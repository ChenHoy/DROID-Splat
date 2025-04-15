import gc
from termcolor import colored
from copy import deepcopy
from typing import Optional, List
from omegaconf import DictConfig
import ipdb

import torch
import torch.multiprocessing as mp

from .utils.loop_utils import TrajectorySegment, TrajectorySegmentManager
from .factor_graph import FactorGraph
import lietorch

# TODO refactor this into a more elegant way
# TODO always check if we have too many factors in the graph, in this case simply use a sliding window
# TODO always use a sliding window in case the optimization gets too large, but still optimize over the whole map


class BackendWrapper(torch.nn.Module):
    """
    Wrapper class for Backend optimization
    """

    def __init__(self, cfg: DictConfig, slam, empty_cache: bool = False):
        super(BackendWrapper, self).__init__()

        self.cfg = cfg
        self.device = cfg.device
        self.net = slam.net
        self.video = slam.video

        self.enable_loop = cfg.tracking.backend.get("use_loop_closure", False)
        self.sparse_opt = cfg.tracking.backend.get("use_sparse_opt", False)

        assert not (self.enable_loop and self.sparse_opt), "Cannot use both loop closure and sparse optimization!"

        # When to start optimizing globally
        self.frontend_window = cfg.tracking.frontend.window
        # Dont consider the state of frontend, but start optimizing after warmup frames
        self.warmup = max(cfg.tracking.backend.get("warmup", 15), cfg.tracking.warmup)
        # Do a final refinement over all keyframes if wanted
        self.do_refinement = cfg.tracking.backend.get("do_refinement", False)

        # Absolute limit for backend optimization to avoid OOM
        # NOTE chen: it is sometimes (e.g. TartanAir) better to use a smaller window like 100
        self.max_window = cfg.tracking.backend.get("window", 200)
        self.steps = cfg.tracking.backend.get("steps", 4)
        self.iters = cfg.tracking.backend.get("iters", 2)
        self.last_t = -1

        self.count = 0
        self.optimizer = Backend(self.net, self.video, self.cfg, self.max_window, self.frontend_window)
        self.empty_cache = empty_cache

    def info(self, msg: str) -> None:
        print(colored("[Backend] " + msg, "blue"))

    # TODO Dont cut off the window, but instead perform multiple BA updates in sliding window fashion
    def forward(
        self,
        local_graph: Optional[FactorGraph] = None,
        add_ii: Optional[torch.Tensor] = None,
        add_jj: Optional[torch.Tensor] = None,
        segments: Optional[TrajectorySegmentManager] = None,
        lock: Optional[mp.Lock] = None,
    ):
        """Run the backend optimization over the whole map."""

        with self.video.get_lock():
            cur_t = self.video.counter.value

        # It makes no sense to optimize inside the frontend
        if cur_t <= self.frontend_window + 1:
            return

        # Safeguard: Run over the whole map only if its within hardware bounds
        # TODO does it make sense to remove the frontend window?
        if cur_t > self.max_window:
            t_start = cur_t - self.max_window
        else:
            t_start = 0
        t_end = cur_t

        # FIXME how exactly is this called in GO-SLAM?!?!?!
        # We did not get the right results here and do it differently, i.e. we use a different t_start and attach the frontend graph
        # TODO But what does GO-SLAM do?!
        if self.enable_loop:
            _, n_edges = self.optimizer.loop_ba(
                t_start,
                t_end,
                self.steps,
                self.iters,
                False,
                local_graph=local_graph,
                add_ii=add_ii,
                add_jj=add_jj,
                lock=lock,
            )
            msg = "Loop BA: [{}, {}]; Using {} edges!".format(t_start, t_end, n_edges)
        elif self.sparse_opt:
            assert segments is not None, "Need to provide segments for sparse optimization!"
            _, n_edges = self.optimizer.sparse_segment_ba(
                cur_t - 1,
                segments,
                self.max_window,
                steps=self.steps,
                iters=self.iters,
                motion_only=False,
                local_graph=local_graph,
                add_ii=add_ii,
                add_jj=add_jj,
                lock=lock,
            )
            msg = "Loop BA: [{}, {}]; Using {} edges!".format(t_start, t_end, n_edges)
        else:
            _, n_edges = self.optimizer.dense_ba(
                t_start, t_end, self.steps, self.iters, motion_only=False, add_ii=add_ii, add_jj=add_jj, lock=lock
            )
            msg = "Full BA: [{}, {}]; Using {} edges!".format(t_start, t_end, n_edges)
        self.info(msg)

        if self.empty_cache:
            torch.cuda.empty_cache()
            gc.collect()

        self.last_t = cur_t
        self.count += 1


class Backend:
    """
    Backend optimization over the whole map and pose graph.
    This exlusively uses the CUDA kernels for optimization and storing the correlation volume to save memory.

    NOTE: main difference to frontend is initializing a new factor graph each time we call
    """

    def __init__(self, net, video, cfg, max_window: int = 200, frontend_window: int = 20):
        self.video = video
        self.device = cfg.device
        self.update_op = net.update

        self.upsample = cfg.tracking.get("upsample", False)
        self.beta = cfg.tracking.get("beta", 0.7)  # Balance rotation and translation for distance computation
        self.thresh = cfg.tracking.backend.get("thresh", 20.0)
        self.radius = cfg.tracking.backend.get("radius", 2)
        self.max_window = max_window
        self.frontend_window = frontend_window
        self.max_factor = cfg.tracking.backend.get("max_factor_mult", 16)
        self.nms = cfg.tracking.backend.get("nms", 0)

        # Loop parameters for loop closure detection
        self.last_loop_t = -1
        self.loop_window = cfg.tracking.backend.get("loop_window", 200)
        self.loop_radius = cfg.tracking.backend.get("loop_radius", 3)
        self.loop_nms = cfg.tracking.backend.get("loop_nms", 2)
        self.loop_thresh = cfg.tracking.backend.get("loop_thresh", 30.0)
        self.loop_max_factor = cfg.tracking.backend.get("loop_max_factor_mult", 16)

    def info(self, msg: str) -> None:
        print(colored("[Backend] " + msg, "blue"))

    @torch.no_grad()
    def accumulate_pose_change(self, pose_prev, pose_cur, t0, t1) -> None:
        """Consider old video.pose_changes to accumulate over multiple backend updates -> dP_cur = dP * dP_prev

        Derivation: g12 = g2 * g1.inv(), g23 = g3 * g2.inv()  -> g3 = g23 * g2 = g23 * g12 * g1 = g13 * g1
        """
        g0, g1 = lietorch.SE3.InitFromVec(pose_prev), lietorch.SE3.InitFromVec(pose_cur)
        dP = g1 * g0.inv()  # Get relative pose in forward direction
        dP_prev = lietorch.SE3.InitFromVec(self.video.pose_changes[t0:t1])
        self.video.pose_changes[t0:t1] = (dP * dP_prev).vec()  # You can now get g_cur = dP * g_prev

    @torch.no_grad()
    def accumulate_scale_change(self, scale_prev, scale_cur, t0, t1) -> None:
        """Consider the old mean disparity and the current one.

        NOTE Since we want to apply scale changes to a 3D point cloud with depth, the inverse scale change needs to be multiplied!
        """
        delta = scale_cur / (scale_prev + 1e-6)
        self.video.scale_changes[t0:t1] = delta
        self.video.scale_changes.clamp_(0.01, 100.0)  # Clamp scale changes to reasonable values

    @torch.no_grad()
    def perform_ba(
        self,
        graph: FactorGraph,
        t_start: int,
        t_end: int,
        steps: int,
        iters: int,
        motion_only: bool,
        lock: Optional[mp.Lock] = None,
    ):
        if lock is None:
            lock = self.video.get_lock()

        with lock:
            # NOTE chen: computing the scale of a scene is not straight-forward, we simply take the mean disparity as proxy
            scales_before = self.video.disps[t_start:t_end].mean(dim=[1, 2]).clone()
            poses_before = self.video.poses[t_start + 1 : t_end].clone()  # Memoize pose before optimization

        # Use t_start + 1 to always fix the first pose!
        graph.update_lowmem(
            t0=t_start + 1, t1=t_end, steps=steps, iters=iters, max_t=t_end, motion_only=motion_only, lock=lock
        )

        with lock:
            poses_after = self.video.poses[t_start + 1 : t_end]  # Memoize pose before optimization
            scales_after = self.video.disps[t_start:t_end].mean(dim=[1, 2]).clone()
            # Memoize pose change in self.video so other Processes can adapt their datastructures
            self.accumulate_pose_change(poses_before, poses_after, t0=t_start + 1, t1=t_end)
            # Memoize scale change in self.video so other Processes can adapt their datastructures
            self.accumulate_scale_change(scales_before, scales_after, t0=t_start, t1=t_end)

    @torch.no_grad()
    def dense_ba(
        self,
        t_start: int = 0,
        t_end: Optional[int] = None,
        steps: int = 8,
        iters: int = 2,
        motion_only: bool = False,
        add_ii: Optional[torch.Tensor] = None,
        add_jj: Optional[torch.Tensor] = None,
        lock: Optional[mp.Lock] = None,
    ):
        """Dense Bundle Adjustment over the whole map. Used for global optimization in the Backend.

        NOTE self.max_factor of the Graph is useless in case the window gets too big, because the function only removes edges not within
        a specific radius/nms neighborhood. The proximity factors will ALWAYS be added to the graph. So if you have a huge window, it will easily exceed the
        number of factors.
        NOTE chen: Since we dont use t0, t1 and max_t here together with t_start and t_end (like DROID-SLAM), not all factors will be used and
        max_factors / n_edges is not as informative of GPU use as you would believe
        """

        with self.video.get_lock():
            cur_t = self.video.counter.value

        if t_end is None:
            t_end = cur_t
        n = t_end - t_start

        # NOTE chen: This is one of the most important numbers for loop closures!
        # If you have a large map, then keep this number really high or else the drift could mess up the map
        # NOTE chen: Using only the frontend keeps sometimes a better global scale than mixing frontend + backend if loop closures are missed!
        # (16 for DROID-SLAM), ((int(self.video.stereo) + (self.radius + 2) * 2) for GO-SLAM)
        max_factors = self.max_factor * self.max_window  # NOTE was n instead of self.max_window in others

        graph = FactorGraph(self.video, self.update_op, self.device, "alt", max_factors, self.upsample)
        n_edges = graph.add_proximity_factors(rad=self.radius, nms=self.nms, beta=self.beta, thresh=self.thresh)
        # FIXME simply del and rebuild the graph in a sliding window fashion in case we have too many factors
        if n_edges > max_factors:
            self.info("Warning. Already going above the max. number of factors from proximity alone")
        # Filter out low confidence edges
        graph.filter_edges()

        if add_ii is not None and add_jj is not None:
            graph.add_factors(add_ii, add_jj)
            # NOTE we get much better results when also including the neighbors of loop edges for more constraints
            graph.add_neighbors(add_ii, add_jj, radius=3)

        self.perform_ba(graph, t_start, t_end, steps, iters, motion_only, lock=lock)
        self.video.dirty[t_start:t_end] = True  # Mark optimized frames, for updating visualization

        # Free up memory again after optimization
        graph.clear_edges()
        del graph
        torch.cuda.empty_cache()
        gc.collect()

        return n, n_edges

    @torch.no_grad()
    def loop_ba(
        self,
        t_start,
        t_end,
        steps=8,
        iters=2,
        motion_only=False,
        lm=5e-5,
        ep=5e-2,
        local_graph: Optional[FactorGraph] = None,
        add_ii: Optional[torch.Tensor] = None,
        add_jj: Optional[torch.Tensor] = None,
        lock: Optional[mp.Lock] = None,
    ):
        """Perform an update on the graph with loop closure awareness according to GO-SLAM. This uses a higher step size
        for optimization than the dense bundle adjustment of the backend and rest of frontend.

        NOTE This does not add explicit loop constraints, but only adds edges with very low optical flow / distance to the graph for a given segment
        within the loop radius.
        NOTE This somehow does not work well on our setup and I suspect that in theory this can only partially reduce drift because in a large graph
        the influence of a small set of edges is not enough to correct the drift.
        """
        with self.video.get_lock():
            if t_end is None:
                t_end = self.video.counter.value

        # NOTE chen: Make sure you have a large enough loop window set in cfg!
        # on larger maps you want to at least have a window of ~100 so we get enough factors
        max_factors = self.loop_max_factor * self.loop_window  # (16 for DROID-SLAM , 8 for GO-SLAM)
        n = t_end - t_start
        t_start_loop = max(0, t_end - self.loop_window)

        graph = FactorGraph(
            self.video,
            self.update_op,
            device=self.device,
            corr_impl="alt",
            max_factors=max_factors,
            upsample=self.upsample,
        )
        # Add local graph if wanted, e.g. this could be the current frontend graph
        if local_graph is not None:
            copy_attr = ["ii", "jj", "age", "net", "target", "weight"]
            for key in copy_attr:
                val = getattr(local_graph, key)
                if val is not None:
                    setattr(graph, key, deepcopy(val))

        # NOTE this removes past edges since remove=True per default (in GO-SLAM)
        n_edges = graph.add_loop_aware_proximity_factors(
            t_start,
            t_end,
            t_start_loop,
            self.loop_radius,
            self.loop_nms,
            self.beta,
            self.loop_thresh,
            max_factors,
            remove=False,
            loop=True,
        )
        if add_ii is not None and add_jj is not None:
            graph.add_factors(add_ii, add_jj)

        # Filter out low confidence edges
        graph.filter_edges()
        self.perform_ba(graph, t_start, t_end, steps, iters, motion_only, lock=lock)
        self.video.dirty[t_start:t_end] = True  # Mark optimized frames, for updating visualization

        # Free up memory again after optimization
        graph.clear_edges()
        del graph
        torch.cuda.empty_cache()
        gc.collect()

        return n, n_edges

    # TODO check if we still have space left in max_window after adding the segments,
    # if yes, then simply add more adjacent frames around the segments until max_window is full
    @torch.no_grad()
    def sparse_segment_ba(
        self,
        index: int,
        segments: TrajectorySegmentManager,
        max_window: int = 500,
        segment_padding: int = 5,
        num_neighbors: int = 1,
        steps=8,
        iters=2,
        motion_only=False,
        lm=5e-5,
        ep=5e-2,
        local_graph: Optional[FactorGraph] = None,
        add_ii: Optional[torch.Tensor] = None,
        add_jj: Optional[torch.Tensor] = None,
        lock: Optional[mp.Lock] = None,
    ):
        """For large-scale outdoor scenes (e.g. KITTI), where we encounter loop closures, use a sparse segment optimization based on the
        loop edges, i.e. we divide the map into distinct segments between loop closures and optimize the segments separately.
        In case we have a very large window, we do multiple optimizations over multiple segments in the window.

        This can be used quite flexibly, most of the time you want to optimize your current segment and its past neighbor.

        TODO we could also cluster the map in different ways, i.e. spatial clustering, octrees, features clustering, etc.
        """
        # Update counter of segments in case index lies in an open segment
        with self.video.get_lock():
            cur_t = self.video.counter.value
        segments.advance_counter(cur_t)  # Limit the current open window to the current time frame
        n_factors, print_segments = 0, []

        # Get the current segment + neighboring segments, with max. num_neighbors segments to left and right (mostly left/past usually)
        segs_in_window = segments.get_neighbor_segments(index, max_window=max_window, num_neighbors=num_neighbors)

        graph = FactorGraph(
            self.video,
            self.update_op,
            device=self.device,
            corr_impl="alt",
            max_factors=self.max_factor,
            upsample=self.upsample,
        )
        # Add local graph if wanted, e.g. this could be the current frontend graph
        if local_graph is not None:
            copy_attr = ["ii", "jj", "age", "net", "target", "weight"]
            for key in copy_attr:
                val = getattr(local_graph, key)
                if val is not None:
                    setattr(graph, key, deepcopy(val))

        # BA will be run over the whole video buffer between [t0, t1], but respect the sparsity of the graph
        t_start, t_end = segs_in_window[0].start - segment_padding, segs_in_window[0].end + segment_padding
        for seg in segs_in_window[1:]:
            t_start = min(t_start, seg.start - segment_padding)
            t_end = max(t_end, seg.end + segment_padding)
        t_start, t_end = max(0, t_start), min(t_end, cur_t)
        n = t_end - t_start

        ### Build up sparse graph
        # Build proximity within each segment
        for i, seg in enumerate(segs_in_window):
            t0, t1 = max(seg.start - segment_padding, 0), min(seg.end + segment_padding, cur_t)
            # NOTE proximity factors are built by interlacing (t0, max_t) and (t1, max_t) for each segment
            # NOTE vanilla BA in DROID-SLAM uses t0=t1=0 and max_t=cur_t, but then runs BA only over [t0, t1]
            print_segments.append((t0, t1))
            new_factors = graph.add_proximity_factors(
                t0=t0, t1=t0, rad=self.radius, nms=self.nms, beta=self.beta, thresh=self.thresh, max_t=t1
            )
            n_factors += new_factors
            if n_factors > self.max_factor:
                self.info(
                    f"Warning. Reached maximum number of factors in sparse segment optimization! Only adding {i}/{len(segs_in_window)} segments"
                )
                break

        # Build edges between current segment and others
        t00, t01 = segs_in_window[0].start - segment_padding, segs_in_window[0].end + segment_padding
        for ii, seg in enumerate(segs_in_window[1:]):
            t10, t11 = seg.start - segment_padding, seg.end + segment_padding
            t10, t11 = max(t10, 0), min(t11, cur_t)
            new_factors = graph.add_cross_window_edges(
                t00, t01, t10, t11, beta=self.beta, thresh=self.thresh, max_factors=self.max_factor
            )
            n_factors += new_factors
            if n_factors > self.max_factor:
                self.info(
                    f"Warning. Reached maximum number of factors in sparse segment optimization! Not adding edges between {index} and ({seg.start}, {seg.end})"
                )
                break

        if add_ii is not None and add_jj is not None:
            graph.add_factors(add_ii, add_jj)

        # Filter out low confidence edges
        graph.filter_edges()
        self.perform_ba(graph, t_start, t_end, steps, iters, motion_only, lock=lock)
        self.video.dirty[t_start:t_end] = True  # Mark optimized frames, for updating visualization

        # Free up memory again after optimization
        graph.clear_edges()
        del graph
        torch.cuda.empty_cache()
        gc.collect()

        self.info("Optimized segments together: {}".format(print_segments))

        return n, n_factors

    # TODO should we use the parent?
    # TODO check if we still have space left in max_window after adding the segments,
    # if yes, then simply add more adjacent frames around the segments until max_window is full
    @torch.no_grad()
    def sparse_loop_segment_ba(
        self,
        index: int,
        loop_edges: List,
        segments: TrajectorySegmentManager,
        max_window: int = 500,
        segment_padding: int = 5,
        steps=8,
        iters=2,
        motion_only=False,
        lm=5e-5,
        ep=5e-2,
        local_graph: Optional[FactorGraph] = None,
        lock: Optional[mp.Lock] = None,
    ):
        """Given an index (usually the current time frame), we optimize the current segment and check if a loop edge lies adjacent to this segment. We then
        use the edge to find spatially adjacent segments and optimize them together with the current segment."""
        # Update counter of segments in case index lies in an open segment
        with self.video.get_lock():
            cur_t = self.video.counter.value
        segments.advance_counter(cur_t)
        n_factors, print_segments = 0, []

        graph = FactorGraph(
            self.video,
            self.update_op,
            device=self.device,
            corr_impl="alt",
            max_factors=self.max_factor,
            upsample=self.upsample,
        )
        # Add local graph if wanted, e.g. this could be the current frontend graph
        if local_graph is not None:
            copy_attr = ["ii", "jj", "age", "net", "target", "weight"]
            for key in copy_attr:
                val = getattr(local_graph, key)
                if val is not None:
                    setattr(graph, key, deepcopy(val))

        # Get the minimal leaf segment and if merged with others due to bigger loops, its parent segment
        # NOTE usually you dont care about the parent
        current_parent, current_min_segment = segments.find_segment_for_index(index)
        t00, t01 = current_min_segment.start - segment_padding, current_min_segment.end + segment_padding
        t00, t01 = max(t00, 0), min(t01, cur_t)
        n = t01 - t00

        # Build proximity within the current segment
        new_factors = graph.add_proximity_factors(
            t0=t00, t1=t00, rad=self.radius, nms=self.nms, beta=self.beta, thresh=self.thresh, max_t=t01
        )
        print_segments.append((t00, t01))
        n_factors += new_factors
        if n_factors > self.max_factor:
            self.info(
                f"Warning. Reached maximum number of factors in sparse segment optimization! Only optimizing over current segment ..."
            )

        # Check if a loop edge (i, j) lies within the current segment, to see if there is a connection
        edge_in_seg = None
        for edge in loop_edges:
            if edge[0] >= t00 and edge[0] <= t01:
                edge_in_seg = edge
                break

        # Add the connected previous segment to our optimization
        if edge_in_seg is not None:
            # Search with j to receiving node
            # FIXME since edge_inseg[1] (j) is right at the end of the segment, its not clear which segment to use (the previous or after the loop closed?)
            # TODO its not clear if we can add both, which one makes more sense?
            # NOTE I think the next one makes sense, because the loop closure usually marks the beginning of overlapping parts
            prev_parent, prev_min_segment = segments.find_segment_for_index(edge_in_seg[1] + 1)
            t10, t11 = prev_min_segment.start - segment_padding, prev_min_segment.end + segment_padding
            t10, t11 = max(t10, 0), min(t11, cur_t)
            print_segments.append((t10, t11))

            new_factors = graph.add_proximity_factors(
                t0=t10, t1=t10, rad=self.radius, nms=self.nms, beta=self.beta, thresh=self.thresh, max_t=t11
            )
            n_factors += new_factors
            if n_factors > self.max_factor:
                self.info(
                    f"Warning. Overshot maximum number of factors: {n_factors} > {self.max_factor}in sparse segment optimization!"
                )
            else:
                # Add the edge between the two segments
                new_factors = graph.add_cross_window_edges(
                    t00, t01, t10, t11, beta=self.beta, thresh=self.thresh, max_factors=self.max_factor
                )
                n_factors += new_factors
                if n_factors > self.max_factor:
                    self.info(
                        f"Warning. Overshot maximum number of factors: {n_factors} > {self.max_factor}in sparse segment optimization!"
                    )

        t_start, t_end = min(t00, t10), max(t01, t11)
        # Filter out low confidence edges
        graph.filter_edges()
        self.perform_ba(graph, t_start, t_end, steps, iters, motion_only, lock=lock)
        self.video.dirty[t_start:t_end] = True  # Mark optimized frames, for updating visualization

        # Free up memory again after optimization
        graph.clear_edges()
        del graph
        torch.cuda.empty_cache()
        gc.collect()

        self.info("Optimized segments together: {}".format(print_segments))

        return n, n_factors
