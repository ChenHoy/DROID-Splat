import gc
from termcolor import colored
from copy import deepcopy
from typing import Optional
import ipdb

import torch
import numpy as np

from .factor_graph import FactorGraph
import lietorch


class BackendWrapper(torch.nn.Module):
    """
    Wrapper class for Backend optimization
    """

    def __init__(self, cfg, slam):
        super(BackendWrapper, self).__init__()

        self.cfg = cfg
        self.device = cfg.device
        self.net = slam.net
        self.video = slam.video

        self.enable_loop = cfg.tracking.backend.get("use_loop_closure", True)
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
        self.optimizer = Backend(self.net, self.video, self.cfg)

    def info(self, msg: str) -> None:
        print(colored("[Backend] " + msg, "blue"))

    def forward(
        self,
        local_graph: Optional[FactorGraph] = None,
        add_ii: Optional[torch.Tensor] = None,
        add_jj: Optional[torch.Tensor] = None,
    ):
        """Run the backend optimization over the whole map."""

        with self.video.get_lock():
            cur_t = self.video.counter.value

        # Safeguard: Run over the whole map only if its within hardware bounds
        if cur_t > self.max_window:
            t_start = cur_t - self.max_window
        else:
            t_start = 0
        t_end = cur_t

        if self.enable_loop:
            _, n_edges = self.optimizer.loop_ba(
                t_start, t_end, self.steps, self.iters, False, local_graph=local_graph, add_ii=add_ii, add_jj=add_jj
            )
            msg = "Loop BA: [{}, {}]; Using {} edges!".format(t_start, t_end, n_edges)
        else:
            _, n_edges = self.optimizer.dense_ba(
                t_start, t_end, self.steps, self.iters, motion_only=False, add_ii=add_ii, add_jj=add_jj
            )
            msg = "Full BA: [{}, {}]; Using {} edges!".format(t_start, t_end, n_edges)
        self.info(msg)

        self.last_t = cur_t
        self.count += 1


class Backend:
    """
    Backend optimization over the whole map and pose graph.
    This exlusively uses the CUDA kernels for optimization and storing the correlation volume to save memory.

    NOTE: main difference to frontend is initializing a new factor graph each time we call
    """

    def __init__(self, net, video, cfg):
        self.video = video
        self.device = cfg.device
        self.update_op = net.update

        self.upsample = cfg.tracking.get("upsample", False)
        self.beta = cfg.tracking.get("beta", 0.7)  # Balance rotation and translation for distance computation
        self.thresh = cfg.tracking.backend.get("thresh", 20.0)
        self.radius = cfg.tracking.backend.get("radius", 2)
        self.max_factor = cfg.tracking.backend.get("max_factor_mult", 16)
        self.nms = cfg.tracking.backend.get("nms", 0)

        # Loop parameters for loop closure detection
        self.last_loop_t = -1
        self.loop_window = cfg.tracking.backend.get("loop_window", 200)
        self.loop_radius = cfg.tracking.backend.get("loop_radius", 3)
        self.loop_nms = cfg.tracking.backend.get("loop_nms", 2)
        self.loop_thresh = cfg.tracking.backend.get("loop_thresh", 30.0)
        self.loop_max_factor = cfg.tracking.backend.get("loop_max_factor_mult", 16)

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
    def dense_ba(
        self,
        t_start: int = 0,
        t_end: Optional[int] = None,
        steps: int = 8,
        iters: int = 2,
        motion_only: bool = False,
        add_ii: Optional[torch.Tensor] = None,
        add_jj: Optional[torch.Tensor] = None,
    ):
        """Dense Bundle Adjustment over the whole map. Used for global optimization in the Backend."""

        with self.video.get_lock():
            if t_end is None:
                t_end = self.video.counter.value
        n = t_end - t_start

        # NOTE chen: This is one of the most important numbers for loop closures!
        # If you have a large map, then keep this number really high or else the drift could mess up the map
        # NOTE chen: Using only the frontend keeps sometimes a better global scale than mixing frontend + backend if loop closures are missed!
        # (16 for DROID-SLAM), ((int(self.video.stereo) + (self.radius + 2) * 2) for GO-SLAM)
        max_factors = self.max_factor * n

        graph = FactorGraph(self.video, self.update_op, self.device, "alt", max_factors, self.upsample)
        n_edges = graph.add_proximity_factors(
            rad=self.radius, nms=self.nms, beta=self.beta, thresh=self.thresh, remove=False
        )
        if add_ii is not None and add_jj is not None:
            graph.add_factors(add_ii, add_jj)

        # NOTE chen: computing the scale of a scene is not straight-forward, we simply take the mean disparity as proxy
        scales_before = self.video.disps[t_start:t_end].mean(dim=[1, 2]).clone()
        poses_before = self.video.poses[t_start + 1 : t_end].clone()  # Memoize pose before optimization
        # fix the start point to avoid drift, be sure to use t_start_loop rather than t_start here.
        graph.update_lowmem(t0=t_start + 1, t1=t_end, steps=steps, iters=iters, max_t=t_end, motion_only=motion_only)
        poses_after = self.video.poses[t_start + 1 : t_end]  # Memoize pose before optimization
        scales_after = self.video.disps[t_start:t_end].mean(dim=[1, 2]).clone()
        # Memoize pose change in self.video so other Processes can adapt their datastructures
        self.accumulate_pose_change(poses_before, poses_after, t0=t_start + 1, t1=t_end)
        # Memoize scale change in self.video so other Processes can adapt their datastructures
        self.accumulate_scale_change(scales_before, scales_after, t0=t_start, t1=t_end)

        graph.clear_edges()
        self.video.dirty[t_start:t_end] = True  # Mark optimized frames, for updating visualization

        # Free up memory again after optimization
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
    ):
        """Perform an update on the graph with loop closure awareness. This uses a higher step size
        for optimization than the dense bundle adjustment of the backend and rest of frontend.
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

        poses_before = self.video.poses[t_start + 1 : t_end].clone()  # Memoize pose before optimization
        # fix the start point to avoid drift, be sure to use t_start_loop rather than t_start here.
        graph.update_lowmem(
            t0=t_start_loop + 1, t1=t_end, steps=steps, iters=iters, max_t=t_end, lm=lm, ep=ep, motion_only=motion_only
        )
        poses_after = self.video.poses[t_start + 1 : t_end].clone()  # Memoize pose before optimization
        # Memoize pose change in self.video so other Processes can adapt their datastructures
        self.accumulate_pose_change(poses_before, poses_after, t0=t_start + 1, t1=t_end)

        graph.clear_edges()
        # Free up memory again after optimization
        del graph
        torch.cuda.empty_cache()
        gc.collect()

        self.video.dirty[t_start:t_end] = True  # Mark optimized frames, for updating visualization

        return n, n_edges

    # TODO implement sparse Pose Graph Optimization similar to HI-SLAM or other systems like ORB-SLAMv3
    # we dont want to optimize all the local frames, only global links in a loop closure fashion
    # This only optimizes the SIM3 poses, not disparity or intrinsics
    # -> How would you transform a normal pose graph into segments with relative pose edges?!
    # i) We need to cluster our graph into distinct segments, we could take e.g. the frontend windows as segments
    # ii) We need to add relative pose edges between the segments, these should not be in SE3, but SIM3 in order to correct scale drift
    # iii) We add a regularization term to our pose graph optimization, which minimizes the difference between the relative pose and the diff. between estimated poses
    #       see https://alida.tistory.com/69#5.-relative-pose-error-pgo
    # -> This would require to either implement a pure Python optimization or adapt the CUDA kernels
    # -> We can use the reference implementation in: https://github.com/princeton-vl/lietorch/blob/master/examples/pgo/main.py, but we first need to define edge sets for this!
    # TODO build up segment datastructure which allows to register a new frontend segment to previous segments
    @torch.no_grad()
    def sparse_ba(
        self, t_start: int = 0, t_end: Optional[int] = None, steps: int = 6, iter: int = 4, motion_only: bool = False
    ):
        """Pose graph optimization over multiple segments of the map. Used as sparse global optimization in the Backend.

        This is much more memory efficient and allows to scale to long-term video with thousands of key frames.
        """

        if t_end is None:
            t_end = self.video.counter.value
        n = t_end - t_start
        max_factors = (int(self.video.stereo) + (self.radius + 2) * 2) * n

        graph = FactorGraph(
            self.video,
            self.update_op,
            device=self.device,
            corr_impl="alt",
            max_factors=max_factors,
            upsample=self.upsample,
        )
        raise NotImplementedError("Sparse BA not yet implemented")
