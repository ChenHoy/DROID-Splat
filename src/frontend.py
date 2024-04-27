import gc
from copy import deepcopy
from termcolor import colored
from time import gmtime, strftime, time

import torch

from .factor_graph import FactorGraph
from .motion_filter import MotionFilter


class Tracker(torch.nn.Module):
    """
    Wrapper class for SLAM frontend tracking.
    """

    def __init__(self, cfg, args, slam):
        super(Tracker, self).__init__()
        self.args = args
        self.cfg = cfg
        self.device = args.device
        self.net = slam.net
        self.video = slam.video

        # filter incoming frames so that there is enough motion
        self.frontend_window = cfg["tracking"]["frontend"]["window"]
        filter_thresh = cfg["tracking"]["motion_filter"]["thresh"]

        self.motion_filter = MotionFilter(self.net, self.video, thresh=filter_thresh, device=self.device)
        self.frontend = Frontend(self.net, self.video, self.args, self.cfg)

    @torch.no_grad()
    def forward(self, timestamp, image, depth, intrinsic, gt_pose=None):
        ### check there is enough motion
        self.motion_filter.track(timestamp, image, depth, intrinsic, gt_pose=gt_pose)

        # local bundle adjustment
        self.frontend()


class Frontend:
    def __init__(self, net, video, args, cfg):
        self.video = video
        self.device = video.device
        self.update_op = net.update

        # NOTE chen: This reduces memory a lot but increases run-time! This potentially saves ~5GB,
        # but is nearly 2x run-time
        # On most scenes its fine to simply not release the cache
        self.release_cache = False  # TODO make configurable

        # Frontend variables
        self.is_initialized = False
        self.count = 0
        self.max_age = 25
        # TODO make configurable
        self.iters1, self.iters2 = 6, 4  # 4, 2
        self.warmup = cfg["tracking"]["warmup"]
        self.upsample = cfg["tracking"]["upsample"]
        self.beta = cfg["tracking"]["beta"]
        self.frontend_max_factors = cfg["tracking"]["frontend"]["max_factors"]
        self.frontend_nms = cfg["tracking"]["frontend"]["nms"]
        self.keyframe_thresh = cfg["tracking"]["frontend"]["keyframe_thresh"]
        self.frontend_window = cfg["tracking"]["frontend"]["window"]
        self.frontend_thresh = cfg["tracking"]["frontend"]["thresh"]
        self.frontend_radius = cfg["tracking"]["frontend"]["radius"]

        # Loop closure parameters
        self.enable_loop = cfg["tracking"]["frontend"]["enable_loop"]
        self.last_loop_t = -1
        self.loop_window = cfg["tracking"]["backend"]["loop_window"]
        self.loop_radius = cfg["tracking"]["backend"]["loop_radius"]
        self.loop_nms = cfg["tracking"]["backend"]["loop_nms"]
        self.loop_thresh = cfg["tracking"]["backend"]["loop_thresh"]

        # Local optimization window
        self.t0, self.t1 = 0, 0

        # Data structure for local map
        self.graph = FactorGraph(
            video,
            net.update,
            device=args.device,
            corr_impl="volume",
            max_factors=self.frontend_max_factors,
            upsample=self.upsample,
        )

    def loop_info(self, t0: int, t1: int, n_kf: int, n_edges: int) -> None:
        msg = "Loop BA: [{}, {}]; Current Keyframe is {}, last is {}. Using {} KFs and {} edges!".format(
            t0, t1, t1, self.last_loop_t, n_kf, n_edges
        )
        self.info(msg)

    def info(self, msg: str):
        print(colored("[Frontend] " + msg, "yellow"))

    def get_ram_usage(self):
        free_mem, total_mem = torch.cuda.mem_get_info(device=self.device)
        used_mem = 1 - (free_mem / total_mem)
        return used_mem, free_mem

    @torch.no_grad()
    def loop_closure_update(self, t_start, t_end, steps=6, motion_only=False, lm=1e-4, ep=1e-1):
        """Perform an update on the graph with loop closure awareness. This uses a higher step size
        for optimization than the dense bundle adjustment of the backend and rest of frontend.

        NOTE chen: this should be used with caution, if you have a longer video / larger map, then using
        loop closures in the frontend can lead to sudden high spikes in RAM! It seems much safer
        to use loop closures in the backend with a fixed optimization window size to not get surprised.
        """
        max_factors = 8 * self.loop_window
        t_start_loop = max(0, t_end - self.loop_window)
        left_factors = max_factors - len(self.graph.ii)

        n_edges = self.graph.add_loop_aware_proximity_factors(
            t_start,
            t_end,
            t_start_loop,
            True,
            self.loop_radius,
            self.loop_nms,
            self.beta,
            self.loop_thresh,
            max_factors=left_factors,
        )
        # fix the start point to avoid drift, be sure to use t_start_loop rather than t_start here.
        self.graph.update_lowmem(
            t_start_loop + 1, t_end, steps=steps, max_t=t_end, lm=lm, ep=ep, motion_only=motion_only
        )
        self.graph.clear_edges()
        torch.cuda.empty_cache()
        # Mark the frames as updated
        self.video.dirty[t_start:t_end] = True
        return t_end - t_start_loop, n_edges

    @torch.no_grad()
    def global_scale_optimization(
        self, t_start, t_end, steps=4, lm=1e-4, ep=1e-1, radius: int = 200, nms: int = 2, thresh: float = 10.0
    ):
        """Optimize the scene globally with the pure Python code for scale optimization. This optimizes poses and structure
        in a block-coordinate descent way."""
        assert self.graph.scale_priors == True, "You need to set scale_priors to True when using this function!"
        assert self.video.optimize_scales == True, "You need to set scale_priors to True when using this function!"

        if t_end is None:
            t_end = self.video.counter.value
        n = t_end - t_start
        max_factors = 16 * t_end  # From DROID-SLAM

        graph = FactorGraph(
            self.video,
            self.update_op,
            device=self.device,
            corr_impl="volume",
            max_factors=max_factors,
            upsample=self.upsample,
        )
        n_edges = graph.add_proximity_factors(rad=radius, nms=nms, beta=self.beta, thresh=thresh, remove=False)
        print(colored(f"[Backend] Performing full scale prior optimization over map with: {n_edges} edges!", "blue"))
        graph.prior_update_lowmem(t0=t_start, t1=t_end, steps=steps, lm=lm, ep=ep)
        print(colored("[Backend] Done!", "blue"))

        # Empty cache immediately and release memory
        graph.clear_edges()
        del graph
        torch.cuda.empty_cache()
        gc.collect()

        with self.video.get_lock():
            self.video.dirty[t_start:t_end] = True  # Mark optimized frames, for updating visualization

    def __update(self):
        """add edges, perform update"""

        self.count += 1
        self.t1 += 1

        # Remove old factors if we already computed a correlation volume
        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        # build edges between [t1-5, video.counter] and [t1-window, video.counter]
        # Add new factors within proximity
        self.graph.add_proximity_factors(
            self.t1 - 5,
            max(self.t1 - self.frontend_window, 0),
            rad=self.frontend_radius,
            nms=self.frontend_nms,
            thresh=self.frontend_thresh,
            beta=self.beta,
            remove=True,
        )

        # NOTE chen: Dont do this with a prior, because it will be harder to correct the scales
        # Condition video.disps based on external sensor data if given before optimizing
        if not self.video.optimize_scales:
            self.video.disps[self.t1 - 1] = torch.where(
                self.video.disps_sens[self.t1 - 1] > 0,
                self.video.disps_sens[self.t1 - 1],
                self.video.disps[self.t1 - 1],
            )

        # Frontend Bundle Adjustment to optimize the current local window
        # This is run for k1 iterations
        for itr in range(self.iters1):
            self.graph.update(t0=None, t1=None, use_inactive=True)

        # set initial pose for next frame
        d = self.video.distance([self.t1 - 3], [self.t1 - 2], beta=self.beta, bidirectional=True)

        # If the distance is too small, remove the last keyframe
        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 2)

            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1
        # Optimize again for k2 iterations (This time we do this loop aware)
        else:
            cur_t = self.video.counter.value
            if self.enable_loop and cur_t > self.frontend_window:
                ### 2nd update
                # FIXME the t_start = 0 will produce a lot of memory in case we have a big map
                n_kf, n_edges = self.loop_closure_update(t_start=0, t_end=cur_t, steps=self.iters2, motion_only=False)
                self.loop_info(0, cur_t, n_kf, n_edges)
                self.last_loop_t = cur_t
            else:
                t0 = max(1, self.graph.ii.min().item() + 1)
                t1 = max(self.graph.ii.max().item(), self.graph.jj.max().item()) + 1
                msg = "Running frontend over [{}, {}] with {} factors.".format(t0, t1, self.graph.ii.numel())
                self.info(msg)

                ### 2nd update
                for itr in range(self.iters2):
                    # TODO make steps of this update configurable!
                    self.graph.update(t0=None, t1=None, use_inactive=True)

        # Manually free memory here as this builds up over time
        if self.release_cache:
            used_mem, total_mem = torch.cuda.mem_get_info(device=self.device)
            if used_mem >= 0.8:
                torch.cuda.empty_cache()
                gc.collect()

        # set pose for next iteration
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1]
        self.video.disps[self.t1] = self.video.disps[self.t1 - 1].mean()

        ### update visualization
        # NOTE chen: Sanity check, because I think sometimes the loop_ba results in [] for ii
        if self.graph.ii.numel() > 0:
            self.video.dirty[self.graph.ii.min() : self.t1] = True

    def __initialize(self):
        """initialize the SLAM system"""

        self.t0, self.t1 = 0, self.video.counter.value

        # build edges between nearby(radius <= 3) frames within local windown [t0, t1]
        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        ### First optimization
        for itr in range(8):
            self.graph.update(t0=1, t1=None, use_inactive=True)

        # build edges between [t0, video.counter] and [t1, video.counter]
        self.graph.add_proximity_factors(t0=0, t1=0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        ### Second optimization
        for itr in range(8):
            self.graph.update(t0=1, t1=None, use_inactive=True)

        self.video.poses[self.t1] = self.video.poses[self.t1 - 1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1 - 4 : self.t1].mean()

        # process complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1 - 1].clone()
        self.last_disp = self.video.disps[self.t1 - 1].clone()
        self.last_time = self.video.timestamp[self.t1 - 1].clone()
        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[: self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup - 4, store=True)

    def __call__(self):
        """main update"""

        # Initialize
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()
            self.info("Initialized!")

        # Update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()

        else:
            pass
