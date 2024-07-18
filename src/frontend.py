import gc
from copy import deepcopy
from termcolor import colored
import ipdb
from time import gmtime, strftime, time

import torch
from lietorch import SE3

from .factor_graph import FactorGraph
from .motion_filter import MotionFilter


class FrontendWrapper(torch.nn.Module):
    """
    Wrapper class for SLAM frontend tracking.
    """

    def __init__(self, cfg, slam):
        super(FrontendWrapper, self).__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.net = slam.net
        self.video = slam.video

        # filter incoming frames so that there is enough motion
        self.window = cfg.tracking.frontend.window
        filter_thresh = cfg.tracking.motion_filter.thresh
        self.motion_filter = MotionFilter(self.net, self.video, thresh=filter_thresh, device=self.device)
        self.optimizer = Frontend(self.net, self.video, self.cfg)

        self.count = 0

    @torch.no_grad()
    def forward(self, timestamp, image, depth, intrinsic, gt_pose=None, static_mask=None):
        """Add new keyframes according to apparent motion and run a local bundle adjustment optimization.
        If there is not enough motion between the new frame and the last inserted keyframe, we dont do anything."""
        self.motion_filter.track(timestamp, image, depth, intrinsic, gt_pose=gt_pose, static_mask=static_mask)
        self.optimizer()  # Local Bundle Adjustment
        self.count = self.optimizer.count  # Synchronize counts of wrapper and actual Frontend


class Frontend:
    def __init__(self, net, video, cfg):
        self.video = video
        self.device = video.device
        self.update_op = net.update

        # NOTE chen: This reduces memory a lot but increases run-time! This potentially saves ~5GB,
        # but its nearly 2x run-time
        self.release_cache = cfg.tracking.frontend.get("release_cache", False)

        # Frontend variables
        self.is_initialized = False
        self.count = 0
        self.max_age = cfg.tracking.frontend.get("max_age", 25)
        self.warmup = cfg.tracking.get("warmup", 8)
        self.upsample = cfg.tracking.get("upsample", True)

        self.beta = cfg.tracking.get("beta", 0.3)
        self.max_factors = cfg.tracking.frontend.get("max_factors", 100)
        self.nms = cfg.tracking.frontend.get("nms", 2)
        self.keyframe_thresh = cfg.tracking.frontend.get("keyframe_thresh", 4.0)
        self.window = cfg.tracking.frontend.get("window", 25)
        self.thresh = cfg.tracking.frontend.get("thresh", 16.0)
        self.radius = cfg.tracking.frontend.get("radius", 2)

        self.steps1 = cfg.tracking.frontend.get("steps1", 4)
        self.steps2 = cfg.tracking.frontend.get("steps2", 2)
        self.iters = cfg.tracking.frontend.get("iters", 4)

        # Local optimization window
        self.t0, self.t1 = 0, 0

        # Data structure for local map
        self.graph = FactorGraph(
            video,
            net.update,
            device=cfg.device,
            corr_impl="volume",
            max_factors=self.max_factors,
            upsample=self.upsample,
        )

    def info(self, msg: str):
        print(colored("[Frontend] " + msg, "yellow"))

    def get_ram_usage(self):
        free_mem, total_mem = torch.cuda.mem_get_info(device=self.device)
        used_mem = 1 - (free_mem / total_mem)
        return used_mem, free_mem

    def __update(self):
        """add edges, perform update"""

        self.t1 += 1

        # Remove old factors if we already computed a correlation volume
        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        # build edges between [t1-5, video.counter] and [t1-window, video.counter]
        # Add new factors within proximity
        self.graph.add_proximity_factors(
            self.t1 - 5, max(self.t1 - self.window, 0), self.radius, self.nms, self.beta, self.thresh, remove=True
        )

        # Condition video.disps based on external sensor data if given before optimizing
        # Dont do this with monocular depth, as every new prior has a yet to be determined scale
        if not self.video.cfg.mode == "prgbd" and not self.video.optimize_scales:
            self.video.disps[self.t1 - 1] = torch.where(
                self.video.disps_sens[self.t1 - 1] > 0,
                self.video.disps_sens[self.t1 - 1],
                self.video.disps[self.t1 - 1],
            )

        # Frontend Bundle Adjustment to optimize the current local window
        for itr in range(self.steps1):
            self.graph.update(t0=None, t1=None, iters=self.iters, use_inactive=True)

        # set initial pose for next frame
        d = self.video.distance([self.t1 - 3], [self.t1 - 2], beta=self.beta, bidirectional=True)

        # If the distance is too small, remove the last keyframe
        if d.item() < self.keyframe_thresh:

            self.count += 1  # Only increase the count when a new frame comes in
            self.graph.rm_keyframe(self.t1 - 2)

            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1
        # Optimize again
        else:
            cur_t = self.video.counter.value
            t0 = max(1, self.graph.ii.min().item() + 1)
            t1 = max(self.graph.ii.max().item(), self.graph.jj.max().item()) + 1
            msg = "Running frontend over [{}, {}] with {} factors.".format(t0, t1, self.graph.ii.numel())
            self.info(msg)

            ### 2nd update
            for itr in range(self.steps2):
                self.graph.update(t0=None, t1=None, iters=self.iters, use_inactive=True)

        # Manually free memory here as this builds up over time
        if self.release_cache:
            used_mem, total_mem = torch.cuda.mem_get_info(device=self.device)
            if used_mem >= 0.8:
                torch.cuda.empty_cache()
                gc.collect()

        ### Set pose & disp for next iteration
        # Naive strategy for initializing next pose as previous pose in DROID-SLAM
        # self.video.poses[self.t1] = self.video.poses[self.t1 - 1]
        # Better: use constant speed assumption and extrapolate
        # (usually gives a boost of 1-4mm in ATE RMSE)
        dP = SE3(self.video.poses[self.t1 - 1]) * SE3(self.video.poses[self.t1 - 2]).inv()  # Get relative pose
        self.video.poses[self.t1] = (dP * SE3(self.video.poses[self.t1 - 1])).vec()

        self.video.disps[self.t1] = self.video.disps[self.t1 - 1].mean()

        ### update visualization
        # NOTE chen: Sanity check, because this was sometimes []
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
        self.graph.add_proximity_factors(t0=0, t1=0, rad=2, nms=2, thresh=self.thresh, remove=False)

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
            self.video.mapping_dirty[: self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup - 4, store=True)

    def __call__(self):
        """main update"""

        # Initialize
        if not self.is_initialized and self.video.counter.value == self.warmup:

            # Deactivate scale optimization during initialization!
            # NOTE chen: changing this attribute does not transmit across Process barriers!
            if self.video.optimize_scales:
                self.graph.scale_priors = False

            self.__initialize()
            self.info("Initialized!")

            # Activate it again, so we optimize the prior scales as well during __update()
            if self.video.optimize_scales:
                self.graph.scale_priors = True

        # Update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()

        else:
            pass
