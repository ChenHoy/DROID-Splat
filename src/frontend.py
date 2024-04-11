import torch
from copy import deepcopy
from time import gmtime, strftime, time

from .factor_graph import FactorGraph


class Frontend:
    def __init__(self, net, video, cfg):
        self.video = video
        self.update_op = net.update
        self.warmup = cfg["tracking"]["warmup"]
        self.upsample = cfg["tracking"]["upsample"]
        self.beta = cfg["tracking"]["beta"]
        self.verbose = cfg.slam.verbose

        self.frontend_max_factors = cfg["tracking"]["frontend"]["max_factors"]
        self.frontend_nms = cfg["tracking"]["frontend"]["nms"]
        self.keyframe_thresh = cfg["tracking"]["frontend"]["keyframe_thresh"]
        self.frontend_window = cfg["tracking"]["frontend"]["window"]
        self.frontend_thresh = cfg["tracking"]["frontend"]["thresh"]
        self.frontend_radius = cfg["tracking"]["frontend"]["radius"]
        self.enable_loop = cfg["tracking"]["frontend"]["enable_loop"]
        self.last_loop_t = -1

        self.graph = FactorGraph(
            video,
            net.update,
            device=cfg.slam.device,
            corr_impl="volume",
            max_factors=self.frontend_max_factors,
            upsample=self.upsample,
        )

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontend variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2

        # Loop closure parameters
        self.loop_window = cfg["tracking"]["backend"]["loop_window"]
        self.loop_radius = cfg["tracking"]["backend"]["loop_radius"]
        self.loop_nms = cfg["tracking"]["backend"]["loop_nms"]
        self.loop_thresh = cfg["tracking"]["backend"]["loop_thresh"]

        self.enable_loop = False
        self.last_loop_t = -1

    @torch.no_grad()
    def loop_closure_update(
        self, t_start, t_end, steps=6, motion_only=False, lm=1e-4, ep=1e-1
    ):
        """Perform an update on the graph with loop closure awareness. This uses a higher step size
        for optimization than the dense bundle adjustment of the backend and rest of frontend.
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
            t_start_loop + 1,
            t_end,
            steps=steps,
            max_t=t_end,
            lm=lm,
            ep=ep,
            motion_only=motion_only,
        )

        self.graph.clear_edges()
        torch.cuda.empty_cache()
        # Mark the frames as updated
        self.video.dirty[t_start:t_end] = True

        return t_end - t_start_loop, n_edges

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

        # Condition video.disps based on external sensor data if given before optimizing
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
        d = self.video.distance(
            [self.t1 - 3], [self.t1 - 2], beta=self.beta, bidirectional=True
        )

        # If the distance is too small, remove the last keyframe
        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 2)

            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1
        # Optimize again for k2 iterations (This time we do this loop aware)
        else:
            cur_t = self.video.counter.value
            t_start = 0
            now = f'{strftime("%Y-%m-%d %H:%M:%S", gmtime())} - Loop BA'
            msg = f"\n\n {now} : [{t_start}, {cur_t}]; Current Keyframe is {cur_t}, last is {self.last_loop_t}."
            if self.enable_loop and cur_t > self.frontend_window:
                n_kf, n_edge = self.loop_closure_update(
                    t_start=0,
                    t_end=cur_t,
                    steps=self.iters2,
                    motion_only=False,
                )

                print(msg + f" {n_kf} KFs, last KF is {self.last_loop_t}! \n")
                self.last_loop_t = cur_t

            else:
                for itr in range(self.iters2):
                    self.graph.update(t0=None, t1=None, use_inactive=True)

        # set pose for next iteration
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1]
        self.video.disps[self.t1] = self.video.disps[self.t1 - 1].mean()

        ### update visualization
        # Sanity check, because I think sometimes the loop_ba results in [] for ii
        if self.graph.ii.numel() > 0:
            self.video.dirty[self.graph.ii.min() : self.t1] = True

    def __initialize(self):
        """initialize the SLAM system"""

        self.t0 = 0
        self.t1 = self.video.counter.value

        # build edges between nearby(radius <= 3) frames within local windown [t0, t1]
        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(t0=1, t1=None, use_inactive=True)

        # build edges between [t0, video.counter] and [t1, video.counter]
        self.graph.add_proximity_factors(
            t0=0, t1=0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False
        )

        for itr in range(8):
            self.graph.update(t0=1, t1=None, use_inactive=True)

        self.video.poses[self.t1] = self.video.poses[self.t1 - 1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1 - 4 : self.t1].mean()

        # initialization complete
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

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()

        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()
            
        else:
            pass
