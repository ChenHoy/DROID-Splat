from copy import deepcopy
import gc

import torch
import numpy as np

from .factor_graph import FactorGraph


class Backend:
    def __init__(self, net, video, args, cfg):
        self.video = video
        self.device = args.device
        self.update_op = net.update

        self.upsample = cfg["tracking"]["upsample"]
        self.beta = cfg["tracking"]["beta"]
        self.backend_thresh = cfg["tracking"]["backend"]["thresh"]
        self.backend_radius = cfg["tracking"]["backend"]["radius"]
        self.backend_nms = cfg["tracking"]["backend"]["nms"]

    @torch.no_grad()
    def dense_ba(
        self, t_start: int = 0, steps: int = 6, iter: int = 2, motion_only: bool = False
    ):
        """Dense Bundle Adjustment over the whole map. Used for global optimization in the Backend."""

        t_end = self.video.counter.value
        n = t_end - t_start
        max_factors = (int(self.video.stereo) + (self.backend_radius + 2) * 2) * n

        graph = FactorGraph(
            self.video,
            self.update_op,
            device=self.device,
            corr_impl="alt",
            max_factors=max_factors,
            upsample=self.upsample,
        )
        n_edges = graph.add_loop_aware_proximity_factors(
            t_start,
            t_end,
            radius=self.backend_radius,
            nms=self.backend_nms,
            beta=self.beta,
            thresh=self.backend_thresh,
            max_factors=max_factors,
        )
        # fix the start point to avoid drift, be sure to use t_start_loop rather than t_start here.
        graph.update_lowmem(
            t0=t_start + 1,
            t1=t_end,
            steps=steps,
            itrs=iter,
            max_t=t_end,
            motion_only=motion_only,
        )
        graph.clear_edges()

        self.video.dirty[t_start:t_end] = True

        # Free up memory again after optimization
        del graph
        torch.cuda.empty_cache()
        gc.collect()

        return n, n_edges
