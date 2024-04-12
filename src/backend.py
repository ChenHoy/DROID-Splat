import gc
from typing import Optional

import torch
import numpy as np

from .factor_graph import FactorGraph


class Backend:
    """
    Backend optimization over the whole map and pose graph.
    This exlusively uses the CUDA kernels for optimization to save memory.

    NOTE: main difference to frontend is initializing a new factor graph each time we call
    """

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
        self, t_start: int = 0, t_end: Optional[int] = None, steps: int = 6, iter: int = 4, motion_only: bool = False
    ):
        """Dense Bundle Adjustment over the whole map. Used for global optimization in the Backend."""

        if t_end is None:
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
        graph.update_lowmem(t0=t_start + 1, t1=t_end, steps=steps, iters=iter, max_t=t_end, motion_only=motion_only)
        graph.clear_edges()
        self.video.dirty[t_start:t_end] = True  # Mark optimized frames, for updating visualization

        # Free up memory again after optimization
        del graph
        torch.cuda.empty_cache()
        gc.collect()

        return n, n_edges
