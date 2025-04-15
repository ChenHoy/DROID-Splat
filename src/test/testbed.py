import ipdb
from typing import Optional, Dict, List
from tqdm import tqdm
from termcolor import colored

import torch
import torch.multiprocessing as mp
import lietorch

import numpy as np

from ..slam import SLAM
from ..loop_closure import LoopDetector, LongTermLoopClosure


class SlamTestbed(SLAM):
    """
    Testing class for SLAM system. This simply acts as a way to debug and test new functionality without running everything in parallel
    """

    def __init__(self, *args, **kwargs):
        super(SlamTestbed, self).__init__(*args, **kwargs)

        self.loop_detector = LoopDetector(self.cfg.loop_closure, self.video, device=self.device)
        # Initialize network during worker process, since torch.hub models need to, see https://github.com/Lightning-AI/pytorch-lightning/issues/17637
        # self.loop_detector.net = self.loop_detector.load_eigen()
        self.lc_closure = LongTermLoopClosure(self.cfg.loop_closure, self.video, device=self.device)

    def get_points(self, index: torch.Tensor, mv_filter_thresh: float = 0.1, mv_filter_count=2):

        import droid_backends
        from lietorch import SE3

        s = self.video.scale_factor
        poses = torch.index_select(self.video.poses, 0, index)
        disps = torch.index_select(self.video.disps, 0, index)
        # convert poses to 4x4 matrix
        Ps = SE3(poses).inv().matrix().cpu().numpy()
        images = torch.index_select(self.video.images, 0, index)
        images = images.cpu()[:, ..., int(s // 2 - 1) :: s, int(s // 2 - 1) :: s].permute(0, 2, 3, 1)
        points = droid_backends.iproj(SE3(poses).inv().data, disps, self.video.intrinsics[0]).cpu()

        thresh = mv_filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))
        count = droid_backends.depth_filter(
            self.video.poses, self.video.disps, self.video.intrinsics[0], index, thresh
        )
        count, disps = count.cpu(), disps.cpu()
        # Only keep points that are consistent across multiple views and not too close by
        masks = (count >= mv_filter_count) & (disps > 0.5 * disps.mean(dim=[1, 2], keepdim=True))
        idx = index[0]
        mask = masks[idx].reshape(-1)
        pts = points[idx].reshape(-1, 3)[mask].numpy()
        clr = images[idx].reshape(-1, 3)[mask].numpy()
        ipdb.set_trace()
        pass

    def test_tracking(self, stream, backend_freq: int = 10) -> None:
        """Test tracking in a sequential manner."""
        # assert self.cfg.run_backend, "Need to run backend for this test to make sense!"

        for frame in tqdm(stream):
            frontend_old_count = self.frontend.optimizer.t1  # How many times did the frontend actually run?

            if self.cfg.with_dyn and stream.has_dyn_masks:
                timestamp, image, depth, intrinsic, gt_pose, static_mask = frame
            else:
                timestamp, image, depth, intrinsic, gt_pose = frame
                static_mask = None

            # Frontend insert new frames
            self.frontend(timestamp, image, depth, intrinsic, gt_pose, static_mask=static_mask)

            # If new keyframe got inserted
            if self.frontend.optimizer.is_initialized and frontend_old_count != self.frontend.optimizer.t1:
                # if self.frontend.optimizer.t1 % backend_freq == 0 and self.cfg.run_backend:
                #     self.backend()
                test_index = torch.arange(8, device=self.video.device)
                self.get_points(test_index)
                ipdb.set_trace()

                # Run Loop Closure
                candidates = self.loop_detector()
                if candidates is not None:
                    self.lc_closure(*candidates)

    def test_rescale(
        self, stream, backend_freq: int = 2, render_freq: int = 5, test_until: Optional[int] = None
    ) -> None:
        assert self.cfg.run_backend, "Need to run backend for this test to make sense!"

        for frame in tqdm(stream):
            frontend_old_count = self.frontend.optimizer.t1  # How many times did the frontend actually run?

            if self.cfg.with_dyn and stream.has_dyn_masks:
                timestamp, image, depth, intrinsic, gt_pose, static_mask = frame
            else:
                timestamp, image, depth, intrinsic, gt_pose = frame
                static_mask = None

            # Control when to start and when to stop the SLAM system from outside
            if timestamp < self.t_start:
                continue
            if self.t_stop is not None and timestamp > self.t_stop:
                break

            # Frontend insert new frames
            self.frontend(timestamp, image, depth, intrinsic, gt_pose, static_mask=static_mask)

            if (
                self.frontend.optimizer.is_initialized
                and self.frontend.optimizer.t1 % backend_freq == 0
                and self.cfg.run_backend
            ):
                self.backend()

            if (
                self.frontend.optimizer.is_initialized
                and self.frontend.optimizer.t1 % render_freq == 0
                and self.cfg.run_mapping
                and frontend_old_count > self.gaussian_mapper.warmup
            ):
                self.maybe_reanchor_gaussians()  # If the backend is also running, we reanchor Gaussians when large map changes occur
                self.gaussian_mapper(None, None)

    def run(self, stream):
        """Test the system by running any function dependent on the input stream directly so we can set breakpoints for inspection."""

        processes = []
        if self.cfg.run_visualization:
            processes.append(
                mp.Process(target=self.visualizing, args=(1, self.cfg.run_visualization), name="Visualizing"),
            )
        self.num_running_thread[0] += len(processes)
        for p in processes:
            p.start()

        backend_freq = 10  # Run backend every 5 frontends

        self.loop_detector.net = self.loop_detector.load_eigen()

        # Check new loop closure
        self.test_tracking(stream, backend_freq=backend_freq)

        # Check if reanchoring/rescaling works correctly
        # self.test_rescale(stream, backend_freq=backend_freq, render_freq=render_freq)

        self.terminate(processes, stream, None)
