from typing import Tuple, List, Optional
from tqdm import tqdm
from termcolor import colored
import ipdb

import torch
import lietorch
from lietorch import SE3
from .factor_graph import FactorGraph


class PoseTrajectoryFiller:
    """This class is used to fill in non-keyframe poses"""

    def __init__(self, cfg, net, video, device="cuda:0"):

        self.cfg = cfg
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.count = 0
        self.video = video
        self.device = device

        # mean, std for image normalization
        self.MEAN = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.STDV = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image: torch.Tensor) -> torch.Tensor:
        """features for correlation volume"""
        return self.fnet(image)

    def __fill(
        self,
        timestamps: List[int],
        images: List[torch.Tensor],
        depths: List[torch.Tensor],
        intrinsics: List[torch.Tensor],
        static_masks: Optional[List[torch.Tensor]] = None,
    ) -> List[SE3]:
        """fill operator"""
        tt = torch.tensor(timestamps, device=self.device)
        images = torch.stack(images, dim=0)
        if depths is not None:
            depths = torch.stack(depths, dim=0)
        if static_masks is not None:
            static_masks = torch.stack(static_masks, dim=0)
        intrinsics = torch.stack(intrinsics, 0)
        inputs = images.to(self.device)

        ### linear pose interpolation ###
        N = self.video.counter.value
        M = len(timestamps)
        ts = self.video.timestamp[:N]
        Ps = SE3(self.video.poses[:N])

        # found the location of current timestamp in keyframe queue
        t0 = torch.tensor([ts[ts <= t].shape[0] - 1 for t in timestamps])
        t1 = torch.where(t0 < N - 1, t0 + 1, t0)

        # time interval between nearby keyframes
        dt = ts[t1] - ts[t0] + 1e-3
        dP = Ps[t1] * Ps[t0].inv()  # Get relative pose
        # Do Linear Interpolation in between segments
        v = dP.log() / dt.unsqueeze(dim=-1)
        w = v * (tt - ts[t0]).unsqueeze(dim=-1)
        Gs = SE3.exp(w) * Ps[t0]

        # extract features (no need for context features)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)
        fmap = self.__feature_encoder(inputs)

        # temporally put the non-keyframe at the end of keyframe queue
        self.video.counter.value += M

        self.video[N : N + M] = (
            tt,
            images[:, 0],
            Gs.data,
            1.0,  # NOTE Initializing with 1.0 disparities somehow works well
            depths,
            intrinsics,
            fmap,
            None,  # No context features
            None,  # No context features
            None,  # No groundtruth pose
            static_masks,
        )

        graph = FactorGraph(self.video, self.update)
        # build edge between current frame and nearby keyframes for optimization
        graph.add_factors(t0.cuda(), torch.arange(N, N + M).cuda())
        graph.add_factors(t1.cuda(), torch.arange(N, N + M).cuda())

        for itr in range(6):
            graph.update(N, N + M, motion_only=True)

        Gs = SE3(self.video.poses[N : N + M].clone())  # Extract updated/interpolated poses

        self.video.counter.value -= M  # Reset counter back to where it previously was
        # Remove the non-keyframes again from the video buffer
        self.video.remove(torch.arange(N, N + M))

        return [Gs]

    # TODO handle corner cases!
    # ii) self.mode == "prgbd" and self.frontend.optimize_scales = True -> We have unscaled priors at non-keyframes!
    @torch.no_grad()
    def __call__(self, image_stream, batch_size: int = 32, return_tstamps: bool = False):
        """fill in poses of non-keyframe images in batch mode. This works by first linear interpolating the
        poses in between the keyframes, then computing the optical flow between these frames and doing a motion only BA
        refinement

        Returns:
        ---
        interpolated poses [lietorch.SE3]
        """
        # Check for enough slack when interpolating
        buffer_size = len(self.video.timestamp)
        slack_buffer = buffer_size - self.video.counter.value
        if slack_buffer < batch_size:
            raise Exception(
                colored(
                    "Buffer size too small for using a slack when interpolating, please increase either video.buffer or decrease the batch size here!",
                    "red",
                )
            )

        timestamps, images, depths, intrinsics, masks = [], [], [], [], []
        # store all camera poses and timestamps
        pose_list, all_timestamps = [], []

        for frame in tqdm(image_stream):
            if self.cfg.with_dyn and image_stream.has_dyn_masks:
                timestamp, image, depth, intrinsic, _, static_mask = frame
            else:
                timestamp, image, depth, intrinsic, _ = frame

            # When optimizing intrinsics as well, choose the optimized ones instead of the ones from the dataset!
            if self.video.opt_intr:
                intrinsic = self.video.intrinsics[0] * self.video.scale_factor

            all_timestamps.append(timestamp)
            timestamps.append(timestamp)
            images.append(image)
            if depth is not None:
                depths.append(depth)
            if static_mask is not None:
                masks.append(static_mask)
            intrinsics.append(intrinsic)

            if len(timestamps) == batch_size:

                # FIXME in prgbd mode together with scale_optimization, we will have the wrong prior here and non-keyframes!
                # TODO either use no depths or interpolate a memoized scale
                depths = depths if len(depths) > 0 else None
                masks = masks if len(masks) > 0 else None
                pose_list += self.__fill(timestamps, images, depths, intrinsics, static_masks=masks)
                timestamps, images, depths, intrinsics = [], [], [], []

        if len(timestamps) > 0:
            depths = depths if len(depths) > 0 else None
            pose_list += self.__fill(timestamps, images, depths, intrinsics)

        if return_tstamps:
            return lietorch.cat(pose_list, dim=0), all_timestamps
        else:
            # stitch pose segments together
            return lietorch.cat(pose_list, dim=0)
