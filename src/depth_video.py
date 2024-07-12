import ipdb
from termcolor import colored
from copy import deepcopy
from typing import Optional, List

import torch
from torch.multiprocessing import Value
import lietorch
import droid_backends

from .droid_net import cvx_upsample
from .geom import projective_ops as pops
from .geom import matrix_to_lie, align_scale_and_shift
from .geom.ba import bundle_adjustment

from .gaussian_splatting.camera_utils import Camera
from .gaussian_splatting.gaussian_renderer import render


class DepthVideo:
    """
    Data structure of multiple buffers to keep track of indices, poses, disparities, images, external disparities and more
    """

    def __init__(self, cfg):
        self.cfg = cfg

        ### Intrinsics / Calibration ###
        if cfg.data.cam.camera_model == "pinhole":
            self.n_intr = 4
            self.model_id = 0
        elif cfg.data.cam.camera_model == "mei":
            self.n_intr = 5
            self.model_id = 1
        else:
            raise Exception("Camera model not implemented! Choose either pinhole or mei model.")
        self.opt_intr = cfg.opt_intr

        self.ready = Value("i", 0)
        self.counter = Value("i", 0)

        ht = cfg.data.cam.H_out
        self.ht = ht
        wd = cfg.data.cam.W_out
        self.wd = wd
        self.stereo = cfg.mode == "stereo"
        device = cfg.device
        self.device = device
        c = 1 if not self.stereo else 2
        self.scale_factor = 8
        s = self.scale_factor
        buffer = cfg.tracking.buffer
        # Whether we upsample the predictions or not
        self.upsampled = cfg.tracking.upsample

        ### state attributes -> Raw map ###
        self.timestamp = torch.zeros(buffer, device=device, dtype=torch.float).share_memory_()
        # List for keeping track of updated frames for visualization
        self.dirty = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        # List for keeping track of updated frames for Map Renderer
        self.mapping_dirty = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()

        self.images = torch.zeros(buffer, 3, ht, wd, device=device, dtype=torch.float)
        self.intrinsics = torch.zeros(buffer, 4, device=device, dtype=torch.float).share_memory_()
        self.poses = torch.zeros(buffer, 7, device=device, dtype=torch.float).share_memory_()  # c2w quaterion
        self.poses_gt = torch.zeros(buffer, 7, device=device, dtype=torch.float).share_memory_()  # c2w quaterion

        # Measure the change of poses before and after backend optimization, so we can track large map changes
        self.pose_changes = torch.zeros(buffer, 7, device=device, dtype=torch.float).share_memory_()  # c2w quaterion

        self.disps = torch.ones(buffer, ht // s, wd // s, device=device, dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        # Estimated Uncertainty weights for Optimization reduced for each node from factor graph
        self.uncertainty = torch.zeros(buffer, ht // s, wd // s, device=device, dtype=torch.float)
        self.uncertainty_up = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()

        self.disps_sens_up = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht // s, wd // s, device=device, dtype=torch.float).share_memory_()
        # Scale and shift parameters for ambiguous monocular depth
        # Optimze the scales and shifts for Pseudo-RGBD mode
        self.optimize_scales = cfg.mode == "prgbd" and cfg.tracking.frontend.optimize_scales
        self.scales = torch.ones(buffer, device=device, dtype=torch.float).share_memory_()
        self.shifts = torch.zeros(buffer, device=device, dtype=torch.float).share_memory_()

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht // s, wd // s, dtype=torch.half, device=device).share_memory_()
        self.nets = torch.zeros(buffer, 128, ht // s, wd // s, dtype=torch.half, device=device).share_memory_()
        self.inps = torch.zeros(buffer, 128, ht // s, wd // s, dtype=torch.half, device=device).share_memory_()

        ### Initialize poses to identity transformation
        self.poses[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)
        self.poses_gt[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)
        self.pose_changes[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)

        ### Additional flags for multi-view filter -> Clean map for rendering ###
        if self.cfg.tracking.upsample:
            self.disps_clean = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        else:
            self.disps_clean = torch.zeros(buffer, ht // s, wd // s, device=device, dtype=torch.float).share_memory_()
        # Poses that have been finetuned by the Rendering module, we could reassign these back to the DepthVideo
        self.poses_clean = torch.zeros(buffer, 7, device=device, dtype=torch.float).share_memory_()  # w2c quaterion
        self.poses_clean[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)

        # Keep track of which frames have been filtered
        self.filtered_id = torch.tensor([0], dtype=torch.int, device=device).share_memory_()

        self.static_masks = torch.ones(buffer, ht, wd, device=device, dtype=torch.bool).share_memory_()

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        s = self.scale_factor
        self.timestamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None and self.cfg.mode != "mono":
            depth_up = item[4]
            self.disps_sens_up[index] = torch.where(depth_up > 0, 1.0 / depth_up, depth_up)
            self.disps_sens[index] = self.disps_sens_up[index][..., int(s // 2 - 1) :: s, int(s // 2 - 1) :: s]
            if self.cfg.mode != "prgbd" and not self.optimize_scales:
                self.disps[index] = self.disps_sens[index].clone()

        if item[5] is not None:
            # NOTE chen: we always work with the downscaled images/disps for optimization so store intrinsics at that scale
            self.intrinsics[index] = item[5] / s

        if len(item) > 6 and item[6] is not None:
            self.fmaps[index] = item[6]

        if len(item) > 7 and item[7] is not None:
            self.nets[index] = item[7]

        if len(item) > 8 and item[8] is not None:
            self.inps[index] = item[8]

        if len(item) > 9 and item[9] is not None:
            self.poses_gt[index] = item[9].to(self.poses_gt.device)

        if len(item) > 10 and item[10] is not None:
            self.static_masks[index] = item[10]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """index the depth video"""

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index > 0:
                index = self.counter.value + index
            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index],
            )

        return item

    def remove(self, index) -> None:
        """Given a list of indices, we want to reset these items to the initial values.

        Example use case: In the trajectory filler we use a small intermediate buffer at the end of the video
        to optimize intermediate poses before returning them. Keeping the overall video buffer clean afterwards should
        be a priority.
        """
        self.timestamp[index] = torch.zeros_like(self.timestamp[index], dtype=torch.float, device=self.device)
        self.images[index] = torch.zeros_like(self.images[index], dtype=torch.float, device=self.device)
        self.intrinsics[index] = torch.zeros_like(self.intrinsics[index], dtype=torch.float, device=self.device)

        zero_poses = torch.zeros_like(self.poses[index], dtype=torch.float, device=self.device)
        zero_poses[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.device)
        self.poses[index] = zero_poses
        self.poses_gt[index] = zero_poses

        self.disps[index] = torch.ones_like(self.disps[index], dtype=torch.float, device=self.device)
        self.disps_up[index] = torch.zeros_like(self.disps_up[index], dtype=torch.float, device=self.device)
        self.disps_sens[index] = torch.zeros_like(self.disps_sens[index], dtype=torch.float, device=self.device)
        self.disps_sens_up[index] = torch.zeros_like(self.disps_sens_up[index], dtype=torch.float, device=self.device)
        self.scales[index] = torch.ones_like(self.scales[index], dtype=torch.float, device=self.device)
        self.shifts[index] = torch.zeros_like(self.shifts[index], dtype=torch.float, device=self.device)

        self.fmaps[index] = torch.zeros_like(self.fmaps[index], dtype=torch.half, device=self.device)
        self.nets[index] = torch.zeros_like(self.nets[index], dtype=torch.half, device=self.device)
        self.inps[index] = torch.zeros_like(self.inps[index], dtype=torch.half, device=self.device)

        self.static_masks[index] = torch.ones_like(self.static_masks[index], dtype=torch.bool, device=self.device)

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)

    def pad_indices(self, indices: torch.Tensor, radius: int = 3) -> torch.Tensor:
        """Given a sequence of indices, we want to include surrounding indices as well within a radius.
        This is useful, as when we want to compute multi-view consistency we need to consider surrounding frames.
        Since we dont want do this for the whole map all the time, we only pad a given list of indices.
        """
        padded_indices = []
        # Dont pad past the sequence, because we might not have visited these frames at all
        last_frame = indices.max().item()
        for ix in indices:
            padded_indices.append(
                torch.arange(max(0, ix - radius), min(last_frame + 1, ix + radius + 1), device=indices.device)
            )
        padded_indices = torch.cat(padded_indices)
        return torch.unique(padded_indices)

    def dummy_filter(self, idx: Optional[torch.Tensor] = None):
        with self.get_lock():
            if idx is None:
                (dirty_index,) = torch.where(self.mapping_dirty.clone())
                dirty_index = dirty_index
            else:
                dirty_index = idx

            if len(dirty_index) == 0:
                return

            if self.upsampled:
                disps = torch.index_select(self.disps_up, 0, dirty_index).clone()
            else:
                disps = torch.index_select(self.disps, 0, dirty_index).clone()

        self.disps_clean[dirty_index] = disps
        self.filtered_id = max(dirty_index.max().item(), self.filtered_id)

    def filter_map(
        self,
        idx: Optional[torch.Tensor] = None,
        radius: int = 2,
        min_count: int = 2,
        bin_thresh: float = 0.1,
        min_disp_thresh: float = 0.01,
        unc_threshold: float = 0.1,
        use_multiview_consistency: bool = True,
        use_uncertainty: bool = True,
        return_mask: bool = False,
    ) -> None:
        """Filter the map based on consistency across multiple views and uncertainty.
        Normally this is done based on only a selected few views. We extend the selection with a local neighborhood
        to achieve a better estimate of consistent points.
        """

        with self.get_lock():
            if idx is None:
                (dirty_index,) = torch.where(self.mapping_dirty.clone())
                dirty_index = dirty_index
            else:
                dirty_index = idx

            if len(dirty_index) == 0:
                return

            # Check for multiview consistency not in the whole map, but also not only in a few local frames
            # -> Pad to neighborhoods, so we can get many consistent points
            dirty_index = self.pad_indices(dirty_index, radius=radius)

            if self.upsampled:
                disps = torch.index_select(self.disps_up, 0, dirty_index).clone()
                intrinsics = self.intrinsics[0] * self.scale_factor
                if use_uncertainty:
                    unc = torch.index_select(self.uncertainty_up, 0, dirty_index).clone()
            else:
                disps = torch.index_select(self.disps, 0, dirty_index).clone()
                intrinsics = self.intrinsics[0]
                if use_uncertainty:
                    unc = torch.index_select(self.uncertainty, 0, dirty_index).clone()

        mask = torch.ones_like(disps, dtype=torch.bool)
        if use_multiview_consistency:
            # Only take pixels where multiple points are consistent across views and they do not have an outlier disparity
            thresh = bin_thresh * torch.ones_like(disps.mean(dim=[1, 2]))
            if self.upsampled:
                count = droid_backends.depth_filter(self.poses, self.disps_up, intrinsics, dirty_index, thresh)
            else:
                count = droid_backends.depth_filter(self.poses, self.disps, intrinsics, dirty_index, thresh)
            mv_mask = (count >= min_count) & (disps > min_disp_thresh * disps.mean(dim=[1, 2], keepdim=True))
            mask = mask & mv_mask

        if use_uncertainty:
            unc_mask = unc > unc_threshold
            mask = mask & unc_mask

        disps[~mask] = 0.0  # Filter away invalid points
        self.disps_clean[dirty_index] = disps
        self.filtered_id = max(dirty_index.max().item(), self.filtered_id)

        if return_mask:
            return mask

    def get_mapping_item(self, index, use_gt=False, device="cuda:0"):
        """Get a part of the video to transfer to the Rendering module"""

        s = self.scale_factor
        with self.get_lock():
            if self.upsampled:
                image = self.images[index].clone().contiguous().to(device)  # [H, W, 3]
                static_mask = self.static_masks[index].clone().to(device)  # [H, W]
                intrinsics = self.intrinsics[0].clone().contiguous().to(device) * s  # [4]
                disp_prior = self.disps_sens_up[index].clone().to(device)  # [H, W]

                print((self.disps_sens_up.sum(dim=[1, 2])>0).sum())


            else:
                # Color is always stored in the original resolution, downsample here to match
                image = self.images[index, ..., int(s // 2 - 1) :: s, int(s // 2 - 1) :: s].clone()
                image = image.contiguous().to(device)  # [C, H // s, W // s]
                static_mask = self.static_masks[index, ..., int(s // 2 - 1) :: s, int(s // 2 - 1) :: s]
                static_mask = static_mask.contiguous().to(device)  # [H // s, W // s]
                intrinsics = self.intrinsics[0].clone().contiguous().to(device)  # [4]
                disp_prior = self.disps_sens[index].clone().to(device)  # [H // s, W // s]

            est_disp = self.disps_clean[index].clone().contiguous().to(device)  # [H, W]
            est_depth = torch.where(est_disp > 0, 1.0 / est_disp, est_disp)
            # Some modes dont have any disps_sens
            if self.cfg.mode in ["rgbd", "prgbd"]:
                depth_prior = torch.where(disp_prior > 0, 1.0 / disp_prior, disp_prior)  # Prior depth
            else:
                depth_prior = est_depth

            # [7, 1]
            if use_gt:
                c2w = lietorch.SE3(self.poses_gt[index].clone()).to(device)
            else:
                w2c = lietorch.SE3(self.poses[index].clone()).to(device)
                c2w = w2c.inv()
            return image, est_depth, depth_prior, intrinsics, c2w, static_mask

    def set_mapping_item(self, index: List[torch.Tensor], poses: List[torch.Tensor], depths: List[torch.Tensor]):
        """Set a part of the video from the Rendering module"""
        # Sanity check for when we did not render anything
        if (len(poses) == 0 and len(depths) == 0) or len(index) == 0:
            return

        # We may get only poses or only depths, so we need to check for both
        if len(poses) != 0:
            has_poses = True
            poses = torch.stack(poses)
            assert len(poses) == len(index), "Index should match the number of poses!"
        else:
            has_poses = False

        if len(depths) != 0:
            has_depths = True
            depths = torch.stack(depths)
            assert len(depths) == len(index), "Index should match the number of depths!"
            valid = depths > 0
        else:
            has_depths = False

        with self.get_lock():

            if has_depths:
                disps = torch.where(valid, 1.0 / depths, depths)
                disps.clamp_(min=1e-5)  # Sanity for optimization

                s = self.scale_factor
                # We work with the original resolution in Rendering
                if self.upsampled:
                    self.disps_up[index] = torch.where(
                        valid[:, 0], disps[:, 0].clone().detach().to(self.device), self.disps_up[index]
                    )
                    self.disps[index] = torch.where(
                        valid[:, 0, int(s // 2 - 1) :: s, int(s // 2 - 1) :: s],
                        disps[:, 0, int(s // 2 - 1) :: s, int(s // 2 - 1) :: s].clone().detach().to(self.device),
                        self.disps[index],
                    )
                # We work with downscaled resolution. This means, we need to upsample the disparities in tracking
                else:
                    self.disps[index] = torch.where(
                        valid[:, 0],
                        disps[:, 0].clone().detach().to(self.device),
                        self.disps[index],
                    )

            if has_poses:
                w2c = poses.clone().detach().to(self.device)  # [4, 4] homogenous matrix
                w2c_vec = matrix_to_lie(w2c)  # [7, 1] Lie element
                self.poses[index] = lietorch.SE3.InitFromVec(w2c_vec).vec()

        self.dirty[index] = True  # Mark frames for visualization

    @staticmethod
    def format_indices(ii, jj, device="cuda"):
        """to device, long, {-1}"""
        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)
        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device=device, dtype=torch.long).reshape(-1)
        jj = jj.to(device=device, dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        disps_up = cvx_upsample(self.disps[ix].unsqueeze(dim=-1), mask)  # [b, h, w, 1]
        self.disps_up[ix] = disps_up.squeeze()  # [b, h, w]

        uncertainty_up = cvx_upsample(self.uncertainty[ix].unsqueeze(-1), mask)  # [b, h, w, 1]
        self.uncertainty_up[ix] = uncertainty_up.squeeze()  # [b, h, w]

    def normalize(self):
        """normalize depth and poses"""
        with self.get_lock():
            cur_ix = self.counter.value
            s = self.disps[:cur_ix].mean()
            self.disps[:cur_ix] /= s
            self.poses[:cur_ix, :3] *= s  # [tx, ty, tz, qx, qy, qz, qw]
            self.dirty[:cur_ix] = True

    def reproject(self, ii, jj):
        """project points from ii -> jj"""
        ii, jj = DepthVideo.format_indices(ii, jj, self.device)
        Gs = lietorch.SE3(self.poses[None, ...])

        coords, valid_mask = pops.general_projective_transform(
            poses=Gs,
            depths=self.disps[None, ...],
            intrinsics=self.intrinsics[None, ...],
            ii=ii,
            jj=jj,
            model_id=self.model_id,
            jacobian=False,
            return_depth=False,
        )

        return coords, valid_mask

    def reduce_uncertainties(self, weights: torch.Tensor, ii: torch.Tensor, strategy: str = "avg") -> torch.Tensor:
        """Given the factor graph for the scene, we optimize poses at different camera locations.
        Each location/pose is a node in the graph and can have multiple edges to other nodes.
        For each edge we have a predicted uncertainty weight map given the learned feature correlations.
        We are interested in viewing these uncertainties or use them later on, as they should correlate
        with moving objects and pixels/points that do not contribute to a good reconstruction.
        Given the indices of an optimization window we reduce edges to get a single uncertainty estimate
        for each frame.

        args:
        ---
        weight [torch.Tensor]: Weight tensor of shape [len(nodes), 2, ht // 8, wd // 8]. Optimization weights for bundle adjustment.
            Each point is a vector [u_x, u_y] \in [0, 1], which measures the uncertainty for x- and y-components.
        ii [torch.Tensor]: Indices of source nodes (We go from i to j, i.e. we have edges e_ij) with same length as jj.
        strategy [str]: How to reduce across edges. Choices: (avg, max). Given multiple uncertainty weight maps for
            each pixel, it is unclear how to correctly reduce this. In the end these are optimal to compute correct camera motion
            and static scene maps. Which edge contributes more to this goal is not straight-forward.
        """
        frames = ii.unique()
        idx = []
        for frame in frames:
            idx.append(frame == ii)

        frame_weights = [weights[ix] for ix in idx]
        if strategy == "avg":
            reduced = [weight.mean(dim=0) for weight in frame_weights]
        elif strategy == "max":
            reduced = [weight.max(dim=0) for weight in frame_weights]
        else:
            raise Exception("Invalid reduction strategy: {}! Use either 'avg' or 'max'".format(strategy))
        return torch.stack(reduced), frames

    def reset_prior(self) -> None:
        """Adjust the prior according to optimized scales, then reset the scale parameters.
        This makes it possible to continue to optimize them, but still provide the correct disps_sens to the
        CUDA kernel for global BA in the backend.
        """
    
        # Rescale the external disparities
        #print((self.disps_sens_up.sum(dim=[1, 2])>0).sum())

        self.disps_sens = self.disps_sens * self.scales[:, None, None] + self.shifts[:, None, None]
        self.disps_sens_up = self.disps_sens_up * self.scales[:, None, None] + self.shifts[:, None, None]
        #self.disps_sens.clamp_(min=1e-5)
        # Reset the scale and shift parameters to initial state
        self.scales = torch.ones_like(self.scales, device=self.device)
        self.shifts = torch.zeros_like(self.shifts, device=self.device)
        #print((self.disps_sens_up.sum(dim=[1, 2])>0).sum())



    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """frame distance metric, where distance = sqrt((u(ii) - u(jj->ii))^2 + (v(ii) - v(jj->ii))^2)"""
        return_matrix = False
        N = self.counter.value
        if ii is None:
            return_matrix = True
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")

        ii, jj = DepthVideo.format_indices(ii, jj)

        intrinsic_common_id = 0  # we assume the intrinsic within one scene is the same
        if bidirectional:
            poses = self.poses[: self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[intrinsic_common_id], ii, jj, beta, self.model_id
            )

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[intrinsic_common_id], jj, ii, beta, self.model_id
            )

            d = 0.5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[intrinsic_common_id], ii, jj, beta, self.model_id
            )

        if return_matrix:
            return d.reshape(N, N)

        return d

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, iters=2, lm=1e-4, ep=0.1, motion_only=False):
        """Wrapper for dense bundle adjustment. This is used both in Frontend and Backend."""

        intrinsic_common_id = 0  # we assume the intrinsic within one scene is the same

        with self.get_lock():

            # Store the uncertainty maps for source frames, that will get updated
            uncertainty, idx = self.reduce_uncertainties(weight, ii)
            # Uncertainties are for [x, y] directions -> Take norm to get single scalar
            self.uncertainty[idx] = torch.norm(uncertainty, dim=1)

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            # FIXME chen: I dont understand why the backend cannot use the scaled disps_sens directly as well
            if self.optimize_scales:
                disps_sens = torch.zeros_like(self.disps_sens, device=self.device)
            else:
                disps_sens = self.disps_sens

            droid_backends.ba(
                self.poses,
                self.disps,
                self.intrinsics[intrinsic_common_id],
                disps_sens,
                target,
                weight,
                eta,
                ii,
                jj,
                t0,
                t1,
                iters,
                self.model_id,
                lm,
                ep,
                motion_only,
                self.opt_intr,
            )
            self.disps.clamp_(min=1e-5)  # Always make sure that Disparities are non-negative!!!
            # Reassigning intrinsics after optimization
            if self.opt_intr:
                self.intrinsics[: self.counter.value] = self.intrinsics[intrinsic_common_id]

    def linear_align_prior(self, eps: float=0.05) -> None:
        """Do a linear alignmnet between the prior and the current map after initialization.
        This strategy is used to align the scales and shifts before running Bundle Adjustment.

        NOTE chen: This is a very different objective than the optical flow one, i.e. there is no guarantee that this helps to converge to the right monocular scale!
        This also can lead to changes, that the BA optimization has to correct back again, thus wasting compute.
        """

        # Filter the map, but use self.disps for scale_optimization
        valid_d = self.filter_map(
            idx=torch.arange(self.counter.value - 1, device=self.device),
            radius=1,
            use_multiview_consistency=True,
            bin_thresh=0.005,
            min_count=2,
            return_mask=True,
        )

        if valid_d.sum() < 300:
            print(colored("Not enough valid points for linear optimization!", "red"))
            return

        if self.upsampled:
            # Map gets filtered at highest resolution, but scale optimization always uses self.disps_sens not self.disps_sens_up
            s = self.scale_factor
            valid_d = valid_d[..., int(s // 2 - 1) :: s, int(s // 2 - 1) :: s]

        scale_t, shift_t, error_t = align_scale_and_shift(
            self.disps_sens[: self.counter.value - 1], self.disps[: self.counter.value - 1], valid_d
        )

        scale_t[torch.isnan(scale_t)], shift_t[torch.isnan(shift_t)] = 1.0, 0.0
        valid_scale = scale_t > eps
        scale_t[~valid_scale] = 1.0

        #print("Frames with wrong scale: ", (~valid_scale).sum().item())

        scale_t[torch.isnan(scale_t)], shift_t[torch.isnan(shift_t)] = 1.0, 0.0
        self.scales[: self.counter.value - 1], self.shifts[: self.counter.value - 1] = scale_t, shift_t
        self.reset_prior()  # Reset the prior and update disps_sens to fit the map

    def ba_prior(self, target, weight, eta, ii, jj, t0=1, t1=None, iters=2, lm=1e-4, ep=0.1, alpha: float = 5e-3):
        """Bundle adjustment over structure with a scalable prior.

        We keep the poses fixed, since this would create an unnecessary ambiguity and can make the system unstable!
        We optimize scale and shift parameters on top of the scene disparity.
        """
        with self.get_lock():
            print((self.disps_sens_up.sum(dim=[1, 2])>0).sum())

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            # Store the uncertainty maps for source frames, that will get updated
            uncertainty, idx = self.reduce_uncertainties(weight, ii)
            # Uncertainties are for [x, y] directions -> Take norm to get single scalar
            self.uncertainty[idx] = torch.norm(uncertainty, dim=1)

            # Precondition


            self.linear_align_prior()  # Align priors to the current (monocular) map with scale and shift from linear optimization

            # Block coordinate descent optimization
            for i in range(iters):
                # Sanity check for non-negative disparities
                self.disps.clamp_(min=1e-5), self.disps_sens.clamp_(min=1e-5)
                # Motion only Bundle Adjustment (MoBA)
                droid_backends.ba(
                    self.poses,
                    self.disps,
                    self.intrinsics[0],
                    self.disps_sens,
                    target,
                    weight,
                    eta,
                    ii,
                    jj,
                    t0,
                    t1,
                    1,
                    self.model_id,
                    lm,
                    ep,
                    True,
                    False,
                )
                # Joint Depth and Scale Adjustment(JDSA)
                bundle_adjustment(
                    target,
                    weight,
                    eta,
                    self.poses,
                    self.disps,
                    self.intrinsics,
                    ii,
                    jj,
                    t0,
                    t1,
                    self.disps_sens,
                    self.scales,
                    self.shifts,
                    iters=1,
                    lm=lm,
                    ep=ep,
                    scale_prior=True,
                    structure_only=True,
                    alpha=alpha,
                )
                # After optimizing the prior, we need to update the disps_sens and reset the scales
                # only then can we use global BA and intrinsics optimization with the CUDA kernel later
                self.reset_prior()

            print((self.disps_sens_up.sum(dim=[1, 2])>0).sum())
