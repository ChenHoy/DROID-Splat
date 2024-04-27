import ipdb
from copy import deepcopy

import torch
from torch.multiprocessing import Value
import lietorch
import droid_backends

from .droid_net import cvx_upsample
from .geom import projective_ops as pops
from .geom.ba import BA_prior, MoBA, BA, bundle_adjustment


class DepthVideo:
    """
    Data structure of multiple buffers to keep track of indices, poses, disparities, images, external disparities and more
    """

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.device = device = args.device

        ### Intrinsics / Calibration ###
        # NOTE you almost always use a pinhole model
        if args.camera_model == "pinhole":
            self.n_intr = 4
            self.model_id = 0
        elif args.camera_model == "mei":
            self.n_intr = 5
            self.model_id = 1
        else:
            raise Exception("Camera model not implemented! Choose either pinhole or mei model.")
        self.opt_intr = args.opt_intr

        self.counter = Value("i", 0)
        self.ready = Value("i", 0)
        self.mapping = Value("i", 0)
        # NOTE we have multiple lock to avoid deadlocks between bundle adjustment and frontend loop closure
        self.ba_lock = {"dense": Value("i", 0), "loop": Value("i", 0)}
        self.global_ba_lock = Value("i", 0)

        self.ht = ht = cfg["cam"]["H_out"]
        self.wd = wd = cfg["cam"]["W_out"]
        self.stereo = cfg["mode"] == "stereo"
        c = 1 if not self.stereo else 2
        self.scale_factor = s = 8
        buffer = cfg["tracking"]["buffer"]

        ### state attributes -> Raw map ###
        self.timestamp = torch.zeros(buffer, device=device, dtype=torch.float).share_memory_()
        self.dirty = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device=device, dtype=torch.float)
        self.intrinsics = torch.zeros(buffer, 4, device=device, dtype=torch.float).share_memory_()
        self.red = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device=device, dtype=torch.float).share_memory_()  # w2c quaterion
        self.poses_gt = torch.zeros(buffer, 4, 4, device=device, dtype=torch.float).share_memory_()  # c2w matrix
        self.disps = torch.ones(buffer, ht // s, wd // s, device=device, dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        # Estimated Uncertainty weights for Optimization reduced for each node from factor graph
        self.uncertainty = torch.zeros(buffer, ht // 8, wd // 8, device=device, dtype=torch.float)
        self.uncertainty_up = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()

        self.disps_sens = torch.zeros(buffer, ht // s, wd // s, device=device, dtype=torch.float).share_memory_()
        # Scale and shift parameters for ambiguous monocular depth
        self.optimize_scales = cfg["mode"] == "prgbd"  # Optimze the scales and shifts for Pseudo-RGBD mode
        self.scales = torch.ones(buffer, device=device, dtype=torch.float).share_memory_()
        self.shifts = torch.zeros(buffer, device=device, dtype=torch.float).share_memory_()
        # In case we have an external groundtruth
        # TODO chen: is this ever used right now?
        self.depths_gt = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht // s, wd // s, dtype=torch.half, device=device).share_memory_()
        self.nets = torch.zeros(buffer, 128, ht // s, wd // s, dtype=torch.half, device=device).share_memory_()
        self.inps = torch.zeros(buffer, 128, ht // s, wd // s, dtype=torch.half, device=device).share_memory_()

        ### Initialize poses to identity transformation
        self.poses[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)
        self.poses_gt[:] = torch.eye(4, dtype=torch.float, device=device)

        ### Additional flags for multi-view filter -> Clean map for rendering ###
        self.disps_filtered = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        self.mask_filtered = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        # FIXME do we already need this object? These are exactly the same as self.poses
        self.poses_filtered = torch.zeros(buffer, 7, device=device, dtype=torch.float).share_memory_()  # w2c quaterion
        self.poses_filtered[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)
        # Keep track of which frames have been filtered
        self.filtered_id = torch.tensor([-1], dtype=torch.int, device=device).share_memory_()
        self.update_priority = torch.zeros(buffer, device=device, dtype=torch.float).share_memory_()
        self.bound = torch.zeros(1, 3, 2, device=device, dtype=torch.float).share_memory_()
        # pose compensation from virtual to real
        self.pose_compensate = torch.zeros(1, 7, dtype=torch.float, device=device).share_memory_()
        self.pose_compensate[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)

    def get_lock(self):
        return self.counter.get_lock()

    def get_ba_lock(self, ba_type):
        return self.ba_lock[ba_type].get_lock()

    def get_mapping_lock(self):
        return self.mapping.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        self.timestamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            self.depths_gt[index] = item[4]
            depth = item[4][..., 3::8, 3::8]
            # Clamp negative depth to a small value
            depth.clamp_(min=0.001)
            self.disps_sens[index] = torch.where(depth > 0, 1.0 / depth, depth)
            # If we have a prior, we initialize disparity with the prior
            self.disps[index] = self.disps_sens[index].clone()

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]

        if len(item) > 9 and item[9] is not None:
            self.poses_gt[index] = item[9].to(self.poses_gt.device)

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

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)

    def get_bound(self):
        with self.mapping.get_lock():
            bound = self.bound[0]

        return bound

    def get_mapping_item(self, index, device="cuda:0", decay=0.1):
        """dense mapping operations to transfer a part of the video to the Rendering module"""

        with self.mapping.get_lock():
            image = self.images[index].clone().permute(1, 2, 0).contiguous().to(device)  # [h, w, 3]
            mask = self.mask_filtered[index].clone().to(device)
            est_disp = self.disps_filtered[index].clone().to(device)  # [h, w]
            est_depth = 1.0 / (est_disp + 1e-7)

        # origin alignment
        w2c = lietorch.SE3(self.poses_filtered[index].clone()).to(device)  # Tw(droid)_to_c
        c2w = lietorch.SE3(self.pose_compensate[0].clone()).to(w2c.device) * w2c.inv()
        c2w = c2w.matrix()  # [4, 4]

        gt_c2w = self.poses_gt[index].clone().to(device)  # [4, 4]

        depth = est_depth
        # gt_depth = self.depths_gt[index].clone().to(device)  # [h, w]
        # depth = gt_depth

        # if updated by mapping, the priority is decreased to lowest level, i.e., 0
        self.update_priority[index] *= decay

        return image, depth, c2w, gt_c2w, mask

    # TODO Backpropagate an updated map from Renderer to the DepthVideo
    # TODO chen: this needs to assume that the Renderer actually produces a better state than from the feature maps / optical flow estimates
    def set_item_from_mapping(self, index, pose=None, depth=None):
        with self.get_lock():
            pass

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
        self.disps_sens = self.disps_sens * self.scales[:, None, None] + self.shifts[:, None, None]
        # Always ensure that disparities are non-negative!!!
        self.disps_sens.clamp_(min=0.001)
        # Reset the scale and shift parameters to initial state
        self.scales = torch.ones_like(self.scales)
        self.shifts = torch.zeros_like(self.shifts)

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

    def ba(
        self, target, weight, eta, ii, jj, t0=1, t1=None, iters=2, lm=1e-4, ep=0.1, motion_only=False, ba_type=None
    ):
        """Wrapper for dense bundle adjustment. This is used both in Frontend and Backend."""

        intrinsic_common_id = 0  # we assume the intrinsic within one scene is the same
        lock = self.get_lock() if ba_type is None else self.get_ba_lock(ba_type)
        with lock:

            # Store the uncertainty maps for source frames, that will get updated
            uncertainty, idx = self.reduce_uncertainties(weight, ii)
            # Uncertainties are for [x, y] directions -> Take norm to get single scalar
            self.uncertainty[idx] = torch.norm(uncertainty, dim=1)

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            droid_backends.ba(
                self.poses,
                self.disps,
                self.intrinsics[intrinsic_common_id],
                self.disps_sens,
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
            self.disps.clamp_(min=0.001)  # Always make sure that Disparities are non-negative!!!

    def ba_prior(self, target, weight, eta, ii, jj, t0=1, t1=None, iters=2, lm=1e-4, ep=0.1):
        """Bundle adjustment over structure with a scalable prior.

        We keep the poses fixed, since this would create an unnecessary ambiguity and can make the system unstable!
        We optimize scale and shift parameters on top of the scene disparity.
        """
        lock = self.get_lock()
        with lock:
            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            # Store the uncertainty maps for source frames, that will get updated
            uncertainty, idx = self.reduce_uncertainties(weight, ii)
            # Uncertainties are for [x, y] directions -> Take norm to get single scalar
            self.uncertainty[idx] = torch.norm(uncertainty, dim=1)

            # MoBA
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
                iters,
                self.model_id,
                lm,
                ep,
                True,
                self.opt_intr,
            )

            # JDSA
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
                iters=iters + 2,
                lm=lm,
                ep=ep,
                scale_prior=True,
                structure_only=True,
            )
