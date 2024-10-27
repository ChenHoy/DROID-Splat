import os
import ipdb
from copy import deepcopy
from typing import List, Dict, Optional, Tuple
import time
import ipdb
import math
import gc
from termcolor import colored
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
from torch.utils.data import WeightedRandomSampler
from lietorch import SE3

import numpy as np
import cv2
import matplotlib.pyplot as plt

from .gaussian_splatting.gui import gui_utils
from .gaussian_splatting.eval_utils import EvaluatePacket
from .gaussian_splatting.utils.general_utils import random_subsample_mask
from .gaussian_splatting.gaussian_renderer import render
from .gaussian_splatting.scene.gaussian_model import GaussianModel
from .gaussian_splatting.camera_utils import Camera
from .losses import mapping_rgbd_loss, plot_losses

from .gaussian_splatting.pose_utils import update_pose
from .gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, focal2fov, getWorld2View2
from .utils.multiprocessing_utils import clone_obj
from .geom import lie_to_matrix
from .trajectory_filler import PoseTrajectoryFiller


"""
Mapping based on 3D Gaussian Splatting. 
We create new Gaussians based on incoming new views and optimize them for dense photometric consistency. 
Since they are initialized with the 3D locations of a VSLAM system, this process converges really fast. 

NOTE this could a be standalone SLAM system itself, but we use it on top of an optical flow based SLAM system.
"""


class GaussianMapper(object):
    """
    SLAM from Rendering with 3D Gaussian Splatting.
    """

    def __init__(self, cfg, slam, gui_qs=None):
        self.cfg = cfg
        self.slam = slam
        self.video = slam.video
        self.device = cfg.device
        self.mode = cfg.mode
        self.evaluate = cfg.evaluate
        self.output = slam.output
        self.delay = cfg.mapping.delay  # Delay between tracking and mapping
        self.warmup = cfg.mapping.warmup
        self.batch_mode = cfg.mapping.online_opt.batch_mode  # Take a batch of all unupdated frames at once

        # Given an external mask for dyn. objects, remove these from the optimization
        self.filter_dyn = cfg.get("with_dyn", False)

        self.save_renders = cfg.mapping.save_renders

        self.opt_params = cfg.mapping.opt_params  # Optimizer
        self.loss_params = cfg.mapping.loss  # Losses

        self.pipeline_params = cfg.mapping.pipeline_params
        self.sh_degree = 3 if cfg.mapping.use_spherical_harmonics else 0
        # Change the downsample factor for initialization depending on cfg.tracking.upsample, so we always have points
        if not self.cfg.tracking.upsample:
            cfg.mapping.input.pcd_downsample_init /= 8
            cfg.mapping.input.pcd_downsample /= 8

        # Online Tracker
        self.update_params = cfg.mapping.online_opt
        self.mapping_iters = self.update_params.iters
        # Which frames to optimize on
        self.n_last_frames = self.update_params.n_last_frames  # Consider the recent n frames
        # Consider additional n random frames (This helps against catastrophic forgetting)# Consider additional n random frames (This helps against catastrophic forgetting)
        self.n_rand_frames = self.update_params.n_rand_frames
        # How to filter the Tracking map before Rendering
        self.filter_params = self.update_params.filter

        # Offline Refinement
        self.refine_params = cfg.mapping.refinement

        # Feedback the map to the Tracker if wanted
        self.feedback_params = cfg.mapping.feedback
        self.feedback_disps = cfg.mapping.feedback.disps
        if self.feedback_disps:
            self.info("Feeding back scene geometry from Renderer -> Tracker!")
        self.feedback_poses = cfg.mapping.feedback.poses
        if self.feedback_poses:
            self.info("Feeding back local pose graph from Renderer -> Tracker!")
        if self.feedback_poses and not (self.update_params.optimize_poses or self.refine_params.optimize_poses):
            self.info(
                "Warning. You are feeding back poses from Mapper to Tracker without optimizing them (either during Tracking or Refinement)!"
            )

        self.gaussians = GaussianModel(self.sh_degree, config=cfg.mapping.input)
        self.gaussians.init_lr(self.opt_params.init_lr)
        self.gaussians.training_setup(self.opt_params)

        bg_color = [1, 1, 1]  # White background
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.z_near = 0.0001
        self.z_far = 10000.0

        if gui_qs is not None:
            self.q_main2vis = gui_qs
            self.use_gui = True
        else:
            self.use_gui = False

        self.last_idx = 0
        self.cameras, self.new_cameras = [], []
        self.loss_list, self.iteration_info = [], []
        self.initialized = False
        self.projection_matrix = None

        self.n_optimized = {}  # Keep track how many times a keyframe was optimized
        self.last_frame_loss = {}  # Keep track of the last loss for each frame
        # Memoize the mapping of Gaussian indices to video indices
        # (this maps cam.uid -> position in video buffer)
        # Since we only store keyframes inside the video buffers, we reassign the mapping ones we add non-keyframes cameras during refinement
        self.cam2buffer, self.buffer2cam = {}, {}
        self.count = 0

    def info(self, msg: str):
        print(colored("[Gaussian Mapper] " + msg, "magenta"))

    def save_render(self, cam: Camera, render_path: str) -> None:
        """Save a rendered frame"""
        render_pkg = render(cam, self.gaussians, self.pipeline_params, self.background, device=self.device)
        rgb = np.uint8(255 * render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(render_path, bgr)

    def __len__(self):
        """Return the number of camera frames in the scene."""
        return len(self.cameras)

    def camera_from_video(self, idx):
        """Extract Camera objects from a part of the video."""
        if self.video.disps_clean[idx].sum() < 1:  # Sanity check:
            self.info(f"Warning. Trying to intialize from empty frame {idx}!")
            return None

        color, depth, depth_prior, intrinsics, w2c_lie, stat_mask = self.video.get_mapping_item(idx, self.device)
        w2c = lie_to_matrix(w2c_lie)
        return self.camera_from_frame(idx, color, w2c, intrinsics, depth, mask=stat_mask)

    def camera_from_frame(
        self,
        idx: int,
        image: torch.Tensor,
        w2c: torch.Tensor,
        intrinsics: torch.Tensor,
        depth_init: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """Given the image, depth, intrinsic and pose, creates a Camera object.
        The depth for supervision and initialization does not need to be the same, e.g. we could initialize
        the Gaussians with a sparse, but certain depth map and supervise with a dense prior.
        We also use an optional mask for the objective function, e.g. for supervising only the static parts of the scene
        explainable by the camera motion."""
        fx, fy, cx, cy = intrinsics

        height, width = image.shape[-2:]
        fovx, fovy = focal2fov(fx, width), focal2fov(fy, height)
        projection_matrix = getProjectionMatrix2(self.z_near, self.z_far, cx, cy, fx, fy, width, height)
        projection_matrix = projection_matrix.transpose(0, 1).to(device=self.device)

        return Camera(
            idx,
            image.contiguous(),
            depth_init,
            depth,
            w2c,
            projection_matrix,
            (fx, fy, cx, cy),
            (fovx, fovy),
            (height, width),
            device=self.device,
            mask=mask,
        )

    def get_new_cameras(self, delay=0):
        """Get all new cameras from the video."""
        # Only add a batch of cameras in batch_mode
        if self.batch_mode:
            to_add = range(self.last_idx, self.cur_idx - delay)
            to_add = to_add[: self.update_params.batch_size]
        else:
            to_add = range(self.last_idx, self.cur_idx - delay)

        for idx in to_add:
            color, depth, depth_prior, intrinsics, w2c_lie, stat_mask = self.video.get_mapping_item(
                idx, device=self.device
            )
            w2c = lie_to_matrix(w2c_lie)

            # HOTFIX Sanity check for when we dont have any good depth
            if (depth > 0).sum() < 100:
                depth = None
            if (depth_prior > 0).sum() < 100:
                depth_prior = None
            cam = self.camera_from_frame(
                idx, color, w2c, intrinsics, depth_init=depth, depth=depth_prior, mask=stat_mask
            )

            # Insert camera into index mapping
            if cam.uid not in self.cam2buffer:
                self.cam2buffer[cam.uid] = cam.uid
                self.buffer2cam[cam.uid] = cam.uid

            if cam.uid not in self.n_optimized:
                self.n_optimized[cam.uid] = 0

            self.new_cameras.append(cam)

    def get_pose_optimizer(self, frames: List) -> torch.optim.Optimizer:
        """Creates an optimizer for the camera poses for all provided frames.
        Since we usually keep the first pose fixed in SLAM, we manually force the require.grad=False in Camera()!
        """
        opt_params = []
        already_in_optimizer = []
        for cam in frames:
            # Dont add duplicates in case we have multiple samples of the same frame in a batch!
            # (this can happen during refinement)
            if cam.uid in already_in_optimizer:
                continue
            opt_params.append(
                {"params": [cam.cam_rot_delta], "lr": self.opt_params.cam_rot_delta, "name": "rot_{}".format(cam.uid)}
            )
            opt_params.append(
                {
                    "params": [cam.cam_trans_delta],
                    "lr": self.opt_params.cam_trans_delta,
                    "name": "trans_{}".format(cam.uid),
                }
            )
            opt_params.append({"params": [cam.exposure_a], "lr": 0.01, "name": "exposure_a_{}".format(cam.uid)})
            opt_params.append({"params": [cam.exposure_b], "lr": 0.01, "name": "exposure_b_{}".format(cam.uid)})
            already_in_optimizer.append(cam.uid)

        return torch.optim.Adam(opt_params)

    def frame_updater(self, delay=0):
        """Gets the list of frames and updates the depth and pose based on the video.

        NOTE: This assumes, that optical flow based tracking overall is more reliable for sparse supervision
        We only use the Renderer for great scene representation and potential densification / correction

        NOTE chen: in some cases we might have keyframes & non-keyframes in self.cameras. We can distinguish keyframes by using
        index mapping since this is a unique mapping from cam.uid to the position in the video buffer.
        """
        all_cameras = self.cameras + self.new_cameras
        with self.video.get_lock():
            (dirty_index,) = torch.where(self.video.mapping_dirty.clone())
            # Only update up to the current frame in Mapper
            dirty_index = dirty_index[dirty_index < self.cur_idx - delay]

        # Check if the dirty indices from video buffer are also in our cam2buffer as values
        # -> Only update already inserted cameras
        to_update = dirty_index[
            torch.isin(dirty_index, torch.tensor(list(self.cam2buffer.values()), device=self.device))
        ]

        for idx in to_update:
            # TODO can the stat_mask potentially change as well?
            color, depth, depth_prior, intrinsics, w2c_lie, stat_mask = self.video.get_mapping_item(
                idx, device=self.device
            )
            w2c = lie_to_matrix(w2c_lie)

            cam = all_cameras[self.buffer2cam[idx.item()]]
            # update intrinsics in case we use opt_intrinsics
            cam.update_intrinsics(intrinsics, color.shape[-2:], self.z_near, self.z_far)
            if self.mode == "prgbd":
                cam.depth_prior = depth_prior.detach()  # Update prior in case of scale_change
            cam.depth = depth.detach()
            R = w2c[:3, :3].unsqueeze(0).detach()
            T = w2c[:3, 3].detach()
            cam.update_RT(R, T)

        self.video.mapping_dirty[to_update] = False

    def get_nonkeyframe_cameras(self, stream, trajectory_filler, batch_size: int = 16) -> List[Camera]:
        """SLAM systems operate on keyframes. This is good enough to build a good map,
        but we can get even finer details when including intermediate keyframes as well!
        Since we dont store all frames when iterating over the datastream, this requires
        reiterating over the stream again and interpolating between poses.

        NOTE Use this only after tracking finished for refining!
        NOTE: If you optimize poses with the Renderer, make sure to feedback
        the latest state of self.video as this is attached to trajectory_filler
        """
        all_poses, all_timestamps = trajectory_filler(stream, batch_size=batch_size, return_tstamps=True)
        # NOTE chen: this assumes that cam.uid will correspond to the position in the video buffer
        # this will later change, where cam.uid will simply correspond to the global timestamp after adding all the cameras
        already_mapped = [int(self.video.timestamp[cam.uid]) for cam in self.cameras]
        s = self.video.scale_factor
        if self.video.upsampled:
            intrinsics = self.video.intrinsics[0].to(self.device) * s
        else:
            intrinsics = self.video.intrinsics[0].to(self.device)

        new_cams = []
        for w2c, timestamp in tqdm(zip(all_poses, all_timestamps)):
            if timestamp in already_mapped:
                continue

            w2c = w2c.matrix()  # [4, 4]
            # TODO chen: if this blows up memory too much, dont use depth during refinement!
            if self.mode in ["rgbd", "prgbd"]:
                depth = stream._get_depth(timestamp).clone().contiguous().to(self.device)
            else:
                depth = None
            color = stream._get_image(timestamp).clone().squeeze(0).contiguous().to(self.device)
            if not self.video.upsampled:
                color = color[..., int(s // 2 - 1) :: s, int(s // 2 - 1) :: s]
                if depth is not None:
                    depth = depth[int(s // 2 - 1) :: s, int(s // 2 - 1) :: s]
            cam = self.camera_from_frame(timestamp, color, w2c, intrinsics, depth_init=depth, depth=depth)
            new_cams.append(cam)

        return new_cams

    def reanchor_gaussians(self, indices: torch.Tensor | List[int], delta_pose: torch.Tensor):
        """After a large map change, we need to reanchor the Gaussians. For this purpose we simply measure the
        rel. pose change for indidividual frames and check for large updates. We can then simply apply the rel. transform
        to the respective Gaussians.

        NOTE indices are positions in the video buffer since we reanchor after updates from video.ba()
        """
        updated_cams = []
        for idx, pose in zip(indices, delta_pose):
            # We have never mapped this frame before
            if int(idx) not in self.buffer2cam:
                continue
            else:
                self.gaussians.reanchor(self.buffer2cam[int(idx)], pose)
                # NOTE chen: We append to our camera list in consecutive order, i.e. this should normally be sorted!
                # this is not a given though! be cautious, e.g. during refinement the list changes due to insertion of non-keyframes
                cam = self.cameras[self.buffer2cam[int(idx)]]
                updated_cams.append(cam)

        # Add the kf from self.cameras[idx] to updated cameras for GUI
        if self.use_gui:
            self.q_main2vis.put_nowait(
                gui_utils.GaussianPacket(
                    gaussians=clone_obj(self.gaussians), keyframes=[cam.detached() for cam in updated_cams]
                )
            )

    def plot_masked_image(self, img: torch.Tensor, mask: torch.Tensor, title: str = "Masked Image"):
        import matplotlib.pyplot as plt

        img = img.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        mask = mask.squeeze().detach().cpu().numpy()
        img[mask] = 0
        plt.imshow(img)
        plt.axis("off")
        plt.title(title)
        plt.show()

    def densify_holes(self, idx: int, mask: torch.Tensor, downsample_factor: float = 1.0) -> None:
        """When initializing the map based on multi-view filtered depths, we often have holes
        in areas that are only visible in a single camera. Since we usually have access to a semi-reliable dense
        depth map (either from SLAM or an external sensor), we can simply patch these holes based on a reference.

        NOTE idx here is the position in our List[Camera] of the Renderer! You can check for valid keyframes by looking at the cam2buffer mapping.
        """
        cam = self.cameras[idx]
        # Only do this for actual keyframes (in case we added others), because we need a reference from Tracking
        if idx not in self.cam2buffer:
            return

        idx_in_video = self.cam2buffer[idx]
        # We have a dense depth prior that is reliable in rgbd mode
        if self.mode == "rgbd":
            dense_disps_ref = self.video.disps_sens_up[idx_in_video].clone().cpu()
            valid = dense_disps_ref > 0
            dense_depth_ref = torch.where(valid, 1.0 / dense_disps_ref, dense_disps_ref)
        # We have at least a scaled prior (from an external source)
        elif self.mode == "prgbd":
            dense_disps_ref = self.video.disps_sens_up[idx_in_video].clone().cpu()
            valid = dense_disps_ref > 0
            dense_depth_ref = torch.where(valid, 1.0 / dense_disps_ref, dense_disps_ref)
        # We have a dense depth reference in the video (from Tracking)
        elif self.video.upsampled:
            dense_disps_ref = self.video.disps_up[idx_in_video].clone().cpu()
            valid = dense_disps_ref > 0
            dense_depth_ref = torch.where(valid, 1.0 / dense_disps_ref, dense_disps_ref)
        else:
            # Get the average scale in this keyframe from the Gaussians
            # Densify the hole from this new point cloud
            # NOTE chen: since holes are usually further away it makes sense to take > median
            avg_depth = self.gaussians.get_avg_scale([idx], factor=1.5)  # Take factor*median
            dense_depth_ref = torch.ones_like(mask, device="cpu") * avg_depth

        # Densify based on the mask
        self.gaussians.extend_from_pcd_seq(
            cam,
            kf_id=idx,
            init=False,
            mask=mask,
            depthmap=dense_depth_ref.numpy(),
            downsample_factor=downsample_factor,
        )

    def get_ram_usage(self) -> Tuple[float, float]:
        free_mem, total_mem = torch.cuda.mem_get_info(device=self.device)
        used_mem = 1 - (free_mem / total_mem)
        return used_mem, free_mem

    def map_refinement(self) -> None:
        """Refine the map with color only optimization. Instead of going over last frames, we always select random frames from the whole map."""

        def draw_random_batch(
            kf_cams: List[Camera],
            batch_size: int = 16,
            weights_kf: Optional[List[float]] = None,
            nonkf_cams: Optional[List[Camera]] = None,
            kf_always: Optional[float] = None,
        ) -> List[Camera]:
            """Draw a random mini batch. If only keyframes are passed, we draw a random batch from them.
            If importance weights are provided, we assign a higher probability to certain keyframes to be drawn.

            If both key- and nonkey-frames are provided, we draw a random batch from all frames. Since usually only the keyframes
            have an attached depth map, you may want to have a certain percentage of keyframes in the batch at all times.
            """

            def draw_frames(cams, n_samples, weights: Optional[List[float]] = None):
                # Sanity check for when the batch size is bigger than our number of keyframes
                if n_samples >= len(cams):
                    return cams
                
                if weights is not None:
                    idx = list(WeightedRandomSampler(weights, n_samples, replacement=False))
                else:
                    idx = np.random.choice(len(cams), n_samples, replace=False)
                return [cams[i] for i in idx]

            if nonkf_cams is None:  # We only draw keyframes
                n_kf = batch_size
                batch = draw_frames(kf_cams, n_kf, weights=weights_kf)
            elif kf_always is not None:  # We draw a fixed ratio of keyframes and non-keyframes
                n_kf = int(batch_size * kf_always)
                n_else = batch_size - n_kf
                keyframes = draw_frames(kf_cams, n_kf, weights=weights_kf)
                others = draw_frames(nonkf_cams, n_else)
                batch = keyframes + others
            else:  # We draw uniformly from all frames
                if weights_kf is not None:
                    print(
                        "Warning. Importance sampling is ignored when we sample from all frames! Define a min. number of keyframes per batch!"
                    )
                batch = draw_frames(kf_cams + nonkf_cams, batch_size)

            return batch

        def draw_random_neighborhood_batch(
            cams: List[Camera], batch_size: int = 16, neighborhood_size: int = 5
        ) -> List[Camera]:
            """Draw multiple mini batches of neighborhood frames from a List of Cameras. Combine these into
            a single batch for optimization. Neighborhoods are non-overlapping and build around an index.
            If the neighborhood size is even, we take more frames from the left side of the index.

            NOTE: we assume that cams is sorted according to the timestamp!
            """

            def draw_neighborhood(cams: List, n_samples: int, idx: int):
                if n_samples % 2 == 0:
                    left = max(0, idx - n_samples // 2)
                    right = min(len(cams), idx + n_samples // 2 - 1)
                else:
                    left = max(0, idx - n_samples // 2)
                    right = min(len(cams), idx + n_samples // 2)
                return cams[left : right + 1], list(range(left, right + 1))

            # Draw batch_size // neighborhood_size non-overlapping neighborhoods
            # Include check to not have overlap
            batch, idxs = [], []
            rest = batch_size % neighborhood_size
            for i in range(batch_size // neighborhood_size):
                if neighborhood_size % 2 == 0:
                    idx = np.random.randint(neighborhood_size // 2 + 1, len(cams) - neighborhood_size // 2)
                else:
                    idx = np.random.randint(neighborhood_size // 2, len(cams) - neighborhood_size // 2)
                while idx in idxs:
                    idx = np.random.randint(len(cams))

                # In case we have a rest, we draw a larger neighborhood rather than have one very small one
                if i == batch_size // neighborhood_size - 1 and rest > 0:
                    cams_neigh, idx_neigh = draw_neighborhood(cams, neighborhood_size + rest, idx)
                else:
                    cams_neigh, idx_neigh = draw_neighborhood(cams, neighborhood_size, idx)

                batch.extend(cams_neigh)
                idxs.extend(idx_neigh)

            return batch

        @torch.no_grad()
        def maybe_fill_holes(render_pkg, view_id: int, size_hole: int = 100, max_mem: float = 0.95) -> bool:
            """We sometimes still have "white holes" left in our Renderings, because we only take covisible areas
            for initialization. Manually filling these speeds up refinement significantly!
            """
            mask = torch.all((render_pkg["render"].squeeze() == self.background[:, None, None]), dim=0)

            if mask.sum() > size_hole:
                has_hole = True
                self.info(f"Detected holes in view {view_id} during importance sampling. Adding higher weight ...")
                # Help out in those frames by refining these more!
                used_mem, free_mem = self.get_ram_usage()
                # NOTE chen: this can add up a lot of memory, only this if we have enough slack
                if used_mem <= max_mem:
                    ng_before = len(self.gaussians)
                    self.info(f"Patching up holes in view {view_id} manually using Depth from Tracking ...")
                    self.densify_holes(view_id, mask, downsample_factor=2.0)
                    self.info(f"Added {len(self.gaussians) - ng_before} Gaussians to fill holes in view {view_id} ...")
            else:
                has_hole = False
            return has_hole

        def importance_weights_from_single_pass(cams: List[Camera], batch_size: int = 16):
            """Run a single forward pass over all frames to gather importance weights.
            Since we need to run over a potential large batch of all frames, we use mini batches.
            This whole operation consumes much time, so we still backpropagate the loss and optimize the Gaussians,
            even though this does not return the "true" importance weights at timestep 0.
            """
            loss, weights = 0.0, []
            for i, view in tqdm(enumerate(cams)):
                loss_i, _ = self.render_compare(view)
                loss += loss_i
                weights.append(loss_i.detach().clone().item())

                # Keep memory in check by only backpropagating in batches
                if i % batch_size == 0:
                    self.gaussians.check_nans()  # NOTE chen: this can happen we have zero depth and an inconvenient pose

                    # Punish anisotropic Gaussians
                    scaling = self.gaussians.get_scaling
                    isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
                    loss += self.loss_params.beta1 * isotropic_loss.mean()
                    # Backpropagate through batch
                    self.gaussians.check_nans()  # Sanity check to avoid invalid Gaussians (e.g. from 0 depths)
                    loss.backward()
                    # Make step
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad()
                    loss = 0.0  # Reset loss

                    gc.collect()
                    torch.cuda.empty_cache()

            return weights

        kf_cams = [cam for cam in self.cameras if cam.uid in self.cam2buffer]
        has_nonkf = len(self.cameras) != len(self.cam2buffer)  # Check if we added non-keyframes
        if has_nonkf:
            non_kf_cams = [cam for cam in self.cameras if cam.uid not in self.cam2buffer]
        else:
            non_kf_cams = None

        # Update GUI
        if self.use_gui:
            self.q_main2vis.put_nowait(
                gui_utils.GaussianPacket(  # NOTE leon: the GUI will lag (even run OOM) if we add thousands of cameras.
                    gaussians=clone_obj(self.gaussians)  # , keyframes=[cam.detach() for cam in self.cameras]
                )
            )

        # Gather importance weights by computing the loss over all frames first
        # Because we dont want to waste too much compute, we backpropagate over this large accumulated batch
        if self.refine_params.sampling.weighted:
            self.info("Gathering importance weights for refinement ...")
            self.opt_params.position_lr_init /= 10  # We use a constant lower learning rate for all frames
            self.gaussians.training_setup(self.opt_params)
            weights = importance_weights_from_single_pass(kf_cams, batch_size=self.refine_params.bs)
            self.opt_params.position_lr_init *= 10
        else:
            weights = None

        ### Refinement loop
        # Reset scheduler and optimizer for refinement
        self.opt_params.position_lr_max_steps = self.refine_params.batch_iters
        # Reduce the learning rate by wanted factor for refinement since we already have a good map
        self.opt_params.position_lr_init *= self.refine_params.lr_factor
        self.opt_params.position_lr_final *= self.refine_params.lr_factor
        self.gaussians.training_setup(self.opt_params)

        for iter1 in tqdm(
            range(self.refine_params.iters), desc=colored("Gaussian Refinement", "magenta"), colour="magenta"
        ):
            # Decide whether to densify / prune this iteration
            do_densify = (
                (iter1 + 1) % self.refine_params.prune_densify_every == 0
            ) and iter1 <= self.refine_params.densify_until
            opacity_densify = self.refine_params.densify.use_opacity

            ### Optimize a random batch from all frames
            # In prgbd mode we cant densify non-keyframes which have the wrong scale information!
            if self.mode == "prgbd" and (do_densify or opacity_densify):
                batch = draw_random_batch(kf_cams, batch_size=self.refine_params.bs)
            ### Optimize a random batch from all frames
            elif self.refine_params.sampling.use_neighborhood:
                # Use multiple random temporal neighborhood, so we have overlap between frames
                batch = draw_random_neighborhood_batch(
                    kf_cams,
                    batch_size=self.refine_params.bs,
                    neighborhood_size=self.refine_params.sampling.neighborhood_size,
                )
            else:
                # Use completely random frames which might not have overlap
                # NOTE this method can have a guarantee to use x% keyframes
                batch = draw_random_batch(
                    kf_cams,
                    self.refine_params.bs,
                    weights_kf=weights,
                    nonkf_cams=non_kf_cams,
                    kf_always=self.refine_params.sampling.kf_at_least,
                )

            # Optimize this batch for a few iterations, so newly added Gaussians can converge
            for iter2 in tqdm(
                range(self.refine_params.batch_iters), desc=colored("Batch Optimization", "magenta"), colour="magenta"
            ):
                loss = self.mapping_step(
                    iter2,
                    batch,
                    self.refine_params.densify.vanilla,
                    prune_densify=do_densify,  # Prune and densify with vanilla 3DGS strategy
                    opacity_densify=opacity_densify,  # Densify based on low opacity regions
                    optimize_poses=self.refine_params.optimize_poses,
                )
                do_densify, opacity_densify = False, False
                print(colored("[Gaussian Mapper] ", "magenta"), colored(f"Refinement loss: {loss}", "cyan"))
                self.loss_list.append(loss)
                if self.use_gui:
                    self.q_main2vis.put_nowait(
                        gui_utils.GaussianPacket(
                            gaussians=clone_obj(self.gaussians), keyframes=[frame.detach() for frame in batch]
                        )
                    )

            del batch

    def get_mapping_update(
        self,
        frames: List[Camera],
        was_pruned: bool = False,
        feedback_poses: bool = True,
        feedback_disps: bool = False,
        opacity_threshold: float = 0.1,
        ignore_frames: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8],
        min_coverage: float = 0.6,  # Min. Density of frame after eliminiating outliers
        max_lonely_gaussians: float = 0.5,  # A frame should have at most <= x % Gaussians that are only visible in its own frameMin. Density of frame after eliminiating outliers
        max_diff_to_video: float = 0.2,  # Maximum abs. rel. deviation from the dense video depth
    ) -> Dict:
        """Get the index, poses and depths of the frames that were already optimized. We can use this to then feedback
        the outputs of the Gaussian Rendering optimization back into the video.map.

        We should only feedback 'good' depths since Rendering can also introduce many outliers, noise or
        just does not converge correctly immediately. The Tracker operates with dense depth frames, so we need to be careful to not feedback too sparse depths.
        For this purpose, we limit the 'step size' of our renderer by ensuring that the rendered depth does not deviate from the original too much. If the
        current Renderer map does not cover enough of the scene in the specific view, we skip the frame.

        Since the map is usually not very good in the beginning, we never feedback the first few keyframes.
        """

        def compare_render_w_video(depth: torch.Tensor, index_in_video: int, max_diff_to_video: float = 0.15):
            """Analyze differences in the rendered depth and the dense video depth. This simply checks how many percent
            of pixels are within an error bound.

            There can be huge differences due to:
            i) Holes in our rendering, because the scene is not fully covered by Gaussians yet
            ii) Outliers in the rendered depth, because the optimization did not converge correctly
            iii) Holes and invalid depths in the dense video depth, because the update network is not perfect
            iv) Occluded areas are usually much better in the rendered depth, but thus are different in video
            """
            # Analyze the video depths before filtering as a reference
            if self.video.upsample:
                disps_ref = self.video.disps_up[index_in_video]
            else:
                disps_ref = self.video.disps[index_in_video]
            valid_ref = disps_ref > 0
            depth_ref = torch.where(valid_ref, 1.0 / disps_ref, disps_ref)

            disps_clean = self.video.disps_clean[index_in_video]
            valid_clean = disps_clean > 0
            depth_clean = torch.where(valid_clean, 1.0 / disps_clean, disps_clean)

            # HACK Limit step size by punishing large deviations from the original video depth
            if not self.video.upsample:
                s = self.video.scale_factor
                depth_down = depth[:, int(s // 2 - 1) :: s, int(s // 2 - 1) :: s]
                diff_to_video = torch.abs(depth_down - depth_ref) / depth_ref  # Abs rel.
                # Filter away outliers
                depth_wo_outliers = torch.where(diff_to_video < max_diff_to_video, depth_down, 0.0)
            else:
                diff_to_video = torch.abs(depth - depth_ref) / depth_ref  # Abs rel.
                depth_wo_outliers = torch.where(diff_to_video < max_diff_to_video, depth, 0.0)  # Filter away outliers
            coverage_gs_wo = (depth_wo_outliers > 0).sum() / (depth_wo_outliers > 0).numel()
            coverage_init = (depth_clean > 0).sum() / (depth_clean > 0).numel()
            return coverage_gs_wo

        index, poses, depths = [], [], []
        # Sanity check
        if not feedback_disps and not feedback_poses:
            return {"index": index, "poses": poses, "depths": depths}

        # Render frames to extract depth
        rejected, accepted = [], []
        for view in frames:
            # Ignore boundary frames, especially in the beginning when we build the map
            # (the first few frames are usually not good for feedback as they are not optimized yet or heavily incomplete)
            if self.cam2buffer[view.uid] in ignore_frames:
                rejected.append(view.uid)
                continue
            render_pkg = render(view, self.gaussians, self.pipeline_params, self.background, device=self.device)

            # We computed covisibility and can therefore count how many Gaussians were observed in how many frames
            if was_pruned:
                in_frame = self.gaussians.unique_kfIDs == view.uid
                n_observed = self.gaussians.n_obs[in_frame]
                # Bad frames usually dont have many covisible Gaussians attached to them
                if (n_observed < 1).sum() / n_observed.numel() > max_lonely_gaussians:
                    rejected.append(view.uid)
                    # self.info(f"Skipping view {view.uid} during Feedback as it has too few covisible Gaussians ...")
                    continue

            # NOTE chen: this can be None when self.gaussians is 0. This could happen in some cases
            if render_pkg is None:
                rejected.append(view.uid)
                # self.info(f"Skipping view {view.uid} as no Gaussians are present in it ...")
                continue

            index_in_video = self.cam2buffer[view.uid]
            depth = render_pkg["depth"].detach()
            # Filter away pixels with very low opacity as these are usually unreliable
            valid_o = render_pkg["opacity"].detach() > opacity_threshold
            depth[~valid_o] = 0.0
            coverage_gs_wo = compare_render_w_video(depth, index_in_video, max_diff_to_video)
            # Disparity in video is dense -> Dont feedback too sparse frames
            if coverage_gs_wo > min_coverage:
                index.append(index_in_video)
                poses.append(view.pose)
                depths.append(clone_obj(depth))
                accepted.append(view.uid)
            else:
                rejected.append(view.uid)
                # self.info(f"Skipping view {view.uid} during Feedback as it has too low coverage with video buffer ...")

        if not feedback_disps:
            depths = []
        if not feedback_poses:
            poses = []

        if len(index) > 0:
            self.info(f"Feeding back good frames: {accepted}, ignoring frames: {rejected}...")
        return {"index": index, "poses": poses, "depths": depths}

    def get_camera_trajectory(self, frames: List[Camera]) -> torch.Tensor:
        """Get the camera trajectory of the frames in world coordinates."""
        poses = []
        for view in frames:
            poses.append(view.pose)
        return torch.stack(poses)

    def maybe_clean_pose_update(self, frames: List[Camera]) -> None:
        """Check if pose updates are not degenerate and set to zero if they are."""
        for view in frames:
            if torch.isnan(view.cam_rot_delta).any() and torch.isnan(view.cam_trans_delta).any():
                print(colored(f"NAN in pose optimizer in view {view.uid}!", "red"))
                print(colored("Setting to zero update ...", "red"))
                view.cam_rot_delta = torch.nn.Parameter(torch.zeros(3, device=self.device))
                view.cam_trans_delta = torch.nn.Parameter(torch.zeros(3, device=self.device))

    def render_compare(self, view: Camera) -> Tuple[float, Dict, Dict]:
        """Render current view and compute loss by comparing with groundtruth"""
        render_pkg = render(view, self.gaussians, self.pipeline_params, self.background, device=self.device)
        # NOTE chen: this can be None when self.gaussians is 0. This can happen in some cases
        if render_pkg is None:
            return 0.0

        image, depth = render_pkg["render"], render_pkg["depth"]
        current_loss = mapping_rgbd_loss(image, depth, view, **self.loss_params)
        return current_loss, render_pkg

    def mapping_step(
        self,
        iter: int,
        frames: List[Camera],
        vanilla_densify_params: Dict,
        prune_densify: bool = False,
        opacity_densify: bool = False,
        optimize_poses: bool = False,
    ) -> float:
        """Takes the list of selected keyframes to optimize and performs one step of the mapping optimization."""
        # Sanity check when we dont have anything to optimize
        if len(self.gaussians) == 0:
            return 0.0

        # NOTE chen: this can happen we have zero depth and an inconvenient pose
        self.gaussians.check_nans()

        if optimize_poses:
            pose_optimizer = self.get_pose_optimizer(frames)

        loss = 0.0
        # Collect for densification and pruning
        opacity_acm, radii_acm = [], []
        visibility_filter_acm, viewspace_point_tensor_acm = [], []
        for view in frames:
            current_loss, render_pkg = self.render_compare(view)
            if render_pkg is None:
                self.info(f"Skipping view {view.uid} as no gaussians are present ...")
                continue

            # Keep track of how often the Gaussians were already optimized
            self.gaussians.increment_n_opt_counter(visibility=render_pkg["visibility_filter"])
            self.n_optimized[view.uid] += 1

            # Accumulate for after loss backpropagation
            opacity_acm.append((view, render_pkg["opacity"]))
            visibility_filter_acm.append(render_pkg["visibility_filter"])
            viewspace_point_tensor_acm.append(render_pkg["viewspace_points"])
            radii_acm.append(render_pkg["radii"])

            self.last_frame_loss[view.uid] = current_loss.item()
            loss += current_loss

        if self.update_params.grad_scaler.do_scale:
            # Scale gradients dependent on how often a Gaussian was already optimized in the past
            grad_hooks = self.gaussians.set_scale_grads(
                self.update_params.grad_scaler.min_scale, self.update_params.grad_scaler.decay_rate
            )

        # Scale the loss with the number of frames so we adjust the learning rate dependent on batch size,
        # (naive adding for huge batches would result on bigger updates)
        # NOTE chen: MonoGS scales their loss with len(frames)
        # NOTE we get better results when not scaling, getting good performance really requires tuning batch size and learning rate together
        avg_loss = loss / len(frames)  # Average over batch

        # Regularizor: Punish anisotropic Gaussians
        scaling = self.gaussians.get_scaling
        isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
        loss += self.loss_params.beta1 * isotropic_loss.mean()

        # NOTE chen: this can happen we have zero depth and an inconvenient pose
        self.gaussians.check_nans()
        loss.backward()

        ### Maybe Densify and Prune before update
        with torch.no_grad():
            for idx in range(len(viewspace_point_tensor_acm)):
                # Dont let Gaussians grow too much by forcing radius to not change
                self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                    radii_acm[idx][visibility_filter_acm[idx]],
                )
                self.gaussians.add_densification_stats(viewspace_point_tensor_acm[idx], visibility_filter_acm[idx])

            # Prune and Densify
            if self.last_idx > self.n_last_frames and prune_densify:
                # General pruning based on opacity and size + densification (from original 3DGS)
                self.gaussians.densify_and_prune(**vanilla_densify_params)

            # Densify in low opacity regions only after the map is stable already
            # (else we waste compute, because densify_and_prune will fill initial holes quickly)
            if prune_densify and opacity_densify and self.count > self.update_params.densify.opacity.after:
                ng_before = len(self.gaussians)
                for i, acm_item in enumerate(opacity_acm):
                    view, opacity = acm_item
                    # Only densify in regions with: i) with low opacity, ii) with high rgb difference, iii) with high depth difference
                    # see https://arxiv.org/pdf/2403.12535
                    mask = opacity.squeeze() < self.update_params.densify.opacity.th
                    # if this mask has too many pixels, then subsample to acceptable lower number
                    if mask.sum() > self.update_params.densify.opacity.max_pixels:
                        mask = random_subsample_mask(mask, self.update_params.densify.opacity.max_pixels)
                        self.gaussians.densify_from_mask(view, mask, depthmap=view.depth_prior.cpu().numpy())
                if (len(self.gaussians) - ng_before) > 0:
                    self.info(f"Added {len(self.gaussians) - ng_before} gaussians based on opacity")

        ### Update states
        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad()
        self.gaussians.update_learning_rate(iter)

        # Delete lists of tensors
        del opacity_acm
        del radii_acm
        del visibility_filter_acm
        del viewspace_point_tensor_acm
        del rgb_diff
        del depth_diff

        if self.update_params.grad_scaler.do_scale:
            for hook in grad_hooks:
                hook.remove()  # Remove again for sanity, the hook needs to be set every iteration again

        if optimize_poses:
            pose_optimizer.step()
            self.maybe_clean_pose_update(frames)  # Sanitize in case of nan's
            pose_optimizer.zero_grad()
            # Actually make optimizer step
            for view in frames:
                if view.uid == 0:  # Keep first pose always fixed!
                    continue
                update_pose(view)
            del pose_optimizer  # We define a new one every time anyways

        return avg_loss.detach().item()

    def covisibility_pruning(
        self,
        n_last_frames: int = 10,
        mode: str = "new",
        last: int = 5,
        dont_prune_latest: int = 1,
        visibility_th: int = 2,
    ):
        """Covisibility based pruning.

        If prune is set to "new", we only prune the last n frames. Else we check for all frames if Gaussians are visible in at least k frames.
        A Gaussian is visible if it touched at least a single pixel in a view.
        """
        # Sanity check
        assert mode in ["new", "abs"], "You can only select 'new' or 'abs' as pruning mode"

        if dont_prune_latest > len(self.cameras + self.new_cameras):
            return

        start = time.time()
        frames = sorted(self.cameras + self.new_cameras, key=lambda x: x.uid)
        last = min(last, len(frames))
        # Make a covisibility check only for the last n frames
        if mode == "new":
            # Dont prune the Last/last-1 frame, since we then would add and prune Gaussians immediately -> super wasteful
            if dont_prune_latest > 0:
                frames = frames[-n_last_frames:-dont_prune_latest]
            else:
                frames = frames[-n_last_frames:]

        occ_aware_visibility = {}
        self.gaussians.n_obs.fill_(0)  # Reset observation count
        for view in frames:
            render_pkg = render(view, self.gaussians, self.pipeline_params, self.background)
            visibility = (render_pkg["n_touched"] > 0).long()
            occ_aware_visibility[view.uid] = visibility
            # Count when at least one pixel was touched by the Gaussian
            self.gaussians.n_obs += visibility.cpu()  # Increase observation count

        to_prune = self.gaussians.n_obs < visibility_th
        if mode == "new":
            ids = [view.uid for view in frames]
            sorted_frames = sorted(ids, reverse=True)
            # Only prune Gaussians added on the last k frames
            last_idx = min(last, len(sorted_frames))
            prune_last = self.gaussians.unique_kfIDs >= sorted_frames[last_idx - 1]
            to_prune = torch.logical_and(to_prune, prune_last)

        if to_prune.sum() > 0:
            self.gaussians.prune_points(to_prune.to(self.device))

        end = time.time()
        self.info(f"({mode}) Covisibility pruning took {(end - start):.2f}s, pruned: {to_prune.sum()} Gaussians")

    def select_keyframes(self, random_weights: Optional[str] = None):
        """Select the last n1 frames and n2 other random frames from all.
        If with_random_weights is set, we assign a higher sampling probability to keyframes, that have not yet been
        optimized a lot. This is measured by self.n_optimized.

        NOTE this method assumes self.cameras to contain sorted keyframes with increasing uid order.

        random_weights:
            If set to "visited", we assign higher probability to frames that have been visited less often.
            If set
        """
        if len(self.cameras) <= self.n_last_frames + self.n_rand_frames:
            keyframes = self.cameras
            keyframes_idx = np.arange(len(self.cameras))
        else:
            last_window = self.cameras[-self.n_last_frames :]
            window_idx = np.arange(len(self.cameras))[-self.n_last_frames :]
            if self.n_rand_frames > 0:
                if random_weights == "visited":
                    to_draw = np.arange(len(self.cameras) - self.n_last_frames)  # Indices we can draw from
                    weights = [self.n_optimized[idx] / self.mapping_iters + 1 for idx in to_draw]
                    weights = [1 / w for w in weights]  # Invert to get higher probability for less visited frames
                    random_idx = list(WeightedRandomSampler(weights, self.n_rand_frames, replacement=False))
                    random_idx = to_draw[random_idx]
                elif random_weights == "loss":
                    to_draw = np.arange(len(self.cameras) - self.n_last_frames)  # Indices we can draw from
                    weights = [self.last_frame_loss[idx] for idx in to_draw]
                    random_idx = list(WeightedRandomSampler(weights, self.n_rand_frames, replacement=False))
                    random_idx = to_draw[random_idx]
                else:
                    random_idx = np.random.choice(
                        len(self.cameras) - self.n_last_frames, self.n_rand_frames, replace=False
                    )
                random_frames = [self.cameras[i] for i in random_idx]
                keyframes = last_window + random_frames
                keyframes_idx = np.concatenate([window_idx, random_idx])
            else:
                keyframes = last_window
                keyframes_idx = window_idx
        return keyframes, keyframes_idx

    def add_nonkeyframe_cameras(self):
        """Use the trajectory filler of the SLAM system to interpolate the current keyframe poses. We will then add
        the rest of the video images as new Cameras to have a better finetuning with more information.
        """

        self.info("Interpolating trajectory to get non-keyframe Cameras for refinement ...")
        non_kf_cams = self.get_nonkeyframe_cameras(self.slam.dataset, self.slam.traj_filler)
        # Reinstatiate an empty traj_filler, we only reuse this during eval
        # this deletes the graph and frees up memory
        del self.slam.traj_filler
        self.slam.traj_filler = PoseTrajectoryFiller(
            self.slam.cfg, net=self.slam.net, video=self.slam.video, device=self.slam.device
        )
        torch.cuda.empty_cache()
        gc.collect()

        self.info(f"Added {len(non_kf_cams)} new cameras: {[cam.uid for cam in non_kf_cams]}")
        new_cam2buffer, new_buffer2cam, masked_mapping, new_n_optimized = {}, {}, {}, {}

        for cam in self.cameras:
            new_id = int(self.video.timestamp[cam.uid].item())
            masked_mapping[cam.uid] = (new_id, self.gaussians.unique_kfIDs == cam.uid)
            new_cam2buffer[new_id] = cam.uid  # Memoize from timestamp to old video index
            new_buffer2cam[cam.uid] = new_id
            new_n_optimized[new_id] = self.n_optimized[cam.uid]
            cam.uid = new_id  # Reassign local keyframe ids to global stream ids

        self.cam2buffer, self.buffer2cam, self.n_optimized = new_cam2buffer, new_buffer2cam, new_n_optimized
        # Update the keyframe ids for each Gaussian, so they fit the new global cam.uid's
        for key, val in masked_mapping.items():
            new_id, mask = val
            self.gaussians.unique_kfIDs[mask] = new_id

        self.cameras += non_kf_cams  # Add to set of cameras
        # Reorder according to global video timestamp uid
        self.cameras = sorted(self.cameras, key=lambda x: x.uid)
        # Add non-keyframes to dictionary
        for cam in self.cameras:
            if not cam.uid in self.n_optimized:
                self.n_optimized[cam.uid] = 0

    def _last_call(self, mapping_queue: mp.Queue, received_item: mp.Event):
        """We already build up the map based on the SLAM system and finetuned over it.
        Depending on compute budget this has been done scarcely.
        This call runs many more iterations for refinement and densification to get a high quality map.

        Since the SLAM system operates on keyframes, but we have many more views in our video stream,
        we can use additional supervision from non-keyframes to get higher detail.
        """
        # Free memory before doing refinement
        torch.cuda.empty_cache()
        gc.collect()

        # NOTE MonoGS does 26k iterations, while we only do 100
        if self.refine_params.iters > 0:

            self.info(f"Gaussians before Map Refinement: {len(self.gaussians)}")
            # Add more information for refinement if wanted
            if self.slam.dataset is not None and self.refine_params.sampling.use_non_keyframes:
                self.add_nonkeyframe_cameras()

            self.info("\nMapping refinement starting")
            self.map_refinement()
            self.info(f"Gaussians after Map Refinement: {len(self.gaussians)}")
            self.info("Mapping refinement finished")

            # Free memory after doing refinement
            torch.cuda.empty_cache()
            gc.collect()

        # Filter out the non-keyframes which are not stored in the video.object
        only_kf = [cam for cam in self.cameras if cam.uid in self.cam2buffer]
        # Only feedback the poses, since we will not work with the video again
        if (self.feedback_poses or self.feedback_disps) and not self.feedback_params.no_refinement:
            self.info(f"Feeding back into video.map ...")
            # HACK allow large differences to the video, else we will filter away occluded regions which we already corrected rightfully
            to_set = self.get_mapping_update(
                only_kf,
                feedback_poses=self.feedback_poses,
                feedback_disps=self.feedback_disps,
                opacity_threshold=0.0,
                ignore_frames=[],
                max_diff_to_video=1.0,
            )
            # There is no need to feed back the
            self.video.set_mapping_item(**to_set)

        self.gaussians.save_ply(f"{self.output}/mesh/final_{self.mode}.ply")
        self.info(f"Mesh saved at {self.output}/mesh/final_{self.mode}.ply")

        if self.save_renders:
            for cam in self.cameras:
                self.save_render(cam, f"{self.output}/intermediate_renders/final/{cam.uid}.png")

        plot_losses(
            self.loss_list,
            self.refine_params.iters,
            title=f"Loss evolution with: {len(self.gaussians)} gaussians",
            output_file=f"{self.output}/loss_{self.mode}.png",
        )

        print(colored("[Gaussian Mapper] ", "magenta"), colored(f"Final mapping loss: {self.loss_list[-1]}", "cyan"))
        self.info(f"{len(self.iteration_info)} iterations, {len(self.cameras)/len(self.iteration_info)} cams/it")

        # Export Cameras and Gaussians to the main Process
        # NOTE chen: we safeguard against None Queues in case we are in test mode ...
        if self.evaluate and mapping_queue is not None:
            mapping_queue.put(
                EvaluatePacket(
                    pipeline_params=clone_obj(self.pipeline_params),
                    cameras=[cam.detach() for cam in self.cameras],
                    gaussians=clone_obj(self.gaussians),
                    background=clone_obj(self.background),
                    cam2buffer=clone_obj(self.cam2buffer),
                    buffer2cam=clone_obj(self.buffer2cam),
                )
            )
        else:
            if mapping_queue is not None:
                mapping_queue.put("None")
        if received_item is not None:
            received_item.wait()  # Wait until the Packet got delivered

    def add_new_gaussians(self, cameras: List[Camera]) -> Camera | None:
        """Initialize new Gaussians based on the provided views (images, poses (, depth))"""
        # Sanity check
        if len(cameras) == 0:
            return None

        for cam in cameras:
            if not self.initialized:
                self.initialized = True
                self.gaussians.extend_from_pcd_seq(cam, cam.uid, init=True)
                self.info(f"Initialized with {len(self.gaussians)} gaussians for view {cam.uid}")
            else:
                ng_before = len(self.gaussians)
                self.gaussians.extend_from_pcd_seq(cam, cam.uid, init=False)

        return cam

    def update_gui(self, last_new_cam: Camera) -> None:
        # Get the latest frame we added together with the input
        if len(self.new_cameras) > 0 and last_new_cam is not None:
            img = last_new_cam.original_image
            if last_new_cam.depth is not None:
                if not self.loss_params.supervise_with_prior:
                    gtdepth = last_new_cam.depth.detach().clone().cpu().numpy()
                else:
                    gtdepth = last_new_cam.depth_prior.detach().clone().cpu().numpy()
            else:
                gtdepth = None
        # We did not have any new input, but refine the Gaussians
        else:
            img, gtdepth = None, None
            last_new_cam = self.cameras[-1]

        self.q_main2vis.put_nowait(
            gui_utils.GaussianPacket(
                gaussians=clone_obj(self.gaussians),
                current_frame=last_new_cam.detach(),
                keyframes=[cam.detach() for cam in self.cameras],
                kf_window=None,
                gtcolor=img,
                gtdepth=gtdepth,
            )
        )

    def _update(self, delay_to_tracking=True, iters: int = 10, release_cache: bool = False):
        """Update our rendered map by:
        i) Pull a filtered update from the sparser SLAM map
        ii) Add new Gaussians based on new views
        iii) Run a bunch of optimization steps to update Gaussians and camera poses
        iv) Prune the resulting Gaussians based on visibility and size
        v) Maybe send back a filtered update to the SLAM system
        """
        self.info("Currently has: {} gaussians".format(len(self.gaussians)))

        ### Filter map based on multiview_consistency and uncertainty
        self.video.filter_map(**self.update_params.filter)

        if delay_to_tracking:
            delay = self.delay
        else:
            delay = 0

        ### Add new cameras based on video index
        self.get_new_cameras(delay=delay)  # Add new cameras
        if len(self.new_cameras) != 0:
            self.last_idx = self.new_cameras[-1].uid + 1
            self.info(f"Added {len(self.new_cameras)} new cameras: {[cam.uid for cam in self.new_cameras]}")

        ### Update the frames based on Tracker and add new Gaussians
        self.frame_updater(delay=delay)  # Update all changed cameras with new information from SLAM system

        last_new_cam = self.add_new_gaussians(self.new_cameras)
        # We might have 0 Gaussians in some cases, so no need to run optimizer
        if len(self.gaussians) == 0:
            self.info("No Gaussians to optimize, skipping mapping step ...")
            return

        ### Optimize gaussians
        for iter in tqdm(range(iters), desc=colored("Gaussian Optimization", "magenta"), colour="magenta"):
            do_densify = (
                iter % self.update_params.prune_densify_every == 0 and iter < self.update_params.prune_densify_until
            )
            frames = (
                self.select_keyframes(random_weights=self.update_params.random_selection_weights)[0] + self.new_cameras
            )
            loss = self.mapping_step(
                iter,
                frames,
                self.update_params.densify.vanilla,
                prune_densify=do_densify,  # Prune and densify with vanilla 3DGS strategy
                opacity_densify=self.update_params.densify.use_opacity,  # Densify based on low opacity regions
                optimize_poses=self.update_params.optimize_poses,
            )
            self.loss_list.append(loss)

        # Keep track of how well the Rendering is doing
        print(colored("\n[Gaussian Mapper] ", "magenta"), colored(f"Loss: {self.loss_list[-1]}", "cyan"))

        ### Prune unreliable Gaussians
        was_pruned = False
        if len(self.iteration_info) % self.update_params.prune_every == 0 and delay_to_tracking:
            if self.update_params.pruning.use_covisibility:
                # Gaussians should be visible in multiple frames
                self.covisibility_pruning(n_last_frames=self.n_last_frames, **self.update_params.pruning.covisibility)
                was_pruned = True

        ### Feedback new state of map to Tracker
        if (self.feedback_poses or self.feedback_disps) and self.count > self.feedback_params.warmup:
            if self.feedback_params.only_last_window and len(self.new_cameras) > 0:
                update_cams = sorted(frames, key=lambda x: x.uid)[-self.n_last_frames :]
            else:
                update_cams = frames
            to_set = self.get_mapping_update(
                update_cams,
                was_pruned,
                feedback_poses=self.feedback_poses and self.update_params.optimize_poses,
                feedback_disps=self.feedback_disps,
                **self.feedback_params.kwargs,
            )
            if len(to_set["index"]) > 0:
                self.video.set_mapping_item(**to_set)

        ### Update visualization
        if self.use_gui:
            self.update_gui(last_new_cam)

        ### Save renders for debugging and visualization
        if self.save_renders:
            # Save only every 5th camera to save disk spacse
            for cam in self.cameras:
                if cam.uid % 5 == 0:
                    self.save_render(cam, f"{self.output}/intermediate_renders/temp/{cam.uid}.png")

        # Free memory each iteration NOTE: this slows it down a bit
        if release_cache:
            torch.cuda.empty_cache()
            gc.collect()

        self.iteration_info.append(len(self.new_cameras))
        # Keep track of added cameras
        self.cameras += self.new_cameras
        self.new_cameras = []

    def __call__(self, mapping_queue: mp.Queue, received_item: mp.Event, the_end=False):

        self.cur_idx = self.video.counter.value

        # Dont update when we get no new frames
        if not the_end and self.last_idx + self.delay < (self.cur_idx + 1) and (self.cur_idx + 1) > self.warmup:
            self._update(iters=self.mapping_iters)
            self.count += 1  # Count how many times we ran the Renderer
            return False

        # We reached the end of the video, but we still have to process some keyframes before last call
        elif the_end and self.last_idx + self.delay < self.cur_idx and (self.cur_idx + 1) > self.warmup:
            self._update(iters=self.mapping_iters)
            self.count += 1  # Count how many times we ran the Renderer
            return False

        elif the_end and (self.last_idx + self.delay) >= self.cur_idx:

            # Allow pruning all frames equally
            self.update_params.pruning.covisibility.dont_prune_latest = 0
            self.update_params.pruning.covisibility.last = 0
            # Run another call to catch the last batch of keyframes
            self._update(iters=self.mapping_iters + 10, delay_to_tracking=False)
            self.count += 1

            self._last_call(mapping_queue=mapping_queue, received_item=received_item)
            return True

        else:
            return False
