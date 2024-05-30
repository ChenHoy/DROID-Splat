import os
import ipdb
from copy import deepcopy
from typing import List, Dict, Optional
import gc
from termcolor import colored
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
from pytorch_msssim import ssim, ms_ssim

import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt

from .gaussian_splatting.gui import gui_utils, slam_gui
from .gaussian_splatting.gaussian_renderer import render
from .gaussian_splatting.scene.gaussian_model import GaussianModel
from .gaussian_splatting.utils.graphics_utils import (
    getProjectionMatrix2,
    getWorld2View2,
    focal2fov,
)
from .gaussian_splatting.utils.loss_utils import l1_loss
from .gaussian_splatting.slam_utils import depth_reg, image_gradient_mask
from .gaussian_splatting.multiprocessing_utils import clone_obj
from .gaussian_splatting.camera_utils import Camera
from .gaussian_splatting.pose_utils import update_pose

import droid_backends
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
        self.refinement_iters = cfg.mapping.refinement_iters
        self.use_non_keyframes = cfg.mapping.use_non_keyframes
        self.mapping_iters = cfg.mapping.mapping_iters
        self.save_renders = cfg.mapping.save_renders
        self.warmup = max(cfg.mapping.warmup, cfg.tracking.warmup)
        self.batch_mode = cfg.mapping.batch_mode  # Take a batch of all unupdated frames at once
        self.optimize_poses = cfg.mapping.optimize_poses
        self.opt_params = cfg.mapping.opt_params
        self.feedback_map = cfg.mapping.feedback_mapping
        self.pipeline_params = cfg.mapping.pipeline_params
        self.kf_mng_params = cfg.mapping.keyframes
        # Always take n last frames from current index and optimize additional random frames globally
        self.n_last_frames = self.kf_mng_params.n_last_frames
        self.n_rand_frames = self.kf_mng_params.n_rand_frames
        # NOTE chen: during refinement we likely go over this number!
        # in order to avoid OOM, we chunk the refinement up
        self.max_frames_refinement = cfg.mapping.max_frames_refinement  # Maximum number of frames to optimize over

        self.filter_uncertainty = cfg.mapping.filter_uncertainty
        self.filter_multiview = cfg.mapping.filter_multiview

        self.filter_dyn = cfg.get("with_dyn", False)

        self.loss_params = cfg.mapping.loss

        self.sh_degree = 3 if cfg.mapping.use_spherical_harmonics else 0

        # Change the downsample factor for initialization depending on cfg.tracking.upsample, so we always have points
        if not self.cfg.tracking.upsample:
            cfg.mapping.input.pcd_downsample_init /= 8
            cfg.mapping.input.pcd_downsample /= 8
        self.gaussians = GaussianModel(self.sh_degree, config=cfg.mapping.input)
        self.gaussians.init_lr(self.opt_params.init_lr)
        self.gaussians.training_setup(self.opt_params)

        bg_color = [1, 1, 1]  # White background
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.z_near = 0.001
        self.z_far = 100000.0

        if gui_qs is not None:
            self.q_main2vis = gui_qs
            self.use_gui = True
        else:
            self.use_gui = False

        self.iteration_info = []

        self.cameras = []
        self.new_cameras = []
        self.loss_list = []
        self.last_idx = 0
        self.initialized = False
        self.projection_matrix = None

    def info(self, msg: str):
        print(colored("[Gaussian Mapper] " + msg, "magenta"))

    def depth_filter(self, idx: int, count_thresh: int = 2, bin_thresh: float = 0.1, min_disp_thresh: float = 0.01):
        """Check for consistency of part of the video"""

        with self.video.get_lock():
            poses = torch.index_select(self.video.poses, 0, idx)
            if self.video.upsampled:
                disps = torch.index_select(self.video.disps_up, 0, idx)
                intrinsics = self.video.intrinsics[0] * self.video.scale_factor
            else:
                disps = torch.index_select(self.video.disps, 0, idx)
                intrinsics = self.video.intrinsics[0]

            thresh = bin_thresh * torch.ones_like(disps.mean(dim=[1, 2]))
            count = droid_backends.depth_filter(poses, disps, intrinsics, idx, thresh)

        # Only take pixels where multiple points are consistent across views and they do not have an outlier disparity
        mask = (count >= count_thresh) & (disps > min_disp_thresh * disps.mean(dim=[1, 2], keepdim=True))
        return mask

    def camera_from_gt(self, queue: mp.Queue):
        """Extract a frame from the Queue and use the gt pose for creation"""
        idx_raw, image_raw, depth_raw, intrinsic_raw, gt_pose_raw = queue.get()
        idx = deepcopy(idx_raw)
        image, depth = image_raw.clone().squeeze().to(self.device), depth_raw.clone().to(self.device)
        intrinsic, gt_pose = intrinsic_raw.clone().to(self.device), gt_pose_raw.clone().to(self.device)

        # Always release objects passed from a mp.Queue after cloning!
        del idx_raw
        del image_raw
        del depth_raw
        del intrinsic_raw
        del gt_pose_raw
        return self.camera_from_frame(idx, image, depth, intrinsic, gt_pose)

    def camera_from_video(self, idx):
        """Extract Camera objects from a part of the video."""
        if self.video.disps_clean[idx].sum() < 1:  # Sanity check:
            self.info(f"Warning. Trying to intialize from empty frame {idx}!")
            return None

        if self.filter_dyn:
            color, depth, intrinsics, c2w, _, stat_mask = self.video.get_mapping_item(idx, self.device)

        else:
            color, depth, intrinsics, c2w, _, _ = self.video.get_mapping_item(idx, self.device)
            stat_mask = None

        return self.camera_from_frame(idx, color, depth, intrinsics, c2w, static_mask=stat_mask)

    # TODO this can be called with depth=None
    # do this for when we interpolate the keyframe poses to include nonkeyframe ones
    def camera_from_frame(
        self,
        idx: int,
        image: torch.Tensor,
        depth: Optional[torch.Tensor],
        intrinsic: torch.Tensor,
        gt_pose: torch.Tensor,
        static_mask: Optional[torch.Tensor] = None,
    ):
        """Given the image, depth, intrinsic and pose, creates a Camera object."""
        fx, fy, cx, cy = intrinsic

        height, width = image.shape[1:]
        gt_pose = torch.linalg.inv(gt_pose)
        fovx = focal2fov(fx, width)
        fovy = focal2fov(fy, height)

        if self.projection_matrix is None:
            self.projection_matrix = (
                getProjectionMatrix2(self.z_near, self.z_far, cx, cy, fx, fy, width, height)
                .transpose(0, 1)
                .to(self.device)
            )

        return Camera(
            idx,
            image,
            depth,
            gt_pose,
            self.projection_matrix,
            fx,
            fy,
            cx,
            cy,
            fovx,
            fovy,
            height,
            width,
            device=self.device,
            stat_mask=static_mask,
        )

    def get_new_cameras(self):
        """Get all new cameras from the video."""
        # Only add a batch of cameras in batch_mode
        if self.batch_mode:
            to_add = range(self.last_idx, self.cur_idx - self.delay)
            to_add = to_add[: self.kf_mng_params.default_batch_size]
        else:
            to_add = range(self.last_idx, self.cur_idx - self.delay)

        for idx in to_add:
            if self.filter_dyn:
                color, depth, intrinsics, c2w, _, stat_mask = self.video.get_mapping_item(idx, self.device)
            else:
                color, depth, intrinsics, c2w, _, _ = self.video.get_mapping_item(idx, self.device)
                stat_mask = None

            cam = self.camera_from_frame(idx, color, depth, intrinsics, c2w, static_mask=stat_mask)
            cam.update_RT(cam.R_gt, cam.T_gt)  # Assuming we found the best pose in tracking
            self.new_cameras.append(cam)

    def pose_optimizer(self, frames: List):
        """Creates an optimizer for the camera poses for all provided frames."""
        opt_params = []
        for cam in frames:
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

        return torch.optim.Adam(opt_params)

    # NOTE chen: this assumes the video object to be a gt reference
    def frame_updater(self):
        """Gets the list of frames and updates the depth and pose based on the video."""
        all_cameras = self.cameras + self.new_cameras
        all_idxs = torch.tensor([cam.uid for cam in all_cameras]).long().to(self.device)

        with self.video.get_lock():
            # print(self.video.timestamp)
            (dirty_index,) = torch.where(self.video.mapping_dirty.clone())
            dirty_index = dirty_index[dirty_index < self.cur_idx - self.delay]
        # Only update already inserted cameras
        to_update = dirty_index[torch.isin(dirty_index, all_idxs)]

        # self.info(f"Updating frames {to_update}")
        for idx in to_update:
            _, depth, _, c2w, _, _ = self.video.get_mapping_item(idx, self.device)
            cam = all_cameras[idx]
            w2c = torch.inverse(c2w)
            R = w2c[:3, :3].unsqueeze(0).detach()
            T = w2c[:3, 3].detach()
            cam.depth = depth.detach()
            cam.update_RT(R, T)

        self.video.mapping_dirty[to_update] = False

    def plot_centers(self) -> None:
        """Plot the optimized 3D Gaussians as a point cloud"""
        means = self.gaussians.get_xyz.detach().cpu().numpy()
        rgb = self.gaussians.get_features[:, 0, :].detach().cpu().numpy()
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(means)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.visualization.draw_geometries([pcd])

    def save_render(self, cam: Camera, render_path: str) -> None:
        """Save a rendered frame"""
        render_pkg = render(cam, self.gaussians, self.pipeline_params, self.background, device=self.device)
        rgb = np.uint8(255 * render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(render_path, bgr)

    def get_nonkeyframe_cameras(self, stream, trajectory_filler, batch_size: int = 16) -> List[Camera]:
        """SLAM systems operate on keyframes. This is good enough to build a good map,
        but we can get even finer details when including intermediate keyframes as well!
        Since we dont store all frames when iterating over the datastream, this requires
        reiterating over the stream again and interpolating between poses.

        NOTE Use this only after tracking finished for refining!
        """
        all_poses, all_timestamps = trajectory_filler(stream, batch_size=batch_size, return_tstamps=True)
        already_mapped = [self.video.timestamp[cam.uid] for cam in self.cameras]
        s = self.video.scale_factor
        if self.video.upsampled:
            intrinsic = self.video.intrinsics[0].to(self.device) * s
        else:
            intrinsic = self.video.intrinsics[0].to(self.device)

        new_cams = []
        for w2c, timestamp in tqdm(zip(all_poses, all_timestamps)):
            if timestamp in already_mapped:
                continue

            c2w = w2c.inv().matrix()  # [4, 4]

            # color = stream._get_image(timestamp).squeeze(0).permute(1, 2, 0).contiguous().to(self.device)
            color = stream._get_image(timestamp).clone().squeeze(0).contiguous().to(self.device)
            if not self.video.upsampled:
                color = color[..., int(s // 2 - 1) :: s, int(s // 2 - 1) :: s]
            cam = self.camera_from_frame(timestamp, color, None, intrinsic, c2w)
            cam.update_RT(cam.R_gt, cam.T_gt)
            new_cams.append(cam)

        return new_cams

    def map_refinement(
        self,
        num_iters: int = 100,
        optimize_poses: bool = False,
        prune: bool = False,
        random_frames: Optional[float] = None,
    ) -> None:
        """Refine the map with color only optimization. Instead of going over last frames, we always select random frames from the whole map."""

        if prune:
            self.abs_visibility_prune(self.kf_mng_params.abs_visibility_th)

        if random_frames is not None:
            n_rand = int(len(self.cameras) * random_frames)
            self.info(f"Info. Going over {n_rand} instead of {len(self.cameras)} of frames for optimization ...")
            if len(n_rand) > self.max_frames_refinement:
                n_chunks = n_rand // self.max_frames_refinement
                self.info(
                    f"Warning. {n_rand} Frames is too many! Optimizing over {n_chunks} chunks of frames with size {self.max_frames_refinement} ..."
                )
        else:
            if len(self.cameras) > self.max_frames_refinement:
                n_chunks = len(self.cameras) // self.max_frames_refinement
                self.info(
                    f"Warning. {len(self.cameras)} Frames is too many! Optimizing over {n_chunks} chunks of frames with size {self.max_frames_refinement} ..."
                )

        for iter in tqdm(range(num_iters)):
            # Select a random subset of frames to optimize over
            if random_frames is not None:
                abs_rand = int(len(self.cameras) * random_frames)
                to_refine = np.random.choice(len(self.cameras), abs_rand, replace=False)
                frames = [self.cameras[i] for i in to_refine]
            else:
                frames = self.cameras

            # Using all frames instead of only keyframes can lead to OOM during optimization
            # -> Use a selection of frames instead!
            if len(frames) > self.max_frames_refinement:
                chunks = [
                    frames[i : i + self.max_frames_refinement]
                    for i in range(0, len(frames), self.max_frames_refinement)
                ]
                loss = 0
                for chunk in chunks:
                    loss += self.mapping_step(iter, chunk, self.kf_mng_params.refinement, densify=False)
                    torch.cuda.empty_cache()
            else:
                loss = self.mapping_step(
                    iter, frames, self.kf_mng_params.refinement, densify=True, optimize_poses=optimize_poses
                )
            self.loss_list.append(loss / len(frames))

            if self.use_gui:
                self.q_main2vis.put_nowait(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        keyframes=self.cameras,
                    )
                )

    def get_mapping_update(self, frames: List[Camera]) -> Dict:
        """Get the index, poses and depths of the frames that were already optimized."""

        # Render frames to extract depth
        index, poses, depths = [], [], []
        for view in frames:
            render_pkg = render(view, self.gaussians, self.pipeline_params, self.background, device=self.device)
            # NOTE chen: this can be None when self.gaussians is 0. This could happen in some cases
            if render_pkg is None:
                self.info(f"Skipping view {view.uid} as no gaussians are present ...")
                if self.optimize_poses:
                    poses.append(None)
                depths.append(None)
                index.append(None)
            else:
                index.append(view.uid)
                # if self.optimize_poses:
                transform = torch.eye(4, device=self.device)
                transform[:3, :3], transform[:3, 3] = view.R, view.T
                poses.append(transform)
                depths.append(render_pkg["depth"])

        return {"index": index, "poses": poses, "depths": depths}

    def mapping_step(
        self, iter: int, frames: List[Camera], kf_mng_params: Dict, densify: bool = True, optimize_poses: bool = False
    ) -> float:
        """
        Takes the list of selected keyframes to optimize and performs one step of the mapping optimization.
        """
        # Sanity check when we dont have anything to optimize
        if len(self.gaussians) == 0:
            return 0.0

        if optimize_poses:
            pose_optimizer = self.pose_optimizer(frames)
            # pose_optimizer = self.pose_optimizer(self.cameras + frames)

        loss = 0.0
        for view in frames:

            render_pkg = render(view, self.gaussians, self.pipeline_params, self.background, device=self.device)
            # NOTE chen: this can be None when self.gaussians is 0. This could happen in some cases
            if render_pkg is None:
                self.info(f"Skipping view {view.uid} as no gaussians are present ...")
                continue

            visibility_filter, viewspace_point_tensor = render_pkg["visibility_filter"], render_pkg["viewspace_points"]
            image, radii, depth = render_pkg["render"], render_pkg["radii"], render_pkg["depth"]
            opacity, n_touched = render_pkg["opacity"], render_pkg["n_touched"]

            loss += self.mapping_rgbd_loss(
                image,
                depth,
                view,
                with_edge_weight=self.loss_params.with_edge_weight,
                with_ssim=self.loss_params.use_ssim,
                with_depth_smoothness=self.loss_params.use_depth_smoothness_reg,
            )

        # Regularize scale changes of the Gaussians
        scaling = self.gaussians.get_scaling
        isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
        loss += self.loss_params.beta * len(frames) * isotropic_loss.mean()
        loss.backward()

        with torch.no_grad():
            # Dont let Gaussians grow too much
            self.gaussians.max_radii2D[visibility_filter] = torch.max(
                self.gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter],
            )

            if densify:
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if self.last_idx > self.n_last_frames and (iter + 1) % self.kf_mng_params.prune_every == 0:

                self.gaussians.densify_and_prune(  # General pruning based on opacity and size + densification
                    kf_mng_params.densify_grad_threshold,
                    kf_mng_params.opacity_th,
                    kf_mng_params.gaussian_extent,
                    kf_mng_params.size_threshold,
                )

            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad()
            self.gaussians.update_learning_rate(iter)

            if optimize_poses:
                pose_optimizer.step()
                pose_optimizer.zero_grad()
                # go over all poses that were affected by the pose optimization
                for view in frames:
                    update_pose(view)

        return loss.item()

    def color_loss(
        self,
        image,
        cam: Camera,
        with_ssim: bool = True,
        alpha2: float = 0.85,
        mask: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute the color loss between the rendered image and the ground truth image.
        This uses a weighted sum of l1 and ssim loss.
        """
        if mask is None:
            mask = torch.ones_like(image, device=self.device)

        # FIXME chen: this exposure trick changes the data range in a data driven way
        # should we always read out the range afterwards?
        image = (torch.exp(cam.exposure_a)) * image + cam.exposure_b
        l1_rgb = l1_loss(image, cam.original_image, mask)
        # NOTE this is configured like is done in most monocular depth estimation supervision pipelines
        # TODO we could also use the multi-scale ssim which is more robust
        if with_ssim:
            # FIXME chen: this sometimes triggers illegal memory access
            # its hard to tell why, but it does not happen when the mapping gui is not used
            # this is likely due to the sent packages, which are immediately accessed while we still optimize?
            ssim_loss = ssim(image.unsqueeze(0), cam.original_image.unsqueeze(0), data_range=1.0, size_average=True)
            rgb_loss = 0.5 * alpha2 * (1 - ssim_loss) + (1 - alpha2) * l1_rgb
        else:
            rgb_loss = l1_rgb
        return rgb_loss

    def depth_loss(
        self,
        depth: torch.Tensor,
        cam: Camera,
        with_smoothness: bool = False,
        beta: float = 0.001,
        mask: Optional[torch.Tensor] = None,
    ) -> float:
        if mask is None:
            mask = torch.ones_like(depth, device=self.device)

        l1_depth = l1_loss(cam.depth, depth, mask)
        if with_smoothness:
            # NOTE this regularizes depth to be smooth in regions with low image gradient but sharp in others
            depth_reg_loss = depth_reg(depth, cam.original_image)
            depth_loss = l1_depth + beta * depth_reg_loss
        else:
            depth_loss = l1_depth

        return depth_loss

    # TODO implement this
    # TODO use a separate list of gaussians for dyn. objects
    # TODO chen: refactor this into a loss file
    # i) static loss
    # ii) dynamic loss in batch mode
    # iii) dynamic regularizer loss for scale changes and trajectory changes
    def dynamic_gaussian_loss(self, image: torch.Tensor, depth: torch.Tensor, cam: Camera):
        raise NotImplementedError()

    # TODO
    def construct_dyn_gaussians(self, mask, depth, image, cam: Camera):
        """Seed new Gaussians around an existing dyn. object mask and use the current scene depth and image as initialization

        i) preoptimize Gaussians for first frame
        ii) then adjust and finetune over whole list of frames
        """
        raise NotImplementedError()

    def mapping_rgbd_loss(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        cam: Camera,
        with_edge_weight: bool = False,
        with_ssim: bool = False,
        with_depth_smoothness: bool = False,
    ):
        has_depth = True
        if cam.depth is None:
            has_depth = False

        alpha1, alpha2 = self.loss_params.alpha1, self.loss_params.alpha2
        beta = self.loss_params.beta2

        # Mask out pixels with little information and invalid depth pixels
        rgb_pixel_mask = (cam.original_image.sum(dim=0) > self.loss_params.rgb_boundary_threshold).view(*depth.shape)
        # Only compute the loss in static regions
        if self.filter_dyn:
            rgb_pixel_mask = rgb_pixel_mask | cam.stat_mask

        if has_depth:
            # Only use valid depths for supervision
            depth_pixel_mask = ((cam.depth > 0.01) * (cam.depth < 1e7)).view(*depth.shape)
            if self.filter_dyn:
                depth_pixel_mask = depth_pixel_mask | cam.stat_mask

        if with_edge_weight:
            edge_mask_x, edge_mask_y = image_gradient_mask(
                cam.original_image
            )  # Use gt reference image for edge weight
            edge_mask = edge_mask_x | edge_mask_y  # Combine with logical OR
            rgb_mask = rgb_pixel_mask.float() * edge_mask.float()
        else:
            rgb_mask = rgb_pixel_mask.float()

        rgb_loss = self.color_loss(image, cam, with_ssim, alpha2, rgb_mask)
        if has_depth:
            depth_loss = self.depth_loss(depth, cam, with_depth_smoothness, beta, depth_pixel_mask)
            return alpha1 * rgb_loss + (1 - alpha1) * depth_loss
        else:
            return rgb_loss

    def plot_losses(self) -> None:
        fig, ax = plt.subplots(2, 1)
        ax[0].set_title(f"Loss evolution.{self.gaussians.get_xyz.shape[0]} gaussians")
        ax[0].set_yscale("log")
        ax[0].plot(self.loss_list)

        ax[1].set_yscale("log")
        ax[1].plot(self.loss_list[-self.refinement_iters :])
        plt.savefig(f"{self.output}/loss_{self.mode}.png")

    def covisibility_pruning(self):
        """Covisibility based pruning"""

        new_frames = (self.cameras + self.new_cameras)[-self.n_last_frames :]

        self.occ_aware_visibility = {}
        for view in new_frames:
            render_pkg = render(view, self.gaussians, self.pipeline_params, self.background)
            self.occ_aware_visibility[view.uid] = (render_pkg["n_touched"] > 0).long()

        new_idx = [view.uid for view in new_frames]
        sorted_frames = sorted(new_idx, reverse=True)

        self.gaussians.n_obs.fill_(0)
        for _, visibility in self.occ_aware_visibility.items():
            self.gaussians.n_obs += visibility.cpu()

        # Gaussians added on the last prune_last frames
        mask = self.gaussians.unique_kfIDs >= sorted_frames[self.kf_mng_params.prune_last - 1]
        to_prune = torch.logical_and(self.gaussians.n_obs <= self.kf_mng_params.visibility_th, mask)
        if to_prune.sum() > 0:
            self.gaussians.prune_points(to_prune.cuda())
            # for idx in new_idx:
            #     self.occ_aware_visibility[idx] = self.occ_aware_visibility[idx][~to_prune]
        self.info(f"Covisibility pruning removed {to_prune.sum()} gaussians")

    def abs_visibility_prune(self, threshold: int = 2):
        """
        Absolute covisibility based pruning. Removes all gaussians
        that are not seen in at least abs_visibility_th views.
        """

        self.occ_aware_visibility = {}
        for view in self.cameras + self.new_cameras:
            render_pkg = render(view, self.gaussians, self.pipeline_params, self.background, device=self.device)
            self.occ_aware_visibility[view.uid] = (render_pkg["n_touched"] > 0).long()

        self.gaussians.n_obs.fill_(0)
        for _, visibility in self.occ_aware_visibility.items():
            self.gaussians.n_obs += visibility.cpu()

        mask = self.gaussians.n_obs < threshold
        if mask.sum() > 0:
            self.gaussians.prune_points(mask.cuda())
        self.info(f"Absolute visibility pruning removed {mask.sum()} gaussians")

    def select_keyframes(self):
        """Select the last n1 frames and n2 other random frames from all."""
        if len(self.cameras) <= self.n_last_frames + self.n_rand_frames:
            keyframes = self.cameras
            keyframes_idx = np.arange(len(self.cameras))
        else:
            keyframes_idx = np.random.choice(len(self.cameras) - self.n_last_frames, self.n_rand_frames, replace=False)
            keyframes = self.cameras[-self.n_last_frames :] + [self.cameras[i] for i in keyframes_idx]
        return keyframes, keyframes_idx

    def _last_call(self, mapping_queue: mp.Queue, received_item: mp.Event):
        """We already build up the map based on the SLAM system and finetuned over it.
        Depending on compute budget this has been done scarcely.
        This call runs many more iterations for refinement and densification to get a high quality map.

        Since the SLAM system operates on keyframes, but we have many more views in our video stream, we can use additional
        supervision from non-keyframes to get higher detail.
        """
        # Free memory before doing refinement
        torch.cuda.empty_cache()
        gc.collect()

        self.info(f"#Gaussians before Map Refinement: {len(self.gaussians)}")

        if self.slam.dataset is not None and self.use_non_keyframes:
            self.info("Interpolating trajectory to get non-keyframe Cameras for refinement ...")
            non_kf_cams = self.get_nonkeyframe_cameras(self.slam.dataset, self.slam.traj_filler)
            # Reinstatiate an empty traj_filler, we only reuse this during eval
            # this deletes the graph and frees up memory
            del self.slam.traj_filler
            self.slam.traj_filler = PoseTrajectoryFiller(
                net=self.slam.net, video=self.slam.video, device=self.slam.device
            )
            torch.cuda.empty_cache()
            gc.collect()

            self.info(f"Added {len(non_kf_cams)} new cameras: {[cam.uid for cam in non_kf_cams]}")
            # Initialize without additional Gaussians, since we only use the additional frames as better supervision
            for cam in non_kf_cams:
                self.initialized = True

            # Keyframe cameras we add during SLAM: cam.uid = video.idx, which is the position in our video buffer
            # Non-keyframe cameras we interpolated: cam.uid = stream.timestamp
            # We can get the global mapping, because: video.timestamp[video.idx] = stream.timestamp
            for cam in self.cameras:
                cam.uid = int(self.video.timestamp[cam.uid].item())  # Reassign local keyframe ids to global stream ids
            self.cameras += non_kf_cams  # Add to set of cameras
            # Reorder according to global uid
            self.cameras = sorted(self.cameras, key=lambda x: x.uid)

        self.info("\nMapping refinement starting")
        # NOTE MonoGS does 26k iterations for a single camera, while we do 100 for multiple cameras
        self.map_refinement(num_iters=self.refinement_iters, optimize_poses=True, random_frames=0.2)
        self.info(f"#Gaussians after Map Refinement: {len(self.gaussians)}")
        self.info("Mapping refinement finished")

        self.gaussians.save_ply(f"{self.output}/mesh/final_{self.mode}.ply")
        self.info(f"Mesh saved at {self.output}/mesh/final_{self.mode}.ply")

        if self.save_renders:
            for cam in self.cameras:
                self.save_render(cam, f"{self.output}/renders/final/{cam.uid}.png")

        self.plot_losses()

        self.info(f"Final mapping loss: {self.loss_list[-1]}")
        self.info(f"{len(self.iteration_info)} iterations, {len(self.cameras)/len(self.iteration_info)} cams/it")

        ## export the cameras and gaussians to the terminate process
        if self.evaluate:
            mapping_queue.put(
                gui_utils.EvaluatePacket(
                    pipeline_params=clone_obj(self.pipeline_params),
                    cameras=self.cameras[:],
                    gaussians=clone_obj(self.gaussians),
                    background=clone_obj(self.background),
                )
            )
        else:
            mapping_queue.put("None")
        received_item.wait()  # Wait until the Packet got delivered

    def _update(self):
        """Update our rendered map by:
        i) Pull a filtered update from the sparser SLAM map
        ii) Add new Gaussians based on new views
        iii) Run a bunch of optimization steps to update Gaussians and camera poses
        iv) Prune the render map based on visibility

        Finally we send the point cloud version of this rendered map back to the SLAM system.
        """
        self.info("Currently has: {} gaussians".format(len(self.gaussians)))

        # Filter map based on multiview_consistency and uncertainty
        # NOTE This could be improved by only filtering the new and existing cameras instead of video.mapping_dirty
        self.video.filter_map(
            min_count=self.kf_mng_params.filter.mv_count_thresh,
            bin_thresh=self.kf_mng_params.filter.bin_thresh,
            unc_threshold=self.kf_mng_params.filter.confidence_thresh,
            use_multiview_consistency=self.filter_multiview,
            use_uncertainty=self.filter_uncertainty,
        )

        self.get_new_cameras()  # Add new cameras
        self.last_idx = self.new_cameras[-1].uid + 1

        self.info(f"Added {len(self.new_cameras)} new cameras: {[cam.uid for cam in self.new_cameras]}")
        self.frame_updater()  # Update all changed cameras with new information from SLAM system

        for cam in self.new_cameras:
            if not self.initialized:
                self.initialized = True
                self.gaussians.extend_from_pcd_seq(cam, cam.uid, init=True)
                self.info(f"Initialized with {self.gaussians.get_xyz.shape[0]} gaussians")
            else:
                ng_before = len(self.gaussians)
                self.gaussians.extend_from_pcd_seq(cam, cam.uid, init=False)
                # self.info(f"Added {len(self.gaussians) - ng_before} gaussians based on view {cam.uid}")

        # We might have 0 Gaussians in some cases, so no need to run optimizer
        if len(self.gaussians) == 0:
            self.info("No Gaussians to optimize, skipping mapping step ...")
            return

        # Optimize gaussians
        for iter in range(self.mapping_iters):
            frames = self.select_keyframes()[0] + self.new_cameras
            if len(frames) == 0:
                self.loss_list.append(0.0)
                continue
            loss = self.mapping_step(
                iter, frames, self.kf_mng_params.mapping, densify=True, optimize_poses=self.optimize_poses
            )
            self.loss_list.append(loss / len(frames))

        # Keep track of how well the Rendering is doing
        self.info(f"Loss:  {self.loss_list[-1]}")

        if len(self.iteration_info) % 1 == 0:
            if self.kf_mng_params.prune_mode == "abs":
                # Absolute visibility pruning for all gaussians
                self.abs_visibility_prune(self.kf_mng_params.abs_visibility_th)
            elif self.kf_mng_params.prune_mode == "new":
                self.covisibility_pruning()  # Covisibility pruning for recently added gaussians

        # Render system -> SLAM system
        if self.feedback_map:
            to_set = self.get_mapping_update(frames)
            self.info("Feeding back to Tracking ...")
            self.video.set_mapping_item(**to_set)

        # Update visualization
        if self.use_gui:
            if cam.depth is not None:
                gtdepth = cam.depth.detach().cpu().numpy()
            else:
                gtdepth = None
            self.q_main2vis.put_nowait(
                gui_utils.GaussianPacket(
                    gaussians=clone_obj(self.gaussians),
                    current_frame=cam,
                    keyframes=self.cameras,
                    kf_window=None,
                    gtcolor=(cam.original_image),
                    gtdepth=gtdepth,
                )
            )

        # Save renders
        if self.save_renders and cam.uid % 5 == 0:
            self.save_render(cam, f"{self.output}/renders/mapping/{cam.uid}.png")

        # Keep track of added cameras
        self.cameras += self.new_cameras
        self.iteration_info.append(len(self.new_cameras))
        self.new_cameras = []

    def __call__(self, mapping_queue: mp.Queue, received_item: mp.Event, the_end=False):

        # self.cur_idx = int(self.video.filtered_id.item())
        self.cur_idx = self.video.counter.value
        if self.last_idx + self.delay < self.cur_idx and self.cur_idx > self.warmup:
            self._update()
            return False

        elif the_end and self.last_idx + self.delay >= self.cur_idx:
            self._update()  # Run another call to catch the last batch of keyframes
            self._last_call(mapping_queue=mapping_queue, received_item=received_item)
            return True
