import os
from copy import deepcopy
from typing import List, Dict
from termcolor import colored

import torch
import torch.multiprocessing as mp
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt

from .gaussian_splatting.gui import gui_utils, slam_gui
from .gaussian_splatting.gaussian_renderer import render
from .gaussian_splatting.scene.gaussian_model import GaussianModel
from .gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2, focal2fov
from .gaussian_splatting.multiprocessing_utils import clone_obj
from .gaussian_splatting.camera_utils import Camera
from .gaussian_splatting.pose_utils import update_pose

import droid_backends


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
        self.mapping_iters = cfg.mapping.mapping_iters
        self.use_gui = cfg.mapping.use_gui
        self.save_renders = cfg.mapping.save_renders
        self.warmup = max(cfg.mapping.warmup, cfg.tracking.warmup)
        self.batch_mode = cfg.mapping.batch_mode  # Take a batch of all unupdated frames at once

        self.optimize_poses = cfg.mapping.optimize_poses
        self.opt_params = cfg.mapping.opt_params
        self.pipeline_params = cfg.mapping.pipeline_params
        self.kf_mng_params = cfg.mapping.keyframes
        # Always take n last frames from current index and optimize additional random frames globally
        self.n_last_frames = self.kf_mng_params.n_last_frames
        self.n_rand_frames = self.kf_mng_params.n_rand_frames

        self.filter_uncertainty = cfg.mapping.filter_uncertainty
        self.filter_multiview = cfg.mapping.filter_multiview

        self.loss_params = cfg.mapping.loss

        self.sh_degree = 3 if cfg.mapping.use_spherical_harmonics else 0
        self.gaussians = GaussianModel(self.sh_degree, config=cfg.mapping.input)
        self.gaussians.init_lr(cfg.mapping.init_lr)
        self.gaussians.training_setup(self.opt_params)

        bg_color = [1, 1, 1]  # White background
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if gui_qs is not None:
            self.q_main2vis, self.q_vis2main = gui_qs
            self.use_gui = True
        else:
            self.use_gui = False


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
        self.video.filter_map(
            min_count=self.kf_mng_params.filter.mv_count_thresh, bin_thresh=self.kf_mng_params.filter.bin_thresh
        )
        color, depth, intrinsics, c2w, _ = self.video.get_mapping_item(idx, self.device)
        return self.camera_from_frame(idx, color, depth, intrinsics, c2w)

    def camera_from_frame(
        self, idx: int, image: torch.Tensor, depth: torch.Tensor, intrinsic: torch.Tensor, gt_pose: torch.Tensor
    ):
        """Given the image, depth, intrinsic and pose, creates a Camera object."""
        fx, fy, cx, cy = intrinsic

        height, width = image.shape[1:]
        gt_pose = torch.linalg.inv(gt_pose)  # They invert the poses in the dataloader
        znear = 0.01
        zfar = 1000.0  # TODO make this configurable?
        fovx = focal2fov(fx, width)
        fovy = focal2fov(fy, height)

        if self.projection_matrix is None:
            self.projection_matrix = (
                getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, width, height).transpose(0, 1).to(self.device)
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
        )

    def get_new_cameras(self):
        """Get all new cameras from the video."""
        if self.batch_mode:
            to_add = range(self.last_idx, self.cur_idx - self.delay)
            to_add = to_add[: self.kf_mng_params.default_batch_size]
        else:
            to_add = range(self.last_idx, self.cur_idx - self.delay)

        for idx in to_add:
            color, depth, intrinsics, c2w, _ = self.video.get_mapping_item(idx, self.device)
            color = color.permute(2, 0, 1)

            cam = self.camera_from_frame(idx, color, depth, intrinsics, c2w)
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

    def frame_updater(self):
        """Gets the list of frames and updates the depth and pose based on the video."""
        all_cameras = self.cameras + self.new_cameras
        all_idxs = torch.tensor([cam.uid for cam in all_cameras]).long().to(self.device)

        with self.video.get_lock():
            (dirty_index,) = torch.where(self.video.mapping_dirty.clone())
            dirty_index = dirty_index[dirty_index < self.cur_idx - self.delay]
        # Only update already inserted cameras
        to_update = dirty_index[torch.isin(dirty_index, all_idxs)]

        #self.info(f"Updating frames {to_update}")
        for idx in to_update:
            _, depth, _, c2w, _ = self.video.get_mapping_item(idx, self.device)
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
        render_pkg = render(cam, self.gaussians, self.pipeline_params, self.background)
        rgb = np.uint8(255 * render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(render_path, bgr)

    def mapping_step(
        self, iter: int, frames: List[Camera], kf_mng_params: Dict, densify: bool = True, optimize_poses: bool = False
    ) -> float:
        """
        Takes the list of selected keyframes to optimize and performs one step of the mapping optimization.
        """
        if optimize_poses:
            pose_optimizer = self.pose_optimizer(self.cameras)

        # Sanity check when we dont have anything to optimize
        if self.gaussians.get_xyz.shape[0] == 0:
            return 0.0

        loss = 0.0
        self.occ_aware_visibility = {}
        for view in frames:

            render_pkg = render(view, self.gaussians, self.pipeline_params, self.background)
            # NOTE chen: this can be None when self.gaussians is 0. This could happen in some cases
            if render_pkg is None:
                continue
            visibility_filter, viewspace_point_tensor = render_pkg["visibility_filter"], render_pkg["viewspace_points"]
            image, radii, depth = render_pkg["render"], render_pkg["radii"], render_pkg["depth"]
            opacity, n_touched = render_pkg["opacity"], render_pkg["n_touched"]

            loss += self.mapping_loss(image, depth, view)
            # TODO chen: why do we need this?
            # Only take into account last frames and not random ones
            if self.last_idx - view.uid <= self.n_last_frames:
                self.occ_aware_visibility[view.uid] = (n_touched > 0).long()

        # Regularize scale changes of the Gaussians
        scaling = self.gaussians.get_scaling
        isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
        loss += self.loss_params.beta * len(frames) * isotropic_loss.mean()

        loss.backward()

        with torch.no_grad():
            self.gaussians.max_radii2D[visibility_filter] = torch.max(
                self.gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter],
            )

            if densify:
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if self.last_idx > self.n_last_frames and (iter + 1) % self.kf_mng_params.prune_every == 0:
                # if self.mode != "rgbd":
                #     self.covisibility_pruning() # Covisibility based pruning for recently added gaussians 
                self.gaussians.densify_and_prune( # General pruning based on opacity and size + densification
                kf_mng_params.densify_grad_threshold,
                kf_mng_params.opacity_th,
                kf_mng_params.gaussian_extent,
                kf_mng_params.size_threshold,
                )

            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad()
            self.gaussians.update_learning_rate(10*self.last_idx) # TODO leon: Only taking keyframes, the update is to small

            if optimize_poses:
                pose_optimizer.step()
                pose_optimizer.zero_grad()
                for view in frames:
                    update_pose(view)

            return loss.item()

    def mapping_loss(self, image: torch.Tensor, depth: torch.Tensor, cam: Camera):
        """Compute a weighted l1 loss between i) the rendered image and the ground truth image and ii) the rendered depth and the ground truth depth.

        NOTE: the groundtruth depth here is the depth from the VSLAM system, not the external sensor depth!
        """

        alpha = self.loss_params.alpha
        rgb_boundary_threshold = self.loss_params.rgb_boundary_threshold

        gt_image, gt_depth = cam.original_image, cam.depth

        rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
        depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)  # Only use valid depths for supervision

        image = (torch.exp(cam.exposure_a)) * image + cam.exposure_b
        l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
        l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

        return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()
    
    def covisibility_pruning(self):
        """Covisibility based pruning"""

        new_frames = (self.cameras + self.new_cameras)[-self.n_last_frames :] 
        new_idx = [view.uid for view in new_frames]
        sorted_frames = sorted(new_idx, reverse=True)

        self.gaussians.n_obs.fill_(0)
        for _, visibility in self.occ_aware_visibility.items():
            self.gaussians.n_obs += visibility.cpu()

        # Gaussians added on the last prune_last frames
        mask = self.gaussians.unique_kfIDs >= sorted_frames[self.kf_mng_params.prune_last - 1]

        to_prune = torch.logical_and(self.gaussians.n_obs <= self.kf_mng_params.visibility_th, mask)
        self.gaussians.prune_points(to_prune.cuda())
        for idx in new_idx:
            self.occ_aware_visibility[idx] = self.occ_aware_visibility[idx][~to_prune]
        self.info(f"Covisibility based pruning removed {to_prune.sum()} gaussians")


    def abs_visibility_prune(self, threshold:int=None):
        """
        Absolute covisibility based pruning. Removes all gaussians 
        that are not seen in at least abs_visibility_th views.
        """

        self.occ_aware_visibility = {}
        for view in (self.cameras + self.new_cameras):
            render_pkg = render(view, self.gaussians, self.pipeline_params, self.background)
            self.occ_aware_visibility[view.uid] = (render_pkg["n_touched"] > 0).long()

        #print(self.occ_aware_visibility[1])
        self.gaussians.n_obs.fill_(0)
        for _, visibility in self.occ_aware_visibility.items():
            self.gaussians.n_obs += visibility.cpu()

        if threshold is None:
            threshold = self.kf_mng_params.abs_visibility_th

        mask = self.gaussians.n_obs < threshold
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
        """Do one last refinement over the map"""

        self.info("\nMapping refinement starting")
        self.abs_visibility_prune(threshold=3)
        for iter in range(self.refinement_iters):
            loss = self.mapping_step(
                iter, self.cameras, self.kf_mng_params.refinement, densify=False, optimize_poses=self.optimize_poses
            )
            self.loss_list.append(loss / len(self.cameras)) # Average loss per frame

            if self.use_gui:
                self.q_main2vis.put_nowait(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        keyframes=self.cameras,
                    )
                )
        self.info("Mapping refinement finished")

        self.gaussians.save_ply(f"{self.output}/mesh/final_{self.mode}.ply")
        self.info(f"Mesh saved at {self.output}/mesh/final_{self.mode}.ply")

        if self.save_renders:
            for cam in self.cameras:
                self.save_render(cam, f"{self.output}/renders/final/{cam.uid}.png")

        fig, ax = plt.subplots(2,1)
        ax[0].set_title("Loss per frame evolution")
        ax[0].set_yscale("log")
        ax[0].plot(self.loss_list)

        ax[1].set_yscale("log")
        ax[1].set_title(f"Mode: {self.mode}. Gaussians: {self.gaussians.get_xyz.shape[0]}")
        ax[1].plot(self.loss_list[-self.refinement_iters:])
        plt.savefig(f"{self.output}/loss_{self.mode}.png")

        self.info(f"Final mapping loss: {self.loss_list[-1]}")

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
            received_item.wait()  # Wait until the Packet got delivered

    def __call__(self, mapping_queue: mp.Queue, received_item: mp.Event, the_end=False):

        # self.cur_idx = int(self.video.filtered_id.item())
        self.cur_idx = self.video.counter.value
        if self.last_idx + self.delay < self.cur_idx and self.cur_idx > self.warmup:

            # Filter map based on multiview_consistency and uncertainty
            # NOTE This could be improved by only filtering the new and existing cameras instead of video.mapping_dirty
            self.video.filter_map(
                min_count=self.kf_mng_params.filter.mv_count_thresh,
                bin_thresh=self.kf_mng_params.filter.bin_thresh,
                unc_threshold=self.kf_mng_params.filter.confidence_thresh,
                use_multiview_consistency=self.filter_multiview,
                use_uncertainty=self.filter_uncertainty,
            )

            # TODO chen: what happens if the map undergoes radical changes in the SLAM system?
            # e.g. a loop closure that rapidly changes the scale of the map
            # would this destory the Gaussians as well?
            self.get_new_cameras()  # Add new cameras
            self.last_idx = self.new_cameras[-1].uid + 1

            self.info(f"Added {len(self.new_cameras)} new cameras: {[cam.uid for cam in self.new_cameras]}")
            self.frame_updater()  # Update all cameras with new information from SLAM system

            for cam in self.new_cameras:
                if not self.initialized:
                    self.initialized = True
                    self.gaussians.extend_from_pcd_seq(cam, cam.uid, init=True)
                    self.info(f"Initialized with {self.gaussians.get_xyz.shape[0]} gaussians")
                else:
                    n_g = self.gaussians.get_xyz.shape[0]
                    self.gaussians.extend_from_pcd_seq(cam, cam.uid, init=False)
                    #self.info(f"Added {self.gaussians.get_xyz.shape[0] - n_g} gaussians based on view {cam.uid}")



            # We might have 0 Gaussians in some cases
            if self.gaussians.get_xyz.shape[0] == 0:
                return

            # Optimize  gaussians
            for iter in range(self.mapping_iters):
                frames = self.select_keyframes()[0] + self.new_cameras
                loss = self.mapping_step(
                    iter, frames, self.kf_mng_params.mapping, densify=True, optimize_poses=self.optimize_poses
                )
                self.loss_list.append(loss / len(frames))

            # for param_group in self.gaussians.optimizer.param_groups:
            #     if param_group["name"] == "xyz":
            #         print(param_group["lr"])

            if self.last_idx % 1 == 0 and self.last_idx > self.n_last_frames:
                self.abs_visibility_prune()
                
                
            # Update visualization
            if self.use_gui:
                self.q_main2vis.put_nowait(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=cam,
                        keyframes=self.cameras, # TODO: only pass cameras that got updated
                        # keyframe=cam,
                        kf_window=None,
                        gtcolor=cam.original_image,
                        gtdepth=cam.depth.detach().cpu().numpy(),
                    )
                )


            # Save renders
            if self.save_renders and cam.uid % 5 == 0:
                self.save_render(cam, f"{self.output}/renders/mapping/{cam.uid}.png")

            # Keep track of added cameras
            self.cameras += self.new_cameras
            self.new_cameras = []


        
        elif the_end and self.last_idx + self.delay >= self.cur_idx:
            self._last_call(mapping_queue=mapping_queue, received_item=received_item)
            return True






    


