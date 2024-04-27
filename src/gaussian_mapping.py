import os
from typing import List
from munch import munchify
from termcolor import colored

import torch
import torch.multiprocessing as mp
import open3d as o3d
import numpy as np
import cv2

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

NOTE this could a be standalone SLAM system itself, but we use it as finetuning here to be independent of rendering alone.
"""


class GaussianMapper(object):
    """
    SLAM from Rendering with 3D Gaussian Splatting.
    """

    def __init__(self, config, args, slam, mapping_queue=None):
        self.config = config
        self.args = args
        self.slam = slam
        self.video = slam.video
        self.device = args.device
        self.mode = args.mode
        self.model_params = munchify(config["model_params"])
        self.opt_params = munchify(config["opt_params"])
        self.pipeline_params = munchify(config["pipeline_params"])
        self.training_params = munchify(config["Training"])
        self.setup = munchify(config["Setup"])

        self.use_spherical_harmonics = False
        self.model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(self.model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(self.opt_params)

        bg_color = [1, 1, 1]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if self.setup.use_gui and not config["evaluate"]:
            self.q_main2vis = mp.Queue()
            self.q_vis2main = mp.Queue()
            self.params_gui = gui_utils.ParamsGUI(
                pipe=self.pipeline_params,
                background=self.background,
                gaussians=self.gaussians,
                q_main2vis=self.q_main2vis,
                q_vis2main=self.q_vis2main,
            )
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            self.info("GUI process started!")

        self.cameras = []
        self.loss_list = []
        self.last_idx = -1
        self.initialized = False

        self.n_last_frames = 10
        self.n_rand_frames = 5

        self.show_filtered = False

        self.mapping_queue = mapping_queue

    def info(self, msg: str):
        print(colored("[Gaussian Mapper] " + msg, "magenta"))

    # TODO only take object here if dirty_index is set to see if this is a new updated frame
    def camera_from_video(self, idx):
        """Takes the frame data from the video and returns a Camera object."""
        color, depth, c2w, _, _ = self.video.get_mapping_item(idx, self.device)
        color = color.permute(2, 0, 1)
        intrinsics = self.video.intrinsics[0] * self.video.scale_factor

        # FIXME chen: the depth filter is not used correctly here!
        # TODO come up with a better way to filter points or use enough views

        return self.camera_from_frame(idx, color, depth, intrinsics, c2w)

    def camera_from_frame(
        self, idx: int, image: torch.Tensor, depth: torch.Tensor, intrinsic: torch.Tensor, gt_pose: torch.Tensor
    ):
        """Given the image, depth, intrinsic and pose, creates a Camera object."""
        fx, fy, cx, cy = intrinsic
        height, width = image.shape[1:]
        gt_pose = torch.linalg.inv(gt_pose)  # They invert the poses in the dataloader
        znear = 0.01
        zfar = 100.0  # TODO make this configurable?
        fovx = focal2fov(fx, width)
        fovy = focal2fov(fy, height)

        if not self.initialized:
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

    def camera_from_gt(self):
        """Extract a frame from the Queue and use the gt pose for creation"""
        idx, image, depth, intrinsic, gt_pose = self.mapping_queue.get()
        image, depth = image.squeeze().to(self.device), depth.to(self.device)
        intrinsic, gt_pose = intrinsic.to(self.device), gt_pose.to(self.device)
        return self.camera_from_frame(idx, image, depth, intrinsic, gt_pose)

    def pose_optimizer(self, frames: list):
        """Creates an optimizer for the camera poses for all provided frames."""
        opt_params = []
        for cam in frames:
            opt_params.append(
                {
                    "params": [cam.cam_rot_delta],
                    "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                    "name": "rot_{}".format(cam.uid),
                }
            )
            opt_params.append(
                {
                    "params": [cam.cam_trans_delta],
                    "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                    "name": "trans_{}".format(cam.uid),
                }
            )
            opt_params.append(
                {
                    "params": [cam.exposure_a],
                    "lr": 0.01,
                    "name": "exposure_a_{}".format(cam.uid),
                }
            )
            opt_params.append(
                {
                    "params": [cam.exposure_b],
                    "lr": 0.01,
                    "name": "exposure_b_{}".format(cam.uid),
                }
            )

        return torch.optim.Adam(opt_params)

    # FIXME this can fail when dirty index is empty
    # TODO chen: why does this even depend on dirty index, when we call it with last_idx?!
    def depth_filter(self, idx: int):
        """
        Gets the video and the time idex and returns the mask.


        NOTE andrei: Tried:
        -setting the cuda device to 1
        -set the debug flags in bashrc
        -using clone_obj
        -checked the locking for dirty_index
        -if you access poses or disps_up or poses directly you get the error
        -setting torch.backends.cudnn.benmark = False
        -tried running it on cpu (got another error Input type (c10::Half) and bias type (float) should be the same)
        """

        # TODO why doesnt it work only with one index?
        with self.video.get_lock():
            (dirty_index,) = torch.where(self.video.dirty.clone())
            dirty_index = dirty_index

            poses = torch.index_select(clone_obj(self.video.poses), 0, dirty_index)
            disps = torch.index_select(clone_obj(self.video.disps_up), 0, dirty_index)
            thresh = 0.1 * torch.ones_like(disps.mean(dim=[1, 2]))
            intrinsics = self.video.intrinsics[0] * self.video.scale_factor
            count = droid_backends.depth_filter(poses, disps, intrinsics, dirty_index, thresh)

        # Heuristic to filter out noisy points
        is_consistent = (count >= 2) & (disps > 0.05 * disps.mean(dim=[1, 2], keepdim=True))
        if len(is_consistent) > 0:
            is_consistent = is_consistent[idx]

        self.info(f"Valid points: {100* is_consistent.sum()/is_consistent.numel()} %")
        return is_consistent

    @torch.no_grad()
    def frame_updater(self, frames: List):
        """Gets the list of frames and updates the depth and pose based on the video."""
        for cam in frames:

            _, depth, c2w, _, _ = self.video.get_mapping_item(cam.uid, self.video.device)
            # FIXME chen: this is not a good strategy, the depth filter check for consistency ACROSS multiple frames
            # dont use it for a single frame
            # as of now this is only useful for filtering out extreme disparities

            w2c = torch.inverse(c2w)
            R = w2c[:3, :3].unsqueeze(0).detach()
            T = w2c[:3, 3].detach()
            cam.depth = depth.detach()
            cam.update_RT(R, T)

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
        self, iter: int, frames: list, pruning_params: dict, densify: bool = True, optimize_poses: bool = False
    ) -> float:
        """Takes the list of selected keyframes to optimize and performs one step of the mapping optimization."""
        if self.setup.optimize_poses:
            pose_optimizer = self.pose_optimizer(self.cameras)

        loss = 0
        self.occ_aware_visibility = {}
        for view in frames:

            render_pkg = render(view, self.gaussians, self.pipeline_params, self.background)
            visibility_filter, viewspace_point_tensor = render_pkg["visibility_filter"], render_pkg["viewspace_points"]
            image, radii, depth = render_pkg["render"], render_pkg["radii"], render_pkg["depth"]
            opacity, n_touched = render_pkg["opacity"], render_pkg["n_touched"]

            loss += self.mapping_loss(image, depth, view)

        scaling = self.gaussians.get_scaling
        isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
        # TODO chen: is this 5 simply because of the number of random frames or a hyperparameter?
        loss += 5 * len(frames) * isotropic_loss.mean()  # TODO chen: why was the 5 not in the previous commits?
        loss.backward()

        with torch.no_grad():
            self.gaussians.max_radii2D[visibility_filter] = torch.max(
                self.gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter],
            )

            if densify:
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if self.last_idx > 0 and (iter + 1) % self.training_params.prune_every == 0:
                self.gaussians.densify_and_prune(
                    self.opt_params.densify_grad_threshold,
                    pruning_params.gaussian_th,
                    pruning_params.gaussian_extent,
                    pruning_params.size_threshold,
                )
            self.gaussians.optimizer.step()

            self.gaussians.optimizer.zero_grad()
            self.gaussians.update_learning_rate(self.last_idx)

            if optimize_poses:
                pose_optimizer.step()
                pose_optimizer.zero_grad()
                for view in frames:
                    update_pose(view)

            return loss.item()

    def mapping_loss(self, image: torch.Tensor, depth: torch.Tensor, cam: Camera) -> float:
        """Compute a weighted l1 loss between i) the rendered image and the ground truth image and ii) the rendered depth and the ground truth depth."""
        alpha = self.config["Training"]["alpha"] if "alpha" in self.config["Training"] else 0.95
        rgb_boundary_threshold = self.config["Training"].get("rgb_boundary_threshold", 0.01)

        gt_image, gt_depth = cam.original_image, cam.depth

        rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
        depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

        image = (torch.exp(cam.exposure_a)) * image + cam.exposure_b
        l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
        l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

        return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()

    # TODO implement and use this
    def prune_gaussians(self):
        """Covisibility based pruning"""

        prune_coviz = 3  # Visibility threshold # TODO: add both parameters to the cfg

        new_frames = self.cameras[-self.n_last_frames :]
        new_idx = [view.uid for view in new_frames]
        sorted_frames = sorted(new_idx, reverse=True)

        self.gaussians.n_obs.fill_(0)
        for _, visibility in self.occ_aware_visibility.items():
            self.gaussians.n_obs += visibility.cpu()

        mask = self.gaussians.unique_kfIDs >= sorted_frames[2]  # Gaussians added on the last 3 frames

        to_prune = torch.logical_and(self.gaussians.n_obs <= prune_coviz, mask)
        self.gaussians.prune_points(to_prune.cuda())
        for idx in new_idx:
            self.occ_aware_visibility[idx] = self.occ_aware_visibility[idx][~to_prune]
        self.info(f"Covisibility based pruning removed {to_prune.sum()} gaussians")

    # TODO chen: make the 5 configurable, because this should be treated as a hyperparameter
    def select_keyframes(self):
        """Select last 5 frames and other 5 random frames"""
        if len(self.cameras) <= 10:
            keyframes = self.cameras
            keyframes_idx = np.arange(len(self.cameras))
        else:
            keyframes_idx = np.random.choice(len(self.cameras) - 5, 5, replace=False)
            keyframes = self.cameras[-5:] + [self.cameras[i] for i in keyframes_idx]
        return keyframes, keyframes_idx

    def _last_call(self) -> bool:
        self.info("Mapping refinement starting")

        for iter in range(self.setup.refinement_iters):
            loss = self.mapping_step(
                iter,
                self.cameras,
                self.training_params.refinement,
                densify=False,
                optimize_poses=self.setup.optimize_poses,
            )
            self.loss_list.append(loss / len(self.cameras))

            if self.setup.use_gui:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        keyframes=self.cameras,
                    )
                )
        self.info("Mapping refinement finished")

        self.gaussians.save_ply(f"{self.setup.mesh_path}/final.ply")
        self.info(f"Mesh saved at {self.setup.mesh_path}/final.ply")

        if self.setup.save_renders:
            for cam in self.cameras:
                self.save_render(cam, f"{self.setup.render_path}/final/{cam.uid}.png")

        ## export the cameras and gaussians to the terminate process
        self.mapping_queue.put(
            gui_utils.EvaluatePacket(
                pipeline_params=self.pipeline_params,
                cameras=clone_obj(self.cameras),
                gaussians=clone_obj(self.gaussians),
                background=self.background,
            )
        )
        return True

    # TODO use self.cameras and batch mode
    # TODO refactor the optimization loop, which we use in both every call and the last call
    def _optimize(self, cam, the_end: bool) -> None:
        # Optimze gaussians
        for iter in range(self.setup.mapping_iters):
            # TODO chen: should we use the random frame_ids here later?
            frames, _ = self.select_keyframes()

            if self.setup.update_frames and not the_end:
                self.frame_updater(frames)

            loss = self.mapping_step(
                iter, frames, self.training_params, densify=True, optimize_poses=self.setup.optimize_poses
            )
            self.loss_list.append(loss / len(frames))

        # Update visualization
        if self.setup.use_gui:
            self.q_main2vis.put(
                # TODO chen: why do we only clone gaussians?
                gui_utils.GaussianPacket(
                    gaussians=clone_obj(self.gaussians),
                    current_frame=cam,
                    # keyframes=self.cameras,  # TODO: only pass cameras that got updated
                    keyframe=cam,
                    kf_window=None,
                    gtcolor=cam.original_image.detach().cpu(),
                    gtdepth=cam.depth.detach().cpu().numpy(),
                )
            )

    def __call__(self, the_end: bool = False):
        cur_idx = int(self.video.filtered_id.item())

        # TODO make conditional more readable
        # TODO refactor so this is cleaner
        if self.last_idx + 2 < cur_idx and cur_idx > self.setup.warmup:

            self.last_idx += 1
            self.info(
                f"\nStarting frame: {self.last_idx}. Gaussians: {self.gaussians.get_xyz.shape[0]}. Video at {cur_idx}"
            )

            # Add camera of the last frame
            cam = self.camera_from_video(self.last_idx)
            cam.update_RT(cam.R_gt, cam.T_gt)  # Assuming we found the best pose
            self.cameras.append(cam)

            # Add gaussians based on the new view
            if not self.initialized:
                self.gaussians.extend_from_pcd_seq(cam, cam.uid, init=True)
                self.initialized = True
            else:
                n_g = self.gaussians.get_xyz.shape[0]
                self.gaussians.extend_from_pcd_seq(cam, cam.uid, init=False)
                self.info(f"Added {self.gaussians.get_xyz.shape[0] - n_g} gaussians for the new view")

            # Optimize gaussians
            self._optimize(cam, the_end)
            msg = "Frame: {}. Gaussians: {}. Video at {}".format(cam.uid, self.gaussians.get_xyz.shape[0], cur_idx)
            self.info(msg)

            # Save renders
            if self.setup.save_renders and cam.uid % 5 == 0:
                self.save_render(cam, f"{self.setup.render_path}/mapping/{cam.uid}.png")

        if the_end and self.last_idx + 2 == cur_idx:
            self._last_call()
