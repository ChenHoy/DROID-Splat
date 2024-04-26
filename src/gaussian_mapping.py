import os
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

NOTE this could a be standalone SLAM system itself, but we use it as finetuning here to be independent of rendering alone.
"""


def plot_3d(rgb: torch.Tensor, depth: torch.Tensor):
    """Use Open3d to plot the 3D point cloud from the monocular depth and input image."""

    def get_calib_heuristic(ht: int, wd: int) -> np.ndarray:
        """On in-the-wild data we dont have any calibration file.
        Since we optimize this calibration as well, we can start with an initial guess
        using the heuristic from DeepV2D and other papers"""
        cx, cy = wd // 2, ht // 2
        fx, fy = wd * 1.2, wd * 1.2
        return fx, fy, cx, cy

    rgb = np.asarray(rgb.cpu())
    depth = np.asarray(depth.cpu())
    invalid = (depth < 0.001).flatten()
    # Get 3D point cloud from depth map
    depth = depth.squeeze()
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.flatten(), y.flatten()
    depth = depth.flatten()

    # Convert to 3D points
    fx, fy, cx, cy = get_calib_heuristic(h, w)
    x3 = (x - cx) * depth / fx
    y3 = (y - cy) * depth / fy
    z3 = depth

    # Convert to Open3D format
    xyz = np.stack([x3, y3, z3], axis=1)
    rgb = np.stack([rgb[0, :, :].flatten(), rgb[1, :, :].flatten(), rgb[2, :, :].flatten()], axis=1)
    depth = depth[~invalid]
    xyz = xyz[~invalid]
    rgb = rgb[~invalid]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Plot the point cloud
    o3d.visualization.draw_geometries([pcd])


class GaussianMapper(object):
    """
    SLAM from Rendering with 3D Gaussian Splatting.
    """

    def __init__(self, cfg, slam):
        self.cfg = cfg
        self.slam = slam
        self.video = slam.video
        self.device = cfg.slam.device
        self.mode = cfg.slam.mode
        self.output = slam.output
        self.model_params = cfg.mapping.model_params
        self.opt_params = cfg.mapping.opt_params
        self.pipeline_params = cfg.mapping.pipeline_params
        self.pruning_params = cfg.mapping.pruning
        self.mapping_params = cfg.mapping
        self.loss_params = cfg.mapping.loss
        self.delay = 2 # Delay between tracking and mapping
        
        self.use_spherical_harmonics = False
        self.model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(self.model_params.sh_degree, config=self.cfg.data)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(self.opt_params)

        bg_color = [1, 1, 1]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if self.mapping_params.use_gui and not self.cfg.slam.evaluate:

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
        self.new_cameras = []
        self.loss_list = []
        self.last_idx = 0
        self.initialized = False
        self.projection_matrix = None

        self.n_last_frames = 10
        self.n_rand_frames = 5

        self.show_filtered = False

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

        # if self.setup.filter_depth:
        #     mask = self.depth_filter(idx)
        #     msg = "Filtered {}% of the points".format(100 * (1 - mask.sum() / mask.numel()))
        #     self.info(msg)

        #     if self.show_filtered:
        #         filt_col = color * mask
        #         filt_col[0, ~mask] = 255
        #         plot_3d(filt_col, depth)

        #     depth = depth * mask

        return self.camera_from_frame(idx, color, depth, intrinsics, c2w)

    def get_new_cameras(self):
        """
        Get all new cameras from the video.
        """
        for idx in range(self.last_idx, self.cur_idx - self.delay):
            color, depth, c2w, _, _ = self.video.get_mapping_item(idx, self.device)
            color = color.permute(2, 0, 1)
            intrinsics = self.video.intrinsics[0] * self.video.scale_factor
            cam = self.camera_from_frame(idx, color, depth, intrinsics, c2w)
            cam.update_RT(cam.R_gt, cam.T_gt) # Assuming we found the best pose in tracking
            self.new_cameras.append(cam)
        

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
            (dirty_index,) = torch.where(self.video.mapping_dirty.clone())
            dirty_index = dirty_index

        device = self.video.device
        poses = torch.index_select(self.video.poses, 0, dirty_index)
        disps = torch.index_select(self.video.disps_up, 0, dirty_index)
        thresh = 0.1 * torch.ones_like(disps.mean(dim=[1, 2]))
        intrinsics = self.video.intrinsics[0] * self.video.scale_factor
        count = droid_backends.depth_filter(poses, disps, intrinsics, dirty_index, thresh)

        mask = (count >= 1) & (disps > 0.05 * disps.mean(dim=[1, 2], keepdim=True))


        self.info(f"Valid points: {100* mask.sum()/mask.numel()} %")
        return mask

    def frame_updater(self):
        """
        Gets the list of frames and updates the depth and pose based on the video.
        All frames are used to compute the filter, only dirty frames are updated
        """
        all_cameras = self.cameras + self.new_cameras
        all_idxs = torch.tensor([cam.uid for cam in all_cameras]).long().to(self.device)

        with self.video.get_lock():
            (dirty_index,) = torch.where(self.video.mapping_dirty.clone())
            dirty_index = dirty_index[dirty_index < self.cur_idx - self.delay]

            poses = torch.index_select(self.video.poses.detach(), dim=0, index=all_idxs)
            disps = torch.index_select(self.video.disps_up.detach(), dim=0, index=all_idxs)
            intrinsics = self.video.intrinsics[0] * self.video.scale_factor

        self.info(f"Updating frames {dirty_index}")


        depths = 1 / (disps + 1e-7)
        thresh = 0.1 * torch.ones_like(disps.mean(dim=[1, 2]))
        count = droid_backends.depth_filter(poses, disps, intrinsics, all_idxs, thresh)
        mask = (count >= 1) & (disps > 0.05 * disps.mean(dim=[1, 2], keepdim=True))

        for idx in dirty_index:
            # self.info(f"Frame {idx}: filtered {100*(1 - mask[idx].sum() / mask[idx].numel()):.2f}% of the points")
            _, _, c2w, _, _ = self.video.get_mapping_item(idx, self.device)
            cam = all_cameras[idx]
            w2c = torch.inverse(c2w)
            R = w2c[:3, :3].unsqueeze(0).detach()
            T = w2c[:3, 3].detach()
            cam.depth = depths[idx].detach()
            cam.update_RT(R, T)
        
        self.video.mapping_dirty[dirty_index] = False

    def plot_centers(self) -> None:
        means = self.gaussians.get_xyz.detach().cpu().numpy()
        rgb = self.gaussians.get_features[:, 0, :].detach().cpu().numpy()
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(means)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.visualization.draw_geometries([pcd])

    def save_render(self, cam: Camera, render_path: str) -> None:
        render_pkg = render(cam, self.gaussians, self.pipeline_params, self.background)
        rgb = np.uint8(255 * render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(render_path, bgr)

    def mapping_step(
        self, iter: int, frames: list, pruning_params: dict, densify: bool = True, optimize_poses: bool = False
    ) -> float:
        """
        Takes the list of selected keyframes to optimize and performs one step of the mapping optimization.
        """
        if self.mapping_params.optimize_poses:
            pose_optimizer = self.pose_optimizer(self.cameras)
        loss = 0
        self.occ_aware_visibility = {}
        for view in frames:

            render_pkg = render(view, self.gaussians, self.pipeline_params, self.background)
            visibility_filter, viewspace_point_tensor = render_pkg["visibility_filter"], render_pkg["viewspace_points"]
            image, radii, depth = render_pkg["render"], render_pkg["radii"], render_pkg["depth"]
            opacity, n_touched = render_pkg["opacity"], render_pkg["n_touched"]

            loss += self.mapping_loss(image, depth, view)
            if self.last_idx - view.uid < self.n_last_frames: # TODO Only take into account last frames and not random ones
                self.occ_aware_visibility[view.uid] = (n_touched > 0).long()

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

            if self.last_idx > self.n_last_frames and (iter + 1) % self.pruning_params.prune_every == 0:
                #self.prune_gaussians() # Covisibility based pruning for recently added gaussians 
                self.gaussians.densify_and_prune( # General pruning based on opacity and size + densification
                pruning_params.densify_grad_threshold,
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

    def mapping_loss(self, image: torch.Tensor, depth: torch.Tensor, cam: Camera):
        """Compute a weighted l1 loss between i) the rendered image and the ground truth image and ii) the rendered depth and the ground truth depth."""

        alpha = self.loss_params.alpha
        rgb_boundary_threshold = self.loss_params.rgb_boundary_threshold

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

        new_frames = self.cameras[-self.n_last_frames :]
        new_idx = [view.uid for view in new_frames]
        sorted_frames = sorted(new_idx, reverse=True)

        self.gaussians.n_obs.fill_(0)
        for _, visibility in self.occ_aware_visibility.items():
            self.gaussians.n_obs += visibility.cpu()

        mask = self.gaussians.unique_kfIDs >= sorted_frames[self.pruning_params.prune_last - 1] # Gaussians added on the last prune_last frames

        to_prune = torch.logical_and(
            self.gaussians.n_obs <= self.pruning_params.visibility_th, mask
        )
        self.gaussians.prune_points(to_prune.cuda())
        for idx in new_idx:
            self.occ_aware_visibility[idx] = self.occ_aware_visibility[idx][~to_prune]
        self.info(f"Covisibility based pruning removed {to_prune.sum()} gaussians")

    # TODO chen: make the 5 configurable, because this should be treated as a hyperparameter
    def select_keyframes(self):
        # Select n_last_frames and other n_rand_frames
        if len(self.cameras) <= self.n_last_frames + self.n_rand_frames:
            keyframes = self.cameras
            keyframes_idx = np.arange(len(self.cameras))
        else:
            keyframes_idx = np.random.choice(len(self.cameras) - self.n_last_frames, self.n_rand_frames, replace=False)
            keyframes = self.cameras[-self.n_last_frames:] + [self.cameras[i] for i in keyframes_idx]
        return keyframes, keyframes_idx

    def __call__(
        self,
        mapping_queue: mp.Queue,
        received_item: mp.Event,
        the_end=False,
    ):
        self.cur_idx = int(self.video.filtered_id.item())

        if the_end and self.last_idx + self.delay == self.cur_idx:
            print("\nMapping refinement starting")

            for iter in range(self.mapping_params.refinement_iters):
                loss = self.mapping_step(
                    iter,
                    self.cameras,
                    self.pruning_params.refinement,
                    densify=False,
                    optimize_poses=self.mapping_params.optimize_poses,
                )
                self.loss_list.append(loss / len(self.cameras))

                if self.mapping_params.use_gui:
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            gaussians=clone_obj(self.gaussians),
                            keyframes=self.cameras,
                        )
                    )
            print("Mapping refinement finished")

            self.gaussians.save_ply(f"{self.output}/mesh/final_{self.mode}.ply")
            print("Mesh saved")

            if self.mapping_params.save_renders:
                for cam in self.cameras:
                    self.save_render(cam, f"{self.output}/renders/final/{cam.uid}.png")

            ## export the cameras and gaussians to the terminate process
            print("Sending the final state to the terminate process")

            mapping_queue.put(
                gui_utils.EvaluatePacket(
                    pipeline_params=clone_obj(self.pipeline_params),
                    cameras=self.cameras[:],
                    gaussians=clone_obj(self.gaussians),
                    background=clone_obj(self.background),
                )
            )

            received_item.wait()  # Wait until the Packet got delivered
            print("Sent the final state to the terminate process successfully")

            if self.cfg.slam.evaluate:

                fig, ax = plt.subplots()
                ax.set_yscale("log")
                ax.set_title(f"Mode: {self.mode}. Optimize poses: {self.mapping_params.optimize_poses}. Gaussians: {self.gaussians.get_xyz.shape[0]}")
                ax.plot(self.loss_list[-self.mapping_params.refinement_iters:])
                plt.savefig(f"{self.output}/loss_{self.mode}.png")
                plt.clf()

            return True

        elif self.last_idx + self.delay < self.cur_idx and self.cur_idx > self.mapping_params.warmup:
            #self.last_idx += 1
            
            # self.info(f"\nStarting frame: {self.last_idx}. Gaussians: {self.gaussians.get_xyz.shape[0]}. Video at {self.cur_idx}")

            # Add all new cameras
            self.get_new_cameras()
            self.info(f"Added {len(self.new_cameras)} new cameras: {[cam.uid for cam in self.new_cameras]}")

            self.frame_updater() # Filter dirty and new cameras



            for cam in self.new_cameras:
                if not self.initialized:
                    self.initialized = True
                    self.gaussians.extend_from_pcd_seq(cam, cam.uid, init=True)
                    self.info(f"Initialized with {self.gaussians.get_xyz.shape[0]} gaussians")
                else:
                    n_g = self.gaussians.get_xyz.shape[0]
                    self.gaussians.extend_from_pcd_seq(cam, cam.uid, init=False)
                    self.info(f"Added {self.gaussians.get_xyz.shape[0] - n_g} gaussians based on view {cam.uid}")



            # Optimze gaussians
            for iter in range(self.mapping_params.mapping_iters):
                frames = self.select_keyframes()[0] + self.new_cameras

                loss = self.mapping_step(
                    iter, frames, self.pruning_params.mapping, densify=True, optimize_poses=False
                )
                self.loss_list.append(loss / len(frames))
                
            # Update visualization
            if self.mapping_params.use_gui:
                self.q_main2vis.put(
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

            msg = "Frame: {}. Gaussians: {}. Video at {}".format(cam.uid, self.gaussians.get_xyz.shape[0], self.cur_idx)
            self.info(msg)

            # Save renders
            if self.mapping_params.save_renders and cam.uid % 5 == 0:
                self.save_render(cam, f"{self.output}/renders/mapping/{cam.uid}.png")


            self.cameras += self.new_cameras
            self.new_cameras = []
            self.last_idx = self.cameras[-1].uid + 1
