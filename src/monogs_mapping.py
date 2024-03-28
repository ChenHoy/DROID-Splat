
from munch import munchify
import torch
import time
import numpy as np
import torch.multiprocessing as mp
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

from gui import gui_utils, slam_gui
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2, focal2fov


from utils.multiprocessing_utils import clone_obj
from utils.multiprocessing_utils import FakeQueue
from utils.camera_utils import Camera
from utils.pose_utils import update_pose

import droid_backends

import lpips

class GaussianMapper(object):
    def __init__(self, config, args, slam, mapping_queue = None,use_gui=False):
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
        
        self.use_spherical_harmonics = False
        self.model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(self.model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(self.opt_params)

        bg_color = [1, 1, 1]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        self.mapping_queue = mapping_queue
        self.use_gui = use_gui
        self.q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        self.q_vis2main = mp.Queue() if self.use_gui else FakeQueue()
        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=self.q_main2vis,
            q_vis2main=self.q_vis2main,
        )
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            print("GUI process started")
            #time.sleep(5)

        self.cameras = [] ## list of views for rendering
        self.last_idx = -1
        self.initialized = False

        self.mapping_iters = 10
        self.refinement_iters = 100

        self.loss_list = []

        self.optimize_poses = True
        self.update_frames = False
        self.mapping_queue = mapping_queue
        self.filter_depth = True
        self.save_renders = False
        self.render_path = "/home/andrei/results/monogs/renders"
        self.mesh_path = "/home/andrei/results/monogs/meshes"



    def camera_from_gt(self):
        idx, image, depth, intrinsic, gt_pose = self.mapping_queue.get()
        image, depth, intrinsic, gt_pose = image.squeeze().to(self.device), depth.to(self.device), intrinsic.to(self.device), gt_pose.to(self.device)
        return self.camera_from_frame(idx, image, depth, intrinsic, gt_pose)

    def camera_from_video(self, idx: int):
        color, depth, c2w, _, _ = self.video.get_mapping_item(idx, self.device)
        color = color.permute(2, 0, 1)
        intrinsics = self.video.intrinsics[0]*self.video.scale_factor

        if self.filter_depth:
            mask = self.depth_filter(idx)
            depth = (depth*mask)

        return self.camera_from_frame(idx, color, depth, intrinsics, c2w)
    

    def camera_from_frame(self, idx: int, image: torch.Tensor, depth: torch.Tensor, intrinsic: torch.Tensor, gt_pose: torch.Tensor):
        fx, fy, cx, cy = intrinsic
        height, width = image.shape[1:]
        gt_pose = torch.linalg.inv(gt_pose)  # They invert the poses in the dataloader 
        znear = 0.01
        zfar = 100.0
        fovx = focal2fov(fx, width)
        fovy = focal2fov(fy, height)

        if not self.initialized:
            self.projection_matrix = getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, width, height).transpose(0, 1).to(self.device)

        return Camera(idx, image, depth, gt_pose, self.projection_matrix, fx, fy, cx, cy, fovx, fovy, height, width, device=self.device)

    def pose_optimizer(self, frames: list):
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

    def depth_filter(self, idx: int):
        """
        Gets the video and the time idex and returns the mask.
        """
        #TODO why doesnt it work only with one index?
        with self.video.get_lock():
            (dirty_index,) = torch.where(self.video.dirty.clone())
            dirty_index = dirty_index

        device = self.video.device
        poses = torch.index_select(self.video.poses, 0, dirty_index)
        disps = torch.index_select(self.video.disps_up, 0, dirty_index)
        thresh = 0.1 * torch.ones_like(disps.mean(dim=[1, 2]))
        intrinsics = self.video.intrinsics[0]*self.video.scale_factor
        count = droid_backends.depth_filter(poses, disps, intrinsics, dirty_index, thresh)

        mask = ((count >= 1) & (disps > 0.5 * disps.mean(dim=[1, 2], keepdim=True)))[idx]

        #print(f"Valid points:{mask.sum()}/{mask.numel()}")

        return mask

    def frame_updater(self, frames):
        with torch.no_grad():
            for cam in frames:
                _, depth, c2w, _, _ = self.video.get_mapping_item(cam.uid, self.video.device)
                if self.filter_depth:
                    mask = self.depth_filter(cam.uid)
                    depth = (depth*mask)

                w2c = torch.inverse(c2w)
                R = w2c[:3, :3].unsqueeze(0).detach()
                T = w2c[:3, 3].detach()
                cam.depth = depth.detach()
                cam.update_RT(R, T)

    def plot_centers(self):
        means = self.gaussians.get_xyz.detach().cpu().numpy()
        rgb = self.gaussians.get_features[:, 0, :].detach().cpu().numpy()
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(means)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.visualization.draw_geometries([pcd])

    def mapping_step(self, iter: int, frames: list, pruning_params: dict, densify: bool = True, optimize_poses: bool = False):
        """
        Takes the list of selected keyframes to optimize and 
        """
        if self.optimize_poses:
            pose_optimizer = self.pose_optimizer(self.cameras)
        loss = 0
        for view in frames:

            render_pkg = render(
                        view, self.gaussians, self.pipeline_params, self.background
            )

            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )

            loss += self.mapping_loss(image, depth, view)


        scaling = self.gaussians.get_scaling
        isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
        loss += len(frames) * isotropic_loss.mean()
        loss.backward()

        with torch.no_grad():
            self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
            )

            if densify:
                self.gaussians.add_densification_stats(
                        viewspace_point_tensor, visibility_filter
                )

            if self.last_idx > 0 and (iter+1) % self.training_params.prune_every == 0:
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


    def mapping_loss(self, image: torch.Tensor, depth: torch.Tensor, cam: Camera):
        '''
        Loss between gt image and rendered image
        '''
        alpha = self.config["Training"]["alpha"] if "alpha" in self.config["Training"] else 0.95
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"] if "rgb_boundary_threshold" in self.config["Training"] else 0.01

        gt_image = cam.original_image
        gt_depth = cam.depth

        rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
        depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

        image = (torch.exp(cam.exposure_a)) * image + cam.exposure_b
        l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
        l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

        return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()
    
    def select_keyframes(self):
        # Select last 5 frames and other 5 random frames
        if len(self.cameras) <= 10:
            keyframes = self.cameras
            keyframes_idx = np.arange(len(self.cameras))
        else:
            keyframes_idx = np.random.choice(len(self.cameras)-5, 5, replace=False)
            keyframes = self.cameras[-5:] + [self.cameras[i] for i in keyframes_idx]
        return keyframes,keyframes_idx
    

    def __call__(self, the_end = False):

        cur_idx = int(self.video.filtered_id.item()) 
        
        if self.last_idx+2 < cur_idx:
            self.last_idx += 1

            # Add camera of the last frame
            cam = self.camera_from_video(self.last_idx)
            cam.update_RT(cam.R_gt, cam.T_gt) # Assuming we found the best pose
            self.cameras.append(cam)


            # Add gaussians based on the new view
            if not self.initialized:
                self.gaussians.extend_from_pcd_seq(
                    cam, cam.uid, init = True
                )
                self.initialized = True

            else:
                n_g = self.gaussians.get_xyz.shape[0]
                self.gaussians.extend_from_pcd_seq(
                    cam, cam.uid, init = False
                )
                print(f"Added {self.gaussians.get_xyz.shape[0] - n_g} gaussians for the new view" )


            # Optimze gaussians
            for iter in range(self.mapping_iters):
                frames, _ = self.select_keyframes()

                if self.update_frames and not the_end: # Tracking finished, no need to update frames
                    self.frame_updater(frames)

                loss = self.mapping_step(
                    iter, frames, self.training_params, 
                    densify = True, optimize_poses = self.optimize_poses
                )
                self.loss_list.append(loss/len(frames))

            # Update visualization
            if self.use_gui:
                self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            gaussians=clone_obj(self.gaussians),
                            current_frame=cam,
                            #keyframes=self.cameras,
                            keyframe = cam,
                            kf_window=None,
                            # Stream visualization
                            gtcolor=cam.original_image,        
                            gtdepth=cam.depth.detach().cpu().numpy(),
                        )
                )

            print(f"Frame: {cam.uid}. Gaussians: {self.gaussians.get_xyz.shape[0]}. Video at {cur_idx}")

            # Save renders
            if self.save_renders and cam.uid % 5 == 0:
                render_pkg = render(
                        cam, self.gaussians, self.pipeline_params, self.background 
                )
                im = np.uint8(255*render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0))
                cv2.imwrite(f"{self.render_path}/{cam.uid}.png", im)
                print(f"Render saved at {self.render_path}/{cam.uid}.png")

        #if the_end and self.mapping_queue.empty():
        if the_end and self.last_idx+2 == cur_idx:
            print("Mapping refinement starting")


            for iter in range(self.refinement_iters):                
                loss = self.mapping_step(
                    iter, self.cameras, self.training_params.refinement, 
                    densify = False, optimize_poses = self.optimize_poses
                )
                self.loss_list.append(loss/len(self.cameras))

                if self.use_gui:
                    self.q_main2vis.put(
                            gui_utils.GaussianPacket(
                                gaussians=clone_obj(self.gaussians),
                                keyframes=self.cameras,
                            )
                    )
        
            
            self.gaussians.save_ply(f"{self.mesh_path}/final.ply")
            print("Mesh saved")

            # fig, ax = plt.subplots()
            # ax.set_yscale("log")
            # ax.set_title(f"Mode: {self.mode}. Optimize poses: {self.optimize_poses}. Gaussians: {self.gaussians.get_xyz.shape[0]}")
            # ax.plot(self.loss_list[-self.refinement_iters:])
            # plt.savefig(f"{self.render_path}/loss_{self.mode}.png")
            # plt.show()

            # fig, ax = plt.subplots()
            # ax.set_yscale("log")
            # ax.plot(self.loss_list)
            # plt.show()
            return True
        """
            TODO:

            - Add more info to GUI
            - If waiting for new frames, optimize in current views
            - Optimize all new frames at the same time
            - Clean config file
            - Pruning criteria (splatam vs monoGS)
            - Call GUI thread on slam and not on mapping 

            

            Notes:

            - the gaussian model is in gaussian_splatting/scene/gaussian_model.py. The whole gaussian splatting is imported from monogs
            - the monogs mapping has a gui flag
            - the plot_centers doesnt seem to update
            """