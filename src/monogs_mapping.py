
from munch import munchify
import torch
import time
import numpy as np
import torch.multiprocessing as mp
import open3d as o3d
import cv2

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
    def __init__(self, config, args, slam, mapping_queue = None):
        self.config = config
        self.args = args
        self.slam = slam
        self.video = slam.video
        self.device = args.device
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
        self.use_gui = True
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
        self.optimize_poses = True
        self.mapping_queue = mapping_queue
        self.filter_depth = True
        self.save_renders = True
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

        # if not self.mapping_queue.empty():
        #     cam = self.camera_from_gt()
        
        if self.last_idx+2 < cur_idx:
            self.last_idx += 1

            print(self.last_idx, cur_idx)
            cam = self.camera_from_video(self.last_idx)

            cam.update_RT(cam.R_gt, cam.T_gt) # Assuming we found the best pose
            self.cameras.append(cam)

            # Add gaussian based on the new view
            
            if not self.initialized:
                self.gaussians.extend_from_pcd_seq(cam, cam.uid, init = True)
                self.initialized = True

            else:
                self.gaussians.extend_from_pcd_seq(cam, cam.uid, init = False)


            # Optimze gaussian
            self.mapping_iters = 20
            for iter in range(self.mapping_iters):
                loss = 0
                frames, _ = self.select_keyframes()

                if self.optimize_poses:
                    pose_optimizer = self.pose_optimizer(frames)

                for view in frames:

                    render_pkg = render(
                        view, self.gaussians, self.pipeline_params, self.background
                    )
                    ## this is the rendered images and depth
                    # print(render_pkg['render'].min(), render_pkg['render'].max())
                    loss += self.mapping_loss(render_pkg["render"], render_pkg["depth"], view)

                    

                scaling = self.gaussians.get_scaling
                isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
                loss += 10 * isotropic_loss.mean()
                loss.backward()

                with torch.no_grad():
                    if self.last_idx > 0 and (iter+1) % self.training_params.prune_every == 0:
                        self.gaussians.densify_and_prune(
                                                self.opt_params.densify_grad_threshold,
                                                self.training_params.gaussian_th,
                                                self.training_params.gaussian_extent,
                                                self.training_params.size_threshold,
                                )
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad()
                    self.gaussians.update_learning_rate(self.last_idx)

                    if self.optimize_poses:
                        pose_optimizer.step()
                        pose_optimizer.zero_grad()
                        for view in frames:
                            update_pose(view)


            # Update visualization
            if self.use_gui:
                self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            gaussians=clone_obj(self.gaussians),
                            current_frame=cam,
                            #keyframes=self.cameras[:],
                            keyframe = cam,
                            kf_window=None,
                            # Stream visualization
                            gtcolor=cam.original_image,        
                            gtdepth=cam.depth.detach().cpu().numpy(),
                        )
                )

            print(f"Frame: {cam.uid}. Gaussians: {self.gaussians.get_xyz.shape[0]}")

            if self.save_renders and cam.uid % 5 == 0:
                im = np.uint8(255*render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0))
                cv2.imwrite(f"{self.render_path}/{cam.uid}.png", im)
                print(f"Render saved at {self.render_path}/{cam.uid}.png")

        #if the_end and self.mapping_queue.empty():
        if the_end and self.last_idx+2 == cur_idx:
            print("Mapping refinement starting")


            for iter in range(40):
                loss = 0
                for view in np.random.permutation(self.cameras):
                    render_pkg = render(
                        view, self.gaussians, self.pipeline_params, self.background
                    )
                    loss += self.mapping_loss(render_pkg["render"], render_pkg["depth"], view)

                scaling = self.gaussians.get_scaling
                isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
                loss += 20 * isotropic_loss.mean()
                loss.backward()

                with torch.no_grad():
                    if self.last_idx > 0 and (iter) % self.training_params.prune_every == 0:
                        self.gaussians.densify_and_prune(
                                                self.opt_params.densify_grad_threshold,
                                                0.7,
                                                1,
                                                20,
                                )
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad()
                    self.gaussians.update_learning_rate(self.last_idx+iter+1)
            
            
                if self.use_gui:
                    self.q_main2vis.put(
                            gui_utils.GaussianPacket(
                                gaussians=clone_obj(self.gaussians),
                            )
                    )
            
            


            self.gaussians.save_ply(f"{self.mesh_path}/final.ply")
            print("Mesh saved")

            return True
        """
            TODO:

            - If waiting for new frames, optimize in current views
            - Make it work with depth_video
            - Optimize all new frames at the same time
            - Clean config file
            - Keyframe selection
            - Pruning criteria (splatam vs monoGS)
            - Optimize poses as well

            

            Notes:

            - the gaussian model is in gaussian_splatting/scene/gaussian_model.py. The whole gaussian splatting is imported from monogs
            - the monogs mapping has a gui flag
            - the plot_centers doesnt seem to update
            """