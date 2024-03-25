
from munch import munchify
import torch
import time

import torch.multiprocessing as mp
import open3d as o3d

from gui import gui_utils, slam_gui
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2, focal2fov


from utils.multiprocessing_utils import clone_obj
from utils.multiprocessing_utils import FakeQueue
from utils.camera_utils import Camera



class GaussianMapper(object):
    def __init__(self, config, args, slam, mapping_queue = None):
        self.config = config
        self.args = args
        self.slam = slam
        self.device = args.device
        self.model_params = munchify(config["model_params"])
        self.opt_params = munchify(config["opt_params"])
        self.pipeline_params = munchify(config["pipeline_params"])
        
        self.use_spherical_harmonics = False
        self.model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(self.model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(self.opt_params)

        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.cameras = []

        self.mapping_queue = mapping_queue
        self.use_gui = False ## change this for gui
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
            time.sleep(5)

        self.initialized = False


    def camera_from_gt(self):
        idx, image, depth, intrinsic, gt_pose = self.mapping_queue.get()
        image, depth, intrinsic, gt_pose = image.squeeze().to(self.device), depth.to(self.device), intrinsic.to(self.device), gt_pose.to(self.device)
        fx, fy, cx, cy = intrinsic
        height, width = image.shape[1:]
        gt_pose = torch.linalg.inv(gt_pose)  # They invert the poses in the dataloader 
        znear = 0.01
        zfar = 100.0
        fovx = focal2fov(fx, width)
        fovy = focal2fov(fy, height)
        if not self.initialized:
            self.projection_matrix = getProjectionMatrix2(znear,zfar, cx, cy, fx, fy, width, height).transpose(0, 1).to(self.device)

        return Camera(idx, image, depth, gt_pose, self.projection_matrix, fx, fy, cx, cy, fovx, fovy, height, width, device=self.device)

    def plot_centers(self):
        '''
        Display just the centers of the gaussians
        '''
        means = self.gaussians.get_xyz.detach().cpu().numpy()
        rgb = self.gaussians.get_features[:, 0, :].detach().cpu().numpy()
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(means)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.visualization.draw_geometries([pcd])

    def __call__(self, the_end = False):

        if not self.mapping_queue.empty():
            cam = self.camera_from_gt()
            self.cameras.append(cam)
            if not self.initialized:
                self.gaussians.extend_from_pcd_seq(cam, cam.uid, init = True)
                self.initialized = True
            else:
                self.gaussians.extend_from_pcd_seq(cam, cam.uid, init = False)

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
            
            if cam.uid % 100 == 0 and not self.use_gui:
                # Simple o3d plot of the centers
                ## NOTE: this runs only one time
                print("Plotted some gaussians")
                self.plot_centers()
            

            """
            TODO:
            - Solve problem with poses

            - Add Gaussian optimization at each keyframe
            - Decide pruning criteria (splatam vs monoGS)
            

            

            Notes:

            - the gaussian model is in gaussian_splatting/scene/gaussian_model.py. The whole gaussian splatting is imported from monogs
            - the monogs mapping has a gui flag
            - the plot_centers doesnt seem to update
            """