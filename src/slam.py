import os
import os.path as osp
from typing import List, Optional
from tqdm import tqdm
from time import gmtime, strftime, sleep
from collections import OrderedDict
from omegaconf import DictConfig
from termcolor import colored
import ipdb

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from lietorch import SE3

from .droid_net import DroidNet
from .frontend import Frontend
from .backend import Backend
from .depth_video import DepthVideo
from .motion_filter import MotionFilter
from .multiview_filter import MultiviewFilter
from .visualization import droid_visualization, depth2rgb
from .trajectory_filler import PoseTrajectoryFiller
from .gaussian_mapping import GaussianMapper
from .gaussian_splatting.eval_utils import eval_ate, eval_rendering, save_gaussians
from .gaussian_splatting.gui import gui_utils, slam_gui

from multiprocessing import Manager


class Tracker(nn.Module):
    """
    Wrapper class for SLAM frontend tracking.
    """

    def __init__(self, cfg, slam):
        super(Tracker, self).__init__()
        self.cfg = cfg
        self.device = cfg.slam.device
        self.net = slam.net
        self.video = slam.video

        # filter incoming frames so that there is enough motion
        self.frontend_window = cfg.tracking.frontend.window
        filter_thresh = cfg.tracking.motion_filter.thresh
        self.motion_filter = MotionFilter(self.net, self.video, thresh=filter_thresh, device=self.device)

        # frontend process
        self.frontend = Frontend(self.net, self.video, self.cfg)

    @torch.no_grad()
    def forward(self, timestamp, image, depth, intrinsic, gt_pose=None):
        ### check there is enough motion
        self.motion_filter.track(timestamp, image, depth, intrinsic, gt_pose=gt_pose)

        # local bundle adjustment
        self.frontend()


class BundleAdjustment(nn.Module):
    """
    Wrapper class for Backend optimization
    """

    def __init__(self, cfg, slam):
        super(BundleAdjustment, self).__init__()
        self.cfg = cfg
        self.device = cfg.slam.device
        self.net = slam.net
        self.video = slam.video

        self.frontend_window = cfg["tracking"]["frontend"]["window"]
        self.last_t = -1
        self.ba_counter = -1

        # backend process
        self.backend = Backend(self.net, self.video, self.cfg)

    def info(self, t_start, t_end, cur_t):
        now = "[Backend] {} - Full BA".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        msg = "\n {} : [{}, {}]; Current Keyframe is {}, last is {}. \n".format(
            now, t_start, t_end, cur_t, self.last_t
        )
        print(colored(msg, "blue"))

    def forward(self):
        cur_t = self.video.counter.value
        t = cur_t

        # Only optimize outside of Frontend
        if cur_t > self.frontend_window:
            t_start = 0
            self.backend.dense_ba(t_start=t_start, t_end=t, steps=6, motion_only=False)

            self.info(t_start, t, cur_t)
            self.last_t = cur_t


class SLAM:
    def __init__(self, cfg):
        super(SLAM, self).__init__()
        self.cfg = cfg
        self.device = cfg.slam.device
        self.verbose = cfg.slam.verbose
        self.mode = cfg.slam.mode
        self.create_out_dirs(cfg)

        self.update_cam(cfg)
        self.load_bound(cfg)

        self.net = DroidNet()

        self.load_pretrained(cfg.tracking.pretrained)
        self.net.to(self.device).eval()
        self.net.share_memory()

        # Manage life time of individual processes
        self.num_running_thread = torch.zeros((1)).int()
        self.num_running_thread.share_memory_()
        self.all_trigered = torch.zeros((1)).int()
        self.all_trigered.share_memory_()
        self.all_finished = torch.zeros((1)).int()
        self.all_finished.share_memory_()
        self.tracking_finished = torch.zeros((1)).int()
        self.tracking_finished.share_memory_()
        self.multiview_filtering_finished = torch.zeros((1)).int()
        self.multiview_filtering_finished.share_memory_()
        self.gaussian_mapping_finished = torch.zeros((1)).int()
        self.gaussian_mapping_finished.share_memory_()
        self.backend_finished = torch.zeros((1)).int()
        self.backend_finished.share_memory_()
        self.visualizing_finished = torch.zeros((1)).int()
        self.visualizing_finished.share_memory_()
        self.sleep_time = 1.0  # Time for giving delays
        # Stream the images into the main thread
        self.input_pipe = mp.Queue()

        # store images, depth, poses, intrinsics (shared between process)
        # NOTE: we can use this for getting both gt and rendered images
        self.video = DepthVideo(cfg)

        self.tracker = Tracker(cfg, self)

        self.multiview_filter = MultiviewFilter(cfg, self)

        self.dense_ba = BundleAdjustment(cfg, self)

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(net=self.net, video=self.video, device=self.device)

        self.do_evaluate = cfg.slam.evaluate
        self.dataset = None

        if cfg.slam.run_mapping_gui:
            self.q_main2vis = mp.Queue()
            self.q_vis2main = mp.Queue()
            self.gaussian_mapper = GaussianMapper(cfg, self, gui_qs = (self.q_main2vis, self.q_vis2main))
        else:
            self.gaussian_mapper = GaussianMapper(cfg, self)

        if cfg.slam.run_mapping_gui:
            self.params_gui = gui_utils.ParamsGUI(
                pipe=cfg.mapping.pipeline_params,
                background=self.gaussian_mapper.background,
                gaussians=self.gaussian_mapper.gaussians,
                q_main2vis=self.q_main2vis,
                q_vis2main=self.q_vis2main,
            )

    def info(self, msg) -> None:
        print(colored("[Main]: " + msg, "green"))

    def create_out_dirs(self, cfg: DictConfig) -> None:
        self.output = cfg.slam.output_folder

        os.makedirs(self.output, exist_ok=True)
        os.makedirs(f"{self.output}/logs/", exist_ok=True)
        os.makedirs(f"{self.output}/renders/mapping/", exist_ok=True)
        os.makedirs(f"{self.output}/renders/final", exist_ok=True)
        os.makedirs(f"{self.output}/mesh", exist_ok=True)

    def update_cam(self, cfg):
        """
        Update the camera intrinsics according to the pre-processing config,
        such as resize or edge crop
        """
        # resize the input images to crop_size(variable name used in lietorch)
        H, W = float(cfg.data.cam.H), float(cfg.data.cam.W)
        fx, fy = cfg.data.cam.fx, cfg.data.cam.fy
        cx, cy = cfg.data.cam.cx, cfg.data.cam.cy

        h_edge, w_edge = cfg.data.cam.H_edge, cfg.data.cam.W_edge
        H_out, W_out = cfg.data.cam.H_out, cfg.data.cam.W_out

        self.fx = fx * (W_out + w_edge * 2) / W
        self.fy = fy * (H_out + h_edge * 2) / H
        self.cx = cx * (W_out + w_edge * 2) / W
        self.cy = cy * (H_out + h_edge * 2) / H
        self.H, self.W = H_out, W_out

        self.cx = self.cx - w_edge
        self.cy = self.cy - h_edge

    def load_bound(self, cfg: DictConfig) -> None:
        """
        Pass the scene bound parameters to different decoders and self.

        ---
        Args:
            cfg [dict], parsed config dict
        """
        # scale the bound if there is a global scaling factor
        self.bound = torch.from_numpy(np.array(cfg.data.bound)).float()

    def load_pretrained(self, pretrained: str) -> None:
        self.info(f"Load pretrained checkpoint from {pretrained}!")

        # TODO why do we have to use the [:2] here?!
        state_dict = OrderedDict([(k.replace("module.", ""), v) for (k, v) in torch.load(pretrained).items()])
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)

    def tracking(self, rank, stream, input_queue: mp.Queue) -> None:
        """Main driver of framework by looping over the input stream"""

        self.info("Tracking thread started!")
        self.all_trigered += 1
        # Wait up for other threads to start
        while self.all_trigered < self.num_running_thread:
            pass

        for frame in tqdm(stream):
            timestamp, image, depth, intrinsic, gt_pose = frame
            if self.mode not in ["rgbd", "prgbd"]:
                depth = None

            if self.cfg.slam.show_stream:
                # Transmit the incoming stream to another visualization thread
                input_queue.put(image)
                input_queue.put(depth)

            self.tracker(timestamp, image, depth, intrinsic, gt_pose)


        self.tracking_finished += 1
        self.all_finished += 1
        self.info("Tracking done!")


    def backend(self, rank, run=True):
        self.info("Full Bundle Adjustment thread started!")
        self.all_trigered += 1

        while self.tracking_finished < 1 and run:
            self.dense_ba()
            sleep(self.sleep_time)  # Let multiprocessing cool down a little bit

        # Run one last time after tracking finished
        if run:
            self.dense_ba()

        self.backend_finished += 1
        self.all_finished += 1
        self.info("Full Bundle Adjustment done!")


    def multiview_filtering(self, rank, run=True):
        self.info("Multiview Filtering thread started!")
        self.all_trigered += 1

        while (self.tracking_finished < 1 or self.backend_finished < 1) and run:
            self.multiview_filter()

        self.multiview_filtering_finished += 1
        self.all_finished += 1
        self.info("Multiview Filtering Done!")


    def gaussian_mapping(self, rank, run=True):
        self.info("Gaussian Mapping thread started!")
        self.all_trigered += 1

        # NOTE chen: We run rendering one last time even after backend is done for finetuning
        # at some point we have to think about whether this makes sense or if rendering could be even better
        while self.tracking_finished < 1 and self.backend_finished < 1 and run:
            self.gaussian_mapper()

        # Run for one last time after everything else finished
        finished = False
        while not finished and run:
            finished = self.gaussian_mapper(the_end=True)

        self.gaussian_mapping_finished += 1
        self.all_finished += 1
        self.info("Gaussian Mapping done!")


    def visualizing(self, rank, run=True):
        self.info("Visualization thread started!")
        self.all_trigered += 1

        while (self.tracking_finished < 1 or self.backend_finished < 1) and run:
            droid_visualization(self.video, device=self.device, save_root=self.output)

        self.visualizing_finished += 1
        self.all_finished += 1
        self.info("Visualization done!")

    def mapping_gui(self, rank, run=True):
        self.info("Mapping GUI thread started!")
        self.all_trigered += 1
        if run:
            slam_gui.run(self.params_gui)

        while (self.gaussian_mapping_finished < 1 ) and run:
            pass

        print(self.gaussian_mapper.gaussians[0].shape)

        self.all_finished += 1
        self.info("Mapping GUI done!")


    def show_stream(self, rank, input_queue: mp.Queue) -> None:
        self.info("OpenCV Image stream thread started!")
        self.all_trigered += 1

        while self.tracking_finished < 1 or self.backend_finished < 1:
            if not input_queue.empty():
                try:
                    rgb = input_queue.get()
                    depth = input_queue.get()

                    rgb_image = rgb[0, [2, 1, 0], ...].permute(1, 2, 0).clone().cpu()
                    cv2.imshow("RGB", rgb_image.numpy())
                    if self.mode in ["rgbd", "prgbd"] and depth is not None:
                        # Create normalized depth map with intensity plot
                        depth_image = depth2rgb(depth.clone().cpu())[0]
                        # Convert to BGR for cv2
                        cv2.imshow("depth", depth_image[..., ::-1])
                    cv2.waitKey(1)
                except Exception as e:
                    print(colored(e, "red"))
                    print(colored("Continue ..", "red"))

        self.all_finished += 1
        self.info("Input data stream done!")


    def evaluate(self, stream) -> None:
        """Evaluate our estimated poses against the ground truth poses.

        TODO evaluate the quality of the map/structure as well
        NOTE this is more tricky, as few datasets (especially no in-the-wild) will have a groundtruth mesh/point cloud
        NOTE on in-the-wild data it is best to use the renderer and compare test images, that have not been observed when building the map as a proxy
        """
        from evo.core.trajectory import PoseTrajectory3D
        import evo.main_ape as main_ape
        from evo.core.metrics import PoseRelation
        from evo.core.trajectory import PosePath3D
        import numpy as np
        import pandas as pd

        eval_save_path = "evaluation_results/"

        ### Trajectory evaluation ###
        self.info("Start evaluating ...")
        self.info("#" * 20 + f" Results for {stream.input_folder} ...")

        self.info("Filling/Interpolating poses for non-keyframes ...")
        timestamps = [i for i in range(len(stream))]
        camera_trajectory = self.traj_filler(stream)  # w2cs
        w2w = SE3(self.video.pose_compensate[0].clone().unsqueeze(dim=0)).to(camera_trajectory.device)
        camera_trajectory = w2w * camera_trajectory.inv()
        traj_est = camera_trajectory.data.cpu().numpy()
        estimate_c2w_list = camera_trajectory.matrix().data.cpu()

        out_path = osp.join(self.output, "checkpoints/est_poses.npy")
        np.save(out_path, estimate_c2w_list.numpy())  # c2ws
        out_path = osp.join(eval_save_path, "metrics_traj.txt")

        if self.mode == "rgbd":
            is_monocular = False
        else:
            is_monocular = True
        result_ate = eval_ate(
            self.video,
            kf_ids=list(range(len(self.video.images))),
            save_dir=self.output,
            iterations=-1,
            final=True,  # TODO chen: what is this?
            monocular=is_monocular,
            keyframes_only=False,
            camera_trajectory=camera_trajectory,
            stream=stream,
        )

        self.info("ATE: ", result_ate)
        trajectory_df = pd.DataFrame([result_ate])
        trajectory_df.to_csv(osp.join(self.output, "trajectory_results.csv"), index=False)

        ### Rendering/Map evaluation ###
        # TODO

    def save_state(self) -> None:
        print("Saving checkpoints...")
        os.makedirs(osp.join(self.output, "checkpoints/"), exist_ok=True)
        torch.save(
            {"tracking_net": self.net.state_dict(), "keyframe_timestamps": self.video.timestamp},
            osp.join(self.output, "checkpoints/go.ckpt"),
        )

    def terminate(self, processes: List, stream=None):
        """fill poses for non-keyframe images and evaluate"""

        print("Initiating termination ...")
        for p in processes:
            p.join()

        self.save_state()
        if self.do_evaluate:
            print("Evaluation predictions ...")
            self.evaluate(stream)
            self.info("Evaluation done!")

        self.info("Terminate done!")

    def run(self, stream):
        # TODO visualizing and guassian mapping cannot be run at the same time, because they both access the dirty_index
        # -> introduce multiple indices so we can keep track of what we already visualized and what we already put into the renderer
        processes = [
            # NOTE The OpenCV thread always needs to be 0 to work somehow
            mp.Process(target=self.show_stream, args=(0, self.input_pipe)),
            mp.Process(target=self.tracking, args=(1, stream, self.input_pipe)),
            mp.Process(target=self.backend, args=(2, self.cfg.slam.run_backend)),
            mp.Process(target=self.multiview_filtering, args=(3, self.cfg.slam.run_multiview_filter)),
            mp.Process(target=self.visualizing, args=(4, self.cfg.slam.run_visualization)),
            mp.Process(target=self.mapping_gui, args=(5, self.cfg.slam.run_mapping_gui and self.cfg.slam.run_mapping)),
            mp.Process(target=self.gaussian_mapping, args=(6, self.cfg.slam.run_mapping)),
        ]

        self.num_running_thread[0] += len(processes)
        for p in processes:
            p.start()

        ###
        # Perform intermediate computations you would want to do, e.g. return the last map for evaluation
        ###

        # Wait until system actually is done
        while self.all_finished < self.num_running_thread:
            # Make exception for configuration where we only have frontend tracking
            if self.num_running_thread == 1 and self.tracking_finished > 0:
                break
            
        self.terminate(processes, stream)
