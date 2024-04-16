import os
import ipdb
from omegaconf import DictConfig
from colorama import Fore, Style
from termcolor import colored
from collections import OrderedDict
from tqdm import tqdm
from time import gmtime, strftime, time, sleep
from evo.core.trajectory import PoseTrajectory3D
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PosePath3D
import numpy as np
import pandas as pd
from typing import List, Optional
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
from multiprocessing import Manager
from .gaussian_splatting.multiprocessing_utils import clone_obj

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
    """
    SLAM system with multiprocessing:
        i) Frontend tracking for local optimization of incoming frames and filter based on perceived motion
        ii) Backend optimization of whole map using bundle adjustment
        iii) Gaussian Renderer initialized from dense point cloud map
        iv) Visualization functionality
    """

    def __init__(self, cfg):
        super(SLAM, self).__init__()

        self.cfg = cfg
        self.device = cfg.slam.device
        self.mode = cfg.slam.mode
        self.create_out_dirs(cfg)

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
        ## Used to share packets for evaluation from the gaussian mapper
        self.gaussian_mapper = GaussianMapper(cfg, self)
        self.mapping_queue = mp.Queue()
        self.received_mapping = mp.Event()

        # Stream the images into the main thread
        self.input_pipe = mp.Queue()
        self.do_evaluate = cfg.slam.evaluate
        self.dataset = None

        self.hang_on = torch.zeros((1)).int()
        self.hang_on.share_memory_()

    def create_out_dirs(self, cfg: DictConfig) -> None:
        self.output = cfg.slam.output_folder
        os.makedirs(f"{self.output}/logs/", exist_ok=True)
        os.makedirs(f"{self.output}/renders/mapping/", exist_ok=True)
        os.makedirs(f"{self.output}/renders/final", exist_ok=True)
        os.makedirs(f"{self.output}/mesh", exist_ok=True)
        os.makedirs(f"{self.output}/evaluation", exist_ok=True)
        
    def info(self, msg) -> None:
        print(colored("[Main]: " + msg, "green"))

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


    def gaussian_mapping(self, rank, run, mapping_queue: mp.Queue, received_mapping: mp.Event):
        self.info("Gaussian Mapping Triggered!")
        self.all_trigered += 1

        while self.tracking_finished < 1 and run:
            while self.hang_on > 0:
                sleep(1.0)
            self.gaussian_mapper(mapping_queue, received_mapping)

        # Run for one last time after everything finished
        finished = False
        while not finished and run:
            finished = self.gaussian_mapper(mapping_queue, received_mapping, True)

        self.gaussian_mapping_finished += 1
        self.all_finished += 1
        self.info("Gaussian Mapping Done!")

    def visualizing(self, rank, run=True):
        self.info("Visualization thread started!")
        self.all_trigered += 1

        while (self.tracking_finished < 1 or self.backend_finished < 1) and run:
            droid_visualization(self.video, device=self.device, save_root=self.output)

        self.visualizing_finished += 1
        self.all_finished += 1
        self.info("Visualization done!")


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
                    self.info(str(e))
                    self.info("Continuing...")
                    pass

        self.all_finished += 1
        self.info("Show stream Done!")

    def evaluate(self, stream, gaussian_mapper_last_state):

        eval_path = os.path.join(self.output,"evaluation")

        def stringify_config():
            tbr = "Config: "

            if self.cfg.slam.run_backend:
                tbr +="Backend"
            if self.cfg.slam.run_frontend:
                tbr += " Frontend"
            if self.cfg.slam.run_mapping:
                tbr += " Mapping"

            tbr += " | Stride: " + str(self.cfg.slam.stride)

            return tbr

        self.info("Saving evaluation results in {}".format(self.output))

        rendering_result = eval_rendering(
            gaussian_mapper_last_state.cameras,
            gaussian_mapper_last_state.gaussians,
            stream,
            eval_path,
            gaussian_mapper_last_state.pipeline_params,
            gaussian_mapper_last_state.background,
            kf_indices=[], ## NOTE: all frames are keyframes
            iteration="final",) ## NOTE: only for printing additional messages

        #### ------------------- ####
        ### Trajectory evaluation ###

        self.info("#" * 20 + f" Results for {stream.input_folder} ...")

        ## Trajectory filler
        timestamps = [i for i in range(len(stream))]
        camera_trajectory = self.traj_filler(stream)  # w2cs
        w2w = SE3(self.video.pose_compensate[0].clone().unsqueeze(dim=0)).to(camera_trajectory.device)
        camera_trajectory = w2w * camera_trajectory.inv()
        traj_est = camera_trajectory.data.cpu().numpy()
        estimate_c2w_list = camera_trajectory.matrix().data.cpu()

        # out_path = os.path.join(self.output, "checkpoints/est_poses.npy")
        # np.save(out_path, estimate_c2w_list.numpy())  # c2ws

        ## TODO: change this for depth videos
        # Set keyframes_only to True to compute the APE and plots on keyframes only.

        monocular = self.cfg.slam.mode == "mono"

        result_ate = eval_ate(
            self.video,
            kf_ids=list(range(len(self.video.images))),
            save_dir=eval_path,
            iterations=-1,
            final=True,
            monocular=monocular,
            keyframes_only=False,
            camera_trajectory=camera_trajectory,
            stream=stream,
        )

        self.info("ATE: {}".format(result_ate))

        trajectory_df = pd.DataFrame([result_ate])
        trajectory_df.to_csv(os.path.join(eval_path, "trajectory_results.csv"), index=False)

        #### ------------------- ####
        ## Joint metrics file ##
        columns = ["config",'dataset','mode', "psnr", "ssim", "lpips","ape"]
        data = [
            [
                stringify_config(),
                self.cfg.data.dataset,
                self.cfg.slam.mode,
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                result_ate['mean']
            ]
        ]

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join(eval_path,"evaluation_results.csv"), index=False)

    def save_state(self):
        self.info("Saving checkpoints...")
        os.makedirs(os.path.join(self.output, "checkpoints/"), exist_ok=True)
        torch.save(
            {
                "tracking_net": self.net.state_dict(),
                "keyframe_timestamps": self.video.timestamp,
            },
            os.path.join(self.output, "checkpoints/go.ckpt"),
        )

    def terminate(self, processes: List, stream=None,gaussian_mapper_last_state=None):
        """fill poses for non-keyframe images and evaluate"""
        self.info("Initiating termination ...")

        for i, p in enumerate(processes):
            p.join()
            self.info("Terminated process {}".format(p.name))


        self.save_state()
        if self.do_evaluate:
            self.info("Doing evaluation!")
            self.evaluate(stream, gaussian_mapper_last_state)
            self.info("Evaluation complete")

        self.info("Terminate: Done!")

    def run(self, stream):
        # TODO visualizing and guassian mapping cannot be run at the same time, because they both access the dirty_index
        # -> introduce multiple indices so we can keep track of what we already visualized and what we already put into the renderer
        processes = [
            # NOTE The OpenCV thread always needs to be 0 to work somehow
            mp.Process(target=self.show_stream, args=(0, self.input_pipe), name="OpenCV Stream"),
            mp.Process(target=self.tracking, args=(1, stream, self.input_pipe), name="Frontend Tracking"),
            mp.Process(target=self.backend, args=(2, self.cfg.slam.run_backend), name="Backend"), 
            mp.Process(target=self.multiview_filtering, args=(3, self.cfg.slam.run_multiview_filter),name="Multiview Filtering"),
            mp.Process(target=self.visualizing, args=(4, self.cfg.slam.run_visualization),name="Visualizing"),
            mp.Process(
                target=self.gaussian_mapping,
                args=(5, self.cfg.slam.run_mapping, self.mapping_queue, self.received_mapping),
                name="Gaussian Mapping",
            ),
        ]

        self.num_running_thread[0] += len(processes)
        for p in processes:
            p.start()

        # Wait for all processes to have finished before terminating and for final mapping update to be transmitted
        while self.mapping_queue.empty() and self.all_finished < self.num_running_thread:
            pass

        # Receive the final update, so we can do something with it ...
        a = self.mapping_queue.get()
        gaussian_mapper_last_state = clone_obj(a)
        self.received_mapping.set()
        del a  # NOTE Always delete receive object from a multiprocessing Queue!


        while self.backend_finished < 1 and self.gaussian_mapping_finished < 1:
            self.info("Waiting Backend and Gaussian Renderer to finish ...")
            # Make exception for configuration where we only have frontend tracking
            if self.num_running_thread == 1 and self.tracking_finished > 0:
                break



        self.terminate(processes,stream,gaussian_mapper_last_state)
