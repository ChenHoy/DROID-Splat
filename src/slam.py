import os
import ipdb
from colorama import Fore, Style
from termcolor import colored
from collections import OrderedDict
from tqdm import tqdm
from time import gmtime, strftime, time, sleep

import cv2
import open3d as o3d
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
import pandas as pd
from .render import Renderer
from multiprocessing import Manager
from .gaussian_splatting.multiprocessing_utils import clone_obj

class Tracker(nn.Module):
    def __init__(self, cfg, slam):
        super(Tracker, self).__init__()
        self.cfg = cfg
        self.device = cfg.slam.device
        self.net = slam.net
        self.video = slam.video
        self.verbose = slam.verbose

        # filter incoming frames so that there is enough motion
        self.frontend_window = cfg.tracking.frontend.window
        filter_thresh = cfg.tracking.motion_filter.thresh
        self.motion_filter = MotionFilter(self.net, self.video, thresh=filter_thresh, device=self.device)

        # frontend process
        self.frontend = Frontend(self.net, self.video, self.cfg)

    def forward(self, timestamp, image, depth, intrinsic, gt_pose=None):
        with torch.no_grad():
            ### check there is enough motion
            self.motion_filter.track(timestamp, image, depth, intrinsic, gt_pose=gt_pose)

            # local bundle adjustment
            self.frontend()


class BundleAdjustment(nn.Module):
    def __init__(self, cfg, slam):
        super(BundleAdjustment, self).__init__()
        self.cfg = cfg
        self.device = cfg.slam.device
        self.net = slam.net
        self.video = slam.video
        self.verbose = slam.verbose
        self.frontend_window = cfg["tracking"]["frontend"]["window"]
        self.last_t = -1
        self.ba_counter = -1

        # backend process
        self.backend = Backend(self.net, self.video, self.cfg)

    def info(self, msg: str) -> None:
        print(colored("[Backend]" + msg, "blue"))

    def forward(self):
        cur_t = self.video.counter.value
        t = cur_t

        if cur_t > self.frontend_window:
            t_start = 0
            now = f'{strftime("%Y-%m-%d %H:%M:%S", gmtime())} - Full BA'
            msg = f"\n\n {now} : [{t_start}, {t}]; Current Keyframe is {cur_t}, last is {self.last_t}."

            self.backend.dense_ba(t_start=t_start, t_end=t, steps=6, motion_only=False)
            self.info(msg + "\n")

            self.last_t = cur_t


class SLAM:
    def __init__(self, cfg):
        super(SLAM, self).__init__()
        self.cfg = cfg
        self.device = cfg.slam.device
        self.verbose = cfg.slam.verbose
        self.mode = cfg.slam.mode
        self.only_tracking = cfg.slam.only_tracking
        self.make_video = cfg.slam.make_video

        self.output = cfg.slam.output_folder

        os.makedirs(f"{self.output}/logs/", exist_ok=True)
        os.makedirs(f"{self.output}/renders/mapping/", exist_ok=True)
        os.makedirs(f"{self.output}/renders/final", exist_ok=True)
        os.makedirs(f"{self.output}/mesh", exist_ok=True)

        self.update_cam(cfg)
        self.load_bound(cfg)

        self.net = DroidNet()

        self.load_pretrained(cfg.tracking.pretrained)
        self.net.to(self.device).eval()
        self.net.share_memory()

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
        self.optimizing_finished = torch.zeros((1)).int()
        self.optimizing_finished.share_memory_()
        self.visualizing_finished = torch.zeros((1)).int()
        self.visualizing_finished.share_memory_()

        self.hang_on = torch.zeros((1)).int()
        self.hang_on.share_memory_()

        self.reload_map = torch.zeros((1)).int()
        self.reload_map.share_memory_()

        # store images, depth, poses, intrinsics (shared between process)
        self.video = DepthVideo(cfg)

        self.tracker = Tracker(cfg, self)
        self.ba = BundleAdjustment(cfg, self)

        self.multiview_filter = MultiviewFilter(cfg, self)

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(net=self.net, video=self.video, device=self.device)

        ## Used to share packets for evaluation from the gaussian mapper
        self.gaussian_mapper = GaussianMapper(cfg, self)
        self.mapping_queue = mp.Queue()
        self.received_mapping = mp.Event()

        # Stream the images into the main thread
        self.input_pipe = mp.Queue()
        self.evaluate = cfg.slam.evaluate
        self.dataset = None

    def set_dataset(self, dataset):
        self.dataset = dataset

    def info(self, msg: str) -> None:
        print(colored("[Main]" + msg, "green"))

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

    def load_bound(self, cfg):
        """
        Pass the scene bound parameters to different decoders and self.
        Args:
            cfg:                        (dict), parsed config dict

        """
        # scale the bound if there is a global scaling factor
        self.bound = torch.from_numpy(np.array(cfg.data.bound)).float()

    def load_pretrained(self, pretrained):
        self.info(f"INFO: load pretrained checkpiont from {pretrained}!")

        state_dict = OrderedDict([(k.replace("module.", ""), v) for (k, v) in torch.load(pretrained).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)

    def tracking(self, rank, stream, input_queue: mp.Queue):
        self.info("Tracking Triggered!")
        self.all_trigered += 1
        # Wait up for other threads to start
        while self.all_trigered < self.num_running_thread:
            pass
        for frame in tqdm(stream):
            timestamp, image, depth, intrinsic, gt_pose = frame
            if self.mode not in ["rgbd", "prgbd"]:
                depth = None
            # Transmit the incoming stream to another visualization thread
            input_queue.put(image)
            input_queue.put(depth)

            self.tracker(timestamp, image, depth, intrinsic, gt_pose)

            if timestamp % 50 == 0 and timestamp > 0 and self.make_video:
                self.hang_on[:] = 1
            while self.hang_on > 0:
                sleep(1.0)

        self.tracking_finished += 1
        self.all_finished += 1
        self.info("Tracking Done!")

    def optimizing(self, rank, dont_run=False):
        self.info("Full Bundle Adjustment Triggered!")
        self.all_trigered += 1
        while self.tracking_finished < 1 and not dont_run:
            while self.hang_on > 0 and self.make_video:
                sleep(1.0)
            self.ba()
            sleep(2.0)  # Let multiprocessing cool down a little bit

        if not dont_run:
            self.ba()

        self.optimizing_finished += 1
        self.all_finished += 1
        self.info("Full Bundle Adjustment Done!")

    def multiview_filtering(self, rank, dont_run=False):
        self.info("Multiview Filtering Triggered!")
        self.all_trigered += 1
        while (self.tracking_finished < 1 or self.optimizing_finished < 1) and not dont_run:
            while self.hang_on > 0 and self.make_video:
                sleep(1.0)
            self.multiview_filter()

        self.multiview_filtering_finished += 1
        self.all_finished += 1
        self.info("Multiview Filtering Done!")


    # FIXME Gaussian mapper gets a different queue here than the one we originally passed
    def gaussian_mapping(self, rank, dont_run, mapping_queue: mp.Queue, received_mapping: mp.Event):
        self.info("Gaussian Mapping Triggered!")

        self.all_trigered += 1
        while self.tracking_finished < 1 and not dont_run:
            while self.hang_on > 0 and self.make_video:
                sleep(1.0)
            self.gaussian_mapper(mapping_queue, received_mapping)

        finished = False
        if not dont_run:
            self.info("[Gaussian mapping] Last run")
            while not finished:
                finished = self.gaussian_mapper(mapping_queue, received_mapping, True)

        self.gaussian_mapping_finished += 1
        self.all_finished += 1
        # self.shared_space['gaussian_mapper'] = self.gaussian_mapper
        self.info("Gaussian Mapping Done!")

    def visualizing(self, rank, dont_run=False):
        self.info("Visualization triggered!")
        self.all_trigered += 1
        while (self.tracking_finished < 1 or self.optimizing_finished < 1) and (not dont_run):
            droid_visualization(self.video, device=self.device, save_root=self.output)

        self.visualizing_finished += 1
        self.all_finished += 1
        self.info("Visualization Done!")

    def show_stream(self, rank, input_queue: mp.Queue) -> None:
        self.info("OpenCV Image stream triggered!")
        self.all_trigered += 1
        while self.tracking_finished < 1 or self.optimizing_finished < 1:
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

    def terminate(self, rank, stream=None):
        """fill poses for non-keyframe images and evaluate"""

        self.info("Initiating termination sequence!")
        while self.optimizing_finished < 1:
            self.info("Waiting for the optimizing...")
            if self.num_running_thread == 1 and self.tracking_finished > 0:
                break

        self.info("Saving checkpoints...")
        os.makedirs(os.path.join(self.output, "checkpoints/"), exist_ok=True)
        torch.save(
            {
                "tracking_net": self.net.state_dict(),
                "keyframe_timestamps": self.video.timestamp,
            },
            os.path.join(self.output, "checkpoints/go.ckpt"),
        )

        self.info("Doing evaluation!")
        if self.evaluate:
            from evo.core.trajectory import PoseTrajectory3D
            import evo.main_ape as main_ape
            from evo.core.metrics import PoseRelation
            from evo.core.trajectory import PosePath3D
            import numpy as np
            import pandas as pd

            eval_save_path = "evaluation_results/"

            #### Rendering evaluation ####
            print("Initialize my evaluation in the termination method")
            print("Shape of images:",self.video.images.shape)
            print("Shape of poses:",self.video.poses.shape)
            print("Shape of ground truth poses:",self.video.poses_gt.shape)
            print("Check that gt poses are not empty:",self.video.poses_gt[25])



            print("Number of cameras / frames:",len(self.evaluation_packet.cameras))
            # print("Last index of gaussian_mapper:",self.gaussian_mapper.last_idx)

            # # self.info("Retrieved mapper camera {}. Retrieved mapper last idx {}".format(retrieved_mapper.cameras,retrieved_mapper.last_idx))
            # _, kf_indices = self.gaussian_mapper.select_keyframes()
            # self.info("Selected keyframe indices:", kf_indices)

            '''
            We're passing only keyframes in self.evaluation_packet.cameras so kf_indcies is not used
            '''
            print("Gaussians:",type(self.evaluation_packet.gaussians),self.evaluation_packet.gaussians)

            rendering_result = eval_rendering(
                self.evaluation_packet.cameras,
                self.evaluation_packet.gaussians,
                self.dataset,
                eval_save_path,
                self.evaluation_packet.pipeline_params,
                self.evaluation_packet.background,
                kf_indices=[],
                iteration="final",
            )

            self.info("Rendering loss: {}".format(rendering_result))


            #### ------------------- ####

            ### Trajectory evaluation ###

            self.info("#" * 20 + f" Results for {stream.input_folder} ...")

            timestamps = [i for i in range(len(stream))]
            camera_trajectory = self.traj_filler(stream)  # w2cs
            w2w = SE3(self.video.pose_compensate[0].clone().unsqueeze(dim=0)).to(camera_trajectory.device)
            camera_trajectory = w2w * camera_trajectory.inv()
            traj_est = camera_trajectory.data.cpu().numpy()
            estimate_c2w_list = camera_trajectory.matrix().data.cpu()

            out_path = os.path.join(self.output, "checkpoints/est_poses.npy")
            np.save(out_path, estimate_c2w_list.numpy())  # c2ws

            ## TODO: change this for depth videos
            """
            Set keyframes_only to True to compute the APE and plots on keyframes only.
            """
            result_ate = eval_ate(
                self.video,
                kf_ids=list(range(len(self.video.images))),
                save_dir=self.output,
                iterations=-1,
                final=True,
                monocular=True,
                keyframes_only=False,
                camera_trajectory=camera_trajectory,
                stream=stream,
            )

            out_path = os.path.join(eval_save_path, "metrics_traj.txt")

            self.info("ATE: {}".format(result_ate))

            trajectory_df = pd.DataFrame([result_ate])
            trajectory_df.to_csv(os.path.join(self.output, "trajectory_results.csv"), index=False)

            #### ------------------- ####

            ## Joint metrics file ##

            # ## TODO: add ATE
            # columns = ["tag", "psnr", "ssim", "lpips","ape"]
            # data = [
            #     [
            #         "optimizing + multiview ",
            #         rendering_result["mean_psnr"],
            #         rendering_result["mean_ssim"],
            #         rendering_result["mean_lpips"],
            #         result_ape.stats['mean']
            #     ]
            # ]

            # df = pd.DataFrame(data, columns=columns)
            # df.to_csv(os.path.join(eval_save_path,"metrics.csv"), index=False)

            self.info("Evaluation complete")
        self.info("Terminate: Done!")

    # NOTE communication between tracking and show_stream works based on input_pipe
    # Tracking feeds objects into input_pipe and show_stream reads from it

    # FIXME communication between gaussian_mapping and the main thread here does not work.
    # Why is that the case?
    def run(self, stream):


        processes = [
            # NOTE The OpenCV thread always needs to be 0 to work somehow
            mp.Process(target=self.show_stream, args=(0, self.input_pipe), name="OpenCV Stream"),
            mp.Process(target=self.tracking, args=(1, stream, self.input_pipe), name="Tracking"),
            mp.Process(target=self.optimizing, args=(2, False), name="Optimizing"),
            mp.Process(target=self.multiview_filtering, args=(3, False), name="Multiview Filtering"),
            mp.Process(target=self.visualizing, args=(4, True), name="Visualizing"),
            mp.Process(
                target=self.gaussian_mapping,
                args=(5, False, self.mapping_queue, self.received_mapping),
                name="Gaussian Mapping",
            ),
        ]

        self.num_running_thread[0] += len(processes)
        for p in processes:
            p.start()

        # Wait for final mapping update
        while self.mapping_queue.empty():
            pass

        a = self.mapping_queue.get()
        self.received_mapping.set()
        # Do something with a
        # self.info("From mapping queue: {}".format(a))
        ## clone the packet for the evaluation process
        self.evaluation_packet = clone_obj(a)
        del a

        # Wait for all processes to have finished, so we can be sure that the mapping queue is not empty, i.e. the last item has been put in
        while self.all_finished < self.num_running_thread:
            pass

        for i, p in enumerate(processes):
            self.info("Joining process {}".format(p.name))
            p.join()
            # FIXME we hang here for gaussian mapping
            self.info("Joined process {}".format(p.name))

        self.info("Run method is finished")  ## BUG: doesnt get here
