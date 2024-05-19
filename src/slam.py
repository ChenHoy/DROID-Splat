import os
import ipdb
from omegaconf import DictConfig
from copy import deepcopy
from termcolor import colored
from collections import OrderedDict
from tqdm import tqdm
from time import sleep
import numpy as np
import pandas as pd
from typing import List, Optional
import os.path as osp
from typing import List, Optional
from tqdm import tqdm
import gc
from time import sleep
from collections import OrderedDict
from omegaconf import DictConfig
from termcolor import colored
import ipdb
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from lietorch import SE3

from .droid_net import DroidNet
from .frontend import FrontendWrapper
from .backend import BackendWrapper
from .depth_video import DepthVideo
from .visualization import droid_visualization, depth2rgb, uncertainty2rgb
from .trajectory_filler import PoseTrajectoryFiller
from .gaussian_mapping import GaussianMapper
from .gaussian_splatting.eval_utils import eval_ate, eval_rendering, save_gaussians
from .gaussian_splatting.gui import gui_utils, slam_gui


from multiprocessing import Manager
from .gaussian_splatting.multiprocessing_utils import clone_obj


class SLAM:
    """SLAM system which bundles together multiple building blocks:
        - Frontend Tracker based on a Motion filter, which successively inserts new frames into a map
            within a local optimization window
        - Backend Bundle Adjustment, which optimizes the map over a global optimization window
        - Gaussian Mapping, which optimizes the map into multiple 3D Gaussian
            based on a dense rendering objective
        - Visualizers for showing the incoming RGB(D) stream, the current pose graph,
            the 3D point clouds of the map, optimized Gaussians

    We combine these building blocks in a multiprocessing environment.
    """

    def __init__(self, cfg, dataset=None, output_folder: Optional[str] = None):
        super(SLAM, self).__init__()

        self.cfg = cfg
        self.device = cfg.get("device", torch.device("cuda:0"))
        self.mode = cfg.mode
        self.create_out_dirs(output_folder)
        self.update_cam(cfg)

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
        self.gaussian_mapping_finished = torch.zeros((1)).int()
        self.gaussian_mapping_finished.share_memory_()
        self.backend_finished = torch.zeros((1)).int()
        self.backend_finished.share_memory_()
        self.visualizing_finished = torch.zeros((1)).int()
        self.visualizing_finished.share_memory_()

        self.mapping_visualizing_finished = torch.zeros((1)).int()
        self.mapping_visualizing_finished.share_memory_()

        # Insert a dummy delay to snychronize frontend and backend as needed
        self.sleep_time = cfg.get("sleep_delay", 3)
        # Delete backend when hitting this threshold, so we can keep going with just frontend
        self.max_ram_usage = cfg.get("max_ram_usage", 0.9)
        self.plot_uncertainty = cfg.get("plot_uncertainty", False)  # Show the optimization uncertainty maps

        # Stream the images into the main thread
        self.input_pipe = mp.Queue()

        # store images, depth, poses, intrinsics (shared between process)
        self.video = DepthVideo(cfg)  # NOTE: we can use this for getting both gt and rendered images
        self.frontend = FrontendWrapper(cfg, self)
        self.backend = BackendWrapper(cfg, self)
        self.traj_filler = PoseTrajectoryFiller(net=self.net, video=self.video, device=self.device)

        self.do_evaluate = cfg.evaluate
        self.dataset = dataset
        self.mapping_queue = mp.Queue()
        self.received_mapping = mp.Event()

        if cfg.data.dataset in ["kitti", "tartanair", "euroc"]:
            self.max_depth_visu = 50.0  # Cut of value to show a consistent depth stream
        else:
            self.max_depth_visu = 5.0

        if cfg.run_mapping_gui and cfg.run_mapping and not cfg.evaluate:
            self.q_main2vis = mp.Queue()
            self.gaussian_mapper = GaussianMapper(cfg, self, gui_qs=(self.q_main2vis))
            self.params_gui = gui_utils.ParamsGUI(
                pipe=cfg.mapping.pipeline_params,
                background=self.gaussian_mapper.background,
                gaussians=self.gaussian_mapper.gaussians,
                q_main2vis=self.q_main2vis,
            )
        else:
            self.gaussian_mapper = GaussianMapper(cfg, self)

    def info(self, msg) -> None:
        print(colored("[Main]: " + msg, "green"))

    def create_out_dirs(self, output_folder: Optional[str] = None) -> None:
        if output_folder is not None:
            self.output = output_folder
        else:
            self.output = "./outputs/"

        os.makedirs(self.output, exist_ok=True)
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

    # TODO chen: do we really still need this?
    def load_bound(self, cfg: DictConfig) -> None:
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

        self.info("Frontend tracking thread started!")
        self.all_trigered += 1

        # Wait up for other threads to start
        while self.all_trigered < self.num_running_thread:
            pass

        for frame in tqdm(stream):
            timestamp, image, depth, intrinsic, gt_pose = frame
            if self.mode not in ["rgbd", "prgbd"]:
                depth = None

            if self.cfg.show_stream:
                # Transmit the incoming stream to another visualization thread
                input_queue.put(image)
                input_queue.put(depth)

            self.frontend(timestamp, image, depth, intrinsic, gt_pose)

        del self.frontend
        torch.cuda.empty_cache()
        gc.collect()

        self.tracking_finished += 1
        self.all_finished += 1
        self.info("Frontend Tracking done!")

    def get_ram_usage(self):
        free_mem, total_mem = torch.cuda.mem_get_info(device=self.device)
        used_mem = 1 - (free_mem / total_mem)
        return used_mem, free_mem

    def ram_safeguard_backend(self, max_ram: float = 0.9, min_ram: float = 0.5) -> None:
        """There are some scenes, where we might get into trouble with memory.
        In order to keep the system going, we simply dont use the backend until we can afford it again.
        """
        used_mem, free_mem = self.get_ram_usage()
        if used_mem > max_ram and self.backend is not None:
            print(colored(f"[Main]: Warning: Deleting Backend due to high memory usage [{used_mem} %]!", "red"))
            print(colored(f"[Main]: Warning: Warning: Got only {free_mem/ 1024 ** 3} GB left!", "red"))
            del self.backend
            self.backend = None
            gc.collect()
            torch.cuda.empty_cache()

        # NOTE chen: if we deleted the backend due to memory issues we likely have not a lot of capacity left for backend
        # only use backend again once we have some slack -> 50% free RAM (12GB in use)
        if self.backend is None and used_mem <= min_ram:
            self.info("Reinstantiating Backend ...")
            self.backend = BackendWrapper(self.cfg, self.args, self)
            self.backend.to(self.device)

    def global_ba(self, rank, run=False):
        self.info("Backend thread started!")
        self.all_trigered += 1

        while self.tracking_finished < 1 and run:

            # Only run backend if we have enough RAM for it
            self.ram_safeguard_backend(max_ram=self.max_ram_usage)
            if self.backend is not None:
                if self.backend.enable_loop:
                    self.backend(local_graph=self.frontend.optimizer.graph)
                else:
                    self.backend()
                sleep(self.sleep_time)  # Let multiprocessing cool down a little bit

        # Run one last time after tracking finished
        if run and self.backend is not None:
            with self.video.get_lock():
                t_end = self.video.counter.value

            msg = "Optimize full map: [{}, {}]!".format(0, t_end)
            self.backend.info(msg)
            _, _ = self.backend.optimizer.dense_ba(t_start=0, t_end=t_end, steps=6)
            _, _ = self.backend.optimizer.dense_ba(t_start=0, t_end=t_end, steps=6)

        del self.backend
        torch.cuda.empty_cache()
        gc.collect()

        self.backend_finished += 1
        self.all_finished += 1
        self.info("Backend done!")

    def gaussian_mapping(self, rank, run, mapping_queue: mp.Queue, received_mapping: mp.Event):
        self.info("Gaussian Mapping Triggered!")
        self.all_trigered += 1

        while (self.tracking_finished + self.backend_finished) < 2 and run:
            self.gaussian_mapper(mapping_queue, received_mapping)
            sleep(self.sleep_time / 4)

        # Run for one last time after everything finished
        finished = False
        while not finished and run:
            finished = self.gaussian_mapper(mapping_queue, received_mapping, True)

        self.gaussian_mapping_finished += 1
        while self.mapping_visualizing_finished < 1:
            pass

        self.all_finished += 1
        self.info("Gaussian Mapping Done!")

    def visualizing(self, rank, run=True):
        self.info("Visualization thread started!")
        self.all_trigered += 1
        finished = False

        while (self.tracking_finished + self.backend_finished < 2) and run and not finished:
            finished = droid_visualization(self.video, device=self.device, save_root=self.output)

        self.visualizing_finished += 1
        self.all_finished += 1
        self.info("Visualization done!")

    def mapping_gui(self, rank, run=True):
        self.info("Mapping GUI thread started!")
        self.all_trigered += 1
        finished = False

        while (self.tracking_finished + self.backend_finished < 2) and run and not finished:
            finished = slam_gui.run(self.params_gui)

        # Wait for Gaussian Mapper to be finished so nothing new is put into the queue anymore
        while self.gaussian_mapping_finished < 1:
            pass

        # empty all the guis that are in params_gui so this will for sure get empty
        if run: # NOTE Leon: It crashes if we dont check this
            while not self.params_gui.q_main2vis.empty():
                obj = self.params_gui.q_main2vis.get()
                a = clone_obj(obj)
                del obj

        self.mapping_visualizing_finished += 1
        self.all_finished += 1
        self.info("Mapping GUI done!")

    def show_stream(self, rank, input_queue: mp.Queue, run=True) -> None:
        self.info("OpenCV Image stream thread started!")
        self.all_trigered += 1

        while (self.tracking_finished + self.backend_finished < 2) and run:
            if not input_queue.empty():
                try:
                    rgb = input_queue.get()
                    depth = input_queue.get()

                    rgb_image = rgb[0, [2, 1, 0], ...].permute(1, 2, 0).clone().cpu()
                    cv2.imshow("RGB", rgb_image.numpy())
                    if self.mode in ["rgbd", "prgbd"] and depth is not None:
                        # Create normalized depth map with intensity plot
                        depth_image = depth2rgb(depth.clone().cpu(), max_depth=self.max_depth_visu)[0]
                        # Convert to BGR for cv2
                        cv2.imshow("depth", depth_image[..., ::-1])
                    cv2.waitKey(1)
                except Exception as e:
                    pass
                    # Uncomment if you observe something weird, this will exit once the stream is finished
                    # print(colored(e, "red"))
                    # print(colored("Continue ..", "red"))

            if self.plot_uncertainty:
                # Plot the uncertainty on top
                with self.video.get_lock():
                    t_cur = max(0, self.video.counter.value - 1)
                    if self.cfg.tracking.get("upsample", False):
                        uncertanity_cur = self.video.uncertainty_up[t_cur].clone()
                    else:
                        uncertanity_cur = self.video.uncertainty[t_cur].clone()
                uncertainty_img = uncertainty2rgb(uncertanity_cur)[0]
                cv2.imshow("Uncertainty", uncertainty_img[..., ::-1])
                cv2.waitKey(1)

        self.all_finished += 1
        self.info("Show stream Done!")

    def evaluate(self, stream, gaussian_mapper_last_state: Optional[gui_utils.EvaluatePacket] = None):

        eval_path = os.path.join(self.output, "evaluation")
        self.info("Saving evaluation results in {}".format(eval_path))
        self.info("#" * 20 + f" Results for {stream.input_folder} ...")

        #### ------------------- ####
        ### Trajectory evaluation ###
        ## Trajectory filler
        timestamps = [i for i in range(len(stream))]
        camera_trajectory = self.traj_filler(stream)  # w2cs
        w2w = SE3(self.video.poses_clean[0].clone().unsqueeze(dim=0)).to(camera_trajectory.device)
        camera_trajectory = w2w * camera_trajectory.inv()
        traj_est = camera_trajectory.data.cpu().numpy()
        estimate_c2w_list = camera_trajectory.matrix().data.cpu()

        # Set keyframes_only to True to compute the APE and plots on keyframes only.
        monocular = self.cfg.mode == "mono"
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
        # out_path = os.path.join(self.output, "checkpoints/est_poses.npy")
        # np.save(out_path, estimate_c2w_list.numpy())  # c2ws

        ## Joint metrics file ##
        configuration_columns = ["run_backend", "run_mapping", "stride", "loop_closure"]
        columns = configuration_columns + ["dataset", "mode", "ape"]
        result_table = [
            str(self.cfg.run_backend),
            str(self.cfg.run_mapping),
            str(self.cfg.stride),
            str(self.backend.enable_loop),
            stream.input_folder,
            self.cfg.mode,
            result_ate["mean"],
        ]

        ### Rendering evaluation ###
        if self.cfg.run_mapping:
            rendering_result = eval_rendering(
                gaussian_mapper_last_state.cameras,
                gaussian_mapper_last_state.gaussians,
                stream,
                eval_path,
                gaussian_mapper_last_state.pipeline_params,
                gaussian_mapper_last_state.background,
                kf_indices=[],  ## NOTE: all frames are keyframes
                iteration="final",
            )
            columns += ["psnr", "ssim", "lpips"]
            result_table += [
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
            ]

        data = [result_table]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join(eval_path, "evaluation_results.csv"), index=False)

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

    def terminate(self, processes: List, stream=None, gaussian_mapper_last_state=None):
        """fill poses for non-keyframe images and evaluate"""
        self.info("Initiating termination ...")

        # self.save_state()  ## this is not reached
        if self.do_evaluate:
            self.info("Doing evaluation!")
            self.evaluate(stream, gaussian_mapper_last_state=gaussian_mapper_last_state)
            self.info("Evaluation complete")

        for i, p in enumerate(processes):
            p.terminate()
            p.join()
            self.info("Terminated process {}".format(p.name))
        self.info("Terminate: Done!")

    def test(self, stream):
        """Test the system by running any function dependent on the input stream directly so we can set breakpoints for inspection."""

        processes = [
            # mp.Process(target=self.show_stream, args=(0, self.input_pipe)),
            mp.Process(target=self.visualizing, args=(1, False))
        ]

        self.num_running_thread[0] += len(processes)
        for p in processes:
            p.start()

        for frame in tqdm(stream):
            timestamp, image, depth, intrinsic, gt_pose = frame
            self.frontend(timestamp, image, depth, intrinsic, gt_pose)

    def run(self, stream):
        processes = [
            # NOTE The OpenCV thread always needs to be 0 to work somehow
            mp.Process(target=self.show_stream, args=(0, self.input_pipe, self.cfg.show_stream), name="OpenCV Stream"),
            mp.Process(target=self.tracking, args=(1, stream, self.input_pipe), name="Frontend Tracking"),
            mp.Process(target=self.global_ba, args=(2, self.cfg.run_backend), name="Backend"),
            mp.Process(target=self.visualizing, args=(3, self.cfg.run_visualization), name="Visualizing"),
            mp.Process(
                target=self.gaussian_mapping,
                args=(4, self.cfg.run_mapping, self.mapping_queue, self.received_mapping),
                name="Gaussian Mapping",
            ),
            mp.Process(
                target=self.mapping_gui,
                args=(5, self.cfg.run_mapping_gui and self.cfg.run_mapping and not self.cfg.evaluate),
                name="Mapping GUI",
            ),
        ]

        self.num_running_thread[0] += len(processes)
        for p in processes:
            p.start()

        # Wait for all processes to have finished before terminating and for final mapping update to be transmitted
        if self.cfg.run_mapping:
            while self.mapping_queue.empty():
                pass
            # Receive the final update, so we can do something with it ...
            a = self.mapping_queue.get()
            self.info("Received final mapping update!")
            if a == "None":
                a = deepcopy(a)
                gaussian_mapper_last_state = None
            else:
                gaussian_mapper_last_state = clone_obj(a)
            del a  # NOTE Always delete receive object from a multiprocessing Queue!
            self.received_mapping.set()

        # Let the processes run until they are finished (When using GUI's these need to be closed manually)
        else:
            gaussian_mapper_last_state = None

        while self.all_finished < self.num_running_thread:
            pass

        self.terminate(processes, stream, gaussian_mapper_last_state)
