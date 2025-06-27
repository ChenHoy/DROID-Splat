import os
import ipdb
import gc
from time import sleep, time, perf_counter
from typing import List, Optional, Tuple
from tqdm import tqdm
import logging
from copy import deepcopy
from termcolor import colored
from collections import OrderedDict
from omegaconf import DictConfig

import cv2
import numpy as np
import pandas as pd

import torch
import torch.multiprocessing as mp
from lietorch import SE3

from .droid_net import DroidNet
from .frontend import FrontendWrapper
from .backend import BackendWrapper
from .depth_video import DepthVideo
from .geom import pose_distance
from .visualization import droid_visualization, depth2rgb, uncertainty2rgb
from .trajectory_filler import PoseTrajectoryFiller
from .loop_detection import LoopDetector, merge_candidates
from .gaussian_mapping import GaussianMapper

from .gaussian_splatting.camera_utils import Camera
from .gaussian_splatting import eval_utils
from .gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, focal2fov
from .gaussian_splatting.gui import gui_utils, slam_gui
from .utils import clone_obj, get_all_queue

# A logger for this file
log = logging.getLogger(__name__)


class SLAM:
    """SLAM system which bundles together multiple building blocks:
        - Frontend Tracker based on a Motion filter, which successively inserts new frames into a map
            within a local optimization window
        - Backend Bundle Adjustment, which optimizes the map over a global optimization window
        - Loop Detector, which computes visual similarity between the current keyframe and all past keyframes. If suitable candidates are found,
            these are send as additional edges to the Backend optimization.
        - Gaussian Mapping, which optimizes the map into multiple 3D Gaussian based on a dense Rendering objective
        - Visualizers for showing the incoming RGB(D) stream, the current pose graph, the 3D point clouds of the map, optimized Gaussians

    We combine these building blocks in a multi-processing environment. By setting frequencies you can synchronize these processes,
    where the Frontend is the leading Process and everything else follows.

    The final runtime will be determined by
        i) how many Processes run in parallel
        ii) Which Process is the slowest and how often it is called.

        In general the Frontend should be the fastest Process and act as an upper bound to how fast we can get.
        You can expect 15 - 25 FPS on normal data with mid resolution.
        You can see the final runtime of our system printed at the end
        NOTE: We usually get a significant slow down only because of the Renderer/Mapper. When configured well, we can still hit ~6 FPS with everything.

    Memory consumption is driven by (many) different building blocks, mainly:
        - Video buffer in float32 with size 256 - 512 keyframes
        - Neural networks for i) DROID optical flow ii) (optional) RAFT optical flow / EIGEN visual place recognition
        - Local Frontend Pose Graph
        - Global Backend Pose Graph
        - Renderer with optimization batch size, number of total Gaussians, resolution of data
    We usually hit 17 - 22GB with all components together. You have to watch out on outdoor scenes, where the buffer
    is either longer or the resolution is too high. Running all components can easily result in an OOM if not careful!
    """

    def __init__(self, cfg: DictConfig, dataset=None, output_folder: Optional[str] = None):
        super(SLAM, self).__init__()

        self.cfg = cfg
        self.device = cfg.get("device", torch.device("cuda:0"))
        self.mode = cfg.get("mode", "mono")
        self.do_evaluate = cfg.get("evaluate", False)
        self.save_predictions = cfg.get("save_rendered_predictions", False)  # save all predictions for later
        self.render_images = cfg.get("render_images", True)  # make a comparison figure
        # Render every 5-th frame during optimization, so we can see the development of our scene representation
        self.save_every = cfg.get("save_every", 5)

        self.create_out_dirs(output_folder)
        self.update_cam(cfg)  # Update output resolution, intrinsics, etc.

        ### Main SLAM neural network
        self.net = DroidNet()
        self.load_pretrained(cfg.tracking.pretrained)
        self.net.to(self.device).eval()
        self.net.share_memory()

        # Insert a dummy delay to snychronize frontend and backend as needed
        self.sleep_time = cfg.get("sleep_delay", 0.1)

        # Delete backend when hitting this threshold, so we can keep going with just frontend
        self.max_ram_usage = cfg.get("max_ram_usage", 0.8)
        self.plot_uncertainty = cfg.get("plot_uncertainty", False)  # Show the optimization uncertainty maps

        ### Main SLAM components
        self.dataset = dataset
        self.video = DepthVideo(cfg)  # store images, depth, poses, intrinsics (shared between process)
        self.traj_filler = PoseTrajectoryFiller(self.cfg, net=self.net, video=self.video, device=self.device)
        self.frontend = FrontendWrapper(cfg, self)

        if self.cfg.run_backend:
            self.backend = BackendWrapper(cfg, self)
            # NOTE self.frontend.window is not an accurate representation over which frames we optimize!
            # Because we delete edges of high age, we usually dont optimize over the whole window, but cut off past frames very fast
            # Example: Window might be [0, 25], but we only optimize [15, 25] -> [0, 15] is untouched and could be handled by backend
            self.backend_warmup = self.backend.warmup
        else:
            self.backend = None
            self.backend_warmup = 9999

        if cfg.run_loop_detection:
            self.loop_detector = LoopDetector(self.cfg.loop_closure, self.net, self.video, self.device)
        else:
            self.loop_detector = None

        if cfg.run_mapping_gui and cfg.run_mapping:
            self.q_main2vis = mp.Queue()
            self.gaussian_mapper = GaussianMapper(cfg, self, gui_qs=(self.q_main2vis))
            self.mapping_warmup = min(self.frontend.window, self.gaussian_mapper.warmup)
            self.params_gui = gui_utils.ParamsGUI(
                pipe=cfg.mapping.pipeline_params,
                background=self.gaussian_mapper.background,
                gaussians=self.gaussian_mapper.gaussians,
                q_main2vis=self.q_main2vis,
            )
        elif cfg.run_mapping:
            self.gaussian_mapper = GaussianMapper(cfg, self)
            self.mapping_warmup = min(self.frontend.window, self.gaussian_mapper.warmup)
        else:
            self.gaussian_mapper = None
            self.mapping_warmup = 9999

        self.sanity_checks()

        # Visualize the Depth maps in right range
        if cfg.data.dataset in ["kitti", "tartanair", "euroc"]:
            self.max_depth_visu = 50.0  # Cut of value to show a consistent depth stream in outdoor datasets
        else:
            self.max_depth_visu = 10.0  # Good value for indoor datasets (maybe make this even lower)

        ### Multi-threading stuff
        # Objects for communicating between processes
        self.input_pipe = mp.Queue()  # Communicate data from Stream -> main thread
        self.mapping_queue = mp.Queue()  # Communicate data between Mapping <-> main thread
        self.ba_lock = mp.Lock()  # Block the bundle adjustment optimization explicitly
        self.received_mapping = mp.Event()  # Ensure we have received the mapping state before moving on
        self.loop_queue = mp.Queue()  # Communicate loop candidates between Loop Detection -> Backend thread

        # Manage life time of individual processes
        self.num_running_thread = torch.zeros((1)).int().share_memory_()
        self.all_trigered = torch.zeros((1)).int().share_memory_()
        self.backend_can_start = torch.zeros((1)).int().share_memory_()  # When to trigger after warmup
        self.mapping_can_start = torch.zeros((1)).int().share_memory_()  # When to trigger after warmup
        self.all_finished = torch.zeros((1)).int().share_memory_()
        self.tracking_finished = torch.zeros((1)).int().share_memory_()
        self.backend_finished = torch.zeros((1)).int().share_memory_()
        self.gaussian_mapping_finished = torch.zeros((1)).int().share_memory_()
        # NOTE chen: we use this flag to avoid frontend to start a new optimization window while the Renderer is working
        # Flag to signal that the current optimization is done or not
        self.mapping_done = torch.zeros((1)).int().share_memory_()
        self.mapping_done += 1  # Set to 1, so Tracking does not wait for mapping
        self.loop_detection_finished = torch.zeros((1)).int().share_memory_()
        self.visualizing_finished = torch.zeros((1)).int().share_memory_()
        self.mapping_visualizing_finished = torch.zeros((1)).int().share_memory_()

        # Synchronization objects
        self.backend_freq = self.cfg.get("backend_every", 10)  # Run the backend every k frontend calls
        self.mapping_freq = self.cfg.get("mapper_every", 5)  # Run the Renderer/Mapper every k frontend calls
        self.sema_backend = mp.Semaphore(1)  # Semaphore allows to keep concurrency
        self.sema_mapping = mp.Semaphore(1)  # Semaphore allows to keep concurrency
        self.communication_lock = mp.Lock()
        self.cond_mapping = mp.Condition(lock=self.communication_lock)  # Conditional for fine-grained synchronization

    def info(self, msg: str, logger=None) -> None:
        if logger is not None:
            logger.info(colored("[Main]: " + msg, "green"))
        else:
            print(colored("[Main]: " + msg, "green"))

    def sanity_checks(self) -> None:
        """Perform sanity checks to see if the system is misconfigured, this is just supposed
        to protect the user when running the system"""
        assert (
            self.cfg.run_frontend
        ), "All other systems rely on the Frontend Tracker. This component must run at all times!"
        if self.cfg.mode == "stereo":
            # NOTE chen: I noticed, that this is really not impemented, i.e.
            # we would need to do some changes in motion_filter, depth_video, BA, etc. to even store the right images, fmaps, etc.
            # NOTE chen: Teed definitely had the idea on dataloader level to return (image_left, image_right) as one sample for images
            raise NotImplementedError(colored("Stereo mode not supported yet!", "red"))
        if self.cfg.mode == "prgbd":
            assert not (self.video.optimize_scales and self.video.opt_intr), colored(
                """Optimizing both poses, disparities, scale & shift and 
            intrinsics creates unforeseen ambiguities!
            This is usually not stable :(
            """,
                "red",
            )
        if self.cfg.run_mapping:
            if self.cfg.mapping.refinement.sampling.use_non_keyframes and self.cfg.mapping.refinement.iters == 0:
                print(
                    colored(
                        """Warning. If you want to use non-keyframes during Gaussian Rendering Optimization, 
                        make sure that you actually refine the map after running tracking!""",
                        "red",
                    )
                )
        if self.cfg.run_mapping_gui:
            assert self.cfg.run_mapping, colored(
                """If you want to use the Mapping GUI, you also need to run the Mapping process!""",
                "red",
            )
        if self.cfg.run_loop_detection:
            assert self.cfg.run_backend, colored(
                """We only do loop closure optimization in the backend, which optimizes the global map. 
                Use the loop detector always together with the backend enabled!""",
                "red",
            )

        if self.cfg.run_backend and self.cfg.run_mapping:
            if self.cfg.mapper_every > self.cfg.backend_every:
                print(colored("Warning. Mapping is run less often than backend!", "red"))

    def create_out_dirs(self, output_folder: Optional[str] = None) -> None:
        if output_folder is not None:
            self.output = output_folder
        else:
            self.output = "./outputs/"

        os.makedirs(self.output, exist_ok=True)
        os.makedirs(f"{self.output}/evaluation", exist_ok=True)

    def update_cam(self, cfg: DictConfig) -> None:
        """Update the camera intrinsics according to the pre-processing config, such as resize or edge crop"""
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

    def load_pretrained(self, pretrained: str) -> None:
        self.info(f"Load pretrained checkpoint from {pretrained}!")

        state_dict = OrderedDict([(k.replace("module.", ""), v) for (k, v) in torch.load(pretrained).items()])
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)

    def tracking(
        self,
        rank: int,
        communication_lock: mp.Lock,
        stream,
        cond_mapping: mp.Condition,
        sema_backend: mp.Semaphore,
        sema_mapping: mp.Semaphore,
        input_queue: mp.Queue,
        ba_lock: mp.Lock = None,
    ) -> None:
        """Main driver of framework by looping over the input stream"""

        def maybe_notify_other_threads_to_start() -> None:
            # Check to notify other threads that they can start
            if self.cfg.run_backend and self.frontend.count > self.backend_warmup and self.backend_can_start < 1:
                self.backend.info("Backend warmup over!")
                self.backend_can_start += 1
            if self.cfg.run_mapping and self.frontend.count > self.mapping_warmup and self.mapping_can_start < 1:
                self.gaussian_mapper.info("Mapper warmup over!")
                self.mapping_can_start += 1

        def synchronize_with_other_threads(sema_backend: mp.Semaphore, sema_mapping: mp.Semaphore) -> None:
            # Synchronize other parallel threads
            if self.frontend.count % self.backend_freq == 0 and self.frontend.count > (self.backend_warmup + 1):
                sema_backend.release()  # Signal Backend that it can run again
            if self.frontend.count % self.mapping_freq == 0 and self.frontend.count > (self.mapping_warmup + 1):
                sema_mapping.release()  # Signal Mapping that it can run again
                self.mapping_done -= 1  # Reset flag for next Mapping call

        self.info("Frontend tracking thread started!")
        self.all_trigered += 1

        # Wait up for other threads to start
        while self.all_trigered < self.num_running_thread:
            pass

        # Main Loop which drives the whole system
        for frame in tqdm(stream):

            old_count = self.frontend.count  # Memoize current state

            if self.cfg.with_dyn and stream.has_dyn_masks:
                timestamp, image, depth, intrinsic, gt_pose, static_mask = frame
            else:
                timestamp, image, depth, intrinsic, gt_pose = frame
                static_mask = None
            if self.mode not in ["rgbd", "prgbd"]:
                depth = None

            # Transmit the incoming stream to another visualization thread
            if self.cfg.show_stream:
                input_queue.put(image)
                input_queue.put(depth)

            # If the Renderer / Mapping is currently optimizing, wait until that it is finished
            if self.frontend.count > self.mapping_warmup:
                with communication_lock:
                    cond_mapping.wait_for(lambda: self.mapping_done > 0)

            self.frontend(timestamp, image, depth, intrinsic, gt_pose, static_mask=static_mask, lock=ba_lock)

            maybe_notify_other_threads_to_start()
            # Check if we actually inserted a new frame and optimized
            if self.frontend.count > old_count:
                synchronize_with_other_threads(sema_backend, sema_mapping)

        self.info(f"Ran Frontend {self.frontend.count} times!")
        del self.frontend
        torch.cuda.empty_cache()
        gc.collect()

        self.tracking_finished += 1
        self.all_finished += 1
        self.info("Frontend Tracking done!")

        # Release the Semaphores to avoid deadlock
        # HACK Increase the counter just by a lot, so it will never go to 0 again
        for i in range(1000):
            sema_backend.release()
            sema_mapping.release()

    def loop_detection(self, rank: int, loop_queue: mp.Queue, run: bool = False) -> None:
        """Run a loop detector in parallel, which either measures optical flow between the current frame
        and all past frames or visual similarity by comparing feature descriptors.
        When two frames could be the same, we add a bidirectional edge to the backend optimization graph.
        """

        if run:
            assert self.loop_detector is not None, "Loop Detection is not enabled, but we are running it!"
            # Initialize network during worker process, since torch.hub models need to, see https://github.com/Lightning-AI/pytorch-lightning/issues/17637
            if self.loop_detector.method == "eigen" and self.loop_detector.net is None:
                self.loop_detector.net = self.loop_detector.load_eigen()

        self.info("Loop Detection thread started!")
        self.all_trigered += 1

        # Run as long as Frontend tracking gives use new frames
        while self.tracking_finished < 1 and run:
            candidates = self.loop_detector.check()
            if candidates is not None:
                self.loop_detector.info("Sending loop candidates ...")
                loop_queue.put(candidates)

        # Free memory
        del self.loop_detector
        torch.cuda.empty_cache()
        gc.collect()

        self.loop_detection_finished += 1
        self.all_finished += 1
        self.info("Loop Detection done!")

    def get_ram_usage(self) -> Tuple[float, float]:
        free_mem, total_mem = torch.cuda.mem_get_info(device=self.device)
        used_mem = 1 - (free_mem / total_mem)
        return used_mem, free_mem

    def ram_safeguard_backend(self, max_ram: float = 0.9, min_ram: float = 0.5, count_to_set: int = 0) -> None:
        """There are some scenes, where we might get into trouble with memory.
        In order to keep the system going, we simply dont use the backend until we can afford it again.
        """
        used_mem, free_mem = self.get_ram_usage()
        if used_mem > max_ram and self.backend is not None:
            print(colored(f"[Main]: Warning: Deleting Backend due to high memory usage [{used_mem} %]!", "red"))
            print(colored(f"[Main]: Warning: Warning: Got only {free_mem/ 1024 ** 3} GB left!", "red"))
            old_count = self.backend.count
            del self.backend
            self.backend = None
            gc.collect()
            torch.cuda.empty_cache()
            return old_count

        # NOTE chen: if we deleted the backend due to memory issues we likely have not a lot of capacity left for backend
        # only use backend again once we have some slack -> 50% free RAM (12GB in use)
        if self.backend is None and used_mem <= min_ram:
            self.info("Reinstantiating Backend ...")
            self.backend = BackendWrapper(self.cfg, self)
            self.backend.count = count_to_set  # Reset with memoized count
            self.backend.to(self.device)

        return count_to_set

    def get_potential_loop_update(self, loop_queue: mp.Queue):
        """Extract the loop candidates from the Queue and merge them into one set of edges."""
        try:
            new_loops = get_all_queue(loop_queue)  # Empty the whole Queue at once
            new_loops = merge_candidates(new_loops)  # In case we had multiple loop updates
            self.backend.info("Received loop candidates!")
            candidates = clone_obj(new_loops)
            del new_loops
            loop_ii, loop_jj = candidates
        except Exception as e:
            self.info("Could not get anything from the Queue! :(")
            print(colored(e, "red"))
            loop_ii, loop_jj = None, None
        return loop_ii, loop_jj

    def backend_op(
        self,
        add_ii: Optional[torch.Tensor] = None,
        add_jj: Optional[torch.Tensor] = None,
        lock: Optional[mp.Lock] = None,
    ) -> None:
        """Simple wrapper to call backend depending on which method we want to use. Reason being, that we also use the
        frontend factor graph for GO-SLAM's loop aware Bundle Adjustment.
        """
        # Use GO-SLAM's loop closure bundle adjustment (This will also consider edges with very small motion, but only in a loop window)
        if self.backend.enable_loop:
            self.backend(local_graph=self.frontend.optimizer.graph, add_ii=add_ii, add_jj=add_jj, lock=lock)
        else:
            # Use vanilla DROID Graph for Bundle adjustment
            self.backend(add_ii=add_ii, add_jj=add_jj, lock=lock)

    def final_backend_op(
        self,
        t_start=0,
        t_end=-1,
        steps: int = 6,
        add_ii=None,
        add_jj=None,
        n_repeat: int = 2,
        lock: Optional[mp.Lock] = None,
    ) -> None:
        """Make two final refinements over the whole global map. This explicitly calls the optimizer function,
        so this does not have a backend window limit, i.e. this actually runs over the whole map."""
        for i in range(n_repeat):
            # Use vanilla Graph for Bundle adjustment
            _, n_edges = self.backend.optimizer.dense_ba(
                t_start=t_start, t_end=t_end, steps=steps, add_ii=add_ii, add_jj=add_jj, lock=lock
            )
            msg = "Full BA: [{}, {}]; Using {} edges!".format(t_start, t_end, n_edges)
            self.backend.info(msg)

    def global_ba(
        self,
        rank: int,
        ba_lock: mp.Lock,
        sema_backend: mp.Semaphore,
        loop_queue: Optional[mp.Queue] = None,
        run: bool = False,
    ) -> None:
        self.info("Backend thread started!")
        self.all_trigered += 1

        memoized_backend_count = 0
        all_lc_candidates = []
        all_loop_ii, all_loop_jj = None, None

        ### Online Loop
        while self.tracking_finished < 1 and run:
            if self.backend_can_start < 1:
                continue

            sema_backend.acquire()  # Aquire the semaphore (If the counter == 0, then this thread will be blocked)
            sleep(self.sleep_time)  # Let multiprocessing cool down a little bit

            ## Only run backend if we have enough RAM for it
            memoized_backend_count = self.ram_safeguard_backend(
                max_ram=self.max_ram_usage, count_to_set=memoized_backend_count
            )
            # Backend got deactivated due to OOM
            if self.backend is None:
                continue

            ## If we run an additional loop detector -> Pull in visually similar candidate edges as well
            loop_ii, loop_jj = None, None
            if self.cfg.run_loop_detection and loop_queue is not None:
                if not loop_queue.empty():
                    loop_ii, loop_jj = self.get_potential_loop_update(loop_queue)
                    # memoize all loop candidates to always give all candidates to the backend as edges!
                    if loop_ii is not None and loop_jj is not None:
                        all_lc_candidates.append((loop_ii, loop_jj))
                        all_loop_ii, all_loop_jj = merge_candidates(all_lc_candidates)

            ### Actual Backend call
            self.backend_op(add_ii=all_loop_ii, add_jj=all_loop_jj, lock=ba_lock)

        sleep(self.sleep_time)  # Let other threads finish their last optimization
        # Try to instantiate again if needed
        memoized_backend_count = self.ram_safeguard_backend(
            max_ram=self.max_ram_usage, count_to_set=memoized_backend_count
        )
        if run:
            if self.backend is not None:
                self.info(f"Ran Backend {self.backend.count} times before refinement!")
            else:
                self.info(
                    """Backend Refinement does not fit into memory! Please run the system in a 
                    different configuration for this to work. (Maybe use lower resolution)"""
                )

        ### Run one last time after tracking finished
        if run and self.backend is not None and self.backend.do_refinement:
            with self.video.get_lock():
                t_end = self.video.counter.value

                msg = "Optimize full map: [{}, {}]!".format(0, t_end)
                self.backend.info(msg)
                self.final_backend_op(t_start=0, t_end=t_end, add_ii=all_loop_ii, add_jj=all_loop_jj, lock=ba_lock)

        if self.backend is not None:
            del self.backend
            torch.cuda.empty_cache()
            gc.collect()

        self.backend_finished += 1
        self.all_finished += 1
        self.info("Backend done!")

    def maybe_reanchor_gaussians(
        self, pose_thresh: float = 0.001, scale_thresh: float = 0.1, with_scales: bool = False
    ) -> None:
        """Reanchor the Gaussians to follow a big map update.
        For this purpose we simply track the pose changes after a backend optimization."""
        with self.video.get_lock():
            unit = self.video.pose_changes.clone()

        # Reanchor the centers with a 3D transformation
        unit[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.device)
        delta = pose_distance(self.video.pose_changes, unit)
        to_update = (delta > pose_thresh).nonzero().squeeze()  # Check for frames with large updates
        if (
            self.gaussian_mapper is not None
            and self.gaussian_mapper.warmup < self.frontend.optimizer.count
            and len(to_update) > 0
        ):
            self.gaussian_mapper.reanchor_gaussians(to_update, self.video.pose_changes[to_update])
            with self.video.get_lock():
                self.video.pose_changes[to_update] = unit[to_update]  # Reset the pose update

        # Rescale the Gaussians to follow a scale changes (This happens prominently when doing loop closures)
        if with_scales:
            scale_delta = torch.abs(self.video.scale_changes - 1.0)
            to_update = (scale_delta > scale_thresh).nonzero().squeeze()  # Check for frames with large updates
            if (
                self.gaussian_mapper is not None
                and self.gaussian_mapper.warmup < self.frontend.optimizer.count
                and to_update.numel() > 0
            ):
                self.info(f"Rescale Gaussians at {to_update} due to large scale change in map!")
                self.gaussian_mapper.rescale_gaussians(to_update, self.video.scale_changes[to_update])
                with self.video.get_lock():
                    self.video.scale_changes[to_update] = 1.0  # Reset the scale update

    def gaussian_mapping(
        self,
        rank: int,
        communication_lock: mp.Lock,
        cond_mapping: mp.Condition,
        sema_mapping: mp.Semaphore,
        mapping_queue: mp.Queue,
        received_mapping: mp.Event,
        run: bool,
    ) -> None:
        self.info("Gaussian Mapping Triggered!")
        self.all_trigered += 1

        while (self.tracking_finished + self.backend_finished) < 2 and run:

            # Wait for warmup phase to finish
            if self.mapping_can_start < 1:
                continue

            sema_mapping.acquire()  # Aquire the semaphore (If the counter == 0, then this thread will be blocked)
            if self.cfg.run_backend:
                self.maybe_reanchor_gaussians()  # If the backend is also running, we reanchor Gaussians when large map changes occur

            self.gaussian_mapper(mapping_queue, received_mapping)

            # Notify leading Tracking thread, that we finished the current Render optimization
            # -> This avoids both Frontend and Renderer to run at the same time and conflict on depth_video
            # (Notify Tracking thread only when it is still alive, else we get a deadlock)
            if self.tracking_finished < 1:
                with communication_lock:
                    self.mapping_done += 1
                    cond_mapping.notify()

            sleep(self.sleep_time)  # Let system cool off a little

        # Run for one last time after everything finished
        if self.gaussian_mapper is not None and run:
            self.info(f"Ran Gaussian Mapper {self.gaussian_mapper.count} times before Refinement!")

        if self.cfg.run_backend:
            self.maybe_reanchor_gaussians()  # If the backend is also running, we reanchor Gaussians when large map changes occur

        finished = False
        while not finished and run:
            finished = self.gaussian_mapper(mapping_queue, received_mapping, True)

        self.gaussian_mapping_finished += 1
        # Let the user still interact with the GUI
        while self.mapping_visualizing_finished < 1:
            pass

        self.all_finished += 1
        self.info("Gaussian Mapping Done!")

    def visualizing(self, rank: int, run=True) -> None:
        """Vanilla Point Cloud Visualizer in Open3D"""
        self.info("Visualization thread started!")
        self.all_trigered += 1
        finished = False

        while (self.tracking_finished + self.backend_finished < 2) and run and not finished:
            finished = droid_visualization(self.video, device=self.device, save_root=self.output)

        self.visualizing_finished += 1
        self.all_finished += 1
        self.info("Visualization done!")

    def mapping_gui(self, rank: int, run=True) -> None:
        """Gaussian Splatting Visualizer in Open3D"""
        self.info("Mapping GUI thread started!")
        self.all_trigered += 1
        finished = False

        while (self.tracking_finished + self.backend_finished < 2) and run and not finished:
            finished = slam_gui.run(self.params_gui)

        # Wait for Gaussian Mapper to be finished so nothing new is put into the queue anymore
        while self.gaussian_mapping_finished < 1:
            pass

        # empty all the guis that are in params_gui so this will for sure get empty
        if run:  # NOTE Leon: It crashes if we dont check this
            while not self.params_gui.q_main2vis.empty():
                obj = self.params_gui.q_main2vis.get()
                a = clone_obj(obj)
                del obj

        self.mapping_visualizing_finished += 1
        self.all_finished += 1
        self.info("Mapping GUI done!")

    def show_stream(self, rank, input_queue: mp.Queue, run=True) -> None:
        """Show the input RGBD stream (+ confidence of network) in separate windows"""
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
                    # print(colored(e, "red"))
                    pass

            if self.plot_uncertainty:
                # Plot the uncertainty on top
                with self.video.get_lock():
                    t_cur = max(0, self.video.counter.value - 1)
                    if self.cfg.tracking.get("upsample", False):
                        uncertanity_cur = self.video.confidence_up[t_cur].clone()
                    else:
                        uncertanity_cur = self.video.confidence[t_cur].clone()
                uncertainty_img = uncertainty2rgb(uncertanity_cur)[0]
                cv2.imshow("Uncertainty", uncertainty_img[..., ::-1])
                cv2.waitKey(1)

        self.all_finished += 1
        self.info("Show stream Done!")

    def get_cams_for_rendering(
        self,
        stream,
        est_c2w_all_lie: np.ndarray,
        tstamps: List[int],
        kf_tstamps: List[int],
        gaussian_mapper_last_state: Optional[eval_utils.EvaluatePacket] = None,
    ) -> List[Camera]:
        """Go over all [B, 7] lie algebra poses and convert these into renderable Camera objects. If we
        already refined using non-keyframes during GaussianMapper Refinement, then we simply return the
        last state of the Mapper.
        """

        # NOTE this is pretty hacky
        # Tum has not a depth groundtruth for all frames, so we need to use these indices to get the right frames
        # Map the indices to frames with actual groundtruth for TUM, we have a reference depth for each frame
        if "tum" in stream.input_folder and self.mode == "prgbd":
            indices = np.where(np.isin(np.array(self.tum_idx), np.array(self.tum_rgbd_idx)))[0]
            kf_idx = np.where(np.isin(np.array(indices), np.array(kf_tstamps)))[0]
            kf_tstamps = [tstamps[i] for i in kf_idx]
            _, _, nonkf_tstamps = eval_utils.torch_intersect1d(torch.tensor(kf_tstamps), torch.tensor(indices))
            nonkf_tstamps = [int(i) for i in nonkf_tstamps]
            nonkf_idx = np.where(np.isin(np.array(indices), np.array(nonkf_tstamps)))[0]
        else:
            indices = range(len(est_c2w_all_lie))
            _, _, nonkf_tstamps = eval_utils.torch_intersect1d(torch.tensor(kf_tstamps), torch.tensor(tstamps))
            nonkf_tstamps = [int(i) for i in nonkf_tstamps]
            kf_idx = kf_tstamps
            nonkf_idx = nonkf_tstamps

        if self.cfg.mapping.refinement.sampling.use_non_keyframes:
            # FIXME this does not work for tum anymore if we evaluate in rgbd mode, but do inference in another mode :/
            if self.mode == "prgbd" and "tum" in stream.input_folder:
                raise Exception("TUM evaluation does not work with non-keyframes in PRGBD mode right now!")
            all_cams = gaussian_mapper_last_state.cameras
        else:
            all_cams = []
            intrinsics = self.video.intrinsics[0]  # We always have the right global intrinsics stored here
            if self.video.upsample:
                intrinsics = intrinsics * self.video.scale_factor

            for i, idx in tqdm(enumerate(indices)):
                view = est_c2w_all_lie[idx]
                _, gt_image, gt_depth, _, _ = stream[i]

                # c2w -> w2c for initialization
                view = SE3.InitFromVec(view.float().to(device=self.device)).inv().matrix()
                fx, fy, cx, cy = intrinsics
                height, width = gt_image.shape[-2:]
                fovx, fovy = focal2fov(fx, width), focal2fov(fy, height)
                projection_matrix = getProjectionMatrix2(
                    self.gaussian_mapper.z_near, self.gaussian_mapper.z_far, cx, cy, fx, fy, width, height
                )
                projection_matrix = projection_matrix.transpose(0, 1).to(device=self.device)
                new_cam = Camera(
                    i,
                    gt_image.contiguous(),
                    gt_depth,
                    gt_depth,
                    view,
                    projection_matrix,
                    (fx, fy, cx, cy),
                    (fovx, fovy),
                    (height, width),
                    device=self.device,
                )
                all_cams.append(new_cam)

        kf_cams = [all_cams[i] for i in kf_idx]
        nonkf_cams = [all_cams[i] for i in nonkf_idx]
        return kf_cams, nonkf_cams, kf_idx, nonkf_idx

    def evaluate(self, stream, gaussian_mapper_last_state: Optional[eval_utils.EvaluatePacket] = None) -> None:
        """Based on the current state of the video buffer and maybe on the last state of the Gaussian Mapper,
        evaluate the following against the ground truth from the (RGBD-) Dataset:
            i) Tracking metrics for the keyframe poses and all poses. People often only report the keyframe trajectory or both.
            ii) Rendering and depth evaluation for keyframes and non-keyframes. Since the keyframes are always in the training dataset,
            people usually consider the non-keyframe metrics.
        """

        eval_path = os.path.join(self.output, "evaluation")
        self.info("Saving evaluation results in {}".format(eval_path), logger=log)
        self.info("#" * 20 + f" Results for {stream.input_folder} ...", logger=log)

        #### ------------------- ####
        ### Trajectory evaluation ###
        #### ------------------- ####
        if self.cfg.mode in ["prgbd", "mono"]:
            monocular = True
        else:
            monocular = False

        est_c2w_all_lie, eval_traj, kf_tstamps, tstamps = self.get_trajectories(stream, gaussian_mapper_last_state)
        # Evo expects floats for timestamps
        kf_tstamps = [float(i) for i in kf_tstamps]
        tstamps = [float(i) for i in tstamps]
        if stream.poses is not None:
            kf_result_ate, all_result_ate = eval_utils.do_odometry_evaluation(
                eval_path, tstamps=tstamps, kf_tstamps=kf_tstamps, monocular=monocular, **eval_traj
            )
            self.info("(Keyframes only) ATE: {}".format(kf_result_ate), logger=log)
            self.info("(All) ATE: {}".format(all_result_ate), logger=log)
            ### Store main results with attributes for ablation/comparison
            odometry_results = eval_utils.create_odometry_csv(
                kf_result_ate, all_result_ate, self.cfg, stream.input_folder
            )
            df = pd.DataFrame(odometry_results)
            df.to_csv(os.path.join(eval_path, "odometry", "evaluation_results.csv"), index=False)

        else:
            self.info(
                "Warning: Dataset has no ground truth poses available for evaluation! Skipping trajectory evaluation ...",
                logger=log,
            )

        #### ------------------- ####
        ### Rendering  evaluation ###
        #### ------------------- ####
        if self.cfg.run_mapping:
            if self.mode == "prgbd":
                if hasattr(stream, "switch_to_rgbd_gt") and callable(stream.switch_to_rgbd_gt):
                    self.info("Switching to RGBD groundtruth for evaluation ...", logger=log)
                    # TUM only has depth for specific frames, so we will remap the indices to the right frames
                    if "tum" in stream.input_folder:
                        self.tum_idx = stream.indices  # Indices our our frames in prgbd mode
                        stream.switch_to_rgbd_gt()
                        self.tum_rgbd_idx = stream.indices  # Indices of the rgbd frames
                    else:
                        stream.switch_to_rgbd_gt()
                else:
                    self.info(
                        "Warning: No RGBD groundtruth available for evaluation! Using monocular depth predictions ...",
                        logger=log,
                    )

            render_eval_path = os.path.join(eval_path, "rendering")

            gaussians = gaussian_mapper_last_state.gaussians
            render_cfg = gaussian_mapper_last_state.pipeline_params
            background = gaussian_mapper_last_state.background

            self.info(f"Rendering finished with {len(gaussians)} Gaussians!", logger=log)

            # Get the whole trajectory as Camera objects, so we can render them
            kf_tstamps, tstamps = [int(i) for i in kf_tstamps], [int(i) for i in tstamps]
            kf_cams, nonkf_cams, kf_tstamps, nonkf_tstamps = self.get_cams_for_rendering(
                stream, est_c2w_all_lie, tstamps, kf_tstamps, gaussian_mapper_last_state
            )

            ### Evalaute only keyframes, which we overfit to see how good that fit is
            save_dir = os.path.join(render_eval_path, "keyframes")
            kf_rnd_metrics = eval_utils.eval_rendering(
                kf_cams,
                kf_tstamps,
                gaussians,
                stream,
                render_cfg,
                background,
                save_dir,
                1,
                monocular,
                self.render_images,
                self.save_predictions,
            )
            self.info(
                "(Keyframes) Rendering: mean PSNR: {}, SSIM: {}, LPIPS: {}".format(
                    kf_rnd_metrics["mean_psnr"], kf_rnd_metrics["mean_ssim"], kf_rnd_metrics["mean_lpips"]
                ),
                logger=log,
            )

            ### Evalaute on non-keyframes, which we have never seen during training (NOTE this is the proper metric, that people compare in papers)
            save_dir = os.path.join(render_eval_path, "non-keyframes")
            nonkf_rnd_metrics = eval_utils.eval_rendering(
                nonkf_cams,
                nonkf_tstamps,
                gaussians,
                stream,
                render_cfg,
                background,
                save_dir,
                self.save_every,
                monocular,
                self.render_images,
                self.save_predictions,
            )
            self.info(
                "(Non-Keyframes) Rendering: mean PSNR: {}, SSIM: {}, LPIPS: {}".format(
                    nonkf_rnd_metrics["mean_psnr"], nonkf_rnd_metrics["mean_ssim"], nonkf_rnd_metrics["mean_lpips"]
                ),
                logger=log,
            )

            rendering_results = eval_utils.create_rendering_csv(
                kf_rnd_metrics, nonkf_rnd_metrics, self.cfg, stream.input_folder
            )
            # Check if the dataset has depth images
            if len(stream.depth_paths) != 0:
                rendering_results["l1_depth"] = [kf_rnd_metrics["mean_l1"], nonkf_rnd_metrics["mean_l1"]]
            render_df = pd.DataFrame(rendering_results)
            render_df.to_csv(os.path.join(render_eval_path, "evaluation_results.csv"), index=False)

    def get_trajectories(self, stream, gaussian_mapper_last_state: Optional[eval_utils.EvaluatePacket] = None):
        """Get the poses both for the whole video sequence and only the keyframes for evaluation.
        Poses are in format [B, 7, 1] as lie elements.

        NOTE the trajectory filler will create keyframe poses, that slightly deviate due to how the interpolation works (we solve local motion_only BA problems)
        NOTE the poses from GaussianMapper are stored as 4x4 homogeneous matrices. When we convert them back into a Lie algebra the mapping is not unique
            (Example: a 180° rotation can be +180° or -180°, i.e. a sign flip of the quaternion).
            Since the evaluation frame evo, will register the estimated trajectory to the groundtruth with a single SE3 transform
            in a best fit way, the sign does not matter as much and might only make small numerical differences
        """
        from .geom import matrix_to_lie, lie_to_matrix, lie_quat_swap_convention

        # When using Gaussian Mapping w. Refinement, we have already interpolated the trajectory and might get
        # better results than when using only the SLAM system for interpolation
        if (
            self.cfg.run_mapping
            and self.cfg.mapping.refinement.iters > 0
            and self.cfg.mapping.refinement.sampling.use_non_keyframes
        ):
            assert (
                gaussian_mapper_last_state is not None
            ), "Missing GaussianMapper state for evaluation even though we ran Mapping!"
            est_w2c_all_matr = self.gaussian_mapper.get_camera_trajectory(gaussian_mapper_last_state.cameras)
            est_w2c_all_lie = matrix_to_lie(est_w2c_all_matr)
            # Evo expects c2w convention (I think)
            est_c2w_all_lie = SE3.InitFromVec(est_w2c_all_lie).inv().vec()

            ## Get timestamps and make sanity check, that Gaussian Mapper has the same
            kf_ids = torch.tensor(list(gaussian_mapper_last_state.cam2buffer.keys()))
            kf_tstamps = self.video.timestamp[: self.video.counter.value].int().cpu()
            assert (
                (kf_ids == kf_tstamps).all().item()
            ), """Gaussian Mapper should contain the keyframes at the same position as in 
            global timestamps from the Video Buffer after interpolation!"""
            assert len(est_w2c_all_matr) == len(
                stream
            ), """After adding non-keyframes, Gaussian Mapper contain 
            all frames of the whole video stream!"""
            tstamps = np.arange(len(est_w2c_all_matr)).tolist()

            # Take from video directly without interpolation optimization
            est_w2c_kf_lie = vid_w2c_kf_lie = self.video.poses[: self.video.counter.value]
            est_c2w_kf_lie = vid_c2w_kf_lie = SE3.InitFromVec(vid_w2c_kf_lie).inv().vec()
            kf_tstamps = kf_tstamps.tolist()

        else:
            # NOTE chen: even if we have optimized the poses with the GaussianMapper, we would have fed them back
            kf_tstamps = self.video.timestamp[: self.video.counter.value].int().cpu().tolist()
            est_w2c_all, tstamps = self.traj_filler(stream, return_tstamps=True)
            est_c2w_all_lie = est_w2c_all.inv().vec().cpu()  # 7x1 Lie algebra
            # Take from video directly without interpolation optimization
            est_w2c_kf_lie = self.video.poses[: self.video.counter.value]
            est_c2w_kf_lie = SE3.InitFromVec(est_w2c_kf_lie).inv().vec()

        est_c2w_all_lie, est_c2w_kf_lie = est_c2w_all_lie.cpu(), est_c2w_kf_lie.cpu()

        # Evo evaluation package assumes lie algebras to be in form [tx, ty, tz, qw, qx, qy, qz]
        # while lietorch uses [qx, qy, qz, qw, tx, ty, tz]
        traj_eval = {}
        traj_eval["est_c2w_all_lie"] = lie_quat_swap_convention(est_c2w_all_lie.clone()).numpy()
        traj_eval["est_c2w_kf_lie"] = lie_quat_swap_convention(est_c2w_kf_lie).numpy()
        if stream.poses is not None:
            gt_c2w_all_lie = eval_utils.get_gt_c2w_from_stream(stream).float().cpu()
            gt_c2w_kf_lie = gt_c2w_all_lie[kf_tstamps]
            traj_eval["gt_c2w_all_lie"] = lie_quat_swap_convention(gt_c2w_all_lie).numpy()
            traj_eval["gt_c2w_kf_lie"] = lie_quat_swap_convention(gt_c2w_kf_lie).numpy()
        return est_c2w_all_lie, traj_eval, kf_tstamps, tstamps

    def get_all_cams_for_rendering(
        self,
        stream,
        est_c2w_all_lie: np.ndarray,
        gaussian_mapper_last_state: Optional[eval_utils.EvaluatePacket] = None,
    ) -> List[Camera]:
        """Go over all [B, 7] lie algebra poses and convert these into renderable Camera objects. If we
        already refined using non-keyframes during GaussianMapper Refinement, then we simply return the
        last state of the Mapper.
        """

        if self.cfg.mapping.refinement.sampling.use_non_keyframes:
            all_cams = gaussian_mapper_last_state.cameras
        else:
            all_cams = []
            intrinsics = self.video.intrinsics[0]  # We always have the right global intrinsics stored here
            if self.video.upsample:
                intrinsics = intrinsics * self.video.scale_factor

            for i, view in tqdm(enumerate(est_c2w_all_lie)):

                _, gt_image, gt_depth, _, _ = stream[i]
                # c2w -> w2c for initialization
                view = SE3.InitFromVec(view.float().to(device=self.device)).inv().matrix()
                fx, fy, cx, cy = intrinsics
                height, width = gt_image.shape[-2:]
                fovx, fovy = focal2fov(fx, width), focal2fov(fy, height)
                projection_matrix = getProjectionMatrix2(
                    self.gaussian_mapper.z_near, self.gaussian_mapper.z_far, cx, cy, fx, fy, width, height
                )
                projection_matrix = projection_matrix.transpose(0, 1).to(device=self.device)
                new_cam = Camera(
                    i,
                    gt_image.contiguous(),
                    gt_depth,
                    gt_depth,
                    view,
                    projection_matrix,
                    (fx, fy, cx, cy),
                    (fovx, fovy),
                    (height, width),
                    device=self.device,
                )
                all_cams.append(new_cam)
        return all_cams

    def save_state(self) -> None:
        self.info("Saving checkpoints...")
        os.makedirs(os.path.join(self.output, "checkpoints/"), exist_ok=True)
        torch.save(
            {
                "tracking_net": self.net.state_dict(),
                "keyframe_timestamps": self.video.timestamp,
            },
            os.path.join(self.output, "checkpoints/droid.ckpt"),
        )

    def terminate(
        self,
        processes: List[mp.Process],
        stream=None,
        gaussian_mapper_last_state: Optional[eval_utils.EvaluatePacket] = None,
    ) -> None:
        """Evaluate the system and then shut down all Process."""

        if self.video.opt_intr:
            self.info("Final estimated video intrinsics: {}".format(self.video.intrinsics[0]), logger=log)

        self.info("Initiating termination ...", logger=log)
        # self.save_state()  # NOTE we dont save this for now, since the network stays the same
        if self.do_evaluate:
            self.info("Doing evaluation!", logger=log)
            self.evaluate(stream, gaussian_mapper_last_state=gaussian_mapper_last_state)
            self.info("Evaluation complete", logger=log)

        for i, p in enumerate(processes):
            p.terminate()
            p.join()
            self.info("Terminated process {}".format(p.name))
        self.info("Terminate: Done!", logger=log)

    def run(self, stream) -> None:
        """Main SLAM function to manage the multi-threaded application."""
        processes = [
            # NOTE The OpenCV thread always needs to be 0 to work somehow
            mp.Process(target=self.show_stream, args=(0, self.input_pipe, self.cfg.show_stream), name="OpenCV Stream"),
            mp.Process(
                target=self.tracking,
                args=(
                    1,
                    self.communication_lock,
                    stream,
                    self.cond_mapping,
                    self.sema_backend,
                    self.sema_mapping,
                    self.input_pipe,
                    self.ba_lock,
                ),
                name="Frontend Tracking",
            ),
            mp.Process(
                target=self.global_ba,
                args=(2, self.ba_lock, self.sema_backend, self.loop_queue, self.cfg.run_backend),
                name="Backend",
            ),
            mp.Process(
                target=self.loop_detection,
                args=(3, self.loop_queue, self.cfg.run_loop_detection),
                name="Loop Detector",
            ),
            mp.Process(target=self.visualizing, args=(4, self.cfg.run_visualization), name="Visualizing"),
            mp.Process(
                target=self.gaussian_mapping,
                args=(
                    5,
                    self.communication_lock,
                    self.cond_mapping,
                    self.sema_mapping,
                    self.mapping_queue,
                    self.received_mapping,
                    self.cfg.run_mapping,
                ),
                name="Gaussian Mapping",
            ),
            mp.Process(
                target=self.mapping_gui,
                args=(6, self.cfg.run_mapping_gui and self.cfg.run_mapping),
                name="Mapping GUI",
            ),
        ]

        self.num_running_thread[0] += len(processes)
        for p in processes:
            p.start()

        start_time = perf_counter()
        self.info(str(start_time), logger=log)

        # Wait for all processes to have finished before terminating and for final mapping update to be transmitted
        if self.cfg.run_mapping:
            while self.mapping_queue.empty():
                pass
            # Receive the final update, so we can do something with it ...
            a = self.mapping_queue.get()
            self.info("Received final mapping update!", logger=log)
            if a == "None":
                a = deepcopy(a)
                gaussian_mapper_last_state = None
            else:
                a.cameras_to("cpu")  # Put all dense tensors on the CPU before cloning to avoid OOM
                gaussian_mapper_last_state = clone_obj(a)

            del a  # NOTE Always delete receive object from a multiprocessing Queue!
            torch.cuda.empty_cache()
            gc.collect()
            self.received_mapping.set()

        # Let the processes run until they are finished (When using GUI's these need to be closed manually)
        else:
            gaussian_mapper_last_state = None

        if gaussian_mapper_last_state is not None:
            gaussian_mapper_last_state.cameras_to(self.device)

        while self.all_finished < self.num_running_thread:
            pass

        self.info("##########", logger=log)
        end_time = perf_counter()
        self.info("Total time elapsed: {:.2f} minutes".format((end_time - start_time) / 60), logger=log)
        self.info(str(start_time), logger=log)
        self.info(str(end_time), logger=log)
        if (end_time - start_time) > 1e-10:
            self.info("Total FPS: {:.2f}".format(len(stream) / (end_time - start_time)), logger=log)
        self.info("##########", logger=log)

        self.terminate(processes, stream, gaussian_mapper_last_state)
