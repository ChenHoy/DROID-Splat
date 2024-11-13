import ipdb
from typing import Optional, Dict, List
from tqdm import tqdm
from termcolor import colored

import torch
import torch.multiprocessing as mp
import lietorch

import numpy as np

from ..gaussian_splatting.camera_utils import Camera
from ..slam import SLAM
from ..geom import pose_distance, matrix_to_lie
from ..gaussian_splatting.eval_utils import EvaluatePacket
from ..gaussian_splatting.gaussian_renderer import render
from ..gaussian_splatting.gui import gui_utils
from ..test.visu import plot_side_by_side, create_animation, show_img
from ..utils import clone_obj
from ..loop_detection import LoopDetector


class SlamTestbed(SLAM):
    """
    Testing class for SLAM system. This simply acts as a way to debug and test new functionality without running everything in parallel
    """

    def __init__(self, *args, **kwargs):
        super(SlamTestbed, self).__init__(*args, **kwargs)

        self.use_mapping_gui = self.cfg.run_mapping_gui and self.cfg.run_mapping
        if self.use_mapping_gui:
            self.q_main2vis = mp.Queue()
            self.params_gui = gui_utils.ParamsGUI(
                pipe=self.cfg.mapping.pipeline_params,
                background=self.gaussian_mapper.background,
                gaussians=self.gaussian_mapper.gaussians,
                q_main2vis=self.q_main2vis,
            )
            self.gaussian_mapper.q_main2vis = self.q_main2vis
            self.gaussian_mapper.use_gui = True

        if self.cfg.run_loop_detection:
            self.loop_detector = LoopDetector(self.cfg.loop_closure, self.net, self.video, device=self.device)
        else:
            self.loop_detector = None

    def get_render_snapshot(self, view: int, title: Optional[str] = None) -> np.ndarray:
        """Get a snapshot of the current rendering state.
        This renders the image & depth of our Gaussians from a specific camera view and compares it with the
        reference stored in the Camera object.
        """

        renderer = self.gaussian_mapper

        assert view < len(renderer.cameras), "View index out of bounds!"
        cam = renderer.cameras[view]
        render_pkg = render(cam, renderer.gaussians, renderer.pipeline_params, renderer.background, device=self.device)
        image_render, radii, depth_render = render_pkg["render"], render_pkg["radii"], render_pkg["depth"]
        image_render.clamp_(0, 1.0)  # Clip numerical errors

        fig = plot_side_by_side(
            image_render.detach().clone(),
            cam.original_image.detach().clone(),
            depth_render.detach().clone(),
            cam.depth.detach().clone(),
            title=title,
            return_image=True,
        )
        return fig

    def render_and_animate_whole_scene(self):
        """Render all views and animate through the scene to see the reconstructed video from the Gaussians"""

        views = []
        for i in range(len(self.gaussian_mapper.cameras)):
            title = f"View: {i}"
            fig = self.get_render_snapshot(i, title=title)
            views.append(fig)
        create_animation(views, interval=100, repeat_delay=500, blit=True)

    def custom_render_update(
        self, record_optimization: bool = False, record_view: int = 0, iters: Optional[int] = None
    ):
        """Update our rendered map by:
        i) Pull a filtered update from the sparser SLAM map
        ii) Add new Gaussians based on new views
        iii) Run a bunch of optimization steps to update Gaussians and camera poses
        iv) Prune the render map based on visibility

        This is an exact replica of the normal GaussianMapper._update function, but here we can
        record and visualize what happens during each optimization step.

        We do this, so we can directly see
            - Does the optimization actually converge to the correct image and depth?
            - Does the scale change somehow, how we observe it in the normal point cloud viewer?
        """

        assert self.cfg.run_mapping, "Need to run mapping to test rendering"
        renderer = self.gaussian_mapper
        renderer.info("Currently has: {} gaussians".format(len(renderer.gaussians)))

        # Filter map based on multiview_consistency and uncertainty
        self.video.filter_map(
            min_count=renderer.kf_mng_params.filter.mv_count_thresh,
            bin_thresh=renderer.kf_mng_params.filter.bin_thresh,
            unc_threshold=renderer.kf_mng_params.filter.confidence_thresh,
            use_multiview_consistency=renderer.filter_multiview,
            use_uncertainty=renderer.filter_uncertainty,
        )

        renderer.get_new_cameras()  # Add new cameras
        if len(renderer.new_cameras) != 0:
            renderer.last_idx = renderer.new_cameras[-1].uid + 1
            renderer.info(
                f"Added {len(renderer.new_cameras)} new cameras: {[cam.uid for cam in renderer.new_cameras]}"
            )

        renderer.frame_updater()  # Update all changed cameras with new information from SLAM system

        for cam in renderer.new_cameras:
            if not renderer.initialized:
                renderer.initialized = True
                renderer.gaussians.extend_from_pcd_seq(cam, cam.uid, init=True)
                renderer.info(f"Initialized with {len(renderer.gaussians)} gaussians")
            else:
                ng_before = len(renderer.gaussians)
                renderer.gaussians.extend_from_pcd_seq(cam, cam.uid, init=False)

        # We might have 0 Gaussians in some cases, so no need to run optimizer
        if len(renderer.gaussians) == 0:
            renderer.info("No Gaussians to optimize, skipping mapping step ...")
            return

        if record_optimization:
            view_over_time = []
            base_title = f"View: {record_view} | Iter: "
            view_over_time.append(self.get_render_snapshot(record_view, title=base_title + "0"))

        # Optimize gaussians
        if iters is None:
            iters = renderer.mapping_iters

        for iter in tqdm(range(iters), desc=colored("Gaussian Optimization", "magenta"), colour="magenta"):
            frames = renderer.select_keyframes()[0] + renderer.new_cameras
            if len(frames) == 0:
                renderer.loss_list.append(0.0)
                continue

            # Make sure that the view we want to visualize over time is in the list
            if record_optimization:
                to_optimize = [cam.uid for cam in frames]
                if record_view not in to_optimize:
                    frames += [renderer.cameras[record_view]]

            loss = renderer.mapping_step(
                iter, frames, renderer.kf_mng_params.mapping, densify=True, optimize_poses=renderer.optimize_poses
            )
            renderer.loss_list.append(loss / len(frames))

            if record_optimization:
                view_over_time.append(self.get_render_snapshot(record_view, title=base_title + str(iter + 1)))

        # Keep track of how well the Rendering is doing
        print(colored("\n[Gaussian Mapper] ", "magenta"), colored(f"Loss: {renderer.loss_list[-1]}", "cyan"))

        if len(self.iteration_info) % self.kf_mng_params.prune_every == 0:
            if self.kf_mng_params.prune_mode == "abs":
                # Absolute visibility pruning for all gaussians
                self.abs_visibility_prune(self.kf_mng_params.abs_visibility_th)
            elif self.kf_mng_params.prune_mode == "new":
                self.covisibility_pruning()  # Covisibility pruning for recently added gaussians

        # Feed back after pruning
        if renderer.feedback_map:
            to_set = renderer.get_mapping_update(frames)
            renderer.info("Feeding back to Tracking ...")
            self.video.set_mapping_item(**to_set)

        renderer.iteration_info.append(len(renderer.new_cameras))
        # Keep track of added cameras
        renderer.cameras += renderer.new_cameras
        renderer.new_cameras = []

        if record_optimization:
            return view_over_time

    def run_tracking_then_check(self, stream, backend_freq: int = 50, check_at: int = 400) -> None:
        """Sequential processing of the stream, where we run
        i) Tracking + Backend and then stop to run
        ii) Rendering.
        This is to check how well the Gaussian Optimization does depending on how it is initialized and how long we run it.
        We can visualize how well a single run of our Renderer can fit a well-initialized scene.
        """
        for frame in tqdm(stream):

            if self.cfg.with_dyn and stream.has_dyn_masks:
                timestamp, image, depth, intrinsic, gt_pose, static_mask = frame
            else:
                timestamp, image, depth, intrinsic, gt_pose = frame
                static_mask = None
            self.frontend(timestamp, image, depth, intrinsic, gt_pose, static_mask=static_mask)

            if timestamp == check_at:
                self.gaussian_mapper(None, None)

                # Go through the whole scene and show the rendered video
                # self.render_and_animate_whole_scene()
                # Render the scene each iteration of the optimization and show animation of it
                view_over_time = self.custom_render_update(record_optimization=True, record_view=20, iters=100)
                # NOTE chen: matplotlib somehow complains when rendering > 100 frames interactively
                # create_animation(view_over_time, interval=200, repeat_delay=500, blit=True)
                create_animation(
                    view_over_time,
                    output_file="/home/chen/code/discriminative/Reconstruction/sfm/my-go-slam/outputs/test.mp4",
                    interval=200,
                    repeat_delay=500,
                    blit=True,
                )
                ipdb.set_trace()

            if self.frontend.optimizer.is_initialized and timestamp % backend_freq == 0:
                self.backend()

    def get_frame_distance_stats(self):
        """How are distances from loop detector distributed?"""
        d_1st, d_2nd, d_3rd = {}, {}, {}
        for key, val in self.loop_detector.of_distances.items():
            # Filter out invalid distances (this mainly happens in bag of words)
            val = val[~torch.isnan(val)]
            val = val[~torch.isinf(val)]
            d_sorted = torch.sort(val)[0]
            if len(d_sorted) > 0:
                d_1st[key] = d_sorted[0]
            if len(d_sorted) > 1:
                d_2nd[key] = d_sorted[1]
            if len(d_sorted) > 2:
                d_3rd[key] = d_sorted[2]
        return d_1st, d_2nd, d_3rd

    def compare_mapping_video(self, idx: int = 0) -> None:
        """Compare render with video"""

        renderer = self.gaussian_mapper

        assert idx < len(renderer.cameras), "View index out of bounds!"
        cam = renderer.cameras[idx]
        render_pkg = render(cam, renderer.gaussians, renderer.pipeline_params, renderer.background, device=self.device)
        image_render, radii, depth_render = render_pkg["render"], render_pkg["radii"], render_pkg["depth"]
        image_render.clamp_(0, 1.0)  # Clip numerical errors

        idx_in_video = renderer.cam2buffer[idx]
        disps_ref = self.video.disps_up[idx_in_video]
        disps_valid = disps_ref > 0
        depth_ref = torch.where(disps_valid, 1.0 / disps_ref, disps_ref)

        plot_side_by_side(
            image_render.detach().clone(),
            cam.original_image.detach().clone(),
            depth_render.detach().clone(),
            depth_ref.clone(),
            title="Render vs. Tracker",
            return_image=False,
        )

    def run_track_render(self, stream, backend_freq: int = 10, render_freq: int = 10) -> None:
        """Sequential processing of the stream, where we run
            i) Tracking
            ii) Rendering
            iii) Check for Loops
            iv) Global Backend optimization
        one after the other.
        This is to check how well the system performs when we run it sequentially.
        Is the system stable? -> Move on to parallel processing but with a well balanced load.
        """

        def convert_to_tensor(x: Dict) -> torch.Tensor:
            return torch.tensor(list(x.values())).to(self.device)

        assert self.cfg.run_mapping, "Need to run mapping to test rendering"

        for frame in tqdm(stream):
            frontend_old_count = self.frontend.optimizer.t1  # How many times did the frontend actually run?

            if self.cfg.with_dyn and stream.has_dyn_masks:
                timestamp, image, depth, intrinsic, gt_pose, static_mask = frame
            else:
                timestamp, image, depth, intrinsic, gt_pose = frame
                static_mask = None

            # Control when to start and when to stop the SLAM system from outside
            if timestamp < self.t_start:
                continue
            if self.t_stop is not None and timestamp > self.t_stop:
                break

            # Frontend insert new frames
            self.frontend(timestamp, image, depth, intrinsic, gt_pose, static_mask=static_mask)

            # If new keyframe got inserted
            if frontend_old_count != self.frontend.optimizer.t1:
                # Render all new incoming frames
                if (
                    self.frontend.optimizer.is_initialized
                    and self.frontend.optimizer.t1 % render_freq == 0
                    and self.gaussian_mapper.warmup < self.frontend.optimizer.count
                ):
                    self.gaussian_mapper(None, None)

                # Run backend and loop closure detection occasianally
                if self.frontend.optimizer.is_initialized and self.frontend.optimizer.t1 % backend_freq == 0:
                    if self.loop_detector is not None:
                        loop_candidates = self.loop_detector.check()
                        if loop_candidates is not None:
                            loop_ii, loop_jj = loop_candidates
                        else:
                            loop_ii, loop_jj = None, None
                        self.backend(add_ii=loop_ii, add_jj=loop_jj)
                    else:
                        self.backend()
                    self.backend()

                    ## Reanchor the Gaussians to follow the big map update
                    threshold = 0.05
                    unit = self.video.pose_changes.clone()
                    unit[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.device)
                    delta = pose_distance(self.video.pose_changes, unit)
                    to_update = (delta > threshold).nonzero().squeeze()
                    if self.gaussian_mapper.warmup < self.frontend.optimizer.count and to_update.ndim > 0:
                        self.gaussian_mapper.reanchor_gaussians(to_update, self.video.pose_changes[to_update])

        self.gaussian_mapper._update(iters=self.gaussian_mapper.mapping_iters + 10, delay_to_tracking=False)
        self.gaussian_mapper._last_call(None, None)

        ipdb.set_trace()

        # self.compare_mapping_video(0)

        # Check distance statistics, so you can select a good threshold depending on the Place Recognition Network
        if self.loop_detector is not None:
            d_1st, d_2nd, d_3rd = self.get_frame_distance_stats()
            d_1st = convert_to_tensor(d_1st)
            d_2nd, d_3rd = convert_to_tensor(d_2nd), convert_to_tensor(d_3rd)

    def test_tracking(self, stream, backend_freq: int = 10) -> None:
        """Test tracking in a sequential manner."""
        assert self.cfg.run_backend, "Need to run backend for this test to make sense!"

        for frame in tqdm(stream):
            frontend_old_count = self.frontend.optimizer.t1  # How many times did the frontend actually run?

            if self.cfg.with_dyn and stream.has_dyn_masks:
                timestamp, image, depth, intrinsic, gt_pose, static_mask = frame
            else:
                timestamp, image, depth, intrinsic, gt_pose = frame
                static_mask = None

            # Control when to start and when to stop the SLAM system from outside
            if timestamp < self.t_start:
                continue
            if self.t_stop is not None and timestamp > self.t_stop:
                break

            # Frontend insert new frames
            self.frontend(timestamp, image, depth, intrinsic, gt_pose, static_mask=static_mask)

            if (
                self.frontend.optimizer.is_initialized
                and self.frontend.optimizer.t1 % backend_freq == 0
                and self.cfg.run_backend
            ):
                self.backend()

    def test_rendering(self, stream, render_freq: int = 10, test_until: Optional[int] = None) -> None:
        """Test Rendering in a sequential manner."""
        i = 0
        assert self.cfg.run_mapping, "Need to run mapping to test rendering"

        for frame in tqdm(stream):
            frontend_old_count = self.frontend.optimizer.t1  # How many times did the frontend actually run?

            if self.cfg.with_dyn and stream.has_dyn_masks:
                timestamp, image, depth, intrinsic, gt_pose, static_mask = frame
            else:
                timestamp, image, depth, intrinsic, gt_pose = frame
                static_mask = None

            # Control when to start and when to stop the SLAM system from outside
            if timestamp < self.t_start:
                continue
            if self.t_stop is not None and timestamp > self.t_stop:
                break
            # HACK If config was not set, overwrite here
            if test_until is not None and timestamp > test_until:
                break

            # Frontend insert new frames
            self.frontend(timestamp, image, depth, intrinsic, gt_pose, static_mask=static_mask)
            if (
                self.frontend.optimizer.is_initialized
                and self.frontend.optimizer.t1 % render_freq == 0
                and self.cfg.run_mapping
                and i > self.gaussian_mapper.warmup
            ):
                self.gaussian_mapper(None, None)
            i += 1

        self.gaussian_mapper(None, None, True)

    # TODO
    def test_nonkeyframes_mapping(self, stream, render_freq: int = 2, test_until: int = 200) -> None:
        """We observe differences when doing a manual trajectory interpolation and when we do it in Gaussian Mapper.
        Test here why the back and forth does not work correctly ...
        """

        def get_poses_from_cameras(cameras: List[Camera]) -> torch.Tensor | List:
            poses = []
            for cam in cameras:
                pose = torch.eye(4)
                pose[:3, :3] = cam.R
                pose[:3, 3] = cam.T
                poses.append(pose)
            poses = torch.stack(poses)
            return poses

        # Overwrite Config to force setup
        self.gaussian_mapper.refinement_iters = 0  # Make sure we never refine during last_call to do it manually

        # Track & Render until early stopping
        self.test_rendering(stream, render_freq=render_freq, test_until=test_until)
        renderer = self.gaussian_mapper

        ####
        # 1. Get the trajectory of all cameras on the conventional way
        ####
        # Since we do early stopping, only look at interpolated poses until last keyframe
        last_kf = self.video.timestamp[self.video.counter.value - 1]
        w2c_all, tstamps = self.traj_filler(stream, return_tstamps=True)

        w2c_all, tstamps = w2c_all[: int(last_kf) + 1], tstamps[: int(last_kf) + 1]
        c2w_all_lie = w2c_all.inv().vec().cpu()  # 7x1 Lie algebra

        ####
        # 2. Get the trajectory inside the Gaussian Mapper
        ####
        # 0. Test: are non_kf_cams_w2c_lie and w2c_all the same for the non-keyframes? -> yes!
        non_kf_cams, non_kf_cams_w2c_lie = renderer.get_nonkeyframe_cameras(stream, self.traj_filler)
        # 1. Test: if we get the pose from the actual Camera object, will this be the same? -> yes!
        test = get_poses_from_cameras([non_kf_cams[0]])  # Is lie(test) == w2c_all[non_kf_cams[0].uid]?

        # Add these Cameras to kf_cams in self.cameras and reassing id's
        new_mapping, masked_mapping = {}, {}
        ## Change id's of existing keyframe to global timestamp in video
        for cam in renderer.cameras:
            new_id = int(self.video.timestamp[cam.uid].item())
            masked_mapping[cam.uid] = (new_id, renderer.gaussians.unique_kfIDs == cam.uid)
            new_mapping[new_id] = cam.uid  # Memoize from timestamp to old video index
            cam.uid = new_id  # Reassign local keyframe ids to global stream ids

        self.cam2buffer = new_mapping
        # Update the keyframe ids for each Gaussian, so they fit the new global uid
        for key, val in masked_mapping.items():
            new_id, mask = val
            renderer.gaussians.unique_kfIDs[mask] = new_id

        rnd_w2c_kf = get_poses_from_cameras(renderer.cameras)
        rnd_w2c_kf_lie = [matrix_to_lie(matr) for matr in rnd_w2c_kf]  # 7x1 Lie algebra
        # Sanity check: Compare these with self.video.poses[:len(renderer.cameras)]
        w2c_kf_lie = self.video.poses[: self.video.counter.value]  # 7x1 Lie algebra
        # 3. test: are these the same? -> yes!

        renderer.cameras += non_kf_cams  # Add to set of cameras
        # Reorder according to global uid
        renderer.cameras = sorted(renderer.cameras, key=lambda x: x.uid)
        # Test: this should give as all cameras sorted and with the right 4x4 positions we also have in w2c_all
        rnd_w2c_all = get_poses_from_cameras(renderer.cameras)
        rnd_w2c_all = rnd_w2c_all[: int(last_kf) + 1]  # Only look at interpolated poses until last keyframe
        # 3. Compare -> Same

    def test_rescale(
        self, stream, backend_freq: int = 2, render_freq: int = 5, test_until: Optional[int] = None
    ) -> None:
        assert self.cfg.run_backend, "Need to run backend for this test to make sense!"

        for frame in tqdm(stream):
            frontend_old_count = self.frontend.optimizer.t1  # How many times did the frontend actually run?

            if self.cfg.with_dyn and stream.has_dyn_masks:
                timestamp, image, depth, intrinsic, gt_pose, static_mask = frame
            else:
                timestamp, image, depth, intrinsic, gt_pose = frame
                static_mask = None

            # Control when to start and when to stop the SLAM system from outside
            if timestamp < self.t_start:
                continue
            if self.t_stop is not None and timestamp > self.t_stop:
                break

            # Frontend insert new frames
            self.frontend(timestamp, image, depth, intrinsic, gt_pose, static_mask=static_mask)

            if (
                self.frontend.optimizer.is_initialized
                and self.frontend.optimizer.t1 % backend_freq == 0
                and self.cfg.run_backend
            ):
                self.backend()

            if (
                self.frontend.optimizer.is_initialized
                and self.frontend.optimizer.t1 % render_freq == 0
                and self.cfg.run_mapping
                and frontend_old_count > self.gaussian_mapper.warmup
            ):
                self.maybe_reanchor_gaussians()  # If the backend is also running, we reanchor Gaussians when large map changes occur
                self.gaussian_mapper(None, None)

    def run(self, stream):
        """Test the system by running any function dependent on the input stream directly so we can set breakpoints for inspection."""

        processes = []
        if self.cfg.run_visualization:
            processes.append(
                mp.Process(target=self.visualizing, args=(1, self.cfg.run_visualization), name="Visualizing"),
            )
        if self.use_mapping_gui:
            processes.append(mp.Process(target=self.mapping_gui, args=(2, self.use_mapping_gui), name="Mapping GUI"))
        self.num_running_thread[0] += len(processes)
        for p in processes:
            p.start()

        render_freq = 10  # Run rendering every k frontends
        backend_freq = 5  # Run backend every 5 frontends

        # self.run_tracking_then_check(stream, backend_freq=backend_freq, check_at=200) # Check how much Rendering can overfit, when initialized correctly
        # self.test_tracking(stream, backend_freq=backend_freq) # Check how tracking works sequentially
        # self.test_rendering(stream, render_freq=render_freq) # Check how Renderer works on top of Tracker sequentially
        # Check the add_nonkeyframe functionality and trajectory interpolation in GaussianMapper
        # self.test_nonkeyframes_mapping(stream)

        # Check how Renderer works on top of all Tracker components sequentially
        # self.run_track_render(
        #     stream, backend_freq=backend_freq, render_freq=render_freq
        # )

        # Check if reanchoring/rescaling works correctly
        self.test_rescale(stream, backend_freq=backend_freq, render_freq=render_freq)
        self.terminate(processes, stream, None)
