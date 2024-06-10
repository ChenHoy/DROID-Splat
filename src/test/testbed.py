import ipdb
from typing import Optional
from tqdm import tqdm

import torch
import torch.multiprocessing as mp

import numpy as np

from ..slam import SLAM
from ..gaussian_splatting.gaussian_renderer import render
from ..test.visu import plot_side_by_side, create_animation
from ..gaussian_splatting.gui import gui_utils, slam_gui
from ..gaussian_splatting.multiprocessing_utils import clone_obj


class SlamTestbed(SLAM):
    """
    Testing class for SLAM system. This simply acts as a way to debug and test new functionality without running everything in parallel
    """

    def __init__(self, *args, **kwargs):
        super(SlamTestbed, self).__init__(*args, **kwargs)

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
                # FIXME why do we get so few gaussians right now from the cameras?!
                # i.e. even when adding 20 views we only initialize with 4k Gaussians
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
        for iter in range(iters):
            frames = renderer.select_keyframes()[0] + renderer.new_cameras
            if len(frames) == 0:
                renderer.loss_list.append(0.0)
                continue

            # Make sure that the view we want to visualize over time is in the list
            if record_optimization:
                to_optimize = [cam.uid for cam in frames]
                if record_view not in to_optimize:
                    frames += renderer.cameras[record_view]

            loss = renderer.mapping_step(
                iter, frames, renderer.kf_mng_params.mapping, densify=True, optimize_poses=renderer.optimize_poses
            )
            renderer.loss_list.append(loss / len(frames))

            if record_optimization:
                view_over_time.append(self.get_render_snapshot(record_view, title=base_title + str(iter + 1)))

        # Keep track of how well the Rendering is doing
        renderer.info(f"Loss: {renderer.loss_list[-1]}")

        if len(renderer.iteration_info) % 1 == 0:
            if renderer.kf_mng_params.prune_mode == "abs":
                # Absolute visibility pruning for all gaussians
                renderer.abs_visibility_prune(renderer.kf_mng_params.abs_visibility_th)
            elif renderer.kf_mng_params.prune_mode == "new":
                renderer.covisibility_pruning()  # Covisibility pruning for recently added gaussians

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

    def run(self, stream):
        """Test the system by running any function dependent on the input stream directly so we can set breakpoints for inspection."""

        processes = []
        if self.cfg.run_visualization:
            processes.append(
                mp.Process(target=self.visualizing, args=(1, self.cfg.run_visualization), name="Visualizing"),
            )
        use_mapping_gui = self.cfg.run_mapping_gui and self.cfg.run_mapping and not self.cfg.evaluate
        if use_mapping_gui:
            processes.append(mp.Process(target=self.mapping_gui, args=(2, use_mapping_gui), name="Mapping GUI"))
        self.num_running_thread[0] += len(processes)
        for p in processes:
            p.start()

        render_freq = 10
        backend_freq = 50

        for frame in tqdm(stream):
            timestamp, image, depth, intrinsic, gt_pose = frame
            self.frontend(timestamp, image, depth, intrinsic, gt_pose)

            if timestamp == 400:
                self.gaussian_mapper(None, None)
                # Go through the whole scene and show the rendered video
                self.render_and_animate_whole_scene()
                self.gaussian_mapper(None, None)
                self.render_and_animate_whole_scene()
                self.gaussian_mapper(None, None)

                # Render the scene each iteration of the optimization and show animation of it
                view_over_time = self.custom_render_update(record_optimization=True, record_view=10)
                create_animation(view_over_time, interval=200, repeat_delay=500, blit=True)
                ipdb.set_trace()

            # Run Gaussian Rendering on top, but only every k frames
            # if self.frontend.optimizer.is_initialized and timestamp % render_freq == 0:
            #     self.gaussian_mapper(None, None)

            # if timestamp == 400:
            #     self.render_and_animate_gaussians()

            if self.frontend.optimizer.is_initialized and timestamp % backend_freq == 0:
                self.backend()

        ipdb.set_trace()

        self.backend()
        self.gaussian_mapper._update()
        self.gaussian_mapper._last_call(None, None)

        while True:
            pass