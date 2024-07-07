import os
import ipdb
from copy import deepcopy
from typing import List, Dict, Optional
import math
import gc
from termcolor import colored
from tqdm import tqdm

import torch
import torch.multiprocessing as mp

import numpy as np
import cv2
import matplotlib.pyplot as plt

from .gaussian_splatting.gui import gui_utils
from .gaussian_splatting.eval_utils import EvaluatePacket
from .gaussian_splatting.gaussian_renderer import render
from .gaussian_splatting.scene.gaussian_model import GaussianModel
from .gaussian_splatting.camera_utils import Camera
from .losses import mapping_rgbd_loss, plot_losses

from .gaussian_splatting.pose_utils import update_pose
from .gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, focal2fov
from .utils.multiprocessing_utils import clone_obj
from .trajectory_filler import PoseTrajectoryFiller


"""
Mapping based on 3D Gaussian Splatting. 
We create new Gaussians based on incoming new views and optimize them for dense photometric consistency. 
Since they are initialized with the 3D locations of a VSLAM system, this process converges really fast. 

NOTE this could a be standalone SLAM system itself, but we use it on top of an optical flow based SLAM system.
"""


class GaussianMapper(object):
    """
    SLAM from Rendering with 3D Gaussian Splatting.
    """

    def __init__(self, cfg, slam, gui_qs=None):
        self.cfg = cfg
        self.slam = slam
        self.video = slam.video
        self.device = cfg.device
        self.mode = cfg.mode
        self.evaluate = cfg.evaluate
        self.output = slam.output
        self.delay = cfg.mapping.delay  # Delay between tracking and mapping
        self.warmup = max(cfg.mapping.warmup, cfg.tracking.warmup)
        self.batch_mode = cfg.mapping.batch_mode  # Take a batch of all unupdated frames at once
        # Feedback the map to the Tracker if wanted
        self.feedback_disps = cfg.mapping.feedback_disps
        if self.feedback_disps:
            self.info("Feeding back scene geometry from Renderer -> Tracker!")
        self.optimize_poses = cfg.mapping.optimize_poses
        self.feedback_poses = cfg.mapping.feedback_poses
        if self.feedback_poses:
            self.info("Feeding back local pose graph from Renderer -> Tracker!")
        if self.feedback_poses and not self.optimize_poses:
            self.info("Warning. You are feeding back poses from Mapper to Tracker without optimizing them!")
        self.feedback_only_last_window = cfg.mapping.feedback_only_last_window
        self.feedback_warmup = cfg.mapping.feedback_warmup

        self.refinement_iters = cfg.mapping.refinement_iters
        self.use_non_keyframes = cfg.mapping.use_non_keyframes
        self.mapping_iters = cfg.mapping.mapping_iters
        self.save_renders = cfg.mapping.save_renders
        self.opt_params = cfg.mapping.opt_params
        self.pipeline_params = cfg.mapping.pipeline_params
        self.kf_mng_params = cfg.mapping.keyframes
        # Always take n last frames from current index and optimize additional random frames globally
        self.n_last_frames = self.kf_mng_params.n_last_frames
        self.n_rand_frames = self.kf_mng_params.n_rand_frames
        # NOTE chen: during refinement we likely go over this number!
        # in order to avoid OOM, we chunk the refinement up
        self.max_frames_refinement = cfg.mapping.max_frames_refinement  # Maximum number of frames to optimize over

        self.filter_uncertainty = cfg.mapping.filter_uncertainty
        self.filter_multiview = cfg.mapping.filter_multiview
        self.supervise_with_prior = cfg.mapping.use_prior_for_supervision

        self.filter_dyn = cfg.get("with_dyn", False)

        self.loss_params = cfg.mapping.loss

        self.sh_degree = 3 if cfg.mapping.use_spherical_harmonics else 0

        # Change the downsample factor for initialization depending on cfg.tracking.upsample, so we always have points
        if not self.cfg.tracking.upsample:
            cfg.mapping.input.pcd_downsample_init /= 8
            cfg.mapping.input.pcd_downsample /= 8
        self.gaussians = GaussianModel(self.sh_degree, config=cfg.mapping.input)
        self.gaussians.init_lr(self.opt_params.init_lr)
        self.gaussians.training_setup(self.opt_params)

        bg_color = [1, 1, 1]  # White background
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.z_near = 0.0001
        self.z_far = 10000.0

        if gui_qs is not None:
            self.q_main2vis = gui_qs
            self.use_gui = True
        else:
            self.use_gui = False

        self.iteration_info = []

        self.cameras = []
        self.new_cameras = []
        self.loss_list = []
        self.last_idx = 0
        self.initialized = False
        self.projection_matrix = None

        # Memoize the mapping of Gaussian indices to video indices
        self.idx_mapping = {}
        self.count = 0

    def info(self, msg: str):
        print(colored("[Gaussian Mapper] " + msg, "magenta"))

    def save_render(self, cam: Camera, render_path: str) -> None:
        """Save a rendered frame"""
        render_pkg = render(cam, self.gaussians, self.pipeline_params, self.background, device=self.device)
        rgb = np.uint8(255 * render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(render_path, bgr)

    def camera_from_video(self, idx):
        """Extract Camera objects from a part of the video."""
        if self.video.disps_clean[idx].sum() < 1:  # Sanity check:
            self.info(f"Warning. Trying to intialize from empty frame {idx}!")
            return None

        color, depth, depth_prior, intrinsics, c2w, stat_mask = self.video.get_mapping_item(idx, self.device)
        w2c = c2w.inv().matrix()
        return self.camera_from_frame(idx, color, w2c, intrinsics, depth, mask=stat_mask)

    def camera_from_frame(
        self,
        idx: int,
        image: torch.Tensor,
        w2c: torch.Tensor,
        intrinsics: torch.Tensor,
        depth_init: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """Given the image, depth, intrinsic and pose, creates a Camera object.
        The depth for supervision and initialization does not need to be the same, e.g. we could initialize
        the Gaussians with a sparse, but certain depth map and supervise with a dense prior.
        We also use an optional mask for the objective function, e.g. for supervising only the static parts of the scene
        explainable by the camera motion."""
        fx, fy, cx, cy = intrinsics

        height, width = image.shape[-2:]
        fovx, fovy = focal2fov(fx, width), focal2fov(fy, height)
        projection_matrix = getProjectionMatrix2(self.z_near, self.z_far, cx, cy, fx, fy, width, height)
        projection_matrix = projection_matrix.transpose(0, 1).to(device=self.device)

        return Camera(
            idx,
            image.contiguous(),
            depth_init,
            depth,
            w2c,
            projection_matrix,
            (fx, fy, cx, cy),
            (fovx, fovy),
            (height, width),
            device=self.device,
            mask=mask,
        )

    def get_new_cameras(self, delay=0):
        """Get all new cameras from the video."""
        # Only add a batch of cameras in batch_mode
        if self.batch_mode:
            to_add = range(self.last_idx, self.cur_idx - delay)
            to_add = to_add[: self.kf_mng_params.default_batch_size]
        else:
            to_add = range(self.last_idx, self.cur_idx - delay)

        for idx in to_add:
            color, depth, depth_prior, intrinsics, c2w, stat_mask = self.video.get_mapping_item(
                idx, device=self.device
            )
            w2c = c2w.inv().matrix()
            # HOTFIX Sanity check for when we dont have any good depth
            if (depth > 0).sum() == 0:
                depth = None

            cam = self.camera_from_frame(
                idx, color, w2c, intrinsics, depth_init=depth, depth=depth_prior, mask=stat_mask
            )

            # get the uid's into the self.idx_mapping
            if cam.uid not in self.idx_mapping:
                self.idx_mapping[cam.uid] = cam.uid
            self.new_cameras.append(cam)

    def get_pose_optimizer(self, frames: List):
        """Creates an optimizer for the camera poses for all provided frames."""
        opt_params = []
        for cam in frames:
            opt_params.append(
                {"params": [cam.cam_rot_delta], "lr": self.opt_params.cam_rot_delta, "name": "rot_{}".format(cam.uid)}
            )
            opt_params.append(
                {
                    "params": [cam.cam_trans_delta],
                    "lr": self.opt_params.cam_trans_delta,
                    "name": "trans_{}".format(cam.uid),
                }
            )
            opt_params.append({"params": [cam.exposure_a], "lr": 0.01, "name": "exposure_a_{}".format(cam.uid)})
            opt_params.append({"params": [cam.exposure_b], "lr": 0.01, "name": "exposure_b_{}".format(cam.uid)})

        return torch.optim.Adam(opt_params)

    def frame_updater(self, delay=0):
        """Gets the list of frames and updates the depth and pose based on the video.

        NOTE: This assumes, that optical flow based tracking overall is more reliable for sparse supervision
        We only use the Renderer for great scene representation and potential densification / correction
        """
        all_cameras = self.cameras + self.new_cameras
        all_idxs = torch.tensor([cam.uid for cam in all_cameras]).long().to(self.device)

        with self.video.get_lock():
            (dirty_index,) = torch.where(self.video.mapping_dirty.clone())
            dirty_index = dirty_index[dirty_index < self.cur_idx - delay]
        # Only update already inserted cameras
        to_update = dirty_index[torch.isin(dirty_index, all_idxs)]

        for idx in to_update:
            # TODO can the stat_mask potentially change as well?
            color, depth, depth_prior, intrinsics, c2w, stat_mask = self.video.get_mapping_item(
                idx, device=self.device
            )
            w2c = c2w.inv().matrix()
            cam = all_cameras[idx]
            R = w2c[:3, :3].unsqueeze(0).detach()
            T = w2c[:3, 3].detach()
            # update intrinsics in case we use opt_intrinsics
            cam.update_intrinsics(intrinsics, color.shape[-2:], self.z_near, self.z_far)
            if self.mode == "prgbd":
                cam.depth_prior = depth_prior.detach()  # Update prior in case of scale_change
            cam.depth = depth.detach()
            cam.update_RT(R, T)

        self.video.mapping_dirty[to_update] = False

    def get_nonkeyframe_cameras(self, stream, trajectory_filler, batch_size: int = 16) -> List[Camera]:
        """SLAM systems operate on keyframes. This is good enough to build a good map,
        but we can get even finer details when including intermediate keyframes as well!
        Since we dont store all frames when iterating over the datastream, this requires
        reiterating over the stream again and interpolating between poses.

        NOTE Use this only after tracking finished for refining!
        """
        all_poses, all_timestamps = trajectory_filler(stream, batch_size=batch_size, return_tstamps=True)
        # NOTE chen: this assumes that cam.uid will correspond to the position in the video buffer
        # this will later change, where cam.uid will simply correspond to the global timestamp after adding all the cameras
        already_mapped = [self.video.timestamp[cam.uid] for cam in self.cameras]
        s = self.video.scale_factor
        if self.video.upsampled:
            intrinsics = self.video.intrinsics[0].to(self.device) * s
        else:
            intrinsics = self.video.intrinsics[0].to(self.device)

        new_cams = []
        for w2c, timestamp in tqdm(zip(all_poses, all_timestamps)):
            if timestamp in already_mapped:
                continue

            w2c = w2c.matrix()  # [4, 4]
            # color = stream._get_image(timestamp).squeeze(0).permute(1, 2, 0).contiguous().to(self.device)
            color = stream._get_image(timestamp).clone().squeeze(0).contiguous().to(self.device)
            if not self.video.upsampled:
                color = color[..., int(s // 2 - 1) :: s, int(s // 2 - 1) :: s]
            cam = self.camera_from_frame(timestamp, color, w2c, intrinsics)
            new_cams.append(cam)

        return new_cams

    def reanchor_gaussians(self, indices: torch.Tensor | List[int], delta_pose: torch.Tensor):
        """After a large map change, we need to reanchor the Gaussians. For this purpose we simply measure the
        rel. pose change for indidividual frames and check for large updates. We can then simply apply the rel. transform
        to the respective Gaussians.
        """
        # NOTE chen: we always update before optimization anyways, but just to be sure and to immediately have the right visuals in GUI
        self.video.filter_map(
            min_count=self.kf_mng_params.filter.mv_count_thresh,
            bin_thresh=self.kf_mng_params.filter.bin_thresh,
            unc_threshold=self.kf_mng_params.filter.confidence_thresh,
            use_multiview_consistency=self.filter_multiview,
            use_uncertainty=self.filter_uncertainty,
        )
        self.frame_updater(delay=self.delay)  # Update all changed cameras with new information from SLAM system

        updated_cams = []
        # TODO chen: implement this in batch mode
        for idx, pose in zip(indices, delta_pose):
            # We have never mapped this frame before
            if int(idx) not in self.idx_mapping:
                continue
            else:
                # Go completely overboard with the reanchoring, so we can see the visuals of a map change
                self.gaussians.reanchor(int(idx), pose)
                # NOTE chen: We append to our camera list in consecutive order, i.e. this should normally be sorted!
                # this is not a given though! be cautious, e.g. during refinement the list changes due to insertion of non-keyframes
                cam = self.cameras[int(idx)]
                updated_cams.append(cam)  # add the kf from self.cameras[idx] to updated cameras for GUI

        if self.use_gui:
            self.q_main2vis.put_nowait(
                gui_utils.GaussianPacket(
                    gaussians=clone_obj(self.gaussians),
                    keyframes=updated_cams,
                )
            )

    def map_refinement(
        self,
        num_iters: int = 100,
        optimize_poses: bool = False,
        densify: bool = False,
        prune_densify: bool = False,
        random_frames: Optional[float] = None,
        kf_at_least: Optional[float] = None,
    ) -> None:
        """Refine the map with color only optimization. Instead of going over last frames, we always select random frames from the whole map."""

        if self.use_gui:
            self.q_main2vis.put_nowait(
                gui_utils.GaussianPacket(
                    gaussians=clone_obj(self.gaussians),
                    keyframes=self.cameras,
                )
            )

        # Optimize over a random subset of all frames
        if random_frames is not None:
            n_frames = int(len(self.cameras) * random_frames)
            self.info(
                f"Info. Going over {n_frames} random frames instead of {len(self.cameras)} of frames for optimization ..."
            )
        else:
            n_frames = len(self.cameras)
        # Divide the frames we want to optimize into handable chunks if too many
        if n_frames > self.max_frames_refinement:
            n_chunks = math.ceil(n_frames / self.max_frames_refinement)
            self.info(
                f"Warning. {n_frames} Frames is too many! Optimizing over {n_chunks} chunks of frames with size {self.max_frames_refinement} ..."
            )

        for iter in tqdm(range(num_iters), desc=colored("Gaussian Refinement", "magenta"), colour="magenta"):
            # Select a random subset of frames to optimize over
            if random_frames is not None:
                total_rand = int(len(self.cameras) * random_frames)
                kf_cams = [cam for cam in self.cameras if cam.uid in self.idx_mapping]
                non_kf_cams = [cam for cam in self.cameras if cam.uid not in self.idx_mapping]
                # make sure that at least k% kf are in each chunk
                if kf_at_least is not None and len(non_kf_cams) > 0:
                    kf_rand = int(total_rand * kf_at_least)
                    to_refine_kf = np.random.choice(len(kf_cams), kf_rand, replace=False)
                    kf_frames = [kf_cams[i] for i in to_refine_kf]
                    nonkf_rand = total_rand - kf_rand
                    to_refine_nonkf = np.random.choice(len(non_kf_cams), nonkf_rand, replace=False)
                    nonkf_frames = [non_kf_cams[i] for i in to_refine_nonkf]
                    frames = kf_frames + nonkf_frames
                else:
                    to_refine = np.random.choice(len(self.cameras), total_rand, replace=False)
                    frames = [self.cameras[i] for i in to_refine]
            else:
                frames = self.cameras

            # Using all frames instead of only keyframes can lead to OOM during optimization
            # -> Use a selection of frames instead!
            if len(frames) > self.max_frames_refinement:
                chunks = [
                    frames[i : i + self.max_frames_refinement]
                    for i in range(0, len(frames), self.max_frames_refinement)
                ]
                loss = 0
                for chunk in chunks:
                    loss += self.mapping_step(
                        iter,
                        chunk,
                        self.kf_mng_params.refinement,
                        densify=densify,
                        prune_densify=prune_densify,
                        optimize_poses=optimize_poses,
                        lr_factor=0.1,  # Force lower learning rate for refinement
                    )
            else:
                loss = self.mapping_step(
                    iter,
                    frames,
                    self.kf_mng_params.refinement,
                    densify=densify,
                    prune_densify=prune_densify,
                    optimize_poses=optimize_poses,
                    lr_factor=0.1,  # Force lower learning rate for refinement
                )

            print(colored("[Gaussian Mapper] ", "magenta"), colored(f"Refinement loss: {loss / len(frames)}", "cyan"))
            self.loss_list.append(loss / len(frames))

            if self.use_gui:
                self.q_main2vis.put_nowait(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        keyframes=frames,
                    )
                )

    def get_mapping_update(
        self,
        frames: List[Camera],
        was_pruned: bool = False,
        feedback_poses: bool = True,
        feedback_disps: bool = False,
        opacity_threshold: float = 0.2,
        ignore_frames: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        min_coverage: float = 0.7,
        max_diff_to_video: float = 0.15,
    ) -> Dict:
        """Get the index, poses and depths of the frames that were already optimized. We can use this to then feedback
        the outputs of the Gaussian Rendering optimization back into the video.map.

        We should only feedback 'good' depths since Rendering can also introduce many outliers, noise or
        just does not converge correctly immediately. The Tracker operates with dense depth frames, so we need to be careful to not feedback too sparse depths.
        For this purpose, we limit the 'step size' of our renderer by ensuring that the rendered depth does not deviate from the original too much. If the
        current Renderer map does not cover enough of the scene in the specific view, we skip the frame.

        Since the map is usually not very good in the beginning, we never feedback the first few keyframes.
        """
        index, poses, depths = [], [], []
        # Sanity check
        if not feedback_disps and not feedback_poses:
            return {"index": index, "poses": poses, "depths": depths}

        # Render frames to extract depth
        for view in frames:
            # Ignore boundary frames, especially in the beginning when we build the map
            # (the first few frames are usually not good for feedback as they are not optimized yet or heavily incomplete)
            if self.idx_mapping[view.uid] in ignore_frames:
                continue
            render_pkg = render(view, self.gaussians, self.pipeline_params, self.background, device=self.device)

            # We computed covisibility and can therefore count how many Gaussians were observed in how many frames
            if was_pruned:
                in_frame = self.gaussians.unique_kfIDs == view.uid
                n_observed = self.gaussians.n_obs[in_frame]
                # Bad frames usually dont have many covisible Gaussians attached to them
                if (n_observed < 1).sum() / n_observed.numel() > 0.2:
                    self.info(f"Skipping view {view.uid} during Feedback as it has too few covisible Gaussians ...")
                    continue

            # NOTE chen: this can be None when self.gaussians is 0. This could happen in some cases
            if render_pkg is None:
                self.info(f"Skipping view {view.uid} as no Gaussians are present in it ...")
                continue

            index_in_video = self.idx_mapping[view.uid]
            depth = render_pkg["depth"].detach()
            # Filter away pixels with very low opacity as these are usually unreliable
            valid_o = render_pkg["opacity"].detach() > opacity_threshold
            depth[~valid_o] = 0.0

            # Analyze the video depths before filtering as a reference
            if self.video.upsample:
                disps_ref = self.video.disps_up[index_in_video]
            else:
                disps_ref = self.video.disps[index_in_video]
            valid_ref = disps_ref > 0
            depth_ref = torch.where(valid_ref, 1.0 / disps_ref, disps_ref)

            disps_clean = self.video.disps_clean[index_in_video]
            valid_clean = disps_clean > 0
            depth_clean = torch.where(valid_clean, 1.0 / disps_clean, disps_clean)

            # Limit step size by punishing large deviations from the original video depth
            if not self.video.upsample:
                s = self.video.scale_factor
                depth_down = depth[:, int(s // 2 - 1) :: s, int(s // 2 - 1) :: s]
                diff_to_video = torch.abs(depth_down - depth_ref) / depth_ref  # Abs rel.
                depth_wo_outliers = torch.where(
                    diff_to_video < max_diff_to_video, depth_down, 0.0
                )  # Filter away outliers
            else:
                diff_to_video = torch.abs(depth - depth_ref) / depth_ref  # Abs rel.
                depth_wo_outliers = torch.where(diff_to_video < max_diff_to_video, depth, 0.0)  # Filter away outliers
            coverage_gs_wo = (depth_wo_outliers > 0).sum() / (depth_wo_outliers > 0).numel()
            coverage_init = (depth_clean > 0).sum() / (depth_clean > 0).numel()

            # Skip if we dont even densify
            # if coverage_gs_wo < coverage_init:
            #     continue

            #### Visualizations for debugging ####
            # disps_clean = self.video.disps_clean[index_in_video]
            # valid_clean = disps_clean > 0
            # depth_clean = torch.where(valid_clean, 1.0 / disps_clean, disps_clean)
            # max_depth = min(max(depth_clean.max(), depth_ref.max(), depth.max()), 4.0)
            # self.plot_depth(
            #     depth.squeeze(), title=f"Depth after Gaussian Optimization, view {view.uid}", max_depth=max_depth
            # )
            # self.plot_depth(
            #     depth_clean.squeeze(),
            #     title=f"Depth (filtered) before Gaussian Optimization, view {view.uid}",
            #     max_depth=max_depth,
            # )

            # max_depth = min(max(depth_ref.max(), depth.max()), 4.0)
            # self.plot_depth(
            #     depth_wo_outliers.squeeze(),
            #     title=f"Depth without outliers after Gaussian Optimization, view {view.uid}",
            #     max_depth=max_depth,
            # )
            # self.plot_depth(
            #     depth_ref.squeeze(),
            #     title=f"Dense Depth before Gaussian Optimization, view {view.uid}",
            #     max_depth=max_depth,
            # )
            # self.plot_error(depth, depth_up, title=f"Difference to Dense Depth, view {view.uid}")
            ####

            # Disparity in video is dense -> Dont feedback too sparse frames
            if coverage_gs_wo > min_coverage:
                index.append(index_in_video)
                transform = torch.eye(4, device=self.device)
                transform[:3, :3], transform[:3, 3] = view.R, view.T
                poses.append(transform)
                depths.append(clone_obj(depth))

            torch.cuda.empty_cache()
            gc.collect()

        if not feedback_disps:
            depths = []
        if not feedback_poses:
            poses = []

        return {"index": index, "poses": poses, "depths": depths}

    def plot_error(
        self, pred: torch.Tensor, ref: torch.Tensor, cmap: str = "viridis", title: Optional[str] = None
    ) -> None:
        import matplotlib.pyplot as plt

        abs_rel = torch.abs(pred - ref) / ref
        plt.imshow(abs_rel.squeeze().cpu().numpy(), cmap=cmap, vmax=0.5)
        if title is not None:
            plt.title(title)
        plt.axis("off")
        plt.show()

    def plot_depth(self, depth: torch.Tensor, title: Optional[str] = None, max_depth: Optional[float] = None) -> None:
        import matplotlib.pyplot as plt
        from .visualization import depth2rgb

        plt.imshow(depth2rgb(depth, max_depth=max_depth).squeeze())
        plt.axis("off")
        if title is not None:
            plt.title(title)
        plt.show()

    def plot_video_disp(self, idx: int, use_clean: bool = False, max_depth: Optional[float] = None) -> None:
        if use_clean:
            disp = self.video.disps_clean[idx]
        else:
            if self.video.upsampled:
                disp = self.video.disps_up[idx]
            else:
                disp = self.video.disps[idx]
        valid = disp > 0
        depth = torch.where(valid, 1.0 / disp, disp)
        self.plot_depth(depth, title=f"Video Depth Map: {idx}", max_depth=max_depth)

    def get_camera_trajectory(self, frames: List[Camera]) -> Dict:
        """Get the camera trajectory of the frames in world coordinates."""
        poses = {}
        for view in frames:
            w2c = torch.eye(4, device=self.device)
            w2c[:3, :3], w2c[:3, 3] = view.R, view.T
            poses[view.uid] = w2c
        return poses

    def maybe_clean_pose_update(self, frames: List[Camera]) -> None:
        """Check if pose updates are not degenerate and set to zero if they are."""
        for view in frames:
            if torch.isnan(view.cam_rot_delta).any() and torch.isnan(view.cam_trans_delta).any():
                print(colored(f"NAN in pose optimizer in view {view.uid}!", "red"))
                print(colored("Setting to zero update ...", "red"))
                view.cam_rot_delta = torch.nn.Parameter(torch.zeros(3, device=self.device))
                view.cam_trans_delta = torch.nn.Parameter(torch.zeros(3, device=self.device))

    def mapping_step(
        self,
        iter: int,
        frames: List[Camera],
        kf_mng_params: Dict,
        densify: bool = True,
        prune_densify: bool = True,
        optimize_poses: bool = False,
        lr_factor: float = 1.0,
    ) -> float:
        """
        Takes the list of selected keyframes to optimize and performs one step of the mapping optimization.
        """
        # Sanity check when we dont have anything to optimize
        if len(self.gaussians) == 0:
            return 0.0

        # NOTE chen: this can happen we have zero depth and an inconvenient pose
        self.gaussians.check_nans()

        if optimize_poses:
            pose_optimizer = self.get_pose_optimizer(frames)

        loss = 0.0
        low_opacity = []
        for view in frames:

            render_pkg = render(view, self.gaussians, self.pipeline_params, self.background, device=self.device)
            # NOTE chen: this can be None when self.gaussians is 0. This could happen in some cases
            if render_pkg is None:
                self.info(f"Skipping view {view.uid} as no gaussians are present ...")
                continue

            visibility_filter, viewspace_point_tensor = render_pkg["visibility_filter"], render_pkg["viewspace_points"]
            image, radii, depth = render_pkg["render"], render_pkg["radii"], render_pkg["depth"]
            opacity, n_touched = render_pkg["opacity"], render_pkg["n_touched"]

            if self.supervise_with_prior and self.mode == "prgbd":
                scale_invariant = True
            else:
                scale_invariant = False
            current_loss = mapping_rgbd_loss(
                image,
                depth,
                view,
                with_edge_weight=self.loss_params.with_edge_weight,
                with_ssim=self.loss_params.use_ssim,
                with_depth_smoothness=self.loss_params.use_depth_smoothness_reg,
                alpha1=self.loss_params.alpha1,
                alpha2=self.loss_params.alpha2,
                beta=self.loss_params.beta2,
                rgb_boundary_threshold=self.loss_params.rgb_boundary_threshold,
                supervise_with_prior=self.supervise_with_prior,
                scale_invariant=scale_invariant,
            )

            # low_opacity.append((view, opacity))
            loss += current_loss

        loss = (
            loss / len(frames) * lr_factor
        )  # NOTE chen: we allow lr_factor to make it possible to change the learning rate on the fly
        # Regularize scale changes of the Gaussians
        scaling = self.gaussians.get_scaling
        isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
        loss += self.loss_params.beta * isotropic_loss.mean()

        # NOTE chen: this can happen we have zero depth and an inconvenient pose
        self.gaussians.check_nans()

        # Scale the loss with the number of frames so we adjust the learning rate dependent on batch size
        # NOTE chen: this is only a valid strategy for standard optimizers
        # if the optimizer has a regularization term (e.g. weight decay), then this changes the trade-off between objective and regularizor!
        # NOTE chen: MonoGS scales their loss with len(frames), while we scale it with sqrt(len(frames))
        scaled_loss = loss * np.sqrt(len(frames))
        scaled_loss.backward()

        with torch.no_grad():
            # Dont let Gaussians grow too much
            self.gaussians.max_radii2D[visibility_filter] = torch.max(
                self.gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter],
            )

            if densify:
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if self.last_idx > self.n_last_frames and (iter + 1) % self.kf_mng_params.prune_densify_every == 0:
                if prune_densify:
                    # General pruning based on opacity and size + densification
                    self.gaussians.densify_and_prune(
                        kf_mng_params.densify_grad_threshold,
                        kf_mng_params.opacity_th,
                        kf_mng_params.gaussian_extent,
                        kf_mng_params.size_threshold,
                    )
                if densify:
                    ng_before = len(self.gaussians)
                    for view, opacity in low_opacity:
                        self.gaussians.densify_w_opacity(opacity, view)
                    # Only print when we added some new Gaussians
                    if (len(self.gaussians) - ng_before) > 0:
                        self.info(f"Added {len(self.gaussians) - ng_before} gaussians based on opacity")

            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad()
            self.gaussians.update_learning_rate(iter)

            if optimize_poses:
                pose_optimizer.step()
                self.maybe_clean_pose_update(frames)
                pose_optimizer.zero_grad()
                # go over all poses that were affected by the pose optimization
                for view in frames:
                    update_pose(view)

        return loss.item()

    def covisibility_pruning(self, prune: str = "new", dont_prune_last: int = 1, visibility_thresh: int = 2):
        """Covisibility based pruning.

        If prune is set to "last", we only prune the last n frames. Else we check for all frames if Gaussians are visible in at least k frames.
        A Gaussian is visible if it touched at least a single pixel in a view.
        """
        # TODO do these a sanity check in __init__ so we dont get this print just once
        # Sanity checks, so this is not misused
        if dont_prune_last >= len(self.new_cameras):
            self.info(
                f"Warning. You selected to not prune the last {dont_prune_last} frames, but this is bigger than the optimization window! Please reconfigure ..."
            )
        if dont_prune_last > len(self.cameras + self.new_cameras):
            return
        occ_aware_visibility = {}

        frames = sorted(self.cameras + self.new_cameras, key=lambda x: x.uid)
        # Make a covisibility check only for the last n frames
        if prune == "new":
            # Dont prune the Last/last-1 frame, since we then would add and prune Gaussians immediately -> super wasteful
            if dont_prune_last > 0:
                frames = frames[-self.n_last_frames : -dont_prune_last]
            else:
                frames = frames[-self.n_last_frames :]

        self.gaussians.n_obs.fill_(0)  # Reset observation count
        for view in frames:
            render_pkg = render(view, self.gaussians, self.pipeline_params, self.background)
            visibility = (render_pkg["n_touched"] > 0).long()
            occ_aware_visibility[view.uid] = visibility
            # Count when at least one pixel was touched by the Gaussian
            self.gaussians.n_obs += visibility.cpu()  # Increase observation count

        to_prune = self.gaussians.n_obs < visibility_thresh
        if prune == "new":
            ids = [view.uid for view in frames]
            sorted_frames = sorted(ids, reverse=True)
            # Only prune Gaussians added on the last k frames
            prune_last = self.gaussians.unique_kfIDs >= sorted_frames[self.kf_mng_params.prune_last - 1]
            to_prune = torch.logical_and(to_prune, prune_last)

        if to_prune.sum() > 0:
            self.gaussians.prune_points(to_prune.to(self.device))
        self.info(f"({prune}) Covisibility pruning removed {to_prune.sum()} gaussians")

    def select_keyframes(self):
        """Select the last n1 frames and n2 other random frames from all."""
        if len(self.cameras) <= self.n_last_frames + self.n_rand_frames:
            keyframes = self.cameras
            keyframes_idx = np.arange(len(self.cameras))
        else:
            keyframes_idx = np.random.choice(len(self.cameras) - self.n_last_frames, self.n_rand_frames, replace=False)
            keyframes = self.cameras[-self.n_last_frames :] + [self.cameras[i] for i in keyframes_idx]
        return keyframes, keyframes_idx

    def add_nonkeyframe_cameras(self):
        self.info("Interpolating trajectory to get non-keyframe Cameras for refinement ...")
        non_kf_cams = self.get_nonkeyframe_cameras(self.slam.dataset, self.slam.traj_filler)
        # Reinstatiate an empty traj_filler, we only reuse this during eval
        # this deletes the graph and frees up memory
        del self.slam.traj_filler
        self.slam.traj_filler = PoseTrajectoryFiller(
            self.slam.cfg, net=self.slam.net, video=self.slam.video, device=self.slam.device
        )
        torch.cuda.empty_cache()
        gc.collect()

        self.info(f"Added {len(non_kf_cams)} new cameras: {[cam.uid for cam in non_kf_cams]}")
        # Initialize without additional Gaussians, since we only use the additional frames as better supervision
        for cam in non_kf_cams:
            self.initialized = True

        # Keyframe cameras we add during SLAM: cam.uid = video.idx, which is the position in our video buffer
        # Non-keyframe cameras we interpolated: cam.uid = stream.timestamp
        # We can get the global mapping, because: video.timestamp[video.idx] = stream.timestamp
        new_mapping, masked_mapping = {}, {}
        for cam in self.cameras:
            new_id = int(self.video.timestamp[cam.uid].item())
            masked_mapping[cam.uid] = (new_id, self.gaussians.unique_kfIDs == cam.uid)
            new_mapping[new_id] = cam.uid  # Memoize from timestamp to old video index
            cam.uid = new_id  # Reassign local keyframe ids to global stream ids

        self.idx_mapping = new_mapping
        # Update the keyframe ids for each Gaussian, so they fit the new global cam.uid's
        for key, val in masked_mapping.items():
            new_id, mask = val
            self.gaussians.unique_kfIDs[mask] = new_id

        self.cameras += non_kf_cams  # Add to set of cameras
        # Reorder according to global uid
        self.cameras = sorted(self.cameras, key=lambda x: x.uid)

    def _last_call(self, mapping_queue: mp.Queue, received_item: mp.Event):
        """We already build up the map based on the SLAM system and finetuned over it.
        Depending on compute budget this has been done scarcely.
        This call runs many more iterations for refinement and densification to get a high quality map.

        Since the SLAM system operates on keyframes, but we have many more views in our video stream, we can use additional
        supervision from non-keyframes to get higher detail.
        """
        # Free memory before doing refinement
        torch.cuda.empty_cache()
        gc.collect()

        # NOTE MonoGS does 26k iterations for a single camera, while we do 100 for multiple cameras
        if self.refinement_iters > 0:

            self.info(f"Gaussians before Map Refinement: {len(self.gaussians)}")
            # Add more information for refinement if wanted
            if self.slam.dataset is not None and self.use_non_keyframes:
                self.add_nonkeyframe_cameras()

            self.info("\nMapping refinement starting")
            # Only optimize over 20% of the whole video and always make sure that 30% of each batch is keyframes
            self.map_refinement(
                num_iters=self.refinement_iters, optimize_poses=self.optimize_poses, random_frames=0.2, kf_at_least=0.3
            )
            self.info(f"Gaussians after Map Refinement: {len(self.gaussians)}")
            self.info("Mapping refinement finished")

            # Free memory after doing refinement
            torch.cuda.empty_cache()
            gc.collect()

        self.info(f"Feeding back into video.map ...")
        # Filter out the non-keyframes which are not stored in the video.object
        only_kf = [cam for cam in self.cameras if cam.uid in self.idx_mapping]
        # Only feedback the poses, since we will not work with the video again
        # TODO chen: do we need a case, where we dont optimize poses during the run, but do it on refinement? 
        if self.feedback_poses or self.feedback_disps:
            # HACK allow large differences to the video, else we will filter away occluded regions which we already corrected rightfully
            to_set = self.get_mapping_update(
                only_kf,
                feedback_poses=self.feedback_poses,  
                feedback_disps=self.feedback_disps,
                opacity_threshold=0.0,
                ignore_frames=[],
                max_diff_to_video=1.0,
            )
            # There is no need to feed back the
            self.video.set_mapping_item(**to_set)

        self.gaussians.save_ply(f"{self.output}/mesh/final_{self.mode}.ply")
        self.info(f"Mesh saved at {self.output}/mesh/final_{self.mode}.ply")

        if self.save_renders:
            for cam in self.cameras:
                self.save_render(cam, f"{self.output}/intermediate_renders/final/{cam.uid}.png")

        plot_losses(
            self.loss_list,
            self.refinement_iters,
            title=f"Loss evolution.{len(self.gaussians)} gaussians",
            output_file=f"{self.output}/loss_{self.mode}.png",
        )

        print(colored("[Gaussian Mapper] ", "magenta"), colored(f"Final mapping loss: {self.loss_list[-1]}", "cyan"))
        self.info(f"{len(self.iteration_info)} iterations, {len(self.cameras)/len(self.iteration_info)} cams/it")

        # Export Cameras and Gaussians to the main Process
        if self.evaluate:
            mapping_queue.put(
                EvaluatePacket(
                    pipeline_params=clone_obj(self.pipeline_params),
                    cameras=self.cameras[:],
                    gaussians=clone_obj(self.gaussians),
                    background=clone_obj(self.background),
                    idx_mapping=clone_obj(self.idx_mapping),
                )
            )
        else:
            mapping_queue.put("None")
        received_item.wait()  # Wait until the Packet got delivered

    def add_new_gaussians(self, cameras: List[Camera]) -> Camera | None:
        """Initialize new Gaussians based on the provided views (images, poses (, depth))"""
        # Sanity check
        if len(cameras) == 0:
            return None

        for cam in cameras:
            if not self.initialized:
                self.initialized = True
                self.gaussians.extend_from_pcd_seq(cam, cam.uid, init=True)
                self.info(f"Initialized with {len(self.gaussians)} gaussians for view {cam.uid}")
            else:
                ng_before = len(self.gaussians)
                self.gaussians.extend_from_pcd_seq(cam, cam.uid, init=False)
                # self.info(f"Added {len(self.gaussians) - ng_before} gaussians based on view {cam.uid}")

        return cam

    def _update(self, delay_to_tracking=True, iters: int = 10):
        """Update our rendered map by:
        i) Pull a filtered update from the sparser SLAM map
        ii) Add new Gaussians based on new views
        iii) Run a bunch of optimization steps to update Gaussians and camera poses
        iv) Prune the resulting Gaussians based on visibility and size
        v) Maybe send back a filtered update to the SLAM system
        """
        self.info("Currently has: {} gaussians".format(len(self.gaussians)))

        ### Filter map based on multiview_consistency and uncertainty
        # self.video.dummy_filter() # TEST: what if we dont apply any multiview_filter?
        self.video.filter_map(
            min_count=self.kf_mng_params.filter.mv_count_thresh,
            bin_thresh=self.kf_mng_params.filter.bin_thresh,
            unc_threshold=self.kf_mng_params.filter.confidence_thresh,
            use_multiview_consistency=self.filter_multiview,
            use_uncertainty=self.filter_uncertainty,
        )

        if delay_to_tracking:
            delay = self.delay
        else:
            delay = 0

        ### Add new cameras based on video index
        self.get_new_cameras(delay=delay)  # Add new cameras
        if len(self.new_cameras) != 0:
            self.last_idx = self.new_cameras[-1].uid + 1
            self.info(f"Added {len(self.new_cameras)} new cameras: {[cam.uid for cam in self.new_cameras]}")

        ### Update the frames based on Tracker and add new Gaussians
        self.frame_updater(delay=delay)  # Update all changed cameras with new information from SLAM system
        last_new_cam = self.add_new_gaussians(self.new_cameras)
        # We might have 0 Gaussians in some cases, so no need to run optimizer
        if len(self.gaussians) == 0:
            self.info("No Gaussians to optimize, skipping mapping step ...")
            return

        ### Optimize gaussians
        do_densify = len(self.iteration_info) % self.kf_mng_params.densify_every == 0
        for iter in tqdm(range(iters), desc=colored("Gaussian Optimization", "magenta"), colour="magenta"):
            frames = self.select_keyframes()[0] + self.new_cameras
            if len(frames) == 0:
                self.loss_list.append(0.0)
                continue

            loss = self.mapping_step(
                iter,
                frames,
                self.kf_mng_params.mapping,
                prune_densify=True,  # Prune and densify with vanilla 3DGS strategy
                densify=do_densify,  # Densify based on low opacity regions
                optimize_poses=self.optimize_poses,
            )
            self.loss_list.append(loss / np.sqrt(len(frames)))
        # Keep track of how well the Rendering is doing
        print(colored("\n[Gaussian Mapper] ", "magenta"), colored(f"Loss: {self.loss_list[-1]}", "cyan"))

        ### Prune unreliable Gaussians
        if len(self.iteration_info) % self.kf_mng_params.prune_every == 0:
            self.covisibility_pruning(
                prune=self.kf_mng_params.prune_mode,  # Prune either for all gaussians or just the last n frames
                dont_prune_last=0,
                visibility_thresh=self.kf_mng_params.visibility_th,
            )
            was_pruned = True
        else:
            was_pruned = False

        ### Feedback new state of map to Tracker
        if self.feedback_poses or self.feedback_disps and self.count > self.feedback_warmup:
            if self.feedback_only_last_window and len(self.new_cameras) > 0:
                update_cams = sorted(frames, key=lambda x: x.uid)[-self.n_last_frames :]
            else:
                update_cams = frames
            to_set = self.get_mapping_update(
                update_cams, was_pruned, feedback_poses=self.feedback_poses, feedback_disps=self.feedback_disps
            )
            self.info("Feeding back to Tracking ...")
            self.video.set_mapping_item(**to_set)

        ### Update visualization
        if self.use_gui:
            # Get the latest frame we added together with the input
            if len(self.new_cameras) > 0 and last_new_cam is not None:
                img = last_new_cam.original_image
                if last_new_cam.depth is not None:
                    if not self.supervise_with_prior:
                        gtdepth = last_new_cam.depth.detach().cpu().numpy()
                    else:
                        gtdepth = last_new_cam.depth_prior.detach().cpu().numpy()
                else:
                    gtdepth = None
            # We did not have any new input, but refine the Gaussians
            else:
                img, gtdepth = None, None
                last_new_cam = self.cameras[-1]

            self.q_main2vis.put_nowait(
                gui_utils.GaussianPacket(
                    gaussians=clone_obj(self.gaussians),
                    current_frame=last_new_cam,
                    keyframes=self.cameras,
                    kf_window=None,
                    gtcolor=img,
                    gtdepth=gtdepth,
                )
            )

        ### Save renders for debugging and visualization
        if self.save_renders:
            # Save only every 5th camera to save disk spacse
            for cam in self.cameras:
                if cam.uid % 5 == 0:
                    self.save_render(cam, f"{self.output}/intermediate_renders/temp/{cam.uid}.png")

        # Free memory each iteration NOTE: this slows it down a bit
        torch.cuda.empty_cache()
        gc.collect()

        self.iteration_info.append(len(self.new_cameras))
        # Keep track of added cameras
        self.cameras += self.new_cameras
        self.new_cameras = []

    def __call__(self, mapping_queue: mp.Queue, received_item: mp.Event, the_end=False):

        self.cur_idx = self.video.counter.value + 1
        if self.last_idx + self.delay < self.cur_idx and self.cur_idx > self.warmup:
            self._update(iters=self.mapping_iters)
            self.count += 1  # Count how many times we ran the Renderer
            return False

        elif the_end and self.last_idx + self.delay >= self.cur_idx:
            self.cur_idx = self.video.counter.value
            # Run another call to catch the last batch of keyframes
            self._update(iters=self.mapping_iters + 10, delay_to_tracking=False)
            self.count += 1

            self._last_call(mapping_queue=mapping_queue, received_item=received_item)
            return True

        else:
            return False
