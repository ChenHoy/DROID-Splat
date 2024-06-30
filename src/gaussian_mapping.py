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


# FIXME after large map changes (we can flag potential loop closures and map changes), the Gaussians normally would need to be recenterd, i.e.
# its not enough to simply change the depth_maps in self.update_frames(), because this still requires further optimization to fit the new depth maps
# We use a heuristic for this: Simply compute a point cloud for
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
        self.refinement_iters = cfg.mapping.refinement_iters
        self.use_non_keyframes = cfg.mapping.use_non_keyframes
        self.mapping_iters = cfg.mapping.mapping_iters
        self.save_renders = cfg.mapping.save_renders
        self.warmup = max(cfg.mapping.warmup, cfg.tracking.warmup)
        self.batch_mode = cfg.mapping.batch_mode  # Take a batch of all unupdated frames at once
        self.optimize_poses = cfg.mapping.optimize_poses
        self.opt_params = cfg.mapping.opt_params
        self.feedback_map = cfg.mapping.feedback_mapping
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

        self.z_near = 0.001
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

        # Memoize the mapping of Gaussian indices to video indices
        # NOTE chen: use this for getting keyframes directly
        self.idx_mapping = {}

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

    # FIXME can we do a readjustment in case of strong changes in the map?
    # normally people anchor frames to a segment of the map, they then connect segments based on PoseGraphOptimization (PGO)
    # When we e.g. do a loop closure, we need to realign the Gaussians of the closed loop segments
    # Since we dont do PGO, we could only recenter Gaussians based on the changed poses and associated unique_kfIDs
    # (-> Compute the distance in world coordinates between the two poses and move the Gaussians accordingly)
    def frame_updater(self, delay=0):
        """Gets the list of frames and updates the depth and pose based on the video.

        NOTE: this assumes that the tracking on the video is a better estimate than the Gaussians alone. We
        prioritize optical flow based tracking over soleley Gaussians due to speed and robustness.
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

    def get_nonkeyframe_cameras(self, stream, trajectory_filler, batch_size: int = 32) -> List[Camera]:
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

            w2c = w2c.matrix()  # Convert to 4x4 homogeneous matrix
            # color = stream._get_image(timestamp).squeeze(0).permute(1, 2, 0).contiguous().to(self.device)
            color = stream._get_image(timestamp).clone().squeeze(0).contiguous().to(self.device)
            if not self.video.upsampled:
                color = color[..., int(s // 2 - 1) :: s, int(s // 2 - 1) :: s]
            cam = self.camera_from_frame(timestamp, color, w2c, intrinsics)
            new_cams.append(cam)

        return new_cams

    # TODO how to still densify and grow? should we adjust this in relation to the current number of Gaussians?
    # FIXME we somehow have some odd Gaussians in the middle of the scene
    # these seems to be initialized randomly from the ones before
    def map_refinement(
        self,
        num_iters: int = 100,
        optimize_poses: bool = False,
        prune: bool = False,
        random_frames: Optional[float] = None,
        kf_at_least: Optional[float] = None,
    ) -> None:
        """Refine the map. Instead of going over last frames, we always select random frames from the whole map."""

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
                        iter, chunk, self.kf_mng_params.refinement, densify=False, optimize_poses=optimize_poses
                    )
            else:
                loss = self.mapping_step(
                    iter, frames, self.kf_mng_params.refinement, densify=False, optimize_poses=optimize_poses
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

    def get_mapping_update(self, frames: List[Camera]) -> Dict:
        """Get the index, poses and depths of the frames that were already optimized. We can use this to then feedback
        the outputs of the Gaussian Rendering optimization back into the video.map."""

        # Render frames to extract depth
        index, poses, depths = [], [], []
        for view in frames:
            # FIXME why do we get an invalid non-invertible matrix for some views?!
            render_pkg = render(view, self.gaussians, self.pipeline_params, self.background, device=self.device)
            # NOTE chen: this can be None when self.gaussians is 0. This could happen in some cases
            if render_pkg is None:
                self.info(f"Skipping view {view.uid} as no gaussians are present ...")
                if self.optimize_poses:
                    poses.append(None)
                depths.append(None)
                index.append(None)
            else:
                index.append(self.idx_mapping[view.uid])
                if self.optimize_poses:
                    transform = torch.eye(4, device=self.device)
                    transform[:3, :3], transform[:3, 3] = view.R, view.T
                    poses.append(transform)
                depths.append(clone_obj(render_pkg["depth"].detach()))

            torch.cuda.empty_cache()
            gc.collect()

        return {"index": index, "poses": poses, "depths": depths}

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

    # TODO implement low_opacity pruning, because sometimes there are artifacts
    def mapping_step(
        self, iter: int, frames: List[Camera], kf_mng_params: Dict, densify: bool = True, optimize_poses: bool = False
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
            loss += mapping_rgbd_loss(
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

        # Regularize scale changes of the Gaussians
        scaling = self.gaussians.get_scaling
        isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
        loss += self.loss_params.beta * len(frames) * isotropic_loss.mean()

        # NOTE chen: this can happen we have zero depth and an inconvenient pose
        self.gaussians.check_nans()

        loss.backward()

        with torch.no_grad():
            # Dont let Gaussians grow too much
            self.gaussians.max_radii2D[visibility_filter] = torch.max(
                self.gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter],
            )

            if densify:
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if self.last_idx > self.n_last_frames and (iter + 1) % self.kf_mng_params.prune_densify_every == 0:
                self.gaussians.densify_and_prune(  # General pruning based on opacity and size + densification
                    kf_mng_params.densify_grad_threshold,
                    kf_mng_params.opacity_th,
                    kf_mng_params.gaussian_extent,
                    kf_mng_params.size_threshold,
                )

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

    def covisibility_pruning(self):
        """Covisibility based pruning"""

        new_frames = (self.cameras + self.new_cameras)[-self.n_last_frames :]

        self.occ_aware_visibility = {}
        for view in new_frames:
            render_pkg = render(view, self.gaussians, self.pipeline_params, self.background)
            self.occ_aware_visibility[view.uid] = (render_pkg["n_touched"] > 0).long()

        new_idx = [view.uid for view in new_frames]
        sorted_frames = sorted(new_idx, reverse=True)

        self.gaussians.n_obs.fill_(0)
        for _, visibility in self.occ_aware_visibility.items():
            self.gaussians.n_obs += visibility.cpu()

        # Gaussians added on the last prune_last frames
        mask = self.gaussians.unique_kfIDs >= sorted_frames[self.kf_mng_params.prune_last - 1]
        to_prune = torch.logical_and(self.gaussians.n_obs <= self.kf_mng_params.visibility_th, mask)
        if to_prune.sum() > 0:
            self.gaussians.prune_points(to_prune.cuda())
            # for idx in new_idx:
            #     self.occ_aware_visibility[idx] = self.occ_aware_visibility[idx][~to_prune]
        self.info(f"Covisibility pruning removed {to_prune.sum()} gaussians")

    def abs_visibility_prune(self, threshold: int = 2):
        """
        Absolute covisibility based pruning. Removes all gaussians
        that are not seen in at least abs_visibility_th views.
        """

        self.occ_aware_visibility = {}
        for view in self.cameras + self.new_cameras:
            render_pkg = render(view, self.gaussians, self.pipeline_params, self.background, device=self.device)
            self.occ_aware_visibility[view.uid] = (render_pkg["n_touched"] > 0).long()

        self.gaussians.n_obs.fill_(0)
        for _, visibility in self.occ_aware_visibility.items():
            self.gaussians.n_obs += visibility.cpu()

        mask = self.gaussians.n_obs < threshold
        if mask.sum() > 0:
            self.gaussians.prune_points(mask.cuda())
        self.info(f"Absolute visibility pruning removed {mask.sum()} gaussians")

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
                num_iters=self.refinement_iters, optimize_poses=True, random_frames=0.2, kf_at_least=0.3
            )
            self.info(f"#Gaussians after Map Refinement: {len(self.gaussians)}")
            self.info("Mapping refinement finished")

            # Free memory after doing refinement
            torch.cuda.empty_cache()
            gc.collect()

        self.info(f"Feeding back into video.map ...")
        # Filter out the non-keyframes which are not stored in the video.object
        only_kf = [cam for cam in self.cameras if cam.uid in self.idx_mapping]
        to_set = self.get_mapping_update(only_kf)
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

        # TODO uncomment after we are done debugging
        ## export the cameras and gaussians to the terminate process
        # if self.evaluate:
        #     mapping_queue.put(
        #         EvaluatePacket(
        #             pipeline_params=clone_obj(self.pipeline_params),
        #             cameras=self.cameras[:],
        #             gaussians=clone_obj(self.gaussians),
        #             background=clone_obj(self.background),
        #         )
        #     )
        # else:
        #     mapping_queue.put("None")
        # received_item.wait()  # Wait until the Packet got delivered

    def add_new_gaussians(self, cameras: List[Camera]) -> Camera | None:
        """Initialize new Gaussians based on the provided views (images, poses (, depth))"""
        # Sanity check
        if len(cameras) == 0:
            return None

        for cam in cameras:
            if not self.initialized:
                self.initialized = True
                self.gaussians.extend_from_pcd_seq(cam, cam.uid, init=True)
                self.info(f"Initialized with {len(self.gaussians)} gaussians")
            else:
                ng_before = len(self.gaussians)
                self.gaussians.extend_from_pcd_seq(cam, cam.uid, init=False)
                # self.info(f"Added {len(self.gaussians) - ng_before} gaussians based on view {cam.uid}")

        return cam

    def _update(self, delay_to_tracking=True):
        """Update our rendered map by:
        i) Pull a filtered update from the sparser SLAM map
        ii) Add new Gaussians based on new views
        iii) Run a bunch of optimization steps to update Gaussians and camera poses
        iv) Prune the render map based on visibility

        Finally we send the point cloud version of this rendered map back to the SLAM system.
        """
        self.info("Currently has: {} gaussians".format(len(self.gaussians)))

        ### Filter map based on multiview_consistency and uncertainty
        # NOTE This could be improved by only filtering the new and existing cameras instead of video.mapping_dirty
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
        # NOTE chen: This assumes, that we get a useful update from the SLAM system!
        # FIXME: we still need to correct loop closures or else we need to rely on expensive re-optimization!
        # TODO solution reanchor the means of the Gaussians to new iproj depth maps
        # Problem: it would not make sense to update the whole map by solving a global registration problem
        # other frameworks resolve this because they have a PoseGraphOptimization between local segments
        # a loop closure is then defined between segments, i.e. we can use the resolving transformations directly on all segments
        # TODO solution would be to store both the video buffer map and the Gaussians in a shared datastructure for spatial grouping
        # -> ultra naive, simply record where we had the biggest changes locally after a global optimization
        # to do this simply memoize the poses before running bundle adjustment and afterwards and then we know which Gaussians from a certain frame need to be relocated
        # TODO are gaussians really anchored?
        # -> naive: use hash map for different temporal segments since they will likely be close to each other, we can memoize during a loop closure which segments are connected
        # -> advanced: use an octree like datastructure to spatially divide frames into group/segments.
        self.frame_updater(delay=delay)  # Update all changed cameras with new information from SLAM system
        last_new_cam = self.add_new_gaussians(self.new_cameras)
        # We might have 0 Gaussians in some cases, so no need to run optimizer
        if len(self.gaussians) == 0:
            self.info("No Gaussians to optimize, skipping mapping step ...")
            return

        ### Optimize gaussians
        do_densify = len(self.iteration_info) % self.kf_mng_params.densify_every == 0
        for iter in tqdm(
            range(self.mapping_iters), desc=colored("Gaussian Optimization", "magenta"), colour="magenta"
        ):
            frames = self.select_keyframes()[0] + self.new_cameras
            if len(frames) == 0:
                self.loss_list.append(0.0)
                continue

            loss = self.mapping_step(
                iter, frames, self.kf_mng_params.mapping, densify=do_densify, optimize_poses=self.optimize_poses
            )
            self.loss_list.append(loss / len(frames))
        # Keep track of how well the Rendering is doing
        print(colored("\n[Gaussian Mapper] ", "magenta"), colored(f"Loss: {self.loss_list[-1]}", "cyan"))

        ### Prune unreliable Gaussians
        if len(self.iteration_info) % self.kf_mng_params.prune_every == 0:
            if self.kf_mng_params.prune_mode == "abs":
                # Absolute visibility pruning for all gaussians
                self.abs_visibility_prune(self.kf_mng_params.abs_visibility_th)
            elif self.kf_mng_params.prune_mode == "new":
                self.covisibility_pruning()  # Covisibility pruning for recently added gaussians

        ### Feedback new state of map to Tracker
        if self.feedback_map:
            to_set = self.get_mapping_update(frames)
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

        self.iteration_info.append(len(self.new_cameras))
        # Keep track of added cameras
        self.cameras += self.new_cameras
        self.new_cameras = []

    def __call__(self, mapping_queue: mp.Queue, received_item: mp.Event, the_end=False):

        # self.cur_idx = int(self.video.filtered_id.item())
        self.cur_idx = self.video.counter.value + 1
        if self.last_idx + self.delay < self.cur_idx and self.cur_idx > self.warmup:
            self._update()
            return False

        elif the_end and self.last_idx + self.delay >= self.cur_idx:
            self._update(delay_to_tracking=False)  # Run another call to catch the last batch of keyframes
            self._last_call(mapping_queue=mapping_queue, received_item=received_item)
            return True
