import os
from typing import Dict, List, Optional, Tuple
import gc
from termcolor import colored
from pathlib import Path
import ipdb
import time

import kornia as K
import kornia.feature as KF
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch_scatter import scatter_max
from einops import asnumpy, rearrange, repeat

import droid_backends
from ..geom import projective_ops as pops
from ..depth_video import DepthVideo
from lietorch import SE3

import dpvo_backends
import pypose as pp
from .patch_projective import transform as patch_transform
from ..utils import indices_to_tuple, TrajectorySegmentManager
from .patch_projective import iproj as patch_iproj
from .optim import SE3_to_Sim3, make_pypose_Sim3, ransac_umeyama, run_DPVO_PGO


def get_matching_keypoints(kp1, kp2, idxs):
    mkpts1 = kp1[idxs[:, 0].cpu()]
    mkpts2 = kp2[idxs[:, 1].cpu()]
    return mkpts1, mkpts2


class LongTermLoopClosure:
    """
    Classical Loop Closure as is done in DPVO-SLAM. We make some modifications to work with out system:
    - We dont use a temporary directory to build up an ImageCache structure, i.e. we dont read/write images to disk.
    Instead, we use the same DepthVideo buffer to index images directly
    - We use the sparser patch style optimization kernel for keypoints only during 3D keypoint triangulation
    We tried using the DROID-SLAM kernels, solely based on dense image maps, with only the 2D keypoints filled in,
    but this gives slightly less accurate and robust results.
    - We dont follow the same logic of doing i) keyframing ii) write to disk iii) wait for loop detector iv) attempt closure
    - We dont synchronize the same way:
    DPVO runs the loop detection and closure in a clear order during frontend/tracking logic, i.e.
    in the beginning of slam.update(), we keyframe the new frame, run bundle adjustment, etc. and finally check for
    loops and attempt closure. NOTE: I think this possible, because the Processes are run asynchronously

    Ours:
    - Let the loop detector alone as a standalone class that is run in a separate process
    - Simply queue up candidates from the detector and only then attempt closures
    - Memoize candidates and past loop closures in this class and simply filter the incoming candidates
    - Run the Loop Closure in its own separate Process with while loop, like all other modules
    """

    def __init__(
        self,
        cfg,
        video_buffer: DepthVideo,
        device: str = "cuda",
        log_folder: Optional[str] = None,
        viz_queue: Optional[mp.Queue] = None,
    ):
        self.cfg = cfg
        print(cfg)

        # 2D feature detector
        self.detector_window = self.cfg.get("detector_window", 15)
        self.score_threshold = self.cfg.get("score_threshold", 40.0)

        # RANSAC Point Cloud Registration
        self.min_num_inliers = self.cfg.get("min_num_inliers", 30)
        self.perc_inliers = self.cfg.get("perc_inliers", 0.3)
        self.ransac_thresh = self.cfg.get("ransac_thresh", 0.1)
        self.ransac_iters = self.cfg.get("ransac_iters", 400)
        self.nms = self.cfg.get("nms", 50)
        self.check_consecutive = self.cfg.get("check_consecutive", False)
        self.num_repeat = self.cfg.get("num_repeat", 3)
        self.triplet_stride = self.cfg.get("triplet_stride", 1)
        self.max_2d_residual = self.cfg.get("max_2d_residual", 2.0)
        self.delay = self.cfg.get("delay", 5)  # Delay to the frontend

        self.max_depth = self.cfg.get("max_depth", 20.0)
        self.device = device

        # Patch graph + loop edges
        self.video = video_buffer
        self.loop_ii = torch.zeros(0, dtype=torch.long)
        self.loop_jj = torch.zeros(0, dtype=torch.long)

        self.lc_count = 0
        self.prev_loop_closes = []
        self.found = []

        self.log_folder = log_folder
        self.viz_queue = viz_queue

        # warmup the jit compiler
        ransac_umeyama(np.random.randn(3, 3), np.random.randn(3, 3), iterations=200, threshold=0.01)

        self.detector = KF.DISK.from_pretrained("depth").to("cuda").eval()
        self.matcher = KF.LightGlue("disk").to("cuda").eval()
        # Keep track of the different spatial segments resulting from the loop closures
        self.segment_manager = TrajectorySegmentManager()

    def info(self, msg) -> None:
        print(colored("[Loop Closure]: " + msg, "cyan"))

    def filter_scores(self, ii, jj, scores):
        """We might detect multiple similarity edges for a given keyframe, i.e.
        a frame produces multiple potential loop closure candidates. However, there might
        be many false positives for the similarity search. Thats why frameworks like ORB-SLAM and
        DVPO only close loops when multiple consecutive detections happen.
        """
        ii_uniq = torch.unique(ii)
        max_scores = torch.zeros_like(ii_uniq, dtype=scores.dtype, device=ii.device)
        jj_uniq = torch.zeros_like(ii_uniq, device=ii.device)
        for i, val in enumerate(ii_uniq):
            indices = (ii == val).nonzero(as_tuple=False)
            scores_ii = scores[indices.squeeze(0)]
            max_idx = indices[torch.argmax(scores_ii.squeeze(0))].item()
            max_scores[i] = scores[max_idx]
            jj_uniq[i] = jj[max_idx]

        return ii_uniq, jj_uniq, max_scores

    def is_runnable(self, i, j, delay: int = 4):
        """Check if we can actually run the loop closure on the given edge.
        This looks if the video is already far enough to close the loop, i.e. we have valid poses for both triplets.
        NOTE: we use a delay here, so that the poses and disps are already optimized at least once
        """
        assert len(self.found) > 0, "Only check on valid loop edges, none were inserted in self.found!"
        with self.video.get_lock():
            cur_t = self.video.counter.value

        if i + self.triplet_stride + delay <= cur_t and j + self.triplet_stride + delay <= cur_t:
            return True
        else:
            return False

    # NOTE DPVO uses a Queue and a multiprocessing.Pool to execut the Levenberg-Marquardt optimization
    # for now we dont have a bottleneck from the Loop Closure and it rarely occurs, so this does not save much amortized costs
    def __call__(
        self,
        ii: Optional[torch.Tensor] = None,
        jj: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
        lc_in_progress: bool = False,
    ) -> bool:
        """Given a set of edges (ii, jj), attempt to close the loop for each candidate edge (i, j).
        If succesful, update pose graph with relative. pose and record completed closure.
        """
        performed_lc = False
        new_loop = None

        # Add batch of new indicies to buffer
        if ii is not None and jj is not None:
            ii, jj, scores = self.filter_scores(ii, jj, scores)
            candidates = indices_to_tuple(ii, jj)
            # Add valid edges to found
            for i, j in candidates:
                assert i > j, f"Loop closure candidate must have i > j, got {i} and {j}"
                # Ensure that this edge is not redundant
                # NOTE DVPO: euclidean distance to both nodes (i, j)
                # dists_sq = [(np.square(i - a) + np.square(j - b)) for a, b in self.prev_loop_closes]
                # if min(dists_sq, default=np.inf) < np.square(self.nms):
                # NOTE chen: I dont like the above solution because in case we find a loop shortly after a previous one, but the past node j is more distant
                # it will still trigger. For that reason we could have multiple back-to-back closures quickly
                dists = [(i - a) for a, _ in self.prev_loop_closes]
                if min(dists, default=np.inf) < self.nms:
                    self.info(
                        f"Rejection for ({i}, {j})! Redundant edge, as we already had a previous loop closure nearby."
                    )
                    continue

                self.found.append((i, j))

        # Pop all edges in the buffer until we find a succesful closure
        if not self.check_consecutive:
            while len(self.found) > 0:
                # Wait for the main system to be ahead here, so we can triangulate with neighboring frames
                # if not self.is_runnable(*self.found[0], delay=self.delay):
                #     break

                i, j = self.found.pop(0)
                closed_loop, new_loop = self.attempt_loop_closure(i, j, lc_in_progress)
                # If closed loop = True, then this edge actually got inserted into the pose graph
                if closed_loop:
                    performed_lc = True
                    self.segment_manager.insert_edge(*new_loop)
                    break

        else:
            # Pop off buffer and perform repetition check, do this until we find a succesful closure
            while len(self.found) >= self.num_repeat:
                # Check if we n consecutive detections for a valid loop closure
                # NOTE we for this reason filter edges for a given frame, i.e. for every i we only allow the edge to the most similar frame > thresh.
                # Use last index of the repetition window for comparison
                # NOTE this sometimes will result in very different edges, e.g. detection 1 and 3 are roughly similar, but 2 can be 5 frames off
                # cands = self._repetition_check(self.found[self.num_repeat - 1][0])
                cands = self._repetition_check(self.found[-1][0])
                if cands is not None:
                    # Wait for main system to be ahead so we can triangulate with neighboring frames
                    # if not self.is_runnable(*cands, delay=self.delay):
                    #     break

                    self.info(f"Attempting loop closure ...")
                    closed_loop, new_loop = self.attempt_loop_closure(*cands, lc_in_progress)
                else:
                    closed_loop = False

                if closed_loop:
                    self.segment_manager.insert_edge(*new_loop)
                    performed_lc = True
                    break

                # Delete half the window of non-robust detections
                del self.found[: self.num_repeat // 2]

        torch.cuda.empty_cache()
        gc.collect()

        return performed_lc, new_loop

    # def _repetition_check(self, idx: int):
    #     """Check that we've retrieved self.num_repeat consecutive frames as candidates"""
    #     if len(self.found) < self.num_repeat:
    #         return None

    #     latest = self.found[: self.num_repeat]
    #     (i, j) = latest[self.num_repeat // 2]
    #     (b, _) = latest[0]
    #     if (1 + idx - b) == self.num_repeat:
    #         return (i, max(j, 1))  # max(j,1) is to avoid centering the triplet on 0
    #     else:
    #         return None

    def _repetition_check(self, idx: int, tolerance: int = 2):
        """Check that we've retrieved <num_repeat> consecutive frames"""
        if len(self.found) < self.num_repeat:
            return
        latest = self.found[-self.num_repeat :]
        (b, _), (i, j), _ = latest

        # self.info(f"Searching self.found: {self.found}\n with idx:{idx}")

        # NOTE we add the tolerance, because on some scenes we will have a long streak of detections,
        # where just a single frame idx is off e.g. (1, 2, 4)
        if abs((1 + idx - b) - self.num_repeat) <= tolerance:
            return (i, max(j, 1))  # max(j,1) is to avoid centering the triplet on 0

    def attempt_loop_closure(
        self, i: int, j: int, lc_in_progress: torch.Tensor
    ) -> Tuple[bool, Optional[Tuple[int, int]]]:

        start = time.time()
        lc_in_progress += 1  # Mark that we are currently running a loop closure
        result = self.close_loop(i, j)  # Try to close the loop
        if result is not None:
            self.lc_count += 1

            self.confirm_loop(i, j)  # Record succesful closures
            new_loop = (i, j)
            self.update_buffer(result)  # Update video buffer with result
            end = time.time()
            self.info(f"Loop closure took {end - start:.2f} seconds")
            was_successful = True
            # NOTE it is extremely important to not have back-to-back loop closures!
            # without clearing, the whole system becomes unstable
            self.found.clear()
        else:
            was_successful = False
            new_loop = None
            lc_in_progress -= 1  # Mark that we are done running a loop closure

        return was_successful, new_loop

    def close_loop(self, i, j):
        """Close a loop closure by using 3D registration based on RANSAC and
        Pose Graph Optimization (PGO) to refine the poses.

        Given a detected loop edge (i, j), we estimate 2D keypoints for separate triplets
        around i and j, triangulate the sparse 3D points and then register two point clouds.
        We optimize for a similarity transformation (Sim3) between the two point clouds and
        use this new information to build a pose graph optimization problem with relative edges.
        We always optimize over all poses self.video.poses[:self.video.counter.value], i.e.
        this updates all poses and disparities in the window [0, self.video.counter.value] when a loop is closed.
        """
        ### 3D Keypoint Estimation
        i_pts, i_feat, i_matches = self.estimate_3d_keypoints_dpvo(
            i, stride=self.triplet_stride, max_residual=self.max_2d_residual, return_matches=True
        )
        j_pts, j_feat, j_matches = self.estimate_3d_keypoints_dpvo(
            j, stride=self.triplet_stride, max_residual=self.max_2d_residual, return_matches=True
        )
        # NOTE Pure DROID-SLAM kernel achieves slighly different results here
        # i_pts_droid, i_feat_droid = self.estimate_3d_keypoints_droid(i, stride=self.triplet_stride, max_residual=self.max_2d_residual)
        # j_pts_droid, j_feat_droid = self.estimate_3d_keypoints_droid(j, stride=self.triplet_stride, max_residual=self.max_2d_residual)

        _, _, iz = i_pts.mT
        _, _, jz = j_pts.mT

        # Filter out points that are too far away
        self.info(f"Triangulated {i_pts.size(0)} and {j_pts.size(0)} points for edge ({i}, {j})")
        i_pts, j_pts = i_pts[iz < self.max_depth], j_pts[jz < self.max_depth]
        for key in ["keypoints", "descriptors"]:
            i_feat[key] = i_feat[key][:, iz < self.max_depth]
            j_feat[key] = j_feat[key][:, jz < self.max_depth]

        self.info(f"{i_pts.size(0)} and {j_pts.size(0)} points left after depth filter for edge ({i}, {j})")

        # We need at least num_inliers triangulated 2D keypoints for each frame i or j
        if i_pts.numel() < self.min_num_inliers or j_pts.numel() < self.min_num_inliers:
            self.info(
                f"Rejection for ({i}, {j})! Too few inliers (A, Detection): {i_pts.numel()} / {self.min_num_inliers}"
            )
            return None

        ### 2D Feature Matching between the two point clouds
        # NOTE chen: this reduces the succesful triangulated matches further
        out = self.matcher({"image0": i_feat, "image1": j_feat})
        i_ind, j_ind = out["matches"][0].mT
        self.info(f"Matched {i_ind.size(0)}, {j_ind.size(0)} 2D keypoints between {i} and {j}")
        i_pts = i_pts[i_ind]
        j_pts = j_pts[j_ind]
        assert i_pts.shape == j_pts.shape, (i_pts.shape, j_pts.shape)
        i_pts, j_pts = asnumpy(i_pts.double()), asnumpy(j_pts.double())

        ### Visualize the matches
        if self.log_folder is not None:
            os.makedirs(os.path.join(self.log_folder, "loop_closures"), exist_ok=True)
            os.makedirs(os.path.join(self.log_folder, "loop_closures", "triangulation"), exist_ok=True)
            # Plot the 2D triangulated matches
            fname2d_i = os.path.join(self.log_folder, "loop_closures", "triangulation", f"lc_tri_{i}.png")
            self.plot_triplet_matches(save=True, fname=fname2d_i, **i_matches)
            fname2d_j = os.path.join(self.log_folder, "loop_closures", "triangulation", f"lc_tri_{j}.png")
            self.plot_triplet_matches(save=True, fname=fname2d_j, **j_matches)
            fname3d = os.path.join(self.log_folder, "loop_closures", f"lc_3d_{i}_{j}.png")
            self.plot_matches(
                i_feat, j_feat, self.video.images[i], self.video.images[j], out, save=True, fname=fname3d
            )

        # Early exit
        num_matched_points = min(i_pts.shape[0], j_pts.shape[0])
        if num_matched_points < self.min_num_inliers:
            self.info(
                f"Rejection for ({i}, {j})! Too few inliers (B, Matching): {num_matched_points} / {self.min_num_inliers}"
            )
            return None

        ### Point Clound Registration
        r, t, s, num_inliers, distances = ransac_umeyama(
            i_pts, j_pts, iterations=self.ransac_iters, threshold=self.ransac_thresh
        )
        ratio_inliers = num_inliers / num_matched_points
        # Exist if number of inlier matches is too small
        self.info(f"RANSAC distances for ({i}, {j}): {distances}")
        if ratio_inliers < self.perc_inliers or num_inliers < self.min_num_inliers:
            self.info(
                f"Rejection for ({i}, {j})! Too few inliers (C, RANSAC): {num_inliers} with {round(ratio_inliers*100, 2)}%"
            )
            return None
        else:
            self.info(
                f"Edge ({i}, {j})! RANSAC inliers: {num_inliers} with {round(ratio_inliers*100, 2)}% >= {self.perc_inliers*100}%"
            )

        ### Pose-Graph Optimization (PGO)
        # newest rel. pose from regristration
        far_rel_pose = make_pypose_Sim3(r, t, s)[None]

        # What if we add fruther poses between the delay?
        with self.video.get_lock():
            cur_t = self.video.counter.value

        # prev rel. poses for loops
        Gi = pp.SE3(self.video.poses.view(1, self.video.buffer_size, 7)[:, self.loop_ii])
        Gj = pp.SE3(self.video.poses.view(1, self.video.buffer_size, 7)[:, self.loop_jj])
        Gij = Gj * Gi.Inv()
        prev_sim3 = SE3_to_Sim3(Gij).data[0].cpu()  # Current state of the pose graph in SIM(3)

        loop_ii = torch.cat((self.loop_ii, torch.tensor([i])))
        loop_jj = torch.cat((self.loop_jj, torch.tensor([j])))

        loop_ii_plus, loop_jj_plus = torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
        for t in range(1, cur_t - i):
            loop_ii_plus = torch.cat((loop_ii_plus, torch.tensor([i + t])))
            loop_jj_plus = torch.cat((loop_jj_plus, torch.tensor([j + t])))
        loop_ii_total = torch.cat((self.loop_ii, torch.tensor([i]), loop_ii_plus))
        loop_jj_total = torch.cat((self.loop_jj, torch.tensor([j]), loop_jj_plus))
        # Get all rel. poses so far
        loop_poses = prev_sim3
        # Assume the same rel. constraint for i and all frames between i and cur_t
        for i in range(0, cur_t - i):
            loop_poses = torch.cat((loop_poses, far_rel_pose))
        loop_poses = pp.Sim3(loop_poses)  # All rel. poses in SIM(3)

        self.loop_ii, self.loop_jj = loop_ii, loop_jj  # Memoize all loop edges without the plus for next closure

        # Get current state of the pose graph in window
        pred_poses = pp.SE3(self.video.poses[:cur_t]).Inv().cpu()
        final_est = run_DPVO_PGO(pred_poses.data, loop_poses.data, loop_ii_total, loop_jj_total, fix_opt_window=False)

        ### Old Code with just a single far_rel_pose and all the old loop rel. poses
        # prev rel. poses for loops
        # Gi = pp.SE3(self.video.poses.view(1, self.video.buffer_size, 7)[:, self.loop_ii])
        # Gj = pp.SE3(self.video.poses.view(1, self.video.buffer_size, 7)[:, self.loop_jj])
        # Gij = Gj * Gi.Inv()
        # prev_sim3 = SE3_to_Sim3(Gij).data[0].cpu()  # Current state of the pose graph in SIM(3)

        # loop_poses = pp.Sim3(torch.cat((prev_sim3, far_rel_pose)))  # All rel. poses
        # loop_ii = torch.cat((self.loop_ii, torch.tensor([i])))
        # loop_jj = torch.cat((self.loop_jj, torch.tensor([j])))

        # with self.video.get_lock():
        #     cur_t = self.video.counter.value
        # pred_poses = pp.SE3(self.video.poses[:cur_t]).Inv().cpu()  # All poses in the window
        # self.loop_ii, self.loop_jj = loop_ii, loop_jj  # Memoize all loop edges
        # final_est = run_DPVO_PGO(pred_poses.data, loop_poses.data, loop_ii, loop_jj, fix_opt_window=False)

        self.info(f"Success! Closed loop between {i} and {j}")
        return final_est

    # NOTE the scale s will be mostly consistent for frames [0: loop_jj], i.e. the segment before the loop starts
    # The scale change then is usually a sequence across the loop
    def update_buffer(self, final_est: torch.Tensor):
        """Update the video buffer by overwriting with the new scales and poses.

        NOTE: compared to DPVO we might have a delay of a couple frames (~3)
        The PGO only optimizes until the last loop_ii, i.e. the frames between [loop_ii, cur_t]
        would have a different scale than the rest of the map if we update naively like DPVO
        -> We scale the rest of the frames with the last scale s[-1] from the PGO optimization and adjust the newest disparity
        The newest disparity is always constant and is the mean of the previous frame, which is adjusted here
        """
        safe_i, _ = final_est.shape
        res, s = final_est.tensor().to(self.video.device).split([7, 1], dim=1)

        # FIXME use segments to only correct drift in the latest segment, i.e. we only go back to either the first frame or last loop_ii - 1
        # Reason: We usually have a strong map for each closure, but later closures of just one segment can suddenly distort the whole map
        # all_segments = self.segment_manager.get_minimal_segments()
        # Only fix drift from latest segment onward, i.e. we already corrected drift in older parts of the map before
        # opt_from = all_segments[-1].start

        cur_t = self.video.counter.value

        # TODO delete in case we dont need the deltas
        s1 = torch.ones(cur_t, device=self.video.device)
        s1[:safe_i] = s.squeeze()

        with self.video.get_lock():
            self.video.poses[:safe_i] = SE3(res).inv().data
            self.video.disps[:safe_i] /= s.view(safe_i, 1, 1)
            self.video.disps_up[:safe_i] /= s.view(safe_i, 1, 1)

            # TODO do we really ever need the deltas?
            self._rescale_deltas(s1)

            # normalize so we always have identity pose for the first frame
            self.video.poses[:cur_t] = (SE3(self.video.poses[:cur_t]) * SE3(self.video.poses[[0]]).inv()).data
            self.info("Feeding back the result to the video buffer ... done!")

            # Overwrite the newest disparity with the mean of the previous frame to get scale of last frame
            self.video.disps[cur_t] = self.video.disps[cur_t - 1].median()
            # Overwrite newest interpolated pose to adjust to previous ones
            dP = SE3(self.video.poses[cur_t - 1]) * SE3(self.video.poses[cur_t - 2]).inv()  # Get relative pose
            self.video.poses[cur_t] = (dP * SE3(self.video.poses[cur_t - 1])).vec()

            # overwrite the newest disp with the mean of cur_t - 1
            # self.video.normalize()
            self.video.dirty[:cur_t] = True  # Update visualization
        # TODO Does the drift come from discrepency between self.video.counter.value and cur_t?

    # TODO do we really need this?
    def _rescale_deltas(self, s):
        """Rescale the poses of removed frames by their predicted scales"""

        tstamp_2_rescale = {}
        with self.video.get_lock():
            cur_t = self.video.counter.value

        for i in range(cur_t):
            tstamp_2_rescale[self.video.timestamp[i]] = s[i]

        # TODO we dont really use the deltas, because our pose interpolation works differently
        for t, (t0, dP) in self.video.delta.items():
            t_src = t
            while t_src in self.video.delta:
                t_src, _ = self.video.delta[t_src]
            s1 = tstamp_2_rescale[t_src]
            self.video.delta[t] = (t0, dP.scale(s1))

    def confirm_loop(self, i, j):
        """Record the loop closure so we don't have redundant edges"""
        assert i > j
        self.prev_loop_closes.append((i, j))

    @torch.no_grad()
    def detect_keypoints(self, images: torch.Tensor, num_features: int = 2048) -> Dict:
        """Use disk w/ lightglue for 2D keypoint detection in an image"""
        _, _, h, w = images.shape
        wh = torch.tensor([w, h]).view(1, 2).float().cuda()
        features = self.detector(
            images,
            num_features,
            pad_if_not_divisible=True,
            window_size=self.detector_window,
            score_threshold=self.score_threshold,
        )
        return [
            {"keypoints": f.keypoints[None], "descriptors": f.descriptors[None], "image_size": wh} for f in features
        ]

    def estimate_3d_keypoints_dpvo(
        self, i: int, stride: int = 1, max_residual: float = 5.0, return_matches: bool = False
    ) -> Dict:
        """Detect, match and triangulate 3D points based on lightglue keypoints and matcher"""

        # Load triplet [i-1, i, i+1]
        # Safeguard against invalid indices
        lower = max(i - stride, 0)
        upper = min(i + stride, self.video.counter.value)
        selected = [lower, i, upper]
        images = self.video.images[selected].float() / 255.0

        fl = self.detect_keypoints(images)

        # Form trajectories from 2D keypoint matches
        trajectories = torch.full((2048, 3), -1, device=self.device, dtype=torch.long)
        trajectories[:, 1] = torch.arange(2048)

        out1 = self.matcher({"image0": fl[0], "image1": fl[1]})
        i0, i1 = out1["matches"][0].mT
        trajectories[i1, 0] = i0

        out2 = self.matcher({"image0": fl[2], "image1": fl[1]})
        i2, i1 = out2["matches"][0].mT
        trajectories[i1, 2] = i2

        # trajectories = trajectories[torch.randperm(2048)]
        trajectories = trajectories[trajectories.min(dim=1).values >= 0]

        ### What do the keypoints look like?
        # self.plot_keypoints(fl, images)
        # self.plot_triplet_matches(
        #     fl, images, out1, out2, save=True, fname=os.path.join(self.log_folder, "loop_closures", f"lc_{i}.png")
        # )

        a, b, c = trajectories.mT
        n, _ = trajectories.shape
        kps0 = fl[0]["keypoints"][:, a]
        kps1 = fl[1]["keypoints"][:, b]
        kps2 = fl[2]["keypoints"][:, c]

        desc1 = fl[1]["descriptors"][:, b]
        image_size = fl[1]["image_size"]

        kk = torch.arange(n).cuda().repeat(2)
        ii = torch.ones(2 * n, device=self.device, dtype=torch.long)
        jj = torch.zeros(2 * n, device=self.device, dtype=torch.long)
        jj[n:] = 2

        # Construct mini graph
        true_disp = self.video.disps_up[i].median()
        patches = torch.cat((kps1, torch.ones(1, n, 1).cuda() * true_disp), dim=-1)
        patches = repeat(patches, "1 n uvd -> 1 n uvd 3 3", uvd=3)
        target = rearrange(torch.stack((kps0, kps2)), "ot 1 n uv -> 1 (ot n) uv", uv=2, n=n, ot=2)
        weight = torch.ones_like(target)

        poses = self.video.poses.view(1, self.video.buffer_size, 7)[:, selected].clone()
        intrinsics = (
            self.video.scale_factor * self.video.intrinsics.view(1, self.video.buffer_size, 4)[:, selected].clone()
        )
        coords = patch_transform(SE3(poses), patches, intrinsics, ii, jj, kk)
        coords = coords[:, :, 1, 1]
        residual = (coords - target).norm(dim=-1).squeeze(0)

        # Structure-only BA
        lmbda = torch.as_tensor([1e-3], device=self.device)

        dpvo_backends.ba(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, -1, 3, 3, 6, False)

        # Only keep small residuals below 2px, i.e. good keypoint matches
        coords = patch_transform(SE3(poses), patches, intrinsics, ii, jj, kk)
        coords = coords[:, :, 1, 1]
        residual = (coords - target).norm(dim=-1).squeeze(0)
        assert residual.numel() == 2 * n
        # NOTE this assumes that the max. reprojection error (between ij and ji) is less than 2px
        # -> we accept a triangulated point only if both matches are below 2px
        mask = scatter_max(residual, kk)[0] < max_residual

        # Backproject into 3D to point cloud
        points = patch_iproj(patches, intrinsics[:, torch.ones(n, device="cuda", dtype=torch.long)])
        points = points[..., 1, 1, :3] / points[..., 1, 1, 3:]

        if return_matches:
            matches = {"images": images, "fl": fl, "matches1": out1, "matches2": out2}
            return (
                points[:, mask].squeeze(0),
                {
                    "keypoints": kps1[:, mask],
                    "descriptors": desc1[:, mask],
                    "image_size": image_size,
                },
                matches,
            )
        else:
            return points[:, mask].squeeze(0), {
                "keypoints": kps1[:, mask],
                "descriptors": desc1[:, mask],
                "image_size": image_size,
            }

    # TODO delete if proven to be unnecessary
    def estimate_3d_keypoints_droid(self, i: int, stride: int = 1, max_residual: float = 5.0) -> Dict:
        """Use matched 2D keypoints for triangulation and return 3D points"""

        # Load triplet [i-1, i, i+1]
        # Safeguard against invalid indices
        lower = max(i - stride, 0)
        upper = min(i + stride, self.video.counter.value)
        selected = [lower, i, upper]
        images = self.video.images[selected].float() / 255.0

        fl = self.detect_keypoints(images)

        # Form trajectories from 2D keypoint matches
        trajectories = torch.full((2048, 3), -1, device=self.device, dtype=torch.long)
        trajectories[:, 1] = torch.arange(2048)

        out1 = self.matcher({"image0": fl[0], "image1": fl[1]})
        # idxs = out["matches"][0]
        i0, i1 = out1["matches"][0].mT
        trajectories[i1, 0] = i0

        out2 = self.matcher({"image0": fl[2], "image1": fl[1]})
        i2, i1 = out2["matches"][0].mT
        trajectories[i1, 2] = i2

        ### What do the keypoints look like?
        # self.plot_keypoints(fl, images)
        # self.plot_triplet_matches(fl, images, out1, out2)

        trajectories = trajectories[torch.randperm(2048)]
        trajectories = trajectories[trajectories.min(dim=1).values >= 0]

        a, b, c = trajectories.mT
        n, _ = trajectories.shape
        kps0, kps1 = fl[0]["keypoints"][:, a], fl[1]["keypoints"][:, b]
        kps2 = fl[2]["keypoints"][:, c]

        desc1 = fl[1]["descriptors"][:, b]
        image_size = fl[1]["image_size"]

        ii = torch.ones(2, device=self.device, dtype=torch.long)
        jj = torch.zeros(2, device=self.device, dtype=torch.long)
        kk = torch.arange(n).cuda().repeat(2)  # Index collapsed 2*N keypoints
        jj[1] = 2

        # Make copies for mini graph
        # Use neg. disparity for each non-keypoint so we will get the right invalid mask back
        disps = -1.0 * torch.ones_like(self.video.disps_up[selected].clone())
        # use mean or median for initialization as we do in frontend)
        init_disp = self.video.disps_up[i].median()
        disps[1, kps1[0, :, 1].int(), kps1[0, :, 0].int()] = init_disp

        # DVPO: [B, 2*N, 2] for keypoints
        # target = rearrange(torch.stack((kps0, kps2)), "ot 1 n uv -> 1 (ot n) uv", uv=2, n=n, ot=2)
        bs, _, h, w = images.shape
        weight = torch.zeros(bs - 1, 2, h, w, device=self.device)
        target = torch.zeros(bs - 1, h, w, 2, device=self.device)
        # -> Use kps1 to index the disps and then filter out the non-keypoints
        target[0, kps1[0, :, 1].int(), kps1[0, :, 0].int()] = kps0
        target[1, kps1[0, :, 1].int(), kps1[0, :, 0].int()] = kps2
        # TODO which points need to be weighted?
        # We have relative outgoing trajectories from kps1 to kps2 and kps0
        # Therefore, we simply need to weight kps1 for each edge ij and ji?
        weight[..., kps1[0, :, 1].int(), kps1[0, :, 0].int()] = 1.0
        target = target.permute(0, 3, 1, 2)  # Ours: [B, 2, H, W]
        poses = self.video.poses[selected].clone()
        intrinsics = self.video.intrinsics[selected].clone() * self.video.scale_factor

        coords, valid_mask = pops.general_projective_transform(
            poses=SE3(poses)[None, ...],  # [1, B]
            depths=disps[None, ...],  # [B, H, W]
            intrinsics=intrinsics[None, ...],  # [B, 4]
            ii=ii,  # N
            jj=jj,  # N
            model_id=self.video.model_id,
            jacobian=False,
            return_depth=False,
        )
        residual = (coords[0] - target.permute(0, 2, 3, 1)).norm(dim=-1).squeeze(0)
        residual[~valid_mask[0].squeeze().bool()] = 0

        # Structure-only BA
        lmbda = 1e-3 * torch.ones_like(disps, device=self.device)  # [B, H, W]
        disps[disps < 0] = 0  # set invalid disparities to 0

        droid_backends.ba(
            poses,  # [N, 7]
            disps,  # [N, H, W]
            intrinsics[0],  # [4]
            torch.zeros_like(disps, device=disps.device),  # disps_sens
            target.contiguous(),  # [B, 2, H, W]
            weight,  # [B, 2, H, W]
            lmbda,  # [B2, H, W]
            ii,
            jj,
            0,  # t-1
            3,  # t1
            6,  # iters
            self.video.model_id,
            1e-4,  # lm
            0.1,  # ep
            False,  # motion_only
            True,  # Structure Only
            False,  # opt_intr
        )

        coords, _ = pops.general_projective_transform(
            poses=SE3(poses)[None, ...],  # [1, B]
            depths=disps[None, ...],  # [B, H, W]
            intrinsics=intrinsics[None, ...],  # [B, 4]
            ii=ii,  # N
            jj=jj,  # N
            model_id=self.video.model_id,
            jacobian=False,
            return_depth=False,
        )
        residual = (coords[0] - target.permute(0, 2, 3, 1)).norm(dim=-1).squeeze(0)
        residual[~valid_mask[0].squeeze().bool()] = 0
        # Collapse [B, 2, H, W] -> [B*N]
        kp_res = residual[:, kps1[0, :, 1].int(), kps1[0, :, 0].int()].flatten()
        # Since we only want to keep N correspondences,
        # only use pixels where both outgoing trajectories are below 2px
        mask = scatter_max(kp_res, kk)[0] < max_residual

        # Get 3D keypoint pointcloud
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])
        kp_3d = points[1, kps1[0, :, 1].int(), kps1[0, :, 0].int()]

        return kp_3d[mask], {
            "keypoints": kps1[:, mask],
            "descriptors": desc1[:, mask],
            "image_size": image_size,
        }

    def plot_keypoints(self, fl: List, images: torch.Tensor) -> None:
        """Plot all 2D keypoints for the image triplet and visualize"""
        import matplotlib.pyplot as plt
        from kornia_moons.viz import visualize_LAF

        fl1, fl2, fl3 = fl
        kp1, kp2, kp3 = fl1["keypoints"], fl2["keypoints"], fl3["keypoints"]
        fig, ax = visualize_LAF(
            K.tensor_to_image(images[0].unsqueeze(0).cpu(), True),
            KF.laf_from_center_scale_ori(kp1.cpu()),
            img_idx=0,
            color="b",
            linewidth=1,
            draw_ori=True,
            fig=None,
            ax=None,
            return_fig_ax=True,
        )
        plt.show()
        visualize_LAF(
            K.tensor_to_image(images[1].unsqueeze(0).cpu(), True),
            KF.laf_from_center_scale_ori(kp2.cpu()),
            img_idx=0,
            color="g",
            linewidth=1,
            draw_ori=True,
        )
        plt.show()
        visualize_LAF(
            K.tensor_to_image(images[2].unsqueeze(0).cpu(), True),
            KF.laf_from_center_scale_ori(kp3.cpu()),
            img_idx=0,
            color="r",
            linewidth=1,
            draw_ori=True,
        )
        plt.show()

    def plot_matches(
        self, fl1, fl2, img1: torch.Tensor, img2: torch.Tensor, matches, save: bool = True, fname: Optional[str] = None
    ):
        """Plot the matches for a single image pair."""
        from kornia_moons.viz import draw_LAF_matches
        import matplotlib.pyplot as plt

        kp1, kp2 = fl1["keypoints"], fl2["keypoints"]
        idxs1 = matches["matches"][0]

        if save:
            fig, ax = draw_LAF_matches(
                KF.laf_from_center_scale_ori(kp1.cpu()),
                KF.laf_from_center_scale_ori(kp2.cpu()),
                idxs1.cpu(),
                K.tensor_to_image(img1.cpu()),
                K.tensor_to_image(img2.cpu()),
                draw_dict={
                    "tentative_color": (1, 1, 0.2, 0.3),
                    "feature_color": None,
                    "vertical": False,
                },
                return_fig_ax=True,
            )
            fig.tight_layout()
            ax.margins(0)
            fig.canvas.draw()
            img_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_from_plot = img_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            if fname is not None:
                fname12 = str(Path(fname).parent / (Path(fname).stem + "_matches_12.png"))
            else:
                fname12 = "matches_12.png"
            self.viz_queue.put((fname12, img_from_plot))
            plt.close(fig)
        else:
            draw_LAF_matches(
                KF.laf_from_center_scale_ori(kp1.cpu()),
                KF.laf_from_center_scale_ori(kp2.cpu()),
                idxs1.cpu(),
                K.tensor_to_image(img1.cpu()),
                K.tensor_to_image(img2.cpu()),
                draw_dict={
                    "tentative_color": (1, 1, 0.2, 0.3),
                    "feature_color": None,
                    "vertical": False,
                },
            )
            plt.show()

    def plot_triplet_matches(
        self, fl: List, images: torch.Tensor, matches1, matches2, save: bool = True, fname: Optional[str] = None
    ) -> None:
        """Plot both pair matches and keypoints for a triplet of frames, as this is what we use to formulate the 3D registration problem."""
        img1, img2, img3 = images
        fl1, fl2, fl3 = fl

        if fname is not None:
            fname12 = str(Path(fname).parent / (Path(fname).stem + "_matches_12.png"))
            fname23 = str(Path(fname).parent / (Path(fname).stem + "_matches_23.png"))
        else:
            fname12, fname23 = "matches_12.png", "matches_23.png"

        self.plot_matches(fl1, fl2, img1, img2, matches1, save, fname12)
        self.plot_matches(fl3, fl2, img3, img2, matches2, save, fname23)
