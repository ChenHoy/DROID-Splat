import json
from termcolor import colored
from tqdm import tqdm
import ipdb
import pickle
from typing import List, Dict, Optional, Tuple
from omegaconf import DictConfig
import os

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from lietorch import SE3
from ..geom import matrix_to_lie

from evo.core import metrics, sync

# NOTE chen: MonoGS uses PosePath3D, everyone else uses PoseTrajectory3D which seems more compatible with our video structure
from evo.core.trajectory import PoseTrajectory3D
from evo.tools.plot import PlotMode, prepare_axis, traj, traj_colormap
from matplotlib import pyplot as plt

from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt

from .gaussian_renderer import render
from .scene.gaussian_model import GaussianModel
from .camera_utils import Camera
from ..depth_video import DepthVideo
from ..losses.image import ssim  # TODO chen: refactor these by simply importing them in __init__ of submodule?
from ..losses.misc import l1_loss
from ..losses.depth import ScaleAndShiftInvariantLoss
from ..utils import psnr, mkdir_p, clone_obj


class EvaluatePacket:
    """This class is used to pass data from the gaussian_mapper to the main thread for evaluation."""

    def __init__(
        self,
        pipeline_params: Dict = None,
        background: torch.Tensor = None,
        gaussians: GaussianModel = None,
        cameras: List[Camera] = None,
        timestamps: torch.Tensor = None,
        cam2buffer: Dict = None,
        buffer2cam: Dict = None,
    ):
        self.has_gaussians = False
        if gaussians is not None:
            self.has_gaussians = True
            self.get_xyz = gaussians.get_xyz.detach().clone()
            self.active_sh_degree = gaussians.active_sh_degree
            self.get_opacity = gaussians.get_opacity.detach().clone()
            self.get_scaling = gaussians.get_scaling.detach().clone()
            self.get_rotation = gaussians.get_rotation.detach().clone()
            self.max_sh_degree = gaussians.max_sh_degree
            self.get_features = gaussians.get_features.detach().clone()

            self._rotation = gaussians._rotation.detach().clone()
            self.rotation_activation = torch.nn.functional.normalize
            self.unique_kfIDs = gaussians.unique_kfIDs.clone()
            self.n_obs = gaussians.n_obs.clone()

        self.pipeline_params = pipeline_params
        self.background = background
        self.cameras = cameras
        self.timestamps = timestamps
        self.gaussians = gaussians
        self.cam2buffer = cam2buffer
        self.buffer2cam = buffer2cam

    def cameras_to(self, device: str) -> None:
        """In case we have a lot of images, it is better to move all dense image tensors from the device to e.g.
        the CPU. We usually run into these problems when we clone the EvaluatePacket in the main process for clean handling.
        """
        for cam in self.cameras:
            cam.image_tensors_to(device)

    def __str__(self) -> str:
        print("Im getting something: {} {}".format(self.pipeline_params, len(self.cameras)))


def create_odometry_csv(results_kf: Dict, results_all: Dict, cfg: DictConfig, input_path: str) -> Dict:
    csv_dict = {
        "ate_on_keyframes_only": [True, False],
        "run_backend": [str(cfg.run_backend), str(cfg.run_backend)],
        "run_mapping": [str(cfg.run_mapping), str(cfg.run_mapping)],
        "stride": [str(cfg.stride), str(cfg.stride)],
        "loop_closure": [
            str(cfg.tracking.backend.use_loop_closure),
            str(cfg.tracking.backend.use_loop_closure),
        ],
        "loop_detector": [str(cfg.run_loop_detection), str(cfg.run_loop_detection)],
        "dataset": [input_path, input_path],
        "mode": [cfg.mode, cfg.mode],
        "ape": [results_kf["mean"], results_all["mean"]],
        "ate": [results_kf["rmse"], results_all["rmse"]],
    }
    return csv_dict


def create_rendering_csv(results_kf, results_nonkf, cfg: DictConfig, input_path: str) -> Dict:
    csv_dict = {
        "run_backend": [str(cfg.run_backend), str(cfg.run_backend)],
        "run_mapping": [str(cfg.run_mapping), str(cfg.run_mapping)],
        "stride": [str(cfg.stride), str(cfg.stride)],
        "loop_closure": [str(cfg.tracking.backend.use_loop_closure), str(cfg.tracking.backend.use_loop_closure)],
        "loop_detector": [str(cfg.run_loop_detection), str(cfg.run_loop_detection)],
        "dataset": [input_path, input_path],
        "mode": [cfg.mode, cfg.mode],
        "psnr": [results_kf["mean_psnr"], results_nonkf["mean_psnr"]],
        "ssim": [results_kf["mean_ssim"], results_nonkf["mean_ssim"]],
        "lpips": [results_kf["mean_lpips"], results_nonkf["mean_lpips"]],
        "extra_non_kf": [
            str(cfg.mapping.refinement.sampling.use_non_keyframes),
            str(cfg.mapping.refinement.sampling.use_non_keyframes),
        ],
        "eval_on_keyframes": [True, False],
    }
    return csv_dict


### Odometry ###


def evaluate_evo(
    poses_est: List[np.ndarray],
    poses_gt: List[np.ndarray],
    timestamps: List[int],
    save_dir: str,
    label: str,
    monocular: bool = False,
) -> Dict:
    """Evaluate the odometry using the evo package. This expect a list/ an array of poses in c2w convention, i.e. you have
    to invert the direct outputs from DROID-SLAM convention.

    NOTE The plotting functionality of evo expects c2w convention.
    """

    plot_dir = os.path.join(save_dir, "plots")
    mkdir_p(plot_dir)

    # NOTE chen: MonoGS uses PosePath3D, where we need to supply 4x4 homogeneous matrices, others use se3 liealgebra directly in PoseTrajectory3D
    # traj_est, traj_ref  = PosePath3D(poses_se3=poses_est), PosePath3D(poses_se3=poses_gt)
    traj_est = PoseTrajectory3D(
        positions_xyz=poses_est[:, :3], orientations_quat_wxyz=poses_est[:, 3:], timestamps=np.array(timestamps)
    )
    traj_ref = PoseTrajectory3D(
        positions_xyz=poses_gt[:, :3], orientations_quat_wxyz=poses_gt[:, 3:], timestamps=np.array(timestamps)
    )

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)  # This synchronizes according to timestamps
    # Scale correct monocular odometry for a fair comparison if needed
    # NOTE chen: monocular can sometimes even be better than RGBD due to the adjustment
    traj_est_aligned = clone_obj(traj_est)
    # traj_est_aligned.align_origin(traj_ref) # this only aligns the origins

    # This computes an SE3 transform to register est on ref
    # (for monocular it contains an additional scale, i.e. sim3 transform)
    traj_est_aligned.align(traj_ref, correct_scale=monocular)

    # Get APE statistics
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    data = (traj_ref, traj_est_aligned)
    ape_metric.process_data(data)
    rmse = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    if monocular:
        print(colored(f"[Eval] scaled RMSE ATE [m]: {rmse}", "red"))  ## Andrei NOTE: this is in m not cm
    else:
        print(colored(f"[Eval] RMSE ATE [m]: {rmse}", "red"))  ## Andrei NOTE: this is in m not cm
    ape_stats = ape_metric.get_all_statistics()
    # NOTE chen: this sometimes contains a numpy.float32 instead of normal float :/
    for key, value in ape_stats.items():
        ape_stats[key] = float(value)

    with open(os.path.join(save_dir, "stats_{}.json".format(str(label))), "w", encoding="utf-8") as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = PlotMode.xy
    fig = plt.figure()
    ax = prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {rmse}")
    traj(ax, plot_mode, traj_ref, "--", "gray", "gt", plot_start_end_markers=True)
    traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
        plot_start_end_markers=True,
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)

    return ape_stats


def get_gt_c2w_from_stream(stream) -> torch.Tensor:
    """Get all 4x4 homogenous matrices in c2w format from the dataset stream.
    We transform these into a (N x 7 x 1) lie vector.
    """
    poses = np.stack(stream.poses)
    if np.isnan(poses).any() or np.isinf(poses).any():
        raise Exception(colored(f"Error. Nan or Inf found in gt poses!", "red"))
    return matrix_to_lie(torch.from_numpy(poses))


def get_odometry_from_video(video: DepthVideo) -> torch.Tensor:
    """Extract the whole odometry from the video object for both our estimated poses and the groundtruth"""

    trj_est, trj_gt = [], []
    for i in range(video.counter.value):
        _, _, _, _, c2w_est, _ = video.get_mapping_item(i, use_gt=False, device=video.device)
        _, _, _, _, c2w_gt, _ = video.get_mapping_item(i, use_gt=True, device=video.device)
        if torch.abs(c2w_gt.vec().sum()) < 1e-7:
            raise ValueError("Groundtruth pose is zero. Video object likely does not have any gt poses!")
        trj_est.append(c2w_est.vec().detach().cpu().numpy())
        trj_gt.append(c2w_gt.vec().detach().cpu().numpy())

    return np.stack(trj_est), np.stack(trj_gt)


def write_out_kitti_style(
    traj: List[np.ndarray] | np.ndarray, poses_in: str = "matrix", outfile: str = "test.txt"
) -> None:
    """Given a list of 4x4 homogeneous matrices, write out the poses in KITTI style format.
    For each pose, we write a line in a .txt file as follows:
        a b c d
        e f g h -> a b c d e f g h i j k l
        i j k l
        0 0 0 1
    """
    with open(outfile, "w") as f:
        for pose in traj:
            if poses_in == "matrix":
                pose = pose.flatten()
            elif poses_in == "lie":
                pose = SE3.InitFromVec(torch.from_numpy(pose)).matrix().numpy().flatten()
            else:
                raise Exception(
                    "Unknown pose format! Please provide them either as a 4x4 homogeneous matrix or as a 7x1 lie element"
                )

            for i in range(12):
                if i == 11:
                    f.write(str(pose[i]))  # Dont leave a trailing space
                else:
                    f.write(str(pose[i]) + " ")
            f.write("\n")


def eval_ate(
    traj_est: np.ndarray,
    traj_gt: np.ndarray,
    timestamps: List | np.ndarray,
    save_dir: str,
    monocular: bool = False,
):
    """
    Evaluate the absolute trajectory error by comparing the estimated camera odometry with a groundtruth reference.

    args:
    ---
        traj_est, traj_gt:  Trajectories of shape (B, 7, 1) in c2w format
        timestamps:         List of timestamps for each frame. These are the global id's in the video!
        save_dir:           Where to save the evaluation results
        monocular:          Whether the odometry is monocular or not. If yes, then we scale adjust the estimates to the gt
    """
    assert traj_est.shape == traj_gt.shape, "Trajectories should have the same shape!"
    assert len(timestamps) == len(traj_est), "Timestamps should have the same length as the trajectories!"

    # Write out serialized string to read later
    trj_data = {"trj_est": traj_est.tolist(), "trj_gt": traj_gt.tolist()}
    with open(os.path.join(save_dir, f"trj_final.json"), "w", encoding="utf-8") as f:
        json.dump(trj_data, f, indent=4)
    ate = evaluate_evo(
        poses_est=traj_est,
        poses_gt=traj_gt,
        timestamps=timestamps,
        save_dir=save_dir,
        label="final",
        monocular=monocular,
    )
    return ate


### Rendering ###


def create_comparison_figure(
    gt_img: np.ndarray,
    est_img: np.ndarray,
    gt_depth: Optional[np.ndarray] = None,
    est_depth: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Create a plot with both gt and estimate images side by side"""

    # Sanity
    assert gt_img.shape == est_img.shape, "Both images should have the same shape!"
    assert gt_img.dtype == est_img.dtype, "Both images should have the same dtype!"
    if gt_depth is not None:
        assert est_depth is not None, "Both gt and estimated depth should be provided!"

        fig, axes = plt.subplots(2, 2, figsize=(10, 5))
        # Display the ground truth image
        axes[0, 0].imshow(gt_img.squeeze()[..., ::-1])
        axes[0, 0].set_title("Ground Truth")
        axes[0, 0].axis("off")

        # Display the predicted image
        axes[0, 1].imshow(est_img.squeeze()[..., ::-1])
        axes[0, 1].set_title("Prediction")
        axes[0, 1].axis("off")

        min_depth = min(gt_depth.min(), est_depth.min())
        max_depth = max(gt_depth.max(), est_depth.max())

        # Display the gt depth
        axes[1, 0].imshow(gt_depth.squeeze(), cmap="Spectral", vmin=min_depth, vmax=max_depth)
        axes[1, 0].set_title("Groundtruth")
        axes[1, 0].axis("off")

        # Display the predicted depth
        axes[1, 1].imshow(est_depth.squeeze(), cmap="Spectral", vmin=min_depth, vmax=max_depth)
        axes[1, 1].set_title("Prediction")
        axes[1, 1].axis("off")

    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Display the ground truth image
        axes[0].imshow(gt_img.squeeeze())
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        # Display the predicted image
        axes[1].imshow(est_img.squeeze())
        axes[1].set_title("Prediction")
        axes[1].axis("off")

    return fig


def save_dense_predictions(
    save_dir: str, idx: int, est_depth: torch.Tensor, est_img: torch.Tensor, gt_depth: Optional[torch.Tensor] = None
) -> None:
    fig1, ax1 = plt.subplots(1, 1)
    # Display the ground truth image
    ax1.imshow(est_img.squeeze()[..., ::-1])
    ax1.axis("off")
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(1, 1)
    # Display the ground truth image
    if gt_depth is not None:
        min_depth, max_depth = gt_depth.min(), gt_depth.max()
    else:
        min_depth, max_depth = est_depth.min(), est_depth.max()
    ax2.imshow(est_depth.squeeze(), cmap="Spectral", vmin=min_depth, vmax=max_depth)
    ax2.axis("off")
    fig2.tight_layout()

    fig1.savefig(os.path.join(save_dir, f"est_img_{str(idx).zfill(4)}.png"))
    fig2.savefig(os.path.join(save_dir, f"est_depth_{str(idx).zfill(4)}.png"))
    plt.close(fig1)
    plt.close(fig2)


def plot_metric_statistics(psnr_array, ssim_array, lpips_array, plot_dir: str):
    """1 bar plot per metric"""

    frames = np.arange(len(psnr_array))

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].bar(frames, psnr_array, color="blue")
    ax[0].set_title("PSNR")
    ax[0].set_ylim([0.0, max(psnr_array) + 1])

    ax[1].bar(frames, ssim_array, color="green")
    ax[1].set_ylim([0.99, 1.0])
    ax[1].set_title("SSIM")

    ax[2].bar(frames, lpips_array, color="red")
    ax[2].set_title("LPIPS")
    ax[2].set_ylim([0.0, max(lpips_array) + 0.005])

    plt.savefig(os.path.join(plot_dir, "metrics_per_frame.pdf"))


def save_gaussians(gaussians: GaussianModel, save_dir: str, iteration: Optional[int] = None) -> None:
    if iteration is not None:
        point_cloud_path = os.path.join(save_dir, "point_cloud/iteration_{}".format(str(iteration)))
    else:
        point_cloud_path = os.path.join(save_dir, "point_cloud/final")
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


def torch_intersect1d(t1: torch.Tensor, t2: torch.Tensor):
    assert t1.dim() == 1 and t2.dim() == 1, "t1, t2 should be 1D Tensors"
    # NOTE: requires t1, t2 to be unique 1D Tensor in advance.
    # Method: based on unique's count
    num_t1, num_t2 = t1.numel(), t2.numel()
    u, inv, cnt = torch.unique(torch.cat([t1, t2]), return_counts=True, return_inverse=True)

    cnt_12 = cnt[inv]
    cnt_t1, cnt_t2 = cnt_12[:num_t1], cnt_12[num_t1:]
    m_t1 = cnt_t1 == 2
    inds_t1 = m_t1.nonzero()[..., 0]
    inds_t1_exclusive = (~m_t1).nonzero()[..., 0]
    inds_t2_exclusive = (cnt_t2 == 1).nonzero()[..., 0]

    intersection = t1[inds_t1]
    t1_exclusive = t1[inds_t1_exclusive]
    t2_exclusive = t2[inds_t2_exclusive]
    return intersection, t1_exclusive, t2_exclusive


def do_odometry_evaluation(
    eval_path: str,
    est_c2w_kf_lie: np.ndarray,
    gt_c2w_kf_lie: np.ndarray,
    est_c2w_all_lie: np.ndarray,
    gt_c2w_all_lie: np.ndarray,
    tstamps: List[int],
    kf_tstamps: List[int],
    monocular: bool,
):
    """Perform evaluation both on keyframe trajectory and the whole trajectory."""
    ### Get the numbers for keyframes only
    kf_eval_path = os.path.join(eval_path, "odometry", "keyframes")
    mkdir_p(kf_eval_path)

    kf_result_ate = eval_ate(est_c2w_kf_lie, gt_c2w_kf_lie, kf_tstamps, save_dir=kf_eval_path, monocular=monocular)
    kf_trajectory_df = pd.DataFrame([kf_result_ate])
    kf_trajectory_df.to_csv(os.path.join(kf_eval_path, "kf_trajectory_results.csv"), index=False)
    # NOTE chen: you can use this file to directly visualize the trajectory using evo
    write_out_kitti_style(est_c2w_kf_lie, poses_in="lie", outfile=os.path.join(kf_eval_path, "kf_est_c2w.txt"))

    ### Get the numbers for the whole trajectory
    all_eval_path = os.path.join(eval_path, "odometry", "all")
    mkdir_p(all_eval_path)

    all_result_ate = eval_ate(est_c2w_all_lie, gt_c2w_all_lie, tstamps, save_dir=all_eval_path, monocular=monocular)
    all_trajectory_df = pd.DataFrame([all_result_ate])
    all_trajectory_df.to_csv(os.path.join(all_eval_path, "all_trajectory_results.csv"), index=False)
    # NOTE chen: you can use this file to directly visualize the trajectory using evo
    write_out_kitti_style(est_c2w_all_lie, poses_in="lie", outfile=os.path.join(all_eval_path, "est_c2w.txt"))
    return kf_result_ate, all_result_ate


def compute_scale_and_shift(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(0)
    if target.ndim == 2:
        target = target.unsqueeze(0)
    # Increase precision because we sum over potentially large arrays
    target, prediction, mask = target.double(), prediction.double(), mask.double()

    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0.squeeze().float(), x_1.squeeze().float()


def eval_rendering(
    cams: List[Camera],
    tstamps: List[int],
    gaussians: GaussianModel,
    dataset,
    render_pipeline_cfg: Dict,
    background: torch.Tensor,
    save_dir: str,
    save_every: int = 1,
    monocular: bool = True,
    save_renders: bool = True,
    save_predictions: bool = True,
):
    """Evaluate the rendering quality of the estimated Scene model and Camera poses by comparing with the dataset groundtruth.

    This function loops over a list of Camera objects, which store an estimated pose. The timestamps are a list of equal size, which
    correspond to the frame indices in the dataset. We use these indices to get the groundtruth depth and image and compare our rendered
    estimates.

    If monocular, we will compute a scale-invariant l1 loss between the rendered depth and gt depth, otherwise we directly compare the depths.
    """
    # Collect all the frames
    img_pred, img_gt, depth_pred, depth_gt, saved_frame_idx = [], [], [], [], []
    psnr_array, ssim_array, lpips_array, depth_l1 = [], [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")

    dataset.return_stat_masks = False  # Dont return dynamic object masks here
    if dataset.depth_paths is not None and len(dataset.depth_paths) > 0:
        has_gt_depth = True
    else:
        has_gt_depth = False

    plot_dir = os.path.join(save_dir, "plots")
    print(colored(f"[Evaluation] Saving Rendering evaluation in: {save_dir}", "green"))
    mkdir_p(save_dir)
    mkdir_p(plot_dir)

    for i, idx in tqdm(enumerate(tstamps)):

        if i % save_every != 0:
            continue

        saved_frame_idx.append(idx)
        cam = cams[i]  # NOTE chen: Make sure that the order of tstamps and cams is the same and corresponding!
        # NOTE we detach tensors to the CPU, because for some scenes we have a lot of images and we want to save memory
        cam.image_tensors_to("cuda")  # Make sure everything is on the GPU for Rendering
        _, gt_image, gt_depth, _, _ = dataset[idx]
        gt_image = gt_image.float().squeeze(0) / 255.0  # Uint8 -> float conversion on demand
        if has_gt_depth:
            gt_depth = gt_depth.squeeze(0)

        render_dict = render(cam, gaussians, render_pipeline_cfg, background)
        image_est, depth_est = render_dict["render"], render_dict["depth"]
        image_est = torch.clamp(image_est, 0.0, 1.0)

        ## Conversion
        gt_img_np = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        est_img_np = (image_est.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        gt_img_np = cv2.cvtColor(gt_img_np, cv2.COLOR_BGR2RGB)  # Convert to RGB since we load images with cv2
        est_img_np = cv2.cvtColor(est_img_np, cv2.COLOR_BGR2RGB)
        img_pred.append(est_img_np)
        img_gt.append(gt_img_np)
        # If we have groundtruth depth
        if has_gt_depth:
            depth_est, gt_depth = depth_est.detach().cpu(), gt_depth.cpu()
            depth_pred.append(depth_est)
            depth_gt.append(gt_depth)

        ### Plot a comparison for inspection
        if save_renders:
            if has_gt_depth:
                fig = create_comparison_figure(gt_img_np, est_img_np, gt_depth.cpu().numpy(), depth_est.cpu().numpy())
            else:
                fig = create_comparison_figure(gt_img_np, est_img_np)
            plt.savefig(os.path.join(plot_dir, "rendered_vs_gt_" + str(idx) + ".png"))
            plt.close(fig)
        if save_predictions:
            if has_gt_depth:
                if monocular:
                    # Align depth with groundtruth, so we have a consistent color scheme
                    valid = gt_depth > 0
                    scale, shift = compute_scale_and_shift(depth_est, gt_depth, valid)
                    depth_est_visu = (depth_est * scale + shift).cpu().numpy()
                else:
                    depth_est_visu = depth_est.cpu().numpy()
                save_dense_predictions(save_dir, idx, depth_est_visu, est_img_np, gt_depth.cpu().numpy())
            else:
                save_dense_predictions(save_dir, idx, depth_est.cpu().numpy(), est_img_np)

        ### Image similarity metrics
        valid_img = gt_image > 0
        psnr_score = psnr(
            (image_est[valid_img]).unsqueeze(0).to("cuda"), (gt_image[valid_img]).unsqueeze(0).to("cuda")
        )
        ssim_score = ssim((image_est).unsqueeze(0).to("cuda"), (gt_image).unsqueeze(0).to("cuda"))
        lpips_score = cal_lpips((image_est).unsqueeze(0).to("cuda"), (gt_image).unsqueeze(0).to("cuda"))
        # Gather scores
        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

        ### Depth Similarity metrics
        if has_gt_depth:
            valid_depth = gt_depth > 0
            if monocular:
                loss_func = ScaleAndShiftInvariantLoss()
            else:
                loss_func = l1_loss
            depth_loss = loss_func(depth_est, gt_depth, mask=valid_depth)
            depth_l1.append(depth_loss.item())

    plot_metric_statistics(psnr_array, ssim_array, lpips_array, plot_dir)

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    # Print this in pretty so we can see it
    loss_str = "[Eval] mean PSNR: {}, SSIM: {}, LPIPS: {}".format(
        output["mean_psnr"], output["mean_ssim"], output["mean_lpips"]
    )
    rnd_statistics = {"psnr": psnr_array, "ssim": ssim_array, "lpips": lpips_array}
    if has_gt_depth:
        output["mean_l1"] = float(np.mean(depth_l1))
        loss_str += ", L1 (depth): {}".format(output["mean_l1"])
        rnd_statistics["l1"] = depth_l1
    print(colored(loss_str, "red"))

    with open(os.path.join(save_dir, "frame_statistics.pkl"), "wb") as f:
        pickle.dump(rnd_statistics, f)
    json.dump(output, open(os.path.join(save_dir, "final_result.json"), "w", encoding="utf-8"), indent=4)
    return output
