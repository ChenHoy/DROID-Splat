import json
import os

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .gaussian_renderer import render
from .utils.image_utils import psnr
from .utils.loss_utils import ssim
from .utils.system_utils import mkdir_p
from .logging_utils import Log
from .multiprocessing_utils import clone_obj

def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    # traj_est_aligned = trajectory.align_trajectory(
    #     traj_est, traj_ref, correct_scale=monocular
    # )
    traj_est_aligned = clone_obj(traj_est) 
    traj_est_aligned.align(traj_ref, correct_scale=monocular) ## throws 
    # traj_est_aligned.align_origin(traj_ref)

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)

    return ape_stats


def get_c2w_list(camera_trajectory, stream):
    '''
    Gets the camera trajectory and the stream of poses and returns the estimated and gt poses for
    all frames after trajectory filler
    '''
    estimate_c2w_list = camera_trajectory.matrix().data.cpu()
    traj_ref = []
    for i in range(len(stream.poses)):
        val = stream.poses[i].sum()
        if np.isnan(val) or np.isinf(val):
            print(f"Nan or Inf found in gt poses, skipping {i}th pose!")
            continue
        traj_ref.append(stream.poses[i])

    gt_c2w_list = torch.from_numpy(np.stack(traj_ref, axis=0))

    return estimate_c2w_list, gt_c2w_list


def eval_ate(video, kf_ids, save_dir, iterations, final=False, monocular=False, 
             keyframes_only=True,
             camera_trajectory=None,
             stream=None):
    '''
    video: DepthVideo
    kf_ids: list of keyframe indices | in case of the video object all of them are keyframes
    used poses_gt for gt and poses for estimated poses
    stream: Dataset
    camera_trajectory: poses after trajectory filler
    '''
    frames = video.images

    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    if keyframes_only:

        for kf_id in kf_ids:
            kf = frames[kf_id] 

            _, _, c2w, _, _ = video.get_mapping_item(kf_id, video.device)
            w2c = torch.inverse(c2w)
            R_est = w2c[:3, :3].unsqueeze(0).detach()
            T_est = w2c[:3, 3].detach()
                
            gt_c2w = video.poses_gt[kf_id].clone().to(video.device)  # [4, 4]
            gt_w2c = torch.inverse(gt_c2w)
            R_gt = gt_w2c[:3, :3].unsqueeze(0).detach()
            T_gt = gt_w2c[:3, 3].detach()

            pose_est = np.linalg.inv(gen_pose_matrix(R_est, T_est))
            pose_gt = np.linalg.inv(gen_pose_matrix(R_gt, T_gt))

            # trj_id.append(frames[kf_id].uid)
            trj_est.append(pose_est.tolist())
            trj_gt.append(pose_gt.tolist())

            trj_est_np.append(pose_est)
            trj_gt_np.append(pose_gt)

        # trj_data["trj_id"] = trj_id
        trj_data["trj_est"] = trj_est
        trj_data["trj_gt"] = trj_gt

        plot_dir = os.path.join(save_dir, "plot")
        mkdir_p(plot_dir)

        label_evo = "final" if final else "{:04}".format(iterations)
        with open(
            os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(trj_data, f, indent=4)

        ate = evaluate_evo(
            poses_gt=trj_gt_np,
            poses_est=trj_est_np,
            plot_dir=plot_dir,
            label=label_evo,
            monocular=monocular,
        )
        return ate
    
    else:
        estimate_c2w_list, gt_c2w_list = get_c2w_list(camera_trajectory, stream)

        assert len(estimate_c2w_list) == len(gt_c2w_list), "Length of estimated and gt poses should be same"

        no_poses = len(estimate_c2w_list)

        for i in range(no_poses):
            c2w_est = estimate_c2w_list[i]
            c2w_gt = gt_c2w_list[i]

            w2c_est = torch.inverse(c2w_est)
            w2c_gt = torch.inverse(c2w_gt)

            R_est = w2c_est[:3, :3].unsqueeze(0).detach()
            T_est = w2c_est[:3, 3].detach()

            R_gt = w2c_gt[:3, :3].unsqueeze(0).detach()
            T_gt = w2c_gt[:3, 3].detach()

            pose_est = np.linalg.inv(gen_pose_matrix(R_est, T_est))
            pose_gt = np.linalg.inv(gen_pose_matrix(R_gt, T_gt))

            # trj_id.append(frames[kf_id].uid)
            trj_est.append(pose_est.tolist())
            trj_gt.append(pose_gt.tolist())

            trj_est_np.append(pose_est)
            trj_gt_np.append(pose_gt)

        trj_data["trj_est"] = trj_est
        trj_data["trj_gt"] = trj_gt

        plot_dir = os.path.join(save_dir, "plot")
        mkdir_p(plot_dir)

        label_evo = "final" if final else "{:04}".format(iterations)
        with open(
            os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(trj_data, f, indent=4)

        ate = evaluate_evo(
            poses_gt=trj_gt_np,
            poses_est=trj_est_np,
            plot_dir=plot_dir,
            label=label_evo,
            monocular=monocular,
        )
        return ate
             
        


def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
):
    '''
    mapper: GaussianMapper
    '''
    interval = 1
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")

    '''
    Runs this only for frames that are not keyframes
    '''
    print("Calculating metrics on non-keyframes. Total keyframes: ", len(kf_indices),"Step used for evaluation: ", interval)
    for idx in range(0, end_idx, interval):
        if idx in kf_indices:
            continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        ## gt_image is an rgb image no depth
        _, gt_image, _, _, _ = dataset[idx] 

        gt_image = gt_image.squeeze(0)

        rendering = render(frame, gaussians, pipe, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0).to("cuda"), (gt_image[mask]).unsqueeze(0).to("cuda"))
        ssim_score = ssim((image).unsqueeze(0).to("cuda"), (gt_image).unsqueeze(0).to("cuda"))
        lpips_score = cal_lpips((image).unsqueeze(0).to("cuda"), (gt_image).unsqueeze(0).to("cuda"))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
