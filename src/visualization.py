import time
import argparse
import numpy as np
from typing import Dict, Optional
import ipdb
from termcolor import colored
from pathlib import Path
from tqdm import tqdm
import os

import torch
import lietorch
import droid_backends
from lietorch import SE3
import cv2
import open3d as o3d

# Only shows errors, not warnings
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

from matplotlib.pyplot import get_cmap
import matplotlib.pyplot as plt
import matplotlib as mpl

# Ignore warnings [DANGEROUS] (activate this when debugging!)
# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
from .geom import projective_ops as pops

CAM_POINTS = np.array(
    [
        [0, 0, 0],
        [-1, -1, 1.5],
        [1, -1, 1.5],
        [1, 1, 1.5],
        [-1, 1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5],
    ]
)

CAM_LINES = np.array([[1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]])


def plot_3d(rgb: torch.Tensor, depth: torch.Tensor):
    """Use Open3d to plot the 3D point cloud from the monocular depth and input image."""

    def get_calib_heuristic(ht: int, wd: int) -> np.ndarray:
        """On in-the-wild data we dont have any calibration file.
        Since we optimize this calibration as well, we can start with an initial guess
        using the heuristic from DeepV2D and other papers"""
        cx, cy = wd // 2, ht // 2
        fx, fy = wd * 1.2, wd * 1.2
        return fx, fy, cx, cy

    rgb = np.asarray(rgb.cpu())
    depth = np.asarray(depth.cpu())
    invalid = (depth < 0.001).flatten()
    # Get 3D point cloud from depth map
    depth = depth.squeeze()
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.flatten(), y.flatten()
    depth = depth.flatten()

    # Convert to 3D points
    fx, fy, cx, cy = get_calib_heuristic(h, w)
    x3 = (x - cx) * depth / fx
    y3 = (y - cy) * depth / fy
    z3 = depth

    # Convert to Open3D format
    xyz = np.stack([x3, y3, z3], axis=1)
    rgb = np.stack([rgb[0, :, :].flatten(), rgb[1, :, :].flatten(), rgb[2, :, :].flatten()], axis=1)
    depth = depth[~invalid]
    xyz = xyz[~invalid]
    rgb = rgb[~invalid]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Plot the point cloud
    o3d.visualization.draw_geometries([pcd])


def depth2rgb(depths: torch.Tensor, min_depth: float = 0.0, max_depth: float = 5.0) -> np.ndarray:
    """Convert a depth array to a color representation.

    args:
    ---
    depths [torch.Tensor]: Depth tensor of shape (N, H, W) or (H, W)
    """
    if depths.ndim == 2:
        depths = depths.unsqueeze(0)

    depths = depths.cpu().numpy()
    rgb = []
    for depth in depths:
        rgb.append(get_clipped_depth_visualization(depth, min_depth, max_depth))
    return np.asarray(rgb)[..., :3]


def uncertainty2rgb(weights: torch.Tensor, min_val: float = 0.0, cmap: str = "turbo") -> np.ndarray:
    """Convert a depth array to a color representation.

    args:
    ---
    depths [torch.Tensor]: Depth tensor of shape (N, H, W) or (H, W)
    """
    if weights.ndim == 2:
        weights = weights.unsqueeze(0)

    weights = weights.cpu().numpy()
    rgb = []
    for weight in weights:
        rgb.append(array2rgb(weight, cmap=cmap, vmin=min_val))
    return np.asarray(rgb)[..., :3]


def get_clipped_depth_visualization(
    depth: np.ndarray,
    min_depth: float = 0.0,
    max_depth: float = 5.0,
    cmap: str = "Spectral",
) -> np.ndarray:
    """
    Get a color image of a depth/disparity array. This normalizes the values
    and cuts off extreme outliers, so that we get a good picture of the scene geometry.
    Color map can be choosen to be any matplotlib colormap, common choices for depth are
    "magma", "gray", "spectral", "plasma" or "turbo".
    """
    vinds = depth > 0
    depth_rgb = array2rgb(depth, cmap=cmap, vmin=min_depth, vmax=max_depth)
    # Just mark invalid pixels black
    depth_rgb[~vinds] = np.array([0.0, 0.0, 0.0])
    return depth_rgb


def get_normalized_depth_visualization(
    depth: np.ndarray,
    pc: int = 98,
    crop_percent: float = 0,
    cmap: str = "magma",
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Get a color image of a depth/disparity array. This normalizes the values
    and cuts off extreme outliers, so that we get a good picture of the scene geometry.
    Color map can be choosen to be any matplotlib colormap, common choices for depth are
    "magma", "gray" or "turbo".

    NOTE If the depth map is a constant 0.0 aka your model produced garbage, this simply
    returns the input array.
    """

    vinds = depth > 0
    # convert to disparity
    depth = 1.0 / (depth + 1)

    z1 = np.percentile(depth[vinds], pc)
    z2 = np.percentile(depth[vinds], 100 - pc)

    depth = (depth - z2) / ((z1 - z2) + eps)
    depth = np.clip(depth, 0, 1)

    depth_rgb = array2rgb(depth, cmap=cmap)
    # NOTE when we use this function for smoothness maps, we sometimes have an all False array
    if np.all(vinds):
        keep_H = int(depth_rgb.shape[0] * (1 - crop_percent))
    else:
        # Just mark invalid pixels black
        depth_rgb[~vinds] = np.array([0.0, 0.0, 0.0])
        keep_H = int(depth_rgb.shape[0] * (1 - crop_percent))
    return depth_rgb[:keep_H]


def array2rgb(
    im: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "gray",
) -> np.ndarray:
    """
    Convert array to color map, if given limits [vmin, vmax], the values are normalized.

    args:
    ---
    im: Numpy array of shape [H x W], [H x W x 1] or [B x H x W x 1]

    returns:
    ---
    rgb_img: RGB array of shape [H x W x 3] or [B x H x W x 3]
    """
    cmap = get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    rgba_img = cmap(norm(im).astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, -1)
    return rgb_img


def white_balance(img):
    # from https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def create_camera_actor(g, scale=0.05):
    """build open3d camera polydata"""
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES),
    )

    color = (g * 1.0, 0.5 * (1 - g), 0.9 * (1 - g))
    camera_actor.paint_uniform_color(color)
    return camera_actor


def create_point_actor(points, colors):
    """open3d point cloud from numpy array"""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def set_load_view_point(filename):
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    ctr.convert_from_pinhole_camera_parameters(param)


def write_pointclouds(pointclouds: Dict, target_folder: str, ext: str = "ply") -> None:
    """Write out a list of point cloud geometries, so we can reload these in another script."""
    if not os.path.exists(os.path.join(os.getcwd(), target_folder)):
        os.makedirs(os.path.join(os.getcwd(), target_folder))

    for key, pointcloud in tqdm(pointclouds.items()):
        o3d.io.write_point_cloud(
            os.path.join(os.getcwd(), target_folder, str(key).zfill(4) + "." + ext),
            pointcloud,
        )
    return


def write_linesets(linesets: Dict, target_folder: str, ext: str = "ply") -> None:
    """Write out a list of camera actor linesets, so we can reload these in another script."""
    if not os.path.exists(os.path.join(os.getcwd(), target_folder)):
        os.makedirs(os.path.join(os.getcwd(), target_folder))

    for key, lineset in tqdm(linesets.items()):
        o3d.io.write_line_set(
            os.path.join(os.getcwd(), target_folder, str(key).zfill(4) + "." + ext),
            lineset,
        )
    return


def droid_visualization(video, save_root: str = "results", device="cuda:0"):
    """DROID visualization frontend"""

    torch.cuda.set_device(device)
    droid_visualization.video = video
    droid_visualization.cameras = {}
    droid_visualization.points = {}
    droid_visualization.warmup = 8
    droid_visualization.scale = 10.0  # 1.0
    droid_visualization.camera_scale = 0.025
    droid_visualization.ix = 0

    # Thresholds for visualization filtering
    droid_visualization.mv_filter_thresh = 0.005
    droid_visualization.mv_filter_count = 4
    droid_visualization.uncertainty_filter_on = True
    droid_visualization.unc_filter_thresh = 0.2

    droid_visualization.do_reset = True

    def increase_mv_filter(vis):
        droid_visualization.mv_filter_thresh *= 2
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[: droid_visualization.video.counter.value] = True

    def decrease_mv_filter(vis):
        droid_visualization.mv_filter_thresh *= 0.5
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[: droid_visualization.video.counter.value] = True

    def increase_unc_filter(vis):
        droid_visualization.unc_filter_thresh *= 2
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[: droid_visualization.video.counter.value] = True

    def decrease_unc_filter(vis):
        droid_visualization.unc_filter_thresh *= 0.5
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[: droid_visualization.video.counter.value] = True

    def increase_mv_count(vis):
        droid_visualization.mv_filter_count += 1
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[: droid_visualization.video.counter.value] = True

    def decrease_mv_count(vis):
        # Count should be > 1!
        if droid_visualization.mv_filter_count > 1:
            droid_visualization.mv_filter_count -= 1
            with droid_visualization.video.get_lock():
                droid_visualization.video.dirty[: droid_visualization.video.counter.value] = True

    def deactivate_update(vis):
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:] = False

    def deactivate_uncertainty(vis):
        droid_visualization.uncertainty_filter_on = False

    def increase_camera(vis):
        droid_visualization.camera_scale *= 1 / 0.8
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[: droid_visualization.video.counter.value] = True

    def decrease_camera(vis):
        droid_visualization.camera_scale *= 0.8
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[: droid_visualization.video.counter.value] = True

    def start_stop_view_resetting(vis):
        droid_visualization.do_reset = not droid_visualization.do_reset

    @torch.no_grad()
    def animation_callback(vis):
        cam_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

        with video.get_lock():
            t = video.counter.value
            (dirty_index,) = torch.where(video.dirty.clone())
            dirty_index = dirty_index

        if len(dirty_index) == 0:
            return

        video.dirty[dirty_index] = False

        poses = torch.index_select(video.poses, 0, dirty_index)
        disps = torch.index_select(video.disps, 0, dirty_index)
        # convert poses to 4x4 matrix
        Ps = SE3(poses).inv().matrix().cpu().numpy()
        images = torch.index_select(video.images, 0, dirty_index)
        images = images.cpu()[:, ..., 3::8, 3::8].permute(0, 2, 3, 1)
        points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()

        thresh = droid_visualization.mv_filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))
        count = droid_backends.depth_filter(video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)
        count, disps = count.cpu(), disps.cpu()
        # Only keep points that are consistent across multiple views and not too close by
        masks = (count >= droid_visualization.mv_filter_count) & (disps > 0.5 * disps.mean(dim=[1, 2], keepdim=True))

        if droid_visualization.uncertainty_filter_on:
            weights = torch.index_select(video.uncertainty, 0, dirty_index)
            masks2 = weights > droid_visualization.unc_filter_thresh
            masks = masks & masks2.cpu()

        for i in range(len(dirty_index)):
            pose = Ps[i]
            ix = dirty_index[i].item()

            if ix in droid_visualization.cameras:
                vis.remove_geometry(
                    droid_visualization.cameras[ix],
                    reset_bounding_box=droid_visualization.do_reset,
                )
                del droid_visualization.cameras[ix]

            if ix in droid_visualization.points:
                vis.remove_geometry(
                    droid_visualization.points[ix],
                    reset_bounding_box=droid_visualization.do_reset,
                )
                del droid_visualization.points[ix]

            ### add camera actor ###
            cam_actor = create_camera_actor(True, droid_visualization.camera_scale)
            cam_actor.transform(pose)
            vis.add_geometry(cam_actor, reset_bounding_box=droid_visualization.do_reset)
            droid_visualization.cameras[ix] = cam_actor

            mask = masks[i].reshape(-1)
            pts = points[i].reshape(-1, 3)[mask].numpy()
            clr = images[i].reshape(-1, 3)[mask].numpy()

            ## add point actor ###
            point_actor = create_point_actor(pts, clr)
            vis.add_geometry(point_actor, reset_bounding_box=droid_visualization.do_reset)
            droid_visualization.points[ix] = point_actor

        # hack to allow interacting with vizualization during inference
        # if len(droid_visualization.cameras) >= droid_visualization.warmup:
        #     cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

        droid_visualization.ix += 1

        cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam_params, True)
        vis.poll_events()
        vis.update_renderer()

    ### create Open3D visualization ###
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)
    vis.register_key_callback(ord("S"), increase_mv_filter)
    vis.register_key_callback(ord("A"), decrease_mv_filter)
    vis.register_key_callback(ord("V"), increase_mv_count)
    vis.register_key_callback(ord("B"), decrease_mv_count)
    vis.register_key_callback(ord("F"), increase_unc_filter)
    vis.register_key_callback(ord("G"), decrease_unc_filter)
    vis.register_key_callback(ord("M"), increase_camera)
    vis.register_key_callback(ord("N"), decrease_camera)
    vis.register_key_callback(ord("Q"), deactivate_update)
    vis.register_key_callback(ord("U"), deactivate_uncertainty)
    vis.register_key_callback(ord("R"), start_stop_view_resetting)

    vis.create_window(height=540, width=960)
    vis.get_render_option().load_from_json("src/renderoption.json")

    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()

    ### Store the geometry and trajectory visualizaton for later inspection when done
    print(colored("[Visu] Saving the visualization for later usage ...!", "grey"))
    try:
        o3d.io.write_pinhole_camera_parameters("final_viewpoint.json", param)
        pcl_path = str(Path(save_root) / "pointclouds")
        cam_path = str(Path(save_root) / "cameras")
        write_pointclouds(droid_visualization.points, pcl_path, ext="xyzrgb")
        write_linesets(droid_visualization.cameras, cam_path)
    except Exception as e:
        print(colored("[Visu] Something went wrong when saving the visualization ...!", "red"))
        print(colored(e, "red"))

    vis.destroy_window()
