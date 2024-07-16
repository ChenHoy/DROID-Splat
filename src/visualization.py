import time
import argparse
import numpy as np
from typing import Dict, Optional, List
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
# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

from matplotlib.pyplot import get_cmap
import matplotlib as mpl

# mpl.use("Qt5Agg")

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


def make_colorwheel() -> np.ndarray:
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def flow_compute_color(u: np.ndarray, v: np.ndarray, convert_to_bgr: bool = False) -> np.ndarray:
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    args:
    ---
    u [np.ndarray]:  input horizontal flow of shape [H x W]
    v [np.ndarray]:  input vertical flow of shape [H x W]
    convert_to_bgr [bool]: whether to change ordering and output BGR instead of RGB

    returns:
    ---
    flow_image [np.ndarray]: RGB image of the flow field of shape [H x W x 3] with range [0, 255]
    """

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    nan_idx = np.isnan(u) | np.isnan(v)
    u[nan_idx] = v[nan_idx] = 0

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    for i in range(colorwheel.shape[1]):

        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    return flow_image


def get_rgb_from_np_optical_flow(flow_uv: np.ndarray, clip_flow: float = None, convert_to_bgr: bool = False):
    """
    Convert a flow vector field into a color visualization of magnitude and direction.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    args:
    ---
    flow_uv [np.ndarray]: Flow array of shape [H,x W x 2]
    clip_flow [float]: maximum clipping value for flow

    returns:
    ---
    flow_image [np.ndarray]: RGB color image of the flow field of shape [H x W x 3] with range [0, 255]
    """

    assert flow_uv.ndim == 3, "input flow must have three dimensions"
    assert flow_uv.shape[2] == 2, "input flow must have shape [H,W,2]"

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)


def opticalflow2rgb(flows: torch.Tensor) -> List[np.ndarray]:
    """Convert an optical flow field (u, v) to a color representation.

    args:
    ---
    flows [torch.Tensor]: a (N, 2, H, W) or (2, H, W) optical flow tensor.
    """
    # We have actual video data
    if flows.ndim == 3:
        flows = flows.unsqueeze(0)

    flows = flows.cpu().numpy()
    flows = flows.transpose((0, 2, 3, 1))
    rgb = []

    for flow in flows:
        rgb.append(get_rgb_from_np_optical_flow(flow))
    # Normalize RGB to [0, 1] for matplotlib
    return np.asarray(rgb) / 255.0


def plot_centers(gaussians) -> None:
    """Plot the optimized 3D Gaussians as a point cloud"""
    means = gaussians.get_xyz.detach().cpu().numpy()
    rgb = gaussians.get_features[:, 0, :].detach().cpu().numpy()
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([pcd])


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

        s = video.scale_factor
        poses = torch.index_select(video.poses, 0, dirty_index)
        disps = torch.index_select(video.disps, 0, dirty_index)
        # convert poses to 4x4 matrix
        Ps = SE3(poses).inv().matrix().cpu().numpy()
        images = torch.index_select(video.images, 0, dirty_index)
        images = images.cpu()[:, ..., int(s // 2 - 1) :: s, int(s // 2 - 1) :: s].permute(0, 2, 3, 1)
        points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()

        thresh = droid_visualization.mv_filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))
        count = droid_backends.depth_filter(video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)
        count, disps = count.cpu(), disps.cpu()
        # Only keep points that are consistent across multiple views and not too close by
        masks = (count >= droid_visualization.mv_filter_count) & (disps > 0.5 * disps.mean(dim=[1, 2], keepdim=True))

        if droid_visualization.uncertainty_filter_on:
            weights = torch.index_select(video.confidence, 0, dirty_index)
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
    # print(colored("[Visu] Saving the visualization for later usage ...!", "grey"))
    # try:
    #     o3d.io.write_pinhole_camera_parameters(save_root + "/final_viewpoint.json", param)
    #     pcl_path = str(Path(save_root) / "pointclouds")
    #     cam_path = str(Path(save_root) / "cameras")
    #     write_pointclouds(droid_visualization.points, pcl_path, ext="xyzrgb")
    #     write_linesets(droid_visualization.cameras, cam_path)
    # except Exception as e:
    #     print(colored("[Visu] Something went wrong when saving the visualization ...!", "red"))
    #     print(colored(e, "red"))

    vis.destroy_window()
    return True
