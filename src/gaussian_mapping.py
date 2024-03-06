import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import open3d as o3d
import torch.nn.functional as F

from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.recon_helpers import setup_camera
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify
from utils.keyframe_selection import keyframe_selection_overlap

from diff_gaussian_rasterization import GaussianRasterizer as Renderer





def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[2], color.shape[1]

    FX, FY, CX, CY = intrinsics

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def initialize_optimizer(params, lrs_dict):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups)


def initialize_first_timestep(first_frame, num_frames, scene_radius_depth_ratio, 
                              mean_sq_dist_method, gaussian_distribution=None,
                              use_gt_stream=False):
    # Get RGB-D Data & Camera Parameters

    if use_gt_stream:
        _, color, depth, intrinsics, pose = first_frame

    else:

        #color, depth, c2w, gt_c2w, mask = self.video.get_mapping_item(frame, device, decay=self.decay)
        pass

    # Process RGB-D Data
    #color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    #depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    #intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())


    densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio


    return params, variables, intrinsics, w2c, cam


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)

    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    
    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def add_new_gaussians(params, variables, curr_data, sil_thres, 
                      time_idx, mean_sq_dist_method, gaussian_distribution):
    # Silhouette Rendering
    transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables


def initialize_camera_pose(params, curr_time_idx, forward_prop):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store

def process_frame(frame,device):
    idx, color, depth, intrinsics, pose = frame
    color = color.squeeze(0).to(device)
    depth = depth.unsqueeze(0).to(device)
    intrinsics = intrinsics.to(device)
    pose = pose.to(device)
    return idx, color, depth, intrinsics, pose


def render_view(params, iter_time_idx, cam):

    transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                                gaussians_grad=False,
                                                camera_grad=False)
    
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    im, _, _, = Renderer(raster_settings=cam)(**rendervar)
    im = im.squeeze(0).permute(1,2,0).cpu().detach().numpy()
    im = (im - im.min()) / (im.max() - im.min()) * 255
    im = im.astype(np.uint8)
    return im

def plot_3d(rgb: np.ndarray, depth: np.ndarray):
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
    x = x.flatten()
    y = y.flatten()
    depth = depth.flatten()
    # Convert to 3D points
    fx, fy, cx, cy = get_calib_heuristic(h, w)
    # Unproject
    x3 = (x - cx) * depth / fx
    y3 = (y - cy) * depth / fy
    z3 = depth
    # Convert to Open3D format
    xyz = np.stack([x3, y3, z3], axis=1)
    rgb = np.stack(
        [rgb[0, :, :].flatten(), rgb[1, :, :].flatten(), rgb[2, :, :].flatten()], axis=1
    )
    depth = depth[~invalid]
    xyz = xyz[~invalid]
    rgb = rgb[~invalid]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    # Plot the point cloud
    o3d.visualization.draw_geometries([pcd])

def plot_centers(params):
    means3D = params['means3D'].detach().cpu().numpy()
    rgb_colors = params['rgb_colors'].detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means3D)
    pcd.colors = o3d.utility.Vector3dVector(rgb_colors)
    o3d.visualization.draw_geometries([pcd])



class GaussianMapper(object):
    """
    Mapper thread.
    """

    def __init__(self, cfg, args, slam, mapping_queue=None):
        self.cfg = cfg
        self.args = args
        self.video = slam.video
        self.verbose = slam.verbose
        self.device = args.device
        #self.device = cfg["mapping"]["device"]
        self.decay = float(cfg["mapping"]["decay"])

        # TODO: get real number of frames
        #self.n_frames = slam.n_frames
        self.n_frames = 2000

        self.num_iters_mapping = self.cfg["mapping"]['num_iters']
        self.last_visit = -1
        self.gt_w2c_all_frames = []
        self.keyframe_list = []
        self.keyframe_time_indices = []


        if mapping_queue is not None:
            self.mapping_queue = mapping_queue
            self.use_gt_stream = True
        else:
            self.use_gt_stream = False



        # seperate dataloader for densification logic ?

        if "gaussian_distribution" not in self.cfg:
            self.cfg["mapping"]['gaussian_distribution'] = "isotropic"



    def __call__(self, the_end = False):

        if self.last_visit == -1:
            if self.use_gt_stream:
                self.first_frame = self.mapping_queue.get()
                self.first_frame = process_frame(self.first_frame, self.device)


            else:
                #TODO
                color, depth, c2w, gt_c2w, mask = self.video.get_mapping_item(time_idx, self.device, decay=self.decay)
                self.first_frame = (color, depth, intrinsics, gt_c2w)

            self.params, self.variables, intrinsics, self.first_frame_w2c, self.cam = initialize_first_timestep(self.first_frame, self.n_frames, 
                                                                                        self.cfg["mapping"]['scene_radius_depth_ratio'],
                                                                                        self.cfg["mapping"]['mean_sq_dist_method'],
                                                                                        gaussian_distribution=self.cfg["mapping"]['gaussian_distribution'],
                                                                                        use_gt_stream=self.use_gt_stream)
            print("Gaussians initialized successfully")
            self.last_visit = 0


        cur_idx = int(self.video.filtered_id.item()) 

        if self.last_visit < cur_idx :
        #if not self.mapping_queue.empty():
            time_idx = self.last_visit
            self.last_visit += 1
        
            if self.use_gt_stream:
                if time_idx == 0:
                    time_idx, color, depth, intrinsics, gt_c2w = self.first_frame

                else:
                    try:
                        frame = self.mapping_queue.get()
                        time_idx, color, depth, intrinsics, gt_c2w = process_frame(frame, self.device)
                    except Exception as e:
                        print(e)
                        print("Continuing...")
                        pass
                    
                    with torch.no_grad():
                        rel_w2c = torch.inverse(gt_c2w)
                        rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                        rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                        rel_w2c_tran = rel_w2c[:3, 3].detach()
                        # Update the camera parameters
                        self.params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                        self.params['cam_trans'][..., time_idx] = rel_w2c_tran

            else:
                color, depth, c2w, gt_c2w, mask = self.video.get_mapping_item(
                            time_idx, self.device, decay=self.decay
                        )


            gt_w2c = torch.inverse(gt_c2w)
            self.gt_w2c_all_frames.append(gt_w2c)

            curr_data = {'cam': self.cam, 'im': color, 'depth': depth, 'id': time_idx, 'intrinsics': intrinsics, 
                        'w2c': self.first_frame_w2c, 'iter_gt_w2c_list': self.gt_w2c_all_frames}

            # Densification & KeyFrame-based Mapping
            if time_idx == 0 or (time_idx+1) % self.cfg["mapping"]['map_every'] == 0:
                # Densification
                if self.cfg["mapping"]['add_new_gaussians'] and time_idx > 0:
                    # Setup Data for Densification

                    densify_curr_data = curr_data

                    # Add new Gaussians to the scene based on the Silhouette
                    self.params, self.variables = add_new_gaussians(self.params, self.variables, densify_curr_data, 
                                                        self.cfg["mapping"]['sil_thres'], time_idx,
                                                        self.cfg["mapping"]['mean_sq_dist_method'], self.cfg["mapping"]['gaussian_distribution'])
                    post_num_pts = self.params['means3D'].shape[0]

                with torch.no_grad():
                    # Get the current estimated rotation & translation
                    curr_cam_rot = F.normalize(self.params['cam_unnorm_rots'][..., time_idx].detach())
                    curr_cam_tran = self.params['cam_trans'][..., time_idx].detach()
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # Select Keyframes for Mapping
                    num_keyframes = self.cfg["mapping"]['mapping_window_size']-2
                    selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, self.keyframe_list[:-1], num_keyframes)
                    selected_time_idx = [self.keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                    if len(self.keyframe_list) > 0:
                        # Add last keyframe to the selected keyframes
                        selected_time_idx.append(self.keyframe_list[-1]['id'])
                        selected_keyframes.append(len(self.keyframe_list)-1)
                    # Add current frame to the selected keyframes
                    selected_time_idx.append(time_idx)
                    selected_keyframes.append(-1)
                    # Print the selected keyframes
                    print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

                # Reset Optimizer & Learning Rates for Full Map Optimization
                optimizer = initialize_optimizer(self.params, self.cfg["mapping"]['lrs']) 

                # Mapping

                for iter in range(self.num_iters_mapping):
                    iter_start_time = time.time()
                    # Randomly select a frame until current time step amongst keyframes
                    rand_idx = np.random.randint(0, len(selected_keyframes))
                    selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                    if selected_rand_keyframe_idx == -1:
                        # Use Current Frame Data
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                    else:
                        # Use Keyframe Data
                        iter_time_idx = self.keyframe_list[selected_rand_keyframe_idx]['id']
                        iter_color = self.keyframe_list[selected_rand_keyframe_idx]['color']
                        iter_depth = self.keyframe_list[selected_rand_keyframe_idx]['depth']

                    iter_gt_w2c = self.gt_w2c_all_frames[:iter_time_idx+1]
                    iter_data = {'cam': self.cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                                'intrinsics': intrinsics, 'w2c': self.first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                    # Loss for current frame
                    loss, self.variables, losses = get_loss(self.params, iter_data, self.variables, iter_time_idx, self.cfg["mapping"]['loss_weights'],
                                                    self.cfg["mapping"]['use_sil_for_loss'], self.cfg["mapping"]['sil_thres'],
                                                    self.cfg["mapping"]['use_l1'], self.cfg["mapping"]['ignore_outlier_depth_loss'], mapping=True)

                    # Backprop
                    loss.backward()
                    with torch.no_grad():
                        # Prune Gaussians
                        if self.cfg["mapping"]['prune_gaussians']:
                            self.params, self.variables = prune_gaussians(self.params, self.variables, optimizer, iter, self.cfg["mapping"]['pruning_dict'])
                        # Gaussian-Splatting's Gradient-based Densification
                        if self.cfg["mapping"]['use_gaussian_splatting_densification']:
                            self.params, self.variables = densify(self.params, self.variables, optimizer, iter, self.cfg["mapping"]['densify_dict'])

                        # Optimizer Update
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

            # Add frame to keyframe list
            if ((time_idx == 0) or ((time_idx+1) % self.cfg["mapping"]['keyframe_every'] == 0) or \
                    (time_idx == self.n_frames-2)) and (not torch.isinf(gt_w2c[-1]).any()) and (not torch.isnan(gt_w2c[-1]).any()):
                with torch.no_grad():
                    # Get the current estimated rotation & translation
                    curr_cam_rot = F.normalize(self.params['cam_unnorm_rots'][..., time_idx].detach())
                    curr_cam_tran = self.params['cam_trans'][..., time_idx].detach()
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # Initialize Keyframe Info
                    curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                    # Add to keyframe list
                    self.keyframe_list.append(curr_keyframe)
                    self.keyframe_time_indices.append(time_idx)
                    #print(f"\nAdded Keyframe at Frame {time_idx}. Total number of keyframes: {len(self.keyframe_list)}")



            if time_idx % 10 == 0:
                print("Number of Gaussians: ", self.params['means3D'].shape[0])
                im = render_view(self.params, time_idx, self.cam)
                cv2.imwrite(f"/home/leon/go-slam-tests/renders/{time_idx}.png", im)

            if time_idx % 100 == 0:
                #plot_3d(color, depth)
                plot_centers(self.params)

            
                

