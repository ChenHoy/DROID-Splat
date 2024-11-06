# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import ipdb
import time
import math
from termcolor import colored
from typing import Optional, List, Tuple

import numpy as np
import open3d as o3d

import torch
from torch import nn
import lietorch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2

from ..utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    helper,
    inverse_sigmoid,
    strip_symmetric,
    mkdir_p,
)
from ..camera_utils import Camera
from ..utils.graphics_utils import BasicPointCloud, getWorld2View2
from ..utils.reloc_utils import compute_relocation_cuda
from ..utils.sh_utils import RGB2SH


"""
3D Gaussian Splatting together with techniques from  Multi-View Gaussian Splatting functions, 
see https://github.com/xiaobiaodu/MVGS for references.

NOTE chen: We vectorized the intersection tests to speed this up and corrected some minor mistakes they had in their code
"""


def normal2rotation(n: torch.Tensor):
    """Construct a random rotation matrix from normal
    adopted from https://github.com/turandai/gaussian_surfels/blob/main/utils/general_utils.py

    NOTE it would better be positive definite and orthogonal
    """
    #
    #
    n = torch.nn.functional.normalize(n)
    w0 = torch.tensor([[1, 0, 0]]).expand(n.shape).to(n.device)
    R0 = w0 - torch.sum(w0 * n, -1, True) * n
    R0 *= torch.sign(R0[:, :1])
    R0 = torch.nn.functional.normalize(R0)
    R1 = torch.linalg.cross(n, R0)

    R1 *= torch.sign(R1[:, 1:2]) * torch.sign(n[:, 2:])
    R = torch.stack([R0, R1, n], -1)
    q = rotmat2quaternion(R)

    return q


def rotmat2quaternion(R: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] + 1e-6
    r = torch.sqrt(1 + tr) / 2
    # print(torch.sum(torch.isnan(r)))
    q = torch.stack(
        [
            r,
            (R[:, 2, 1] - R[:, 1, 2]) / (4 * r),
            (R[:, 0, 2] - R[:, 2, 0]) / (4 * r),
            (R[:, 1, 0] - R[:, 0, 1]) / (4 * r),
        ],
        -1,
    )
    if normalize:
        q = torch.nn.functional.normalize(q, dim=-1)
    return q


class GradientScaler(object):
    """
    Tracks the number of times each variable has been optimized and scales gradients with diminishing effect.
    We use an exponential decay schedule here.
    """

    def __init__(self, min_scale: float = 0.1, decay_rate: float = 0.01, counts: torch.Tensor = None):
        self.min_scale = min_scale  # Dont lower gradients below 1/10, so we never stop optimizing some old gaussians
        self.decay_rate = decay_rate
        self.counts = counts if counts is not None else torch.ones(1)

    def get_scale(self, counts: torch.Tensor) -> torch.Tensor:
        """
        Scales the gradients of a variable based on its optimization count with diminishing effect.

        Args:
            variable: The torch.Tensor variable for which to scale gradients.

        Returns:
            The scaling factor as a Tensor of floats.
        """
        decay = torch.exp(-self.decay_rate * counts)
        scale = self.min_scale + (1 - self.min_scale) * decay
        return scale

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        # Apply scaling factor to gradients during backward pass
        assert len(grad) == len(
            self.counts
        ), f"Gradient and count shape mismatch! Expected shape {grad.shape}, got {self.counts.shape}"
        scaling = self.get_scale(self.counts)

        if grad.ndim == 1:
            return grad * scaling
        elif grad.ndim == 2:
            return grad * scaling[:, None]
        elif grad.ndim == 3:
            return grad * scaling[:, None, None]
        else:
            raise Exception(f"Gradient shape {grad.shape} not supported! Use either 1D, 2D or 3D tensors.")


class InverseScaler(GradientScaler):
    def __init__(self, min_scale: float = 0.1, decay_rate: float = 0.01, counts: torch.Tensor = None):
        super().__init__(min_scale, decay_rate, counts)

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        # Apply scaling factor to gradients during backward pass
        assert len(grad) == len(
            self.counts
        ), f"Gradient and count shape mismatch! Expected shape {grad.shape}, got {self.counts.shape}"
        scaling = self.get_scale(self.counts)

        if grad.ndim == 1:
            return grad / scaling
        elif grad.ndim == 2:
            return grad / scaling[:, None]
        elif grad.ndim == 3:
            return grad / scaling[:, None, None]
        else:
            raise Exception(f"Gradient shape {grad.shape} not supported! Use either 1D, 2D or 3D tensors.")


def scale_gradients(variable: torch.nn.Parameter, scale_fn):
    """Attach a hook, so the gradients are automatically scaled by the GradientScaler for each backward pass."""

    def hook(parameter):
        parameter.grad = scale_fn(parameter.grad)

    h = variable.register_post_accumulate_grad_hook(hook)
    return h


def unscale_gradients(variable: torch.nn.Parameter, inverse_fn) -> torch.Tensor:
    """Because we use the gradients when densifying and pruning Gaussians, we might want to use the
    original gradients, else we easily get stuck with a fixed number of Gaussians and dont densify anymore.

    The densification call lies between loss.backward() and optimizer.step()! Therefore we need to unscale the gradients
    after computation in backward(), but leave them in place for the following optimizer.step() scall.
    For this reason we dont change the gradient in place, but return the unscaled value for the densification function.
    """
    return inverse_fn(variable.grad)


class GaussianModel:
    def __init__(self, sh_degree: int, config=None, device: str = "cuda:0"):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.device = device

        self._xyz = torch.empty(0, device=self.device)
        self._features_dc = torch.empty(0, device=self.device)
        self._features_rest = torch.empty(0, device=self.device)
        self._scaling = torch.empty(0, device=self.device)
        self._rotation = torch.empty(0, device=self.device)
        self._opacity = torch.empty(0, device=self.device)
        self.max_radii2D = torch.empty(0, device=self.device)
        self.xyz_gradient_accum = torch.empty(0, device=self.device)

        self.unique_kfIDs = torch.empty(0, device=self.device).int()
        self.n_obs = torch.empty(0, device=self.device).int()
        self.n_optimized = torch.empty(0, device=self.device).int()

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        # Cap the scale so Gaussian dont grow to big and create degenerate cases during rendering
        self.max_scale = 10000.0

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.cfg = config
        self.ply_input = None

        self.isotropic = False

    def build_covariance_from_scaling_rotation(self, scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    def __len__(self):
        """Returns the number of 3D Gaussians we have"""
        return len(self._xyz)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_3d_bounding_box(self, return_volume: bool = False):
        """Return a bounding box of the 3D Gaussians in the scene.
        This is in format [min_x, min_y, min_z, max_x, max_y, max_z]
        """
        min_point = torch.min(self.get_xyz, dim=0)[0]
        max_point = torch.max(self.get_xyz, dim=0)[0]
        box3d = [min_point[0], min_point[1], min_point[2], max_point[0], max_point[1], max_point[2]]
        volume3d = (max_point[0] - min_point[0]) * (max_point[1] - min_point[1]) * (max_point[2] - min_point[2])
        if return_volume:
            return box3d, volume3d
        else:
            return box3d

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        l = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.lr_delay_mult = training_args.position_lr_delay_mult
        self.max_steps = training_args.position_lr_max_steps

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps,
                )

                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {
            "xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling": self._scaling,
            "rotation": self._rotation,
        }

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)

            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                # self.info(group["name"], stored_state["exp_avg"].shape, extension_tensor.shape)
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def create_pcd_from_image_and_depth(
        self, cam, rgb, depth, init=False, downsample_factor=None, with_normals: bool = False
    ):
        if downsample_factor is None:
            if init:
                downsample_factor = self.cfg.pcd_downsample_init
            else:
                downsample_factor = self.cfg.pcd_downsample

        point_size = self.cfg.get("point_size", 0.05)
        if self.cfg.get("adaptive_pointsize", True):
            point_size = min(0.05, point_size * np.median(depth))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth, depth_scale=1.0, depth_trunc=100.0, convert_rgb_to_intensity=False
        )

        # NOTE chen: this is literally just a scaling operation, where we do: 1. invert 2. scale 3. invert
        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(cam.image_width, cam.image_height, cam.fx, cam.fy, cam.cx, cam.cy),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )
        if with_normals:
            # FIXME should we tune these parameters since they depend on the scene scale?
            pcd_tmp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)
        # NOTE some monocular depth prediction networks like OmniDepth actually can output normals on top
        # We could also use this information if we wanted
        if with_normals:
            new_normals = np.asarray(pcd_tmp.normals)
            pcd = BasicPointCloud(points=new_xyz, colors=new_rgb, normals=new_normals)
        else:
            pcd = BasicPointCloud(points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3)))

        pcd = BasicPointCloud(points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3)))

        if pcd.points.shape[0] <= 5:
            return

        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        fused_normals = torch.from_numpy(np.asarray(pcd.normals)).float().cuda()
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = (
            torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001) * point_size
        )

        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        if with_normals:
            rots = normal2rotation(fused_normals)
        else:
            # NOTE chen: Random normals seem to work better than unit ones
            # rots = torch.rand((fused_point_cloud.shape[0], 4), device=self.device)
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
            rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device)
        )

        return fused_point_cloud, features, scales, rots, opacities

    def create_pcd_from_image(
        self,
        cam: Camera,
        init=False,
        scale: float = 2.0,
        depthmap: np.ndarray = None,
        mask: torch.Tensor = None,
        downsample_factor: float = None,
    ):
        image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        ### Get depth map
        if depthmap is not None and depthmap.sum() > 0:
            depth_raw = depthmap
        else:
            # Take the attached depth
            if cam.depth is not None and cam.depth.sum() > 0:
                depth_raw = cam.depth.contiguous().cpu().numpy()
            # Take the prior if given
            elif cam.depth_prior is not None and cam.depth_prior.sum() > 0:
                depth_raw = cam.depth_prior.contiguous().cpu().numpy()
            # If we don't have a depth signal, initialize from random
            else:
                print(colored("Initializing Gaussians from RANDOM depth ...!", "red"))
                if cam.uid == 0:
                    # Introduce random Gaussians, this is how MonoGS works in monocular mode
                    depth_raw = (
                        np.ones(rgb_raw.shape[:2]) + (np.random.randn(rgb_raw.shape[:2]) - 0.5) * 0.05
                    ) * scale
                else:
                    # Take the depth of a small neighborhood of Gaussian keyframes
                    neighbors = torch.arange(max(0, cam.uid - 1), cam.uid + 1, device=self.unique_kfIDs.device)
                    # Take 1.5*median of neighboring frames
                    neighbors_scale = self.get_avg_scale(kfIdx=neighbors, factor=1.5)
                    if neighbors_scale is not None:
                        depth_raw = np.ones(rgb_raw.shape[:2]) * neighbors_scale
                    else:
                        depth_raw = (
                            np.ones(rgb_raw.shape[:2])
                            + (np.random.randn(rgb_raw.shape[0], rgb_raw.shape[1]) - 0.5) * 0.05
                        ) * scale

            # Introduce random Gaussians, this is how MonoGS works in monocular mode
            if self.cfg.sensor_type == "monocular":
                depth_raw = (
                    np.ones_like(depth_raw) + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5) * 0.05
                ) * scale

        if mask is not None:
            depth_raw[~mask.cpu().numpy()] = 0.0
        depth = o3d.geometry.Image(depth_raw.astype(np.float32))
        rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init, downsample_factor=downsample_factor)

    def extend_from_pcd(self, fused_point_cloud, features, scales, rots, opacities, kf_id):
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        new_features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0]), device=self.device).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0]), device=self.device).int()
        new_n_opt = torch.zeros((new_xyz.shape[0]), device=self.device).int()

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
            new_n_opt=new_n_opt,
        )

    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None, mask=None, downsample_factor=None
    ):
        features = self.create_pcd_from_image(
            cam_info, init, scale=scale, depthmap=depthmap, mask=mask, downsample_factor=downsample_factor
        )
        if features is not None:
            fused_point_cloud, features, scales, rots, opacities = features
            self.extend_from_pcd(fused_point_cloud, features, scales, rots, opacities, kf_id)
        else:
            print("No points in the point cloud")

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]

        self.max_radii2D = self.max_radii2D[valid_points_mask]
        # FIXME highly suspect of creating memory bug
        # raises sometimes RuntimeError: out_ptr == out_accessor[thread_count_nonzero[tid + 1]].data()
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask]
        self.n_obs = self.n_obs[valid_points_mask]
        self.n_optimized = self.n_optimized[valid_points_mask]

    # def prune_floaters(
    #     self, search_radius: float = 0.1, min_nn_distance: float = 0.05, return_mask: bool = True
    # ) -> None:
    #     """Prune isolated outlier points which are floaters without any neighbors"""
    #     pcd_temp = self._xyz.unsqueeze(0)
    #     # NOTE since we search within the same point cloud, we will always get the original point as the closest neighbor with distance 0.0
    #     all_dists, idxs, _, _ = frnn.frnn_grid_points(pcd_temp, pcd_temp, K=2, r=search_radius)
    #     nn_dists = all_dists[..., 1]  # Get the actual nearest neighbor distance
    #     floaters = nn_dists > min_nn_distance  # Points without a nearest neighbor in this radius are likely floaters
    #     self.prune_points(floaters.squeeze())
    #     if return_mask:
    #         return floaters

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
        new_n_opt=None,
        reset_params: bool = True,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Reset attributes for thresholding
        if reset_params:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

        # Add new keyframe ID and observation/optimized count
        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()
        if new_n_opt is not None:
            self.n_optimized = torch.cat((self.n_optimized, new_n_opt)).int()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, scale_std: float = 1.0):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )

        stds = scale_std * self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask].repeat(N)
        new_n_opt = self.n_optimized[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            new_n_opt=new_n_opt,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool),
            )
        )

        self.prune_points(prune_filter)

    def densify_from_mask(self, cam, mask, downsample_factor=1, depthmap=None):
        features = self.create_pcd_from_image(cam, mask=mask, downsample_factor=downsample_factor, depthmap=depthmap)
        if features is not None:
            fused_point_cloud, features, scales, rots, opacities = features
            self.extend_from_pcd(fused_point_cloud, features, scales, rots, opacities, cam.uid)

    def add_densification_stats(
        self, viewspace_point_tensor: torch.Tensor, update_filter: torch.Tensor, pixels: Optional[torch.Tensor] = None
    ):
        """From Pixel GS / Abs GS: Accumulate the gradient by averaging over all pixels that touched the Gaussian.

        see Abs Gaussian Splatting: https://arxiv.org/pdf/2404.10484
        """
        if pixels is not None:
            self.xyz_gradient_accum[update_filter] += torch.norm(
                viewspace_point_tensor.grad[: len(update_filter)][update_filter], dim=-1, keepdim=True
            ) * pixels[update_filter].unsqueeze(-1)
            self.denom[update_filter] += pixels[update_filter].unsqueeze(-1)
        else:
            self.xyz_gradient_accum[update_filter] += torch.norm(
                viewspace_point_tensor.grad[: len(update_filter)][update_filter], dim=-1, keepdim=True
            )
            self.denom[update_filter] += 1

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=self.device).requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device=self.device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device=self.device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device=self.device).requires_grad_(True)
        )
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=self.device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=self.device).requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device=self.device)
        self.unique_kfIDs = torch.zeros((self._xyz.shape[0]))
        self.n_obs = torch.zeros((self._xyz.shape[0]), device=self.device).int()
        self.n_optimized = torch.zeros((self._xyz.shape[0]), device=self.device).int()

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(self, visibility_filters):  ##Reset opacity for only non-visible gaussians
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def clean_scales(self) -> None:
        """Sometimes the scales get out of hand due to degenerate supervision"""
        invalid = torch.isnan(self._scaling) & torch.isinf(self._scaling)
        if not invalid.any():
            self._scaling.clamp_(max=self.max_scale)
            return

        # Reset the scales of out of bound Gaussians to some dummy
        dummy_scaling = self._scaling[~invalid].mean()
        scaling_new = self._scaling.clone()
        scaling_new[invalid] = dummy_scaling
        scaling_new.clamp_(max=self.max_scale)

        optimizable_tensors = self.replace_tensor_to_optimizer(scaling_new, "scaling")
        self._scaling = optimizable_tensors["scaling"]

    def check_nans(self):
        """Remove Gaussians with an invalid state, i.e. nan or inf attributes."""
        invalid = torch.isnan(self._xyz) | torch.isinf(self._xyz)
        invalid = invalid | torch.isnan(self._scaling) | torch.isinf(self._scaling)
        if invalid.ndim > 1:
            for i in range(invalid.ndim - 1):
                invalid = invalid.any(dim=-1)

        if invalid.sum() > 0:
            idx = torch.where(invalid)[0]
            print(colored(f"[Gaussian Mapper] Found degenerate Gaussians: {idx}", "red"))
            print(colored("[Gaussian Mapper] Cleaning up by removal ...", "red"))
            self.prune_points(invalid)

    def reanchor(self, kf_idx: torch.Tensor | List[int], delta_pose: torch.Tensor) -> None:
        """Transform the centers of Gaussians attached to a keyframe idx with an SE3 transform."""

        to_update = self.unique_kfIDs == kf_idx
        xyz_to_re = self.get_xyz[to_update]
        # Make homogenous coordinates
        xyz_re = torch.cat((xyz_to_re, torch.ones(xyz_to_re.shape[0], 1, device=self.device)), dim=1)
        xyz_new = self.get_xyz.clone()  # Make copy to replace old Variable

        dP = lietorch.SE3.InitFromVec(delta_pose.to(self.device))
        xyz_new[to_update] = (dP[None] * xyz_re)[:, :3]  # Transform and extract new 3D coordinates

        optimizable_tensors = self.replace_tensor_to_optimizer(xyz_new, "xyz")
        self._xyz = optimizable_tensors["xyz"]

    @torch.no_grad()
    def increment_n_opt_counter(
        self, visibility: Optional[torch.Tensor] = None, kf_idx: Optional[torch.Tensor] = None
    ) -> None:
        to_increment = torch.zeros_like(self.unique_kfIDs, dtype=torch.bool)
        if kf_idx is not None:
            to_increment = self.unique_kfIDs == kf_idx
        if visibility is not None:
            to_increment = torch.logical_and(to_increment, visibility.to(to_increment.device))

        self.n_optimized[to_increment] += 1

    def get_avg_scale(self, factor: float = 1.0, kfIdx: Optional[torch.Tensor | List[int]] = None) -> float:
        """Get the average depth of the Gaussians in the scene. Optionally, filter by keyframe index to select only Gaussians in a specific area."""
        points = self.get_xyz

        if kfIdx is not None:
            print(f"Computing scale based on {kfIdx}")
            select = torch.zeros(points.shape[0], dtype=torch.bool)
            for idx in kfIdx:
                # Sanity check if this index actually exists
                if not (self.unique_kfIDs == idx).any():
                    continue
                idx_array = idx * torch.ones_like(self.unique_kfIDs, device=self.unique_kfIDs.device)
                select = torch.logical_or(select, (self.unique_kfIDs == idx_array))
            print(f"Selecting {select.sum()} points from {len(points)}")
            points = points[select]

        if len(points) == 0:
            return None

        depth = points[:, 2]
        avg_scale = factor * torch.median(depth, dim=0)[0]
        print(f"Using average scale: {avg_scale}")
        return avg_scale.item()

    def set_scale_grads(self, min_scale: float = 0.1, decay_rate: float = 0.01) -> None:
        """Attach a hook with the scaling factor to scale the gradients during the backward pass dependent on self.n_optimized."""
        scale_fn = GradientScaler(min_scale=min_scale, decay_rate=decay_rate, counts=self.n_optimized.to(self.device))
        h_xyz = scale_gradients(self._xyz, scale_fn)
        h_features_dc = scale_gradients(self._features_dc, scale_fn)
        h_features_rest = scale_gradients(self._features_rest, scale_fn)
        h_opacity = scale_gradients(self._opacity, scale_fn)
        h_scaling = scale_gradients(self._scaling, scale_fn)
        h_rot = scale_gradients(self._rotation, scale_fn)
        return [h_xyz, h_features_dc, h_features_rest, h_opacity, h_scaling, h_rot]

    def densify_and_clone(self, grads: torch.Tensor, grad_threshold: float, scene_extent: float):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # NOTE these operations create a malloc error with wrong pytorch version
        new_kf_id = self.unique_kfIDs[selected_pts_mask]
        new_n_obs = self.n_obs[selected_pts_mask]
        new_n_opt = self.n_optimized[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            new_n_opt=new_n_opt,
        )

    def densify_and_prune(
        self, max_grad: float, min_opacity: float, extent: float, max_screen_size: float, scale_std: float = 1.0
    ):
        """Densify and prune"""
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        n_g = self.get_xyz.shape[0]  # Memoize number of Gaussians before operations

        # Densify in carved out 3D regions of high 2D projected image losses
        self.densify_and_clone(grads, max_grad, extent)
        # Vanilla Densify and Split based on gradients and extent
        self.densify_and_split(grads, max_grad, extent, scale_std=scale_std)

        # Prune based on Opacity
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        self.info(f"Pruning & densification added {self.get_xyz.shape[0] - n_g} gaussians")

        torch.cuda.empty_cache()

    #####
    # Monte-Carlo Markov Chain model methods
    #####

    def _update_params(self, idxs, ratio):
        new_opacity, new_scaling = compute_relocation_cuda(
            opacity_old=self.get_opacity[idxs, 0], scale_old=self.get_scaling[idxs], N=ratio[idxs, 0] + 1
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))

        return (
            self._xyz[idxs],
            self._features_dc[idxs],
            self._features_rest[idxs],
            new_opacity,
            new_scaling,
            self._rotation[idxs],
        )

    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio

    def relocate_gs(self, dead_mask=None):

        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = self.get_opacity[alive_indices, 0]
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        (
            self._xyz[dead_indices],
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices],
        ) = self._update_params(reinit_idx, ratio=ratio)

        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx)

    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        probs = self.get_opacity.squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation) = self._update_params(
            add_idx, ratio=ratio
        )
        # Update opacity and scaling of the existing Gaussians
        self._opacity[add_idx] = new_opacity
        self._scaling[add_idx] = new_scaling

        new_unique_kfIDs = self.unique_kfIDs[add_idx]
        new_n_obs = self.n_obs[add_idx]  # FIXME should this be reinitialized to zero, as the Gaussian is newly born?
        new_n_opt = self.n_optimized[add_idx]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
            new_n_opt=new_n_opt,
            reset_params=False,
        )
        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs

    def info(self, msg: str) -> None:
        print(colored(f"[Gaussian Mapping] {msg}", "magenta"))
