#
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
from termcolor import colored

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from ..utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from ..utils.graphics_utils import BasicPointCloud, getWorld2View2
from ..utils.sh_utils import RGB2SH
from ...utils.system_utils import mkdir_p
from .gaussian_model import GaussianModel


class DynamicGaussianModel(GaussianModel):
    def __init__(self, sh_degree: int, lifespan: int = 100, config=None, device: str = "cuda:0"):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.device = device
        self.lifespan = lifespan

        # Time dependent properties
        self._xyz = torch.empty(0, device=self.device)
        self._rotation = torch.empty(0, device=self.device)
        self.xyz_gradient_accum = torch.empty(0, device=self.device)

        # Constant properties
        self._features_dc = torch.empty(0, device=self.device)
        self._features_rest = torch.empty(0, device=self.device)
        self._scaling = torch.empty(0, device=self.device)
        self._opacity = torch.empty(0, device=self.device)
        self.max_radii2D = torch.empty(0, device=self.device)

        self.unique_kfIDs = torch.empty(0).int()
        self.n_obs = torch.empty(0).int()

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.cfg = config
        self.ply_input = None

        self.isotropic = False

        self.type = "dynamic"

    def build_covariance_from_scaling_rotation(self, scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    def __len__(self):
        if self.get_xyz.shape[0] == 0:
            return 0
        else:
            return self.get_xyz.shape[-2]

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

    def get_covariance(self, t, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation[t])

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def extend_from_pcd(self, fused_point_cloud, features, scales, rots, opacities, kf_id):

        if fused_point_cloud.shape[0] == 0:  # No new points
            self.info(f"No gaussians added for kf_id {kf_id}")
            return

        new_gaussians = fused_point_cloud.shape[0]

        new_point_cloud = torch.zeros((self.lifespan, new_gaussians, 3))
        new_point_cloud[kf_id] = fused_point_cloud
        new_xyz = nn.Parameter(new_point_cloud.requires_grad_(True)).to(self.device)

        new_rot = torch.zeros((self.lifespan, new_gaussians, rots.shape[1]))
        new_rot[kf_id] = rots
        new_rotation = nn.Parameter(new_rot.requires_grad_(True)).to(self.device)

        new_features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        new_features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_gaussians)).int() * kf_id
        new_n_obs = torch.zeros((new_gaussians)).int()

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
        )

    def extend_from_pcd_seq(self, cam_info, kf_id=None, init=False, scale=2.0, depthmap=None):
        if kf_id is None:
            kf_id = cam_info.kf_id
        fused_point_cloud, features, scales, rots, opacities = self.create_pcd_from_image(
            cam_info, init, scale=scale, depthmap=depthmap
        )
        self.extend_from_pcd(fused_point_cloud, features, scales, rots, opacities, kf_id)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.lr_delay_mult = training_args.position_lr_delay_mult
        self.max_steps = training_args.position_lr_max_steps

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)

            if (
                group["name"] == "xyz" or group["name"] == "rotation"
            ):  # NOTE leon: xyz and rot have different dimensions. #TODO: cleaner solution
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][:, mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][:, mask]

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][:, mask].requires_grad_(True)))
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][:, mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
            else:
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

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[:, valid_points_mask]
        self.denom = self.denom[:, valid_points_mask]

        self.max_radii2D = self.max_radii2D[valid_points_mask]
        # FIXME highly suspsect of creating memory bug
        # raises sometimes RuntimeError: out_ptr == out_accessor[thread_count_nonzero[tid + 1]].data()
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            # print(extension_tensor.shape, group["params"][0].shape)
            # print(group["name"])
            if group["name"] == "xyz" or group["name"] == "rotation":

                if stored_state is not None:
                    # self.info(group["name"], stored_state["exp_avg"].shape, extension_tensor.shape)
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=1
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                        dim=1,
                    )

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=1).requires_grad_(True)
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=1).requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]
            else:
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
        self._rotation = optimizable_tensors["rotation"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], self.get_xyz.shape[1], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], self.get_xyz.shape[1], 1), device=self.device)

        self.max_radii2D = torch.zeros((self.get_xyz.shape[1]), device=self.device)

        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.__len__()
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)

        new_xyz = torch.zeros((self.lifespan, N * selected_pts_mask.sum(), 3), device=self.device)

        for t in range(self.lifespan):
            rot = build_rotation(self._rotation[t, selected_pts_mask]).repeat(N, 1, 1)
            new_xyz[t] = torch.bmm(rot, samples.unsqueeze(-1)).squeeze(-1)

        new_xyz += self.get_xyz[:, selected_pts_mask].repeat(1, N, 1)
        new_rotation = self._rotation[:, selected_pts_mask].repeat(1, N, 1)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        # NOTE these operations create a malloc error with wrong pytorch version
        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool),
            )
        )

        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[:, selected_pts_mask]
        new_rotation = self._rotation[:, selected_pts_mask]

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]

        # NOTE these operations create a malloc error with wrong pytorch version
        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

    def densify_and_prune(self, kf_id, max_grad, min_opacity, extent, max_screen_size):
        grads = (
            self.xyz_gradient_accum[kf_id] / self.denom[kf_id]
        )  # NOTE leon: we densify on a given view, based on the gradients of that frame
        grads[grads.isnan()] = 0.0
        n_g = self.__len__()

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size  # Size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)
        self.info(f"Pruning & densification added {self.__len__() - n_g} gaussians")

    def add_densification_stats(self, kf_id, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[kf_id, update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[kf_id, update_filter] += 1

    def info(self, msg: str) -> None:
        print(colored(f"[Gaussian Mapping] {msg}", "magenta"))
