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
import math
from termcolor import colored
from typing import Optional, List

import numpy as np
import open3d as o3d

import torch
from torch import nn
import lietorch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2

from ..utils.general_utils import build_rotation, build_scaling_rotation, inverse_sigmoid, mkdir_p, get_expon_lr_func
from ..camera_utils import Camera
from ..utils.graphics_utils import BasicPointCloud, getWorld2View2
from ..utils.sh_utils import RGB2SH


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

        # Cap the scale so Gaussian dont grow to big and create degenerate cases during rendering
        self.max_scale = 10000.0

        self.denom = torch.empty(0, device=self.device)
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.optimizer = None
        self.setup_functions()

        self.cfg = config

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(
                torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation
            ).permute(0, 2, 1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device=self.device)
            trans[:, :3, :3] = RS
            trans[:, 3, :3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __len__(self):
        """Returns the number of 3D Gaussians we have"""
        return len(self._xyz)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)  # .clamp(max=1)

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

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

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

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # TODO we lose the unique_kfIDs here
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

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

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
        self.n_obs = torch.zeros((self._xyz.shape[0]), device="cpu").int()
        self.n_optimized = torch.zeros((self._xyz.shape[0]), device="cpu").int()

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

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
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

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N: int = 2, scale_std: float = 1.0):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        stds = scale_std * self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
        )

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool))
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_kf_id = self.unique_kfIDs[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, scale_std=1.0):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        n_g = self.get_xyz.shape[0]

        self.densify_and_clone(grads, max_grad, extent)
        # New points are seeded within normal deviation with mean and std=scale, i.e. around the current point
        # We allow to manually change the standard deviation to have many more points lie with high chance very close to old Gaussian!
        self.densify_and_split(grads, max_grad, extent, scale_std=scale_std)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        self.info(f"Pruning & densification added {self.get_xyz.shape[0] - n_g} gaussians")

        # TODO maybe delete for faster runtime
        torch.cuda.empty_cache()

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

    # TODO chen: we use exposure parameters to linear transform the image on top in our other pipelines!
    def create_pcd_from_image(
        self,
        cam: Camera,
        init=False,
        scale: float = 2.0,
        depthmap: np.ndarray = None,
        mask: torch.Tensor = None,
        downsample_factor: float = None,
    ):
        # image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        image_ab = cam.original_image
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

    # TODO chen: this is incredibly wasteful as we go back and forth between CPU and GPU here
    # and all that just to convert 2D -> 3D with a pinhole model, which we could do based on lietorch alone
    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False, downsample_factor=None):
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

        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)

        pcd = BasicPointCloud(points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3)))

        if pcd.points.shape[0] <= 5:
            return

        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = (
            torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001) * point_size
        )

        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device)
        )

        return fused_point_cloud, features, scales, rots, opacities

    def extend_from_pcd(self, fused_point_cloud, features, scales, rots, opacities, kf_id):
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True), device=self.device)
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True), device=self.device
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True), device=self.device
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True), device=self.device)
        new_rotation = nn.Parameter(rots.requires_grad_(True), device=self.device)
        new_opacity = nn.Parameter(opacities.requires_grad_(True), device=self.device)

        new_unique_kfIDs = torch.ones((new_xyz.shape[0]), device=self.device).int() * kf_id

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
        )

    def densify_from_mask(self, cam, mask, downsample_factor=1, depthmap=None):
        features = self.create_pcd_from_image(cam, mask=mask, downsample_factor=downsample_factor, depthmap=depthmap)
        if features is not None:
            fused_point_cloud, features, scales, rots, opacities = features
            self.extend_from_pcd(fused_point_cloud, features, scales, rots, opacities, cam.uid)

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

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

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

    def info(self, msg: str) -> None:
        print(colored(f"[Gaussian Mapping] {msg}", "magenta"))
