from typing import Optional, Tuple
from omegaconf import DictConfig

import torch
from torch import nn

import numpy as np
from .utils.graphics_utils import getProjectionMatrix, getWorld2View2, focal2fov, fov2focal
from ..utils.image_utils import PILtoTorch


class Camera(nn.Module):
    def __init__(
        self,
        uid: int,
        color: torch.Tensor,
        depth_est: torch.Tensor,
        depth_gt: torch.Tensor,
        pose_w2c: torch.Tensor,
        projection_matrix: torch.Tensor,
        intrinsics: Tuple[float, float, float, float],
        fov: Tuple[float, float],
        img_size: Tuple[int, int],
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        device: str = "cuda:0",
        mask: Optional[torch.Tensor] = None,
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        self.fx, self.fy, self.cx, self.cy = intrinsics
        self.fov_x, self.fov_y = fov
        self.image_height, self.image_width = img_size

        self.R_gt = pose_w2c[:3, :3]
        self.T_gt = pose_w2c[:3, 3]
        self.update_RT(self.R_gt, self.T_gt)

        self.original_image = color.clamp(0.0, 1.0).to(self.device)
        self.depth = depth_est.to(self.device)
        self.depth_prior = depth_gt.to(self.device)

        self.grad_mask = None
        self.mask = mask  # NOTE chen: this is used for dynamic objects if we know that info
        if mask is not None:
            # FIXME chen: so this is not really used? Why do I need this mask?
            # self.original_image *= mask.to(self.data_device)
            self.mask = mask.to(self.device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.device)
            self.mask = None

        self.trans, self.scale = trans, scale

        self.exposure_a = nn.Parameter(torch.tensor([0.0], requires_grad=True, device=device))
        self.exposure_b = nn.Parameter(torch.tensor([0.0], requires_grad=True, device=device))

        self.projection_matrix = projection_matrix.to(device=device)

    def image_tensors_to(self, new_device: str) -> None:
        self.original_image = self.original_image.to(new_device)
        if self.depth is not None:
            self.depth = self.depth.to(new_device)
        if self.depth_prior is not None:
            self.depth_prior = self.depth_prior.to(new_device)
        if self.mask is not None:
            self.mask = self.mask.to(new_device)

    def to(self, device: str) -> None:
        self.device = device
        self.image_tensors_to(device)

        self.R = self.R.to(device=device)
        self.T = self.T.to(device=device)

        self.projection_matrix = self.projection_matrix.to(device=device)

        if self.grad_mask is not None:
            self.grad_mask = self.grad_mask.to(device=device)

    def detach(self):
        """Clone and detach all tensors from the camera object"""
        return Camera(
            self.uid,
            self.original_image.clone().detach(),
            self.depth.clone().detach() if self.depth is not None else None,
            self.depth_prior.clone().detach() if self.depth_prior is not None else None,
            self.pose.clone().detach(),
            self.projection_matrix.clone().detach(),
            (self.fx, self.fy, self.cx, self.cy),
            (self.fov_x, self.fov_y),
            (self.image_height, self.image_width),
            self.device,
            self.mask.clone().detach() if self.mask is not None else None,
        )

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_depth,
            gt_pose,
            projection_matrix,
            (dataset.fx, dataset.fy, dataset.cx, dataset.cy),
            (dataset.fovx, dataset.fovy),
            (dataset.height, dataset.width),
            device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, fov_x, fov_y, fx, fy, cx, cy, H, W):
        # projection_matrix = getProjectionMatrix2(
        #     znear=0.001, zfar=10000.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        # ).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=0.001, zfar=10000.0, fovX=fov_x, fovY=fov_y).transpose(0, 1)

        return Camera(
            uid,
            None,
            None,
            None,
            T,
            projection_matrix,
            (fx, fy, cx, cy),
            (fov_x, fov_y),
            (H, W),
        )

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def pose(self):
        tensor = torch.eye(4, device=self.device)
        tensor[:3, :3], tensor[:3, 3] = self.R, self.T
        return tensor

    @property
    def full_proj_transform(self):
        return (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def update_intrinsics(
        self, intrinsics: torch.Tensor, image_shape: Tuple[int, int], znear: float, zfar: float
    ) -> None:
        self.fx, self.fy, self.cx, self.cy = intrinsics
        height, width = image_shape

        self.fov_x, self.fov_y = focal2fov(self.fx, width), focal2fov(self.fy, height)
        # projection_matrix = getProjectionMatrix2(znear, zfar, self.cx, self.cy, self.fx, self.fy, width, height)
        projection_matrix = getProjectionMatrix(znear, zfar, self.fov_x, self.fov_y)
        self.projection_matrix = projection_matrix.transpose(0, 1).to(device=self.device)

    def clean(self):
        self.original_image = None
        self.depth = None
        self.depth_prior = None
        self.grad_mask = None
