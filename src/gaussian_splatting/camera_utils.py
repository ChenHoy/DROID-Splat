from typing import Optional, Tuple
from omegaconf import DictConfig

import torch
from torch import nn

from .utils.graphics_utils import getProjectionMatrix2, getWorld2View2, focal2fov
from .slam_utils import image_gradient, image_gradient_mask


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
        device: str = "cuda:0",
        mask: Optional[torch.Tensor] = None,
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        self.fx, self.fy, self.cx, self.cy = intrinsics
        self.FoVx, self.FoVy = fov
        self.image_height, self.image_width = img_size

        self.R_gt = pose_w2c[:3, :3]
        self.T_gt = pose_w2c[:3, 3]
        self.update_RT(self.R_gt, self.T_gt)

        self.original_image = color
        self.depth = depth_est
        self.depth_prior = depth_gt
        self.grad_mask = None

        self.mask = mask

        self.cam_rot_delta = nn.Parameter(torch.zeros(3, requires_grad=True, device=device))
        self.cam_trans_delta = nn.Parameter(torch.zeros(3, requires_grad=True, device=device))

        self.exposure_a = nn.Parameter(torch.tensor([0.0], requires_grad=True, device=device))
        self.exposure_b = nn.Parameter(torch.tensor([0.0], requires_grad=True, device=device))

        self.projection_matrix = projection_matrix.to(device=device)

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
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.001, zfar=10000.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid,
            None,
            None,
            None,
            T,
            projection_matrix,
            (fx, fy, cx, cy),
            (FoVx, FoVy),
            (H, W),
        )

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

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

        self.FoVx, self.FoVy = focal2fov(self.fx, width), focal2fov(self.fy, height)
        projection_matrix = getProjectionMatrix2(znear, zfar, self.cx, self.cy, self.fx, self.fy, width, height)
        self.projection_matrix = projection_matrix.transpose(0, 1).to(device=self.device)

    def compute_grad_mask(self, config):
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        if config["Dataset"]["type"] == "replica":
            row, col = 32, 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            for r in range(row):
                for c in range(col):
                    block = img_grad_intensity[
                        :,
                        r * int(h / row) : (r + 1) * int(h / row),
                        c * int(w / col) : (c + 1) * int(w / col),
                    ]
                    th_median = block.median()
                    block[block > (th_median * multiplier)] = 1
                    block[block <= (th_median * multiplier)] = 0
            self.grad_mask = img_grad_intensity
        else:
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = img_grad_intensity > median_img_grad_intensity * edge_threshold

    def clean(self):
        self.original_image = None
        self.depth = None
        self.depth_prior = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None
