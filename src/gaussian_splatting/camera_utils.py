from typing import Optional, Tuple
from omegaconf import DictConfig

import torch
from torch import nn

from kornia import create_meshgrid
from .utils.graphics_utils import getProjectionMatrix2, getWorld2View2, focal2fov


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

        self.mask = mask

        # Always fix first frame!
        if self.uid == 0:
            self.cam_rot_delta = nn.Parameter(torch.zeros(3, requires_grad=False, device=device))
            self.cam_trans_delta = nn.Parameter(torch.zeros(3, requires_grad=False, device=device))
        else:
            self.cam_rot_delta = nn.Parameter(torch.zeros(3, requires_grad=True, device=device))
            self.cam_trans_delta = nn.Parameter(torch.zeros(3, requires_grad=True, device=device))

        self.exposure_a = nn.Parameter(torch.tensor([0.0], requires_grad=True, device=device))
        self.exposure_b = nn.Parameter(torch.tensor([0.0], requires_grad=True, device=device))

        self.projection_matrix = projection_matrix.to(device=device)
        self.rayo, self.rayd = None, None
        self.set_ray()  # Get the camera origin and direction vector for later

    def set_ray(self):
        projectinverse = self.projection_matrix.T.inverse()
        camera2wold = self.world_view_transform.T.inverse()
        pixgrid = create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device="cpu")[0]
        pixgrid = pixgrid.cuda()  # H,W,
        xindx = pixgrid[:, :, 0]  # x
        yindx = pixgrid[:, :, 1]  # y
        ndcy, ndcx = pix2ndc(yindx, self.image_height), pix2ndc(xindx, self.image_width)
        ndcx = ndcx.unsqueeze(-1)
        ndcy = ndcy.unsqueeze(-1)  # * (-1.0)
        ndccamera = torch.cat((ndcx, ndcy, torch.ones_like(ndcy) * (1.0), torch.ones_like(ndcy)), 2)  # N,4
        projected = ndccamera @ projectinverse.T
        diretioninlocal = projected / projected[:, :, 3:]  # v
        rays_d = diretioninlocal[:, :, :3] @ camera2wold[:3, :3].T
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # Make unit

        # TODO why put this on the cpu?
        self.rayo = self.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0).cpu()
        self.rayd = rays_d.permute(2, 0, 1).unsqueeze(0).cpu()

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
        self.cam_rot_delta = self.cam_rot_delta.to(device=device)
        self.cam_trans_delta = self.cam_trans_delta.to(device=device)

        self.exposure_a = self.exposure_a.to(device=device)
        self.exposure_b = self.exposure_b.to(device=device)
        self.projection_matrix = self.projection_matrix.to(device=device)

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
            (self.FoVx, self.FoVy),
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

        self.FoVx, self.FoVy = focal2fov(self.fx, width), focal2fov(self.fy, height)
        projection_matrix = getProjectionMatrix2(znear, zfar, self.cx, self.cy, self.fx, self.fy, width, height)
        self.projection_matrix = projection_matrix.transpose(0, 1).to(device=self.device)

    def clean(self):
        self.original_image = None
        self.depth = None
        self.depth_prior = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None


def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0
