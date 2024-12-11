from typing import List, Optional, Tuple, Union
import ipdb

import torch
import torch.nn.functional as F

from .misc import l1_loss, edge_weighted_tv, pearson_loss, log_l1_loss, l1_huber_loss
from ..utils import gradient_map

MAX_DEPTH = 1e7
MIN_DEPTH = 0.01
MIN_NUM_POINTS = 50  # At least have 100 points for supervision


def depth_loss(
    depth_est: torch.Tensor,
    depth_gt: torch.Tensor,
    with_edge_weight: bool = False,
    with_smoothness: bool = False,
    beta: float = 0.001,
    original_image: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    depth_func: str = "l1",
):
    """L1 (+ smoothness) loss between estimated and ground truth depth maps.

    We support: 'l1', 'log_l1', 'l1_huber', 'pearson' for depth loss computation

    NOTE: DN-Splatter https://arxiv.org/pdf/2403.17822 computes an edge-aware log-depth loss
    """
    # Sanity check against missing depths (e.g. everything got filtered out)
    if (depth_gt > 0).sum() < MIN_NUM_POINTS or mask.sum() < MIN_NUM_POINTS:
        depth_loss = torch.tensor(0.0, device=depth_est.device, requires_grad=True)
    else:
        if with_edge_weight:
            grad_img = gradient_map(original_image)
            weights = torch.exp(-grad_img)
        else:
            weights = None

        if depth_func == "l1":
            depth_loss = l1_loss(depth_est, depth_gt, mask=mask, weights=weights)
        elif depth_func == "log_l1":
            depth_loss = log_l1_loss(depth_est, depth_gt, mask=mask, weights=weights)
        elif depth_func == "l1_huber":
            depth_loss = l1_huber_loss(depth_est, depth_gt, mask=mask)
        elif depth_func == "pearson":
            depth_loss = pearson_loss(depth_est, depth_gt, mask=mask)
        else:
            raise NotImplementedError(f"Depth loss function {depth_func} not implemented!")

    # Sanity check to avoid division by zero
    if with_smoothness and original_image is not None and mask.sum() > 0:
        depth_loss = depth_loss + beta * edge_weighted_tv(depth_est, original_image, mask=mask)

    return depth_loss


class ScaleAndShiftInvariantLoss(torch.nn.Module):
    """
    Scale-invariant L1 loss from ZoeDepth (monocular depth prediction)

    see: https://github.com/isl-org/ZoeDepth/blob/main/zoedepth/trainers/loss.py
    """

    def __init__(self):
        super().__init__()
        self.name = "SSILoss"

    # TODO does this have a sanity check for when the linear system if not solvable?
    def compute_scale_and_shift(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[float, float]:
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if prediction.ndim == 2:
            prediction = prediction.unsqueeze(0)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        # Increase precision because we sum over potentially large arrays
        target, prediction, mask = target.double(), prediction.double(), mask.double()

        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        # A needs to be a positive definite matrix.
        valid = det > 0

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0.squeeze().float(), x_1.squeeze().float()

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, interpolate: bool = True
    ) -> torch.Tensor:
        if prediction.shape[-1] != target.shape[-1] and interpolate:
            prediction = F.interpolate(prediction, target.shape[-2:], mode="bilinear", align_corners=True)

        prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze()
        assert (
            prediction.shape == target.shape
        ), f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        scale, shift = self.compute_scale_and_shift(prediction, target, mask)
        scaled_prediction = scale * prediction + shift
        return F.l1_loss(scaled_prediction[mask], target[mask])


def monogs_depth_reg(depth: torch.Tensor, gt_image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """Ensure that the depth is smooth in regions where the image gradient is low."""

    def image_gradient_mask(image: torch.Tensor, eps=0.01):
        # Compute image gradient mask
        c = image.shape[0]
        conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=image.device)
        conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=image.device)
        p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
        p_img = torch.abs(p_img) > eps
        img_grad_v = torch.nn.functional.conv2d(p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c)
        img_grad_h = torch.nn.functional.conv2d(p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c)

        return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)

    mask_v, mask_h = image_gradient_mask(depth)
    if mask is not None:
        mask_v = torch.logical_and(mask_v, mask)
        mask_h = torch.logical_and(mask_h, mask)
    gray_grad_v, gray_grad_h = gradient_map(gt_image.mean(dim=0, keepdim=True), return_xy=True)
    depth_grad_v, depth_grad_h = gradient_map(depth, return_xy=True)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (w_v * torch.abs(depth_grad_v)).mean()
    return err

def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
