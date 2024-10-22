from typing import List, Optional, Tuple, Union
import ipdb

import torch
import torch.nn.functional as F

from .misc import l1_loss
from ..utils import image_gradient_mask, image_gradient

MAX_DEPTH = 1e7
MIN_DEPTH = 0.01
MIN_NUM_POINTS = 50  # At least have 100 points for supervision


def depth_loss(
    depth_est: torch.Tensor,
    depth_gt: torch.Tensor,
    with_smoothness: bool = False,
    beta: float = 0.001,
    original_image: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    scale_invariant: bool = False,
):
    if mask is None:
        mask = torch.ones_like(depth_est, device=depth_est.device)

    if scale_invariant:
        loss_func = ScaleAndShiftInvariantLoss()
    else:
        loss_func = l1_loss

    # Sanity check against missing depths (e.g. everything got filtered out)
    if (depth_gt > 0).sum() < MIN_NUM_POINTS or mask.sum() < MIN_NUM_POINTS:
        l1_depth = 0.0
    else:
        l1_depth = loss_func(depth_est, depth_gt, mask)

    # Sanity check to avoid division by zero
    if with_smoothness and original_image is not None and mask.sum() > 0:
        depth_loss = l1_depth + beta * depth_reg(depth_est, original_image, mask=mask)
    else:
        depth_loss = l1_depth

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


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):

    mask_v, mask_h = image_gradient_mask(depth)
    if mask is not None:
        mask_v = torch.logical_and(mask_v, mask)
        mask_h = torch.logical_and(mask_h, mask)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
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
