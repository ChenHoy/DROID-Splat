from typing import List, Optional, Tuple, Union
from ..utils import gradient_map
import ipdb

import torch


def l1(pred: torch.Tensor, gt: torch.Tensor):
    return torch.abs(pred - gt)


def l2(pred: torch.Tensor, gt: torch.Tensor):
    return (pred - gt) ** 2


def log_l1(pred: torch.Tensor, gt: torch.Tensor):
    return torch.log(1 + torch.abs(pred - gt))


def tv(pred: torch.Tensor, mask: Optional[torch.Tensor] = None):
    grad_x, grad_y = gradient_map(pred, return_xy=True)
    return torch.masked_select(grad_x, mask).mean() + torch.masked_select(grad_y, mask).mean()


def edge_weighted_tv(
    pred: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor] = None, weight_fn: str = "exp"
):
    grad_x, grad_y = gradient_map(pred, return_xy=True)
    ref_x, ref_y = gradient_map(ref, return_xy=True)
    # Regions of high gradient will have lower weights
    if weight_fn == "exp":
        wx, wy = torch.exp(-ref_x), torch.exp(-ref_y)
    # NOTE reference @https://github.com/hugoycj/2d-gaussian-splatting-great-again/blob/main/utils/loss_utils.py uses non-exponential weights
    else:
        assert ref.max() <= 1, "Reference needs to have normalized values for this option!"
        wx, wy = (ref_x - 1) ** 500, (ref_y - 1) ** 500
    grad_x, grad_y = wx * grad_x, wy * grad_y
    return torch.masked_select(grad_x, mask).mean() + torch.masked_select(grad_y, mask).mean()


def l1_huber_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    delta: float = 0.2,
    mask: Optional[torch.Tensor] = None,
    return_array: bool = False,
):
    l1_err = l1(pred, gt)
    if mask is None:
        mask = torch.ones_like(pred, device=pred.device)
    d = delta * torch.max(l1_err[mask])  # Get threshold
    err = torch.where(l1_err < d, ((pred - gt) ** 2 + d**2) / (2 * d), l1_err)

    # Take masked mean
    num_valid = mask.sum()
    if num_valid.item() > 0:
        loss = (mask * err).sum() / num_valid
    else:
        loss = torch.tensor(0.0, device=pred.device, requires_grad=True)

    if return_array:
        return loss, err
    else:
        return loss


def pearson_loss(
    pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None, return_array: bool = False
):
    if mask is None:
        mask = torch.ones_like(pred, device=pred.device)

    src = pred - pred[mask].mean()
    target = gt - gt[mask].mean()
    src = src / (src[mask].std() + 1e-6)
    target = target / (target[mask].std() + 1e-6)
    co = src[mask] * target[mask]
    err = 1 - co

    # Filter invalids
    # NOTE chen: for some reason this was checked in reference
    invalid = torch.isnan(co)
    if torch.any(invalid):
        print("Warning. Invalid Pearson correlation values in loss!")
    err[invalid] = 0

    # Take masked mean
    num_valid = mask.sum()
    if num_valid.item() > 0:
        # NOTE we already took the masked values in the error computation
        loss = err.sum() / num_valid
    else:
        loss = torch.tensor(0.0, device=pred.device, requires_grad=True)

    if return_array:
        return loss, err
    else:
        return loss


def l1_loss(pred: torch.Tensor, gt: torch.Tensor, **kwargs):
    return masked_loss(pred, gt, l1, **kwargs)


def l2_loss(pred: torch.Tensor, gt: torch.Tensor, **kwargs):
    return masked_loss(pred, gt, l2, **kwargs)


def log_l1_loss(pred: torch.Tensor, gt: torch.Tensor, **kwargs):
    return masked_loss(pred, gt, log_l1, **kwargs)


### Generic callable for any loss function to allow weighting and masking functionality
def masked_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    loss_func: callable,
    weights: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    return_array: bool = False,
) -> float:
    """Compute an element wise weighted loss between two tensors and compute a masked average"""
    if mask is None:
        mask = torch.ones_like(pred, device=pred.device)

    # We need the mask to compute the Huber threshold -> Pass mask to func
    if loss_func.__name__ == "l1_huber":
        assert weights is None, "Its not recommended to use weights with Huber loss!"
        return loss_func(pred, gt, mask=mask, return_array=return_array)
    else:
        err = loss_func(pred, gt)

    if weights is not None:
        err *= weights

    # Take masked mean
    num_valid = mask.sum()
    if num_valid.item() > 0:
        loss = (mask * err).sum() / num_valid
    else:
        loss = torch.tensor(0.0, device=pred.device, requires_grad=True)

    if return_array:
        return loss, err
    else:
        return loss

# NOTE this is inspired by the loss in DN-Splatter
def ms_masked_loss(
    loss_func: callable,
    pred: torch.Tensor,
    gt: torch.Tensor,
    scales=[1, 2, 4],
    weights: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    interpolation: str = "bilinear",
):
    total_loss = 0.0

    # TODO weights dont sum up to one?
    lvl_weights = [1, 0.5, 0.25, 0.125]
    for scale, w_l in zip(scales, lvl_weights):
        if scale == 1:
            # Original resolution
            total_loss += w_l * masked_loss(pred, gt, loss_func, weights=weights, mask=mask)
        else:
            # Downsampled resolution
            scaled_output = torch.nn.functional.interpolate(
                pred, scale_factor=1 / scale, mode=interpolation, align_corners=False
            )
            scaled_gt = torch.nn.functional.interpolate(
                gt, scale_factor=1 / scale, mode=interpolation, align_corners=False
            )

            if mask is not None:
                # Always use nearest interpolation for masks
                scaled_mask = torch.nn.functional.interpolate(mask.float(), scale_factor=1 / scale, mode="nearest")
            if weights is not None:
                scaled_weights = torch.nn.functional.interpolate(
                    weights, scale_factor=1 / scale, mode=interpolation, align_corners=False
                )
            total_loss += w_l * masked_loss(
                scaled_output, scaled_gt, loss_func, weights=scaled_weights, mask=scaled_mask
            )
    return total_loss
