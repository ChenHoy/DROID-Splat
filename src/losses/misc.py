from typing import List, Optional, Tuple, Union
import ipdb

import torch


def l1_loss(
    pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None, return_diff: bool = False
) -> float:
    diff = torch.abs(pred - gt)
    if mask is not None:
        diff = diff * mask
    loss = diff.mean()

    if return_diff:
        return loss, diff
    else:
        return loss


def l2_loss(pred, gt, mask: Optional[torch.Tensor] = None, return_diff: bool = False) -> float:
    diff = (pred - gt) ** 2
    if mask is not None:
        diff = diff * mask
    loss = diff.mean()

    if return_diff:
        return loss, diff
    else:
        return loss
