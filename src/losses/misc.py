from typing import List, Optional, Tuple, Union
import ipdb

import torch
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def l1_loss(
    pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None, return_diff: bool = False
) -> float:
    diff = torch.abs(pred - gt)
    if mask is not None:
        loss = (diff * mask).mean()
        diff = diff * mask

    if return_diff:
        return loss, diff
    else:
        return loss


def l2_loss(pred, gt, mask: Optional[torch.Tensor] = None, return_diff: bool = False) -> float:
    diff = (pred - gt) ** 2
    if mask is not None:
        loss = (diff * mask).mean()
        diff = diff * mask

    if return_diff:
        return loss, diff
    else:
        return loss
