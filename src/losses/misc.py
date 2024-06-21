from typing import List, Optional, Tuple, Union

import torch


def l1_loss(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    if mask is not None:
        return torch.abs((pred - gt) * mask).mean()
    else:
        return torch.abs((pred - gt)).mean()


def l2_loss(pred, gt, mask: Optional[torch.Tensor] = None) -> float:
    if mask is not None:
        return ((pred - gt) ** 2 * mask).mean()
    else:
        return ((pred - gt) ** 2).mean()
