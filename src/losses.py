from typing import List, Optional

import numpy as np
import torch
from pytorch_msssim import ssim, ms_ssim

from .gaussian_splatting.utils.loss_utils import l1_loss
from .gaussian_splatting.camera_utils import Camera
from .gaussian_splatting.slam_utils import depth_reg, image_gradient_mask

MAX_DEPTH = 1e7
MIN_DEPTH = 0.01


def mapping_rgbd_loss(
    image: torch.Tensor,
    depth: torch.Tensor,
    cam: Camera,
    with_edge_weight: bool = False,
    with_ssim: bool = False,
    with_depth_smoothness: bool = False,
    alpha1: float = 0.8,
    alpha2: float = 0.85,
    beta: float = 0.001,
    rgb_boundary_threshold: float = 0.01,
    filter_dyn: bool = False,
) -> float:
    has_depth = True
    if cam.depth is None:
        has_depth = False

    # Mask out pixels with little information and invalid depth pixels
    rgb_pixel_mask = (cam.original_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    # Only compute the loss in static regions
    if filter_dyn:
        rgb_pixel_mask = rgb_pixel_mask | cam.stat_mask

    if has_depth:
        # Only use valid depths for supervision
        depth_pixel_mask = ((cam.depth > MIN_DEPTH) * (cam.depth < MAX_DEPTH)).view(*depth.shape)
        if filter_dyn:
            depth_pixel_mask = depth_pixel_mask | cam.stat_mask

    if with_edge_weight:
        edge_mask_x, edge_mask_y = image_gradient_mask(cam.original_image)  # Use gt reference image for edge weight
        edge_mask = edge_mask_x | edge_mask_y  # Combine with logical OR
        rgb_mask = rgb_pixel_mask.float() * edge_mask.float()
    else:
        rgb_mask = rgb_pixel_mask.float()

    loss_rgb = color_loss(image, cam, with_ssim, alpha2, rgb_mask)
    if has_depth:
        loss_depth = depth_loss(depth, cam, with_depth_smoothness, beta, depth_pixel_mask)
        return alpha1 * loss_rgb + (1 - alpha1) * loss_depth
    else:
        return loss_rgb


def color_loss(
    image: torch.Tensor,
    cam: Camera,
    with_ssim: bool = True,
    alpha2: float = 0.85,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute the color loss between the rendered image and the ground truth image.
    This uses a weighted sum of l1 and ssim loss.
    """
    if mask is None:
        mask = torch.ones_like(image, device=image.device)

    image = (torch.exp(cam.exposure_a)) * image + cam.exposure_b
    l1_rgb = l1_loss(image, cam.original_image, mask)
    # NOTE this is configured like is done in most monocular depth estimation supervision pipelines
    if with_ssim:
        ssim_loss = ssim(image.unsqueeze(0), cam.original_image.unsqueeze(0), data_range=1.0, size_average=True)
        rgb_loss = 0.5 * alpha2 * (1 - ssim_loss) + (1 - alpha2) * l1_rgb
    else:
        rgb_loss = l1_rgb
    return rgb_loss


def depth_loss(
    depth: torch.Tensor,
    cam: Camera,
    with_smoothness: bool = False,
    beta: float = 0.001,
    mask: Optional[torch.Tensor] = None,
) -> float:
    if mask is None:
        mask = torch.ones_like(depth, device=depth.device)

    l1_depth = l1_loss(cam.depth, depth, mask)
    if with_smoothness:
        depth_reg_loss = depth_reg(depth, cam.original_image)
        depth_loss = l1_depth + beta * depth_reg_loss
    else:
        depth_loss = l1_depth

    return depth_loss


# TODO implement this
# TODO use a separate list of gaussians for dyn. objects
# TODO chen: refactor this into a loss file
# i) static loss
# ii) dynamic loss in batch mode
# iii) dynamic regularizer loss for scale changes and trajectory changes
def dynamic_gaussian_loss(image: torch.Tensor, depth: torch.Tensor, cam: Camera) -> float:
    raise NotImplementedError()


def plot_losses(
    loss_list: List[float],
    refinement_iters: int,
    title: str = "Loss evolution",
    output_file: Optional[str] = "losses.png",
) -> None:
    """Plot the loss curve.

    Example usage:
        plot_losses(
            self.loss_list,
            self.refinement_iters,
            title=f"Loss evolution.{len(self.gaussians)} gaussians",
            output_file=f"{self.output}/loss_{self.mode}.png",
        )
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1)
    ax[0].set_title(title)
    ax[0].set_yscale("log")
    ax[0].plot(loss_list)

    ax[1].set_yscale("log")
    ax[1].plot(loss_list[-refinement_iters:])
    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()
