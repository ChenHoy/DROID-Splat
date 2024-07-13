from typing import List, Optional, Tuple, Union
import ipdb

import torch
import torch.nn.functional as F

from ..gaussian_splatting.camera_utils import Camera

from ..utils import image_gradient_mask
from .depth import depth_loss
from .image import color_loss

MAX_DEPTH = 1e7
MIN_DEPTH = 0.01
MIN_NUM_POINTS = 50  # At least have 100 points for supervision


def mapping_rgbd_loss(
    image: torch.Tensor,
    depth: torch.Tensor,
    cam: Camera,
    with_edge_weight: bool = False,
    with_ssim: bool = False,
    with_depth_smoothness: bool = False,
    alpha1: float = 0.8,
    alpha2: float = 0.85,
    beta2: float = 0.001,
    rgb_boundary_threshold: float = 0.01,
    supervise_with_prior: bool = False,
    scale_invariant: bool = False,
    **kwargs,
) -> float:
    if cam.depth is not None or (cam.depth_prior is not None and supervise_with_prior):
        has_depth = True
        if supervise_with_prior:  # NOTE leon: this can be active on mono mode, but both depths are the same
            depth_gt = cam.depth_prior
        else:
            depth_gt = cam.depth
    else:
        has_depth = False

    # Transform with exposure (done in other papers)
    # image = (torch.exp(cam.exposure_a)) * image + cam.exposure_b
    image_gt = cam.original_image

    # Mask out pixels with little information and invalid depth pixels
    rgb_pixel_mask = (image_gt.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)

    # Include additional attached masks if they exist
    if cam.mask is not None:
        rgb_pixel_mask = rgb_pixel_mask & cam.mask

    if with_edge_weight:
        edge_mask_x, edge_mask_y = image_gradient_mask(image_gt)  # Use gt reference image for edge weight
        edge_mask = edge_mask_x | edge_mask_y  # Combine with logical OR
        rgb_mask = rgb_pixel_mask.float() * edge_mask.float()
    else:
        rgb_mask = rgb_pixel_mask.float()

    loss_rgb = color_loss(image, image_gt, with_ssim, alpha2, rgb_mask)
    if has_depth:
        # Only use valid depths for supervision
        depth_pixel_mask = ((depth_gt > MIN_DEPTH) * (depth_gt < MAX_DEPTH)).view(*depth.shape)
        if cam.mask is not None:
            depth_pixel_mask = depth_pixel_mask & cam.mask
        loss_depth = depth_loss(
            depth, depth_gt, with_depth_smoothness, beta2, image_gt, depth_pixel_mask, scale_invariant=scale_invariant
        )
        return alpha1 * loss_rgb + (1 - alpha1) * loss_depth
    else:
        return loss_rgb


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


def show_image(image: torch.Tensor) -> None:
    import matplotlib.pyplot as plt

    if image.shape[1] == 3:
        plt.imshow(image[0].detach().permute(1, 2, 0).cpu().numpy())
    else:
        plt.imshow(image[0].detach().cpu().numpy())
    plt.axis("off")
    plt.show()
