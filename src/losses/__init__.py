from typing import List, Optional, Tuple, Union
import ipdb

import torch
import torch.nn.functional as F

from ..gaussian_splatting.camera_utils import Camera

from .depth import depth_loss
from .image import color_loss

MAX_DEPTH = 1e3
MIN_DEPTH = 0.001
MIN_NUM_POINTS = 50  # At least have 100 points for supervision


def mapping_rgbd_loss(
    image: torch.Tensor,
    depth: torch.Tensor,
    cam: Camera,
    with_ssim: bool = False,
    with_edge_weight: bool = False,
    with_depth_smoothness: bool = False,
    supervise_with_prior: bool = False,
    rgb_boundary_threshold: float = 0.01,
    use_ms_ssim: bool = False,
    depth_func: str = "l1",
    alpha1: float = 0.8,
    alpha2: float = 0.85,
    beta2: float = 0.001,
    **kwargs
):

    if (cam.depth is not None and not supervise_with_prior) or (cam.depth_prior is not None and supervise_with_prior):
        has_depth = True
        if supervise_with_prior:  # NOTE leon: this can be active on mono mode, but both depths are the same
            depth_gt = cam.depth_prior.clone()
        else:
            depth_gt = cam.depth
    else:
        has_depth = False

    # Transform with exposure (done in other papers)
    # image = (torch.exp(cam.exposure_a)) * image + cam.exposure_b
    image_gt = cam.original_image

    # Mask out pixels with little information (from MonoGS)
    rgb_pixel_mask = (image_gt.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    # Include additional attached masks if they exist
    if cam.mask is not None:
        rgb_pixel_mask = rgb_pixel_mask & cam.mask
    rgb_mask = rgb_pixel_mask.float()
    loss_rgb = color_loss(image, image_gt, with_ssim, use_ms_ssim=use_ms_ssim, mask=rgb_mask, alpha2=alpha2)

    if has_depth:
        # Only use valid depth
        depth_pixel_mask = ((depth_gt > MIN_DEPTH) * (depth_gt < MAX_DEPTH)).view(*depth.shape)
        if cam.mask is not None:
            depth_pixel_mask = depth_pixel_mask & cam.mask

        loss_depth = depth_loss(
            depth.squeeze(),
            depth_gt.squeeze(),
            with_edge_weight,
            with_depth_smoothness,
            beta2,
            image_gt.squeeze(),
            depth_pixel_mask.squeeze(),
            depth_func,
        )
        loss = alpha1 * loss_rgb + (1 - alpha1) * loss_depth
        # loss = loss_rgb + (1 - alpha1) * loss_depth
    else:
        loss = loss_rgb

    return loss


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
