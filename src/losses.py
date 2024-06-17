from typing import List, Optional, Tuple, Union
import warnings
import ipdb

import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor

from .gaussian_splatting.utils.loss_utils import l1_loss
from .gaussian_splatting.camera_utils import Camera
from .gaussian_splatting.slam_utils import depth_reg, image_gradient_mask

MAX_DEPTH = 1e7
MIN_DEPTH = 0.01


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

        return x_0, x_1

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        interpolate: bool = True,
    ) -> bool:
        if prediction.shape[-1] != target.shape[-1] and interpolate:
            prediction = F.interpolate(prediction, target.shape[-2:], mode="bilinear", align_corners=True)

        prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze()
        assert (
            prediction.shape == target.shape
        ), f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        scale, shift = self.compute_scale_and_shift(prediction, target, mask)
        scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        return F.l1_loss(scaled_prediction[mask], target[mask])


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
    supervise_with_prior: bool = False,
    scale_invariant: bool = False,
) -> float:
    if cam.depth is None and cam.depth_prior is None:
        has_depth = False
    else:
        has_depth = True
        if supervise_with_prior:
            depth_gt = cam.depth_prior
        else:
            depth_gt = cam.depth

    image = (torch.exp(cam.exposure_a)) * image + cam.exposure_b  # Transform with exposure to get more realistic image
    image_gt = cam.original_image

    # Mask out pixels with little information and invalid depth pixels
    rgb_pixel_mask = (image_gt.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    # Include additional attached masks if they exist
    if cam.mask is not None:
        rgb_pixel_mask = rgb_pixel_mask & cam.mask

    if has_depth:
        # Only use valid depths for supervision
        depth_pixel_mask = ((depth_gt > MIN_DEPTH) * (depth_gt < MAX_DEPTH)).view(*depth.shape)
        if cam.mask is not None:
            depth_pixel_mask = depth_pixel_mask & cam.mask

    if with_edge_weight:
        edge_mask_x, edge_mask_y = image_gradient_mask(image_gt)  # Use gt reference image for edge weight
        edge_mask = edge_mask_x | edge_mask_y  # Combine with logical OR
        rgb_mask = rgb_pixel_mask.float() * edge_mask.float()
    else:
        rgb_mask = rgb_pixel_mask.float()

    loss_rgb = color_loss(image, image_gt, with_ssim, alpha2, rgb_mask)
    if has_depth:
        loss_depth = depth_loss(
            depth, depth_gt, with_depth_smoothness, beta, image_gt, depth_pixel_mask, scale_invariant=scale_invariant
        )
        return alpha1 * loss_rgb + (1 - alpha1) * loss_depth
    else:
        return loss_rgb


def color_loss(
    image_est: torch.Tensor,
    image_gt: torch.Tensor,
    with_ssim: bool = True,
    alpha2: float = 0.85,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute the color loss between the rendered image and the ground truth image.
    This uses a weighted sum of l1 and ssim loss.
    """
    if mask is None:
        mask = torch.ones_like(image_est, device=image_est.device)

    l1_rgb = l1_loss(image_est, image_gt, mask)
    # NOTE this is configured like is done in most monocular depth estimation supervision pipelines
    if with_ssim:
        # ssim_loss = ms_ssim(
        #     image_est.unsqueeze(0), image_gt.unsqueeze(0), data_range=1.0, mask=mask.bool(), size_average=False
        # )
        ssim_loss = ssim(
            image_est.unsqueeze(0),
            image_gt.unsqueeze(0),
            data_range=1.0,
            mask=mask.unsqueeze(0).bool(),
            size_average=True,
        )
        rgb_loss = 0.5 * alpha2 * (1 - ssim_loss) + (1 - alpha2) * l1_rgb
    else:
        rgb_loss = l1_rgb
    return rgb_loss


def depth_loss(
    depth_est: torch.Tensor,
    depth_gt: torch.Tensor,
    with_smoothness: bool = False,
    beta: float = 0.001,
    original_image: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    scale_invariant: bool = False,
) -> float:
    if mask is None:
        mask = torch.ones_like(depth_est, device=depth_est.device)

    if scale_invariant:
        loss_func = ScaleAndShiftInvariantLoss()
    else:
        loss_func = l1_loss

    # Sanity check against missing depths (e.g. everything got filtered out)
    if depth_gt.sum() == 0 or mask.sum() == 0:
        l1_depth = 0.0
    else:
        l1_depth = loss_func(depth_est, depth_gt, mask)

    if with_smoothness and original_image is not None:
        # Sanity check against missing depths (e.g. everything got filtered out)
        if mask.sum() == 0:
            depth_reg_loss = 0.0
        else:
            depth_reg_loss = depth_reg(depth_est, original_image, mask=mask)
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


def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: Tensor, win: Tensor) -> Tensor:
    r"""Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def show_image(image: torch.Tensor) -> None:
    import matplotlib.pyplot as plt

    if image.shape[1] == 3:
        plt.imshow(image[0].detach().permute(1, 2, 0).cpu().numpy())
    else:
        plt.imshow(image[0].detach().cpu().numpy())
    plt.axis("off")
    plt.show()


def _ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float,
    win: Tensor,
    size_average: bool = True,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        data_range (float or int): value range of input images. (usually 1.0 or 255)
        win (torch.Tensor): 1-D gauss kernel
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        mask (torch.Tensor): boolean mask same size as X and Y

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if mask is not None:
        kernel_size = (win.shape[-1] - 1) // 2
        # Since we dont use padding, we dont have the same resolution after convolution
        mask_conv = mask[..., kernel_size:-kernel_size, kernel_size:-kernel_size]
        ssim = torch.masked_select(ssim_map, mask_conv[:, None]).mean()
        cs = torch.masked_select(cs_map, mask_conv[:, None]).mean()
        return ssim, cs
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
    mask: Optional[Tensor] = None,
) -> Tensor:
    r"""interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
        mask (torch.Tensor): boolean mask same size as X and Y

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    if mask is not None and mask.shape != X.shape:
        raise ValueError(
            f"Input mask should have the same dimensions as input images, but got {mask.shape} and {X.shape}."
        )

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)
        if mask is not None:
            mask = mask.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")
    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K, mask=mask)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if mask is not None:
        return ssim_per_channel
    elif size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    weights: Optional[List[float]] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    mask: Optional[Tensor] = None,
) -> Tensor:
    r"""interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        mask (torch.Tensor): boolean mask same size as X and Y
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2**4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2**4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights_tensor = X.new_tensor(weights)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights_tensor.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K, mask=mask)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)
            mask = None if mask is None else avg_pool(mask.float(), kernel_size=2, padding=padding)
            mask = None if mask is None else mask.bool()

    ssim_per_channel = torch.relu(ssim_per_channel)  # type: ignore  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights_tensor.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
        nonnegative_ssim: bool = False,
    ) -> None:
        r"""class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X: Tensor, Y: Tensor, mask: Optional[Tensor]) -> Tensor:
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
            mask=mask,
        )


class MS_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        weights: Optional[List[float]] = None,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    ) -> None:
        r"""class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X: Tensor, Y: Tensor, mask: Optional[Tensor]) -> Tensor:
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
            mask=mask,
        )
