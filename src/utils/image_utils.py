# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import matplotlib.pyplot as plt
import numpy as np


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def colormap(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map = (map - map.min()) / (map.max() - map.min())
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2, 0, 1)
    return map


def gradient_map(image: torch.Tensor, operator: str = "sobel", return_xy: bool = False):
    """Compute the image gradient with a differntial operator."""
    if operator == "prewitt":
        kernel_x = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0) / 3
        kernel_y = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]).float().unsqueeze(0).unsqueeze(0) / 3
    elif operator == "sobel":
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0) / 4
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0) / 4
    elif operator == "scharr":
        # NOTE original from wikipedia has weights 47, 162 and not 3, 10, seems just slightly different
        kernel_x = torch.tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]]).float().unsqueeze(0).unsqueeze(0) / 16
        kernel_y = torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]]).float().unsqueeze(0).unsqueeze(0) / 16
    else:
        raise Exception(f"Operator {operator} not supported. Use 'sobel' or 'scharr'.")
    kernel_x, kernel_y = kernel_x.to(image.device), kernel_y.to(image.device)

    # We have a batch of images of size [B, C, H, W]
    if image.ndim == 4:
        grad_x = torch.cat(
            [torch.nn.functional.conv2d(image[:, i], kernel_x, padding=1) for i in range(image.shape[1])], dim=1
        )
        grad_y = torch.cat(
            [torch.nn.functional.conv2d(image[:, i], kernel_y, padding=1) for i in range(image.shape[1])], dim=1
        )
        # Reduce RGB channels
        # NOTE chen: reduction over RGB channels is not straight forward, some simply use: |R| + |G| + |B|
        # Norm also ensures that gradients with opposing directions not cancel each other out
        grad_x, grad_y = torch.abs(grad_x).mean(1), torch.abs(grad_y).mean(1)

    # We have a single image of size [C, H, W]
    elif image.ndim == 3:
        grad_x = torch.cat(
            [torch.nn.functional.conv2d(image[i].unsqueeze(0), kernel_x, padding=1) for i in range(image.shape[0])]
        )
        grad_y = torch.cat(
            [torch.nn.functional.conv2d(image[i].unsqueeze(0), kernel_y, padding=1) for i in range(image.shape[0])]
        )
        # Reduce RGB channels
        grad_x, grad_y = torch.abs(grad_x).mean(0), torch.abs(grad_y).mean(0)
    # We have a single image of size [H, W]
    else:
        grad_x = torch.nn.functional.conv2d(image.unsqueeze(0), kernel_x, padding=1).squeeze()
        grad_y = torch.nn.functional.conv2d(image.unsqueeze(0), kernel_y, padding=1).squeeze()

    if return_xy:
        return grad_x, grad_y
    else:
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        return magnitude


# TODO this needs to be correctly imported somewhere else
# NOTE chen: this is kinda useless, as we simply can use the dictionary ourselves, use this as a guide for how to compute e.g. normals
def render_net_image(render_pkg, render_items, render_mode, camera):
    output = render_items[render_mode].lower()
    if output == "alpha":
        net_image = render_pkg["rend_alpha"]
    elif output == "normal":
        net_image = render_pkg["rend_normal"]
        net_image = (net_image + 1) / 2
    elif output == "depth":
        net_image = render_pkg["surf_depth"]
    elif output == "edge":
        net_image = gradient_map(render_pkg["render"])
    elif output == "curvature":
        net_image = render_pkg["rend_normal"]
        net_image = (net_image + 1) / 2
        net_image = gradient_map(net_image)
    else:
        net_image = render_pkg["render"]

    if net_image.shape[0] == 1:
        net_image = colormap(net_image)
    return net_image
