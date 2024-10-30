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


def gradient_map(image: torch.Tensor, operator: str = "scharr", return_xy: bool = False):
    """Compute the image gradient with a differntial operator."""

    if operator == "sobel":
        operator_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda() / 4
        operator_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda() / 4
    elif operator == "scharr":
        operator_x = torch.tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]]).float().unsqueeze(0).unsqueeze(0).cuda() / 16
        operator_y = torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]]).float().unsqueeze(0).unsqueeze(0).cuda() / 16
    else:
        raise Exception(f"Operator {operator} not supported. Use 'sobel' or 'scharr'.")

    grad_x = torch.cat(
        [torch.nn.functional.conv2d(image[i].unsqueeze(0), operator_x, padding=1) for i in range(image.shape[0])]
    )
    grad_y = torch.cat(
        [torch.nn.functional.conv2d(image[i].unsqueeze(0), operator_y, padding=1) for i in range(image.shape[0])]
    )

    if return_xy:
        return grad_x, grad_y
    else:
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        magnitude = magnitude.norm(dim=0, keepdim=True)
        return magnitude


def colormap(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map = (map - map.min()) / (map.max() - map.min())
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2, 0, 1)
    return map


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
