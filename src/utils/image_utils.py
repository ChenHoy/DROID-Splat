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


def mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# NOTE chen: this is similar to MonoGS, but instead of Scharr we use Sobel
# TODO exchange gradient_map with image_gradient in where this is used
def gradient_map(image: torch.Tensor):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda() / 4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda() / 4

    grad_x = torch.cat(
        [torch.nn.functional.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])]
    )
    grad_y = torch.cat(
        [torch.nn.functional.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])]
    )
    magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    magnitude = magnitude.norm(dim=0, keepdim=True)

    return magnitude


def image_gradient(image: torch.Tensor):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device=image.device)
    conv_x = torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device=image.device)
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c)
    img_grad_h = normalizer * torch.nn.functional.conv2d(p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c)
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image: torch.Tensor, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=image.device)
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=image.device)
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c)
    img_grad_h = torch.nn.functional.conv2d(p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c)

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


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
