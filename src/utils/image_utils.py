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


def mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


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
