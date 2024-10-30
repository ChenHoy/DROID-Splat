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
import ipdb


def mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gradient_map(image: torch.Tensor, operator: str = "scharr", return_xy: bool = False):
    """Compute the image gradient with a differntial operator."""
    if operator == "prewitt":
        operator_x = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda() / 3
        operator_y = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]).float().unsqueeze(0).unsqueeze(0).cuda() / 3
    elif operator == "sobel":
        operator_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda() / 4
        operator_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda() / 4
    elif operator == "scharr":
        # NOTE original from wikipedia has weights 47, 162 and not 3, 10, seems just slightly different
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
        # Reduce RGB channels
        # NOTE chen: we could also simply average, I have not seen it done like this before
        magnitude = magnitude.norm(dim=0, keepdim=True)
        return magnitude
