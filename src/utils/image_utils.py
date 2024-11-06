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
