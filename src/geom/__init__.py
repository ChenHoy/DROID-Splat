from typing import Tuple

import torch
import lietorch
from .projective_ops import coords_grid, projective_transform, proj, iproj


def matrix_to_lie(matrix: torch.Tensor) -> torch.Tensor:
    """Transforms a batch of 4x4 homogenous matrix into a 7D lie vector,
    see https://github.com/princeton-vl/lietorch/issues/14
    """
    from pytorch3d.transforms import matrix_to_quaternion

    # Ensure we always have a batched tensor
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)

    quat = matrix_to_quaternion(matrix[:, :3, :3])
    quat = torch.cat((quat[:, 1:], quat[:, 0][:, None]), 1)  # swap real first to real last
    trans = matrix[:, :3, 3]
    vec = torch.cat((trans, quat), 1)
    return vec


@torch.no_grad()
def align_scale_and_shift(
    prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    weighted least squares problem to solve scale and shift:
        min sum{ weight[i,j] * (prediction[i,j] * scale + shift - target[i,j])^2 }

    see: https://github.com/zhangganlin/GlORIE-SLAM/ as Reference
    NOTE chen: This is the exact same as is standard in monocular depth prediction, see our Scale-Invariant Loss from ZoeDepth

    prediction: [B, H, W]
    target: [B, H, W]
    weights: [B, H, W]
    """

    if weights is None:
        weights = torch.ones_like(prediction).to(prediction.device)
    if len(prediction.shape) < 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
        weights = weights.unsqueeze(0)

    a_00 = torch.sum(weights * prediction * prediction, dim=[1, 2])
    a_01 = torch.sum(weights * prediction, dim=[1, 2])
    a_11 = torch.sum(weights, dim=[1, 2])
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(weights * prediction * target, dim=[1, 2])
    b_1 = torch.sum(weights * target, dim=[1, 2])
    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    det = a_00 * a_11 - a_01 * a_01
    scale = (a_11 * b_0 - a_01 * b_1) / det
    shift = (-a_01 * b_0 + a_00 * b_1) / det
    error = (scale[:, None, None] * prediction + shift[:, None, None] - target).abs()
    masked_error = error * weights
    error_sum = masked_error.sum(dim=[1, 2])
    error_num = weights.sum(dim=[1, 2])
    avg_error = error_sum / error_num

    return scale, shift, avg_error


def pose_distance(
    g1: torch.Tensor | lietorch.SE3, g2: torch.Tensor | lietorch.SE3, beta: float = 0.5, radians: bool = False
) -> float:
    """Compute the distance between two SE3 elements. Since there is not left-invariant metric on SE(3),
    we need to heuristically weight the translation and rotation components.

    NOTE: beta does not depent on the scale of the scene, so be cautious here and select a good value!
    Example:
        We have a translation difference of 1m and a rotation difference of 45 degrees (0.785 in radians).
        The translation could significantly put distance between the frames,
        but we also want to take the rotation into account here! Large rotation differences are much more critical since they
        affect the final position depending on the scale of the scene as well.
    """
    if isinstance(g1, torch.Tensor):
        g1 = lietorch.SE3.InitFromVec(g1)
    if isinstance(g2, torch.Tensor):
        g2 = lietorch.SE3.InitFromVec(g2)

    g12 = g1 * g2.inv()
    d_se3 = g12.log()
    tau, phi = d_se3.split([3, 3], dim=-1)  # Separate into translation and rotation
    dt = tau.norm(dim=-1)  # NOTE We dont need the translation here
    dr = phi.norm(dim=-1)
    if not radians:
        dr = (180 / torch.pi) * dr  # convert radians to degrees
    return beta * dr + (1 - beta) * dt
