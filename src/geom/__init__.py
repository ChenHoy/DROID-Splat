from typing import Tuple
from termcolor import colored
import ipdb

import torch
import lietorch
from .projective_ops import coords_grid, projective_transform, proj, iproj


# NOTE both functions are numerically precise to ~1e-7
def matrix_to_lie(matrix: torch.Tensor) -> torch.Tensor:
    """Transforms a batch of 4x4 homogenous matrix into a 7D lie vector,
    see https://github.com/princeton-vl/lietorch/issues/14

    Caution: This conversion can introduce problems, since it is not unique, e.g.
    a 180 degree rotation can be represented by two different quaternion with opposite signs.
    This case happens a lot when comparing the lie vectors in self.video and converting the homogeneous matrices
    from the Rendering ...
    """
    from pytorch3d.transforms import matrix_to_quaternion

    # Ensure we always have a batched tensor
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)

    quat = matrix_to_quaternion(matrix[:, :3, :3])
    quat = torch.cat((quat[:, 1:], quat[:, 0][:, None]), 1)  # swap real first to real last

    trans = matrix[:, :3, 3]
    # FIXME this does not work in all cases, there seems to be something significantly wrong with pytorch3d
    vec = torch.cat((trans, quat), 1)  # FIX PyTorch3D diff. coordinate system
    return vec


def lie_to_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """Transform a 7x1 tensor into a 4x4 homogenous matrix."""
    tensor_lie = lietorch.SE3.InitFromVec(tensor)
    return tensor_lie.matrix()


def quat_swap_convention(
    q: torch.Tensor,
    is_in: str = "xyzw",
) -> torch.Tensor:
    """Swap the quaternion convention from [x, y, z, w] to [w, x, y, z] or vice versa"""
    if q.ndim == 1:  # Add extra batch dimension
        had_batch_dim = False
        q = q.unsqueeze(0)
    else:
        had_batch_dim = True

    if is_in == "xyzw":
        xyz, w = q[:, :3], q[:, 3:]
        q_swapped = torch.cat((w, xyz), dim=-1)
    else:
        w, xyz = q[:, :1], q[:, 1:]
        q_swapped = torch.cat((xyz, w), dim=-1)

    if had_batch_dim:
        return q_swapped
    else:
        return q_swapped.squeeze(0)


def lie_quat_swap_convention(lie_tensor: torch.Tensor, is_in: str = "xyzw") -> torch.Tensor:
    """A lie algebra element is stored as a translation t and rotation q, e.g. [tx, ty, tz, qx, qy, qz, qw].
    However, a quaternion can be stored as either [x, y, z, w] or [w, x, y, z].

    Different frameworks use different conventions:
        lietorch / DROID-SLAM: [tx, ty, tz, qx, qy, qz, qw]
        pytorch3D / evo: [tx, ty, tz, qw, qx, qy, qz]

    returns:
    ---
    swapped [torch.Tensor]: [B, 7] lie algebra tensor
    """
    if lie_tensor.ndim == 1:  # Add extra batch dimension
        had_batch_dim = False
        lie_tensor = lie_tensor.unsqueeze(0)
    else:
        had_batch_dim = True

    t, q = lie_tensor.split([3, 4], dim=-1)
    q_swapped = quat_swap_convention(q, is_in=is_in)
    swapped = torch.cat((t, q_swapped), dim=-1)
    if had_batch_dim:
        return swapped
    else:
        return swapped.squeeze(0)


# NOTE when both quaternions are not exactly the same (which is the case for us), eps needs to at least account for the dissimilarity!
# e.g. q1 has a flipped sign of q2 and q2 = -q1 +- 0.1 -> eps should be ~0.1
def check_and_correct_rotation(q1_raw: torch.Tensor, q2_raw: torch.Tensor, eps: float = 1e-1) -> torch.Tensor:
    """Multiple rotations can achieve the same orientation, but have different signs, e.g.
    (w1, x1, y1, z1) and -(w1, x1, y1, z1) achieve similar .
    Because we would like to have unique mappings between 4x4 homogenous matrices and 7D lie vectors,
    we check with a reference.

    This reorientates q1 so it has the same sign as q2!
    """
    assert q1_raw.shape == q2_raw.shape, f"Shapes of q1 and q2 do not match: {q1_raw.shape} vs. {q2_raw.shape}"
    if q1_raw.ndim == 1:
        is_batched = False
        q1_raw = q1_raw.unsqueeze(0)
        q2_raw = q2_raw.unsqueeze(0)
    else:
        is_batched = True

    # Force unit quaternion
    q1 = q1_raw.clone() / torch.linalg.vector_norm(q1_raw, dim=-1)[:, None]
    q2 = q2_raw.clone() / torch.linalg.vector_norm(q2_raw, dim=-1)[:, None]
    x1, y1, z1, w1 = torch.split(q1, 1, dim=1)
    x2, y2, z2, w2 = torch.split(q2, 1, dim=1)
    dot_product = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2

    correction = torch.ones(q1.shape[0], device=q1.device)
    make_neg = torch.abs(dot_product.squeeze(-1) + 1.0) < eps  # dot_product = -1 +- eps indicates sign flip
    correction[make_neg] = -1.0
    # make_neg = dot_product < eps # The two quaternions are not the same, but they definitely have different signs

    corrected_q1 = correction[:, None] * q1
    if is_batched:
        return corrected_q1
    else:
        return corrected_q1.squeeze(0)


def check_and_correct_transform(g1: lietorch.SE3 | torch.Tensor, g2: lietorch.SE3 | torch.Tensor) -> torch.Tensor:
    """Given two lie algebras of shape [B, 7], we check if the rotation is the same and correct it if necessary.
    This compares the rotation of g1 with the handedness of g2 and corrects the signs if necessary.

    This reorientates g1 so it has the same sign as g2!

    NOTE this assumes that the Lie algebra is stored in convention [tx, ty, tz, qx, qy, qz, qw]!!
    """
    assert g1.shape == g2.shape, f"Shapes of g1 and g2 do not match: {g1.shape} vs. {g2.shape}"
    if isinstance(g1, lietorch.SE3):
        g1 = g1.vec()
    if isinstance(g2, lietorch.SE3):
        g2 = g2.vec()

    if g1.ndim == 1:
        is_batched = False
        g1 = g1.unsqueeze(0)
        g2 = g2.unsqueeze(0)
    else:
        is_batched = True

    q1, q2 = g1[:, 3:], g2[:, 3:]
    q1_corrected = check_and_correct_rotation(q1, q2)
    g1[:, 3:] = q1_corrected

    if is_batched:
        return g1
    else:
        return g1.squeeze(0)


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
    scale = torch.ones_like(b_0)
    shift = torch.zeros_like(b_1)

    # NOTE chen: degenerate cases do happen!
    det = a_00 * a_11 - a_01 * a_01

    # A needs to be a positive definite matrix!
    valid = det > 0
    scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

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

    NOTE: This assumes the lietorch convention for lie algebra [tx, ty, tz, qx, qy, qz, qw]!
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
