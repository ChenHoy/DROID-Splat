from .projective_ops import coords_grid, projective_transform, proj, iproj
import lietorch
import torch


def matrix_to_lie(matrix: torch.Tensor) -> lietorch.SE3:
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
    return lietorch.SE3.InitFromVec(vec)
