import ipdb
from typing import Optional

import torch
from einops import rearrange, einsum
from torch_scatter import scatter_sum

from .chol import schur_solve, block_solve
from .projective_ops import projective_transform
from lietorch import SE3
import lietorch

"""
Bundle Adjustment for SLAM implemented in pure Python. 
This is supposed to replicate the CUDA kernel.

NOTE PyTorch 2.1.2 and Python3.11 somehow results in float16 output of matmul 
of two float32 inputs! :/
We had better precision using Python 3.9 and this version of torch. This the reason 
why all matmul / einsum operations are cast to float32 afterwards.

NOTE DO NOT USE THESE FUNCTIONS FOR LARGE-SCLAE SYSTEMS!
The dense Cholesky decomposition solver is numerically unstable for very sparse systems.
Example: These work fine in the frontend until we hit a loop-closure, which suddenly creates a very 
large window [t0, t1] with a lot of sparse entries. The dense solver will then result in nan's.

NOTE Hessian H needs to be positive definite, the Schur complement S should also be positive definite!
we observe not only here but also in the working examples from lietorch, that this is not the case :/
H is symmetric, but only the diagonal elements Hii and Hjj have positive real eigenvalues. This due to 
how we define the residuals! If you define them using two poses gi & gj, there will be off diagonal elements. 
DROID-SLAM uses the same formulation like DSO does!
"""


def safe_scatter_add_mat(A: torch.Tensor, ii, jj, n: int, m: int) -> torch.Tensor:
    """Turn a dense (B, N, D, D) matrix into a sparse (B, n*m, D, D) matrix by
    scattering with the indices ii and jj.

    Example:
        In Bundle Adjustment we compute dense Hessian blocks for N camera nodes that are part of the whole scene
        defined by ii, jj. We now need the Hessian for the whole Scene window.
        We thus scatter this depending on the indices into a bigger N*N matrix,
        where each indices ij is the dependency between camera i and camera j.
    """
    # Filter out any negative and out of bounds indices
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:, v], ii[v] * m + jj[v], dim=1, dim_size=n * m)


def safe_scatter_add_vec(b, ii, n):
    """Turn a dense (B, k, D) vector into a sparse (B, n, D) vector by
    scattering with the indices ii, where n >> k
    """
    # Filter out any negative and out of bounds indices
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:, v], ii[v], dim=1, dim_size=n)


def safe_scatter_add_mat_inplace(H, data, ii, jj, B, M, D):
    v = (ii >= 0) & (jj >= 0)
    H.scatter_add_(1, (ii[v] * M + jj[v]).view(1, -1, 1, 1).repeat(B, 1, D, D), data[:, v])


def safe_scatter_add_vec_inplace(b, data, ii, B, M, D):
    v = ii >= 0
    b.scatter_add_(1, ii[v].view(1, -1, 1).repeat(B, 1, D), data[:, v])


def disp_retr(disps: torch.Tensor, dz: torch.Tensor, ii) -> torch.Tensor:
    """Apply retraction operator to inv-depth maps where dz can be
    multiple updates (results from multiple constraints) which are scatter summed
    to update the key frames ii.

    d_k2 = d_k1 + dz
    """
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])


def pose_retr(poses: SE3, dx: torch.Tensor, ii) -> SE3:
    """Apply retraction operator to poses, where dx are lie algebra
    updates which come from multiple constraints and are scatter summed
    to the key frame poses ii.

    g_k2 = exp^(dx) * g_k1
    """
    ii = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))


def is_positive_definite(matrix: torch.Tensor, eps=2e-5) -> bool:
    """Check if a Matrix is positive definite by checking symmetry looking at the eigenvalues"""
    return bool((abs(matrix - matrix.mT) < eps).all() and (torch.linalg.eigvals(matrix).real >= 0).all())


# i) Implement the disps_sens part like in the CUDA kernel
# ii) Implement our scale optimization and dont optimize the poses
# NOTE the original RGBD SLAM by Teed formulation actually is || r || ** 2 + alpha * || D - D_prior || ** 2
# They already adapted the Jacobian / Hessian to account for this
# They use alpha = 0.05 per default
def BA_prior(
    target: torch.Tensor,
    weight: torch.Tensor,
    eta: torch.Tensor,
    poses: SE3,
    disps: torch.Tensor,
    disps_sens: torch.Tensor,
    intrinsics: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    t0: int,
    t1: int,
    ep: float = 0.1,
    lm: float = 1e-4,
    alpha: float = 0.05,
):
    """Bundle Adjustment for optimizing the disparity according to current poses and a depth prior.
    Normally we could also update the poses with this combined objective, but HI-SLAM only optimizes the depths.

    How is disps_sens aka a prior used in the original cuda BA?
    ---
    m = disps_sens[kx] > 0 (Check for where depth is defined)
    alpha = 0.05
    C = C[ii, jj, kx] + alpha * m + (1 - m) * eta (Add alpha only for where there is a prior, else only add normal damping)
        (This makes sense as the Jacobian of the second objective will just be alpha*r)
        (If disps_sens is 0 -> C = C + eta * I)
    w = w[ii, kx] - m * alpha * (disps[kx] - disps_sens[kx]) (This just adds the second residual on top)
    """
    raise NotImplementedError("This function is not implemented yet!")


def bundle_adjustment(
    target: torch.Tensor,
    weight: torch.Tensor,
    damping: torch.Tensor,
    poses: torch.Tensor,
    disps: torch.Tensor,
    disps_sens: Optional[torch.Tensor],
    intrinsics: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    t0: int,
    t1: int,
    iters: int = 4,
    lm: float = 1e-4,
    ep: float = 0.1,
    structure_only: bool = False,
    motion_only: bool = False,
) -> None:
    """Wrapper function around different bundle adjustment methods."""

    assert sum([structure_only, motion_only]) <= 1, "You can either optimize only motion or structure or both!"

    Gs = lietorch.SE3(poses[None, ...])
    # Batch up the tensors to work with the pure Python code
    disps, weight = disps.unsqueeze(0), weight.unsqueeze(0)
    target, intrinsics = target.unsqueeze(0), intrinsics.unsqueeze(0)
    if disps_sens is not None:
        disps_sens = disps_sens.unsqueeze(0)
    else:
        disps_sens = None

    for i in range(iters):
        if motion_only:
            MoBA(target, weight, damping, Gs, disps, intrinsics, ii, jj, t0, t1, ep, lm)

        if structure_only:
            BA(target, weight, damping, Gs, disps, disps_sens, intrinsics, ii, jj, t0, t1, ep, lm, True)
            disps.clamp(min=0.001)

        if not motion_only and not structure_only:
            BA(target, weight, damping, Gs, disps, disps_sens, intrinsics, ii, jj, t0, t1, ep, lm)
            disps.clamp(min=0.001)

    # Update the video
    poses = Gs.data[0]
    disps = disps[0]  # Remove the batch dimension again


def get_hessian_and_rhs(
    Jz: torch.Tensor,
    Ji: torch.Tensor,
    Jj: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    coords: torch.Tensor,
    valid: torch.Tensor,
    with_structure: bool = True,
):
    """Get mixed terms of Jacobian and Hessian for constructing the linear system"""
    B, N, _, ht, wd = target.shape
    # Reshape to residuals vector
    r = rearrange(target, "b n xy h w -> b n h w xy") - coords
    # Filter out super large residuals
    valid *= (r.norm(dim=-1) < 250.0).float().unsqueeze(-1)
    r = rearrange(r.double(), "b n h w xy -> b n (h w xy)")
    w = 0.001 * (valid * rearrange(weight, "b n xy h w -> b n h w xy"))
    w = rearrange(w.double(), "b n h w xy -> b n (h w xy) 1")

    Ji = rearrange(Ji, "b n h w xy D -> b n (h w xy) D")
    Jj = rearrange(Jj, "b n h w xy D -> b n (h w xy) D")
    wJiT, wJjT = (w * Ji).mT, (w * Jj).mT

    # Each block is B x N x D x D
    Hii = einsum(wJiT, Ji, "b n i j , b n j k -> b n i k").double()
    Hij = einsum(wJiT, Jj, "b n i j , b n j k -> b n i k").double()
    Hji = einsum(wJjT, Ji, "b n i j , b n j k -> b n i k").double()
    Hjj = einsum(wJjT, Jj, "b n i j , b n j k -> b n i k").double()
    # Each rhs term is B x N x D x 1
    vi = einsum(wJiT, r, "b n D hwxy, b n hwxy -> b n D").double()
    vj = einsum(wJjT, r, "b n D hwxy, b n hwxy -> b n D").double()

    if not with_structure:
        return Hii, Hij, Hji, Hjj, vi, vj

    # Mixed term of camera and disp blocks
    # (BNHW x D x 2) x (BNHW x 2 x 1) -> (BNHW x D x 1)
    Jz = rearrange(Jz, "b n h w xy 1 -> b n (h w) xy 1")
    wJiT = rearrange(wJiT, "b n D (hw xy) -> b n hw D xy", hw=ht * wd, xy=2)
    wJjT = rearrange(wJjT, "b n D (hw xy) -> b n hw D xy", hw=ht * wd, xy=2)
    Eik = torch.matmul(wJiT, Jz).squeeze(-1).double()
    Ejk = torch.matmul(wJjT, Jz).squeeze(-1).double()

    # Sparse diagonal block of disparities only
    w = rearrange(w, "b n (hw xy) 1 -> b n hw xy", hw=ht * wd)
    r = rearrange(r, "b n (hw xy) -> b n hw xy", hw=ht * wd)
    wJzT = (w[..., None] * Jz).mT  # (B N HW 1 XY)
    wk = einsum(wJzT.squeeze(-2), r, "b n hw xy, b n hw xy -> b n hw").double()
    Ck = einsum(wJzT.squeeze(-2), Jz.squeeze(-1), "b n hw xy, b n hw xy -> b n hw")
    Ck = Ck.double()

    return Hii, Hij, Hji, Hjj, Eik, Ejk, Ck, vi, vj, wk


def MoBA(
    target: torch.Tensor,
    weight: torch.Tensor,
    eta: torch.Tensor,
    all_poses: SE3,
    all_disps: torch.Tensor,
    all_intrinsics: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    t0: int,
    t1: int,
    ep: float = 0.1,
    lm: float = 1e-4,
    rig: int = 1,
) -> None:
    """Motion only bundle adjustment for optimizing pose nodes inside a window [t0, t1].
    The factor graph is defined by ii, jj.

    NOTE This always builds the system for poses 0:t1, but
    then excludes all poses as fixed before t0.
    """
    # Select keyframe window to optimize over!
    disps, poses, intrinsics = all_disps[:, :t1], all_poses[:, :t1], all_intrinsics[:, :t1]

    # Always fix the first pose and then fix all poses outside of optimization window
    fixedp = max(t0, 1)

    B, M, ht, wd = disps.shape  # M is a all cameras
    N = ii.shape[0]  # Number of edges for relative pose constraints
    D = poses.manifold_dim  # 6 for SE3, 7 for SIM3

    ### 1: compute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = projective_transform(poses, disps, intrinsics, ii, jj, jacobian=True)
    # NOTE normally this should be -Ji / -Jj and then vi / vj = -J^T @ r
    Ji, Jj, Jz = Ji.double(), Jj.double(), Jz.double()
    ### 2: Construct linear system
    Hii, Hij, Hji, Hjj, vi, vj = get_hessian_and_rhs(Jz, Ji, Jj, target, weight, coords, valid, with_structure=False)

    # only optimize keyframe poses
    M = M - fixedp
    ii = ii / rig - fixedp
    jj = jj / rig - fixedp
    ii, jj = ii.to(torch.int64), jj.to(torch.int64)

    # Assemble larger sparse system for optimization window
    H = torch.zeros(B, M * M, D, D, device=target.device, dtype=torch.float64)
    safe_scatter_add_mat_inplace(H, Hii, ii, ii, B, M, D)
    safe_scatter_add_mat_inplace(H, Hij, ii, jj, B, M, D)
    safe_scatter_add_mat_inplace(H, Hji, jj, ii, B, M, D)
    safe_scatter_add_mat_inplace(H, Hjj, jj, jj, B, M, D)
    H = H.reshape(B, M, M, D, D)

    v = torch.zeros(B, M, D, device=target.device, dtype=torch.float64)
    safe_scatter_add_vec_inplace(v, vi, ii, B, M, D)
    safe_scatter_add_vec_inplace(v, vj, jj, B, M, D)

    ### 3: solve the system + apply retraction ###
    dx = block_solve(H, v, ep=ep, lm=lm)
    if torch.isnan(dx).any():
        dx = torch.zeros_like(dx)
        print("Cholesky decomposition failed, using zero update instead")

    # Update only un-fixed poses
    poses1, poses2 = poses[:, :fixedp], poses[:, fixedp:]
    poses2 = poses2.retr(dx)
    poses = lietorch.cat([poses1, poses2], dim=1)
    # Update global poses
    all_poses[:, :t1] = poses


def BA(
    target: torch.Tensor,
    weight: torch.Tensor,
    eta: torch.Tensor,
    all_poses: SE3,
    all_disps: torch.Tensor,
    all_disps_sens: Optional[torch.Tensor],
    all_intrinsics: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    t0: int,
    t1: int,
    ep: float = 0.1,
    lm: float = 1e-4,
    alpha: float = 0.05,
    rig: int = 1,
    structure_only: bool = False,
) -> None:
    """Bundle Adjustment for optimizing both poses and disparities.

    Shapes that should be expected:
        target / r: (B, N, 2, H, W) -> (BNHW x 2) = (M x 2), i.e. we have BNHW points and 2 (xy) coordinates
        Ji / Jj: (M x 2 x 6) since our function maps a 6D input to a 2D vector
        Jz: (M x 2 x 1) since for the structure we only optimize the 1D disparity
        H: J^T * J = (M x 6 x 6)
        Ji^T * r / Jj^T * r (v): (M x 6 x 2) * (M x 2 x 1) = (M x 6 x 1)
        Jz^T * r (w): (M x 1 x 2) x (M x 2 x 1) = (M x 1 x 1)

    args:
    ---
        target: Predicted reprojected coordinates from node ci -> cj of shape (|E|, 2, H, W)
        weight: Predicted confidence weights of "target" of shape (|E|, 2, H, W)
        eta: Levenberg-Marqhart damping factor on depth of shape (|V|, H, W)
        ii: Timesteps of outgoing edges of shape (|E|)
        jj: Timesteps of incoming edges of shape (|E|)
        t0: Optimization window start time
        t1: Optimitation window end time
    """
    # Select keyframe window to optimize over!
    disps, poses, intrinsics = all_disps[:, :t1], all_poses[:, :t1], all_intrinsics[:, :t1]
    if all_disps_sens is not None:
        disps_sens = all_disps_sens[:, :t1]
    else:
        disps_sens = None

    # Always fix the first pose and then fix all poses outside of optimization window
    fixedp = max(t0, 1)

    B, M, ht, wd = disps.shape  # M is a all cameras
    num_edges = ii.shape[0]  # Number of edges for relative pose constraints
    D = poses.manifold_dim  # 6 for SE3, 7 for SIM3

    ### 1: compute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = projective_transform(poses, disps, intrinsics, ii, jj, jacobian=True)
    Jz, Ji, Jj = Jz.double(), Ji.double(), Jj.double()

    ### 2: Assemble linear system ###
    Hii, Hij, Hji, Hjj, Eik, Ejk, Ck, vi, vj, wk = get_hessian_and_rhs(Jz, Ji, Jj, target, weight, coords, valid)

    # Construct larger sparse system
    kx, kk = torch.unique(ii, return_inverse=True)
    N = len(kx)  # Actual unique key frame nodes to be updated

    # only optimize keyframe poses
    M = M - fixedp
    ii = ii // rig - fixedp
    jj = jj // rig - fixedp

    # Scatter add all edges and assemble full Hessian
    # 4 x (B, N, 6, 6) -> (B, M x M, 6, 6)
    H = torch.zeros(B, M * M, D, D, device=target.device, dtype=torch.float64)
    safe_scatter_add_mat_inplace(H, Hii, ii, ii, B, M, D)
    safe_scatter_add_mat_inplace(H, Hij, ii, jj, B, M, D)
    safe_scatter_add_mat_inplace(H, Hji, jj, ii, B, M, D)
    safe_scatter_add_mat_inplace(H, Hjj, jj, jj, B, M, D)
    H = H.reshape(B, M, M, D, D)

    v = safe_scatter_add_vec(vi, ii, M) + safe_scatter_add_vec(vj, jj, M)
    v = rearrange(v, "b m d -> b m 1 d 1")

    E = safe_scatter_add_mat(Eik, ii, kk, M, N) + safe_scatter_add_mat(Ejk, jj, kk, M, N)
    E = rearrange(E, "b (m n) hw d -> b m (n hw) d 1", m=M, n=N)

    # Depth only appears if k = i, therefore the gradient is 0 for k != i
    # This reduces the number of elements to M defined by kk
    # kk basically scatters all edges ij onto i, i.e. we sum up all edges that are connected to i
    C = safe_scatter_add_vec(Ck, kk, N)
    w = safe_scatter_add_vec(wk, kk, N)
    eta = rearrange(eta, "n h w -> 1 n (h w)")

    if disps_sens is not None:
        m = disps_sens[:, kx].view(B, -1, ht*wd) > 0
        m = m.int()
        # Add alpha only for where there is a prior, else only add normal damping
        C = C + alpha * m + (1 - m) * eta + 1e-7
        w = w - m * alpha * (disps[:, kx] - disps_sens[:, kx]).view(B, -1, ht*wd)
    else:
        C = C + eta + 1e-7  # Apply damping
    C = rearrange(C, "b n hw -> b (n hw) 1 1")
    w = rearrange(w, "b n hw -> b (n hw) 1 1")

    ### 3: solve the system ###
    if structure_only:
        dz = schur_solve(H, E, C, v, w, ep=ep, lm=lm, structure_only=True)
        dz = rearrange(dz, "b (n h w) 1 1 -> b n h w", n=N, h=ht, w=wd)
        ### 4: apply retraction ###
        all_disps[:, :t1] = disp_retr(disps, dz, kx)

    else:
        dx, dz = schur_solve(H, E, C, v, w, ep=ep, lm=lm)
        dz = rearrange(dz, "b (n h w) 1 1 -> b n h w", n=N, h=ht, w=wd)
        ### 4: apply retraction ###
        # Update only un-fixed poses
        poses1, poses2 = poses[:, :fixedp], poses[:, fixedp:]
        poses2 = poses2.retr(dx)
        poses = lietorch.cat([poses1, poses2], dim=1)
        # Update global poses and disparities
        all_poses[:, :t1] = poses
        all_disps[:, :t1] = disp_retr(disps, dz, kx)
