import ipdb
from typing import Optional, List

import numpy as np
import numba as nb
import torch
from einops import parse_shape, rearrange
from scipy.spatial.transform import Rotation as R
from ..utils import TrajectorySegment

import dpvo_backends
import pypose as pp


def make_pypose_Sim3(rot, t, s):
    q = R.from_matrix(rot).as_quat()
    data = np.concatenate([t, q, np.array(s).reshape((1,))])
    return pp.Sim3(data)


def SE3_to_Sim3(x: pp.SE3):
    out = torch.cat((x.data, torch.ones_like(x.data[..., :1])), dim=-1)
    return pp.Sim3(out)


@nb.njit(cache=True)
def _format(es):
    return np.asarray(es, dtype=np.int64).reshape((-1, 2))[1:]


@nb.njit(cache=True)
def reduce_edges(flow_mag, ii, jj, max_num_edges, nms):
    es = [(-1, -1)]

    if ii.size == 0:
        return _format(es)

    Ni, Nj = (ii.max() + 1), (jj.max() + 1)
    ignore_lookup = np.zeros((Ni, Nj), dtype=nb.bool_)

    idxs = np.argsort(flow_mag)
    for idx in idxs:  # edge index

        if len(es) > max_num_edges:
            break

        i = ii[idx]
        j = jj[idx]
        mag = flow_mag[idx]

        if (j - i) < 30:
            continue

        if mag >= 1000:  # i.e., inf
            continue

        if ignore_lookup[i, j]:
            continue

        es.append((i, j))

        for di in range(-nms, nms + 1):
            i1 = i + di

            if 0 <= i1 < Ni:
                ignore_lookup[i1, j] = True

    return _format(es)


@nb.njit(cache=True)
def umeyama_alignment(x: np.ndarray, y: np.ndarray):
    """
    The following function was copied from:
    https://github.com/MichaelGrupp/evo/blob/3067541b350528fe46375423e5bc3a7c42c06c63/evo/core/geometry.py#L35

    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.sum(axis=1) / n
    mean_y = y.sum(axis=1) / n

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        print("Warning. Degenerate covariance matrix in SVD for Umeyama")
        return None, None, None  # Degenerate covariance rank, Umeyama alignment is not possible

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s))
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


@nb.njit(cache=True)
def ransac_umeyama(src_points, dst_points, iterations=1, threshold=0.1, optim_inliers: int = 100):
    best_inliers, best_distances = 0, None
    best_R, best_t, best_s = None, None, None

    for _ in range(iterations):
        # Randomly select three points
        indices = np.random.choice(src_points.shape[0], 3, replace=False)
        src_sample = src_points[indices]
        dst_sample = dst_points[indices]

        # Estimate transformation
        R, t, s = umeyama_alignment(src_sample.T, dst_sample.T)
        # Just skip over degenerate cases
        if t is None:
            continue

        # Apply transformation
        transformed = (src_points @ (R * s).T) + t

        # Count inliers (not ideal because depends on scene scale)
        # FIXME how to make this scale-invariant?
        distances = np.sum((transformed - dst_points) ** 2, axis=1) ** 0.5
        inlier_mask = distances < threshold
        inliers = np.sum(inlier_mask)

        # Update best transformation
        if inliers > best_inliers:
            best_inliers = inliers
            best_R, best_t, best_s = umeyama_alignment(src_points[inlier_mask].T, dst_points[inlier_mask].T)
            best_distances = distances

        # If we have > 100 inliers, the model is likely pretty good
        # TODO chen: I dont like this, as this depends on the number of input points
        if inliers > optim_inliers:
            break

    return best_R, best_t, best_s, best_inliers, best_distances


def batch_jacobian(func, x):
    def _func_sum(*x):
        return func(*x).sum(dim=0)

    _, b, c = torch.autograd.functional.jacobian(_func_sum, x, vectorize=True)
    return rearrange(torch.stack((b, c)), "N O B I -> N B O I", N=2)


def _residual(C, Gi, Gj):
    assert parse_shape(C, "N _") == parse_shape(Gi, "N _") == parse_shape(Gj, "N _")
    out = C @ pp.Exp(Gi) @ pp.Exp(Gj).Inv()
    return out.Log().tensor()


def residual(Ginv, input_poses, dSloop, ii, jj, jacobian=False):
    """
    Given a set of poses and loop closures, compute the residual of the loop closure constraints.
    """
    device = Ginv.device
    assert parse_shape(input_poses, "_ d") == dict(d=7)
    # NOTE Ginv = SE3_to_Sim3(input_poses).Inv().Log()
    pred_inv_poses = SE3_to_Sim3(input_poses).Inv()

    # free variables
    n, _ = pred_inv_poses.shape
    kk = torch.arange(1, n, device=device)
    ll = kk - 1

    # constants
    # NOTE these are just the rel. poses for each i to i-1
    Ti = pred_inv_poses[kk]
    Tj = pred_inv_poses[ll]
    dSij = Tj @ Ti.Inv()

    constants = torch.cat((dSij, dSloop), dim=0)
    iii = torch.cat((kk, ii))
    jjj = torch.cat((ll, jj))
    resid = _residual(constants, Ginv[iii], Ginv[jjj])

    if not jacobian:
        return resid

    J_Ginv_i, J_Ginv_j = batch_jacobian(_residual, (constants, Ginv[iii], Ginv[jjj]))
    return resid, (J_Ginv_i, J_Ginv_j, iii, jjj)


# NOTE DPVO actually runs this in separate Process Pool asynchronously
# Since we rarely run a loop closure and we let the other threads wait for this to finish
# or system stability, we just run this synchronously and eat the small amortized cost
def run_PGO(
    pred_poses: torch.Tensor, loop_poses: torch.Tensor, loop_ii: torch.Tensor, loop_jj: torch.Tensor, **kwargs
):
    """Run a PoseGraph Optimiztion on a window of poses given certain loop edges.
    This takes n pred_poses and optimizes all poses in the window [0, n].

    args:
    ---
    pred_poses: torch.Tensor [cur_t, 7] Su(3) abs. poses
    loop_poses: torch.Tensor [num_prev_loops + 1, 7] SIM(3) rel. poses
        These are all previous loop edge relative poses and the current loop edge, where we estimated the rel. pose with 3D registration
    loop_ii: torch.Tensor [num_prev_loops] int64
        The i indices of the loop edges
    loop_jj: torch.Tensor [num_prev_loops] int64
        The j indices of the loop edges
    """
    final_est = perform_updates(pred_poses, loop_poses, loop_ii, loop_jj, **kwargs)

    # NOTE we make sure that loop_ii will be up to the most recent frame, as we usually close loops as soon as they appear
    # NOTE chen: DPVO actually has a + 1 here, but they index and progress in a different way than us
    safe_i = loop_ii.max().item()  # Only optimized until the most recent loop closure

    aa = SE3_to_Sim3(pred_poses.cpu())
    # Get the relative pose, but only for the newest loop node (i)
    rel_delta = aa[[safe_i]] * final_est[[safe_i]].Inv()

    # Update all poses with this optimized relative pose, so the loop get closed
    final_est = rel_delta * final_est
    return final_est[:safe_i]


def perform_updates(
    input_poses: torch.Tensor,
    dSloop: torch.Tensor,
    ii_loop: torch.Tensor,
    jj_loop: torch.Tensor,
    iters: int = 30,
    ep: float = 0.0,
    lmbda: float = 1e-6,
    fix_opt_window: bool = False,
    segments_to_fix: Optional[List[TrajectorySegment]] = None,
    segment_padding: int = 2,
):
    """Run Levenberg-Marquardt optimization on a PGO problem. We additionally allow to fix certain segments of the trajectory, i.e. we
    augment the Hessian matrix for these poses, so that they are not updated.
    NOTE freezing the gradients and residuals manually in these areas did not work!

    Since PGO is a relative pose edge optimization, iii and jjj until the fixed loop edges are simply intertwined,
    e.g. [1, 2, ..., last_ii] <-> [0, 1, ..., last_ii-1], this means these are just the relative poses fixed between consecutive frames
    Given a segment [t0, t1], which we want to fix, we now just need to find the position in iii and jjj including the border indices [t0, t1] (maybe +1)

    NOTE Our SegmentManager gives us sorted segments in ascending order, so we can simply iterate over them and fix the poses in the given range.
    The segments will always fragment the map/trajectory into non-loops and loops and maintains an ongoing open window as last segment.

    NOTE iii: [arange(1, last_ii), ii_loop], jjj: [arange(0, last_jj), jj_loop] with len(iii) = len(jjj) > cur_t
    J_Ginv_i, J_Ginv_j are the Jacobians of the residuals w.r.t. the poses -> len(iii) x 7 x 7 and len(iii) x 7 for residuals

    NOTE DPVO uses a very conservative 1e-6 lambda initial step size (This is for good reason as I found out the hard way)
    NOTE DPVO does not use any numerical damping that is applied on the system diagonal for some reason
    """

    input_poses = input_poses.clone()

    if fix_opt_window:
        freen = torch.cat((ii_loop, jj_loop)).max().item() + 1
    else:
        freen = -1

    Ginv = SE3_to_Sim3(input_poses).Inv().Log()  # Use Lie algebra for updates

    residual_history = []

    for itr in range(iters):

        resid, (J_Ginv_i, J_Ginv_j, iii, jjj) = residual(Ginv, input_poses, dSloop, ii_loop, jj_loop, jacobian=True)

        # We fix segments of previous loop closures to keep the map intact
        # i.e. we only want to drift-correct the current and previous segment to align the new loop edge (i, j)
        # Go over all given segments and create a fixed interval [t0, t1] out of them
        # NOTE make sure the segments are connected, or else we will have a bigger super interval with poses outside the segments fixed
        if segments_to_fix is not None:
            t0, t1 = 999, 0
            for segment in segments_to_fix:
                t0 = min(t0, segment[0])
                t1 = max(t1, segment[1])  # TODO should it be -1 or +1 here?
        else:
            t0, t1 = -1, -1

        residual_history.append(resid.square().mean().item())

        (delta_pose,) = dpvo_backends.solve_system_fixed(J_Ginv_i, J_Ginv_j, iii, jjj, resid, ep, lmbda, freen, t0, t1)
        assert Ginv.shape == delta_pose.shape

        Ginv_tmp = Ginv + delta_pose  # Update in exponential space

        new_resid = residual(Ginv_tmp, input_poses, dSloop, ii_loop, jj_loop)
        # Old school Levenberg-Marquart update rule for step size
        if new_resid.square().mean() < residual_history[-1]:
            Ginv = Ginv_tmp
            lmbda /= 2
        else:
            lmbda *= 2

        if (residual_history[-1] < 1e-5) and (itr >= 4) and ((residual_history[-5] / residual_history[-1]) < 1.5):
            break

    return pp.Exp(Ginv).Inv()  # Map back to group in c2w
