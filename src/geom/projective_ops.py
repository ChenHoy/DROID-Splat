import torch
from lietorch import SE3, Sim3

MIN_DEPTH = 0.001


def extract_intrinsics(intrinsics):
    """
    Args:
        intrinsics:                     (Tensor), fx, fy, cx, cy
                                        [..., N, 4]
    """
    return intrinsics[..., None, None, :].unbind(dim=-1)  # fx, fy, cx, cy with shape [..., N, 1, 1]


def coords_grid(ht, wd, device="cuda"):
    y, x = torch.meshgrid(
        torch.arange(ht).to(device).float(),
        torch.arange(wd).to(device).float(),
        indexing="ij",
    )

    return torch.stack([x, y], dim=-1)  # [ht, wd, 2]


def iproj(disps, intrinsics, jacobian=False):
    """pinhole camera inverse projection"""
    ht, wd = disps.shape[-2:]
    device = disps.device
    fx, fy, cx, cy = extract_intrinsics(intrinsics)  # [..., N, 1, 1]

    y, x = torch.meshgrid(
        torch.arange(ht).to(device).float(),
        torch.arange(wd).to(device).float(),
        indexing="ij",
    )
    i = torch.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts = torch.stack([X, Y, i, disps], dim=-1)

    J = None
    if jacobian:
        JX = torch.zeros_like(X)[..., None]
        JY = torch.zeros_like(Y)[..., None]
        Ji = torch.zeros_like(i)[..., None]
        Jd = torch.zeros_like(disps)[..., None]
        Jd[..., -1] = 1.0
        J = torch.cat([JX, JY, Ji, Jd], dim=-1)

    return pts, J


def actp(Gij, X0, jacobian=False):
    """action on point cloud"""
    # X0: [batch, N, h, w, 4], 4: x, y, 1, d, Gij: SE3, [batch, N, 7]
    X1 = Gij[:, :, None, None] * X0

    Ja = None
    if jacobian:
        X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        B, N, H, W = d.shape

        if isinstance(Gij, SE3):
            Ja = torch.stack(
                [
                    d,
                    o,
                    o,
                    o,
                    Z,
                    -Y,
                    o,
                    d,
                    o,
                    -Z,
                    o,
                    X,
                    o,
                    o,
                    d,
                    Y,
                    -X,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                ],
                dim=-1,
            ).view(B, N, H, W, 4, 6)

        elif isinstance(Gij, Sim3):
            Ja = torch.stack(
                [
                    d,
                    o,
                    o,
                    o,
                    Z,
                    -Y,
                    X,
                    o,
                    d,
                    o,
                    -Z,
                    o,
                    X,
                    Y,
                    o,
                    o,
                    d,
                    Y,
                    -X,
                    o,
                    Z,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                ],
                dim=-1,
            ).view(B, N, H, W, 4, 7)
        else:
            raise TypeError(type(Gij))

    return X1, Ja


def proj(Xs, intrinsics, jacobian=False, return_depth=False):
    """pinhole camera projection"""
    fx, fy, cx, cy = extract_intrinsics(intrinsics)  # [..., N, 1, 1]
    X, Y, Z, D = Xs.unbind(dim=-1)  # for each [batch, N, h, w]

    Z = torch.where(Z < 0.5 * MIN_DEPTH, torch.ones_like(Z), Z)
    x = fx * (X / Z) + cx
    y = fy * (Y / Z) + cy

    if return_depth:
        coords = torch.stack([x, y, D / Z], dim=-1)
    else:
        coords = torch.stack([x, y], dim=-1)

    proj_jac = None
    if jacobian:
        B, N, H, W = Z.shape
        o = torch.zeros_like(Z)
        proj_jac = torch.stack(
            [
                fx / Z,
                o,
                -(fx / Z) * (X / Z),
                o,
                o,
                fy / Z,
                -(fy / Z) * (Y / Z),
                o,
            ],
            dim=-1,
        ).view(B, N, H, W, 2, 4)

    return coords, proj_jac


def projective_transform(
    poses: SE3,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    jacobian: bool = False,
    return_depth: bool = False,
    use_double: bool = False,
):
    """map points from ii -> jj"""

    # inverse project (pinhole)
    # depths: [batch, num, h, w]; ii: [N, ]; jj: [N, ]; intrinsics: [batch, num, 4]
    X0, Jz = iproj(depths[:, ii], intrinsics[:, ii], jacobian=jacobian)
    # X0: [batch, N, h, w, 4], 4: x, y, 1, d; Jz: [batch, N, h, w, 4], 4: 0, 0, 0, 1

    # poses: SE3, [batch, num, 7]; Gij: SE3, [batch, N, 7]
    Gij = poses[:, jj] * poses[:, ii].inv()
    # FIXME what is it with the -0.1?!
    Gij.data[:, ii == jj] = torch.tensor([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=Gij.device)
    X1, Ja = actp(Gij, X0, jacobian=jacobian)  # X1: [batch, N, h, w, 4], Ja: [batch, N, h, w, 4, 6]

    # project (pinhole), x1: [batch, N, h, w, 2/3], 2: x, y, 3: x, y, z, Jp: [batch, N, h, w, 2, 4]
    x1, Jp = proj(X1, intrinsics[:, jj], jacobian=jacobian, return_depth=return_depth)

    # exclude points too close to camera
    # NOTE chen: for some reason this was indexed with 2 in previous code
    # however, it is pretty clear, that depth/Z is at index 3
    valid = ((X1[..., 3] > MIN_DEPTH) & (X0[..., 3] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(dim=-1)  # [batch, N, h, w, 1]

    if jacobian:
        # Ji transforms according to dual adjoint
        # NOTE chen: using double here saves us sometimes from inf/nan
        if use_double:
            Jj = torch.matmul(Jp.double(), Ja.double()).float()  # [batch, N, h, w, 2, 6]
            Ji = -Gij[:, :, None, None, None].adjT(Jj)
            Jz = Gij[:, :, None, None] * Jz
            Jz = torch.matmul(Jp.double(), Jz.double().unsqueeze(dim=-1)).float()
        else:
            Jj = torch.matmul(Jp, Ja).float()  # [batch, N, h, w, 2, 6]
            Ji = -Gij[:, :, None, None, None].adjT(Jj)
            Jz = Gij[:, :, None, None] * Jz
            Jz = torch.matmul(Jp, Jz.unsqueeze(dim=-1)).float()

        return x1, valid, (Ji, Jj, Jz)

    return x1, valid


def induced_flow(poses, disps, intrinsics, ii, jj):
    """optical flow induced by camera motion"""

    ht, wd = disps.shape[2:]
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float(),
    )

    coords0 = torch.stack([x, y], dim=-1)
    coords1, valid = projective_transform(poses, disps, intrinsics, ii, jj, False)

    return coords1[..., :2] - coords0, valid


###
# Non-pinhole camera models from Annika
###


def general_projective_transform(poses, depths, intrinsics, ii, jj, jacobian=False, return_depth=False, model_id=0):
    if model_id == 0:  # pinhole
        return projective_transform(poses, depths, intrinsics, ii, jj, jacobian, return_depth)

    elif model_id == 1:  # mei
        return projective_transform_mei(poses, depths, intrinsics, ii, jj, jacobian, return_depth)
    else:
        raise Exception("Camera model not implemented.")


def projective_transform_mei(poses, depths, intr, ii, jj, jacobian=False, return_depth=False):
    """map points from ii->jj"""
    if torch.sum(torch.isnan(depths)) > 0:
        raise Exception("nan values in depth")

    # inverse project
    X0, _, _ = iproj_mei(depths[:, ii], intr[:, ii], jacobian=jacobian)

    # transform
    Gij = poses[:, jj] * poses[:, ii].inv()

    Gij.data[:, ii == jj] = torch.as_tensor([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda")
    X1, _ = actp(Gij, X0, jacobian=jacobian)

    # project
    x1, _, _ = proj_mei(X1, intr[:, jj], jacobian=jacobian, return_depth=return_depth)

    # exclude points too close to camera
    valid = ((X1[..., 2] > MIN_DEPTH) & (X0[..., 2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    if jacobian:
        raise Exception("Jacobian for mei model currently not supported.")

    return x1, valid


def iproj_mei(disps, intr, jacobian=False):
    """mei camera inverse projection"""
    ht, wd = disps.shape[2:]
    fx, fy, cx, cy, xi = extract_intrinsics(intr)

    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float(),
    )

    rhat = ((x - cx) / fx) ** 2 + ((y - cy) / fy) ** 2
    factor = (xi + torch.sqrt(1 + (1 - xi**2) * rhat)) / (1 + rhat)

    X = (x - cx) * factor / fx
    Y = (y - cy) * factor / fy
    Z = factor - xi

    pts = torch.stack([X / Z, Y / Z, Z / Z, disps], dim=-1)

    if jacobian:
        raise Exception("Jacobian for mei model currently not supported.")

    return pts, None, None


def proj_mei(Xs, intr, jacobian=False, return_depth=False):
    """mei camera projection"""
    fx, fy, cx, cy, xi = extract_intrinsics(intr)
    X, Y, Z, D = Xs.unbind(dim=-1)

    Z = torch.where(Z < 0.5 * MIN_DEPTH, torch.ones_like(Z), Z)

    d = 1.0 / Z
    r = torch.sqrt(X**2 + Y**2 + Z**2)
    factor = 1.0 / (Z + xi * r)

    x = fx * (X * factor) + cx
    y = fy * (Y * factor) + cy

    if return_depth:
        coords = torch.stack([x, y, D * d], dim=-1)
    else:
        coords = torch.stack([x, y], dim=-1)

    if jacobian:
        raise Exception("Jacobian for mei model currently not supported.")

    return coords, None, None
