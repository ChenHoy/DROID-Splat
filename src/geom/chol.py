import torch
from termcolor import colored
import ipdb
from einops import einsum, rearrange

"""
Dense Cholesky solver for block structured matrices and symmetric pos. definite matrices. 

We adapted Zachary Teed's code using einops, double precision and smarter indexing. 
The precision is crucial when solving large systems, but results in a bigger memory footprint!
The indexing turned out to be very important because the CUDA kernel uses a Sparse Cholesky solver, while 
torch relies on a dense Cholesky decomposition. For very sparse systems, the dense solver will be numerically unstable :/

NOTE very large problems require float64 precision to be numerically stable (else we overflow -> inf/-inf), the reason for this is simply the large number of 
variables due to computing dense and pixel wise. We dont observe this problem in the CUDA kernels, since computations can be run elementwise in parallel and not vectorized 
over very large system matrices. 
However, the main system runs in float32 in order to achieve a medium memory footprint (< 24GB) and run on high resolution images.
Since we have a multi-threaded system, we store tensors in shared_memory in order to have quick read/write access. Therefore all tensors must have 
the same datatype like the rest of the system! For this reason we provide the option to run the system in a given precision (Input is always float32).
In case we detect an overflow, we simply clip the Jacobians / Gradients, which effectively limits the step size of the optimizer.
"""


def is_positive_definite(matrix: torch.Tensor, eps=2e-5) -> bool:
    """Check if a Matrix is positive definite by checking symmetry looking at the eigenvalues.

    NOTE This is in reality less efficient then performing a test Cholesky decomposition and checking if it fails
    NOTE A rigorous test would also include the submatrix determinants
    """
    return bool((abs(matrix - matrix.mT) < eps).all() and (torch.linalg.eigvals(matrix).real >= 0).all())


class LUSolver(torch.autograd.Function):
    # TODO is this really correct? we apply the same derivative like in CholeskySolver
    @staticmethod
    def forward(ctx, H, b):
        # don't crash training if decomp fails
        try:
            lu, pivots = torch.linalg.lu_factor(H)
            xs = torch.linalg.lu_solve(lu, pivots, b)
            ctx.save_for_backward(lu, pivots, xs)
            ctx.failed = False
            success = True
        except Exception as e:
            print(colored("Warning. LU Decomposition failed!", "red"))
            ctx.failed = True
            success = False
            xs = torch.zeros_like(b)

        return xs, success

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        lu, pivots, xs = ctx.saved_tensors
        # P * H = LU with P being the pivots
        dz = torch.linalg.lu_solve(lu, pivots, grad_x)
        dH = -torch.matmul(xs, dz.transpose(-1, -2)).to(lu.dtype)

        return dH, dz


class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        # don't crash training if cholesky decomp fails
        try:
            U = torch.linalg.cholesky(H)
            xs = torch.cholesky_solve(b, U)
            ctx.save_for_backward(U, xs)
            ctx.failed = False
            success = True
        except Exception as e:
            print(colored("Warning. Cholesky Decomposition failed!", "red"))
            success = False
            ctx.failed = True
            xs = torch.zeros_like(b)

        return xs, success

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1, -2)).to(xs.dtype)

        return dH, dz


def show_matrix(A: torch.Tensor) -> None:
    import matplotlib.pyplot as plt

    plt.imshow(A[0].detach().cpu().numpy())
    plt.grid(False)
    plt.show()


def block_show(A: torch.Tensor) -> None:
    """Plot the matrix to investigate the sparsity pattern.
    The block matrix is reshaped into a proper 2D matrix before plotting.

    args:
    ---
    A [torch.Tensor]: batched matrix (B, N1, M1, P1, Q1)
    """
    import matplotlib.pyplot as plt

    b, n1, m1, p1, q1 = A.shape
    A = A.permute(0, 1, 3, 2, 4).reshape(b, n1 * p1, m1 * q1)
    plt.imshow(A[0].detach().cpu().numpy())
    plt.show()


def block_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """matrix multiply for block structured matrices of shape (b, n, m, p, q)."""
    b, n1, m1, p1, q1 = A.shape
    b, n2, m2, p2, q2 = B.shape
    A = rearrange(A, "b n1 m1 p1 q1 -> b (n1 p1) (m1 q1)")
    B = rearrange(B, "b n2 m2 p2 q2 -> b (n2 p2) (m2 q2)")
    C = torch.matmul(A, B).to(A.dtype)
    C = rearrange(C, "b (n1 p1) (m2 q2) -> b n1 m2 p1 q2", n1=n1, m2=m2, p1=p1, q2=q2)
    return C


def cholesky_block_solve(
    H: torch.Tensor, b: torch.Tensor, ep: float = 0.1, lm: float = 1e-4, use_double: bool = False
) -> torch.Tensor:
    """solve normal equations for block structure matrices of shape (n1 n2 d1 d2)"""
    if use_double:
        dtype = torch.float64
    else:
        dtype = torch.float32

    bs, n, _, d, _ = H.shape
    I = torch.eye(d, dtype=dtype).to(H.device)
    H = H.to(dtype) + (ep + lm * H.to(dtype)) * I  # Damp H

    H = rearrange(H, "b n1 n2 d1 d2 -> b (n1 d1) (n2 d2)")
    b = rearrange(b, "b n d -> b (n d) 1")

    x = CholeskySolver.apply(H, b)
    return rearrange(x, "b (n1 d1) 1 -> b n1 d1", n1=n, d1=d)


def schur_block_solve(
    H: torch.Tensor,
    E: torch.Tensor,
    C: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    ep: float = 0.1,
    lm: float = 1e-4,
    structure_only: bool = False,
    motion_only: bool = False,
    return_state: bool = False,
    use_double: bool = False,
):
    """Solve the sparse block-structured linear system Hx = v using the Schur complement,
    i.e. instead of solving a larger M*N x M*N system, we solve a smaller M x M system and N x N system,
    where the N x N system for the structure can be easily solved since C is a diagonal matrix.

    adapted from the DPVO repository.
    """
    b = H.shape[0]

    Q = 1.0 / C  # Since C is diagonal we can just divide for inversion
    EQ = E * Q[:, None]
    Et = rearrange(E, "b n mhw d 1 -> b mhw n 1 d")

    if structure_only:
        dZ = (Q * w).view(b, -1, 1, 1)
        if return_state:
            return dZ, success
        else:
            return dZ

    else:
        S = H - block_matmul(EQ, Et)
        y = v - block_matmul(EQ, w.unsqueeze(dim=2))
        dX, success = cholesky_block_solve(S, y, ep=ep, lm=lm, use_double=use_double)

        if motion_only:
            if return_state:
                return dX, success
            else:
                return dX

        dZ = Q * (w - block_matmul(Et, dX).squeeze(dim=-1))
        dX = dX.view(b, -1, 6)
        dZ = dZ.view(b, -1, 1, 1)

        if return_state:
            return dX, dZ, success
        else:
            return dX, dZ


def schur_solve(
    H: torch.Tensor,
    E: torch.Tensor,
    C: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    ep: float = 0.1,
    lm: float = 1e-4,
    structure_only: bool = False,
    motion_only: bool = False,
    solver: str = "cholesky",
    return_state: bool = False,
    use_double: bool = False,
):
    """Solve the linear system Hx = v by the Schur complement,
    i.e. instead of solving a larger M*N x M*N system, we solve a smaller M x M system and N x N system,
    where the N x N system for the structure can be easily solved since we know C^-1.

    H and the Schur complement S should be positive definite, so we can invert them using Cholesky decomposition.
    However, for some problems this is not the case! When optimizing poses, structure, scale and shift we notice
    that S will not be positive definite, but still invertible. In this case we use LU decomposition instead.
    """
    if solver == "cholesky":
        Solver = CholeskySolver
    else:
        Solver = LUSolver
    if use_double:
        dtype = torch.float64
    else:
        dtype = torch.float32

    bs = C.shape[0]
    C = C.view(bs, -1)
    Q = 1.0 / C  # Since C is diagonal we can just divide for inversion

    if structure_only:
        dZ = Q * w.view(bs, -1)
        return dZ.float()

    EQ = E * Q[:, None]
    S = H - torch.matmul(EQ, E.mT).to(dtype)
    y = v - torch.matmul(EQ, w.squeeze(dim=-1)).to(dtype)

    A = S + (ep + lm * S) * torch.eye(S.shape[1], device=S.device, dtype=dtype)  # Damping
    dX, success = Solver.apply(A, y)

    if motion_only:
        if return_state:
            return dX.squeeze(-1), success
        else:
            return dX.squeeze(-1)

    dZ = Q * (w.view(bs, -1) - torch.matmul(E.mT, dX).view(bs, -1)).to(dtype)
    if return_state:
        return dX, dZ, success
    else:
        return dX, dZ
