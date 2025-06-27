"""Assembly of 1-D Galerkin blocks."""

from __future__ import annotations
from typing import Optional

import numpy as np

from .orthopoly import (
    gauss_legendre_rule,
    shifted_legendre,
)

__all__ = [
    "assemble_block",
    "assemble_duffy",
    "assemble_gauss2d",
    "assemble_rectangle",
    "convergence_vs_ref",
]


def _legendre_phi(i: int, x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Evaluate shifted, L2-orthonormal Legendre polynomial."""
    return shifted_legendre(i, x, a, b)


# def leg_vals(n_max: int, x: np.ndarray) -> np.ndarray:
#     """Values of orthonormal Legendre polynomials on ``[0, 1]``."""
#     return legendre_table(n_max, x)
def leg_vals(n: int, x: np.ndarray) -> np.ndarray:
    """
    Evaluate the first *n* orthonormal (shifted) Legendre polynomials
    on the interval [0, 1].

    Parameters
    ----------
    n : int
        Number of polynomials to return (π̃₀ … π̃_{n-1}).
    x : np.ndarray
        Evaluation points in [0, 1].  May be any shape.

    Returns
    -------
    vals : np.ndarray
        Array of shape (n, *x.shape) with
        vals[k] == π̃_k(x) for k = 0 … n-1.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    x = np.asarray(x, dtype=float)
    vals = np.empty((n,) + x.shape, dtype=float)

    # π̃₀(t) = 1
    vals[0] = 1.0
    if n == 1:
        return vals

    # π̃₁(t) = √3·(2t − 1)
    vals[1] = np.sqrt(3.0) * (2.0 * x - 1.0)

    # helper: α_k  (k ≥ 1)  ─ recurrence coefficient
    def alpha(k: int) -> float:
        return k / (2.0 * np.sqrt(4.0 * k * k - 1.0))

    # three-term recurrence (k starts at 1 → produces π̃_{k+1})
    for k in range(1, n - 1):
        a_k = alpha(k)
        a_k1 = alpha(k + 1)
        vals[k + 1] = ((x - 0.5) / a_k1) * vals[k] - (a_k / a_k1) * vals[k - 1]

    return vals


def assemble_block(interval: tuple[float, float], coeff_b: float, n: int, *, quad_order: int = 40) -> np.ndarray:
    """Assemble a single Galerkin block.

    Parameters
    ----------
    interval : tuple of float
        Integration limits ``(a, b)``.
    coeff_b : float
        Coefficient of ``(x - y)^2`` in the Gaussian kernel ``exp(-b (x - y)^2)``.
    n : int
        Number of Legendre polynomials ``\\phi_i``.
    quad_order : int, optional
        Order of the Gauss--Legendre quadrature.

    Returns
    -------
    ndarray
        ``n \times n`` Galerkin matrix ``A`` with
        ``A[i, j] = \\int_a^b \\int_a^b e^{-b (x - y)^2} \\phi_i(y) \\phi_j(x) dy dx``.
    """
    a, b_ = interval
    x, w = gauss_legendre_rule(a, b_, quad_order)

    phi = np.array([_legendre_phi(i, x, a, b_) for i in range(n)])
    weighted_phi = phi * w

    K = np.exp(-coeff_b * (x[:, None] - x[None, :]) ** 2)

    A = np.zeros((n, n))
    even = np.arange(0, n, 2)
    odd = np.arange(1, n, 2)

    if even.size:
        wphi_even = weighted_phi[even]
        A[np.ix_(even, even)] = wphi_even @ K @ wphi_even.T

    if odd.size:
        wphi_odd = weighted_phi[odd]
        A[np.ix_(odd, odd)] = wphi_odd @ K @ wphi_odd.T

    return A


def assemble_duffy_slow(f: float, degree: int, quad: int,
                        gx: float = 4.0, gy: float | None = None) -> np.ndarray:
    """Duffy-split assembly on ``[0, 1]`` with polynomial stretching."""
    if gy is None:
        gy = gx

    xi, wx = gauss_legendre_rule(0.0, 1.0, quad)
    X, Y = np.meshgrid(xi, xi, indexing="ij")
    W = np.outer(wx, wx)

    u = X ** gx
    v = Y ** gy
    J = gx * gy * X ** (gx - 1) * Y ** (gy - 1)
    Khat = J * u * np.exp(-f * (u * v) ** 2)

    x1, y1 = u, (1.0 - v) * u
    x2, y2 = 1.0 - u, (v - 1.0) * u + 1.0

    phix1 = leg_vals(degree, x1.ravel()).reshape(degree, *x1.shape)
    phiy1 = leg_vals(degree, y1.ravel()).reshape(degree, *y1.shape)
    phix2 = leg_vals(degree, x2.ravel()).reshape(degree, *x2.shape)
    phiy2 = leg_vals(degree, y2.ravel()).reshape(degree, *y2.shape)

    weight = W * Khat

    even = np.arange(0, degree, 2)
    odd = np.arange(1, degree, 2)

    A = np.zeros((degree, degree))

    if even.size:
        A_even = (
            np.einsum("mn,imn,jmn->ij", weight, phix1[even], phiy1[even])
            + np.einsum("mn,imn,jmn->ij", weight, phix2[even], phiy2[even])
        )
        A[np.ix_(even, even)] = A_even

    if odd.size:
        A_odd = (
            np.einsum("mn,imn,jmn->ij", weight, phix1[odd], phiy1[odd])
            + np.einsum("mn,imn,jmn->ij", weight, phix2[odd], phiy2[odd])
        )
        A[np.ix_(odd, odd)] = A_odd

    return A


def assemble_duffy(
    f: float,
    degree: int,
    quad: int,
    gx: float = 4.0,
    gy: Optional[float] = None,
) -> np.ndarray:
    """
    Duffy-split assembly on [0,1] with polynomial stretching.
    Same output as `assemble_duffy`, but avoids redundant evaluations
    of Legendre bases on identical coordinates.
    """
    if gy is None:
        gy = gx

    # -- quadrature rule -----------------------------------------------------
    xi, wx = gauss_legendre_rule(0.0, 1.0, quad)           # (N,), (N,)
    N = xi.size

    # -- 1-D stretched coordinates ------------------------------------------
    u = xi ** gx                                            # shape (N,)
    v = xi ** gy                                            # shape (N,)

    # promote to 2-D with broadcasting (no copies)
    U = u[:, None]                                          # (N,1)
    V = v[None, :]                                          # (1,N)

    # -- Jacobian, kernel and weights ---------------------------------------
    J = gx * gy * (xi ** (gx - 1))[:, None] * (xi ** (gy - 1))[None, :]
    Khat = J * U * np.exp(-f * (U * V) ** 2)               # (N,N)
    W = np.outer(wx, wx)                                # (N,N)
    weight = W * Khat                                       # (N,N)

    # -- mapped points -------------------------------------------------------
    #  block 1
    # x1 = U                         # (N,1) … constant in 'n'
    y1 = (1.0 - V) * U             # (N,N)
    #  block 2
    # x2 = 1.0 - U                   # (N,1)
    y2 = (V - 1.0) * U + 1.0       # (N,N)

    # -- Legendre basis values ----------------------------------------------
    #  x-direction: only N distinct points → evaluate once
    phi_x1 = leg_vals(degree, u)           # (deg, N)
    phi_x2 = leg_vals(degree, 1.0 - u)     # (deg, N)

    #  y-direction: full grid still needed
    phi_y1 = leg_vals(degree, y1.ravel()).reshape(degree, N, N)
    phi_y2 = leg_vals(degree, y2.ravel()).reshape(degree, N, N)

    # add a singleton axis so einsum can broadcast along 'n'
    phi_x1 = phi_x1[:, :, None]            # (deg, N, 1)
    phi_x2 = phi_x2[:, :, None]            # (deg, N, 1)

    # -- assemble ------------------------------------------------------------
    A = np.zeros((degree, degree))
    even = np.arange(0, degree, 2)
    odd = np.arange(1, degree, 2)

    if even.size:
        A[np.ix_(even, even)] = (
            np.einsum("mn,imn,jmn->ij", weight, phi_x1[even], phi_y1[even])
            + np.einsum("mn,imn,jmn->ij", weight, phi_x2[even], phi_y2[even])
        )

    if odd.size:
        A[np.ix_(odd, odd)] = (
            np.einsum("mn,imn,jmn->ij", weight, phi_x1[odd], phi_y1[odd])
            + np.einsum("mn,imn,jmn->ij", weight, phi_x2[odd], phi_y2[odd])
        )

    return A


def assemble_gauss2d(f: float, degree: int, quad: int) -> np.ndarray:
    """Direct tensor-product Gauss--Legendre on ``[0, 1]``."""
    x, wx = gauss_legendre_rule(0.0, 1.0, quad)
    phi = leg_vals(degree, x)
    K = np.exp(-f * (x[:, None] - x[None, :]) ** 2)

    even = np.arange(0, degree, 2)
    odd = np.arange(1, degree, 2)

    A = np.zeros((degree, degree))

    if even.size:
        wphi_even = phi[even] * wx
        A[np.ix_(even, even)] = wphi_even @ K @ wphi_even.T

    if odd.size:
        wphi_odd = phi[odd] * wx
        A[np.ix_(odd, odd)] = wphi_odd @ K @ wphi_odd.T

    return A


def assemble_rectangle(f: float, degree: int, m: int) -> np.ndarray:
    """Midpoint rectangle rule on ``[0, 1]``²."""
    x = (np.arange(m) + 0.5) / m
    dx = 1.0 / m
    phi = leg_vals(degree, x)
    K = np.exp(-f * (x[:, None] - x[None, :]) ** 2)

    even = np.arange(0, degree, 2)
    odd = np.arange(1, degree, 2)

    A = np.zeros((degree, degree))

    if even.size:
        wphi_even = phi[even] * dx
        A[np.ix_(even, even)] = wphi_even @ K @ wphi_even.T

    if odd.size:
        wphi_odd = phi[odd] * dx
        A[np.ix_(odd, odd)] = wphi_odd @ K @ wphi_odd.T

    return A


def convergence_vs_ref(
    f: float, degree: int, g: float, quad_list: list[int], quad_ref: int
) -> tuple[list[int], list[float]]:
    """Convergence study helper for :func:`assemble_duffy`."""
    A_ref = assemble_gauss2d(f, degree, quad_ref)
    errs = [np.linalg.norm(assemble_duffy(f, degree, q, g) - A_ref) for q in quad_list]
    return quad_list, errs
