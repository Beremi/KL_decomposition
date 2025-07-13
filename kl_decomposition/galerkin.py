"""Assembly of 1-D Galerkin blocks."""

from __future__ import annotations
import opt_einsum as oe
from typing import Optional

import numpy as np

from .orthopoly import (
    gauss_legendre_rule,
    shifted_legendre,
    leg_vals,
)

__all__ = [
    "assemble_block",
    "assemble_duffy",
    "assemble_gauss2d",
    "assemble_rectangle",
]


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

    phi = np.array([shifted_legendre(i, x, a, b_) for i in range(n)])
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
    Duffy-split assembly on [0,1]×[0,1] with polynomial stretching.
    Uses the parity relation P̃_k(1-t)=(-1)^k P̃_k(t) to
    * evaluate each Legendre basis only once, and
    * perform only one tensor contraction per parity block
      (the second contribution is identical, so we just double it).
    """
    if gy is None:
        gy = gx

    # ---------------------------------------------------------------- quadrature
    xi, wx = gauss_legendre_rule(0.0, 1.0, quad)           # (N,), (N,)
    N = xi.size

    # -------------------------------------------------------------- stretch maps
    u = xi ** gx                                           # (N,)
    v = xi ** gy                                           # (N,)
    U, V = u[:, None], v[None, :]                          # (N,1), (1,N)

    # -------------------------------------------------------------- weight term
    J = gx * gy * (xi ** (gx - 1))[:, None] * (xi ** (gy - 1))[None, :]
    Khat = J * U * np.exp(-f * (U * V) ** 2)             # (N,N)
    weight = np.outer(wx, wx) * Khat                       # (N,N)

    # -------------------------------------------------------------- mapped grid
    y1 = (1.0 - V) * U                                     # (N,N)

    # -------------------------------------------------------------- basis values
    phi_x = leg_vals(degree, u)[:, :, None]                # (deg,N,1)
    phi_y = leg_vals(degree, y1.ravel()).reshape(degree, N, N)  # (deg,N,N)

    # -------------------------------------------------------------- assemble
    A = np.zeros((degree, degree))
    even = np.arange(0, degree, 2)
    odd = np.arange(1, degree, 2)

    if even.size:
        block = oe.contract("mn,imn,jmn->ij",
                            weight, phi_x[even], phi_y[even], optimize=True)
        A[np.ix_(even, even)] = 2.0 * block                # region-1 + region-2

    if odd.size:
        block = oe.contract("mn,imn,jmn->ij",
                            weight, phi_x[odd], phi_y[odd], optimize=True)
        A[np.ix_(odd, odd)] = 2.0 * block                  # region-1 + region-2

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
