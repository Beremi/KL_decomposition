"""Assembly of 1-D Galerkin blocks."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import Legendre, legval

from .kernel_fit import gauss_legendre_rule

__all__ = [
    "assemble_block",
    "assemble_duffy",
    "assemble_gauss2d",
    "assemble_rectangle",
    "convergence_vs_ref",
]


def _legendre_phi(i: int, x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Evaluate shifted, L2-orthonormal Legendre polynomial."""
    u = 2.0 * (x - a) / (b - a) - 1.0
    return np.sqrt((2 * i + 1) / (b - a)) * legval(u, [0.0] * i + [1.0])


def leg_vals(n_max: int, x: np.ndarray) -> np.ndarray:
    """Values of orthonormal Legendre polynomials on ``[0, 1]``."""
    t = 2.0 * x - 1.0
    vals = np.empty((n_max, x.size))
    pref = 1.0 / np.sqrt(2.0)
    for n in range(n_max):
        vals[n] = np.sqrt(2 * n + 1) * pref * Legendre.basis(n)(t)
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
        Number of Legendre polynomials ``\phi_i``.
    quad_order : int, optional
        Order of the Gauss--Legendre quadrature.

    Returns
    -------
    ndarray
        ``n \times n`` Galerkin matrix ``A`` with
        ``A[i, j] = \int_a^b \int_a^b e^{-b (x - y)^2} \phi_i(y) \phi_j(x) dy dx``.
    """
    a, b_ = interval
    x, w = gauss_legendre_rule(a, b_, quad_order)

    phi = np.array([_legendre_phi(i, x, a, b_) for i in range(n)])
    weighted_phi = phi * w

    K = np.exp(-coeff_b * (x[:, None] - x[None, :]) ** 2)
    A = weighted_phi @ K @ weighted_phi.T
    return A


def assemble_duffy(f: float, degree: int, quad: int,
                   gx: float = 4.0, gy: float | None = None) -> np.ndarray:
    """Duffy-split assembly on ``[0, 1]`` with polynomial stretching."""
    if gy is None:
        gy = gx

    xi, wx = np.polynomial.legendre.leggauss(quad)
    xi = 0.5 * (xi + 1.0)
    wx = 0.5 * wx
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
    A = (
        np.einsum("mn,imn,jmn->ij", weight, phix1, phiy1)
        + np.einsum("mn,imn,jmn->ij", weight, phix2, phiy2)
    )
    return A


def assemble_gauss2d(f: float, degree: int, quad: int) -> np.ndarray:
    """Direct tensor-product Gauss--Legendre on ``[0, 1]``."""
    x, wx = np.polynomial.legendre.leggauss(quad)
    x = 0.5 * (x + 1.0)
    wx = 0.5 * wx
    phi = leg_vals(degree, x)
    K = np.exp(-f * (x[:, None] - x[None, :]) ** 2)
    return phi @ (K * (wx[:, None] * wx[None, :])) @ phi.T


def assemble_rectangle(f: float, degree: int, m: int) -> np.ndarray:
    """Midpoint rectangle rule on ``[0, 1]``²."""
    x = (np.arange(m) + 0.5) / m
    dx = 1.0 / m
    phi = leg_vals(degree, x)
    K = np.exp(-f * (x[:, None] - x[None, :]) ** 2)
    return phi @ (K * dx * dx) @ phi.T


def convergence_vs_ref(
    f: float, degree: int, g: float, quad_list: list[int], quad_ref: int
) -> tuple[list[int], list[float]]:
    """Convergence study helper for :func:`assemble_duffy`."""
    A_ref = assemble_gauss2d(f, degree, quad_ref)
    errs = [np.linalg.norm(assemble_duffy(f, degree, q, g) - A_ref) for q in quad_list]
    return quad_list, errs
