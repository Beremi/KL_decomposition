"""Assembly of 1-D Galerkin blocks."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import legval

from .kernel_fit import gauss_legendre_rule

__all__ = ["assemble_block"]


def _legendre_phi(i: int, x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Evaluate shifted, L2-orthonormal Legendre polynomial."""
    u = 2.0 * (x - a) / (b - a) - 1.0
    return np.sqrt((2 * i + 1) / (b - a)) * legval(u, [0.0] * i + [1.0])


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
