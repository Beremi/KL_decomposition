"""Eigen-decomposition of the 1-D exponential kernel.

This module provides helper routines for computing the analytic eigenpairs
of the kernel

    K(x, y) = exp(-alpha * |x - y|),  0 <= x, y <= L.

The eigenfunctions come in cosine (Neumann-like) and sine (Dirichlet-like)
families.  The corresponding frequencies ``omega`` are solutions of

* ``tan(omega * L) =  omega / alpha``     (cosine family)
* ``tan(omega * L) = -alpha / omega``    (sine family)

and the eigenvalues are ``lambda = 2 * alpha / (alpha**2 + omega**2)``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from typing import Callable

__all__ = ["exp_kernel_eigen"]


def _find_roots(func: Callable[[float], float], intervals: list[tuple[float, float]]) -> list[float]:
    roots = []
    for a, b in intervals:
        try:
            root = brentq(func, a, b)
            roots.append(root)
        except ValueError:
            continue
    return roots


def exp_kernel_eigen(alpha: float, L: float = 1.0, n_cos: int = 3, n_sin: int = 3):
    """Return eigenvalues and frequencies of ``exp(-alpha*|x-y|)`` on ``[0, L]``.

    Parameters
    ----------
    alpha : float
        Decay parameter of the kernel.
    L : float, optional
        Length of the interval ``[0, L]``.
    n_cos, n_sin : int, optional
        Number of cosine and sine modes to compute.

    Returns
    -------
    cos_vals : ndarray
        Eigenvalues associated with the cosine family.
    cos_freqs : ndarray
        Frequencies ``omega`` of the cosine eigenfunctions.
    sin_vals : ndarray
        Eigenvalues associated with the sine family.
    sin_freqs : ndarray
        Frequencies ``omega`` of the sine eigenfunctions.
    """
    eps = 1e-6
    cos_intervals = [
        (n * np.pi + eps, (n + 0.5) * np.pi - eps) for n in range(n_cos)
    ]
    sin_intervals = [
        ((n + 0.5) * np.pi + eps, (n + 1) * np.pi - eps) for n in range(n_sin)
    ]

    f_cos = lambda w: np.tan(w * L) - w / alpha
    f_sin = lambda w: np.tan(w * L) + alpha / w

    cos_freqs = np.array(_find_roots(f_cos, cos_intervals))
    sin_freqs = np.array(_find_roots(f_sin, sin_intervals))

    cos_vals = 2 * alpha / (alpha**2 + cos_freqs**2)
    sin_vals = 2 * alpha / (alpha**2 + sin_freqs**2)

    return cos_vals, cos_freqs, sin_vals, sin_freqs
