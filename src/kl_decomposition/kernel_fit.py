"""Tools for approximating 1-D covariance kernels.

This module implements routines for fitting a covariance function
with a sum of exponentials

    C(d) \approx \sum_{i=1}^{N} a_i * exp(-b_i * d**2)

where ``a_i, b_i > 0``.  The coefficients are found by solving a
non-linear least squares problem using a combination of global
(differential evolution) and local (L-BFGS-B) optimisation from
:mod:`scipy.optimize`.

The high level function :func:`fit_exp_sum` performs the fit.  Utility
functions for constructing quadrature nodes and weights are provided as
simple building blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize

__all__ = [
    "rectangle_rule",
    "gauss_legendre_rule",
    "fit_exp_sum",
]


def rectangle_rule(
    a: float,
    b: float,
    n: int,
    *,
    open_left: bool = False,
    open_right: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform rectangle rule on ``[a, b]``.

    Parameters
    ----------
    a, b : float
        Interval end points with ``a < b``.
    n : int
        Number of panels.
    open_left, open_right : bool, optional
        Whether to drop the left or right end point.

    Returns
    -------
    x, w : ndarray
        Quadrature nodes and weights.
    """
    h = (b - a) / n
    start = 1 if open_left else 0
    stop = n - 1 if open_right else n
    idx = np.arange(start, stop)
    x = a + (idx + 0.5) * h
    w = np.full_like(x, h, dtype=float)
    return x, w


def gauss_legendre_rule(a: float, b: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss--Legendre quadrature on ``[a, b]`` with ``n`` points."""
    x, w = np.polynomial.legendre.leggauss(n)
    # map from [-1, 1] to [a, b]
    x = 0.5 * (b - a) * x + 0.5 * (b + a)
    w = 0.5 * (b - a) * w
    return x, w


@dataclass
class OptimiserOptions:
    """Configuration for the optimisation routines."""

    de_options: dict | None = None
    local_options: dict | None = None


def _objective(
    params: np.ndarray,
    d: np.ndarray,
    target: np.ndarray,
    w: np.ndarray,
    n_terms: int,
) -> float:
    a = np.exp(params[:n_terms])
    b = np.exp(params[n_terms:])
    pred = np.sum(a[:, None] * np.exp(-b[:, None] * d[None, :] ** 2), axis=0)
    diff = pred - target
    return np.sum(w * diff * diff)


def fit_exp_sum(
    n_terms: int,
    x: ArrayLike,
    w: ArrayLike,
    func: Callable[[ArrayLike], ArrayLike],
    *,
    optimiser: OptimiserOptions | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit ``func`` on ``x`` with weights ``w`` by a sum of exponentials.

    Parameters
    ----------
    n_terms : int
        Number of exponential terms ``N``.
    x, w : array_like
        Nodes and weights describing the fitting error.
    func : callable
        One-dimensional function to approximate.
    optimiser : OptimiserOptions, optional
        Parameters forwarded to the optimisation routines.

    Returns
    -------
    a, b : ndarray
        Arrays of positive coefficients of length ``N``.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    target = np.asarray(func(x), dtype=float)

    opt = optimiser or OptimiserOptions()

    bounds = [(-5.0, 5.0)] * (2 * n_terms)

    def obj(p: np.ndarray) -> float:
        return _objective(p, x, target, w, n_terms)

    result = optimize.differential_evolution(
        obj,
        bounds=bounds,
        **(opt.de_options or {}),
    )
    p0 = result.x
    res_local = optimize.minimize(
        obj,
        p0,
        method="L-BFGS-B",
        options=opt.local_options,
    )
    params = res_local.x
    a = np.exp(params[:n_terms])
    b = np.exp(params[n_terms:])
    return a, b
