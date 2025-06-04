"""Tools for approximating 1-D covariance kernels.

This module implements routines for fitting a covariance function
with a sum of exponentials

    C(d) \\approx \\sum_{i=1}^{N} a_i * exp(-b_i * d**2)

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
from typing import Callable, Iterable, Tuple, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize
import jax
import jax.numpy as jnp
import numba as nb

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


@jax.jit
def _objective_jax(
    params: jnp.ndarray,
    d: jnp.ndarray,
    target: jnp.ndarray,
    w: jnp.ndarray,
    n_terms: int,
) -> jnp.ndarray:
    a = jnp.exp(params[:n_terms])
    b = jnp.exp(params[n_terms:])
    pred = jnp.sum(a[:, None] * jnp.exp(-b[:, None] * d[None, :] ** 2), axis=0)
    diff = pred - target
    return jnp.sum(w * diff * diff)


def _prepare_jax_funcs(d: np.ndarray, target: np.ndarray, w: np.ndarray, n_terms: int):
    def obj(p):
        return _objective_jax(p, d, target, w, n_terms)

    jitted_obj = jax.jit(obj)
    jitted_grad = jax.jit(jax.grad(obj))
    jitted_hess = jax.jit(jax.hessian(obj))
    return jitted_obj, jitted_grad, jitted_hess


def bisection_line_search(
    f: Callable[[float], float],
    df: Callable[[float], float],
    a: float = 0.0,
    b: float = 1.0,
    *,
    tol: float = 1e-6,
    max_iter: int = 20,
) -> float:
    fa = df(a)
    fb = df(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = df(mid)
        if abs(fm) < tol:
            return mid
        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)


def newton_with_line_search(
    x0: np.ndarray,
    obj: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    hess: Callable[[np.ndarray], np.ndarray],
    *,
    max_iter: int = 10,
    tol: float = 1e-6,
) -> np.ndarray:
    x = np.asarray(x0, dtype=float)
    for _ in range(max_iter):
        g = np.asarray(grad(x))
        if np.linalg.norm(g) < tol:
            break
        H = np.asarray(hess(x))
        try:
            step = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = -g

        def line_obj(alpha: float) -> float:
            return obj(x + alpha * step)

        def line_grad(alpha: float) -> float:
            return float(jax.grad(line_obj)(alpha))

        alpha = bisection_line_search(line_obj, line_grad)
        x = x + alpha * step
    return x


@nb.njit(cache=True)
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


def _differential_evolution(
    obj: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    *,
    max_gen: int = 100,
    pop_size: int = 15,
    rng: np.random.Generator | None = None,
    newton: Callable[[np.ndarray], np.ndarray] | None = None,
    n_newton: int = 0,
) -> np.ndarray:
    dim = len(bounds)
    rng = rng or np.random.default_rng()
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    pop = rng.uniform(size=(pop_size, dim))
    pop = lower + pop * (upper - lower)
    scores = np.array([obj(ind) for ind in pop])
    for _ in range(max_gen):
        for i in range(pop_size):
            a, b, c = pop[rng.choice(pop_size, 3, replace=False)]
            mutant = np.clip(a + 0.8 * (b - c), lower, upper)
            cross = rng.random(dim) < 0.9
            trial = np.where(cross, mutant, pop[i])
            score = obj(trial)
            if score < scores[i]:
                pop[i] = trial
                scores[i] = score

        if newton is not None and n_newton > 0:
            idx = np.argsort(scores)[:n_newton]
            for i in idx:
                refined = newton(pop[i])
                refined = np.clip(refined, lower, upper)
                s = obj(refined)
                if s < scores[i]:
                    pop[i] = refined
                    scores[i] = s

    best_idx = int(np.argmin(scores))
    return pop[best_idx]


def fit_exp_sum(
    n_terms: int,
    x: ArrayLike,
    w: ArrayLike,
    func: Callable[[ArrayLike], ArrayLike],
    *,
    optimiser: OptimiserOptions | None = None,
    method: Literal["de", "de_newton"] = "de",
    max_gen: int = 100,
    pop_size: int = 15,
    n_newton: int = 2,
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
    method : {"de", "de_newton"}, optional
        Optimisation scheme. ``"de"`` uses plain differential evolution,
        while ``"de_newton"`` applies Newton refinement with bisection line
        search to the best individuals in each generation.
    max_gen : int, optional
        Number of generations for the differential evolution.
    pop_size : int, optional
        Population size.
    n_newton : int, optional
        Number of individuals to refine with Newton in each generation
        when ``method='de_newton'``.

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

    if method == "de":
        params = _differential_evolution(
            obj,
            bounds,
            max_gen=max_gen,
            pop_size=pop_size,
            rng=None,
        )
    elif method == "de_newton":
        jax_obj, jax_grad, jax_hess = _prepare_jax_funcs(x, target, w, n_terms)

        def newton_fn(p: np.ndarray) -> np.ndarray:
            return newton_with_line_search(p, jax_obj, jax_grad, jax_hess)

        params = _differential_evolution(
            obj,
            bounds,
            max_gen=max_gen,
            pop_size=pop_size,
            newton=newton_fn,
            n_newton=n_newton,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    a = np.exp(params[:n_terms])
    b = np.exp(params[n_terms:])
    return a, b
