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
import functools
import time

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
    "DEStats",
    "NewtonStats",
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


@dataclass
class DEStats:
    """Statistics returned by the differential evolution."""

    iterations: int
    best_score: float
    eval_count: int
    history: list
    runtime: float


@dataclass
class NewtonStats:
    """Statistics for Newton optimisation."""

    iterations: int
    runtime: float


@functools.partial(jax.jit, static_argnums=(4,))
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


def _prepare_jax_funcs(
    d: np.ndarray,
    target: np.ndarray,
    w: np.ndarray,
    n_terms: int,
    compiled: bool = True,
):
    def obj(p):
        return _objective_jax(p, d, target, w, n_terms)

    if compiled:
        jitted_obj = jax.jit(obj)
        jitted_grad = jax.jit(jax.grad(obj))
        jitted_hess = jax.jit(jax.hessian(obj))
        return jitted_obj, jitted_grad, jitted_hess
    return obj, jax.grad(obj), jax.hessian(obj)


def _prepare_numpy_funcs(
    d: np.ndarray,
    target: np.ndarray,
    w: np.ndarray,
    n_terms: int,
):
    def obj(p: np.ndarray) -> float:
        return _objective_py(p, d, target, w, n_terms)

    def grad(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        g = np.zeros_like(p)
        for i in range(len(p)):
            step = np.zeros_like(p)
            step[i] = eps
            g[i] = (obj(p + step) - obj(p - step)) / (2 * eps)
        return g

    def hess(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        n = len(p)
        H = np.zeros((n, n))
        for i in range(n):
            step_i = np.zeros_like(p)
            step_i[i] = eps
            for j in range(n):
                step_j = np.zeros_like(p)
                step_j[j] = eps
                fpp = obj(p + step_i + step_j)
                fpm = obj(p + step_i - step_j)
                fmp = obj(p - step_i + step_j)
                fmm = obj(p - step_i - step_j)
                H[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps ** 2)
        return H

    return obj, grad, hess


def bisection_line_search(
    f: Callable[[float], float],
    df: Callable[[float], float],
    a: float = 0.0,
    b: float = 1.0,
    *,
    tol: float = 1e-6,
    max_iter: int = 20,
) -> tuple[float, int]:
    fa = df(a)
    fb = df(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    steps = 0
    for _ in range(max_iter):
        steps += 1
        mid = 0.5 * (a + b)
        fm = df(mid)
        if abs(fm) < tol:
            return mid, steps
        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b), steps


def newton_with_line_search(
    x0: np.ndarray,
    obj: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    hess: Callable[[np.ndarray], np.ndarray],
    *,
    max_iter: int = 10,
    tol: float = 1e-6,
    compiled: bool = True,
    return_stats: bool = False,
) -> tuple[np.ndarray, NewtonStats] | np.ndarray:
    x = np.asarray(x0, dtype=float)
    start = time.time()
    n_iter = 0
    for i in range(max_iter):
        g = np.asarray(grad(x))
        n_iter += 1
        if np.linalg.norm(g) < tol:
            break
        H = np.asarray(hess(x))
        try:
            step = -np.linalg.solve(H, g)
            if np.dot(step, g) > 0:
                step = -g
        except np.linalg.LinAlgError:
            step = -g

        def line_obj(alpha: float) -> float:
            return obj(x + alpha * step)

        def line_grad(alpha: float) -> float:
            if compiled:
                return float(jax.grad(line_obj)(alpha))
            eps = 1e-6
            return (line_obj(alpha + eps) - line_obj(alpha - eps)) / (2 * eps)

        alpha, ls_steps = bisection_line_search(line_obj, line_grad)
        print(
            f"iter {i}: grad_norm={np.linalg.norm(g):.2e}, "
            f"alpha={alpha:.2e}, ls_steps={ls_steps}"
        )
        x = x + alpha * step
    runtime = time.time() - start
    if return_stats:
        return x, NewtonStats(n_iter, runtime)
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


def _objective_py(
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
    bounds: list[tuple[float, float]] | None,
    *,
    max_gen: int = 100,
    pop_size: int = 15,
    rng: np.random.Generator | None = None,
    newton: Callable[[np.ndarray], np.ndarray] | None = None,
    n_newton: int = 0,
    mean: np.ndarray | float | None = None,
    sigma: np.ndarray | float | None = None,
    return_stats: bool = False,
) -> np.ndarray:
    if mean is None:
        mean = 1.0
    if sigma is None:
        sigma = 1.0
    if bounds is not None:
        dim = len(bounds)
    else:
        dim = len(np.atleast_1d(mean))
    rng = rng or np.random.default_rng()
    start = time.time()
    eval_count = 0
    if bounds is not None:
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        pop = rng.normal(loc=mean, scale=sigma, size=(pop_size, dim))
        pop = np.clip(pop, lower, upper)
    else:
        pop = rng.normal(loc=mean, scale=sigma, size=(pop_size, dim))
    scores = np.array([obj(ind) for ind in pop])
    eval_count += pop_size
    history: list[float] = []
    for _ in range(max_gen):
        for i in range(pop_size):
            a, b, c = pop[rng.choice(pop_size, 3, replace=False)]
            mutant = a + 0.8 * (b - c)
            if bounds is not None:
                mutant = np.clip(mutant, lower, upper)
            cross = rng.random(dim) < 0.9
            trial = np.where(cross, mutant, pop[i])
            score = obj(trial)
            eval_count += 1
            if score < scores[i]:
                pop[i] = trial
                scores[i] = score

        if newton is not None and n_newton > 0:
            idx = np.argsort(scores)[:n_newton]
            for i in idx:
                refined = newton(pop[i])
                if isinstance(refined, tuple):
                    refined = refined[0]
                if bounds is not None:
                    refined = np.clip(refined, lower, upper)
                s = obj(refined)
                eval_count += 1
                if s < scores[i]:
                    pop[i] = refined
                    scores[i] = s

        history.append(float(np.min(scores)))

    runtime = time.time() - start
    best_idx = int(np.argmin(scores))
    best = pop[best_idx]
    if return_stats:
        stats = DEStats(
            iterations=max_gen,
            best_score=float(scores[best_idx]),
            eval_count=eval_count,
            history=history,
            runtime=runtime,
        )
        return best, stats
    return best


def fit_exp_sum(
    n_terms: int,
    x: ArrayLike,
    w: ArrayLike,
    func: Callable[[ArrayLike], ArrayLike],
    *,
    optimiser: OptimiserOptions | None = None,
    method: Literal["de", "de_newton", "de_newton1", "de_ls"] = "de",
    max_gen: int = 100,
    pop_size: int = 15,
    n_newton: int = 2,
    de_mean: ArrayLike = 1.0,
    de_sigma: ArrayLike = 1.0,
    newton_max_iter: int = 10,
    compiled: bool = True,
    return_info: bool = False,
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
    method : {"de", "de_newton", "de_newton1", "de_ls"}, optional
        Optimisation scheme. ``"de"`` uses plain differential evolution,
        ``"de_newton"`` applies multi-step Newton refinement, ``"de_newton1"``
        performs only one Newton iteration, and ``"de_ls"`` performs
        differential evolution only on the ``b_i`` parameters with the
        ``a_i`` coefficients obtained by weighted least squares.
    max_gen : int, optional
        Number of generations for the differential evolution.
    pop_size : int, optional
        Population size.
    n_newton : int, optional
        Number of individuals to refine with Newton in each generation
        when ``method='de_newton'``.
    de_mean, de_sigma : array_like, optional
        Mean and standard deviation for the initial population of the
        differential evolution. Defaults to ``1``.
    newton_max_iter : int, optional
        Maximum Newton iterations when ``method='de_newton'``.
    compiled : bool, optional
        If ``False``, use pure NumPy routines without JIT compilation.
    return_info : bool, optional
        If ``True``, return optimisation statistics in addition to the
        fitted coefficients.

    Returns
    -------
    a, b : ndarray
        Arrays of coefficients of length ``N``.
    info : DEStats, optional
        Returned when ``return_info`` is ``True`` and contains
        convergence statistics of the optimisation.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    target = np.asarray(func(x), dtype=float)

    opt = optimiser or OptimiserOptions()

    bounds = [(-5.0, 5.0)] * (2 * n_terms)

    objective_fn = _objective if compiled else _objective_py

    def obj(p: np.ndarray) -> float:
        return objective_fn(p, x, target, w, n_terms)

    if method == "de":
        params, info = _differential_evolution(
            obj,
            bounds,
            max_gen=max_gen,
            pop_size=pop_size,
            rng=None,
            mean=de_mean,
            sigma=de_sigma,
            return_stats=True,
        )
    elif method in {"de_newton", "de_newton1"}:
        if compiled:
            obj_n, grad_n, hess_n = _prepare_jax_funcs(
                x, target, w, n_terms, compiled=True
            )
        else:
            obj_n, grad_n, hess_n = _prepare_numpy_funcs(
                x, target, w, n_terms
            )

        max_it = 1 if method == "de_newton1" else newton_max_iter

        def newton_fn(p: np.ndarray) -> np.ndarray:
            return newton_with_line_search(
                p, obj_n, grad_n, hess_n, max_iter=max_it, compiled=compiled
            )[0]

        params, info = _differential_evolution(
            obj,
            bounds,
            max_gen=max_gen,
            pop_size=pop_size,
            newton=newton_fn,
            n_newton=n_newton,
            mean=de_mean,
            sigma=de_sigma,
            return_stats=True,
        )
    elif method == "de_ls":
        def obj_b(b_params: np.ndarray) -> float:
            b_sorted = np.sort(b_params)
            F = np.exp(-b_sorted[None, :] * x[:, None] ** 2)
            A = F * np.sqrt(w)[:, None]
            y = target * np.sqrt(w)
            a_ls, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            pred = F @ a_ls
            diff = pred - target
            return float(np.sum(w * diff * diff))

        params_b, info = _differential_evolution(
            obj_b,
            None,
            max_gen=max_gen,
            pop_size=pop_size,
            mean=de_mean,
            sigma=de_sigma,
            return_stats=True,
        )
        b_sorted = np.sort(params_b)
        F = np.exp(-b_sorted[None, :] * x[:, None] ** 2)
        A = F * np.sqrt(w)[:, None]
        y = target * np.sqrt(w)
        a_ls, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        params = np.concatenate([a_ls, b_sorted])
    else:
        raise ValueError(f"Unknown method: {method}")

    a = np.exp(params[:n_terms])
    b = np.exp(params[n_terms:])
    if method == "de_ls":
        a = params[:n_terms]
        b = params[n_terms:]
    if return_info:
        return a, b, info
    return a, b
