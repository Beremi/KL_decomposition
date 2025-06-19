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
from typing import Callable, Tuple, Literal
import time

import numpy as np
from .orthopoly import gauss_legendre_rule
from numpy.typing import ArrayLike
import jax
import jax.numpy as jnp
try:
    import numba as nb
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    class _NB:
        def njit(self, *args, **kwargs):
            if args and callable(args[0]):
                return args[0]

            def wrapper(func):
                return func

            return wrapper

    nb = _NB()

__all__ = [
    "rectangle_rule",
    "gauss_legendre_rule",
    "fit_exp_sum",
    "fit_exp_sum_ce",
    "fit_exp_sum_sorted",
    "DEStats",
    "NewtonStats",
    "CEStats",
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
    grad_history: list | None = None


@dataclass
class NewtonStats:
    """Statistics for Newton optimisation."""

    iterations: int
    runtime: float


@dataclass
class CEStats:
    """Statistics for the cross-entropy optimisation."""

    iterations: int
    best_score: float
    eval_count: int
    history: list
    runtime: float


@jax.jit
def _objective_jax_de(
    params: jnp.ndarray,
    d: jnp.ndarray,
    target: jnp.ndarray,
    w: jnp.ndarray,
) -> jnp.ndarray:
    n_terms = params.shape[0] // 2
    a = jnp.exp(params[:n_terms])
    b = jnp.exp(params[n_terms:])
    pred = jnp.sum(a[:, None] * jnp.exp(-b[:, None] * d[None, :] ** 2), axis=0)
    diff = pred - target
    return jnp.sum(w * diff * diff)


@jax.jit
def _objective_jax_newton(
    params: jnp.ndarray,
    d: jnp.ndarray,
    target: jnp.ndarray,
    w: jnp.ndarray,
) -> jnp.ndarray:
    n_terms = params.shape[0] // 2
    a = params[:n_terms]
    b = jnp.exp(params[n_terms:])
    pred = jnp.sum(a[:, None] * jnp.exp(-b[:, None] * d[None, :] ** 2), axis=0)
    diff = pred - target
    return jnp.sum(w * diff * diff)


def _prepare_jax_funcs(
    d: np.ndarray,
    target: np.ndarray,
    w: np.ndarray,
    compiled: bool = True,
    newton: bool = False,
):
    if newton:

        def obj(p):
            return _objective_jax_newton(p, d, target, w)

    else:

        def obj(p):
            return _objective_jax_de(p, d, target, w)

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
    newton: bool = False,
):
    if newton:

        def obj(p: np.ndarray) -> float:
            return _objective_py_newton(p, d, target, w)

    else:

        def obj(p: np.ndarray) -> float:
            return _objective_py_de(p, d, target, w)

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
                H[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps**2)
        return H

    return obj, grad, hess


def bisection_line_search(
    f: Callable[[float], float],
    a: float = -1.0,
    b: float = 2.0,
    *,
    tol: float = 1e-6,
    max_iter: int = 20,
) -> tuple[float, int]:
    """Minimise ``f`` on ``[a, b]`` by a golden-section search."""

    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = f(c)
    fd = f(d)
    steps = 0

    for _ in range(max_iter):
        steps += 1
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
        if b - a < tol:
            break
    return 0.5 * (a + b), steps


def newton_with_line_search(
    x0: np.ndarray,
    obj: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    hess: Callable[[np.ndarray], np.ndarray],
    *,
    max_iter: int = 10,
    grad_tol: float = 1e-6,
    compiled: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, NewtonStats] | np.ndarray:
    x = np.asarray(x0, dtype=float)
    start = time.time()
    n_iter = 0
    for i in range(max_iter):
        g = np.asarray(grad(x))
        n_iter += 1
        if np.linalg.norm(g) < grad_tol:
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

        alpha, ls_steps = bisection_line_search(line_obj)
        if verbose:
            print(
                f"iter {i}: grad_norm={np.linalg.norm(g):.2e}, "
                f"alpha={alpha:.2e}, ls_steps={ls_steps}"
            )
        x = x + alpha * step
    runtime = time.time() - start
    return x, NewtonStats(n_iter, runtime)


@nb.njit(cache=True)
def _objective_de(
    params: np.ndarray,
    d: np.ndarray,
    target: np.ndarray,
    w: np.ndarray,
) -> float:
    n_terms = params.shape[0] // 2
    a = np.exp(params[:n_terms])
    b = np.exp(params[n_terms:])
    pred = np.sum(a[:, None] * np.exp(-b[:, None] * d[None, :] ** 2), axis=0)
    diff = pred - target
    return np.sum(w * diff * diff)


def _objective_py_de(
    params: np.ndarray,
    d: np.ndarray,
    target: np.ndarray,
    w: np.ndarray,
) -> float:
    n_terms = params.shape[0] // 2
    a = np.exp(params[:n_terms])
    b = np.exp(params[n_terms:])
    pred = np.sum(a[:, None] * np.exp(-b[:, None] * d[None, :] ** 2), axis=0)
    diff = pred - target
    return np.sum(w * diff * diff)


def _objective_py_newton(
    params: np.ndarray,
    d: np.ndarray,
    target: np.ndarray,
    w: np.ndarray,
) -> float:
    n_terms = params.shape[0] // 2
    a = params[:n_terms]
    b = np.exp(params[n_terms:])
    pred = np.sum(a[:, None] * np.exp(-b[:, None] * d[None, :] ** 2), axis=0)
    diff = pred - target
    return np.sum(w * diff * diff)


def _differential_evolution(
    obj: Callable[[np.ndarray], float],
    *,
    max_gen: int = 100,
    pop_size: int = 15,
    rng: np.random.Generator | None = None,
    newton: Callable[[np.ndarray], np.ndarray] | None = None,
    n_newton: int = 0,
    mean: np.ndarray | float | None = None,
    sigma: np.ndarray | float | None = None,
    grad: Callable[[np.ndarray], np.ndarray] | None = None,
    grad_tol: float = 0.0,
    verbose: bool = False,
) -> np.ndarray:
    if mean is None:
        mean = 1.0
    if sigma is None:
        sigma = 1.0
    dim = len(np.atleast_1d(mean))
    rng = rng or np.random.default_rng()
    start = time.time()
    eval_count = 0
    pop = rng.normal(loc=mean, scale=sigma, size=(pop_size, dim))
    scores = np.array([obj(ind) for ind in pop])
    eval_count += pop_size
    history: list[float] = []
    grad_history: list[float] = []
    for gen in range(max_gen):
        for i in range(pop_size):
            a, b, c = pop[rng.choice(pop_size, 3, replace=False)]
            mutant = a + 0.8 * (b - c)
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
                s = obj(refined)
                eval_count += 1
                if s < scores[i]:
                    pop[i] = refined
                    scores[i] = s

        best_idx = int(np.argmin(scores))
        best_gen = float(scores[best_idx])
        history.append(best_gen)
        if grad is not None:
            g_norm = float(np.linalg.norm(grad(pop[best_idx])))
            grad_history.append(g_norm)
            if verbose:
                print(
                    f"gen {gen}: best_score={best_gen:.2e}, grad_norm={g_norm:.2e}"
                )
            if grad_tol > 0 and g_norm < grad_tol:
                break
        elif verbose:
            print(f"gen {gen}: best_score={best_gen:.2e}")

    runtime = time.time() - start
    best_idx = int(np.argmin(scores))
    best = pop[best_idx]
    stats = DEStats(
        iterations=len(history),
        best_score=float(scores[best_idx]),
        eval_count=eval_count,
        history=history,
        runtime=runtime,
        grad_history=grad_history,
    )
    return best, stats


def _sort_params(p: np.ndarray, n_terms: int) -> np.ndarray:
    """Return ``p`` with the second half sorted and first half permuted."""
    order = np.argsort(p[n_terms:])
    return np.concatenate([p[:n_terms][order], p[n_terms:][order]])


def _sort_population(pop: np.ndarray, n_terms: int) -> None:
    """Sort parameters of each individual in ``pop`` in-place."""
    for i in range(pop.shape[0]):
        pop[i] = _sort_params(pop[i], n_terms)


def _differential_evolution_sorted(
    obj: Callable[[np.ndarray], float],
    n_terms: int,
    *,
    max_gen: int = 100,
    pop_size: int = 15,
    rng: np.random.Generator | None = None,
    newton: Callable[[np.ndarray], np.ndarray] | None = None,
    n_newton: int = 0,
    mean: np.ndarray | float | None = None,
    sigma: np.ndarray | float | None = None,
    grad: Callable[[np.ndarray], np.ndarray] | None = None,
    grad_tol: float = 0.0,
    verbose: bool = False,
) -> tuple[np.ndarray, DEStats]:
    if mean is None:
        mean = 1.0
    if sigma is None:
        sigma = 1.0
    dim = 2 * n_terms
    rng = rng or np.random.default_rng()
    start = time.time()
    eval_count = 0
    pop = rng.normal(loc=mean, scale=sigma, size=(pop_size, dim))
    _sort_population(pop, n_terms)
    scores = np.array([obj(ind) for ind in pop])
    eval_count += pop_size
    history: list[float] = []
    grad_history: list[float] = []

    for gen in range(max_gen):
        for i in range(pop_size):
            a, b, c = pop[rng.choice(pop_size, 3, replace=False)]
            mutant = a + 0.8 * (b - c)
            cross = rng.random(dim) < 0.9
            trial = np.where(cross, mutant, pop[i])
            trial = _sort_params(trial, n_terms)
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
                refined = _sort_params(refined, n_terms)
                s = obj(refined)
                eval_count += 1
                if s < scores[i]:
                    pop[i] = refined
                    scores[i] = s

        best_idx = int(np.argmin(scores))
        best_gen = float(scores[best_idx])
        history.append(best_gen)
        if grad is not None:
            g_norm = float(np.linalg.norm(grad(pop[best_idx])))
            grad_history.append(g_norm)
            if verbose:
                print(
                    f"gen {gen}: best_score={best_gen:.2e}, grad_norm={g_norm:.2e}"
                )
            if grad_tol > 0 and g_norm < grad_tol:
                break
        elif verbose:
            print(f"gen {gen}: best_score={best_gen:.2e}")

    runtime = time.time() - start
    best_idx = int(np.argmin(scores))
    best = pop[best_idx]
    stats = DEStats(
        iterations=len(history),
        best_score=float(scores[best_idx]),
        eval_count=eval_count,
        history=history,
        runtime=runtime,
        grad_history=grad_history,
    )
    return best, stats


def _cross_entropy(
    obj: Callable[[np.ndarray], float],
    *,
    dim: int,
    max_iter: int = 50,
    pop_size: int = 50,
    elite_frac: float = 0.1,
    mean: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, CEStats]:
    """Cross-entropy minimisation for ``obj``."""

    rng = rng or np.random.default_rng()
    mean = np.ones(dim) if mean is None else np.asarray(mean, dtype=float)
    if cov is None:
        cov = np.eye(dim)
    cov = np.asarray(cov, dtype=float)

    history: list[float] = []
    eval_count = 0
    start = time.time()
    best_params = mean
    best_score = float("inf")

    elite_num = max(1, int(pop_size * elite_frac))

    for it in range(max_iter):
        samples = rng.multivariate_normal(mean, cov, size=pop_size)
        scores = np.array([obj(s) for s in samples])
        eval_count += pop_size
        idx = np.argsort(scores)
        elite = samples[idx[:elite_num]]
        mean = elite.mean(axis=0)
        cov = np.cov(elite, rowvar=False) + 1e-6 * np.eye(dim)
        history.append(float(scores[idx[0]]))
        if scores[idx[0]] < best_score:
            best_score = float(scores[idx[0]])
            best_params = samples[idx[0]]

    runtime = time.time() - start
    stats = CEStats(
        iterations=len(history),
        best_score=best_score,
        eval_count=eval_count,
        history=history,
        runtime=runtime,
    )
    return best_params, stats


def fit_exp_sum(
    n_terms: int,
    x: ArrayLike,
    w: ArrayLike,
    func: Callable[[ArrayLike], ArrayLike],
    *,
    optimiser: OptimiserOptions | None = None,
    method: Literal["de_newton", "de_ls"] = "de_newton",
    max_gen: int = 100,
    pop_size: int = 15,
    n_newton: int = 2,
    de_mean: ArrayLike | None = None,
    de_sigma: ArrayLike | None = None,
    newton_max_iter: int = 10,
    compiled: bool = True,
) -> Tuple[np.ndarray, np.ndarray, DEStats]:
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
    method : {"de_newton", "de_ls"}, optional
        Optimisation scheme. ``"de_newton"`` applies multi-step Newton
        refinement while ``"de_ls"`` performs differential evolution only on
        parameters ``c_i`` such that ``b_i = exp(c_i)``.  The corresponding
        ``a_i`` coefficients are obtained by weighted least squares.
    max_gen : int, optional
        Number of generations for the differential evolution.
    pop_size : int, optional
        Population size.
    n_newton : int, optional
        Number of individuals to refine with Newton in each generation
        when ``method='de_newton'``.
    de_mean, de_sigma : array_like, optional
        Mean and standard deviation for the initial population of the
        differential evolution. If not provided, arrays of ones are used
        (length ``2*N`` for the standard methods and ``N`` for ``'de_ls'``).
    newton_max_iter : int, optional
        Maximum Newton iterations when ``method='de_newton'``.
    compiled : bool, optional
        If ``False``, use pure NumPy routines without JIT compilation.

    Returns
    -------
    a, b : ndarray
        Arrays of coefficients of length ``N``.
    info : DEStats
        Convergence statistics of the optimisation.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    target = np.asarray(func(x), dtype=float)

    # opt = optimiser or OptimiserOptions()

    if de_mean is None:
        de_mean = np.ones(n_terms if method == "de_ls" else 2 * n_terms)
    else:
        de_mean = np.asarray(de_mean, dtype=float)
    if de_sigma is None:
        de_sigma = np.ones_like(de_mean)
    else:
        de_sigma = np.asarray(de_sigma, dtype=float)

    objective_fn = _objective_de if compiled else _objective_py_de

    def obj(p: np.ndarray) -> float:
        return objective_fn(p, x, target, w)

    if method == "de_newton":
        if compiled:
            obj_n, grad_n, hess_n = _prepare_jax_funcs(
                x, target, w, compiled=True, newton=True
            )
        else:
            obj_n, grad_n, hess_n = _prepare_numpy_funcs(
                x, target, w, newton=True
            )

        max_it = newton_max_iter

        def newton_fn(p: np.ndarray) -> np.ndarray:
            start = np.concatenate([np.exp(p[:n_terms]), p[n_terms:]])
            refined, _ = newton_with_line_search(
                start,
                obj_n,
                grad_n,
                hess_n,
                max_iter=max_it,
                compiled=compiled,
            )
            return np.concatenate([np.log(refined[:n_terms]), refined[n_terms:]])

        params, info = _differential_evolution(
            obj,
            max_gen=max_gen,
            pop_size=pop_size,
            newton=newton_fn,
            n_newton=n_newton,
            mean=de_mean,
            sigma=de_sigma,
            verbose=False,
        )
    elif method == "de_ls":

        def obj_c(c_params: np.ndarray) -> float:
            b_sorted = np.sort(np.exp(c_params))
            F = np.exp(-b_sorted[None, :] * x[:, None] ** 2)
            A = F * np.sqrt(w)[:, None]
            y = target * np.sqrt(w)
            a_ls, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            pred = F @ a_ls
            diff = pred - target
            return float(np.sum(w * diff * diff))

        params_c, info = _differential_evolution(
            obj_c,
            max_gen=max_gen,
            pop_size=pop_size,
            mean=de_mean,
            sigma=de_sigma,
            verbose=False,
        )
        b_sorted = np.sort(np.exp(params_c))
        F = np.exp(-b_sorted[None, :] * x[:, None] ** 2)
        A = F * np.sqrt(w)[:, None]
        y = target * np.sqrt(w)
        a_ls, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        params = np.concatenate([a_ls, b_sorted])
    else:
        raise ValueError(f"Unknown method: {method}")

    if method == "de_ls":
        a = params[:n_terms]
        b = params[n_terms:]
    else:
        a = np.exp(params[:n_terms])
        b = np.exp(params[n_terms:])
    return a, b, info


def fit_exp_sum_ce(
    n_terms: int,
    x: ArrayLike,
    w: ArrayLike,
    func: Callable[[ArrayLike], ArrayLike],
    *,
    iterations: int = 50,
    pop_size: int = 50,
    elite_frac: float = 0.1,
    ce_mean: ArrayLike | None = None,
    ce_cov: ArrayLike | None = None,
    compiled: bool = True,
) -> Tuple[np.ndarray, np.ndarray, CEStats]:
    """Fit ``func`` using a cross-entropy search."""

    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    target = np.asarray(func(x), dtype=float)

    if ce_mean is None:
        ce_mean = np.ones(2 * n_terms)
    else:
        ce_mean = np.asarray(ce_mean, dtype=float)
    if ce_cov is None:
        ce_cov = np.eye(2 * n_terms)
    else:
        ce_cov = np.asarray(ce_cov, dtype=float)

    objective_fn = _objective_de if compiled else _objective_py_de

    def obj(p: np.ndarray) -> float:
        return objective_fn(p, x, target, w)

    params, info = _cross_entropy(
        obj,
        dim=2 * n_terms,
        max_iter=iterations,
        pop_size=pop_size,
        elite_frac=elite_frac,
        mean=ce_mean,
        cov=ce_cov,
    )

    a = np.exp(params[:n_terms])
    b = np.exp(params[n_terms:])
    return a, b, info


def fit_exp_sum_sorted(
    n_terms: int,
    x: ArrayLike,
    w: ArrayLike,
    func: Callable[[ArrayLike], ArrayLike],
    *,
    max_gen: int = 100,
    pop_size: int = 15,
    n_newton: int = 2,
    de_mean: ArrayLike | None = None,
    de_sigma: ArrayLike | None = None,
    compiled: bool = False,
) -> Tuple[np.ndarray, np.ndarray, DEStats]:
    """Fit ``func`` using differential evolution with sorted Newton steps."""

    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    target = np.asarray(func(x), dtype=float)

    if de_mean is None:
        de_mean = np.ones(2 * n_terms)
    else:
        de_mean = np.asarray(de_mean, dtype=float)
    if de_sigma is None:
        de_sigma = np.ones_like(de_mean)
    else:
        de_sigma = np.asarray(de_sigma, dtype=float)

    if compiled:
        obj, grad_n, hess_n = _prepare_jax_funcs(
            x, target, w, compiled=True, newton=False
        )
    else:
        obj, grad_n, hess_n = _prepare_numpy_funcs(
            x, target, w, newton=False
        )

    def newton_fn(p: np.ndarray) -> np.ndarray:
        refined, _ = newton_with_line_search(
            p, obj, grad_n, hess_n, max_iter=1, compiled=compiled
        )
        return refined

    params, info = _differential_evolution_sorted(
        obj,
        n_terms,
        max_gen=max_gen,
        pop_size=pop_size,
        newton=newton_fn,
        n_newton=n_newton,
        mean=de_mean,
        sigma=de_sigma,
        verbose=True,
    )
    a = np.exp(params[:n_terms])
    b = np.exp(params[n_terms:])
    return a, b, info
