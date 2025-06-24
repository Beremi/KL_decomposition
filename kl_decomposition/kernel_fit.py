"""Tools for approximating 1-D covariance kernels.

This module implements routines for fitting a covariance function
with a sum of exponentials

    C(d) \\approx \\sum_{i=1}^{N} a_i * exp(-b_i * d**2)

where ``a_i, b_i > 0``.  The coefficients are found by solving a
non-linear least squares problem.
"""

from scipy.optimize import nnls
import numpy as np
import jax.numpy as jnp
import jax
from typing import Callable, Tuple, List, Dict, Any, Optional

# Enable 64-bit (double) precision in JAX
jax.config.update("jax_enable_x64", True)


def cross_entropy(
    user_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]],
    initial_mean: np.ndarray,
    initial_std: np.ndarray,
    pop_size: int = 100,
    elite_frac: float = 0.2,
    alpha_mean: float = 0.7,
    alpha_std: float = 0.7,
    n_iters: int = 100,
    random_state: Optional[int] = None,
    verbose: bool = False,
    tol_std: float = 1e-6,
    tol_g: float = 1e-6
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Cross-Entropy Method with independent Gaussian distribution and stopping criteria.

    Args:
        user_fn: function taking samples (shape Nxd) and returning
            (updated_samples, scores, info_dict)
        initial_mean: initial mean vector (d,)
        initial_std: initial std deviation vector (d,)
        pop_size: number of samples per iteration
        elite_frac: fraction of samples selected as elites
        alpha_mean: learning rate for mean update
        alpha_std: learning rate for std update
        n_iters: maximum number of iterations
        random_state: RNG seed for reproducibility
        verbose: print debug info each iteration
        tol_std: stop if max(std) <= tol_std
        tol_g: stop if info_dict['normg'] <= tol_g

    Returns:
        best_sample: best found sample (d,)
        best_score: best objective value
        info: dict with:
            'iter': iteration index at stop
            'best_score_history': list of best_score after each iteration
            'max_std_history': list of max_std after each iteration
    """
    rng = np.random.RandomState(random_state)
    mean = np.array(initial_mean, dtype=float)
    std = np.array(initial_std, dtype=float)
    n_elite = max(1, int(np.ceil(pop_size * elite_frac)))
    best_sample: Optional[np.ndarray] = None
    best_score = np.inf

    # history lists
    best_score_history: list = []
    max_std_history: list = []

    for iteration in range(n_iters):
        samples = rng.randn(pop_size, mean.size) * std + mean
        samples, scores, info_dict = user_fn(samples)
        idx = int(np.argmin(scores))
        # update best if improved
        if scores[idx] < best_score:
            best_score = float(scores[idx])
            best_sample = samples[idx]
        # select elites and update distribution
        elite = samples[np.argsort(scores)[:n_elite]]
        elite_mean = elite.mean(axis=0)
        elite_std = elite.std(axis=0, ddof=0)
        mean = alpha_mean * elite_mean + (1 - alpha_mean) * mean
        std = alpha_std * elite_std + (1 - alpha_std) * std
        # record history
        max_std = float(np.max(std))
        best_score_history.append(best_score)
        max_std_history.append(max_std)
        # debug print
        if verbose:
            print(f"Iter {iteration}: best_score={best_score}, max_std={max_std}, info={info_dict}")
        # stopping criteria
        if max_std <= tol_std:
            break
        if info_dict.get('normg', np.inf) <= tol_g:
            break

    info = {
        'iter': iteration,
        'best_score_history': best_score_history,
        'max_std_history': max_std_history
    }
    return best_sample, best_score, info


def build_se_obj_grad_hess(x: np.ndarray, w: np.ndarray, target: np.ndarray):
    """
    Builds objective, gradient, and Hessian functions for squared exponential approximation.

    Parameters:
        x (np.ndarray): Input coordinates (1D array).
        w (np.ndarray): Quadrature weights (same size as x).
        target (np.ndarray): Target function values at x.

    Returns:
        obj (callable): Objective function.
        grad (callable): Gradient of the objective.
        hess (callable): Hessian of the objective.
    """
    # Convert to JAX float64 arrays
    x_j = jnp.array(x, dtype=jnp.float64)
    w_j = jnp.array(w, dtype=jnp.float64)
    target_j = jnp.array(target, dtype=jnp.float64)

    def obj_jax(p):
        n_param = p.shape[0] // 2
        a = p[:n_param]
        b = jnp.exp(p[n_param:])
        pred = jnp.sum(a[:, None] * jnp.exp(-b[:, None] * x_j[None, :] ** 2), axis=0)
        diff = pred - target_j
        return jnp.sqrt(jnp.sum(w_j * diff ** 2))

    obj_jit = jax.jit(obj_jax)
    grad_jit = jax.jit(jax.grad(obj_jax))
    hess_jit = jax.jit(jax.hessian(obj_jax))

    def obj(p):
        return float(obj_jit(jnp.array(p, dtype=jnp.float64)))

    def grad(p):
        return np.array(grad_jit(jnp.array(p, dtype=jnp.float64)))

    def hess(p):
        return np.array(hess_jit(jnp.array(p, dtype=jnp.float64)))

    return obj, grad, hess


def adam_minimize(
    f: Callable[[np.ndarray], float],
    g: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    lr: float = 1e-2,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    max_iter: int = 10_000,
    tol: float = 1e-8,
    verbose: bool = False,
    print_every: int = 100,
    # NEW adaptive-LR knobs
    patience: int = 20,
    decay_factor: float = 0.5,
    callback: Callable[[Dict[str, Any]], None] | None = None,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Adam minimiser with optional adaptive learning-rate decay.

    When the gradient-norm has not improved for `patience` consecutive
    steps, the current learning rate is multiplied by `decay_factor`.
    """
    x = np.asarray(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    f_hist: List[float] = []

    beta1_t = beta2_t = 1.0
    best_grad_norm = np.inf
    wait = 0  # iterations since last grad improvement

    if verbose:
        hdr = " iter |     f         |grad|      lr   "
        print(hdr + "-" * (2 * len(hdr)))

    for t in range(1, max_iter + 1):
        grad = g(x)
        grad_norm = np.linalg.norm(grad)

        # adaptive-LR bookkeeping
        improved = grad_norm < best_grad_norm - 1e-12  # tiny margin
        if improved:
            best_grad_norm = grad_norm
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                lr *= decay_factor
                wait = 0  # reset counter
                if verbose:
                    print(f"{t:6d} | lr decayed → {lr:g}")

        # record objective & early stop
        f_val = f(x)
        f_hist.append(f_val)
        if grad_norm < tol:
            if verbose:
                print(f"{t:6d} | {f_val:11.4e} | {grad_norm:9.2e} | {lr:8g}  (converged)")
            break

        # moment updates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        beta1_t *= beta1
        beta2_t *= beta2
        m_hat = m / (1 - beta1_t)
        v_hat = v / (1 - beta2_t)

        # parameter step
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if verbose and (t == 1 or t % print_every == 0):
            print(f"{t:6d} | {f_val:11.4e} | {grad_norm:9.2e} | {lr:8g}")

        if callback is not None:
            callback(
                dict(iter=t, x=x, f=f_val, grad=grad, grad_norm=grad_norm, lr=lr)
            )

    return x, f_hist[-1], f_hist


def golden_section_search(f, a=0.0, b=1.0, tol=1e-6, max_iter=100):
    """
    Golden-section search to find the minimum of a unimodal function f on [a, b].

    Parameters
    ----------
    f : callable
        The objective function to minimize.
    a : float
        Left endpoint of the initial interval.
    b : float
        Right endpoint of the initial interval.
    tol : float
        Tolerance for the interval width (stopping criterion).
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    x_min : float
        Estimated position of the minimum.
    f_min : float
        Value of f at x_min.
    """
    # Golden ratio constant
    phi = (np.sqrt(5.0) - 1) / 2

    # Initialize interior points
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    f_c = f(c)
    f_d = f(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break

        if f_c < f_d:
            b, d, f_d = d, c, f_c
            c = b - phi * (b - a)
            f_c = f(c)
        else:
            a, c, f_c = c, d, f_d
            d = a + phi * (b - a)
            f_d = f(d)

    # Choose the best of the final points
    if f_c < f_d:
        x_min, f_min = c, f_c
    else:
        x_min, f_min = d, f_d
    # print(f"Iterations: {_ + 1}")
    return x_min, f_min


def newton(f, grad, hess, x0,
           tol_grad=1e-6,
           tol_step=1e-8,
           stall_iter=5,
           max_iter=100,
           ls_bounds=(0.0, 1.0),
           verbose=False):
    """
    Newton's method with backtracking line search and fallback to gradient descent.

    Parameters
    ----------
    f : callable
        Objective function f(x).
    grad : callable
        Gradient of f: grad(x).
    hess : callable
        Hessian of f: hess(x).
    x0 : ndarray
        Initial guess.
    tol_grad : float, optional
        Tolerance for gradient norm.
    tol_step : float, optional
        Tolerance for step size norm for stall criterion.
    stall_iter : int, optional
        Number of consecutive small steps to trigger stop.
    max_iter : int, optional
        Maximum number of iterations.
    ls_bounds : (float, float), optional
        Initial interval for line search.

    Returns
    -------
    x : ndarray
        Estimated minimizer.
    grad_norm : float
        Norm of gradient at the solution.
    f_history : list of float
        History of f values.
    grad_norm_history : list of float
        History of gradient norms.
    """
    x = x0.astype(float)
    f_history = []
    grad_norm_history = []
    fx = f(x)

    stall_count = 0
    trust = 1e-3 * np.eye(len(x))  # trust region to ensure positive definiteness

    for k in range(max_iter):
        gx = grad(x)
        grad_norm = np.linalg.norm(gx)

        f_history.append(fx)
        grad_norm_history.append(grad_norm)

        # Check gradient convergence
        if grad_norm < tol_grad:
            break

        # Try Newton step
        try:
            Hx = hess(x)
            p = -np.linalg.solve(Hx + trust, gx)
            trust /= 2
            # ensure descent direction
            if np.dot(p, gx) >= 0:
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            # fallback to steepest descent
            # print("Hessian not positive definite, using gradient descent")
            p = -gx
            trust *= 10
        # Backtracking line search
        alpha, fx = golden_section_search(
            lambda alpha: f(x + alpha * p),
            a=ls_bounds[0], b=ls_bounds[1], tol=1e-12, max_iter=10000
        )

        # Update
        x_new = x + alpha * p
        step_norm = np.linalg.norm(x_new - x)
        x = x_new

        # Stall criterion
        if step_norm < tol_step:
            stall_count += 1
            if stall_count >= stall_iter:
                break
        else:
            stall_count = 0

        # print debug info
        if verbose:
            print(f"Iter {k}: f={fx:.6e}, grad_norm={grad_norm:.6e}, step_norm={step_norm:.6e}, alpha={alpha:.6f}")

    return x, grad_norm, f_history, grad_norm_history


def optimal_a(d: np.ndarray,
              w: np.ndarray,
              target: np.ndarray,
              b: np.ndarray) -> np.ndarray:
    """
    Solve for a >= 0 minimizing
        sum_j w[j] * (sum_k a[k] * exp(-b[k] * d[j]**2) - target[j])**2

    Parameters
    ----------
    d : (J,) array
        Sample points x_j.
    w : (J,) array
        Quadrature weights.
    target : (J,) array
        Target values at each d[j].
    b : (K,) array
        Exponents b_k.

    Returns
    -------
    a : (K,) array
        Non-negative coefficient vector minimizing the weighted LS error.
    """
    # Build design matrix Φ[j,k] = exp(-b[k] * d[j]**2)
    # Note: np.outer(d**2, b) yields shape (J, K)
    Phi = np.exp(-np.outer(d**2, b))     # shape (J, K)

    # Incorporate weights by scaling rows
    W_sqrt = np.sqrt(w)                  # shape (J,)
    Phi_w = Phi * W_sqrt[:, None]        # shape (J, K)
    y_w = target * W_sqrt              # shape (J,)

    a, _ = nnls(Phi_w, y_w)

    return a


def optimal_a_allow_negative(d: np.ndarray,
                             w: np.ndarray,
                             target: np.ndarray,
                             b: np.ndarray) -> np.ndarray:
    """
    Solve for a minimizing the weighted LS error without non-negativity constraint.

    Parameters
    ----------
    d : (J,) array
        Sample points x_j.
    w : (J,) array
        Quadrature weights.
    target : (J,) array
        Target values at each d[j].
    b : (K,) array
        Exponents b_k.

    Returns
    -------
    a : (K,) array
        Coefficient vector minimizing the weighted LS error.
    """
    # Build design matrix Φ[j,k] = exp(-b[k] * d[j]**2)
    Phi = np.exp(-np.outer(d**2, b))     # shape (J, K)

    # Incorporate weights by scaling rows
    W_sqrt = np.sqrt(w)                  # shape (J,)
    Phi_w = Phi * W_sqrt[:, None]        # shape (J, K)
    y_w = target * W_sqrt              # shape (J,)

    a = np.linalg.lstsq(Phi_w, y_w, rcond=None)[0]

    return a


def interp_extrap_linear_mid(b_old, n_terms, domain=(0.0, 1.0)):
    """
    Like interp_extrap_linear, but assumes *b_old* samples are taken at the
    **mid-points** of *m* equal sub-intervals spanning *domain*.

        ┌─┬─┬─┬─┐
        │•│•│•│•│   • = sample positions (midpoints)
        └─┴─┴─┴─┘

    Parameters
    ----------
    b_old : 1-D numpy array
        Original samples (length m).
    n_terms : int
        Desired length of the output array.
    domain : tuple(float, float), optional
        The (start, end) of the x-axis for *b_old* and *b_new*.

    Returns
    -------
    b_new : 1-D numpy array
        Resampled array of length *n_terms* with linear interpolation and
        edge-extrapolation based on the first/last segment slopes.
    """
    m = len(b_old)
    if m < 2:
        raise ValueError("b_old must contain at least two points.")

    tmp_x_bounds = np.linspace(domain[0], domain[1], m + 1)
    x_old = (tmp_x_bounds[:-1] + tmp_x_bounds[1:]) / 2  # mid-points
    tmp_x_bounds = np.linspace(domain[0], domain[1], n_terms + 1)
    x_new = (tmp_x_bounds[:-1] + tmp_x_bounds[1:]) / 2  # mid-points

    # Interpolate within the convex hull of x_old
    b_new = np.interp(x_new, x_old, b_old)

    # Linear extrapolation on the left
    slope_left = (b_old[1] - b_old[0]) / (x_old[1] - x_old[0])
    mask_left = x_new < x_old[0]
    if np.any(mask_left):
        b_new[mask_left] = b_old[0] + slope_left * (x_new[mask_left] - x_old[0])

    # Linear extrapolation on the right
    slope_right = (b_old[-1] - b_old[-2]) / (x_old[-1] - x_old[-2])
    mask_right = x_new > x_old[-1]
    if np.any(mask_right):
        b_new[mask_right] = b_old[-1] + slope_right * (x_new[mask_right] - x_old[-1])

    return b_new


def square_exp_approximations_newton(
    max_terms: int,
    precision: float,
    x: np.ndarray,
    w: np.ndarray,
    target: np.ndarray,
    obj: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    hess: Callable[[np.ndarray], np.ndarray],
    # defaults for Newton – customise at call-time if needed
    tol_grad: float = 1e-9,
    tol_step: float = 1e-9,
    stall_iter: int = 500,
    max_iter: int = 10_000,
    ls_bounds: Tuple[float, float] = (0.0, 1.0),
    verbose: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Compute square-exponential approximations of orders 1 … max_terms.

    Returns
    -------
    all_a   : list of ndarray
        The `a`-coefficients for each order.
    all_b   : list of ndarray
        The `b`-coefficients (log-length-scales) for each order.
    all_vals: list of float
        Final objective `obj(x_vals)` for each order.
    """
    if max_terms < 1:
        raise ValueError("max_terms must be ≥ 1")

    all_a, all_b, all_vals = [], [], []

    # --- helper to run the optimiser once ---------------------------------
    def _optimise(init_means: np.ndarray, n_terms: int) -> np.ndarray:
        x_vals, *_ = newton(
            f=obj,
            grad=grad,
            hess=hess,
            x0=init_means,
            tol_grad=tol_grad,
            tol_step=tol_step,
            stall_iter=stall_iter,
            max_iter=max_iter,
            ls_bounds=ls_bounds,
            verbose=False
        )
        a_curr, b_curr = x_vals[:n_terms], x_vals[n_terms:]
        all_a.append(a_curr)
        all_b.append(b_curr)
        all_vals.append(obj(x_vals))
        if verbose:
            print(f"n_terms={n_terms:2d}  obj={all_vals[-1]:.6e}")
        return x_vals  # for the next order

    # --- order 1 ----------------------------------------------------------
    b_guess = np.array([0.0])
    a_guess = optimal_a(x, w, target, np.exp(b_guess))
    x_prev = _optimise(np.concatenate((a_guess, b_guess)), 1)

    # --- order 2 ----------------------------------------------------------
    if max_terms >= 2:
        b_guess = np.array([all_b[-1][0] - 1.0, all_b[-1][0] + 1.0])
        a_guess = optimal_a(x, w, target, np.exp(b_guess))
        x_prev = _optimise(np.concatenate((a_guess, b_guess)), 2)

    # --- orders 3 … max_terms --------------------------------------------
    for n_terms in range(3, max_terms + 1):
        # previous x_prev is [a₁…a_{n-1}, b₁…b_{n-1}]
        b_old = x_prev[n_terms - 1:]           # grab the (n-1) previous b's
        b_new = interp_extrap_linear_mid(b_old, n_terms)
        a_new = optimal_a(x, w, target, np.exp(b_new))
        means = np.concatenate((a_new, b_new))
        x_prev = _optimise(means, n_terms)
        if all_vals[-1] < precision:
            if verbose:
                print(f"Early stop at n_terms={n_terms} with obj={all_vals[-1]:.6e}")
            break

    return all_a, all_b, all_vals
