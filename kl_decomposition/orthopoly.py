from typing import Tuple, Sequence
import numpy as np
from scipy.linalg import eigh_tridiagonal

__all__ = [
    "eval_legendre",
    "legendre_table",
    "shifted_legendre",
    "gauss_legendre_rule",
    "gauss_legendre_rule_multilevel"
]


def eval_legendre(n: int, x: np.ndarray) -> np.ndarray:
    """Evaluate Legendre polynomial P_n at points x using recurrence."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    p_nm2 = np.ones_like(x)
    p_nm1 = x
    for k in range(2, n + 1):
        p_n = ((2 * k - 1) * x * p_nm1 - (k - 1) * p_nm2) / k
        p_nm2, p_nm1 = p_nm1, p_n
    return p_n


def legendre_table(n_max: int, x: np.ndarray) -> np.ndarray:
    """Return table of orthonormal Legendre polynomials on [0, 1]."""
    x = np.asarray(x, dtype=float)
    t = 2.0 * x - 1.0
    vals = np.empty((n_max, x.size))
    for n in range(n_max):
        vals[n] = np.sqrt(2 * n + 1) * eval_legendre(n, t)
    return vals


def shifted_legendre(n: int, x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Evaluate shifted, L2-orthonormal Legendre polynomial on [a, b]."""
    u = 2.0 * (x - a) / (b - a) - 1.0
    return np.sqrt((2 * n + 1) / (b - a)) * eval_legendre(n, u)


def gauss_legendre_rule(a: float, b: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classic n-point Gauss–Legendre rule on the interval [a, b] (Golub–Welsch).

    Returns
    -------
    x : ndarray
        Quadrature nodes.
    w : ndarray
        Corresponding weights.
    """
    if n < 1:
        raise ValueError("n must be positive")

    i = np.arange(1, n, dtype=float)
    beta = i / np.sqrt(4.0 * i * i - 1.0)
    nodes, vecs = eigh_tridiagonal(np.zeros(n), beta)
    weights = 2.0 * (vecs[0] ** 2)

    # affine map [-1,1] → [a,b]
    x = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    w = 0.5 * (b - a) * weights
    return x, w


def _make_breakpoints(a: float, b: float, L: int, ratio: float) -> Sequence[float]:
    """
    Compute the break-points that split [a,b] into a geometric sequence of
    sub-intervals clustered near *a*.

    For example with a = 0, b = 1, L = 3, ratio = 0.2 the list returned is
    [0, 0.2**3, 0.2**2, 0.2, 1].
    """
    if not (0.0 < ratio < 1.0):
        raise ValueError("ratio must lie in (0, 1)")
    if L < 0:
        raise ValueError("L must be non-negative")

    if L == 0:
        return [a, b]                         # no splitting

    # geometric points measured from the *left* endpoint a
    length = b - a
    # ratio**L, ratio**(L-1), …, ratio**1
    mids = [a + length * ratio**k for k in range(L, 0, -1)]
    return [a, *mids, b]


def gauss_legendre_rule_multilevel(
    a: float,
    b: float,
    n: int,
    L: int = 0,
    ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-level Gauss–Legendre rule on [a,b].

    Parameters
    ----------
    a, b : float
        Integration limits (``a`` < ``b``).  Typically ``a = 0`` for the
        singular/peaky behaviour you described.
    n : int
        Degree of each *individual* Gauss–Legendre rule.
    L : int, default 0
        Number of *levels*.  ``L = 0`` means the usual single interval.
        ``L ≥ 1`` splits the left part of the interval into ``L`` geometrically
        shrinking sections, plus one final “catch-all’’ on the right.
    ratio : float, default 0.5
        Geometric ratio between successive break-points (must lie in (0, 1)).
        Smaller values push more nodes towards the left endpoint.

    Returns
    -------
    x : ndarray
        All nodes on all sub-intervals (concatenated, sorted ascending).
    w : ndarray
        Corresponding weights (one weight per node).

    Notes
    -----
    *Total* number of nodes = ``(L + 1) * n``.
    """
    breaks = _make_breakpoints(a, b, L, ratio)

    xs, ws = [], []
    for left, right in zip(breaks[:-1], breaks[1:]):
        x, w = gauss_legendre_rule(left, right, n)
        xs.append(x)
        ws.append(w)

    # Concatenate and (optionally) sort – the breakpoints are already ascending
    x_all = np.concatenate(xs)
    w_all = np.concatenate(ws)
    return x_all, w_all


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
