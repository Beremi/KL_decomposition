import numpy as np
from scipy.linalg import eigh_tridiagonal

__all__ = [
    "eval_legendre",
    "legendre_table",
    "shifted_legendre",
    "gauss_legendre_rule",
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


def gauss_legendre_rule(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss--Legendre quadrature on [a, b] using Golub--Welsch."""
    if n < 1:
        raise ValueError("n must be positive")
    i = np.arange(1, n, dtype=float)
    beta = i / np.sqrt(4 * i * i - 1)
    nodes, vecs = eigh_tridiagonal(np.zeros(n), beta)
    weights = 2.0 * (vecs[0] ** 2)
    x = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    w = 0.5 * (b - a) * weights
    return x, w
