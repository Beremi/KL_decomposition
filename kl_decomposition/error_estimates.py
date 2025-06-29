import numpy as np
from .orthopoly import gauss_legendre_rule


def residual_l2(
    c,                    # c(x,y) kernel (vectorised)
    u,                    # eigenfunction approximation (vectorised)
    lam,                  # eigenvalue approximation
    quad_outer=30,        # #pts for the outer (x) quadrature
    quad_inner=30,        # #pts for EACH inner (y) sub-interval
):
    """
    L2–norm of the residual  R(x) = ∫_0^1 c(x,y)u(y)dy − λu(x)
    on the interval T = [0,1].

    The y-integral is split at y = x so that each sub-integrand is smooth.
    """
    # --- outer quadrature in x ------------------------------------------------
    x, w_x = gauss_legendre_rule(0.0, 1.0, quad_outer)  # (N,), (N,)
    u_x = u(x)                                      # cache u(x_k)
    R_vals = np.empty_like(x)

    # --- loop over x-abscissae ------------------------------------------------
    for k, xk in enumerate(x):
        # sub-interval 1 : 0 <= y <= x
        if xk > 0.0:
            y1, w1 = gauss_legendre_rule(0.0, xk, quad_inner)
            int1 = np.sum(w1 * c(xk, y1) * u(y1))
        else:
            int1 = 0.0

        # sub-interval 2 : x <= y <= 1
        if xk < 1.0:
            y2, w2 = gauss_legendre_rule(xk, 1.0, quad_inner)
            int2 = np.sum(w2 * c(xk, y2) * u(y2))
        else:
            int2 = 0.0

        R_vals[k] = int1 + int2 - lam * u_x[k]

    # --- L2 norm --------------------------------------------------------------
    return np.sqrt(np.sum(w_x * R_vals**2))


def truncated_covariance(x, y, lambdas, phis):
    """
    x, y : array-like (broadcastable to a common shape)
    lambdas : (N,) array of eigenvalues
    phis    : list of N callables
    returns : ndarray with shape broadcast(x, y)
    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    # Evaluate all eigenfunctions at once
    Phi_x = np.stack([phi(x) for phi in phis])   # shape (N, …)
    Phi_y = np.stack([phi(y) for phi in phis])   # shape (N, …)
    # Weighted inner product over the KL index n
    return np.tensordot(lambdas, Phi_x * Phi_y, axes=1)


def l2_error_quad(cov_exact, lambdas, phis, n_pts=200):
    """
    cov_exact : callable C(x, y)
    lambdas, phis, N : as above
    n_pts     : # points in each direction for Gauss–Legendre
    """
    # 1-D Gauss–Legendre nodes/weights on [-1,1]
    x, w = gauss_legendre_rule(0.0, 1.0, n_pts)
    X, Y = np.meshgrid(x, x, indexing='ij')
    W = w[:, None] * w[None, :]  # tensor weights
    diff = cov_exact(X, Y) - truncated_covariance(X, Y,
                                                  lambdas, phis)
    integral = np.sum(W * diff**2)
    return np.sqrt(integral)


def l2_error_tri_quad(cov_exact, lambdas, phis, n_outer=80, n_inner=80):
    """
    Spectral-accuracy quadrature of
        ‖C – C_N‖_L2((0,1)²)  where C_N is a KL truncation,
    by splitting the unit square into the two right triangles
        T₁ = {(x,y) : 0≤y≤x≤1}   (below the diagonal)
        T₂ = {(x,y) : 0≤x≤y≤1}   (above the diagonal).

    Parameters
    ----------
    cov_exact  : callable  C(x, y)
    lambdas    : (M,) eigenvalues
    phis       : list/array of eigenfunctions
    n_outer    : # Gauss points in the *outer* integral (x or y)
    n_inner    : # Gauss points in the *inner* integral (variable upper limit)

    Returns
    -------
    float   Hilbert–Schmidt (L²) error  ‖C − C_N‖
    """
    # 1-D Gauss–Legendre rule on (0,1)
    x_nodes, x_wts = gauss_legendre_rule(0.0, 1.0, n_outer)
    t_nodes, t_wts = gauss_legendre_rule(0.0, 1.0, n_inner)   # param 0…1

    # Pre-allocate the inner-quadrature work arrays
    T = t_nodes[None, :]                 # shape (1, n_inner)
    WT = t_wts[None, :]  # "

    # ------------------------------------------------------------------
    # Triangle T₁  :  ∫₀¹ dx ∫₀ˣ dy  f(x,y)
    # ------------------------------------------------------------------
    X = x_nodes[:, None]                 # broadcast (n_outer, 1)
    WX = x_wts[:, None]

    Y = X * T                            # y = t * x
    J = X                                # Jacobian |∂y/∂t| = x

    diff_T1 = cov_exact(X, Y) - truncated_covariance(X, Y, lambdas, phis)
    int_T1 = np.sum(WX * J * np.sum(WT * diff_T1**2, axis=1, keepdims=True))

    # ------------------------------------------------------------------
    # Triangle T₂  :  swap roles of x and y
    # ------------------------------------------------------------------
    Y = x_nodes[:, None]
    X = Y * T
    J = Y

    diff_T2 = cov_exact(X, Y) - truncated_covariance(X, Y, lambdas, phis)
    int_T2 = np.sum(WX * J * np.sum(WT * diff_T2**2, axis=1, keepdims=True))

    return np.sqrt(int_T1 + int_T2)
