import json
import numpy as np
from . import galerkin, kl_tensor
from copy import deepcopy


def get_kl_decomp(path_decomp):
    print(f"Loading decomposition from {path_decomp}...")
    with open(path_decomp, "r") as f:
        squared_approx = json.load(f)

    all_a, all_b, all_vals = [], [], []
    # available decomposition sizes
    for id, decomp in squared_approx.items():
        all_a.append(decomp["a"])
        all_b.append(decomp["b"])
        all_vals.append(decomp["val"])
    return all_a, all_b, all_vals


def get_decomp_at_tol(all_decomp, tol):
    """
    Get the decomposition at a given tolerance.
    """
    all_a, all_b, all_vals = all_decomp["all_a"], all_decomp["all_b"], all_decomp["all_vals"]
    for i, val in enumerate(all_vals):
        if val < tol:
            return all_a[i], all_b[i], all_vals[i]
    return all_a[-1], all_b[-1], all_vals[-1]


def get_kl_decomp_gal(b_coeff, comp_degree=100, quad=600, g=4.0):
    A_all = []
    for b in b_coeff:
        print(f"Processing {b=:.3e}.")
        if b < 1e5:
            A_all.append(galerkin.assemble_gauss2d(
                         f=b, degree=comp_degree, quad=quad * 5))
        else:
            A_all.append(galerkin.assemble_duffy(f=b, degree=comp_degree, quad=quad, gx=g))
    return A_all


def get_kl_decomp_from_gal_to_operators(a_coeff, A_all, comp_degree=100, N_eval=1000):
    A_all_copy = deepcopy(A_all)
    for idx, A in enumerate(A_all_copy):
        A_all_copy[idx] = A[:comp_degree, :comp_degree]

    eigenvalues, eigenvectors = kl_tensor.spectral_blocks(A_all_copy, a_coeff, 1, N_eval=N_eval)
    eigenvalues, eigenvectors = kl_tensor.evaluate_eigenfunctions_1d(eigenvalues, eigenvectors, max_degree=comp_degree)
    return eigenvalues, eigenvectors


def to_latex_sci(x):
    """Format number in LaTeX scientific notation with 4 significant digits."""
    if x == 0:
        return r"$0$"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / 10**exponent
    return rf"${mantissa:.3f}\cdot 10^{{{exponent}}}$"
