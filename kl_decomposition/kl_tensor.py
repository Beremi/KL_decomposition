"""spectral_decomp.py
====================

Spectral decomposition of checker‑board tensor–product operators
----------------------------------------------------------------

Let
    A = Σ_{k=1}^{M} g_k ⊗_{d=1}^{n} A_k ,
where every local matrix *A_k* is the **same** along all tensor
modes *d* and has a checker‑board pattern \\(A_k =\begin{pmatrix}E_k & 0\\0 & O_k\\end{pmatrix}\\)
after permuting indices `(0,2,4, … | 1,3,5, …)`.

With that permutation the global matrix becomes block–diagonal with
`2**n` parity sectors.  The routines below

* build a `scipy.sparse.linalg.LinearOperator` for **each** block without
  ever materialising the huge Kronecker product,
* extract the *N* largest eigenpairs per block, **sorted descendingly**
  across *all* sectors, and
* evaluate the resulting eigenfunctions on an arbitrary tensor grid
  using shifted, orthonormal Legendre polynomials (`leg_vals`).

The public API consists of

```python
spectral_blocks( … )          # Lanczos diagonalisation per block
evaluate_eigenfunctions( … )  # ψ_j(x_1,…,x_n) on a tensor grid
```
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

# local import for 1‑D basis functions -------------------------------------------------------------
from .orthopoly import leg_vals  # noqa: F401 (project‑local module)

__all__ = [
    "spectral_blocks",
    "evaluate_eigenfunctions",
]


# ================================================================================================ #
# helper: left‑multiplication of a tensor mode
# ================================================================================================ #

def _left_mult_axis(X: np.ndarray, M: np.ndarray, axis: int) -> np.ndarray:
    """Left‑multiply *axis* of *X* by matrix *M*.

    Parameters
    ----------
    X
        *n*-way array.
    M
        Two‑dimensional matrix with shape ``(d_out, d_in)`` so that
        ``X.shape[axis] == d_in``.
    axis
        Which axis of *X* to transform.

    Returns
    -------
    Y
        New tensor with ``Y.shape[axis] == d_out``.
    """
    X_move = np.moveaxis(X, axis, 0)            # (d_in, …)
    Y = np.einsum("ij,j...->i...", M, X_move)   # contraction along first index
    return np.moveaxis(Y, 0, axis)


# ================================================================================================ #
# checker‑board split of a *local* matrix
# ================================================================================================ #

def _even_odd_blocks(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the even–even and odd–odd dense blocks of *A*.

    Assumes the index order ``0,2,4,… | 1,3,5,…``.
    """
    even_idx = np.arange(0, A.shape[0], 2)
    odd_idx = np.arange(1, A.shape[0], 2)
    return A[np.ix_(even_idx, even_idx)], A[np.ix_(odd_idx, odd_idx)]


def _preprocess_blocks(A_all: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Split every *A_k* into *(E_k, O_k).*"""
    even_blocks, odd_blocks = [], []
    for A in A_all:
        EvenBlocks, OddBlocks = _even_odd_blocks(A)
        even_blocks.append(EvenBlocks)
        odd_blocks.append(OddBlocks)
    return even_blocks, odd_blocks


# ================================================================================================ #
# build a *single* parity‑block operator
# ================================================================================================ #

def _make_block_operator(
    bits: Tuple[int, ...],
    even_blocks: Sequence[np.ndarray],
    odd_blocks: Sequence[np.ndarray],
    g_coeff: Sequence[float],
) -> LinearOperator:
    """LinearOperator representing one parity block *A_p*.

    Parameters
    ----------
    bits
        Parity pattern ``(0|1, …, 0|1)`` of length *n*.
    even_blocks / odd_blocks
        Dense matrices ``E_k`` / ``O_k`` (`len == M`).
    g_coeff
        Scalars ``g_k`` (`len == M`).

    Returns
    -------
    op
        A Hermitian `LinearOperator` acting on the block of dimension
        ``np.prod(dims)``, where ``dims[d] = E_k.shape[0]`` if
        ``bits[d] == 0`` and ``O_k.shape[0]`` otherwise.
    """
    M = len(g_coeff)
    d_even, d_odd = even_blocks[0].shape[0], odd_blocks[0].shape[0]
    dims = [d_even if b == 0 else d_odd for b in bits]
    size = int(np.prod(dims))

    # ------------------------------------------------------------------------------------------ #
    def matvec(x: np.ndarray) -> np.ndarray:  # noqa: D401 – short name is fine here
        X = x.reshape(dims)                  # vector → rank‑n tensor
        Y = np.zeros_like(X)
        for k in range(M):
            mats = [even_blocks[k] if b == 0 else odd_blocks[k] for b in bits]
            T = X
            for axis, Mk in enumerate(mats):
                T = _left_mult_axis(T, Mk, axis)
            Y += g_coeff[k] * T
        return Y.ravel()

    return LinearOperator(
        shape=(size, size),
        matvec=matvec,
        rmatvec=matvec,   # Hermitian ⇒ same action
        dtype=even_blocks[0].dtype,
    )


# ================================================================================================ #
# public 1 / 2 :  Lanczos diagonalisation per block
# ================================================================================================ #

def spectral_blocks(
    A_all: Sequence[np.ndarray],
    g_coeff: Sequence[float],
    n: int,
    N_eval: int = 6,
) -> Tuple[Dict[Tuple[int, ...], np.ndarray], Dict[Tuple[int, ...], np.ndarray]]:
    """Diagonalise **every** parity block and return eigenpairs.

    Parameters
    ----------
    A_all
        The *M* local checker‑board matrices ``A_k``.
    g_coeff
        The coefficients ``g_k`` appearing in the definition of *A*.
    n
        Tensor order — i.e. the number of modes/sites.
    N_eval
        How many *largest* eigenvalues/eigenvectors per block to compute.

    Returns
    -------
    eigvals, eigvecs
        Two dictionaries keyed by the parity tuple ``bits``:

        * ``eigvals[bits]`` – ``(N_eval,)`` array, **descending** order.
        * ``eigvecs[bits]`` – matrix ``(block_dim, N_eval)`` whose columns
          are the matching eigenvectors.
    """
    even_blocks, odd_blocks = _preprocess_blocks(A_all)
    eigvals: Dict[Tuple[int, ...], np.ndarray] = {}
    eigvecs: Dict[Tuple[int, ...], np.ndarray] = {}

    for bits in product((0, 1), repeat=n):
        op = _make_block_operator(bits, even_blocks, odd_blocks, g_coeff)
        k = min(N_eval, op.shape[0] - 1)  # eigsh requires k < dim
        w, v = eigsh(op, k=k, which="LA")
        idx = np.argsort(w)[::-1]          # descending
        eigvals[bits] = w[idx]
        eigvecs[bits] = v[:, idx]

    return eigvals, eigvecs


# ================================================================================================ #
# public 2 / 2 :  Evaluate eigenfunctions on a tensor grid
# ================================================================================================ #

def evaluate_eigenfunctions(
    eigvals: Dict[Tuple[int, ...], np.ndarray],
    eigvecs: Dict[Tuple[int, ...], np.ndarray],
    grid_axes: Sequence[np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Evaluate every eigenfunction on a tensor grid (vectorised).

    The algebraic eigenvectors in *eigvecs* are first reshaped into
    tensors of shape ``dims + (N_eval,)`` per parity block and *all*
    eigenfunctions of the block are projected onto the Legendre basis in
    **one** `numpy.einsum` call.  This is both clearer and faster than a
    per‑vector loop.
    """

    n = len(grid_axes)

    # --- infer local basis sizes ---------------------------------------- #
    size_even_block = next(v.shape[0] for bits, v in eigvecs.items() if all(b == 0 for b in bits))
    d_even = int(round(size_even_block ** (1 / n)))

    size_odd_block = next(v.shape[0] for bits, v in eigvecs.items() if all(b == 1 for b in bits))
    d_odd = int(round(size_odd_block ** (1 / n)))

    d_max = max(d_even, d_odd)

    # --- pre‑compute Legendre basis values ------------------------------ #
    phi_full: List[np.ndarray] = [leg_vals(d_max, x) for x in grid_axes]  # (d_max, N_x)
    phi_even: List[np.ndarray] = [phi[:d_even] for phi in phi_full]
    phi_odd: List[np.ndarray] = [phi[:d_odd] for phi in phi_full]

    # index labels for einsum ------------------------------------------- #
    import string

    coeff_idx = string.ascii_lowercase[:n]  # a, b, c, …  for coefficient axes
    grid_idx = string.ascii_lowercase[n:2 * n]  # u, v, w, …  for grid axes
    vec_idx = 'z'  # eigenvector index (last axis)

    # template:  "abc z, aU, bV, cW -> UVW z"  (for n=3)
    einsum_rhs = (
        f"{''.join(coeff_idx)}{vec_idx}, " +
        ", ".join(f"{c}{g}" for c, g in zip(coeff_idx, grid_idx)) +
        f" -> {''.join(grid_idx)}{vec_idx}"
    )

    # -------------------------------------------------------------------- #
    lambdas: List[float] = []
    psi_grid: List[np.ndarray] = []

    for bits, w_block in eigvals.items():
        v_block = eigvecs[bits]                 # (block_dim, N_eval)
        N_eval_here = v_block.shape[1]

        dims = tuple(d_even if b == 0 else d_odd for b in bits)
        coeff_tensor = v_block.reshape(*dims, N_eval_here)  # dims + (N_eval,)

        # pick the right 1‑D bases for this parity pattern
        phi_list = [phi_even[k] if bit == 0 else phi_odd[k] for k, bit in enumerate(bits)]

        # einsum: (dims,N_eval) × Π_k (d_k, N_xk)  → (grid..., N_eval)
        psi_all = np.einsum(einsum_rhs, coeff_tensor, *phi_list, optimize=True)
        # move eigenvector axis to front? keep at end for consistency

        lambdas.extend(w_block.tolist())
        # split the last axis into individual eigenfunctions
        for i in range(N_eval_here):
            psi_grid.append(psi_all[..., i])

    # --- global descending sort ----------------------------------------- #
    order = np.argsort(lambdas)[::-1]
    Λ = np.asarray(lambdas)[order]
    Ψ = [psi_grid[i] for i in order]

    return Λ, Ψ
