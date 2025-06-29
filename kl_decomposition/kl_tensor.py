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
from functools import reduce

from itertools import product
from typing import Dict, List, Sequence, Tuple, Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import eigh

# local import for 1‑D basis functions -------------------------------------------------------------
from .orthopoly import leg_vals  # noqa: F401 (project‑local module)

__all__ = [
    "spectral_blocks",
    "evaluate_eigenfunctions",
    "generate_raandom_sample",
    "evaluate_eigenfunctions_1d",
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


def _make_block_matrix(
    bits: Tuple[int, ...],
    even_blocks: Sequence[np.ndarray],
    odd_blocks: Sequence[np.ndarray],
    g_coeff: Sequence[float],
) -> np.ndarray:
    """
    Dense matrix representing one parity block.

    Parameters
    ----------
    bits
        Parity pattern (0|1,…,0|1) of length n.
    even_blocks / odd_blocks
        Dense matrices E_k / O_k (len == M).
    g_coeff
        Scalars g_k (len == M).

    Returns
    -------
    op_mat
        A (size × size) ndarray, where size = ∏_d dims[d], dims[d] =
        even_blocks[0].shape[0] if bits[d]==0 else odd_blocks[0].shape[0].
    """
    M = len(g_coeff)
    # dims for each tensor axis
    dims = [
        even_blocks[0].shape[0] if b == 0 else odd_blocks[0].shape[0]
        for b in bits
    ]
    size = int(np.prod(dims))

    # accumulate sum_k g_k * ⨂_d (E_k or O_k)
    op_mat = np.zeros((size, size), dtype=even_blocks[0].dtype)
    for k in range(M):
        # pick the correct block for each axis
        mats = [even_blocks[k] if b == 0 else odd_blocks[k] for b in bits]
        # form the Kronecker product of all mats using numpy.kron
        K = reduce(lambda A, B: np.kron(A, B), mats)
        op_mat += g_coeff[k] * K

    return op_mat


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
        k = min(N_eval, op.shape[0])  # eigsh requires k < dim
        if op.shape[0] == k:
            M = _make_block_matrix(bits, even_blocks, odd_blocks, g_coeff)
            w, v = eigh(M)             # returns (eigenvalues, eigenvectors)
        else:
            w, v = eigsh(op, k=k, which="LM")
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
    max_degree: int,
    grid: np.ndarray,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """General evaluator for arbitrary spatial dimension *n*.

    Assumes the **same 1‑D grid** (``grid``) and the **same polynomial
    degree``max_degree`` in every dimension.

    Parameters
    ----------
    eigvals / eigvecs
        Output of :func:`spectral_blocks` for arbitrary `n`.
    max_degree
        Highest Legendre degree per dimension (must cover even/odd
        blocks).  The local basis is split into even/odd subsets
        automatically.
    grid
        One‑dimensional array of evaluation points in ``[0,1]`` – shared
        by *all* dimensions.

    Returns
    -------
    Λ, Ψ
        Same layout as in the 1‑D and 2‑D helpers: a flat array of
        eigenvalues and a list of eigenfunction arrays, each shaped
        ``(len(grid),)*n``.
    """
    # -------------------------------------------------------------------- #
    # spatial dimension n inferred from any key in eigvals
    n = len(next(iter(eigvals)))

    # 1. Legendre basis (once)
    phi_all = leg_vals(max_degree, grid)        # (D, N)
    phi_even, phi_odd = phi_all[0::2], phi_all[1::2]
    d_even, d_odd = phi_even.shape[0], phi_odd.shape[0]

    # 2. Build einsum signature dynamically -------------------------------- #
    import string
    letters = string.ascii_letters  # 52 letters -> good up to n=25
    if 2 * n + 1 > len(letters):
        raise ValueError("Dimension too large for current einsum letter pool.")

    coeff_idx = letters[:n]                    # a, b, c, …
    grid_idx = letters[n:2 * n]                # u, v, w, …
    vec_idx = 'z'

    einsum_rhs = (
        f"{''.join(coeff_idx)}{vec_idx}, " +
        ', '.join(f"{c}{g}" for c, g in zip(coeff_idx, grid_idx)) +
        f" -> {''.join(grid_idx)}{vec_idx}"
    )

    # 3. Loop over parity blocks ------------------------------------------- #
    lambdas: List[float] = []
    psi_grid: List[np.ndarray] = []

    for bits, w_block in eigvals.items():
        v_block = eigvecs[bits]               # (block_dim, N_eval)
        N_eval_here = v_block.shape[1]

        dims = tuple(d_even if b == 0 else d_odd for b in bits)
        coeff_tensor = v_block.reshape(*dims, N_eval_here)

        phi_list = [(phi_even if bit == 0 else phi_odd) for bit in bits]

        psi_all = np.einsum(einsum_rhs, coeff_tensor, *phi_list, optimize=True)

        lambdas.extend(w_block.tolist())
        psi_grid.extend([psi_all[..., k] for k in range(N_eval_here)])

    # 4. global sort ------------------------------------------------------- #
    order = np.argsort(lambdas)[::-1]
    eigenvalues = np.asarray(lambdas)[order]
    eigenvectors_on_grid = [psi_grid[i] for i in order]

    return eigenvalues, eigenvectors_on_grid


def evaluate_eigenfunctions_1d(
    eigvals: Dict[Tuple[int], np.ndarray],
    eigvecs: Dict[Tuple[int], np.ndarray],
    max_degree: int
) -> Tuple[np.ndarray, List[Callable[[np.ndarray], np.ndarray]]]:
    """
    1-D version that returns ψ-functions rather than pre-evaluated grids.

    Parameters
    ----------
    eigvals / eigvecs : output of ``spectral_blocks`` for n = 1
        • keys are ``(0,)`` for the even block, ``(1,)`` for the odd block
        • ``eigvecs[(b,)].shape == (d_block, N_fns_block)``
    max_degree : int
        Highest Legendre degree used in the local basis.
    leg_vals : callable
        Same utility you already call in 2-D/3-D code:
            ``phi = leg_vals(max_degree, x)   # shape (max_degree+1, |x|)``
        It must return the *shifted* Legendre values on [0, 1].

    Returns
    -------
    Λ : (N,)  array
        Eigenvalues sorted from largest to smallest.
    Ψ : list[callable]
        One callable per eigenfunction.  Each ψ(x) accepts
        an arbitrary NumPy array ``x`` (any shape, broadcastable)
        with entries in [0, 1] and returns an array of the same shape.
    """
    D_tot = max_degree
    d_odd = max_degree // 2         # #even orders ≤ max_degree
    d_even = D_tot - d_odd              # #odd  orders ≤ max_degree

    lambdas: List[float] = []
    psi_list: List[Callable[[np.ndarray], np.ndarray]] = []

    for bits, w_block in eigvals.items():       # bits is (0,) or (1,)
        v_block = eigvecs[bits]                # (d_even|odd, N_block)
        parity = bits[0]                      # 0 = even, 1 = odd
        d_block = d_even if parity == 0 else d_odd
        assert v_block.shape[0] == d_block

        # --- create one closure per eigenfunction in this block ---------- #
        for k, λ in enumerate(w_block):
            coeff = v_block[:, k].copy()        # 1-D coefficients

            def _make_psi(coeff=coeff, parity=parity):
                """Return ψ(x) with coeff & parity bound."""
                def ψ(x: np.ndarray) -> np.ndarray:
                    x_arr = np.asarray(x)
                    flat = x_arr.ravel()                     # 1-D view

                    phi_all = leg_vals(max_degree, flat)     # (D_tot, M)
                    phi_sel = phi_all[0::2] if parity == 0 else phi_all[1::2]

                    res = coeff @ phi_sel                    # dot → (M,)
                    return res.reshape(x_arr.shape)

                ψ.__doc__ = f"Eigenfunction for Λ = {λ:.6e}, parity={parity}"
                return ψ

            psi_list.append(_make_psi())
            lambdas.append(float(λ))

    # -------- global sort by eigenvalue (descending) -------------------- #
    order = np.argsort(lambdas)[::-1]
    eigenvalues_sorted = np.asarray(lambdas)[order]
    eigenfunctions_sorted = [psi_list[i] for i in order]

    return eigenvalues_sorted, eigenfunctions_sorted


def generate_raandom_sample(lambdas_sorted, psi_sorted):
    sample = np.zeros_like(psi_sorted[0])
    for lambda_val, psi in zip(lambdas_sorted, psi_sorted):
        sample += np.random.randn() * np.sqrt(max(lambda_val, 0)) * psi
    return sample
