
# KL_decomposition

**Efficient Karhunen-Loève (KL) decomposition for isotropic Gaussian random fields on multi-dimensional boxes.**

This repository turns the algorithmic framework described in Béreš (2023) into ready-to-run Python code.  
The package fits a low-rank separable approximation of the covariance kernel, assembles Galerkin matrices in compressed (Kronecker) format, and computes the leading KL modes with an iterative eigen decomposition.

This repository acompanies the paper:
Béreš, M., Legendre polynomials and their use for Karhunen--Loève expansion, *Applications of Mathematics* (submitted), 2025

---

## Key features

* **Low-rank kernel fit** using JAX-based automatic differentiation in Newton optimiser with special initial guess.
* **Memory-lean Galerkin assembly** – dense matrix of size $(n+1)^{2D}$ is replaced by a sum of $k$ Kronecker products of $D$ blocks of size $(n+1)^{2}$.
* **Tensor Krylov eigensolver** that touches the full matrix only through `matmul`, enabling 2-D and 3-D KL decompositions on a laptop.

---

## Repository structure

```
kl_decomposition/     Python package with algorithms
├── kernel_fit.py     – optimise squared–exponential kernel fits
├── galerkin.py       – assemble 1‑D Galerkin blocks
├── kl_tensor.py      – tensor‑product eigendecomposition
├── orthopoly.py      – Legendre basis utilities
├── exp_kernel.py     – analytic eigenpairs of exp(|x−y|) kernel
├── error_estimates.py– helper error norms
└── misc.py           – convenience wrappers
data/                 Precomputed kernel fits (JSON files)
*.ipynb               Demonstration notebooks
```

---

## Computation pipeline

### 1 Approximation of the autocovariance

The isotropic kernel is fitted in 1-D distance $d=\lVert\mathbf x-\mathbf y\rVert$:

![img](https://latex.codecogs.com/svg.image?{\color{Gray}C(d)\approx\sum_{i=1}^{k}%20a_i%20e^{-b_i%20d^{2}},})

with $a_i,b_i>0$ optimised by Newton method with special initial guess.

### 2 Galerkin matrix assembly
The Galerkin matrix is assembled in a separable form

![img](https://latex.codecogs.com/svg.image?{\color{Gray}A_{n}=\sum_{i=1}^{k}a_{i}\bigotimes_{l=1}^{D}\mathbf{A}_{n,i}^{(l)}})


Where all 1-D blocks are given by

![img](https://latex.codecogs.com/svg.image?{\color{Gray}\bigl(\mathbf{A}_{n,i}^{(l)}\bigr)_{\beta_{l}\alpha_{l}}=\int_{a_{l}}^{b_{l}}\int_{a_{l}}^{b_{l}}e^{-b_{i}(x-y)^{2}}\varphi_{\alpha_{l}}^{(l)}(y)\varphi_{\beta_{l}}^{(l)}(x)\,\mathrm{d}y\,\mathrm{d}x})



using shifted, $L^{2}$-orthonormal Legendre polynomials $\phi^{(l)}_{\alpha_l}$ so that the mass matrix is the identity.

#### 2.1 Fast quadrature for very large $b_k$

The integrals become numerically challenging when $b_i\gg1$: the Gaussian ridge collapses onto the
diagonal $x=y$ and a naive Gauss–Legendre rule needs
$\mathcal O(\sqrt{b_i})$ points to see it.  
We keep the cost **independent of** $b_i$ with two tricks.  

1. **Duffy‐type mapping**  
   * Split the square $[a,b]^2$ along its diagonal into two triangles.  
   * Map each triangle back to the unit square so that the diagonal becomes a
     coordinate axis:

        ![img](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7D%28x%2Cy%29%5Clongrightarrow%28x%2Cy%29%3D%28%5Ctilde%20x%2C%281-%5Ctilde%20y%29%5Ctilde%20x%29%5Cquad%5Ctext%7Bor%7D%5Cquad%28x%2Cy%29%3D%281-%5Ctilde%20x%2C%28%5Ctilde%20y-1%29%5Ctilde%20x%2B1%29.%7D)

     The kernel transforms to $\tilde xe^{-b_i\tilde x^{2}\tilde y^{2}}$ and no longer sharply aligned with the diagonal.

2. **Adaptive stretching**  
   * To keep the remaining peak resolvable with a *fixed* Gauss–Legendre grid we stretch the coordinates with a simple power transformation:

        ![img](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7D%5Chat%7Bx%7D%20%3D%20%5Ctilde%7Bx%7D%5Eg%2C%20%5Cquad%20%5Chat%7By%7D%20%3D%20%5Ctilde%7By%7D%5Eg%2C%20%5Cquad%20g%20%5Cge%201.%20%5Ctag%7B2%7D%7D)


---

## Dependencies

* **JAX** – gradient-based kernel fit
* **NumPy & SciPy** – matrix assembly, quadrature, linear algebra

---

## Quickstart

```python
import numpy as np
from kl_decomposition import kernel_fit, galerkin, kl_tensor

# 1) squared exponential fit of exp(-d**2)
d = np.linspace(0.0, 3.0, 200)
w = np.full_like(d, d[1]-d[0])
target = np.exp(-d**2)
obj, grad, hess = kernel_fit.build_se_obj_grad_hess(d, w, target)
a_all, b_all, _ = kernel_fit.square_exp_approximations_newton(
    max_terms=3,
    precision=1e-12,
    x=d,
    w=w,
    target=target,
    obj=obj,
    grad=grad,
    hess=hess,
)
a_coeff = a_all[-1]
b_coeff = np.exp(b_all[-1])

# 2) Galerkin blocks (1‑D example)
degree = 40
A_all = [galerkin.assemble_block((0.0, 1.0), b, degree) for b in b_coeff]

# 3) eigen-decomposition and values on a grid
eigvals, eigvecs = kl_tensor.spectral_blocks(A_all, a_coeff, n=1)
grid = np.linspace(0.0, 1.0, 200)
lambdas, psi_grid = kl_tensor.evaluate_eigenfunctions(eigvals, eigvecs, degree, grid)
```

The lists `lambdas` and `psi_grid` contain the sorted eigenvalues and the corresponding eigenfunctions evaluated on `grid`.

---

## Demonstration notebooks

| Notebook | Purpose |
| --- | --- |
| `kernels_fit.ipynb` | Fit squared exponential approximations to several standard kernels and save them to `data/*.json`. |
| `plot_kernel_fits.ipynb` | Visualize the distribution of $b$ coefficients for all stored kernel fits. |
| `assembly_demo.ipynb` | Benchmark Duffy-mapped and Gauss-Legendre quadrature for assembling Galerkin blocks. |
| `kl_decomposition.ipynb` | Load precomputed fits and build KL decompositions in 2‑D. |
| `eigenfunctions_exp_cov.ipynb` | Show the eigenfunctions of a 2‑D exponential covariance kernel. |
| `exponential_kernel_eig.ipynb` | Demonstrate analytic eigenpairs of the 1‑D exponential kernel. |
| `error_covariance.ipynb` | Compare approximate KL modes with precise decompositions for exponential and Matérn kernels. |
| `residual_1D.ipynb` | Tabulate reference eigenvalues for 1‑D kernels and plot residual errors. |

---

## References

* M. Béreš, *Methods for the Solution of Differential Equations with Uncertainties in Parameters*, phd thesis, VŠB – Technical University of Ostrava, Ostrava, 2023.
