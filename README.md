
# KL_decomposition

**Efficient Karhunen-Loève (KL) decomposition for isotropic Gaussian random fields on multi-dimensional boxes.**

This repository turns the algorithmic framework described in Béreš (2023) into ready-to-run Python code.  
The package fits a low-rank separable approximation of the covariance kernel, assembles Galerkin matrices in compressed (Kronecker) format, and computes the leading KL modes with an iterative eigen decomposition.

---

## Key features

* **Low-rank kernel fit** using JAX-based automatic differentiation or a hybrid global/​local optimiser (cross-entropy + trust-region Newton).  
* **Memory-lean Galerkin assembly** – every dense $m^{2d}$ block is replaced by a sum of $M$ Kronecker products of $m^{2}$ blocks.  
* **Tensor Krylov eigensolver** that touches the full matrix only through `matmul`, enabling 2-D and 3-D KL decompositions on a laptop.  
* Pure-Python, no compiled extensions.

---

## Computation pipeline

### 1 Approximation of the autocovariance

The isotropic kernel is fitted in 1-D distance $d=\lVert\mathbf x-\mathbf y\rVert$:

![img](https://latex.codecogs.com/svg.image?{\color{Gray}C(d)\approx\sum_{k=1}^{M}%20a_k%20e^{-b_k%20d^{2}},})

with $a_k,b_k>0$ optimised by differential evolution + Newton search.

### 2 Galerkin matrix assembly

For each exponential term we form the 1-D block

![img](https://latex.codecogs.com/svg.image?{\color{Gray}\left(A^{(j)}_{k}\right)_{i\ell}=\int_{a_j}^{b_j}\int_{a_j}^{b_j}e^{-b_k(x-y)^{2}}\phi^{(j)}_{i}(y)\phi^{(j)}_{\ell}(x)dydx})


using shifted, $L^{2}$-orthonormal Legendre polynomials $\phi^{(j)}_{i}$ so that the mass matrix is the identity .

Example usage::

    from kl_decomposition import assemble_block
    A = assemble_block((0.0, 1.0), b=1.0, n=3)

#### 2.1 Fast quadrature for very large $b_k$

The integrals  

![img](https://latex.codecogs.com/svg.image?{\color{Gray}(A_{k})_{i\ell}=\int_{a}^{b}\int_{a}^{b}e^{-b_k(x-y)^{2}}\phi_i(y)\phi_\ell(x)dydx})

become numerically challenging when $b_k\gg1$: the Gaussian ridge collapses onto the
diagonal $x=y$ and a naive Gauss–Legendre rule needs
$\mathcal O(\sqrt{b_k})$ points to see it.  
We keep the cost **independent of** $b_k$ with two tricks.  

1. **Duffy‐type mapping**  
   * Split the square $[a,b]^2$ along its diagonal into two triangles.  
   * Map each triangle back to the unit square so that the diagonal becomes a
     coordinate axis:

        ![img](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7D%28x%2Cy%29%5Clongrightarrow%28x%2Cy%29%3D%28%5Ctilde%20x%2C%281-%5Ctilde%20y%29%5Ctilde%20x%29%5Cquad%5Ctext%7Bor%7D%5Cquad%28x%2Cy%29%3D%281-%5Ctilde%20x%2C%28%5Ctilde%20y-1%29%5Ctilde%20x%2B1%29.%7D)

     The kernel transforms to $\tilde xe^{-b_k\tilde x^{2}\tilde y^{2}}$ and no longer sharply aligned with the diagonal.

2. **Adaptive stretching**  
   To keep the remaining peak resolvable with a *fixed* Gauss–Legendre grid we stretch the coordinates with a simple power transformation:

![img](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7D%5Chat%20x%20%3D%20%5Ctilde%20x%5E%7Bn%7D%2C%20%5Cqquad%20%5Chat%20y%20%3D%20%5Ctilde%20y%5E%7Bn%7D%2C%20%5Cqquad%20n%5Cin%280%2C%5Cinfty%29.%20%5Ctag%7B2%7D%7D)




### 3 Tensor-compressed eigenpair computation

#### 3.1 Low-rank separable kernel

Because $e^{-b_k\lVert\mathbf x-\mathbf y\rVert^{2}}$ factorises over coordinates, the fitted kernel reads

![img](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7Dc%28%5Cmathbf%20x%2C%5Cmathbf%20y%29%5Capprox%0A%5Csum_%7Bk%3D1%7D%5E%7BM%7D%20a_k%5Cprod_%7Bj%3D1%7D%5E%7Bd%7De%5E%7B-b_k%28x_j-y_j%29%5E%7B2%7D%7D.%20%5Ctag%7B1%7D%7D)


#### 3.2 Kronecker-product Galerkin matrix

Substituting (1) and the tensor basis $\varphi_{(i_1,\dots,i_d)}=\prod_{j}\phi^{(j)}_{i_j}$ gives

![img](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BGray%7D%20A%3D%5Csum_%7Bk%3D1%7D%5E%7BM%7D%20a_k%20%5Cbigotimes_%7Bj%3D1%7D%5E%7Bd%7D%20A%5E%7B%28j%29%7D_%7Bk%7D%2C%20%5Ctag%7B2%7D%7D)


a sum of $M$ Kronecker products of size $m^{d}\times m^{d}$ .

*Assembly cost:* $Mdm^{2}$ 2-D integrals instead of $m^{2d}$ 2$d$-D ones.
*Mat-vec cost:* $O(Mdm^{d+1})$ instead of $O(m^{2d})$; for $d=3,m=20,M=10$ this is a ×133 flop reduction .

#### 3.3 Iterative solver

A lightweight power-iteration + Rayleigh-Ritz routine to work entirely with the format (2). No dense matrix is ever formed.

---

## Dependencies

* **JAX** – gradient-based kernel fit
* **NumPy & SciPy** – matrix assembly, quadrature, linear algebra

---

## References

* M. Béreš, *Efficient Numerical Approximation of the Karhunen-Loève Decomposition* (PhD thesis, 2023) – Algorithmic foundations.
* R. Beylkin & G. Mohlenkamp, “Algorithms for numerical analysis in high dimensions,” *PNAS* 102-27 (2005).
* D. Kressner & C. Tobler, “Krylov subspace methods for tensor structured linear systems,” *SIAM J. Sci. Comput.*, 2011.

