
# KL_decomposition

**Efficient Karhunen-Loève (KL) decomposition for isotropic Gaussian random fields on multi-dimensional boxes.**

This repository turns the algorithmic framework described in Béreš (2023) into ready-to-run Python code.  
The package fits a low-rank separable approximation of the covariance kernel, assembles Galerkin matrices in compressed (Kronecker) format, and computes the leading KL modes with an iterative eigen decomposition.

---

## Key features

* **Low-rank kernel fit** using JAX-based automatic differentiation or a hybrid global/​local optimiser (cross-entropy + trust-region Newton).  
* **Memory-lean Galerkin assembly** – every dense \(m^{2d}\) block is replaced by a sum of \(M\) Kronecker products of \(m^{2}\) blocks.  
* **Tensor Krylov eigensolver** that touches the full matrix only through `matmul`, enabling 2-D and 3-D KL decompositions on a laptop.  
* Pure-Python, no compiled extensions.

---

## Computation pipeline

### 1 Approximation of the autocovariance

The isotropic kernel is fitted in 1-D distance $d=\lVert\mathbf x-\mathbf y\rVert$:

$$
C(d)\;\approx\;\sum_{k=1}^{M} a_k e^{-b_k d^{2}},
$$

with $a_k,b_k>0$ optimised by differential evolution + Newton search.

### 2 Galerkin matrix assembly

For each exponential term we form the 1-D block

$$
\bigl(A^{(j)}_{k}\bigr)_{i\ell}=\int_{a_j}^{b_j}\!\!\int_{a_j}^{b_j}
e^{-b_k(x-y)^{2}}\,
\phi^{(j)}_{i}(y)\,\phi^{(j)}_{\ell}(x)\,dy\,dx,
$$

using shifted, $L^{2}$-orthonormal Legendre polynomials $\phi^{(j)}_{i}$ so that the mass matrix is the identity .

#### 2.1 Fast quadrature for very large $b_k$

The integrals  
\[
(A_{k})_{i\ell}
=\int_{a}^{b}\!\!\int_{a}^{b}
e^{-\,b_k(x-y)^{2}}\,
\phi_i(y)\,\phi_\ell(x)\,dy\,dx
\]
become numerically challenging when \(b_k\gg1\): the Gaussian ridge collapses onto the
diagonal \(x=y\) and a naive Gauss–Legendre rule needs
\(\mathcal O(\sqrt{b_k})\) points to see it.  
We keep the cost **independent of** \(b_k\) with two tricks.  

1. **Duffy‐type mapping**  
   * Split the square \([a,b]^2\) along its diagonal into two triangles.  
   * Map each triangle back to the unit square so that the diagonal becomes a
     coordinate axis:
     \[
       (x,y)\;\longrightarrow\;
       (x,\;y)=(\tilde x,\,(1-\tilde y)\tilde x)
       \quad\text{or}\quad
       (x,y)=(1-\tilde x,\,( \tilde y-1)\tilde x+1).
     \]
     The kernel transforms to \(\tilde x\,e^{-\,b_k\tilde x^{2}\tilde y^{2}}\) and no longer sharply aligned with the diagonal.

2. **Adaptive stretching**  
   To keep the remaining peak resolvable with a *fixed* Gauss–Legendre grid we stretch the coordinates with a simple power transformation:

   \[
   \hat x = \tilde x^{\,n}, \qquad \hat y = \tilde y^{\,n}, \qquad n\in(0,\infty).
   \tag{2}
   \]



### 3 Tensor-compressed eigenpair computation

#### 3.1 Low-rank separable kernel

Because $e^{-b_k\lVert\mathbf x-\mathbf y\rVert^{2}}$ factorises over coordinates, the fitted kernel reads

$$
c(\mathbf x,\mathbf y)\;\approx\;
\sum_{k=1}^{M} a_k\prod_{j=1}^{d}e^{-b_k(x_j-y_j)^{2}}. \tag{1}
$$

#### 3.2 Kronecker-product Galerkin matrix

Substituting (1) and the tensor basis $\varphi_{(i_1,\dots,i_d)}=\prod_{j}\phi^{(j)}_{i_j}$ gives

$$
A\;=\;\sum_{k=1}^{M} a_k
\bigotimes_{j=1}^{d} A^{(j)}_{k}, \tag{2}
$$

a sum of $M$ Kronecker products of size $m^{d}\times m^{d}$ .

*Assembly cost:* $M\,d\,m^{2}$ 2-D integrals instead of $m^{2d}$ 2$d$-D ones.
*Mat-vec cost:* $O(M\,d\,m^{d+1})$ instead of $O(m^{2d})$; for $d=3,m=20,M=10$ this is a ×133 flop reduction .

#### 3.3 Iterative solver

A lightweight power-iteration + Rayleigh-Ritz routine to work entirely with the format (2). No dense matrix is ever formed.

---

## Dependencies

* **JAX** – gradient-based kernel fit
* **NumPy & SciPy** – matrix assembly, quadrature, linear algebra

---

## References

* M. Béreš, *Efficient Numerical Approximation of the Karhunen-Loève Decomposition* (PhD thesis, 2023) – Algorithmic foundations (§ 2.4)&#x20;
* R. Beylkin & G. Mohlenkamp, “Algorithms for numerical analysis in high dimensions,” *PNAS* 102-27 (2005).
* D. Kressner & C. Tobler, “Krylov subspace methods for tensor structured linear systems,” *SIAM J. Sci. Comput.*, 2011.

