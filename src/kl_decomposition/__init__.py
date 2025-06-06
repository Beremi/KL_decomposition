"""KL decomposition utilities."""

from .orthopoly import gauss_legendre_rule
from .kernel_fit import (
    rectangle_rule,
    fit_exp_sum,
    fit_exp_sum_sorted,
    OptimiserOptions,
    DEStats,
    NewtonStats,
    newton_with_line_search,
    bisection_line_search,
)
from .galerkin import (
    assemble_block,
    assemble_duffy,
    assemble_gauss2d,
    assemble_rectangle,
    convergence_vs_ref,
)
from .exp_kernel import exp_kernel_eigen

__all__ = [
    "rectangle_rule",
    "gauss_legendre_rule",
    "fit_exp_sum",
    "fit_exp_sum_sorted",
    "OptimiserOptions",
    "DEStats",
    "NewtonStats",
    "newton_with_line_search",
    "bisection_line_search",
    "assemble_block",
    "assemble_duffy",
    "assemble_gauss2d",
    "assemble_rectangle",
    "convergence_vs_ref",
    "exp_kernel_eigen",
]
