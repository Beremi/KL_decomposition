"""KL decomposition utilities."""

from .kernel_fit import (
    rectangle_rule,
    gauss_legendre_rule,
    fit_exp_sum,
    fit_exp_sum_sorted,
    OptimiserOptions,
    DEStats,
    NewtonStats,
    newton_with_line_search,
    bisection_line_search,
)
from .galerkin import assemble_block

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
]
