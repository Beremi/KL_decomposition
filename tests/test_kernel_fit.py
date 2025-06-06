import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kl_decomposition import rectangle_rule, fit_exp_sum
from kl_decomposition.kernel_fit import fit_exp_sum_sorted
from kl_decomposition.kernel_fit import _objective_jax_de
import jax.numpy as jnp


class TestKernelFit(unittest.TestCase):
    def test_fit_single_exp(self):
        x, w = rectangle_rule(0.0, 2.0, 50)
        f = lambda t: 2.0 * np.exp(-3.0 * t**2)
        a, b, _ = fit_exp_sum(1, x, w, f, method="de")
        self.assertTrue(np.allclose(a, 2.0, rtol=1e-2, atol=1e-2))
        self.assertTrue(np.allclose(b, 3.0, rtol=1e-2, atol=1e-2))

    def test_objective_jax_static_indexing(self):
        params = jnp.zeros(2)
        x = jnp.linspace(0.0, 1.0, 5)
        w = jnp.ones_like(x)
        target = jnp.exp(-3.0 * x**2)
        val = _objective_jax_de(params, x, target, w, 1)
        self.assertIsInstance(float(val.block_until_ready()), float)

    def test_de_newton_one_step_nocompile(self):
        x, w = rectangle_rule(0.0, 1.0, 20)
        f = lambda t: 1.0 * np.exp(-2.0 * t**2)
        a, b, _ = fit_exp_sum(1, x, w, f, method="de_newton1", compiled=False)
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 1)

    def test_de_ls(self):
        x, w = rectangle_rule(0.0, 2.0, 50)
        f = lambda t: 2.0 * np.exp(-3.0 * t**2) + 0.5 * np.exp(-1.0 * t**2)
        a, b, info = fit_exp_sum(
            2,
            x,
            w,
            f,
            method="de_ls",
            max_gen=10,
            pop_size=20,
        )
        self.assertEqual(len(a), 2)
        self.assertEqual(len(b), 2)
        self.assertLess(info.best_score, 1e-3)

    def test_de_ls_nocompile(self):
        x, w = rectangle_rule(0.0, 2.0, 30)
        f = lambda t: 2.0 * np.exp(-3.0 * t**2) + 0.5 * np.exp(-1.0 * t**2)
        a, b, _ = fit_exp_sum(
            2,
            x,
            w,
            f,
            method="de_ls",
            compiled=False,
            max_gen=5,
            pop_size=10,
        )
        self.assertTrue(np.all(np.isfinite(a)))
        self.assertTrue(np.all(np.isfinite(b)))

    def test_sorted_newton(self):
        x, w = rectangle_rule(0.0, 1.0, 20)
        f = lambda t: np.exp(-2.0 * t**2)
        a, b, _ = fit_exp_sum_sorted(
            1,
            x,
            w,
            f,
            max_gen=5,
            pop_size=10,
            n_newton=2,
        )
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 1)


if __name__ == "__main__":
    unittest.main()
