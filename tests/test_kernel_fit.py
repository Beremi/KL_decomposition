import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kl_decomposition import rectangle_rule, fit_exp_sum


class TestKernelFit(unittest.TestCase):
    def test_fit_single_exp(self):
        x, w = rectangle_rule(0.0, 2.0, 50)
        f = lambda t: 2.0 * np.exp(-3.0 * t ** 2)
        a, b = fit_exp_sum(1, x, w, f, method="de")
        self.assertTrue(np.allclose(a, 2.0, rtol=1e-2, atol=1e-2))
        self.assertTrue(np.allclose(b, 3.0, rtol=1e-2, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
