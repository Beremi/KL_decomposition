import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kl_decomposition.orthopoly import legendre_table, shifted_legendre, gauss_legendre_rule
from numpy.polynomial import legendre as npleg

class TestOrthopoly(unittest.TestCase):
    def test_legendre_table_matches_numpy(self):
        x = np.linspace(0.0, 1.0, 7)
        table = legendre_table(5, x)
        t = 2.0 * x - 1.0
        for n in range(5):
            expected = np.sqrt(2 * n + 1) * npleg.Legendre.basis(n)(t)
            self.assertTrue(np.allclose(table[n], expected, atol=1e-14))

    def test_shifted_legendre_matches_numpy(self):
        a, b = -0.3, 1.7
        x = np.linspace(a, b, 9)
        for n in range(4):
            y = shifted_legendre(n, x, a, b)
            u = 2.0 * (x - a) / (b - a) - 1.0
            expected = np.sqrt((2 * n + 1) / (b - a)) * npleg.Legendre.basis(n)(u)
            self.assertTrue(np.allclose(y, expected, atol=1e-14))

    def test_gauss_legendre_rule_matches_numpy(self):
        a, b = -1.0, 2.0
        n = 6
        x, w = gauss_legendre_rule(a, b, n)
        xn, wn = np.polynomial.legendre.leggauss(n)
        xn = 0.5 * (b - a) * xn + 0.5 * (b + a)
        wn = 0.5 * (b - a) * wn
        self.assertTrue(np.allclose(x, xn, atol=1e-14))
        self.assertTrue(np.allclose(w, wn, atol=1e-14))

if __name__ == '__main__':
    unittest.main()
