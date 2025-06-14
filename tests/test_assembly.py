import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kl_decomposition import (
    assemble_block,
    assemble_duffy,
    assemble_gauss2d,
    assemble_rectangle,
)


class TestAssembly(unittest.TestCase):
    def test_zero_b(self):
        A = assemble_block((0.0, 1.0), 0.0, 3)
        expected = np.zeros((3, 3))
        expected[0, 0] = 1.0
        self.assertTrue(np.allclose(A, expected, atol=1e-12))

    def test_positive_b(self):
        A = assemble_block((0.0, 1.0), 1.0, 3)
        self.assertTrue(np.allclose(A, A.T, atol=1e-12))
        eigs = np.linalg.eigvalsh(A)
        self.assertTrue(np.all(eigs > 0))

    def test_methods_agree(self):
        f = 0.7
        degree = 4
        q = 30
        A1 = assemble_duffy(f, degree, q)
        A2 = assemble_gauss2d(f, degree, q)
        A3 = assemble_rectangle(f, degree, 300)
        self.assertTrue(np.allclose(A1, A2, atol=1e-8))
        self.assertTrue(np.allclose(A1, A3, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
