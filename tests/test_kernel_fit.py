import numpy as np
from kl_decomposition import rectangle_rule, fit_exp_sum


def test_fit_single_exp():
    x, w = rectangle_rule(0.0, 2.0, 50)
    f = lambda t: 2.0 * np.exp(-3.0 * t ** 2)
    a, b = fit_exp_sum(1, x, w, f)
    assert np.allclose(a, 2.0, rtol=1e-2, atol=1e-2)
    assert np.allclose(b, 3.0, rtol=1e-2, atol=1e-2)
