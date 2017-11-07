import numpy as np

import stack

def test_multivariate_complex_normal():
    n_dim = 3
    n_samples = 100
    cov = np.eye(n_dim)
    samples = stack.multivariate_complex_normal(cov, n_samples)

    assert samples.shape == (n_samples, n_dim)
    assert samples.dtype == np.complex
