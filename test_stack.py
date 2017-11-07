import numpy as np
import pytest

import stack

def test_multivariate_complex_normal():
    n_dim = 3
    n_samples = 100
    cov = np.eye(n_dim)
    samples = stack.multivariate_complex_normal(cov, n_samples)

    assert samples.shape == (n_samples, n_dim)
    assert samples.dtype == np.complex


def test_amp_phi_coh2cov_exceptions():

    # test for amp and phi vectors
    with pytest.raises(ValueError):
        amp = np.ones((3, 2))
        coh = np.ones((amp.size, amp.size))
        phi = np.ones(4)
        stack.amp_phi_coh2cov(amp, phi, coh)

    # test for matching sizes
    with pytest.raises(ValueError):
        amp = np.ones(3)
        coh = np.ones((amp.size, amp.size))
        phi = np.ones(4)
        stack.amp_phi_coh2cov(amp, phi, coh)

    # test for quadratic coherence
    with pytest.raises(ValueError):
        amp = np.ones(3)
        coh = np.ones((3, 4))
        phi = np.ones(4)
        stack.amp_phi_coh2cov(amp, phi, coh)


def test_amp_phi_coh2cov():
    n_dim = 3
    amp = np.ones(n_dim)
    phi = np.ones(n_dim)
    coh = np.eye(n_dim)

    cov = stack.amp_phi_coh2cov(amp, phi, coh)

    assert cov.shape == (n_dim, n_dim)
    assert cov.dtype == np.complex
    np.testing.assert_equal(cov, cov.T.conj())
