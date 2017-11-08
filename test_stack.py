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


def test_amps_phis_cohs2covs():

    shape = (7, 5, 3)

    amps = np.ones(shape)
    phis = np.ones(shape)
    cohs = np.ones((shape[0], *shape))

    covs = stack.amps_phis_cohs2covs(amps, phis, cohs)

    assert covs.shape == cohs.shape

    covs_swap = np.swapaxes(covs, 0, 1)
    np.testing.assert_equal(covs, covs_swap)


def test_exp_decay_coh_mat():
    M = 7

    coh = stack.exp_decay_coh_mat(M, 0.1)

    assert coh.shape == (M, M)
    np.testing.assert_equal(coh, coh.T)
    np.testing.assert_equal(coh[0], np.sort(coh[0])[::-1])
