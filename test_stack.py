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
    np.testing.assert_equal(np.diag(coh), np.ones(M))


def test_too_close():

    outl = (3, 4, 5)
    min_dis = (0, 2, 2)
    prev_outls = [
        (0, 0, 0),
        (3, 8, 9),
        (3, 5, 9),
    ]

    assert stack.too_close(outl, prev_outls, min_dis)
    assert not stack.too_close(outl, prev_outls[:-1], min_dis)


def test_gen_outliers():
    stack_shape = (5, 4, 2)
    amp = 1
    size = 3

    outliers, coords = stack.gen_outliers(stack_shape, amp, size)

    assert len(outliers) == len(coords) == size

    np.testing.assert_array_almost_equal(np.abs(outliers), np.ones(size))

    assert all(c >= (0, 0, 0) for c in coords)
    assert all(c < stack_shape for c in coords)


def test_add_outliers2stack():
    stk = np.zeros((1, 4, 5), dtype=np.complex)
    coords = [(0, 1, 2), (0, 3, 0)]
    outliers = [1, 1]
    selem = np.ones((1, 3, 3))

    des_stk = np.array([[[0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [1, 2, 1, 1, 0],
                         [1, 1, 0, 0, 0]]])

    np.testing.assert_array_equal(stack.add_outliers2stack(stk, outliers, coords, selem), des_stk)
