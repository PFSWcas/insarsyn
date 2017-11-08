import numpy as np
from scipy.stats import multivariate_normal


def multivariate_complex_normal(cov, size=1):
    """ multivariate complex normal distribution """

    n_dim = cov.shape[0]

    # complex isomorphism
    cov_real = np.kron(np.eye(2), cov.real) + np.kron(
        np.array([[0, -1], [1, 0]]), cov.imag)
    cov_real *= 0.5
    xy = multivariate_normal.rvs(cov=cov_real, size=size)
    if xy.ndim == 1:
        return xy[:n_dim] + 1j * xy[-n_dim:]
    else:
        return xy[:, :n_dim] + 1j * xy[:, -n_dim:]


def amp_phi_coh2cov(amp, phi, coh):
    """ creates a covariance matrix from given amplidute and phase vectors.
    Coherence is a quadratic matrix with matching dimensions.

    """

    if amp.ndim != 1 or phi.ndim != 1:
        raise ValueError('amp and phi must be one dimensional')
    if coh.ndim != 2 or coh.shape[0] != coh.shape[1]:
        raise ValueError(
            'coh must be two dimensional and quadratic. Shape is {}'.format(
                coh.shape))
    if amp.size != phi.size or coh.shape[0] != amp.size:
        raise ValueError('dimension mismatch')

    slc = amp * np.exp(1j * phi)

    return coh * np.outer(slc, slc.conj())


def amps_phis_cohs2covs(amps, phis, cohs):
    """ creates covariance matrices for a stack of SAR SLCs defined by amps, phis and cohs """

    if amps.ndim != 3 or phis.ndim != 3:
        raise ValueError('amp and phi must be three dimensional')
    if cohs.ndim != 4 or cohs.shape[0] != cohs.shape[1]:
        raise ValueError(
            'coh must be four dimensional and first two dimensions must have equal size. Shape is {}'.
            format(cohs.shape))
    if amps.shape != phis.shape or cohs.shape[1:] != amps.shape:
        raise ValueError('dimension mismatch')

    covs = np.empty(cohs.shape, dtype=np.complex)

    for x in range(amps.shape[1]):
        for y in range(amps.shape[2]):
            covs[:, :, x, y] = amp_phi_coh2cov(amps[:, x, y], phis[:, x, y], cohs[:, :, x, y])

    return covs


def exp_decay_coh_mat(M, lbda):
    """ generates a coherence matrix with exponential decay """

    coh_vec = np.exp(-np.arange(0, M)*lbda)

    return np.outer(coh_vec, coh_vec.T)
