import numpy as np
from scipy.stats import multivariate_normal

def multivariate_complex_normal(cov, size=1):
    """ multivariate complex normal distribution """

    n_dim = cov.shape[0]

    # complex isomorphism
    cov_real = np.kron(np.eye(2), cov.real) + np.kron(np.array([[0, -1], [1, 0]]), cov.imag)
    cov_real *= 0.5
    xy = multivariate_normal.rvs(cov=cov_real, size=size)
    if xy.ndim == 1:
        return xy[:n_dim] + 1j*xy[-n_dim:]
    else:
        return xy[:, :n_dim] + 1j*xy[:, -n_dim:]


def amp_phi_coh2cov(amp, phi, coh):
    """ creates a covariance matrix from given amplidute and phase vectors.
    Coherence is a quadratic matrix with matching dimensions.

    """

    if amp.ndim != 1 or phi.ndim != 1:
        raise ValueError('amp and phi must be one dimensional')
    if coh.ndim != 2 or coh.shape[0] != coh.shape[1]:
        raise ValueError('coh must be two dimensional and quadratic. Shape is {}'.format(coh.shape))
    if amp.size != phi.size or coh.shape[0] != amp.size:
        raise ValueError('dimension mismatch')

    slc = amp*np.exp(1j*phi)

    return coh * np.outer(slc, slc.conj())
