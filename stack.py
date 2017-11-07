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
