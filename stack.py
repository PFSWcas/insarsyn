import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.linalg import toeplitz
from scipy.ndimage import convolve


def multivariate_complex_normal(cov, size=1):
    """ multivariate complex normal distribution

    :params cov: covariance matrix. Must be positive-definite
    :params size: number of samples
    """

    n_dim = cov.shape[0]

    # complex isomorphism
    cov_real = np.kron(np.eye(2), cov.real) + np.kron(
        np.array([[0, -1], [1, 0]]), cov.imag)
    cov_real *= 0.5

    xy = __multivariate_normal(cov_real, size)

    if xy.ndim == 1:
        return xy[:n_dim] + 1j * xy[-n_dim:]
    return xy[:, :n_dim] + 1j * xy[:, -n_dim:]


def __multivariate_normal(cov, size):
    # for some covariance matrices the cholesky decomposition may fail
    # in this case fall back to scipy's, i.e. numpy's routine, which relies
    # on singular value decomposition.
    # We try to use the Cholesky decomposition for performance reasons.
    # See https://github.com/numpy/numpy/pull/3938 for a dicussion.
    try:
        # Choleksy decomposition: COV = LL^T
        L = np.linalg.cholesky(cov)

        xy = norm.rvs(size=cov.shape[0]*size).reshape((-1, size))
        # so L @ xy has covariance LL^T = cov
        xy = np.dot(L, xy)

        # first dimension: number of samples:
        xy = xy.T

        if size == 1:
            xy = xy.flatten()

    except np.linalg.linalg.LinAlgError:
        xy = multivariate_normal.rvs(cov=cov, size=size)

    return xy


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
            covs[:, :, x, y] = amp_phi_coh2cov(amps[:, x, y], phis[:, x, y],
                                               cohs[:, :, x, y])

    return covs


def exp_decay_coh_mat(M, lbda):
    """ generates a coherence matrix with exponential decay """

    coh_vec = np.exp(-np.arange(0, M) * lbda)

    return toeplitz(coh_vec)


def too_close(new_outl, prev_outls, min_dis):
    """ returns true if any of the distances between

    new_outl to prev_outls is smaller than min_dis

    """

    for po in prev_outls:
        dis = (abs(x - y) for x, y in zip(new_outl, po))
        if any(x < y for x, y in zip(dis, min_dis)):
            return True
    return False


def gen_outliers(stack_shape, amp, size, min_dis=None):
    """ generates a list of outliers

    :params stack_shape: tuple, shape of the stack,
        i.e. interval for the outliers

    :returns: a tuple containing two lists: the outliers and their coordinates

    """
    outliers = []
    coords = []

    if min_dis is None:
        min_dis = (0, 0, 0)

    while len(outliers) < size:
        coord = tuple(
            np.random.randint(o, x - o) for x, o in zip(stack_shape, min_dis))
        if not too_close(coord, coords, min_dis):
            coords.append(coord)
            outliers.append(amp * np.exp(1j * np.random.uniform(-np.pi,
                                                                np.pi)))
    return outliers, coords


def add_outliers2stack(stack, outliers, coords, selem=np.ones((1, 3, 3))):
    outlier_stack = np.zeros_like(stack)
    for outl, coord in zip(outliers, coords):
        outlier_stack[coord] = outl

    outlier_stack_conv = convolve(outlier_stack.real, selem, mode='constant') + 1j*convolve(
        outlier_stack.imag, selem, mode='constant')


    mask = outlier_stack_conv != 0
    stack[mask] = outlier_stack_conv[mask]

    return stack
