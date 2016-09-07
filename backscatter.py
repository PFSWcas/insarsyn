import numpy as np
from scipy.special import gamma


def incidence_angle(terrain, look_angle=30):
    """computes the radar incidence angle given its look angle and the terrain

    assumes the look angle is constant for the whole terrain
    """

    if terrain.ndim != 2:
        raise ValueError('terrain must be 2 dimensional')
    grad = np.gradient(terrain)[1]
    incidence_angle = np.arctan(-grad)+np.radians(look_angle)
    if np.any(incidence_angle < 0):
        raise ValueError('negative incidence angle would result in shadowing')
    return incidence_angle


def radar_cross_section(theta, k=120, H=0.3, T=100):
    """compute the radar cross section

    A simplified version of Equation (4.4) of "A Fractal-Based Theoretical
    Framework for Retrieval of Surface Parameters from Electromagnetic
    Backscattering Data"; Giorgioa Franceschetti et al.; IEEE TGRS 2000

    Parameters
    ----------
    theta : float numpy array
            incidence_angle
    k :     float
            wavenumber
    H :     float [0, 1]
            Hurst coefficient/exponent
    T :     float
            topothesy

    """
    def calc_S0(H, T):
        return H*T**(2*(1-H))*2**(2*H)*gamma(1+H)/gamma(1-H)

    S0 = calc_S0(H, T)
    # suppress warnings if theta is equal to zero and set the radar cross
    # section to 1
    zeros = theta == 0
    theta[zeros] = np.finfo(np.float32).eps
    rcs = np.cos(theta)**4*S0/(2*k*np.sin(theta))**(2+2*H)
    rcs[zeros] = 1
    return rcs
