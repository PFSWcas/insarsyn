# formulae for coherences taken from
# H. A. Zebker and J. Villasenor, "Decorrelation in interferometric radar
# echoes," in IEEE Transactions on Geoscience and Remote Sensing, vol. 30, no.
# 5, pp. 950-959, Sep 1992.

def coh_thermal(signal_power, noise_power):
    """thermal coherence due to signal to noise ratio"""
    return signal_power/(signal_power+noise_power)
