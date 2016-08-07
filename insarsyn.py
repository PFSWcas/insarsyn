#!/usr/bin/python3

import numpy as np
from collections import OrderedDict

import patterns
import fractals


def wrap_phase(phi):
    return (phi % (2*np.pi)) - np.pi


def gen_noisy_slcs(amp, phi, coh):

    def noisegen(shape):
        real = np.random.standard_normal(shape)
        imag = np.random.standard_normal(shape)
        return (real + 1j*imag) / np.sqrt(2)

    noise1 = noisegen(amp.shape)
    noise2 = noisegen(amp.shape)

    l2 = amp*coh*np.exp(-1j*phi)
    l3 = amp*np.sqrt(1-coh**2)

    slc1 = amp*noise1
    slc2 = l2*noise1 + l3*noise2

    return (slc1, slc2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    shape = (129, 129)

    areas = OrderedDict()
    areas[patterns.plateau]        = {'shape': shape, 'mid_width': 32}
    areas[patterns.const_slope]    = {'shape': shape}
    areas[patterns.sine]           = {'shape': shape, 'omega': 0.3}
    areas[patterns.step_slope]     = {'shape': shape}
    areas[patterns.unit_step]      = {'shape': shape}
    areas[patterns.raised_cos]     = {'shape': shape}
    areas[patterns.banana]         = {'shape': shape}
    areas[patterns.peaks]          = {'shape': shape}
    areas[patterns.zebra]          = {'shape': shape}
    areas[patterns.squares]        = {'shape': shape}
    areas[patterns.checkers]       = {'shape': shape}
    areas[fractals.diamond_square] = {'size': shape[0]}

    fig = plt.figure(figsize=(12, 9))
    fig.subplots_adjust(left=0.02,
                        right=0.98,
                        wspace=0.2,
                        hspace=0.2,
                        bottom=0.02,
                        top=0.94)
    gs = gridspec.GridSpec(3, 4)

    for idx, (method, args) in enumerate(areas.items()):
        name = method.__name__
        print(name)
        data = method(**args)

        ax = plt.subplot(gs[idx])
        ax.imshow(data, cmap=plt.get_cmap('viridis'))
        ax.set_title(name)

    plt.show()

