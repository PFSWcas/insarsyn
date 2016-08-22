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
    areas[patterns.sine]           = {'shape': shape, 'omega': 0.1}
    areas[patterns.logfreq]        = {'shape': shape, 'doublerate': 43}
    areas[patterns.step_slope]     = {'shape': shape}
    areas[patterns.unit_step]      = {'shape': shape}
    areas[patterns.raised_cos]     = {'shape': shape}
    areas[patterns.banana]         = {'shape': shape}
    areas[patterns.peaks]          = {'shape': shape}
    areas[patterns.zebra]          = {'shape': shape}
    areas[patterns.logbar]         = {'shape': shape, 'doublerate': 43}
    areas[patterns.squares]        = {'shape': shape}
    areas[patterns.checkers]       = {'shape': shape}
    areas[fractals.diamond_square] = {'size': shape[0], 'seed': 42}

    plots = [('Unwrapped Phase',
              lambda x: x,
              {'cmap': plt.get_cmap('viridis')}),
             ('Wrapped Phase',
              lambda x: wrap_phase(5*np.pi*x),
              {'cmap': plt.get_cmap('hsv'), 'vmin': -np.pi, 'vmax': np.pi})]

    for (name, modder, plot_opts) in plots:
        fig = plt.figure(figsize=(8, 5.2))
        fig.suptitle(name, fontsize=12)
        fig.subplots_adjust(left=0.05,
                            right=0.95,
                            wspace=0.2,
                            hspace=0.2,
                            bottom=0.05,
                            top=0.9)
        gs = gridspec.GridSpec(3, 5)

        for idx, (method, args) in enumerate(areas.items()):
            ax = plt.subplot(gs[idx])
            ax.imshow(modder(method(**args)), **plot_opts)
            ax.set_title(method.__name__, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        filename = 'insarsyn_{}.png'.format(name.lower().replace(' ', '_'))
        fig.savefig(filename, dpi=150)

