import numpy as np
from math import pi

class InputGenerator(object):
    """Generates inputs for networks."""

    def get_constant_input(
            self, network, peaks, peaktype="exp", peakwidth=8, which_ring=0
            ):
        """
        Returns input centered at PEAKS coherent in the first implicit ring of
        the network.
        """

        T = 1000
        Ns = np.arange(network.N)
        input_t = np.zeros(network.N)
        for peak in peaks:
            if peaktype == "exp":
                bump = -np.square(np.arange(-peakwidth, peakwidth+1))/(peakwidth**2)
                bump += 1 
            elif peaktype == "step":
                bump = np.ones(peakwidth*2 + 1)
            for idx, unit in enumerate(np.arange(peak-peakwidth, peak + peakwidth+1)):
                input_t[unit] += bump[idx]
        if which_ring == 1:
            input_t = input_t[network.remap_order]
        input_ext = np.tile(input_t, (T, 1))
        return input_ext 

    def get_moving_input(
            self, network, which_ring=0
            ):
        """
        Returns input centered at PEAKS coherent in the first implicit ring of
        the network.
        """

        T = 1000 
        locs = np.linspace(0, 2*pi, T)
        input_ext = []
        for loc in locs:
            input_ext.append(
                np.cos(loc - network.thetas[:, which_ring])
                )
        return np.array(input_ext), locs
            
