import numpy as np
from math import pi

class InputGenerator(object):
    """Generates inputs for networks."""

    def get_constant_input(self, network, peaks, peaktype="exp"):
        """
        Returns input centered at PEAKS coherent in the first implicit ring of
        the network.
        """

        T = 1000
        Ns = np.arange(network.N)
        input_t = np.zeros(network.N)
        peakwidth = 8 
        for peak in peaks:
            if peaktype == "exp":
                bump = -np.square(np.arange(-peakwidth, peakwidth+1))/(peakwidth**2)
                bump += 1 
            elif peaktype == "step":
                bump = np.ones(peakwidth*2 + 1)
            for idx, unit in enumerate(np.arange(peak-peakwidth, peak + peakwidth+1)):
                input_t[unit] += bump[idx]
        return np.tile(input_t, (T, 1))
            
