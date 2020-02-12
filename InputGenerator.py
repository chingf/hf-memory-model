import numpy as np
from math import pi

class InputGenerator(object):
    """Generates inputs for networks."""

    def get_noise_input(self, network):
        """ Random noise to feed into the network, sustained by some time """

        # Generate input for one time step over all units
        T = 500
        input_ext = np.zeros((T, network.num_units))
        input_ext[:,:network.num_separate_units] = np.random.normal(
            0, 1, input_ext[:, :network.num_separate_units].shape
            )
        alphas = np.zeros(T)
        alphas[:T//3] = 0.6
        return input_ext, alphas

