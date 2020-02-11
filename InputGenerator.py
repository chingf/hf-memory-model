import numpy as np
from math import pi

class InputGenerator(object):
    """Generates inputs for networks."""

    def get_noise_input(self, network):
        """ Random noise to feed into the network, sustained by some time """

        # Generate input for one time step over all units
        input_t = np.zeros(network.num_units)
        input_t[:network.num_separate_units] = np.random.normal(
            0, 1, network.num_separate_units
            )
        for i in network.shared_unit_map[0]:
            J_idx = network.J_episode_indices[i]
            input_t[J_idx] = np.random.normal(0, 1)

        T = 500
        input_ext = np.tile(input_t, (T, 1))
        alphas = np.zeros(T)
        alphas[:T//3] = 0.6
        return input_ext, alphas

