import numpy as np
from math import pi

class InputGenerator(object):
    """Generates inputs for networks."""

    def get_noise_input(self, network):
        """ Random noise to feed into the network, sustained by some time """

        # Generate input for one time step over all units
        T = 600
        input_ext = np.zeros((T, network.num_units))
        input_ext[:,:network.num_separate_units] = np.random.normal(
            0, 1, input_ext[:, :network.num_separate_units].shape
            )
        input_ext[:,network.J_episode_indices] = np.random.normal(
            0, 1, input_ext[:,network.J_episode_indices].shape
            )
        alphas = np.zeros(T)
        alphas[:100] = 0.6
        return input_ext, alphas

    def get_sin_input(self, network, loc=pi/4):
        """ Random noise to feed into the network, sustained by some time """

        # Generate input for one time step over all units
        T = 600
        cos = np.cos(np.linspace(-pi, pi, network.N))
        input_ext = np.zeros((T, network.num_units))
        input_ext[:,network.J_episode_indices] = np.tile(
            np.roll(cos, int((loc/(2*pi))*network.N) - network.N//2),
            (T, 1)
            )
        alphas = np.zeros(T)
        alphas[:100] = 0.6
        return input_ext, alphas

