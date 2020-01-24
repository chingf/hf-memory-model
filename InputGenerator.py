import numpy as np
from math import pi

class InputGenerator(object):
    """Generates inputs for networks."""

    def get_input(self, T, N_c):
        """
        Returns the standard toy input used. 

        Args:
            T (int): Number of time steps
            N_c (int): Number of context units
        Returns:
            input_ext (numpy array): Size (T,) array of float radians
                representing the external stimulus
            input_c (numpy array): Size (T, N_c) array of floats representing
                the activation of context units over time
            alphas (numpy array): Size (T,) array of floats representing the
                strength of the external input.
        """

        input_ext = np.concatenate([
            np.linspace(0, 2*pi, T//5),
            np.linspace(2*pi, 0, T//5),
            np.linspace(0, 0, T//5),
            np.linspace(0, 0, T//5),
            np.linspace(0, 2*pi, T//5)
            ])
        alphas = np.ones(input_ext.size)*0.6
        input_c = np.zeros((input_ext.size, N_c))
        alphas[int(0.4*T):int(0.8*T),] = 0
        input_c = np.zeros((input_ext.size, N_c))
        input_c[int(0.52*T):int(0.64*T), 0] = 1
        input_c[int(0.68*T):int(0.8*T), 1] = 1
        return input_ext, input_c, alphas
