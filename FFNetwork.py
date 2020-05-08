import numpy as np
from math import pi
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from PlotMaker import PlotMaker
import warnings
import traceback
warnings.filterwarnings("error")

class FFNetwork(object):
    """
    An attractor network made of two separate ring attractors: a place network
    and an episode network. The two networks may overlap to a user-specified
    amount.

    In this model, one timestep corresponds to 100 ms. 

    Args:

    Attributes:
    """

    dt = 0.1
    steps_in_s = int((1/dt)*50)

    def __init__(
        self, num_units=100, btsp_scale=5, J_mean=-0.1, J_std=0.1
        ):

        self.num_units = num_units
        self.btsp_scale = btsp_scale
        self.J_mean = J_mean
        self.J_std = J_std
        self._init_J()
        self._init_J_ext()
        self._set_btsp_params()
        self.t = 0
        self.memories = []

    def step(self, inputs, prev_m, prev_f, btsp):
        """
        Steps the network forward one time step. Evolves the current network
        activity according to the defined first-order dynamics.

        Args:
            prev_m (numpy array): (num_units,) size array of floats; the current
            input_t (float): Radian representing the external stimulus.

        Returns:
            m_{t} and f_{t}: numpy arrays representing the current and the
                firing rates, respectively, of each unit in the next time step.
        """

        input_t = inputs[:, -1]
        h_ext = self.J_ext @ input_t
        total_input = self.J @ self._g(prev_m[:, -1]) + h_ext - 0.2
        dmdt = -prev_m[:, -1] + total_input
        m_t = prev_m[:, -1] + self.dt*dmdt
        f_t = self._g(m_t)
        self._update_recurrent_synapses(inputs, prev_f, f_t, btsp)
        #self._update_ext_synapses(inputs, prev_f, f_t, btsp)
        return m_t, f_t

    def _update_recurrent_synapses(self, inputs, prev_f, f_t, btsp):
        if btsp == 0:
            return
        elapsed_t = prev_f.shape[1]
        if elapsed_t > self.eligibility_size:
            eligibility_trace = prev_f[:, -self.eligibility_size:]
        else:
            pad_length = self.eligibility_size - elapsed_t
            eligibility_trace = np.pad(
                prev_f[:,:elapsed_t], ((0,0),(pad_length, 0)), 'constant'
                )
        plasticity_change = np.sum(
            eligibility_trace*self.eligibility_kernel, axis=1
            )
        plasticity_change = self._btsp_g(plasticity_change)
        self.memories.append(plasticity_change)
        plastic_synapses = plasticity_change > 0
        plastic_synapses = np.logical_and(
            plastic_synapses, np.random.uniform(size=self.num_units) < btsp
            )
        plastic_synapses = np.logical_and(
            plastic_synapses, np.logical_not(self.btsp_history)
            )
        print("Number of Plastic Synapses: %d"%np.sum(plastic_synapses))
        self.J[plastic_synapses, :] = np.tile(
            plasticity_change, (np.sum(plastic_synapses), 1)
            )
        np.fill_diagonal(self.J, 0)
        self.btsp_history[plastic_synapses] = True

    def _update_ext_synapses(self, inputs, prev_f, f_t, btsp):
        if btsp == 0:
            return
        elapsed_t = prev_f.shape[1]
        if elapsed_t > self.eligibility_size:
            eligibility_trace = inputs[:, -self.eligibility_size:]
        else:
            pad_length = self.eligibility_size - elapsed_t
            eligibility_trace = np.pad(
                inputs[:,:elapsed_t], ((0,0),(pad_length, 0)), 'constant'
                )
        plasticity_change = np.sum(
            eligibility_trace*self.eligibility_kernel, axis=1
            )
        plasticity_change = self._btsp_g(plasticity_change)
        plastic_synapses = (np.random.uniform(size=self.num_units) < btsp)
        print(np.sum(plastic_synapses))
        plastic_synapses = np.logical_and(
            plastic_synapses, np.logical_not(self.btsp_history)
            )
        self.J_ext[plastic_synapses, :] = np.tile(
            plasticity_change, (np.sum(plastic_synapses), 1)
            )
        self.btsp_history[plastic_synapses] = True

    def _init_J(self):
        self.J = np.zeros((self.num_units, self.num_units))
        self.J = np.random.normal(
            loc=self.J_mean, scale=self.J_std,
            size=(self.num_units, self.num_units)
            )
        np.fill_diagonal(self.J, 0)

    def _init_J_ext(self):
        self.J_ext = np.random.normal(
            loc=0, scale=0.2, size=(self.num_units, self.num_units)
            )
        self.J_ext = np.diag(np.ones(self.num_units))

    def _set_btsp_params(self):
        eligibility_size = self.steps_in_s
        eligibility_kernel = np.ones(eligibility_size)/eligibility_size
        self.eligibility_size = eligibility_size
        self.eligibility_kernel = eligibility_kernel
        self.btsp_history = np.zeros(self.num_units).astype(bool)

    def _btsp_g(self, x):
        x = np.clip(x, 0, 1)
        x -= np.sum(x)/x.size
        return x*self.btsp_scale

    def _g(self, x):
        """
        Rectifies and saturates a given vector.
        """

        return np.clip(x, 0, 1)

