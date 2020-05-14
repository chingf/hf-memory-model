import numpy as np
from math import pi
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from PlotMaker import PlotMaker
from RingAttractorRNN import RingAttractorRNN

class HebbRNN(object):
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
        self, N_pl=100, N_ep=100, J_mean=-0.1, J_std=0.1, K_inhib=0.3,
        plasticity_scale=0.4
        ):

        self.N_pl = N_pl
        self.N_ep = N_ep
        self.J_ep_indices = np.arange(N_ep)
        self.J_pl_indices = np.arange(N_pl) + N_ep
        self.num_units = N_pl + N_ep
        self.J_mean = J_mean
        self.J_std = J_std
        self.K_inhib = K_inhib
        self.plasticity_scale = plasticity_scale
        self._set_plasticity_params()
        self._init_J()
        self._init_J_ring()
        self._init_J_ext()
        self.J = np.clip(self.J, -J_std*12, J_std*12)
        self.t = 0
        self.memories = []

    def step(self, inputs, prev_m, prev_f, plasticity, ext_plasticity):
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
        total_input = self.J @ self._g(prev_m[:, -1]) + h_ext - self.K_inhib
        dmdt = -prev_m[:, -1] + total_input
        m_t = prev_m[:, -1] + self.dt*dmdt
        f_t = self._g(m_t)
        self._update_recurrent_synapses(inputs, prev_f, f_t, plasticity)
        self._update_ext_synapses(inputs, prev_f, f_t, ext_plasticity)
        return m_t, f_t

    def _update_recurrent_synapses(self, inputs, prev_f, f_t, plasticity):
        if plasticity == 0:
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
        plasticity_change = self._plasticity_g(
            plasticity_change
            )*self.plasticity_scale
        self.memories.append(plasticity_change)
        plastic_synapses = plasticity_change > 0
        plastic_synapses = np.logical_and(
            plastic_synapses, np.random.uniform(size=self.num_units) < plasticity
            )
        plastic_synapses = np.logical_and(
            plastic_synapses, np.logical_not(self.plasticity_history)
            )
        self.plasticity_history[plastic_synapses] = True
        plasticity_change = np.outer(plasticity_change, plasticity_change)
        self.J[plastic_synapses,:] = plasticity_change[plastic_synapses,:]
        self.J[:,plastic_synapses] = plasticity_change[:,plastic_synapses]
        np.fill_diagonal(self.J, 0)
        self.J = np.clip(self.J, -self.J_std*12, self.J_std*12)

    def _update_ext_synapses(self, inputs, prev_f, f_t, plasticity):
        pass

    def _set_plasticity_params(self):
        eligibility_size = int(self.steps_in_s/6)
        #eligibility_kernel = np.ones(eligibility_size)/eligibility_size
        eligibility_kernel = self._exponential(eligibility_size*2, tau=70)
        eligibility_kernel = eligibility_kernel[:eligibility_size]
        self.eligibility_size = eligibility_size
        self.eligibility_kernel = eligibility_kernel
        self.plasticity_history = np.zeros(self.num_units).astype(bool)

    def _init_J(self):
        self.J = np.random.normal(
            loc=self.J_mean, scale=self.J_std,
            size=(self.num_units, self.num_units)
            )
        np.fill_diagonal(self.J, 0)

    def _init_J_ring(self):
        J_ring = RingAttractorRNN(num_units=self.N_pl).J
        self.J[-self.N_pl:, -self.N_pl:] = J_ring
        #self.plasticity_history[-self.N_pl:] = True

    def _init_J_ext(self):
        self.J_ext = np.diag(np.ones(self.num_units))

    def _g(self, x):
        return np.clip(x, 0, 1)

    def _plasticity_g(self, x):
        x = np.clip(x, 0, 1)
        x -= np.sum(x)/x.size
        return x

    def _exponential(self, M, tau):
        n = np.arange(M*2)
        center = (M*2 - 1)/2
        kernel = np.exp(-np.abs(n-center)/tau)
        return kernel[:M]
