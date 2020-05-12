import numpy as np
from math import pi
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from PlotMaker import PlotMaker
from RingAttractorRNN import RingAttractorRNN
from HebbRNN import HebbRNN

class MixedRNN(HebbRNN):
    """
    An attractor network made of two separate ring attractors: a place network
    and an episode network. The two networks may overlap to a user-specified
    amount.

    In this model, one timestep corresponds to 100 ms. 

    Args:

    Attributes:
    """

    def __init__(
        self, N_pl=100, N_ep=100, J_mean=-0.1, J_std=0.1, K_inhib=0.3,
        plasticity_scale=0.4
        ):

        super().__init__(
            self, N_pl=100, N_ep=100, J_mean=-0.1, J_std=0.1, K_inhib=0.3,
            plasticity_scale=0.4
            )
        self.ext_memories = []

    def _update_ext_synapses(self, inputs, prev_f, f_t, plasticity):
        if plasticity == 0:
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
        plasticity_change = self._plasticity_g(plasticity_change)
        self.ext_memories.append(plasticity_change)
        plastic_synapses = plasticity_change > 0
        plastic_synapses = np.logical_and(
            plastic_synapses, np.random.uniform(size=self.num_units) < plasticity
            )
        plastic_synapses = np.logical_and(
            plastic_synapses, np.logical_not(self.ext_plasticity_history)
            )
        self.ext_plasticity_history[plastic_synapses] = True
        plt.plot(plasticity_change); plt.show()
        self.J_ext[plastic_synapses, :] = np.tile(
            plasticity_change, (np.sum(plastic_synapses), 1)
            )

    def _set_plasticity_params(self):
        super()._set_plasticity_params()
        self.ext_plasticity_history = np.zeros(self.num_units).astype(bool)

    def _init_J_ext(self):
        self.J_ext = np.random.normal(
            loc=self.J_mean, scale=self.J_std, size=self.num_units
            )

