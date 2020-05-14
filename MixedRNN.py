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
        ext_plasticity_scale=0.4, plasticity_scale=0.8
        ):

        super().__init__(
            N_pl=N_pl, N_ep=N_ep, J_mean=J_mean, J_std=J_std, K_inhib=K_inhib,
            plasticity_scale=plasticity_scale
            )
        self.ext_memories = []
        self.ext_plasticity_scale = ext_plasticity_scale

    def _update_ext_synapses(self, inputs, prev_f, f_t, ext_plasticity):
        if ext_plasticity == 0:
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
        plasticity_change = self._plasticity_g(
            plasticity_change
            )*self.ext_plasticity_scale
        plastic_synapses = plasticity_change > 0
        plastic_synapses = np.logical_and(
            plastic_synapses, np.random.uniform(size=self.num_units) < ext_plasticity
            )
        plastic_synapses = np.logical_and(
            plastic_synapses, np.logical_not(self.ext_plasticity_history)
            )
        self.ext_plasticity_history[plastic_synapses] = True
        self.ext_memories.append(plasticity_change)
        self.J_ext[plastic_synapses, :] = np.tile(
            plasticity_change, (np.sum(plastic_synapses), 1)
            )

    def _set_plasticity_params(self):
        super()._set_plasticity_params()
        self.ext_plasticity_history = np.zeros(self.num_units).astype(bool)

    def _init_J_ext(self):
#        self.J_ext = np.random.normal(
#            loc=self.J_mean, scale=self.J_std,
#            size=(self.num_units, self.num_units)
#            )
        self.J_ext = np.random.normal(
            loc=0, scale=0.2,
            size=(self.num_units, self.num_units)
            )
        self.J_ext[np.ix_(self.J_pl_indices, self.J_pl_indices)] = np.diag(
            np.ones(self.N_pl)
            )
        self.J_ext[np.ix_(self.J_pl_indices, self.J_ep_indices)] = 0
