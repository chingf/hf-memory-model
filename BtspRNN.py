import numpy as np
from math import pi
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from PlotMaker import PlotMaker
from HebbRNN import HebbRNN
from RingAttractorRNN import RingAttractorRNN

class BtspRNN(HebbRNN):
    """
    An attractor network made of two separate ring attractors: a place network
    and an episode network. The two networks may overlap to a user-specified
    amount.

    In this model, one timestep corresponds to 100 ms. 

    Args:

    Attributes:
    """

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
        plasticity_change = self._plasticity_g(plasticity_change)
        self.memories.append(plasticity_change)
        plastic_synapses = plasticity_change > 0
        plastic_synapses = np.logical_and(
            plastic_synapses, np.random.uniform(size=self.num_units) < plasticity
            )
        plastic_synapses = np.logical_and(
            plastic_synapses, np.logical_not(self.plasticity_history)
            )
        print("Number of Plastic Synapses: %d"%np.sum(plastic_synapses))
        self.plasticity_history[plastic_synapses] = True
        plt.plot(plasticity_change); plt.show()
        self.J[plastic_synapses, :] = np.tile(
            plasticity_change, (np.sum(plastic_synapses), 1)
            )
        np.fill_diagonal(self.J, 0)
        self.J = np.clip(self.J, -self.J_std*12, self.J_std*12)
