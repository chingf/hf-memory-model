import numpy as np
from math import pi
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from PlotMaker import PlotMaker
from LearningNetwork import LearningNetwork

import warnings
import traceback
warnings.filterwarnings("error")

class IsolatedNetwork(LearningNetwork):
    """ Just one network """

    base_J0 = 0.3
    base_J2 = 5.
    dt = 0.1
    kappa = 4. 
    vonmises_gain = 3.2
    norm_scale = 5

    def __init__(self, N, K_inhib, mode, args=None):
        self.N = N
        self.K_inhib = K_inhib
        self._set_variables(N, K_inhib)
        if mode == "wta":
            self.num_ep_modules = args
            self._init_wta()
        elif mode == "ring":
            self.J0 = self.base_J0/N
            self.J2 = self.base_J2/N
            self._init_ring()
        else: # Random
            self._init_random()
        self.t = 0
        self.J = normalize(self.J, axis=1, norm="l1")*self.norm_scale


    def _init_wta(self):
        self.ep_modules = np.array_split(
            np.arange(self.N).astype(int), self.num_ep_modules
            )
        J = np.zeros((self.N, self.N))
        ep_weight = 1.
        ep_excit = ep_weight/(self.N//self.num_ep_modules)
        ep_inhib = -ep_weight/(self.N - self.N//self.num_ep_modules)
        for m_i in range(self.num_ep_modules):
            module = self.ep_modules[m_i]
            for i in module:
                weights = np.ones(self.N)*ep_inhib
                weights[module] = ep_excit
                J[i, :weights.size] = weights
        self.J = J

    def _init_ring(self):
        J = np.zeros((self.N, self.N))
        for i in range(self.N):
            weights = self._get_vonmises(i)
            J[i, :] = weights
        self.J = J

    def _init_random(self):
        self.J = np.random.normal(0, 0.1, (self.N, self.N))

    def _set_variables(self, N, K_inhib):
        self.num_units = N; self.N_ep = N; self.N_pl = N
        self.J_episode_indices = np.arange(self.N)
        self.J_place_indices = np.arange(self.N)
        self.shared_unit_map = np.array([[],[]])
        self.internetwork_units = np.array([[],[]])
        self.J_episode_indices_unshared = np.arange(N)

