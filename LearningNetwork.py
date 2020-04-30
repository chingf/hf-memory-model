import numpy as np
from math import pi
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from PlotMaker import PlotMaker

import warnings
import traceback
warnings.filterwarnings("error")

class LearningNetwork(object):
    """
    An attractor network made of two separate ring attractors: a place network
    and an episode network. The two networks may overlap to a user-specified
    amount.

    In this model, one timestep corresponds to 100 ms. 

    Args:
        N_pl (int): Number of units in the place network. Their preferred tuning
            evenly covers the ring
        N_ep (int): Number of units in the winner-take-all episode network.
        K_inhib (float): Value of global inhibition

    Attributes:
        N_pl (int): Number of units in the place network. Their preferred tuning
            evenly covers the ring
        N_ep (int): Number of units in the episode network.
        K_inhib (float): Value of global inhibition
        base_J0 (float): parameter representing uniform all-to-all inhibition.
        base_J2 (float): parameter representing amplitude of angle-specific
            interaction.
        dt (float): parameter representing the size of one timestep.
        J0 (float): parameter base_J0 normalized for the number of units.
        J2 (float): parameter base_J2 normalized for the number of units.
        N (int): integer number of units in the network. Thus unit i will
            represent neurons with preferred tuning at 2pi/i radians.
        K_inhib (float): Value of global inhibition
        J (numpy array): (N, N) array of floats representing the connectivity
            between units. V_ij will represent the connection from j to i
        num_units (int): number of total unique units over both networks
        num_shared_units (int): number of unique units in both networks
        ep_modules (numpy array)
        internetwork_units (numpy array): (2, M) array of ints. Indicates
            unidirectional (or bidirectional if feedback is enabled) connections
            between the episode and place network.
        shared_unit_map (numpy array): (2, num_shared_units) array of ints.
            The value at (0,j) is the unit from the episode network that is at
            position (1,j) in the place network.
        J_episode_indices (numpy array): (N,) array of ints. The value at
            position (i,) maps the ith unit of the episode network to the
            corresponding index in the J matrix.
        J_place_indices (numpy array): (N,) array of ints. The value at
            position (i,) maps the ith unit of the palce network to the
            corresponding index in the J matrix.
    """

    base_J0 = 0.3
    base_J2 = 5
    dt = 0.1
    steps_in_s = (1/dt)*50
    kappa = 4. 
    vonmises_gain = 3.2
    norm_scale = 2.5 #5 TODO
    isolated = False

    def __init__(
            self, N_pl, N_ep, K_pl, K_ep,
            overlap=0, num_wta_modules=3, start_random=False, start_wta=False
            ):
        self.N_pl = N_pl
        self.N_ep = N_ep
        self.K_pl = K_pl
        self.K_ep = K_ep
        self.K_inhib = K_pl
        self.overlap = overlap
        self.num_ep_modules = int(N_ep//13)#num_wta_modules
        self.num_pl_modules = num_wta_modules
        self.start_random = start_random
        self.J0 = self.base_J0/N_pl
        self.J2 = self.base_J2/N_pl
        self.internetwork_units = np.array([[], []])
        self._init_wta_modules()
        self._init_shared_units()
        self._init_J(start_wta)
        self.steps_in_s = 10*50
        self.t = 0

    def step(self, prev_m, prev_f, input_t, alpha_t, fastlearn):
        """
        Steps the network forward one time step. Evolves the current network
        activity according to the defined first-order dynamics.

        Args:
            prev_m (numpy array): (N,) size array of floats; the current
            input_t (float): Radian representing the external stimulus.
            alpha_t (float): The strength of the external stimulus

        Returns:
            m_{t} and f_{t}: numpy arrays representing the current and the
                firing rates, respectively, of each unit in the next time step.
        """

        h_ext = alpha_t*input_t
        total_input = self.J @ self._g(prev_m[:, -1]) + h_ext - self.K_inhib
        dmdt = -prev_m[:, -1] + total_input
        m_t = prev_m[:, -1] + self.dt*dmdt
        f_t = self._g(m_t)
        self._update_synapses(prev_f, f_t, fastlearn)
        return m_t, f_t

    def _update_synapses(self, prev_f, f_t, fastlearn):
        if type(fastlearn) is int:
            if fastlearn == 1:
                alpha = 3e-4
                pl_only = False
            elif fastlearn == 0:
                alpha = 0
                pl_only = False
        else:
            alpha = 8e-2 if fastlearn else 3e-4
            pl_only = False
        elapsed_t = prev_f.shape[1]
        window_size = 2000
        if elapsed_t > window_size:
            activity_window = np.sum(prev_f[:, -window_size:], axis=1)
        else:
            activity_window = np.sum(prev_f, axis=1)
        inactive_units = np.argwhere(
            activity_window/activity_window.size < 3/2000
            )
        pre = prev_f[:, -1]
        post = f_t
        threshold = 0.3 #self.K_pl
        pre = pre - threshold
        post = post - threshold 
        pre[np.abs(pre) < 1e-10] = 0
        post[np.abs(post) < 1e-10] = 0
        delta = alpha * np.outer(post, pre)
        inhibitory_idxs = np.ix_(
            np.argwhere(post < 0).squeeze(), np.argwhere(pre < 0).squeeze()
            )
        delta[inhibitory_idxs] = 0
        delta[inactive_units, :] = 0
        delta[:, inactive_units] = 0
        np.fill_diagonal(delta, 0)
        if pl_only and not self.isolated:
            delta[self.J_episode_indices_unshared, :] = 0
            delta[:, self.J_episode_indices_unshared] = 0
        self.J += delta
        self.J = normalize(self.J, axis=1, norm="l1")*self.norm_scale
        #self.J = np.clip(self.J, -1e-2, 1e-1)

    def _init_wta_modules(self):
        self.ep_modules = np.array_split(
            np.arange(self.N_ep).astype(int), self.num_ep_modules
            )
        self.pl_modules = np.array_split(
            np.arange(self.N_pl).astype(int), self.num_pl_modules
            )

    def _init_shared_units(self):
        """ Determines which units are shared between the two networks """

        num_shared_units = int(self.N_pl*self.overlap)
        shared_episode_units = np.random.choice(
            [i for i in range(self.N_ep)], num_shared_units, replace=False
            )
        shared_place_units =  np.random.choice(
            [i for i in range(self.N_pl)], num_shared_units, replace=False
            )
        shared_unit_map = np.vstack((shared_episode_units, shared_place_units))
        self.shared_unit_map = shared_unit_map
        self.num_shared_units = num_shared_units
        self.num_units = self.N_ep + self.N_pl - num_shared_units

    def _init_J(self, start_wta):
        """ Initializes the connectivity matrix J """

        if self.start_random:
            J_episode_indices = np.zeros(self.N_ep).astype(int)
            J_place_indices = np.zeros(self.N_pl).astype(int)
            self.J = np.random.normal(0, 1, (self.num_units, self.num_units))
            J_idx = 0
            for i in range(self.N_ep):
                if i in self.shared_unit_map[0]:continue
                J_episode_indices[i] = J_idx
                J_idx += 1
            for i in range(self.N_pl):
                if i in self.shared_unit_map[1]: continue
                J_place_indices[i] = J_idx
                J_idx += 1
            for i in range(self.num_shared_units):
                ep = self.shared_unit_map[0, i]
                pl = self.shared_unit_map[1, i]
                J_episode_indices[ep] = J_idx
                J_place_indices[pl] = J_idx
                J_idx += 1
            self.J = normalize(self.J, axis=1)
            self.J_episode_indices = J_episode_indices
            self.J_place_indices = J_place_indices
            return

        num_unshared_ep = self.N_ep - self.num_shared_units
        num_unshared_pl = self.N_pl - self.num_shared_units
        J = np.zeros((self.num_units, self.num_units))
        J_episode_indices = np.zeros(self.N_ep).astype(int)
        J_place_indices = np.zeros(self.N_pl).astype(int)
        J_idx = 0

        iw = 1.75
        # Fill in unshared episode network connectivity matrix
        wta_weight = 1.
        wta_excit = wta_weight/(self.N_ep//self.num_ep_modules)
        wta_inhib = -iw*wta_weight/(self.N_ep - self.N_ep//self.num_ep_modules)
        for m_i in range(self.num_ep_modules):
            module = self.ep_modules[m_i]
            for i in module:
                if i in self.shared_unit_map[0]: continue
                weights = np.ones(self.N_ep)*wta_inhib
                weights[module] = wta_excit
                weights = np.delete(weights, self.shared_unit_map[0])
                J[J_idx, :weights.size] = weights
                J_episode_indices[i] = int(J_idx)
                J_idx += 1

        # Fill in unshared place network connectivity matrix
        if start_wta:
            wta_weight = 1.
            wta_excit = wta_weight/(self.N_pl//self.num_pl_modules)
            wta_inhib = -iw*wta_weight/(self.N_pl - self.N_pl//self.num_pl_modules)
            for m_i in range(self.num_pl_modules):
                module = self.pl_modules[m_i]
                for i in module:
                    if i in self.shared_unit_map[1]: continue
                    weights = np.ones(self.N_pl)*wta_inhib
                    weights[module] = wta_excit
                    weights = np.delete(weights, self.shared_unit_map[1])
                    J[J_idx,
                        num_unshared_ep:num_unshared_ep + weights.size
                        ] = weights
                    J_place_indices[i] = int(J_idx)
                    J_idx += 1
        else:
            for i in range(self.N_pl):
                if i in self.shared_unit_map[1]: continue
                weights = self._get_vonmises(i)
                weights = np.delete(weights, self.shared_unit_map[1])
                J[J_idx, num_unshared_ep:num_unshared_ep + num_unshared_pl] = weights
                J_place_indices[i] = int(J_idx)
                J_idx += 1

        # Fill in shared units for episode and place
        for i in range(self.num_shared_units):
            ep_unit = self.shared_unit_map[0, i]
            pl_unit = self.shared_unit_map[1, i]

            # Weights between shared unit and unshared episode unit
            ep_module_idx = np.argwhere(
                [ep_unit in m for m in self.ep_modules]
                )[0,0]
            wta_weights = np.ones(self.N_ep)*wta_inhib
            wta_weights[self.ep_modules[ep_module_idx]] = wta_excit
            for ep in range(self.N_ep):
                if ep in self.shared_unit_map[0]: continue
                J[J_idx, J_episode_indices[ep]] = wta_weights[ep]
                J[J_episode_indices[ep], J_idx] = wta_weights[ep]

            # Weights between shared unit and unshared place unit
            if start_wta:
                pl_module_idx = np.argwhere(
                    [pl_unit in m for m in self.pl_modules]
                    )[0,0]
                place_weights = np.ones(self.N_pl)*wta_inhib
                place_weights[self.pl_modules[pl_module_idx]] = wta_excit
            else:
                place_weights = self._get_vonmises(pl_unit)
            for place in range(self.N_pl):
                if place in self.shared_unit_map[1]: continue
                J[J_idx, J_place_indices[place]] = place_weights[place]
                J[J_place_indices[place], J_idx] = place_weights[place]

            # Weights between shared unit and shared unit
            for j in np.arange(1, self.num_shared_units - i):
                j_ep_unit = self.shared_unit_map[0, j]
                j_pl_unit = self.shared_unit_map[1, j]
                total_weight = 0.5*(
                    wta_weights[j_ep_unit] + place_weights[j_pl_unit]
                    )
                J[J_idx, J_idx + j] = total_weight
                J[J_idx + j, J_idx] = total_weight

            J_episode_indices[ep_unit] = int(J_idx)
            J_place_indices[pl_unit] = int(J_idx)
            J_idx += 1

        # Remove any self-excitation
        np.fill_diagonal(J, 0)
        self.J_episode_indices = J_episode_indices
        self.J_place_indices = J_place_indices
        self.J_episode_indices_unshared = [
            i for u, i in enumerate(J_episode_indices) if u not in self.shared_unit_map[0]
            ]
        self.J = J
        self.J = normalize(self.J, axis=1, norm="l1")*self.norm_scale

    def _g(self, x):
        """
        Rectifies and saturates a given vector.
        """

        return np.clip(x, 0, 1)

    def _get_vonmises_weight(self, i ,center):
        curve = self._get_vonmises(center=center)
        return curve[i]

    def _get_vonmises(self, center):
        """ Returns a sharp sinusoidal curve that drops off rapidly """

        mu = 0
        x = np.linspace(-pi, pi, self.N_pl, endpoint=False)
        curve = np.exp(self.kappa*np.cos(x-mu))/(2*pi*np.i0(self.kappa))
        curve -= np.max(curve)/2.
        curve *= self.vonmises_gain
        curve = np.roll(curve, center - self.N_pl//2)
        return -self.J0 + self.J2*curve
