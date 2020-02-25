import numpy as np
from math import pi
import matplotlib.pyplot as plt

class OverlapNetwork(object):
    """
    An attractor network made of two separate ring attractors: a place network
    and an episode network. The two networks may overlap to a user-specified
    amount.

    In this model, one timestep corresponds to 100 ms. 

    Args:
        N (int): Number of units in one network. Their preferred tuning evenly
           covers the ring
        K_inhib (float): Value of global inhibition

    Attributes:
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
        thetas (numpy array): (N,2) array of float radians. The value at i,j
            represents the preferred tuning of unit i in network j
        num_units (int): number of total unique units over both networks
        num_shared_units (int): number of unique units in both networks
        num_separate_units (int): number of unique units in only one network
        shared_unit_map (numpy array): (2, num_shared_units) array of ints.
            The value at (0,j) is the unit from the episode network that is at
            position (1,j) in the place network.
        J_episode_indices (numpy array): (N,) array of ints. The value at
            position (i,) maps the ith unit of the episode network to the
            corresponding index in the J matrix.
        J_place_indices (numpy array): (N,) array of ints. The value at
            position (i,) maps the ith unit of the palce network to the
            corresponding index in the J matrix.
        interacting_units (numpy array): (2, M) array of ints. Indicates
            unidirectional (or bidirectional if feedback is enabled) connections
            between the episode and place network.
    """

    base_J0 = 0.3 
    base_J2 = 5.
    dt = 0.1
    kappa = 4
    vonmises_gain = 3.

    def __init__(
            self, N, K_inhib, overlap=0, num_interactions=3,
            add_feedback=False, add_attractor=False
            ):
        self.N = N
        self.K_inhib = K_inhib
        self.overlap = overlap
        self.num_interactions = num_interactions
        self.add_feedback = add_feedback
        self.add_attractor = add_attractor
        self.J0 = self.base_J0/N
        self.J2 = self.base_J2/N
        self._init_interacting_units()
        self._init_shared_units()
        self._init_thetas()
        self._init_J()
        self._init_J_interactions()
        self._init_episode_attractors()

    def step(self, prev_m, input_t, alpha_t):
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

        if type(input_t) is np.ndarray:
            h_ext = alpha_t*input_t
        else:
            h_ext = np.zeros(self.num_units)
            episode_input = alpha_t*np.cos(input_t - self.thetas)
            for idx, J_idx in enumerate(self.J_episode_indices):
                h_ext[J_idx] = episode_input[idx]
        f_t = self.J @ self._g(prev_m) + self._g(h_ext)
        dmdt = -prev_m + f_t - self.K_inhib
        m_t = prev_m + self.dt*dmdt 
        return m_t, f_t

    def _init_interacting_units(self):
        """ Determines which units connect between the two networks """

        if self.num_interactions == 0:
            self.interacting_units = np.array([[], []])
        else:
            center_units = np.linspace(
                0, self.N, self.num_interactions, endpoint=False
                )
            episode_units = []
            place_units = []
            for c in center_units:
                neighbors = [int(c + i)%self.N for i in np.arange(-2, 3)]
                episode_units.extend(neighbors)
                place_units.extend(neighbors)
            self.interacting_units = np.array([episode_units, place_units])

    def _init_shared_units(self):
        """ Determines which units are shared between the two networks """

        episode_units = self.interacting_units[0] 
        num_shared_units = int(self.N*self.overlap)
        shared_episode_units = np.random.choice(
            [i for i in range(self.N) if i not in episode_units],
            num_shared_units, replace=False
            )
        shared_place_units =  np.random.choice(
            [i for i in range(self.N) if i not in episode_units],
            num_shared_units, replace=False
            )
        shared_unit_map = np.vstack((shared_episode_units, shared_place_units))
        self.shared_unit_map = shared_unit_map
        self.num_shared_units = num_shared_units
        self.num_separate_units = self.N - num_shared_units
        self.num_units = self.N*2 - num_shared_units
        np.random.seed()

    def _init_thetas(self):
        """ Initializes the preferred tuning within a network """

        thetas = np.linspace(0, 2*pi, self.N, endpoint=False)
        self.thetas = thetas

    def _init_J(self):
        """ Initializes the connectivity matrix J """

        num_separate_units = self.N - self.num_shared_units
        J = np.zeros((self.num_units, self.num_units))
        J_episode_indices = np.zeros(self.N).astype(int)
        J_place_indices = np.zeros(self.N).astype(int)
        curr_unit = 0

        # Fill in unshared episode units
        for i in range(self.N):
            if i in self.shared_unit_map[0,:]:
                continue
            weights = self._get_weight_vector(i)
            weights = np.delete(weights, self.shared_unit_map[0,:])
            J[curr_unit, :num_separate_units] = weights
            J_episode_indices[i] = int(curr_unit)
            curr_unit += 1

        # Fill in unshared place units
        for i in range(self.N):
            if i in self.shared_unit_map[1,:]:
                continue
            weights = self._get_weight_vector(i)
            weights = np.delete(weights, self.shared_unit_map[1,:])
            J[curr_unit, num_separate_units:num_separate_units*2] = weights 
            J_place_indices[i] = int(curr_unit)
            curr_unit += 1

        # Fill in shared units for episode and place
        for i in range(self.num_shared_units):
            episode_unit = self.shared_unit_map[0, i]
            place_unit = self.shared_unit_map[1, i]
            for j_ep in range(self.N):
                if j_ep in self.shared_unit_map[0,:]:
                    continue
                J_idx = J_episode_indices[j_ep]
                weight = self._get_weight_value(episode_unit, j_ep)
                J[curr_unit, J_idx] = weight
                J[J_idx, curr_unit] = weight 
            for j_place in range(self.N):
                if j_place in self.shared_unit_map[1,:]:
                    continue
                J_idx = J_place_indices[j_place]
                weight = self._get_weight_value(place_unit, j_place)
                J[curr_unit, J_idx] = weight
                J[J_idx, curr_unit] = weight 
            for j_shared in np.arange(1, self.num_shared_units - i):
                j_shared_ep = self.shared_unit_map[0, j_shared]
                j_shared_pl = self.shared_unit_map[1, j_shared]
                weight_ep = self._get_weight_value(episode_unit, j_shared_ep)
                weight_pl = self._get_weight_value(place_unit, j_shared_pl)
                total_weight = -self.J0 + 0.5*(weight_ep + weight_pl)
                J[curr_unit, curr_unit + j_shared] = total_weight 
                J[curr_unit + j_shared, curr_unit] = total_weight 
            J_episode_indices[episode_unit] = int(curr_unit)
            J_place_indices[place_unit] = int(curr_unit)
            curr_unit += 1

        # Remove self-excitation
        for i in range(self.num_units):
            J[i,i] = 0

        self.J_episode_indices = J_episode_indices
        self.J_place_indices = J_place_indices
        self.J = J

    def _init_J_interactions(self):
        """ Adds the interactions between networks to J matrix """

        episode_units = self.interacting_units[0]
        place_units = self.interacting_units[1]
        interaction_support = np.arange(-self.N//2, self.N//2 + 1)

        for idx, episode_unit in enumerate(episode_units):
            episode_unit = self.J_episode_indices[episode_unit]
            for i in interaction_support:
                weight_offset = self._get_weight_value(i, 0)
                scale = 2
                weight_offset *= scale
                place_unit = self.J_place_indices[(place_units[idx]+i)%self.N]
                self.J[place_unit, episode_unit] = weight_offset

        if self.add_feedback:
            for idx, place_unit in enumerate(place_units):
                place_unit = self.J_place_indices[place_unit]
                for i in interaction_support:
                    weight_offset = self._get_weight_value(i, 0)
                    scale = 2
                    weight_offset *= scale
                    episode_unit = self.J_episode_indices[(episode_units[idx]+i)%self.N]
                    self.J[episode_unit, place_unit] = weight_offset
        for i in range(self.J.shape[0]):
            self.J[i,i] = 0

    def _init_episode_attractors(self):
        """ Tries to burn in stronger attractors in the episode network """

        if self.add_attractor is False:
            self.episode_attractors = []
            return
        episode_attractors = [self.N//4, 3*self.N//4]
        for attractor in episode_attractors:
            for offset in np.arange(-3, 4):
                for other_idx in range(self.N):
                    attractor_idx = attractor + offset
                    sharp_cos = np.roll(
                        self._get_sharp_cos(int(self.N*0.18)),
                        attractor_idx - self.N//2,
                        )
                    new_weights = -self.J0 + self.J2*sharp_cos
                    new_weight = new_weights[other_idx]
                    other_idx = self.J_episode_indices[other_idx]
                    attractor_idx = self.J_episode_indices[attractor_idx]
                    if other_idx == attractor_idx:
                        continue
                    self.J[other_idx, attractor_idx] = new_weight
        self.episode_attractors = episode_attractors

    def _g(self, f_t):
        """
        Rectifies and saturates a given firing rate.
        """

        return np.clip(f_t, 0, 1)

    def _get_weight_vector(self, i):
        sharp_cos = np.roll(self._get_sharp_cos(), i - self.N//2)
        return -self.J0 + self.J2*sharp_cos

    def _get_weight_value(self, i ,j):
        sharp_cos = np.roll(self._get_sharp_cos(), i - self.N//2)
        return self.J0 + self.J2*sharp_cos[j]

    def _get_sharp_cos(self):
        """ Returns a sharp sinusoidal curve that drops off rapidly """

        mu = 0
        x = np.linspace(-pi, pi, self.N, endpoint=False)
        curve = np.exp(self.kappa*np.cos(x-mu))/(2*pi*np.i0(self.kappa))
        curve -= np.max(curve)/2.
        curve *= self.vonmises_gain
        return curve
