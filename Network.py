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
            between units. V_ij will represent the connection from j to i.
        thetas (numpy array): (N,2) array of float radians. The value at i,j
            represents the preferred tuning of unit i in network j
        num_units
        num_shared_units
        num_separate_units
        shared_unit_map
        J_episode_indices
        J_place_indices

    """

    base_J0 = 0.3 
    base_J2 = 5.
    dt = 0.1

    def __init__(self, N, K_inhib, overlap=0):
        self.N = N
        self.K_inhib = K_inhib
        self.overlap = overlap
        self.J0 = self.base_J0/N
        self.J2 = self.base_J2/N
        self._init_networks()
        self._init_thetas()
        self._init_J()

    def simulate(self, input_ext, alphas=None):
        """
        Simulates the behavior of the ring attractor over some period of time.
        The input only arrives to units involved in the episode network.
    
        Args:
            input_ext (numpy array): Size (T,) array of float radians representing
                the external stimulus. Here, the stimulus is some external cue
                that indicates what the correct orientation theta_0 is.
            alphas (numpy array): Size (T,) array of float representing the
                strength of the external input. Optional.

        Returns:
            m (numpy array): Size (N,T) array of floats representing current
                of each unit at each time step
            f (numpy array): Size (N,T) array of floats representing firing
                rate of each unit at each time step.

        Raises:
            ValueError: If alphas is provided but is not the same size as input.
        """
 
        if (alphas is not None) and (input_ext.shape[0] != alphas.size):
            raise ValueError(
                "If alphas is provided, it should be the same size as input."
                )
        T = input_ext.shape[0]
        m = np.zeros((self.num_units, T)) # Current
        f = np.zeros((self.num_units, T)) # Firing rate
        m0 = 0.1*np.random.normal(0, 1, self.num_units)
        for t in range(T):
            alpha_t = 0 if alphas is None else alphas[t]
            if t == 0:
                m_t, f_t = self._step(m0, input_ext[t], alpha_t)
            else:
                m_t, f_t = self._step(m[:, t-1], input_ext[t], alpha_t)
            m[:,t] = m_t
            f[:,t] = f_t
        return m, f

    def _step(self, prev_m, input_t, alpha_t):
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

        if input_t == np.nan:
            h_ext = np.zeros(self.num_units)
            h_ext[:self.num_separate_units] = np.random.normal(
                0, 1, self.num_separate_units
                )
            for i in self.shared_unit_map[0]:
                J_idx = self.J_episode_indices[i]
                h_ext[J_idx] = np.random.normal(0, 1)
            h_ext *= alpha_t
        else:
            h_ext = np.zeros(self.num_units)
            episode_input = alpha_t*np.cos(input_t - self.thetas)
            for idx, J_idx in enumerate(self.J_episode_indices):
                h_ext[J_idx] = episode_input[idx]
        f_t = self.J @ self._g(prev_m) + self._g(h_ext)
        dmdt = -prev_m + f_t - self.K_inhib
        m_t = prev_m + self.dt*dmdt 
        return m_t, f_t

    def _init_networks(self):
        """ Determines which units are shared between the two networks """

        num_shared_units = int(self.N*self.overlap)
        shared_episode_units = np.random.choice(
            self.N, num_shared_units, replace=False
            )
        shared_place_units =  np.random.choice(
            self.N, num_shared_units, replace=False
            )
        shared_unit_map = np.vstack((shared_episode_units, shared_place_units))
        self.shared_unit_map = shared_unit_map
        self.num_shared_units = num_shared_units
        self.num_separate_units = self.N - num_shared_units
        self.num_units = self.N*2 - num_shared_units

    def _init_thetas(self):
        """ Initializes the preferred tuning within a network """

        thetas = np.linspace(0, 2*pi, self.N)
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
            weights = -self.J0 + self.J2*np.cos(self.thetas[i] - self.thetas[:])
            weights = np.delete(weights, self.shared_unit_map[0,:])
            J[curr_unit, :num_separate_units] = weights
            J_episode_indices[i] = int(curr_unit)
            curr_unit += 1

        # Fill in unshared place units
        for i in range(self.N):
            if i in self.shared_unit_map[1,:]:
                continue
            weights = -self.J0 + self.J2*np.cos(self.thetas[i] - self.thetas[:])
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
                weight = -self.J0 + self.J2*np.cos(
                    self.thetas[episode_unit] - self.thetas[j_ep]
                    )
                J[curr_unit, J_idx] += weight
                J[J_idx, curr_unit] += weight 
            for j_place in range(self.N):
                if j_place in self.shared_unit_map[1,:]:
                    continue
                J_idx = J_place_indices[j_place]
                weight = -self.J0 + self.J2*np.cos(
                    self.thetas[place_unit] - self.thetas[j_place]
                    )
                J[curr_unit, J_idx] += weight
                J[J_idx, curr_unit] += weight 
            for j_shared in np.arange(1, self.num_shared_units - i):
                j_shared_ep = self.shared_unit_map[0, j_shared]
                j_shared_pl = self.shared_unit_map[1, j_shared]
                weight_ep = -self.J0 + self.J2*np.cos(
                    self.thetas[episode_unit] - self.thetas[j_shared_ep]
                    )
                weight_place = -self.J0 + self.J2*np.cos(
                    self.thetas[place_unit] - self.thetas[j_shared_pl]
                    )
                J[curr_unit, curr_unit + j_shared] += (weight_ep + weight_place)
                J[curr_unit + j_shared, curr_unit] += (weight_ep + weight_place)
            J_episode_indices[episode_unit] = int(curr_unit)
            J_place_indices[place_unit] = int(curr_unit)
            curr_unit += 1

        # Remove self-excitation
        for i in range(self.num_units):
            J[i,i] = 0

        self.J_episode_indices = J_episode_indices
        self.J_place_indices = J_place_indices
        self.J = J

    def _g(self, f_t):
        """
        Rectifies and saturates a given firing rate.
        """

        return np.clip(f_t, 0, 1)

