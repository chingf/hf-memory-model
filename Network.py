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
        interacting_units
    """

    base_J0 = 0.3 
    base_J2 = 5.
    dt = 0.1

    def __init__(
            self, N, K_inhib, overlap=0,
            add_feedback=False, add_attractor=False
            ):
        self.N = N
        self.K_inhib = K_inhib
        self.overlap = overlap
        self.add_feedback = add_feedback
        self.add_attractor = add_attractor
        self.J0 = self.base_J0/N
        self.J2 = self.base_J2/N
        self._init_shared_units()
        self._init_thetas()
        self._init_J()
        self._init_J_interactions()
        self._init_episode_attractors()

    def simulate(self, input_ext, alphas):
        """
        Simulates the behavior of the ring attractor over some period of time.
        The input only arrives to units involved in the episode network.
    
        Args:
            input_ext (numpy array): Size (T,) array of float radians representing
                the external stimulus. Here, the stimulus is some external cue
                that indicates what the correct orientation theta_0 is.
            alphas (numpy array): Size (T,) array of float representing the
                strength of the external input.

        Returns:
            m (numpy array): Size (N,T) array of floats representing current
                of each unit at each time step
            f (numpy array): Size (N,T) array of floats representing firing
                rate of each unit at each time step.

        Raises:
            ValueError: If alphas is not the same size as input.
        """
 
        if input_ext.shape[0] != alphas.size:
            raise ValueError("Alphas should be the same size as input.")
        T = input_ext.shape[0]
        m = np.zeros((self.num_units, T)) # Current
        f = np.zeros((self.num_units, T)) # Firing rate
        m0 = 0.1*np.random.normal(0, 1, self.num_units)
        for t in range(T):
            alpha_t = alphas[t]
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

    def _init_shared_units(self):
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
                weight_place = self._get_weight_value(place_unit, j_shared_pl)
                total_weight = -self.J0 + (weight_ep + weight_place)/2
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

        episode_units = [0, self.N//3, 2*self.N//3]
        place_units = [0, self.N//3, 2*self.N//3]
        self.interacting_units = np.array([episode_units, place_units])
        interaction_support = np.arange(-16, 17)
        interaction_peakwidth = 20 

        for idx, episode_unit in enumerate(episode_units):
            episode_unit = self.J_episode_indices[episode_unit]
            for i in interaction_support:
                weight_offset = self._get_weight_value(i, 0, interaction_peakwidth)
                place_unit = (place_units[idx]+i)%self.N + self.num_separate_units
                self.J[place_unit, episode_unit] += weight_offset

        if self.add_feedback:
            for idx, place_unit in enumerate(place_units):
                place_unit = self.J_place_indices[place_unit]
                for i in interaction_support:
                    weight_offset = self._get_weight_value(i, 0, interaction_peakwidth)
                    weight_offset *= 1
                    episode_unit = (episode_units[idx]+i)%self.N
                    self.J[episode_unit, place_unit] += weight_offset

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
                        self._get_sharp_cos(), attractor_idx - self.N//2
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

    def _get_weight_vector(self, i, peakwidth=30):
        sharp_cos = np.roll(self._get_sharp_cos(peakwidth), i - self.N//2)
        return -self.J0 + self.J2*sharp_cos

    def _get_weight_value(self, i ,j, peakwidth=30):
        sharp_cos = np.roll(self._get_sharp_cos(peakwidth), i - self.N//2)
        return self.J0 + self.J2*sharp_cos[j]

    def _get_sharp_cos(self, peakwidth=18):
        """ Returns a sharp sinusoidal curve that drops off rapidly """

        cos_bump = np.cos(pi-np.linspace(0, 2*pi, 3*peakwidth, endpoint=False))
        flat_inhibition = np.ones((self.N - peakwidth*3)//2)*-1
        curve = np.concatenate(
            (flat_inhibition, cos_bump, flat_inhibition)
            )
        return curve

