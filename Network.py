import numpy as np
from sklearn.preprocessing import normalize
from math import pi

class OverlapNetwork(object):
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
    base_J2 = 5.5
    tau = 1 # time constant (in ms)
    dt = 0.1 #
    steps_in_s = (1/dt)*(1/tau)*50
    kappa = 4.
    vonmises_gain = 3.2
    norm_scale = 3.

    def __init__(
            self, N_pl, N_ep, K_pl, K_ep,
            overlap=0, num_internetwork_connections=3, num_ep_modules=3,
            add_feedback=False
            ):
        self.N_pl = N_pl
        self.N_ep = N_ep
        self.K_pl = K_pl
        self.K_ep = K_ep
        self.overlap = overlap
        self.num_internetwork_connections = num_internetwork_connections
        self.num_ep_modules = num_ep_modules
        self.add_feedback = add_feedback
        self.J0 = self.base_J0/N_pl
        self.J2 = self.base_J2/N_pl
        self._init_episode_modules()
        self._init_internetwork_units()
        self._init_shared_units()
        self._init_J()
        self._init_J_interactions()
        self.J = normalize(self.J, axis=1, norm="l1")*self.norm_scale
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
        total_input = self.J @ self._g(prev_m) + h_ext
        total_input[self.J_place_only] -= self.K_pl
        total_input[self.J_episode_only] -= self.K_ep
        if self.overlap != 0:
            total_input[self.J_shared] -= (self.K_pl + self.K_ep)/2
        try:
            dmdt = (-prev_m + total_input)/self.tau
        except:
            import pdb; pdb.set_trace()
        m_t = prev_m + self.dt*dmdt
        f_t = self._g(m_t)
        return m_t, f_t

    def _init_episode_modules(self):
        self.ep_modules = np.array_split(
            np.arange(self.N_ep).astype(int), self.num_ep_modules
            )

    def _init_internetwork_units(self):
        """ Determines which units connect between the two networks """

        if self.num_internetwork_connections == 0:
            ep_modules, place_units = []
        elif self.num_internetwork_connections == 1:
            ep_modules = [self.num_ep_modules//2]
            pl_units = [self.N_pl//2]
        else:
            ep_modules = np.linspace(
                0, self.num_ep_modules, self.num_internetwork_connections,
                endpoint=False
                )
            pl_units = np.linspace(
                0, self.N_pl, self.num_internetwork_connections, endpoint=False
                )
        self.internetwork_units = np.array([ep_modules, pl_units])

    def _init_shared_units(self):
        """ Determines which units are shared between the two networks """

        num_shared_units = int(self.N_ep*self.overlap)
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

    def _init_J(self):
        """ Initializes the connectivity matrix J """

        num_unshared_ep = self.N_ep - self.num_shared_units
        num_unshared_pl = self.N_pl - self.num_shared_units
        J = np.zeros((self.num_units, self.num_units))
        J_episode_indices = np.zeros(self.N_ep).astype(int)
        J_place_indices = np.zeros(self.N_pl).astype(int)
        J_episode_only = []
        J_place_only = []
        J_shared = []
        J_idx = 0

        # Fill in unshared episode network connectivity matrix
        ep_weight = 0.5
        ep_excit = ep_weight/(self.N_ep//self.num_ep_modules)
        ep_inhib = -ep_weight/(self.N_ep - self.N_ep//self.num_ep_modules)
        for m_i in range(self.num_ep_modules):
            module = self.ep_modules[m_i]
            for i in module:
                if i in self.shared_unit_map[0]: continue
                weights = np.ones(self.N_ep)*ep_inhib
                weights[module] = ep_excit
                weights = np.delete(weights, self.shared_unit_map[0])
                J[J_idx, :weights.size] = weights
                J_episode_indices[i] = int(J_idx)
                J_episode_only.append(int(J_idx))
                J_idx += 1

        # Fill in unshared place network connectivity matrix
        for i in range(self.N_pl):
            if i in self.shared_unit_map[1]: continue
            weights = self._get_vonmises(i)
            weights = np.delete(weights, self.shared_unit_map[1])
            J[J_idx, num_unshared_ep:num_unshared_ep + num_unshared_pl] = weights
            J_place_indices[i] = int(J_idx)
            J_place_only.append(int(J_idx))
            J_idx += 1

        # Fill in shared units for episode and place
        for i in range(self.num_shared_units):
            ep_unit = self.shared_unit_map[0, i]
            pl_unit = self.shared_unit_map[1, i]

            # Weights between shared unit and unshared episode unit
            ep_module_idx = np.argwhere(
                [ep_unit in m for m in self.ep_modules]
                )[0,0]
            ep_weights = np.ones(self.N_ep)*ep_inhib
            ep_weights[self.ep_modules[ep_module_idx]] = ep_excit
            for ep in range(self.N_ep):
                if ep in self.shared_unit_map[0]: continue
                J[J_idx, J_episode_indices[ep]] = ep_weights[ep]
                J[J_episode_indices[ep], J_idx] = ep_weights[ep]

            # Weights between shared unit and unshared place unit
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
                    ep_weights[j_ep_unit] + place_weights[j_pl_unit]
                    )
                J[J_idx, J_idx + j] = total_weight
                J[J_idx + j, J_idx] = total_weight

            J_episode_indices[ep_unit] = int(J_idx)
            J_place_indices[pl_unit] = int(J_idx)
            J_shared.append(int(J_idx))
            J_idx += 1

        self.J_episode_indices = J_episode_indices
        self.J_place_indices = J_place_indices
        self.J_episode_only = np.array(J_episode_only)
        self.J_place_only = np.array(J_place_only)
        self.J_shared = np.array(J_shared)
        self.J = J

    def _init_J_interactions(self):
        """ Adds the interactions between networks to J matrix """

        ep_modules, pl_units = self.internetwork_units
        scale = 1.
        for cache in range(ep_modules.size):
            ep_module = int(ep_modules[cache])
            ep_units = self.ep_modules[ep_module]
            pl_unit = int(pl_units[cache])
            pl_support = 4
            pl_support = np.arange(pl_unit - pl_support, pl_unit + pl_support + 1)
            pl_ep_weight = 0.03
            ep_pl_weight = 0.1
            pl_ep_weights = np.ones(self.N_ep)*-pl_ep_weight
            pl_ep_weights[ep_units] *= -1
            ep_pl_weights = np.ones(self.N_pl)*-ep_pl_weight
            ep_pl_weights[pl_support] *= -1
            for pl in pl_support:
                J_pl = self.J_place_indices[pl]
                self.J[self.J_episode_indices, J_pl] = pl_ep_weights
            for ep in ep_units:
                J_ep = self.J_episode_indices[ep]
                self.J[self.J_place_indices, J_ep] = ep_pl_weights
        for i in range(self.J.shape[0]):
            self.J[i,i] = 0

    def _g(self, f_t):
        """
        Rectifies and saturates a given firing rate.
        """

        return np.clip(f_t, 0, 1)

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
