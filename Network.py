import numpy as np
from math import pi

class RingNetwork():
    """
    A ring attractor network as described by Ben-Yishai & Sompolinsky (1994).

    In this model, one timestep corresponds to 100 ms. Wherever possible,
    variable notation corresponds with the notation used in the original paper.

    Args:
        N (int): Number of units in the network. Their preferred tuning evenly
           covers the ring

    Attributes:
        base_J0 (float): parameter representing uniform all-to-all inhibition.
        base_J2 (float): parameter representing amplitude of angle-specific
            interaction.
        dt (float): parameter representing the size of one timestep.
        J0 (float): parameter base_J0 normalized for the number of units.
        J2 (float): parameter base_J2 normalized for the number of units.
        N (int): integer number of units in the network. Thus unit i will
            represent neurons with preferred tuning at 2pi/i radians.
        J (numpy array): (N, N) array of floats representing the connectivity
            between units. V_ij will represent the connection from j to i.
        thetas (numpy array): (N,) array of float radians. The value at i
            represents the preferred tuning of unit i: 2pi/i
    """

    base_J0 = 0.3 
    base_J2 = 4.5
    dt = 0.1

    def __init__(self, N):
        self.N = N
        self.J0 = self.base_J0/N
        self.J2 = self.base_J2/N
        self.thetas = np.linspace(0, 2*pi, N)
        self._init_J()

    def simulate(self, input, alphas=None):
        """
        Simulates the behavior of the ring attractor over some period of time.
    
        Args:
            input (numpy array): Size (T,) array of float radians representing
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
 
        if (alphas is not None) and (input.size != alphas.size):
            raise ValueError(
                "If alphas is provided, it should be the same size as input."
                )
        T = input.size
        m = np.zeros((self.N, T)) # Current
        f = np.zeros((self.N, T)) # Firing rate
        m0 = 0.1*np.random.normal(0, 1, self.N)
        for t in range(T):
            alpha_t = 0 if alphas is None else alphas[t]
            if t == 0:
                m_t, f_t = self._step(m0, input[t], alpha_t)
            else:
                m_t, f_t = self._step(m[:, t-1], input[t], alpha_t)
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

        h_ext = alpha_t*np.cos(input_t - self.thetas)
        f_t = self.J @ prev_m + h_ext
        dmdt = -prev_m + self._g(f_t)
        m_t = prev_m + self.dt*dmdt #TODO: see how this changes with odeint
        return m_t, f_t

    def _g(self, f_t):
        """
        Rectifies and saturates a given firing rate.
        """

        return np.clip(f_t, 0, 1)

    def _init_J(self):
        """
        Initializes the connectivity matrix J
        """

        J = np.zeros((self.N,self.N))
        for i in range(self.N):
            J[i,:]= -self.J0 + self.J2*np.cos(self.thetas[i] - self.thetas)
        self.J = J

class MixedNetwork(RingNetwork):
    """
    A ring attractor network with context units external to the main network.

    In this model, one timestep corresponds to 100 ms. 

    Args:
        N (int): Number of units in the network. Their preferred tuning evenly
           covers the ring
        N_c (int): Number of context units.
        C (float): parameter for the constant gain added to dmdt during the
            context mode.
        indices_c (numpy array): Optional; size (N_c,) array containing the
            indices of the context units. If not provided, they will be randomly
            drawn.

    Attributes:
        base_J0 (float): parameter representing uniform all-to-all inhibition.
        base_J2 (float): parameter representing amplitude of angle-specific
            interaction.
        dt (float): parameter representing the size of one timestep.
        J0 (float): parameter base_J0 normalized for the number of units.
        J2 (float): parameter base_J2 normalized for the number of units.
        N (int): integer number of units in the network. Thus unit i will
            represent neurons with preferred tuning at 2pi/i radians.
        N_c (int): Number of external context units.
        C (float): parameter for the constant gain added to dmdt during the
            context mode.
        indices_c (numpy array): Optional; size (N_c,) array containing the
            indices of the context units. If not provided, they will be randomly
            drawn.
        J (numpy array): (N, N) array of floats representing the connectivity
            between units. V_ij will represent the connection from j to i.
        thetas (numpy array): (N,) array of float radians. The value at i
            represents the preferred tuning of unit i: 2pi/i

    Raises:
        ValueError: If indices_c.size != N_c
    """

    def __init__(self, N, N_c, C, indices_c=None):
        self.N = N
        self.N_c = N_c
        self.C = C
        self.J0 = self.base_J0/N
        self.J2 = self.base_J2/N
        self.thetas = np.linspace(0, 2*pi, N)
        if indices_c is None:
            self.indices_c = np.random.choice(N, N_c, replace=False)
        else:
            if indices_c.size != N_c:
                raise ValueError("Context unit indices are of incorrect size.")
            self.indices_c = indices_c
        self._init_J()

    def simulate(self, input, input_c, alphas=None):
        """
        Simulates the behavior of the ring attractor over some period of time.
    
        Args:
            input (numpy array): Size (T,) array of float radians representing
                the external stimulus. Here, the stimulus is some external cue
                that indicates what the correct orientation theta_0 is.
            input_c (numpy array): Size (T, N_c) array of floats representing
                the activation of context units over time.
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
 
        if (alphas is not None) and (input.size != alphas.size):
            raise ValueError(
                "If alphas is provided, it should be the same size as input."
                )
        T = input.size
        m = np.zeros((self.N, T)) # Current
        f = np.zeros((self.N, T)) # Firing rate
        m0 = 0.1*np.random.normal(0, 1, self.N)
        for t in range(T):
            alpha_t = 0 if alphas is None else alphas[t]
            if t == 0:
                m_t, f_t = self._step(m0, input[t], input_c[t], alpha_t)
            else:
                m_t, f_t = self._step(m[:, t-1], input[t], input_c[t], alpha_t)
            m[:,t] = m_t
            f[:,t] = f_t
        return m, f

    def _step(self, prev_m, input_t, input_c_t, alpha_t):
        """
        Steps the network forward one time step. Evolves the current network
        activity according to the defined first-order dynamics.

        Args:
            prev_m (numpy array): (N,) size array of floats; the current
            input_t (float): Radian representing the external stimulus.
            input_c_t (numpy array): (N_c,) size array of floats; the activation
                of context units at this time step.
            alpha_t (float): The strength of the external stimulus

        Returns:
            m_{t} and f_{t}: numpy arrays representing the current and the
                firing rates, respectively, of each unit in the next time step.
        """

        h_ext = alpha_t*np.cos(input_t - self.thetas)
        f_t = self.J @ prev_m + h_ext
        c_offset = np.zeros(self.N)
        c_offset[self.indices_c] = input_c_t
        c_offset *= self.C
        dmdt = -prev_m + self._g(f_t) + c_offset
        m_t = prev_m + self.dt*dmdt #TODO: see how this changes with odeint
        return m_t, f_t
