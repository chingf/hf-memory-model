import pdb
import numpy as np
from math import pi
import matplotlib.pyplot as plt

class RingNetwork(object):
    """
    A ring attractor network as described by Ben-Yishai & Sompolinsky (1994).

    In this model, one timestep corresponds to 100 ms. Wherever possible,
    variable notation corresponds with the notation used in the original paper.

    Args:
        N (int): Number of units in the network. Their preferred tuning evenly
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
        thetas (numpy array): (N,) array of float radians. The value at i
            represents the preferred tuning of unit i: 2pi/i
    """

    base_J0 = 0.3 
    base_J2 = 5.
    dt = 0.1

    def __init__(self, N, K_inhib):
        self.N = N
        self.K_inhib = K_inhib
        self.J0 = self.base_J0/N
        self.J2 = self.base_J2/N
        self._init_thetas()
        self._init_J()

    def simulate(self, input_ext, alphas=None):
        """
        Simulates the behavior of the ring attractor over some period of time.
    
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
        m = np.zeros((self.N, T)) # Current
        f = np.zeros((self.N, T)) # Firing rate
        m0 = 0.1*np.random.normal(0, 1, self.N)
        m0[:5] += 0.02
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

        if input_t.size == 1:
            h_ext = alpha_t*np.cos(input_t - self.thetas[:,0])
        else:
            h_ext = alpha_t*input_t
        f_t = self.J @ self._g(prev_m) + self._g(h_ext)
        dmdt = -prev_m + f_t - self.K_inhib
        m_t = prev_m + self.dt*dmdt 
        return m_t, f_t

    def _g(self, f_t):
        """
        Rectifies and saturates a given firing rate.
        """

        return np.clip(f_t, 0, 1)

    def _init_thetas(self):
        """ Initializes the preferred tuning of each unit """

        self.thetas = np.linspace(0, 2*pi, self.N)

    def _init_J(self):
        """
        Initializes the connectivity matrix J
        """

        J = np.zeros((self.N,self.N))
        for i in range(self.N):
            J[i,:]= -self.J0 + self.J2*np.cos(self.thetas[i] - self.thetas)
        self.J = J

class RemapNetwork(RingNetwork):
    """
    An attractor network made implicity of two separate ring attractors. The
    units of one ring attractor are 'remapped' to form the second ring attractor.

    In this model, one timestep corresponds to 100 ms. 

    Args:
        N (int): Number of units in the network. Their preferred tuning evenly
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
    """

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

        if input_t.size == 1:
            h_ext = alpha_t*np.cos(input_t - self.thetas[:,0]*2)
        else:
            h_ext = alpha_t*input_t
        f_t = self.J @ self._g(prev_m) + self._g(h_ext)
        dmdt = -prev_m + f_t - self.K_inhib
        m_t = prev_m + self.dt*dmdt 
        return m_t, f_t

    def _init_thetas(self):
        """ Initializes the preferred tuning of each unit """

        remap_order = np.arange(self.N)
        np.random.shuffle(remap_order)
        thetas = np.zeros((self.N, 2))
        thetas[:,0] = np.linspace(0, 2*pi, self.N)
        thetas[:,1] = np.linspace(0, 2*pi, self.N)[remap_order]
        self.thetas = thetas
        self.remap_order = remap_order

    def _init_J(self):
        """ Initializes the connectivity matrix J """

        J = np.zeros((self.N,self.N))
        for i in range(self.N):
            sharp_cos = self._get_sharp_cos()
            net1_J = self.J2*np.roll(sharp_cos, i - self.N//2)
            net2_J = self.J2*np.cos(self.thetas[i,1] - self.thetas[:,1])
            J[i,:]= -self.J0 + (net1_J + net2_J)
        for i in range(10):
            offset = 0.2
            J[25, 0 + i + 1] += offset
            J[25, 0 - i - 1] += offset
            J[75, 30 + i + 1] += offset
            J[75, 30 - i - 1] += offset

        for i in range(self.N):
            J[i,i] = 0

        self.J = J

    def _get_sharp_cos(self):
        """ Returns a sharp sinusoidal curve that drops off rapidly """

        intra_peakwidth = 16
        cos_bump = np.cos(pi-np.linspace(0, 2*pi, 3*intra_peakwidth))
        flat_inhibition = np.ones((self.N - intra_peakwidth*3)//2) * -1
        curve = np.concatenate(
            (flat_inhibition, cos_bump, flat_inhibition)
            )
        return curve

