import numpy as np
from math import pi

class WTANetwork(object):
    """
    A winner-take-all network. For now, all modules are the same size.
    """

    dt = 0.1

    def __init__(self, N, num_modules, uniform_inhib=0.):
        self.N = N
        self.num_modules = num_modules
        self.uniform_inhib = uniform_inhib
        num_excit = N//num_modules
        num_inhib = N - num_excit
        total_weight = .3
        self.excit = total_weight/num_excit 
        self.inhib = -total_weight/num_inhib 
        self._init_modules()
        self._init_J()

    def simulate(self, T):
        """
        Simulates the behavior of the network over some period of time.
    
        Returns:
            m (numpy array): Size (N,T) array of floats representing current
                of each unit at each time step
            f (numpy array): Size (N,T) array of floats representing firing
                rate of each unit at each time step.
        """
 
        m = np.zeros((self.N, T)) # Current
        f = np.zeros((self.N, T)) # Firing rate
        inputs = np.zeros((self.N, T))
        m0 = 0.1*np.random.normal(0, 1, self.N)
        thetas = np.linspace(0, 2*pi, self.N, endpoint=False)
        for t in range(T): # 2pi/5 sec
            loc = ((t%200)/200)*2*pi
            loc = 0 if t < T//2 else pi/2
            input_t = np.cos(loc - thetas)
            input_t[input_t < 0] = 0
            #input_t = 0.1*np.random.normal(0, 1, self.N)
            alpha_t = 0.6
            if t == 0:
                m_t, f_t = self.step(m0, input_t, alpha_t)
            else:
                m_t, f_t = self.step(m[:, t-1], input_t, alpha_t)
            m[:,t] = m_t
            f[:,t] = f_t
            inputs[:,t] = input_t*alpha_t
        return m, f, inputs

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

        h_ext = input_t*alpha_t
        f_t = self.J @ self._g(prev_m) + self._g(h_ext)
        dmdt = -prev_m + f_t - self.uniform_inhib
        m_t = prev_m + self.dt*dmdt 
        return m_t, f_t

    def _init_modules(self):
        """ Determines which units are assigned to which module. """

        self.modules = np.array_split(np.arange(self.N), self.num_modules)

    def _init_J(self):
        """ Sets the weights between two neurons. """

        J = np.zeros((self.N, self.N))
        for m_i in range(self.num_modules):
            # Intra-excitation
            module = self.modules[m_i]
            for i1 in module:
                for i2 in module:
                    if i1 == i2:
                        continue
                    J[i1, i2] = self.excit

            # Cross-inhibition
            for m_j in range(self.num_modules):
                if m_i == m_j:
                    continue
                module_i = self.modules[m_i]
                module_j = self.modules[m_j]
                for i in module_i:
                    for j in module_j:
                        J[i, j] = self.inhib
        self.J = J

    def _g(self, f_t):
        """
        Rectifies and saturates a given firing rate.
        """

        return np.clip(f_t, 0, 1)

