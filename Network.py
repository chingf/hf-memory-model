import numpy as np

class Network():
    """
    A recurrently connected network

    Args:
        N: Integer count of units in the network.
        g: Positive float; the network gain

    Attributes:
        g: Positive float; the network gain
        N: Integer count of units in the network
        x: (N,) Numpy array. The ith element is the current electrical activity
            of the ith unit. This may be subthreshold.
        J: (N,N) Numpy array. The synaptic weights between units. J_ij is the
            weight from j into i.
    """

    def __init__(self, g, N):
        self.g = g
        self.N = N
        self._init_activity()
        self._init_synaptic_weights()

    def get_activity(self):
        """
        Returns the current network activity, phi(x)
        """

        r0 = 0.1
        x_leq0 = np.where(self.x <= 0) # Indices where x <= 0
        x_g0 = np.where(self.x > 0) # Indices where x > 0
        phi_x = self.x.copy()
        phi_x[x_leq0] = r0*np.tanh(self.x[x_leq0]/r0)
        phi_x[x_g0] = (2-r0)*np.tanh(self.x[x_g0]/(2-r0))
        return phi_x

    def step(self):
        """
        Simulates the network for one time step. Evolves the current network
        activity according to the defined first-order dynamics.
        """

        next_x = []
        for i in range(self.N):
            x_i = self.x[i]
            dx_i = self.get_dx(i)
            next_x.append(x_i + dx_i)
        self.x = np.array(next_x)

    def get_dx(self, i):
        """
        Calculates dx_i/dt. A unit of t is 10 ms

        Args
            i: Integer indexing into units.
        Returns:
            The change in x_i over time, where x_i is the activity of the ith
            unit.
        """

        r0 = 0.1
        phi_x = self.get_activity()
        input_to_unit = np.dot(self.J[i,:], phi_x)
        return -self.x[i] + input_to_unit

    def _init_activity(self):
        """
        Populates an array with randomly initialized unit activity
        """

        self.x = np.random.normal(0, 5, self.N)

    def _init_synaptic_weights(self):
        """
        Initialize synaptic weights. There are no self weights. For each i,j
        where i != j, J_ij ~ Normal(0, 1/N)
        """

        J_std = np.sqrt((self.g**2)/self.N)
        self.J = np.random.normal(0, J_std, (self.N, self.N))
        for n in range(self.N):
            self.J[n,n] = 0.
