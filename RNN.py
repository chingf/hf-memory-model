import numpy as np
from Network import Network

class RNN():
    """
    A recurrent neural network.
    
    See Sompolinsky (1988), Abbott & Sompolinsky (2010), and
    Sussillo & Abbott (2009).

    Args:
        N: number of units in the network
        input_size: Dimension of input driving the network
        output_size: Dimension of network output.

    Attributes:
        N: Integer count of units in the network
        U: (input_size, N) Numpy array. The weights from the input to the RNN
        W: (N, output_size) Numpy array. The weights to generate RNN output
        g: Positive float; the network gain
        network: Network object; the recurrent network
        input_size: Dimension of input driving the network.
        output_size: Dimension of network output.

    Raises:
        ValueError: If gain is negative
    """

    def __init__(self, g, N, input_size=None, output_size=None):
        if g < 0:
            raise ValueError("Network gain cannot be negative.")
        self.g = g
        self.N = N
        self._init_inputs(input_size)
        self._init_outputs(output_size)
        self._init_network(N)

    def simulate(self, T):
        """
        Runs the RNN for some amount of time.

        Args:
            T: Integer count of number of time steps

        Returns:
            A (output_size, T) numpy array
        """

        output = np.zeros((self.output_size, T))
        output[:, 0] = self.network.get_activity() @ self.W
        for t in np.arange(1,T):
            self.network.step()
            network_activity = self.network.get_activity()
            output[:, t] = network_activity @ self.W
        return output

    def _init_inputs(self, input_size):
        if input_size is None:
            self.U = None 
        else:
            self.U = np.random.uniform(-1, 1, (input_size, self.N))
        self.input_size = input_size

    def _init_outputs(self, output_size):
        if output_size is None:
            self.W = np.eye(self.N)
            self.output_size = self.N
        else:
            self.W = np.random.uniform(-1, 1, (self.N, output_size))
            self.output_size = output_size

    def _init_network(self, N):
        self.network = Network(self.g, N)

