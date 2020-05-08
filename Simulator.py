import numpy as np
from math import pi
from PlotMaker import PlotMaker
from Input import *

class Simulator(object):
    """
    An object that wraps around a Network and an Input. Simulator will query
    Input at each time step and feed the input into Network. Simulator will
    collect the results of the Network. Here, one timestep corresponds to 100 ms

    Args:
        network: A Network object
        inputgen: An Input object that generates inputs to be fed to network
    """

    def simulate(self, network, inputgen):
        """
        Simulates the behavior of the network over some period of time.
    
        Returns:
            m (numpy array): Size (N,T) array of floats representing current
                of each unit at each time step
            f (numpy array): Size (N,T) array of floats representing firing
                rate of each unit at each time step.
        """

        inputgen.set_network(network)
        m = np.zeros((network.num_units, inputgen.T)) # Current
        f = np.zeros((network.num_units, inputgen.T)) # Firing rate
        inputs = np.zeros((network.num_units, inputgen.T)) # Input
        m0 = 0.1*np.random.normal(0, 1, network.num_units)
        f0 = np.clip(m0, 0, 1)
        m0 = m0.reshape((-1, 1))
        f0 = f0.reshape((-1, 1))
        t = 0
        while True:
            try:
                input_t, btsp = inputgen.get_inputs()
                inputs[:, t] = input_t
            except StopIteration:
                break
            if t == 0:
                m_t, f_t = network.step(inputs, m0, f0, btsp)
            else:
                m_t, f_t = network.step(
                    inputs[:, :t], m[:, :t], f[:, :t], btsp
                    )
            m[:,t] = m_t
            f[:,t] = f_t
            inputgen.set_current_activity(f_t)
            t += 1
            network.t += 1
        return m, f, inputs

