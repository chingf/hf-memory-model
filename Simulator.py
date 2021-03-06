import numpy as np
from math import pi
from Network import OverlapNetwork
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

    def __init__(self, network, inputgen):
        self.network = network
        self.inputgen = inputgen
        self.inputgen.set_network(network)
        self.J_samplerate = self.inputgen.J_samplerate
        self.J_samples = []

    def simulate(self):
        """
        Simulates the behavior of the network over some period of time.
    
        Returns:
            m (numpy array): Size (N,T) array of floats representing current
                of each unit at each time step
            f (numpy array): Size (N,T) array of floats representing firing
                rate of each unit at each time step.
        """
 
        m = np.zeros((self.network.num_units, self.inputgen.T)) # Current
        f = np.zeros((self.network.num_units, self.inputgen.T)) # Firing rate
        m0 = 0.1*np.random.normal(0, 1, self.network.num_units)
        f0 = np.clip(m0, 0, 1)
        m0 = m0.reshape((-1, 1))
        f0 = f0.reshape((-1, 1))
        t = 0
        while True:
            try:
                input_t, alpha_t, fastlearn = self.inputgen.get_inputs()
            except StopIteration:
                break
            if t == 0:
                m_t, f_t = self.network.step(m0, f0, input_t, alpha_t, fastlearn)
            else:
                m_t, f_t = self.network.step(
                    m[:, :t], f[:, :t], input_t, alpha_t, fastlearn
                    )
            if self.J_samplerate > 0 and t % self.J_samplerate == 0:
                self.J_samples.append(self.network.J)
            m[:,t] = m_t
            f[:,t] = f_t
            self.inputgen.set_current_activity(f_t)
            t += 1
            self.network.t += 1
        return m, f
