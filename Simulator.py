import numpy as np
from math import pi
from Input import *
from utils import *
from multiprocessing import Pool

class Simulator(object):
    """
    An object that wraps around a Network and an Input. Simulator will query
    Input at each time step and feed the input into Network. Simulator will
    collect the results of the Network. Here, one timestep corresponds to 100 ms

    Args:
        network: A Network object
        inputgen: An Input object that generates inputs to be fed to network
    """

    def __init__(self, network):
        self.network = network

    def simulate(self, inputgen):
        """
        Simulates the behavior of the network over some period of time.
    
        Returns:
            m (numpy array): Size (N,T) array of floats representing current
                of each unit at each time step
            f (numpy array): Size (N,T) array of floats representing firing
                rate of each unit at each time step.
        """

        network = self.network
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
                input_t, plasticity, ext_plasticity, inhib = inputgen.get_inputs()
                inputs[:, t] = input_t
            except StopIteration:
                break
            if t == 0:
                m_t, f_t = network.step(
                    inputs, m0, f0, plasticity, ext_plasticity, inhib
                    )
            else:
                m_t, f_t = network.step(
                    inputs[:, :t], m[:, :t], f[:, :t],
                    plasticity, ext_plasticity, inhib
                    )
            m[:,t] = m_t
            f[:,t] = f_t
            inputgen.set_current_activity(f_t)
            t += 1
            network.t += 1
        return m, f, inputs

    def eval(
        self, num_eval_locs, num_iters, multiprocess=True,
        plot=False, shorten=True
        ):
        """
        Evaluates network behavior by running memory retrieval on the network at
        various locations. Transient activation of a place cell is sufficent to
        be classifed as a memory retrieval. In the absence of this, transient
        activation of the episode cell pattern will be necessary for retrieval
        classification.
    
        Returns:
            P (numpy array): Size (num_locs, num_memories + 1) array of floats
                representing the probability of recalling a memory at a given
                location. The probability is estimated by running the same
                input for ITERS number of times. The values at P[:,-1] will
                represent the probability of no memories being recalled
        """

        locs = np.linspace(0, 2*pi, num_eval_locs, endpoint=False)
        network = self.network
        P = np.zeros((num_eval_locs, len(network.memories) + 1))
        FR = [[] for _ in range(num_eval_locs)]
        args = [
            (loc_idx, loc, num_iters, plot, shorten) for loc_idx, loc in enumerate(locs)
            ]
        if multiprocess:
            pool = Pool(processes=5)
            pool_results = pool.starmap(self._eval_pool_func, args)
            pool.close()
            pool.join()
        else:
            pool_results = []
            for arg in args:
                pool_results.append(self._eval_pool_func(*arg))
        for pool_result in pool_results:
            loc_idx, P_i, FR_i = pool_result
            P[loc_idx, :] = P_i
            FR[loc_idx] = FR_i
        P = P/num_iters
        results = {"P": P, "FR": FR}
        return results

    def _eval_pool_func(self, loc_idx, loc, num_iters, plot, shorten):
        network = self.network
        loc = (loc/(2*pi))*network.N_pl
        P_i = np.zeros(len(network.memories) + 1)
        FR_i = [[] for _ in range(len(network.memories) + 1)]
        for _ in range(num_iters):
            inputgen = TestNavFPInput(recall_loc=loc, network=network, shorten=shorten)
            m, f, inputs = self.simulate(inputgen)
            if plot:
                plot_formation(
                    f, network, inputs, sortby=network.memories[0], title="Recall"
                    )
            memories = network.memories
            recall_start = inputgen.recall_start
            recall_inhib_start = inputgen.recall_inhib_start
            recall_frames = np.arange(recall_start, recall_inhib_start).astype(int)
            recalled_mem = -1
            recalled_mem_strength = -1
            for idx, memory in enumerate(network.memories):
                binary_mem = memory.copy()
                binary_mem[binary_mem > 0] = 1
                binary_mem[binary_mem < 0] = -1
                mem_support = np.sum(binary_mem > 0)
                recall_strength = np.mean(f[:,recall_frames[-10:]], axis=1)
                binary_recall = recall_strength.copy()
                binary_recall[recall_strength > 0.001] = 1
                mem_match = np.sum(binary_mem*binary_recall)/mem_support
                if mem_match > 0.1:
                    recalled_mem = idx
                    recalled_mem_strength = np.mean(recall_strength[binary_mem > 0])
                    break
            P_i[recalled_mem] += 1
            FR_i[recalled_mem].append(recalled_mem_strength)
        return loc_idx, P_i, FR_i
