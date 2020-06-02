import pickle
import itertools
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from math import pi, sin, cos
from MixedRNN import MixedRNN
from HebbRNN import HebbRNN
from BtspRNN import BtspRNN
from Simulator import Simulator
from Input import NavigationInput, EpisodeInput, TestFPInput, AssocInput
from Input import TestNavFPInput
from utils import *

def run_and_plot_assoc(
    noise_mean=-0., noise_std=0.1, J_mean=-0.1, J_std=0.1,
    ext_plasticity=1, plasticity=1.,
    ext_plasticity_scale=0.1, plasticity_scale=0.4
    ):
    """ Runs and plots a random network learning the ring structure. """

    network = MixedRNN(
        N_pl=100, N_ep=150, J_mean=J_mean, J_std=J_std,
        ext_plasticity_scale=ext_plasticity_scale, plasticity_scale=plasticity_scale
        )
    sim = Simulator()
    locs = np.linspace(0, 2*pi, 4, endpoint=False)
    # Form caches
    for idx, loc in enumerate(locs):
        inputgen = AssocInput(
            noise_mean=noise_mean, noise_std=noise_std, cache_loc=loc,
            ext_plasticity=ext_plasticity, plasticity=plasticity
            )
        m, f, inputs = sim.simulate(network, inputgen)
        plot_J(network.J, network, sortby=network.memories[idx], title="J Matrix")
        plot_J(
            network.J_ext, network, sortby=network.ext_memories[idx],
            title="Read-In Matrix"
            )
        plot_formation(
            f, network, inputs, sortby=network.memories[idx],
            title="Navigation and Association %d"%idx
            )
        plot_formation(
            f, network, inputs, sortby=network.memories[0],
            title="Navigation and Association %d (Sorted by RNN Memory 1)"%(idx+1)
            )

    memory_grid = np.array(network.memories)
    plt.figure(figsize=(8,4))
    plt.imshow(memory_grid, aspect="auto")
    plt.show()
    test_navfp(network)
    import pdb; pdb.set_trace()

def test_navfp(network=None):
    if network is None:
        with open("btsphebb4-0.p", "rb") as f:
            network = pickle.load(f)
        for memory in network.memories:
            plot_J(network.J, network, sortby=memory, title="J Matrix")
        for memory in network.ext_memories:
            plot_J(network.J_ext, network, sortby=memory, title="Read-In Matrix")
    sim = Simulator()
    locs = [60 for _ in range(10)]#np.arange(10, 100, 5)
    for loc in locs:
        print(loc)
        np.random.seed(1)
        inputgen = TestNavFPInput(recall_loc=loc, network=network)
        m, f, inputs = sim.simulate(network, inputgen)
        for idx, memory in enumerate(network.memories):
            plot_formation(
                f, network, inputs, sortby=memory,
                title="Recall (Sorted by RNN Memory %d)"%(idx+1)
                )

def eval_navfp(network=None):
    if network is None:
        with open("btsphebb4-0.p", "rb") as f:
            network = pickle.load(f)
    sim = Simulator(network)
    P = sim.eval(25, 40)
    with open("eval-bh4-0.p", "wb") as f:
        pickle.dump(P, f)

def main():
    #run_and_plot_assoc()
    #test_navfp()
    eval_navfp()

main()
