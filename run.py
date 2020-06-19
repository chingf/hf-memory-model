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
    ext_plasticity_scale=0.2, plasticity_scale=np.sqrt(0.25),
    plot=True
    ):
    """ Runs and plots a random network learning the ring structure. """

    network = MixedRNN(
        N_pl=200, N_ep=250, J_mean=J_mean, J_std=J_std,
        ext_plasticity_scale=ext_plasticity_scale, plasticity_scale=plasticity_scale
        )
    sim = Simulator(network)
    locs = np.linspace(0, 2*pi, 5, endpoint=False)
    results = {}
    results["Caching"] = []
    # Form caches
    for idx, loc in enumerate(locs):
        inputgen = AssocInput(
            noise_mean=noise_mean, noise_std=noise_std, cache_loc=loc,
            ext_plasticity=ext_plasticity, plasticity=plasticity
            )
        m, f, inputs = sim.simulate(inputgen)
        results["Caching"].append((m, f, inputs, network.J, network.J_ext))
        if plot:
            plot_J(
                network.J, network, sortby=network.memories[idx],
                title="J Matrix"
                )
            plot_J(
                network.J_ext, network, sortby=network.ext_memories[idx],
                title="Read-In Matrix"
                )
            plot_formation(
                f, network, inputs, sortby=network.memories[idx],
                title="Navigation and Association %d"%idx
                )
    results["Network"] = network
    memory_grid = np.array(network.memories)
    plt.figure(figsize=(8,4))
    plt.imshow(memory_grid, aspect="auto")
    plt.show()
    for memory in network.memories:
        plot_J(network.J, network, sortby=memory, title="J Matrix")
    for memory in network.ext_memories:
        plot_J(network.J_ext, network, sortby=memory, title="Read-In Matrix")
    plt.show()
    with open("200btsphebb5-0.p", "wb") as f:
        pickle.dump(results, f)
    test_navfp(network)
    import pdb; pdb.set_trace()

def test_navfp(network=None):
    if network is None:
        with open("btsphebb4-6.p", "rb") as f:
            network = pickle.load(f)
            if type(network) is dict:
                network = network["Network"]
        for memory in network.memories:
            plot_J(network.J, network, sortby=memory, title="J Matrix")
        for memory in network.ext_memories:
            plot_J(network.J_ext, network, sortby=memory, title="Read-In Matrix")
    sim = Simulator(network)
    locs = np.linspace(0, network.N_pl, 10, endpoint=False)
    for loc in locs:
        print(loc)
        #np.random.seed(1)
        inputgen = TestNavFPInput(recall_loc=loc, network=network)
        m, f, inputs = sim.simulate(inputgen)
        for idx, memory in enumerate(network.memories):
            plot_formation(
                f, network, inputs, sortby=memory,
                title="Recall (Sorted by RNN Memory %d)"%(idx+1)
                )

def eval_navfp(network=None):
    if network is None:
        with open("200btsphebb5-0.p", "rb") as f: #btsphebb4-5
            network = pickle.load(f)
            if type(network) is dict:
                network = network["Network"]
    sim = Simulator(network)
    results = sim.eval(20, 10)
    with open("eval.p", "wb") as f:
        pickle.dump(results, f)

def main():
    with open("400btsphebb5-0.p", "rb") as f:
        results = pickle.load(f)
        net = results["Network"]
    run_and_plot_assoc(plot=False)
    #test_navfp(net)
    #eval_navfp()

main()
