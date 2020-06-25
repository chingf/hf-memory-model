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
from Simulator import Simulator
from Input import NavigationInput, EpisodeInput, TestFPInput, AssocInput
from Input import TestNavFPInput
from utils import *

def run_assoc(
    N_pl=100, N_ep=150, num_locs=3, plot=False, save_pickle=False,
    ext_plasticity_scale=0.2, plasticity_scale=np.sqrt(0.25)
    ):
    """ Runs and plots a random network learning the ring structure. """

    network = MixedRNN(
        N_pl=N_pl, N_ep=N_ep, ext_plasticity_scale=ext_plasticity_scale,
        plasticity_scale=plasticity_scale
        )
    sim = Simulator(network)
    locs = np.linspace(0, 2*pi, num_locs, endpoint=False)
    results = {}
    results["Caching"] = []
    # Form caches
    for idx, loc in enumerate(locs):
        inputgen = AssocInput(cache_loc=loc)
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
    if plot:
        memory_grid = np.array(network.memories)
        plt.figure(figsize=(8,4))
        plt.imshow(memory_grid, aspect="auto")
        plt.show()
        for memory in network.memories:
            plot_J(network.J, network, sortby=memory, title="J Matrix")
        for memory in network.ext_memories:
            plot_J(network.J_ext, network, sortby=memory, title="Read-In Matrix")
        plt.show()
    if save_pickle:
        with open("sim_results.p", "wb") as f:
            pickle.dump(results, f)
    return results

def run_nav(
    N_pl=200, N_ep=0, save_pickle=False,
    ext_plasticity_scale=0.1, plasticity_scale=np.sqrt(0.1)
    ):
    """ Runs and plots a random network learning the ring structure. """

    network = MixedRNN(
        N_pl=N_pl, N_ep=N_ep, ext_plasticity_scale=ext_plasticity_scale,
        plasticity_scale=plasticity_scale, init_ring=False
        )
    plot_J(network.J, network, title="J Matrix (Pre-Learning)")
    plot_J(network.J_ext, network, title="Read-In Matrix (Pre-Learning)")
    sim = Simulator(network)
    inputgen = NavigationInput()
    m, f, inputs = sim.simulate(inputgen)
    results = {}
    results["Navigating"] = [m, f, inputs]
    results["Network"] = network
    plot_J(network.J, network, title="J Matrix")
    plot_J(network.J_ext, network, title="Read-In Matrix")
    plot_formation(f, network, inputs, title="Navigation")
    if save_pickle:
        with open("nav_results.p", "wb") as f:
            pickle.dump(results, f)
    return results

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
    locs = np.linspace(0, network.N_pl, 3, endpoint=False)
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

def eval_navfp(network=None, num_eval_locs=20, num_iters=10, save_pickle=False):
    if network is None:
        with open("200btsphebb5-0.p", "rb") as f: #btsphebb4-5
            network = pickle.load(f)
            if type(network) is dict:
                network = network["Network"]
    sim = Simulator(network)
    results = sim.eval(num_eval_locs, num_iters)
    if save_pickle:
        with open("eval_results.p", "wb") as f:
            pickle.dump(results, f)
    return results

def gridsearch_pool_func(N_pl, N_ep, num_locs, ext_plasticity_scale, plasticity_scale):
    assoc_results = run_assoc(
        N_pl=N_pl, N_ep=N_ep, num_locs=num_locs, 
        ext_plasticity_scale=ext_plasticity_scale,
        plasticity_scale=plasticity_scale
        )
    sim = Simulator(assoc_results["Network"])
    num_eval_locs = num_locs*2
    eval_results = sim.eval(num_eval_locs, num_locs*4, multiprocess=False)
    P = eval_results["P"]
    self_recall_P = P[::2,:num_locs]
    self_recall_threshold = 1.5/num_locs
    self_recall_failed = np.any(np.diagonal(self_recall_P) < self_recall_threshold)
    spont_recall_P = 1 - P[1::2,-1]
    spont_recall_threshold = 0.75/num_locs
    spont_recall_failed = np.any(spont_recall_P < spont_recall_threshold)
    print(np.diag(self_recall_P))
    print(spont_recall_P)
    print()
    sim.eval(num_eval_locs, 2, multiprocess=False, plot=True, shorten=False)
    if self_recall_failed or spont_recall_failed:
        return 0
    print("success")
    return 1 - np.mean(spont_recall_P)

def gridsearch():
    N_ep_grid = np.array([150])
    ep_pl_ratio = 1.5
    plasticity_grid = np.array([0.25, 0.35, 0.4, 0.45, 0.5])
    gridsearch_results = {}
    gridsearch_matrices = {}
    for N_ep in N_ep_grid:
        N_pl = int(N_ep/ep_pl_ratio)
        gridsearch_matrix = np.zeros((len(plasticity_grid), len(plasticity_grid)))
        num_locs = 3
        while True:
            for i, plasticity_scale in enumerate(np.sqrt(plasticity_grid)):
                for j, ext_plasticity_scale in enumerate(plasticity_grid):
                    if i != j: continue
                    print(str(i) + ", " + str(j) + ": " + str(num_locs) + "\n")
                    val = 0
                    num_iters = 3
                    for _ in range(num_iters):
                        val += gridsearch_pool_func(
                            N_pl, N_ep, num_locs,
                            ext_plasticity_scale, plasticity_scale
                            )
                    val = val/num_iters
                    gridsearch_matrix[i,j] = val
            print("==========")
            print(gridsearch_matrix)
            print("==========\n")
            if np.any(gridsearch_matrix > 0):
                gridsearch_matrices[N_ep] = [num_locs, gridsearch_matrix]
                num_locs += 1
            else:
                break
    gridsearch_results["Matrices"] = gridsearch_matrices
    gridsearch_results["N_ep_grid"] = N_ep_grid
    gridsearch_results["ep_pl_ratio"] = ep_pl_ratio
    gridsearch_results["plasticity_grid"] = plasticity_grid
    with open("gridsearch_results.p", "wb") as f:
        pickle.dump(gridsearch_results, f)

def main():
    #results = run_assoc()
    #test_navfp(results["Network"])
    #gridsearch()
    run_nav()

main()
