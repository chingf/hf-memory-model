import pdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from math import pi
from PlotMaker import PlotMaker
from GridSearch import GridSearch
from Network import RingNetwork, SimpleMixedNetwork, MixedNetwork, PlasticMixedNetwork

def gridsearch(overlap, name):
    scores, std = GridSearch(overlap).run_search()
    pdb.set_trace()
    results = {}
    results['scores'] = scores
    results['std'] = std
    with open("gridsearch-" + name + ".p", "wb") as f:
        pickle.dump(results, f)

def plot():
    plotmaker = PlotMaker()
    plotmaker.plot_slider()

def main():
    network = RingNetwork(100)
    num_steps = 500
    input = np.concatenate([
        np.linspace(0, 2*pi, num_steps//4),
        np.linspace(2*pi, 0, num_steps//4),
        np.linspace(0, 2*pi, num_steps//4),
        np.linspace(2*pi, 0, num_steps//4)
        ])
    alphas = np.ones(input.size)*0.6
    m, f = network.simulate(input, alphas)
    pdb.set_trace()

print("STARTING OVERLAPPING GRID SEARCH")
gridsearch(True, "overlap")
print()
print("STARTING NON-OVERLAPPING GRID SEARCH")
gridsearch(True, "nonoverlap")
