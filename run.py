import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from math import pi
from math import sin
from math import cos
from PlotMaker import PlotMaker
from Network import RingNetwork, SimpleContextNetwork, MixedNetwork
from GridSearch import GridSearch
from InputGenerator import InputGenerator

pm = PlotMaker()

def make_ring(network):
    for context in range(2):
        plt.figure(figsize=(3,3))
        target_index = network.target_indices[context]
        seed_units = network.ring_indices[context,:]
        x = []
        y = []
        for seed_unit in seed_units:
            seed_angle = (seed_unit/network.N)*(2*pi)
            x.append(cos(seed_angle))
            y.append(sin(seed_angle))
        target_angle = (target_index/network.N)*(2*pi)
        plt.scatter(x, y, color="blue")
        plt.scatter(
            cos(target_angle), sin(target_angle), color="red"
            )
        plt.xticks([])
        plt.yticks([])
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.show()

def run_network():
    N = 102
    N_c = 2
    C = 1
    N_cr = 24
    J_cr = 0.6
    K_inhib = 0.
    target_indices = np.array([50, 0])
    network = MixedNetwork(N, N_c, C, K_inhib, N_cr, J_cr, target_indices)
    inputgen = InputGenerator()
    input_ext, input_c, alphas = inputgen.get_input2(1250, N_c)
    m, f, dmdt = network.simulate(input_ext, input_c, alphas)
    make_ring(network)
    pm.plot_main(
        input_ext, alphas, f, input_c, target_indices/N, m
        )

pm.plot_slider()
