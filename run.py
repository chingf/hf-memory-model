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
from Network import OverlapNetwork 
from GridSearch import GridSearch
from InputGenerator import InputGenerator

pm = PlotMaker()

def run_and_plot_network(overlap=0.):
    np.random.seed(1)
    N = 100 
    K_inhib = 0
    network = OverlapNetwork(
        N=N, K_inhib=K_inhib, overlap=overlap, add_feedback=True
        )
    inputgen = InputGenerator()
    input_ext, alphas = inputgen.get_noise_input(network)
    m, f = network.simulate(input_ext, alphas)
    pm.plot_main(input_ext, alphas, f, network)
    pm.plot_J(network)

for o in [0.4]:
    run_and_plot_network(o)
