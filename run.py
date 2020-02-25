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
from Simulator import Simulator
from Input import NoisyInput, BehavioralInput 

pm = PlotMaker()

def run_and_plot_network(overlap=0.):
    N = 100 
    K_inhib = 0.
    network = OverlapNetwork(
        N=N, K_inhib=K_inhib, overlap=overlap, add_feedback=True,
        num_interactions=2
        )
    inputgen = BehavioralInput()
    sim = Simulator(network, inputgen)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)

for o in [0.4]:
    run_and_plot_network(o)
