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
from LearningNetwork import LearningNetwork
from Simulator import Simulator
from Input import NoisyInput, IndependentInput, BehavioralInput, NavigationInput

pm = PlotMaker()

def run_and_plot_overlapnet(overlap=0.):
    N_pl = 100
    N_ep = 200
    K_inhib = 0.
    network = OverlapNetwork(
        N_pl=N_pl, N_ep=N_ep, K_inhib=K_inhib, overlap=overlap, add_feedback=True,
        num_internetwork_connections=3, num_ep_modules=12
        )
    inputgen = BehavioralInput(pre_seed_loc=pi)
    sim = Simulator(network, inputgen)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)
    import pdb; pdb.set_trace()

def run_and_plot_learningnet(overlap=0.):
    N_pl = 100
    N_ep = 100
    K_inhib = 0.
    network = LearningNetwork(
        N_pl=N_pl, N_ep=N_ep, K_inhib=K_inhib, overlap=overlap,
        num_ep_modules=12, start_random=True, add_feedback=True
        )
    inputgen = NavigationInput()
    sim = Simulator(network, inputgen)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)

for o in [0.]:
    print("Overlap: %1.2f"%o)
    run_and_plot_learningnet(o)
