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
from LearningNetwork import LearningNetwork, HalfLearningNetwork
from Simulator import Simulator
from Input import BehavioralInput, NavigationInput
from Input import CacheInput, MultiCacheInput, WTANavigationInput

pm = PlotMaker()

def run_and_plot_overlapnet(overlap=0.):
    """ Runs and plots the hand-tuned network. """

    N_pl = 100
    N_ep = 100 
    K_inhib = 0.18
    network = OverlapNetwork(
        N_pl=N_pl, N_ep=N_ep, K_inhib=K_inhib, overlap=overlap, add_feedback=True,
        num_internetwork_connections=3, num_ep_modules=10
        )
    inputgen = BehavioralInput(pre_seed_loc=pi)
    sim = Simulator(network, inputgen)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)

def run_and_plot_learningnet(overlap=0.):
    """ Runs and plots a random network learning the ring structure. """

    N_pl = 100
    N_ep = 100
    K_inhib = 0.18
    np.random.seed(1)
    network = LearningNetwork(
        N_pl=N_pl, N_ep=N_ep, K_inhib=K_inhib, overlap=overlap,
        num_ep_modules=12, start_random=True
        )
    inputgen = NavigationInput(T=2000)
    sim = Simulator(network, inputgen)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)
    import pdb; pdb.set_trace()

def run_and_plot_learningwtanet(overlap=0.):
    """ Runs and plots a WTA network learning the ring structure. """

    N_pl = 100
    N_ep = 100
    K_inhib = 0.18
    np.random.seed(3)
    network = LearningNetwork(
        N_pl=N_pl, N_ep=N_ep, K_inhib=K_inhib, overlap=overlap,
        num_ep_modules=6, start_random=False
        )
    inputgen = WTANavigationInput(T=2000)
    sim = Simulator(network, inputgen)
    pm.plot_J(sim)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)
    import pdb; pdb.set_trace()

def run_and_plot_halfnet(overlap=0.):
    """
    Runs and plots a place and episode network learning inter-network
    connections.
    """

    np.random.seed(0)
    N_pl = 100
    N_ep = 100
    K_inhib = 0.18
    network = HalfLearningNetwork(
        N_pl=N_pl, N_ep=N_ep, K_inhib=K_inhib, overlap=overlap,
        num_ep_modules=6
        )
    inputgen = MultiCacheInput()
    sim = Simulator(network, inputgen)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)
    import pdb; pdb.set_trace()

def main():
    for o in [0.]:
        print("Overlap: %1.2f"%o)
        run_and_plot_learningwtanet(o)

if __name__ == "__main__":
    main()
