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
from IsolatedNetwork import IsolatedNetwork
from Simulator import Simulator
from Input import BehavioralInput, NavigationInput
from Input import MultiCacheInput

pm = PlotMaker()

def run_and_plot_overlapnet(overlap=0.):
    """ Runs and plots the hand-tuned network. """

    N_pl = 100
    N_ep = 100 
    K_pl=K_ep=0.2
    network = OverlapNetwork(
        N_pl=N_pl, N_ep=N_ep, K_pl=K_pl, K_ep=K_ep, overlap=overlap,
        add_feedback=True, num_internetwork_connections=3, num_ep_modules=7
        )
    inputgen = BehavioralInput(pre_seed_loc=pi/2, K_pl=K_pl, K_ep=K_ep)
    sim = Simulator(network, inputgen)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)
    with open("overlapnet.p", "wb") as p:
        pickle.dump({"sim": sim, "m": m, "f": f}, p)

def run_and_plot_learningring(overlap=0.):
    """ Runs and plots a random network learning the ring structure. """

    N = 100
    K_inhib = 0.2
    network = IsolatedNetwork(N, K_inhib, "random")
    inputgen = NavigationInput(T=10000)
    sim = Simulator(network, inputgen)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)

def run_and_plot_halfnet(overlap=0.):
    """
    Runs and plots a place and episode network learning inter-network
    connections.
    """

    np.random.seed(0)
    N_pl = 100
    N_ep = 100
    K_inhib = 0.18
    network = LearningNetwork(
        N_pl=N_pl, N_ep=N_ep, K_inhib=K_inhib, overlap=overlap,
        num_ep_modules=6, start_random=False
        )
    inputgen = MultiCacheInput()
    sim = Simulator(network, inputgen)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)
    import pdb; pdb.set_trace()

def run_and_plot_endtoend(overlap=0.):
    """ Runs and plots the end-to-end learning process """

    #np.random.seed(0)
    N_pl = 100
    N_ep = 100
    K_inhib = 0.2
    network = LearningNetwork(
        N_pl=N_pl, N_ep=N_ep, K_inhib=K_inhib, overlap=overlap,
        num_wta_modules=6, start_random=False
        )
    inputgen = NavigationInput(T=30000)
    sim = Simulator(network, inputgen)
    pm.plot_J(sim)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)
    inputgen = MultiCacheInput()
    sim = Simulator(network, inputgen)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)
    inputgen = BehavioralInput(pre_seed_loc=pi/2)
    sim = Simulator(network, inputgen)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)
    import pdb; pdb.set_trace()

def main():
    for o in [0.6]:
        print("Overlap: %1.2f"%o)
        run_and_plot_overlapnet(o)

if __name__ == "__main__":
    main()
