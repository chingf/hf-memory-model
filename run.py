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
    K_ep = 0.8
    K_pl = 0.2
    network = OverlapNetwork(
        N_pl=N_pl, N_ep=N_ep, K_pl=K_pl, K_ep=K_ep, overlap=overlap,
        add_feedback=True, num_internetwork_connections=3, num_ep_modules=7
        )
    inputgen = BehavioralInput(pre_seed_loc=7, K_pl=K_pl, K_ep=K_ep)
    sim = Simulator(network, inputgen)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)
    with open("overlapnet.p", "wb") as p:
        pickle.dump({"sim": sim, "m": m, "f": f}, p)

def run_and_plot_learningring(overlap=0., network=None):
    """ Runs and plots a random network learning the ring structure. """

    N = 100
    K_inhib = 0.2
    if network is None:
        network = IsolatedNetwork(N, K_inhib, "random")
    else:
        with open(network, "rb") as p:
            dic = pickle.load(p)
        network = dic["sim"].network
    inputgen = NavigationInput(T=2000) # 13000
    sim = Simulator(network, inputgen)
    pm.plot_J(sim)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)
    with open("learnedring.p", "wb") as p:
        pickle.dump({"sim": sim}, p)

def run_and_plot_learningassociations(overlap=0.):
    """
    Runs and plots a place and episode network learning inter-network
    connections.
    """

    N_pl = 100
    N_ep = 100
    K_pl = K_ep = 0.3
    network = LearningNetwork(
        N_pl=N_pl, N_ep=N_ep, K_pl=K_pl, K_ep=K_ep, overlap=overlap,
        num_wta_modules=9, start_random=False, start_wta=False
        )
    inputgen = MultiCacheInput(K_ep=K_ep)
    sim = Simulator(network, inputgen)
    pm.plot_J(sim)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)

def run_and_plot_endtoend(overlap=0.):
    """ Runs and plots the end-to-end learning process """

    #np.random.seed(0)
    N_pl = 100
    N_ep = 100
    K_pl = 0.3
    K_ep = 0.3
    network = LearningNetwork(
        N_pl=N_pl, N_ep=N_ep, K_pl=K_pl, K_ep=K_ep, overlap=overlap,
        num_wta_modules=10, start_random=False, start_wta=True
        )
    inputgen = NavigationInput(T=8000)
    sim = Simulator(network, inputgen)
    pm.plot_J(sim)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)
    inputgen = MultiCacheInput(K_ep=K_ep)
    sim = Simulator(network, inputgen)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)
    with open("learnednet.p", "wb") as p:
        pickle.dump({"sim": sim, "m": m, "f": f}, p)
    inputgen = BehavioralInput(pre_seed_loc=7, K_pl=K_pl, K_ep=K_ep)
    sim = Simulator(network, inputgen)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)
    import pdb; pdb.set_trace()

def test_net(p):
    with open(p, "rb") as p:
        dic = pickle.load(p)
    network = dic["sim"].network
    K_pl = network.K_pl
    K_ep = network.K_ep
    inputgen = BehavioralInput(pre_seed_loc=13, K_pl=K_pl, K_ep=K_ep)
    sim = Simulator(network, inputgen)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)

def main():
    test_net("overlapnet.p")
    return
    for o in [0.4]:
        print("Overlap: %1.2f"%o)
        run_and_plot_overlapnet(o)

main()
