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
from WTANetwork import WTANetwork
from Simulator import Simulator
from Input import NoisyInput, BehavioralInput 

pm = PlotMaker()

def run_and_plot_overlapnet(overlap=0.):
    N = 100 
    K_inhib = 0.
    network = OverlapNetwork(
        N=N, K_inhib=K_inhib, overlap=overlap, add_feedback=True,
        num_interactions=3
        )
    inputgen = NoisyInput()
    sim = Simulator(network, inputgen)
    m, f = sim.simulate()
    pm.plot_main(sim, f)
    pm.plot_J(sim)

def run_and_plot_wtanet():
    N = 100 
    network = WTANetwork(N=N, num_modules=5)
    m, f, inputs = network.simulate(200)
    norm = mcolors.DivergingNorm(vmin=f.min(), vmax = f.max(), vcenter=0)
    fig, axs = plt.subplots(2, 1, figsize=(12, 7))
    axs[0].imshow(f, cmap=plt.cm.coolwarm, norm=norm, aspect="auto")
    axs[1].imshow(inputs, cmap=plt.cm.coolwarm, norm=norm, aspect="auto")
    plt.show()
    plt.figure()
    plt.imshow(network.J)
    plt.show()

for o in [0.4]:
    run_and_plot_wtanet()
