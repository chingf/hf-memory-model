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
from Network import OverlapNetwork, VMNetwork
from GridSearch import GridSearch
from InputGenerator import InputGenerator

pm = PlotMaker()

mu = 0
x = np.linspace(-pi, pi, 100)
N = 100 
K_inhib = 0
def vonmises(x, kappa, mu):
    return np.exp(kappa*np.cos(x-mu))/(2*pi*np.i0(kappa))

kappas = np.arange(1, 20, 0.5)
heights = []
widths = []
gains = np.zeros(kappas.size)*np.nan
for idx, kappa in enumerate(kappas):
    vm = vonmises(x, kappa, mu)
    vm -= (np.max(vm)/2.)
    vm *= 2
    width = np.sum(vm > 0)
    widths.append(width)
    for gain in np.arange(1, 10, 0.1):
        np.random.seed(1)
        network = VMNetwork(
            N=N, K_inhib=K_inhib, overlap=0.4, add_feedback=True,
            gain=gain, kappa=kappa
            )
        inputgen = InputGenerator()
        input_ext, alphas = inputgen.get_sin_input(network)
        m, f = network.simulate(input_ext, alphas)
        if gain == 1:
            heights.append(np.max(m))
        if np.max(m) >= 1:
            gains[idx] = gain
            if kappa % 5 == 0:
                print(kappa)
                pm.plot_main(input_ext, alphas, f, network)
                pm.plot_J(network)
            break;

# Plot height vs width
plt.figure()
plt.scatter(widths, heights)
plt.xlabel("Width of Curve")
plt.ylabel("Bump Height Obtained")
plt.show()

# Plot width vs gain
plt.figure()
plt.scatter(widths, gains)
plt.ylim((1, 3.5))
plt.xlabel("Width of Curve")
plt.ylabel("Gain")
plt.title("Paramters to obtain bump height = 1")
plt.show()
import pdb; pdb.set_trace()

