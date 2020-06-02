import pickle
import itertools
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from math import pi, sin, cos

def plot_J(J, network, sortby=False, title=None):
    J = J.copy()
    if sortby is not False:
        memory = sortby.copy()
        sorting = np.argsort(memory[network.J_ep_indices])
        sorting = np.concatenate((sorting, network.J_pl_indices))
        sorting = np.ix_(sorting, sorting)
        J = J[sorting]
    gridspec.GridSpec(1, 10)
    plt.subplot2grid((1, 10), (0,0), rowspan=1, colspan=9)
    norm = mcolors.DivergingNorm(vmin=J.min(), vmax = J.max(), vcenter=0)
    plt.imshow(J, aspect="auto", cmap=plt.cm.coolwarm, norm=norm)
    if title is not None:
        plt.title(title)
    plt.show()

def plot_formation(f, network, inputs, sortby=False, title=None):
    if sortby is not False:
        memory = sortby.copy()
        sorting = np.argsort(memory[network.J_ep_indices])
        sorting = np.concatenate((sorting, network.J_pl_indices))
        f = f[sorting, :]
        inputs = inputs[sorting,:]
        memory = np.expand_dims(memory[sorting], axis=1)
    else:
        memory = np.array([0])

    norm = mcolors.DivergingNorm(
        vmin=min(inputs.min(), f.min(), memory.min()),
        vmax=max(inputs.max(), f.max(), memory.max()), vcenter=0.01
        )
    fig = plt.figure(figsize=(6,5))
    nrows = 2 
    ncols = 10
    gridspec.GridSpec(nrows,ncols)
    plt.subplot2grid((nrows,ncols), (0,0), rowspan=1, colspan=9)
    plt.imshow(inputs, aspect="auto", cmap=plt.cm.coolwarm, norm=norm)
    plt.subplot2grid((nrows,ncols), (1,0), rowspan=1, colspan=9)
    plt.imshow(f, aspect="auto", cmap=plt.cm.coolwarm, norm=norm)
    if sortby is not False:
        plt.subplot2grid((nrows,ncols), (0,9), rowspan=1, colspan=1)
        plt.imshow(memory, cmap=plt.cm.coolwarm, norm=norm, aspect="auto")
        plt.subplot2grid((nrows,ncols), (1,9), rowspan=1, colspan=1)
        plt.imshow(memory, cmap=plt.cm.coolwarm, norm=norm, aspect="auto")
    plt.suptitle(title)
    plt.show()

