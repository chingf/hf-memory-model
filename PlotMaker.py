import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, CheckButtons
from math import pi
from math import cos
from math import sin
from Input import NoisyInput, BehavioralInput
from Network import OverlapNetwork 

class PlotMaker(object):
    """Makes plots to visualize a ring network."""

    def plot_main(self, sim, f):
        """
        Plots the basic visualization needed: input, episode network activity,
        place network activity

        Args:
            sim: Simulator object
            f (numpy array): Size (N,T) array of floats representing firing
                rate of each unit at each time step.
        """

        width = 12
        height = 7
        fig = plt.figure(1)
        self._make_main_grid(sim, f)
        fig.tight_layout()
        fig.set_size_inches(w=width, h=height)
        plt.show()

    def plot_J(self, sim):
        """
        Plots the J of the network with indices arranged by order on ring
        """

        network = sim.network
        full_J = np.zeros((network.N*2, network.N*2))*np.nan
        for idx_i, i in enumerate(network.J_episode_indices):
            for idx_j, j in enumerate(network.J_episode_indices):
                full_J[idx_i, idx_j] = network.J[i, j]
            for idx_j, j in enumerate(network.J_place_indices):
                full_J[idx_i, idx_j + network.N] = network.J[i, j]
        for idx_i, i in enumerate(network.J_place_indices):
            for idx_j, j in enumerate(network.J_episode_indices):
                full_J[idx_i + network.N, idx_j] = network.J[i, j]
            for idx_j, j in enumerate(network.J_place_indices):
                full_J[idx_i + network.N, idx_j + network.N] = network.J[i, j]
        plt.figure()
        plt.imshow(full_J)
        plt.show()

    def _make_main_grid(self, sim, f):
        """
        Fills the current PyPlot object with the basic visualization needed:
        animal location, input/context strength, network activity.

        Args:
            sim: A Simulator object
            f (numpy array): Size (N,T) array of floats representing firing
                rate of each unit at each time step.
        """

        network = sim.network
        inputs = sim.inputgen.inputs
        alphas = sim.inputgen.alphas
        T = inputs.shape[0]
        N = network.N
        gridrows = 9
        gridcols = 12
        rowspan = 3
        colspan = 10 # Column span for main time-dependent plots
        gridspec.GridSpec(gridrows, gridcols)

        # Segregate the two networks
        f_ep = f[network.J_episode_indices, :]
        f_pl = f[network.J_place_indices, :]

        # Plot input
        for t, alpha in enumerate(alphas):
            inputs[t,:] *= alpha
        inputs = np.concatenate((
            inputs[:, network.J_place_indices],
            inputs[:, network.J_episode_indices]
            ), axis=1)
        plt.subplot2grid(
            (gridrows, gridcols), (0,0), rowspan=rowspan, colspan=colspan
            )
        plt.imshow(np.flip(inputs.T, axis=0))
        plt.xticks(np.arange(0, T, 100), np.arange(0, T, 100))
        plt.yticks(
            [0, N//2, 3*N//2, 2*N], ["0", "Pi", "Pi", "2Pi"]
            )
        plt.axhline(N, color='black')
        plt.ylabel("Input to Unit", fontsize=14)
        plt.title("Activity over Time", fontsize=16)
        plt.gca().set_aspect('auto')
      
        # Plot residuals and episode network activity
        plt.subplot2grid(
            (gridrows,gridcols), (rowspan,colspan), rowspan=rowspan, colspan=2
            )
        plt.plot(f_ep[:,-1], np.arange(N))
        for interacting_unit in network.interacting_units[0]:
            plt.axhline(interacting_unit, color="red", linewidth=0.5)
        for attractor_unit in network.episode_attractors:
            plt.axhline(attractor_unit, color="green", linewidth=0.5)
        plt.axvline(0, color="gray")
        plt.yticks([])
        plt.subplot2grid(
            (gridrows, gridcols), (rowspan,0), rowspan=rowspan, colspan=colspan
            )
        norm = mcolors.DivergingNorm(vmin=f.min(), vmax = f.max(), vcenter=0)
        aximg_ep = plt.imshow(
            np.flip(f_ep, axis=0), cmap=plt.cm.coolwarm, norm=norm, aspect='auto'
            )
        plt.xticks(np.arange(0, T, 100), np.arange(0, T//10, 10))
        plt.yticks(
            [0, f_ep.shape[0]//2, f_ep.shape[0] - 1], ["2Pi", "Pi", "0"]
            )
        plt.ylabel("Episode Network", fontsize=14)

        # Plot residuals and place network activity
        plt.subplot2grid(
            (gridrows,gridcols), (rowspan*2,colspan), rowspan=rowspan, colspan=2
            )
        plt.plot(f_pl[:,-1], np.arange(N))
        for interacting_unit in network.interacting_units[1]:
            plt.axhline(interacting_unit, color="red", linewidth=0.5)
        plt.axvline(0, color="gray")
        plt.yticks([])
        plt.subplot2grid(
            (gridrows, gridcols), (rowspan*2,0), rowspan=rowspan, colspan=colspan
            )
        norm = mcolors.DivergingNorm(vmin=f.min(), vmax = f.max(), vcenter=0)
        aximg_pl = plt.imshow(
            np.flip(f_pl, axis=0), cmap=plt.cm.coolwarm, norm=norm, aspect='auto'
            )
        plt.yticks(
            [0, f_ep.shape[0]//2, f_ep.shape[0] - 1], ["2Pi", "Pi", "0"]
            )
        plt.ylabel("Place Network", fontsize=14)

        # Plots the seconds on the x axis of the last subplot
        plt.xticks(np.arange(0, T, 100), np.arange(0, T//10, 10))
        plt.xlabel("Seconds")

        return aximg_ep, aximg_pl

