import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, CheckButtons
from math import pi
from math import cos
from math import sin
from InputGenerator import InputGenerator
from Network import OverlapNetwork 

class PlotMaker(object):
    """Makes plots to visualize a ring network."""

    def plot_main(
            self, input_ext, alphas, f, network
            ):
        """
        Plots the basic visualization needed: input, episode network activity,
        place network activity

        Args:
            input_ext (numpy array): Size (T, num_units) array of floats
                representing the external stimulus to the episode network. 
            alphas (numpy array): Size (T,) array of float representing the
                strength of the external input.
            f (numpy array): Size (N,T) array of floats representing firing
                rate of each unit at each time step.
            network (OverlapNetwork): the network that generated the activity
        """

        width = 15
        height = 7
        fig = plt.figure(1)
        self._make_main_grid(input_ext, alphas, f, network)
        fig.tight_layout()
        fig.set_size_inches(w=width, h=height)
        plt.show()

    def plot_slider(
            self, K_max=0.3, Ncr_max=80, Jcr_max=0.1,
            K_init=0., Ncr_init=20, Jcr_init=0.015
            ):
        """
        Makes the main 3-chunk plot with sliders to change MixedNetwork
        activity based on parameters

        Args:
            Arguments will initialize the following parameters and specify the
            maximum value these parameters can attain for visualization purposes
            K (float): parameter for the global inhibition 
            N_cr (int): Number of ring units that a single context unit synapses
                onto.
            J_cr (float): parameter; the synaptic strength from ring unit to
                target unit during context mode
        """

        fig, ax = plt.subplots(figsize=(13, 8))
        plt.subplots_adjust(bottom=0.4, right=0.95, left = 0.1)

        # Define fixed parameters
        N = 100
        N_c = 2
        C = 1
        target_indices = np.array([N//2, 0])
        T = 1250
        input_ext, input_c, alphas = InputGenerator().get_input(T, N_c)

        # Initialize graph
        network = MixedNetwork(
            N, N_c, C, K_init, Ncr_init, Jcr_init, target_indices
            )
        self.network = network
        m, f, dmdt = network.simulate(input_ext, input_c, alphas)
        aximg1, aximg2, ringaxs = self._make_main_grid(
            input_ext, alphas, f, input_c, target_indices/N,
            network=network, make_ring=True
            )
        self.view_avg = False

        # Define parameter sliders 
        axcolor = 'lightgoldenrodyellow'
        axK = plt.axes([0.1, 0.3, 0.705, 0.03], facecolor=axcolor)
        axNcr = plt.axes([0.1, 0.25, 0.705, 0.03], facecolor=axcolor)
        axJcr = plt.axes([0.1, 0.2, 0.705, 0.03], facecolor=axcolor)
        axExtIn = plt.axes([0.1, 0.15, 0.705, 0.03], facecolor=axcolor)
        axAlpha = plt.axes([0.1, 0.1, 0.705, 0.03], facecolor=axcolor)
        sK = Slider(axK, "K_inhib", 0, K_max, valinit=K_init, valstep=0.01)
        sNcr = Slider(axNcr, "N_cr", 2, Ncr_max, valinit=Ncr_init, valstep=2)
        sJcr = Slider(axJcr, "J_cr", 0, Jcr_max, valinit=Jcr_init, valstep=0.0005)
        sExtIn = Slider(
            axExtIn, "Location",
            0, 2*pi, valinit=0, valstep=0.01)
        sAlpha = Slider(
            axAlpha, "External Input\nStrength",
            0, T, valinit=np.where(alphas == 0)[0][0], valstep=1
            )
        def update(val):
            K = sK.val
            Ncr = int(sNcr.val)
            Jcr = sJcr.val
            input_ext = self._get_input_ext(sExtIn.val, int(sAlpha.val), T)
            alphas = self._get_alphas(int(sAlpha.val), T)
            num_reps = 10 if self.view_avg else 1
            f = np.zeros((N, T)) 
            for _ in range(num_reps):
                network = MixedNetwork(
                    N, N_c, C, K, Ncr, Jcr, target_indices
                    )
                _, _f, _ = network.simulate(input_ext, input_c, alphas)
                f += _f
            f /= num_reps
            aximg1.set_data(np.tile(alphas, (50,1)))
            aximg2.set_data(np.flip(f, axis=0))
            self._make_ring(network, ringaxs)
            self.network = network
            fig.canvas.draw_idle()

        # Define reset button
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        resetbutton = Button(
            resetax, 'Reset', color=axcolor, hovercolor='0.975'
            )
        def reset(event):
            sK.reset()
            sNcr.reset()
            sJcr.reset()

        # Define random refresh button
        refreshax = plt.axes([0.6, 0.025, 0.15, 0.04])
        refreshbutton = Button(
            refreshax, 'Random Refresh', color=axcolor, hovercolor='0.975'
            )

        # Define check box for viewing averages
        avgax = plt.axes([0.4, 0.025, 0.15, 0.04])
        avgbutton = CheckButtons(avgax, ["View Average"])
        def toggle_avg(event):
            self.view_avg = False if self.view_avg else True

        # Define save button
        saveax = plt.axes([0.2, 0.025, 0.15, 0.04])
        savebutton = Button(
            saveax, 'Save Network', color=axcolor, hovercolor='0.975'
            )
        def save(event):
            with open('network.p', 'wb') as f:
                pickle.dump(self.network, f)

        # Link each widget to its action 
        sK.on_changed(update)
        sNcr.on_changed(update)
        sJcr.on_changed(update)
        sExtIn.on_changed(update)
        sAlpha.on_changed(update)
        resetbutton.on_clicked(reset)
        refreshbutton.on_clicked(update)
        avgbutton.on_clicked(toggle_avg)
        savebutton.on_clicked(save)

        plt.show()

    def _make_main_grid(self, input_ext, alphas, f, network):
        """
        Fills the current PyPlot object with the basic visualization needed:
        animal location, input/context strength, network activity.

        Args:
            input_ext (numpy array): Size (T, num_units) array of floats
                representing the external stimulus to the episode network. 
            alphas (numpy array): Size (T,) array of float representing the
                strength of the external input.
            f (numpy array): Size (N,T) array of floats representing firing
                rate of each unit at each time step.
            network (OverlapNetwork): the network that generated the activity
        """

        T = input_ext.shape[0]
        N = network.N
        for t, alpha in enumerate(alphas):
            input_ext[t,:] *= alpha
        gridrows = 9
        gridcols = 12
        rowspan = 3
        colspan = 10 # Column span for main time-dependent plots
        gridspec.GridSpec(gridrows, gridcols)

        # Segregate the two networks
        f_ep = f[network.J_episode_indices, :]
        f_pl = f[network.J_place_indices, :]

        # Plot input
        plt.subplot2grid(
            (gridrows, gridcols), (0,0), rowspan=rowspan, colspan=colspan
            )
        plt.imshow(input_ext.T[:network.num_separate_units,:])
        plt.xticks(np.arange(0, T, 100), np.arange(0, T, 100))
        plt.yticks(
            [0, pi, 2*pi], ["0", "Pi", "2Pi"]
            )
        plt.ylabel("Input to Unit", fontsize=14)
        plt.title("Activity over Time", fontsize=16)
      
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
        plt.xticks([])
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
        plt.xticks([])
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

    def _make_ring(self, network, ringaxs):
        for context in range(2):
            target_index = network.target_indices[context]
            seed_units = network.ring_indices[context,:]
            x = []
            y = []
            for seed_unit in seed_units:
                seed_angle = (seed_unit/N)*(2*pi)
                x.append(cos(seed_angle))
                y.append(sin(seed_angle))
            target_angle = (target_index/network.N)*(2*pi)
            ringaxs[context].clear()
            ringaxs[context].scatter(x, y, color="blue")
            ringaxs[context].scatter(
                cos(target_angle), sin(target_angle), color="red"
                )
            ringaxs[context].set_xticks([])
            ringaxs[context].set_yticks([])
            ringaxs[context].set_xlim([-1.1, 1.1])
            ringaxs[context].set_ylim([-1.1, 1.1])
        ringaxs[0].set_title("Target 1 (\u03C0)")
        ringaxs[1].set_title("Target 2 (0)")

    def _get_input_ext(self, location, location_timestep, T):
        pause_length = T - T//5 - location_timestep 
        input_ext = np.concatenate([
            np.linspace(0, location, location_timestep),
            np.linspace(location, location, pause_length),
            np.linspace(location, 2*pi, T//5),
            ])
        return input_ext

    def _get_alphas(self, timestep, T):
        alphas = np.ones(T)*0.6
        alphas[timestep:int(0.8*T),] = 0
        return alphas
