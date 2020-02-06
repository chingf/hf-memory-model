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
from Network import RemapNetwork 

class PlotMaker(object):
    """Makes plots to visualize a ring network."""

    def plot_main(
            self, input_ext, alphas, f, input_c=None, target_indices=None, m=None
            ):
        """
        Plots the basic visualization needed: animal location, input/context
        strength, network activity.

        Args:
            input_ext (numpy array): Size (T,) array of float radians representing
                the external stimulus. Here, the stimulus is some external cue
                that indicates what the correct orientation theta_0 is.
            alphas (numpy array): Size (T,) array of float representing the
                strength of the external input.
            f (numpy array): Size (N,T) array of floats representing firing
                rate of each unit at each time step.
            input_c (numpy array): Size (T, N_c) array of floats representing
                the activation of context units over time.
            target_indices (numpy array): size (N_c,) array containing the
                indices of the target units.
        """

        width = 15
        height = 7 if m is None else 10
        fig = plt.figure(1)
        self._make_main_grid(input_ext, alphas, f, input_c, target_indices, m)
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

    def _make_main_grid(
            self, input_ext, alphas, f,
            input_c=None, target_indices=None, m=None,
            make_ring=False, network=None
            ):
        """
        Fills the current PyPlot object with the basic visualization needed:
        animal location, input/context strength, network activity.

        Args:
            input_ext (numpy array): Size (T,) array of float radians representing
                that indicates what the correct orientation theta_0 is.
            alphas (numpy array): Size (T,) array of float representing the
                strength of the external input.
            f (numpy array): Size (N,T) array of floats representing firing
                rate of each unit at each time step.
            input_c (numpy array): Size (T, N_c) array of floats representing
                the activation of context units over time.
            target_indices (numpy array): size (N_c,) array containing the
                normalized indices of the target units. e.g., index 0.5
                corresponds to pi
        """

        alphas = np.tile(alphas, (50, 1))
        T = input_ext.size
        gridrows = 9 if m is None else 13
        gridcols = 12 
        gridspec.GridSpec(gridrows, gridcols)
        colspan = gridcols - 2 if make_ring else gridcols

        # Plot location 
        plt.subplot2grid((gridrows, gridcols), (0,0), rowspan=4, colspan=colspan)
        plt.plot(np.arange(T), input_ext, linewidth=2)
        if (input_c is not None) and (target_indices is not None):
            if input_c.size == T:
                input_c = input_c.reshape((-1, 1))
            for c_idx, activity_c in enumerate(input_c.T): # Iterate over context
                on_time = np.where(activity_c > 0)
                target_index = target_indices[c_idx]*2*pi
                plt.axhline(
                    target_index, np.min(on_time)/T, np.max(on_time)/T,
                    linewidth=4, color="red"
                    )
        plt.xticks(np.arange(0, T, 100), np.arange(0, T, 100))
        plt.yticks(
            [0, pi, 2*pi], ["0", "Pi", "2Pi"]
            )
        plt.ylabel("Location", fontsize=14)
        plt.title("Activity over Time", fontsize=16)
   
        # Plot input strength
        plt.subplot2grid((gridrows, gridcols), (4,0), colspan=colspan)
        aximg1 = plt.imshow(alphas, aspect='auto')
        plt.yticks([])
        plt.xticks([])
        plt.ylabel("Input\nStrength\n\n", fontsize=14)
       
        # Plot network firing rate activity
        plt.subplot2grid((gridrows, gridcols), (5,0), rowspan=4, colspan=colspan)
        norm = mcolors.DivergingNorm(vmin=f.min(), vmax = f.max(), vcenter=0)
        aximg2 = plt.imshow(
            np.flip(f, axis=0), cmap=plt.cm.coolwarm, norm=norm, aspect='auto'
            )
        plt.yticks(
            [0, f.shape[0]//2, f.shape[0] - 1], ["2Pi", "Pi", "0"]
            )
        plt.ylabel("f_\u03B8", fontsize=14)

        # Plot network current activity if it was provided
        if m is not None:
            plt.subplot2grid((gridrows, gridcols), (9,0), rowspan=4, colspan=colspan)
            norm = mcolors.DivergingNorm(vmin=f.min(), vmax = f.max(), vcenter=0)
            plt.imshow(
                np.flip(m, axis=0), cmap=plt.cm.coolwarm, norm=norm, aspect='auto'
                )
            plt.yticks(
                [0, m.shape[0]//2, m.shape[0] - 1], ["2Pi", "Pi", "0"]
                )
            plt.ylabel("m_\u03B8", fontsize=14)

        # Plots the seconds on the x axis of the last subplot
        plt.xticks(
            np.arange(0, T, 100), np.arange(0, T//10, 10)
            )
        plt.xlabel("Seconds")

        # Plot seed units if desired
        ringaxs = []
        if make_ring is True:
            plt.subplot2grid(
                (gridrows, gridcols), (0, colspan), rowspan=4, colspan=2
                )
            ringaxs.append(plt.gca())
            plt.subplot2grid(
                (gridrows, gridcols), (5, colspan), rowspan=4, colspan=2
                )
            ringaxs.append(plt.gca())
            self._make_ring(network, ringaxs)

        return aximg1, aximg2, ringaxs 

    def _make_ring(self, network, ringaxs):
        for context in range(2):
            target_index = network.target_indices[context]
            seed_units = network.ring_indices[context,:]
            x = []
            y = []
            for seed_unit in seed_units:
                seed_angle = (seed_unit/network.N)*(2*pi)
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
