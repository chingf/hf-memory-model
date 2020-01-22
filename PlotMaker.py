import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, CheckButtons
from math import pi
from Network import PlasticMixedNetwork

class PlotMaker(object):
    """Makes plots to visualize a ring network."""

    def plot_main(self, input, alphas, f, input_c=None, target_indices=None):
        """
        Plots the basic visualization needed: animal location, input/context
        strength, network activity.

        Args:
            input (numpy array): Size (T,) array of float radians representing
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

        fig = plt.figure(1)
        self._make_main_grid(input, alphas, f, input_c, target_indices)
        fig.tight_layout()
        fig.set_size_inches(w=15, h=7)
        plt.show()

    def plot_slider(
            self, C_max=15, Ncr_max=50, Jcr_max=1.5,
            C_init=1, Ncr_init=40, Jcr_init=0.06
            ):
        """
        Makes the main 3-chunk plot with sliders to change PlasticMixedNetwork
        activity based on parameters

        Args:
            Arguments will initialize the following parameters and specify the
            maximum value these parameters can attain for visualization purposes
            C (float): parameter for the constant gain added to the dmdt for a
                ring unit when its connected context unit is activated.
            N_cr (int): Number of ring units that a single context unit synapses
                onto.
            J_cr (float): parameter; the synaptic strength from ring unit to
                target unit during context mode
        """
        fig, ax = plt.subplots(figsize=(13, 8))
        plt.subplots_adjust(bottom=0.4)

        # Define fixed parameters
        N = 100
        N_c = 2
        target_indices = np.array([N//2, 0])
        T = 1000
        input, input_c, alphas = self._get_input(T, N_c)

        # Initialize graph
        network = PlasticMixedNetwork(
            N, N_c, C_init, Ncr_init, Jcr_init, target_indices
            )
        m, f, dmdt = network.simulate(input, input_c, alphas)
        aximg = self._make_main_grid(input, alphas, f, input_c, target_indices)
        self.view_avg = False

        # Define parameter sliders 
        axcolor = 'lightgoldenrodyellow'
        axC = plt.axes([0.15, 0.3, 0.65, 0.03], facecolor=axcolor)
        axNcr = plt.axes([0.15, 0.2, 0.65, 0.03], facecolor=axcolor)
        axJcr = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
        sC = Slider(axC, "C", 0, C_max, valinit=C_init, valstep=0.1)
        sNcr = Slider(axNcr, "N_cr", 2, Ncr_max, valinit=Ncr_init, valstep=2)
        sJcr = Slider(axJcr, "J_cr", 0, Jcr_max, valinit=Jcr_init, valstep=0.02)
        def update(val):
            C = sC.val
            Ncr = int(sNcr.val)
            Jcr = sJcr.val
            num_reps = 10 if self.view_avg else 1
            f = np.zeros((N, T)) 
            for _ in range(num_reps):
                network = PlasticMixedNetwork(
                    N, N_c, C, Ncr, Jcr, target_indices
                    )
                _, _f, _ = network.simulate(input, input_c, alphas)
                f += _f
            f /= num_reps
            aximg.set_data(np.flip(f, axis=0))
            fig.canvas.draw_idle()

        # Define reset button
        resetax = plt.axes([0.8, 0.025, 0.1, 0.06])
        resetbutton = Button(
            resetax, 'Reset', color=axcolor, hovercolor='0.975'
            )
        def reset(event):
            sC.reset()
            sNcr.reset()
            sJcr.reset()

        # Define random refresh button
        refreshax = plt.axes([0.6, 0.025, 0.15, 0.06])
        refreshbutton = Button(
            refreshax, 'Random Refresh', color=axcolor, hovercolor='0.975'
            )

        # Define check box for viewing averages
        avgax = plt.axes([0.4, 0.025, 0.15, 0.06])
        avgbutton = CheckButtons(avgax, ["View Average"])
        def toggle_avg(event):
            self.view_avg = False if self.view_avg else True

        # Link each widget to its action 
        sC.on_changed(update)
        sNcr.on_changed(update)
        sJcr.on_changed(update)
        resetbutton.on_clicked(reset)
        refreshbutton.on_clicked(update)
        avgbutton.on_clicked(toggle_avg)

        plt.show()

    def _get_input(self, T, N_c):
        """
        Helper method that returns the general toy input used.

        Args:
            T (int): Number of time steps
            N_c (int): Number of context units
        """

        input = np.concatenate([
            np.linspace(0, 2*pi, T//5),
            np.linspace(2*pi, 0, T//5),
            np.linspace(0, 0, T//5),
            np.linspace(0, 0, T//5),
            np.linspace(0, 2*pi, T//5)
            ])
        alphas = np.ones(input.size)*0.6
        input_c = np.zeros((input.size, N_c))
        alphas[500:1000,] = 0
        input_c = np.zeros((input.size, N_c))
        input_c[650:800, 0] = 1
        input_c[850:, 1] = 1
        return input, input_c, alphas

    def _make_main_grid(
            self, input, alphas, f, input_c=None, target_indices=None
            ):
        """
        Fills the current PyPlot object with the basic visualization needed:
        animal location, input/context strength, network activity.

        Args:
            input (numpy array): Size (T,) array of float radians representing
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

        alphas = np.tile(alphas, (50,1))
        T = input.size
        gridspec.GridSpec(9,1)

        # Plot location 
        plt.subplot2grid((9,1), (0,0), rowspan=4)
        plt.plot(np.arange(T), input, linewidth=2)
        if (input_c is not None) and (target_indices is not None):
            for c_idx, activity_c in enumerate(input_c.T): # Iterate over context
                on_time = np.where(activity_c > 0)
                target_index = input[target_indices[c_idx]]
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
        plt.subplot2grid((9,1), (4,0))
        plt.imshow(alphas, aspect='auto')
        plt.yticks([])
        plt.xticks([])
        plt.ylabel("Input\nStrength\n\n", fontsize=14)
       
        # Plot network activity
        plt.subplot2grid((9,1), (5,0), rowspan=4)
        norm = mcolors.DivergingNorm(vmin=f.min(), vmax = f.max(), vcenter=0)
        aximg = plt.imshow(
            np.flip(f, axis=0), cmap=plt.cm.coolwarm, norm=norm, aspect='auto'
            )
        plt.xticks(
            np.arange(0, T, 100), np.arange(0, T//10, 10)
            )
        plt.xlabel("Seconds")
        plt.yticks(
            [0, f.shape[0]//2, f.shape[0] - 1], ["0", "Pi", "2Pi"]
            )
        plt.ylabel("m_\u03B8", fontsize=14)
        return aximg
