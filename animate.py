import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat
import matplotlib.colors as mcolors
import os
from math import pi
from PlotMaker import PlotMaker
from Input import PresentationInput
from Simulator import Simulator
import matplotlib.cm as cm
import pandas as pd
import time
from matplotlib import rcParams
import matplotlib.animation as animation
from celluloid import Camera

def animate_learning():
    with open("pickles/learnednet5.p", "rb") as p:
        dic = pickle.load(p)
    sim1 = dic["sim1"]
    sim2 = dic["sim2"]
    sim3 = dic["sim3"]
    net = sim3.network
    m = dic["m"]
    f = dic["f"]
    JsRing = sim1.J_samples
    JsAssoc = sim2.J_samples
    Js = []
    Js.extend(JsRing)
    Js.extend(JsAssoc[:2600:100])
    Js.extend(JsAssoc[2585:2600])
    Js.extend(JsAssoc[2600::100])
    fig, ax = plt.subplots()
    camera = Camera(fig)
    N_ep = net.N_ep
    N_pl = net.N_pl
    for step in range(len(Js)-1):
        J = Js[step]
        full_J = np.zeros((N_ep + N_pl, N_ep + N_pl))*np.nan
        for idx_i, i in enumerate(net.J_episode_indices):
            for idx_j, j in enumerate(net.J_episode_indices):
                full_J[idx_i, idx_j] = J[i, j]
            for idx_j, j in enumerate(net.J_place_indices):
                full_J[idx_i, idx_j + N_ep] = J[i, j]
        for idx_i, i in enumerate(net.J_place_indices):
            for idx_j, j in enumerate(net.J_episode_indices):
                full_J[idx_i + N_ep, idx_j] = J[i, j]
            for idx_j, j in enumerate(net.J_place_indices):
                full_J[idx_i + N_ep, idx_j + N_ep] = J[i, j]
        norm = mcolors.DivergingNorm(
            vmin=full_J.min(), vmax = full_J.max(), vcenter=0
            )
        ax.imshow(full_J, animated=True, cmap=plt.cm.coolwarm, norm=norm)
        ax.set_xticks([])
        ax.set_yticks([])
        camera.snap()
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    anim = camera.animate(interval=500, repeat_delay=3000, blit=True)
    anim.save('Js.mp4', writer=writer)

def animate_behav():
    with open("pickles/learnednet5.p", "rb") as p:
        dic = pickle.load(p)
    sim3 = dic["sim3"]
    m = dic["m"]
    f = dic["f"]
    net = sim3.network
    start_idx = 8500
    end_idx = 11250
    interval = 25
    query_start = (9500 - 8500)/25
    query_end = (10250 - 8500)/25
    f_pl = f[net.J_place_indices, start_idx:end_idx:interval] # 9500-10250 query
    f_pl = np.flip(f_pl, axis=0)
    f_ep = f[net.J_episode_indices, start_idx:end_idx:interval] # 9500-10250 query
    f_ep = np.flip(f_ep, axis=0)
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 4]})
    camera = Camera(fig)
    f_colors = [cm.hot(i) for i in f_pl[:, 0]]
    my_circle=plt.Circle( (0,0), 0.7, color='white')
    axs[0].imshow(repmat(f_ep[:,0], 5, 1), cmap='hot')
    axs[0].set_yticks([])
    axs[0].set_xticks([])
    axs[1].pie(x=np.ones(net.N_pl), colors=f_colors)
    axs[1].add_artist(my_circle)
    for i in range(f_pl.shape[1]):
        f_colors = [cm.hot(i) for i in f_pl[:, i]]
        axs[0].imshow(repmat(f_ep[:,i], 5, 1), cmap='hot')
        axs[1].pie(x=np.ones(net.N_pl), colors=f_colors)
        axs[1].add_artist(my_circle)
        camera.snap()
    Writer = ani.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    
    anim = camera.animate(interval=500, repeat_delay=3000, blit=True)
    anim.save('behav.mp4', writer=writer)
