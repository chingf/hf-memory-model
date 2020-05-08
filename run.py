import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from math import pi
from math import sin
from math import cos
from PlotMaker import PlotMaker
from FFNetwork import FFNetwork
from Simulator import Simulator
from Input import NavigationInput, EpisodeInput, TestFPInput

pm = PlotMaker()

def run_and_plot_nav():
    """ Runs and plots a random network learning the ring structure. """

    num_units = 100
    network = FFNetwork(num_units)
    inputgen = NavigationInput()
    sim = Simulator()
    m, f, inputs = sim.simulate(network, inputgen)
    with open("ffnet_nav.p", "wb") as p:
        pickle.dump(
            {"network": network, "inputgen": inputgen, "m": m, "f": f}, p
            )
    J = network.J
    norm = mcolors.DivergingNorm(
        vmin=J.min(), vmax = J.max(), vcenter=0
        )
    plt.imshow(J, aspect="auto", cmap=plt.cm.coolwarm)
    plt.show()
    plt.imshow(f, aspect="auto")
    plt.show()

def run_and_plot_ep(
        btsp=0.75, noise_mean=0.1, noise_std=0.2,
        J_mean=-0.1, J_std=0.3, btsp_scale=5
        ):
    """ Runs and plots a random network learning the ring structure. """

    num_units = 200
    network = FFNetwork(
        num_units=num_units, btsp_scale=btsp_scale, J_mean=J_mean, J_std=J_std
        )
    inputgen = EpisodeInput(
        T=80, btsp=btsp, noise_mean=noise_mean, noise_std=noise_std
        )
    sim = Simulator()
    m, f, inputs = sim.simulate(network, inputgen)
    plot_J(network, title="J Matrix with one memory")
    plot_formation(
        f, network, inputs, sort=1,
        title="FR, Memory Formation 1\nSorted by Memory 1"
        )
    inputgen = EpisodeInput(
        T=80, btsp=btsp, noise_mean=noise_mean, noise_std=noise_std
        )
    sim = Simulator()
    m, f, inputs = sim.simulate(network, inputgen)
    plot_J(network, title="J Matrix with two memories")
    plot_formation(
        f, network, inputs, sort=1,
        title="FR, Memory Formation 2\nSorted by Memory 1"
        )
    for _ in np.arange(2):
        for use_memory in np.arange(3):
            inputgen = TestFPInput(
                T=80, btsp=btsp, noise_mean=noise_mean, noise_std=noise_std,
                use_memory=use_memory, memory_noise_std=0.02
                )
            m, f, inputs = sim.simulate(network, inputgen)
            plot_recall(f, network, inputs)
    mem = network.memories[0].copy()
    recall = f[:,-1].squeeze()
    mem[mem <= 0] = 0
    mem[mem > 0] = 1
    recall[recall <= 0] = 0
    recall[recall > 0] = 1
    print(np.sum(recall == mem)/recall.size)
    import pdb; pdb.set_trace()

def gridsearch_ep():
    def eval(btsp, noise_mean, noise_std, J_mean, J_std, btsp_scale):
        successes = 0
        num_iters = 40
        sim = Simulator()
        for _ in range(num_iters):
            num_units = 100
            network = FFNetwork(
                num_units=num_units, btsp_scale=btsp_scale,
                J_mean=J_mean, J_std=J_std
                )
            inputgen = EpisodeInput(
                T=80, btsp=btsp, noise_mean=noise_mean, noise_std=noise_std
                )
            m, f, inputs = sim.simulate(network, inputgen)
            inputgen = TestFPInput(
                T=80, btsp=btsp, noise_mean=noise_mean, noise_std=noise_std
                )
            m, f, inputs = sim.simulate(network, inputgen)
            memory = network.memories[0].copy()
            recall = f[:,-1].squeeze()
            memory[memory <= 0] = 0
            memory[memory > 0] = 1
            recall[recall <= 0] = 0
            recall[recall > 0] = 1
            match = np.sum(recall == memory)/recall.size
            is_success = False
            if match > 0.93 and np.sum(memory) > 10:
                is_success = True
            successes += 1 if is_success else 0
        return successes/num_iters
    params = [
        [0.50, 0.75, 1.],
        [-0.2, -0.1, 0., 0.1, 0.2],
        [0.1, 0.2, 0.3],
        [-0.2, -0.1, 0., 0.1, 0.2],
        [0.1, 0.2, 0.3],
        [1, 2.5, 5]
        ]
    eval_results = {
        "BTSP": [], "NoiseMean": [], "NoiseStd": [],
        "JMean": [], "JStd": [], "BTSPScale": [], "Score": []
        }
    for param in list(itertools.product(*params)):
        btsp, noise_mean, noise_std, J_mean, J_std, btsp_scale = param
        score = eval(btsp, noise_mean, noise_std, J_mean, J_std, btsp_scale)
        eval_results["BTSP"].append(btsp)
        eval_results["NoiseMean"].append(noise_mean)
        eval_results["NoiseStd"].append(noise_std)
        eval_results["JMean"].append(J_mean)
        eval_results["JStd"].append(J_std)
        eval_results["BTSPScale"].append(btsp_scale)
        eval_results["Score"].append(score)
    with open("grideval.p", "wb") as f:
        pickle.dump(eval_results, f)

def plot_J(network, sort=False, title=None):
    J = network.J
    gridspec.GridSpec(1, 10)
    plt.subplot2grid((1, 10), (0,0), rowspan=1, colspan=9)
    norm = mcolors.DivergingNorm(vmin=J.min(), vmax = J.max(), vcenter=0)
    plt.imshow(J, aspect="auto", cmap=plt.cm.coolwarm, norm=norm)
    if title is not None:
        plt.title(title)
    plt.show()

def plot_formation(f, network, inputs, sort=False, title=None):
    if sort:
        memory = network.memories[sort-1].copy()
        sorting = np.argsort(memory)
        f = f[sorting, :]
        inputs = inputs[sorting,:]
        memory = np.expand_dims(memory[sorting], axis=1)

    norm = mcolors.DivergingNorm(
        vmin=min(inputs.min(), memory.min()),
        vmax=max(inputs.max(), f.max(), memory.max()), vcenter=0
        )
    fig = plt.figure(figsize=(6,5))
    nrows = 2 
    ncols = 10
    gridspec.GridSpec(nrows,ncols)
    plt.subplot2grid((nrows,ncols), (0,0), rowspan=1, colspan=9)
    plt.imshow(inputs, aspect="auto", cmap=plt.cm.coolwarm, norm=norm)
    plt.subplot2grid((nrows,ncols), (0,9), rowspan=1, colspan=1)
    plt.imshow(memory, cmap=plt.cm.coolwarm, norm=norm, aspect="auto")
    plt.subplot2grid((nrows,ncols), (1,0), rowspan=1, colspan=9)
    plt.imshow(f, aspect="auto", cmap=plt.cm.coolwarm, norm=norm)
    plt.subplot2grid((nrows,ncols), (1,9), rowspan=1, colspan=1)
    plt.imshow(memory, cmap=plt.cm.coolwarm, norm=norm, aspect="auto")
    plt.suptitle(title)
    plt.show()

def plot_recall(f, network, inputs):
    num_memories = len(network.memories)
    fig = plt.figure(figsize=(5*num_memories, 8))
    nrows = 2
    ncols = 10*num_memories
    rowspan = 1
    gridspec.GridSpec(nrows, ncols, figure=fig)
    vmax = max(f.max(), inputs.max())
    vcenter = 0.01
    norm = mcolors.DivergingNorm(vmin=0, vmax = vmax, vcenter=vcenter)
    for m_idx in np.arange(num_memories):
        start_col = m_idx*10
        memory = network.memories[m_idx].copy()
        sorting = np.argsort(memory)
        sorted_inputs = inputs[sorting, :]
        sorted_f = f[sorting, :]
        sorted_mem = np.expand_dims(memory[sorting], axis=1)
        plt.subplot2grid((nrows, ncols), (0,start_col), rowspan=rowspan, colspan=9)
        plt.imshow(sorted_inputs, aspect="auto", cmap=plt.cm.coolwarm, norm=norm)
        plt.subplot2grid((nrows, ncols), (0,start_col+9), rowspan=rowspan, colspan=1)
        plt.imshow(sorted_mem, cmap=plt.cm.coolwarm, norm=norm, aspect="auto")
        plt.xticks([]); plt.yticks([])
        plt.subplot2grid((nrows, ncols), (rowspan,start_col), rowspan=rowspan, colspan=9)
        plt.imshow(sorted_f, aspect="auto", cmap=plt.cm.coolwarm, norm=norm)
        plt.subplot2grid((nrows, ncols), (rowspan,start_col+9), rowspan=rowspan, colspan=1)
        plt.imshow(sorted_mem, cmap=plt.cm.coolwarm, norm=norm, aspect="auto")
        plt.xticks([]); plt.yticks([])
    plt.suptitle("Recall, sorted by both memories")
    plt.show()

def main():
    run_and_plot_ep()

main()
