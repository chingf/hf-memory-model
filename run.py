import pickle
import itertools
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from math import pi
from math import sin
from math import cos
from PlotMaker import PlotMaker
from MixedRNN import MixedRNN
from HebbRNN import HebbRNN
from BtspRNN import BtspRNN
from Simulator import Simulator
from Input import NavigationInput, EpisodeInput, TestFPInput, AssocInput
from Input import TestNavFPInput

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
        plasticity=0.75, noise_mean=0.1, noise_std=0.2,
        J_mean=-0.1, J_std=0.3, plasticity_scale=5
        ):
    """ Runs and plots a random network learning the ring structure. """

    num_units = 200
    network = FFNetwork(
        num_units=num_units, plasticity_scale=plasticity_scale, J_mean=J_mean, J_std=J_std
        )
    inputgen = EpisodeInput(
        T=80, plasticity=plasticity, noise_mean=noise_mean, noise_std=noise_std
        )
    sim = Simulator()
    m, f, inputs = sim.simulate(network, inputgen)
    plot_J(network, title="J Matrix with one memory")
    plot_formation(
        f, network, inputs, sort=1,
        title="FR, Memory Formation 1\nSorted by Memory 1"
        )
    inputgen = EpisodeInput(
        T=80, plasticity=plasticity, noise_mean=noise_mean, noise_std=noise_std
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
                T=80, plasticity=plasticity, noise_mean=noise_mean, noise_std=noise_std,
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

def run_and_plot_assoc(
    noise_mean=-0., noise_std=0.1, J_mean=-0.1, J_std=0.1,
    ext_plasticity=1, plasticity=1,
    ext_plasticity_scale=0.1, plasticity_scale=0.4
    ):
    """ Runs and plots a random network learning the ring structure. """

    network = MixedRNN(
        N_pl=100, N_ep=150, J_mean=J_mean, J_std=J_std,
        ext_plasticity_scale=ext_plasticity_scale, plasticity_scale=plasticity_scale
        )
    sim = Simulator()

    # Association 1
    inputgen = AssocInput(
        noise_mean=noise_mean, noise_std=noise_std, cache_loc=2*pi/3,
        ext_plasticity=ext_plasticity, plasticity=plasticity
        )
    m, f, inputs = sim.simulate(network, inputgen)
    plot_J(network.J, network, sortby=network.memories[0], title="J Matrix")
    plot_J(network.J_ext, network, sortby=network.ext_memories[0], title="Read-In Matrix")
    plot_formation(
        f, network, inputs, sortby=network.memories[0],
        title="Navigation and Association 1 (Sorted by RNN Memory)"
        )

    # Association 2
    inputgen = AssocInput(
        noise_mean=noise_mean, noise_std=noise_std, cache_loc=4*pi/3,
        ext_plasticity=ext_plasticity, plasticity=plasticity
        )
    m, f, inputs = sim.simulate(network, inputgen)
    plot_J(network.J, network, sortby=network.memories[1], title="J Matrix")
    plot_J(network.J_ext, network, sortby=network.memories[1], title="Read-In Matrix")
    plot_formation(
        f, network, inputs, sortby=network.memories[0],
        title="Navigation and Association 2 (Sorted by RNN Memory)"
        )

    # Random Recall
    inputgen = TestFPInput(
        plasticity=plasticity, noise_mean=noise_mean, noise_std=noise_std,
        memory_noise_std=0.02
        )
    m, f, inputs = sim.simulate(network, inputgen)
    plot_formation(
        f, network, inputs, sortby=network.ext_memories[0],
        title="Recall (Sorted by Readin Memory)"
        )
    ep_mem_0 = np.argwhere(network.memories[0][network.J_ep_indices] > 0).squeeze()
    ep_mem_1 = np.argwhere(network.memories[1][network.J_ep_indices] > 0).squeeze()
    print("Size of Memory 0`: %d"%np.sum(ep_mem_0 > 0))
    print("Size of Memory 1`: %d"%np.sum(ep_mem_1 > 0))
    print(np.intersect1d(ep_mem_0, ep_mem_1))
    test_navfp(network)
    import pdb; pdb.set_trace()

def eval_recall(memory, recall, network):
    memory = memory.copy()
    recall = recall.copy()
    memory = memory[network.J_pl_indices]
    recall = recall[network.J_pl_indices]
    memory[memory <= 0] = 0
    memory[memory > 0] = 1
    recall[recall <= 0] = 0
    recall[recall > 0] = 1
    match = np.sum(recall == memory)/recall.size
    return match

def eval_memory(memory):
    return np.sum(memory[145:155]) < 0.01

def eval_random_recall(recall, network):
    recall = recall[network.J_pl_indices]
    recall_0 = np.sum(recall[45:55]) > 0
    recall_1 = (np.sum(recall[-5:]) + np.sum(recall[:5])) > 0
    return recall_0 != recall_1

def gridsearch_ep():
    def eval(plasticity, noise_mean, noise_std, J_mean, J_std, plasticity_scale):
        successes = 0
        num_iters = 40
        sim = Simulator()
        for _ in range(num_iters):
            num_units = 100
            network = FFNetwork(
                num_units=num_units, plasticity_scale=plasticity_scale,
                J_mean=J_mean, J_std=J_std
                )
            inputgen = EpisodeInput(
                T=80, plasticity=plasticity, noise_mean=noise_mean, noise_std=noise_std
                )
            m, f, inputs = sim.simulate(network, inputgen)
            inputgen = TestFPInput(
                T=80, plasticity=plasticity, noise_mean=noise_mean, noise_std=noise_std
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
        plasticity, noise_mean, noise_std, J_mean, J_std, plasticity_scale = param
        score = eval(plasticity, noise_mean, noise_std, J_mean, J_std, plasticity_scale)
        eval_results["BTSP"].append(plasticity)
        eval_results["NoiseMean"].append(noise_mean)
        eval_results["NoiseStd"].append(noise_std)
        eval_results["JMean"].append(J_mean)
        eval_results["JStd"].append(J_std)
        eval_results["BTSPScale"].append(plasticity_scale)
        eval_results["Score"].append(score)
    with open("grideval.p", "wb") as f:
        pickle.dump(eval_results, f)

def gridsearch_assoc():
    J_mean = -0.1
    J_std = 0.3
    def eval(
        noise_mean, noise_std, plasticity_scale, ext_plasticity_scale,
        plasticity, ext_plasticity, K_inhib, recall_scale
        ):
        successes = 0
        num_iters = 20
        sim = Simulator()
        for _ in range(num_iters):
            network = MixedRNN(
                N_pl=100, N_ep=100, J_mean=J_mean, J_std=J_std,
                ext_plasticity_scale=ext_plasticity_scale,
                plasticity_scale=plasticity_scale, K_inhib=K_inhib
                )
            inputgen = AssocInput(
                noise_mean=noise_mean, noise_std=noise_std, cache_loc=pi,
                ext_plasticity=ext_plasticity, plasticity=plasticity
                )
            m, f, inputs = sim.simulate(network, inputgen)
            inputgen = AssocInput(
                noise_mean=noise_mean, noise_std=noise_std, cache_loc=2*pi,
                ext_plasticity=ext_plasticity, plasticity=plasticity
                )
            m, f, inputs = sim.simulate(network, inputgen)
            # Check that second memory has some kind of place memory
            if np.sum(network.memories[1][network.J_pl_indices] > 0) < 2:
                continue
            # Check that second memory doesn't overlap with first
            if not eval_memory(network.memories[1]):
                continue
            # Check that random recall doesn't recall both memories
            inputgen = TestFPInput(
                T=80, plasticity=plasticity, noise_mean=noise_mean,
                noise_std=noise_std, recall_scale=recall_scale
                )
            m, f, inputs = sim.simulate(network, inputgen)
            if not eval_random_recall(f[:,-1].squeeze(), network):
                continue
            # Check that memory noise std is what you want
            inputgen = TestFPInput(
                T=80, plasticity=plasticity, noise_mean=noise_mean, noise_std=noise_std,
                use_memory=network.memories[0], memory_noise_std=0.02
                )
            m, f, inputs = sim.simulate(network, inputgen)
            score_0 = eval_recall(network.memories[0], f[:,-1].squeeze(), network)
            success_0 = (1 - score_0) < np.sum(network.memories[0]/network.memories[0].size)
            inputgen = TestFPInput(
                T=80, plasticity=plasticity, noise_mean=noise_mean, noise_std=noise_std,
                use_memory=network.memories[1], memory_noise_std=0.02
                )
            m, f, inputs = sim.simulate(network, inputgen)
            score_1 = eval_recall(network.memories[1], f[:,-1].squeeze(), network)
            success_1 = (1 - score_1) < np.sum(network.memories[1]/network.memories[1].size)
            is_success = False
            if success_0 and success_1:
                is_success = True
            successes += 1 if is_success else 0
        return successes/num_iters
    params = [
        [-0.1, 0, 0.1],
        [0.2],
        [0.3, 0.6, 0.9],
        [0.3, 0.6, 0.9],
        [0.25, 0.5, 0.75],
        [0.25, 0.5, 0.75],
        [0.2],
        [2]
        ]
    eval_results = {
        "NoiseMean": [], "NoiseStd": [], "HebbScale": [], "BTSPScale":[],
        "HebbProb": [], "BTSPProb": [], "Inhib": [], "RecallScale": [],
        "Score": []
        }
    param_idx = 0
    for param in list(itertools.product(*params)):
        if param_idx % 20 == 0:
            print("On parameter set %d"%param_idx)
        noise_mean, noise_std, plasticity_scale, ext_plasticity_scale, plasticity, ext_plasticity, K_inhib, recall_scale = param
        score = eval(
            noise_mean, noise_std, plasticity_scale, ext_plasticity_scale,
            plasticity, ext_plasticity, K_inhib, recall_scale
            )
        eval_results["NoiseMean"].append(noise_mean)
        eval_results["NoiseStd"].append(noise_std)
        eval_results["HebbScale"].append(plasticity_scale)
        eval_results["BTSPScale"].append(ext_plasticity_scale)
        eval_results["HebbProb"].append(plasticity)
        eval_results["BTSPProb"].append(ext_plasticity)
        eval_results["Inhib"].append(K_inhib)
        eval_results["RecallScale"].append(recall_scale)
        eval_results["Score"].append(score)
        param_idx += 1
    with open("grideval_assoc.p", "wb") as f:
        pickle.dump(eval_results, f)

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
    #run_and_plot_assoc()
    test_navfp()

def test(): 
    with open("btsphebb-5.p", "rb") as f:
        network = pickle.load(f)
    sim = Simulator()
    locs = [125, 123, 121, 119]
    for loc in locs:
        mu = 0
        kappa = 5
        vonmises_gain = 8
        x = np.linspace(-pi, pi, network.num_units, endpoint=False)
        curve = np.exp(kappa*np.cos(x-mu))/(2*pi*np.i0(kappa))
        curve -= np.max(curve)/1.1
        curve *= vonmises_gain
        probe = np.roll(curve, loc - network.num_units//2)
        inputgen = TestFPInput(use_memory=probe)
        m, f, inputs = sim.simulate(network, inputgen)
        plot_formation(
            f, network, inputs, sortby=network.memories[1],
            title="Recall of Memory 2 (Sorted by RNN Memory)"
            )
        plot_formation(
            m, network, inputs, sortby=network.memories[1],
            title="Recall of Memory 2 (Sorted by RNN Memory)"
            )

def test_navfp(network=None):
    if network is None:
        with open("btsphebb2.p", "rb") as f:
            network = pickle.load(f)
        plot_J(network.J, network, sortby=network.memories[0], title="J Matrix")
    sim = Simulator()
    locs = [99 for _ in range(10)]
    for loc in locs:
        print(loc)
        np.random.seed(1)
        inputgen = TestNavFPInput(recall_loc=loc, network=network)
        m, f, inputs = sim.simulate(network, inputgen)
        plot_formation(
            f, network, inputs, sortby=network.memories[1],
            title="Recall (Sorted by RNN Memory 2)"
            )
        plot_formation(
            m, network, inputs, sortby=network.memories[1],
            title="Recall, Membrane Potential (Sorted by RNN Memory 2)"
            )

main()
