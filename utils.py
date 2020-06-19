import pickle
import itertools
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import seaborn as sns
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

def plot_eval(eval_f, net_f):
    with open(eval_f, "rb") as f:
        results = pickle.load(f)
        P = results["P"]
        FR = results["FR"]
    with open(net_f, "rb") as f:
        network = pickle.load(f)
        if type(network) is dict:
            network = network["Network"]
    
    mem_fr = [[] for _ in range(len(network.memories))]
    mem_fr = []
    mem_id = []
    cmap = cm.get_cmap('Accent')
    for loc in range(len(FR)):
        for mem_idx in range(len(network.memories)):
            mem_fr.extend(FR[loc][mem_idx])
            mem_id.extend([mem_idx+1 for _ in range(len(FR[loc][mem_idx]))])
    df = pd.DataFrame({"Memory":mem_id, "FR": mem_fr})
    fig, ax = plt.subplots(figsize=(6,4))
    sns.stripplot(
        x="Memory", y="FR", data=df, alpha=0.3, size=4, jitter=.15, ax=ax
        )
    plt.title("Activity During Recall", fontsize=14)
    plt.xlabel("Memory", fontsize=12)
    plt.ylabel("Mean FR of Encoding Units", fontsize=12)
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    num_locs = P.shape[0]
    mem_locs = np.linspace(0, 2*pi, len(network.memories), endpoint=False)
    current_palette = sns.color_palette()
    for idx in range(P.shape[1]):
        if idx == P.shape[1] - 1:
            plt.plot(
                np.linspace(0, 2*pi, num_locs, endpoint=False), P[:,idx],
                linewidth=2, color="gray", label="No Recall"
                )
        else:
            rgba = current_palette[idx]
            plt.plot(
                np.linspace(0, 2*pi, num_locs, endpoint=False), P[:,idx],
                linewidth=2, color=rgba, label="Memory %d"%(idx+1)
                )
            memory = network.memories[idx]
            mem_loc = mem_locs[idx]
            mem_pl = np.argwhere(memory[network.J_pl_indices] > 0).squeeze()
            mem_pl = np.split(mem_pl, np.where(np.diff(mem_pl) != 1)[0]+1)
            for mem_pl_segment in mem_pl:
                min_loc = (mem_pl_segment[0]/network.N_pl)*2*pi
                max_loc = (mem_pl_segment[-1]/network.N_pl)*2*pi
                plt.axvspan(min_loc, max_loc, color=rgba, alpha=0.4)
    plt.legend()
    plt.xlabel("Location of Recall", fontsize=12)
    plt.ylabel("Probability of Recall", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.xticks([0, pi, 2*pi], ["0", "\u03C0", "2\u03C0"])
    plt.title("Memory Recall Distribution", fontsize=14)
    plt.show()

def get_spatial_fields(f, inputs, network, locs):
    #locs = inputs[network.J_pl_indices, :]
    #locs = np.argmax(locs, axis=0).squeeze()
    spatial_f = np.zeros((network.num_units, network.N_pl))
    for idx, loc in enumerate(np.arange(network.N_pl)):
        bin_fr = np.mean(f[:,locs == loc], axis=1)
        spatial_f[:,idx] = bin_fr
    f_mean = np.mean(f, axis=1)
    f_std = np.mean(f, axis=1)
    threshold = f_mean + 2*f_std
    threshold = np.tile(threshold, (network.N_pl, 1)).T
    sig_fields_f = spatial_f.copy()
    sig_fields_f[spatial_f < threshold] = 0
    sig_fields = np.argmax(sig_fields_f, axis=1)
    sig_fields[np.sum(sig_fields_f, axis=1)==0] = -1
    return spatial_f, sig_fields

def get_spatial_mi(f, inputs, network, locs):
    norm_spatial_info = np.zeros(f.shape[0])
    #locs = inputs[network.J_pl_indices, :]
    #locs = np.argmax(locs, axis=0).squeeze()
    spatial_info = get_mutual_info(locs, f)
    shuffled_info = []
    num_shuffs = 110
    for _ in range(num_shuffs):
        shuffled_f = circular_shuffle(f)
        shuffled_info.append(get_mutual_info(locs, shuffled_f))
    shuffled_info = np.array(shuffled_info)
    shuffled_exceeded = shuffled_info < np.tile(spatial_info, (num_shuffs, 1))
    sig_neurs = np.sum(shuffled_exceeded, axis=0) > 0.99*num_shuffs
    norm_spatial_info = \
        spatial_info/np.mean(shuffled_info, axis=0)
    return norm_spatial_info, sig_neurs

def get_mutual_info(contexts, f):
    mean_f = np.mean(f, axis=1)
    mutual_info = np.zeros(mean_f.shape)
    for ctxt in np.unique(contexts):
        prob = np.sum(contexts==ctxt)/contexts.size
        ctxt_mean_f = np.mean(f[:,contexts==ctxt], axis=1)
        log_term = np.log2(ctxt_mean_f/mean_f)
        log_term[np.isnan(log_term)] = 0
        log_term[np.isinf(log_term)] = 0
        mutual_info += prob*ctxt_mean_f*log_term
    return mutual_info

def circular_shuffle(arr):
    arr = arr.copy()
    shift = np.random.choice(np.arange(1, arr.size))
    if len(arr.shape) == 2:
        for neur in range(arr.shape[0]):
            shift = np.random.choice(np.arange(1, arr.size))
            arr[neur,:] = np.roll(arr[neur,:], shift)
        return arr
    else:
        return np.roll(arr, shift)

def collect_cache_noncache(f, inputs, cache_sites, cache_or_retriev, network, locs):
    #locs = inputs[network.J_pl_indices, :]
    #locs = np.argmax(locs, axis=0).squeeze()
    all_cache_frames = np.argwhere(cache_or_retriev > 0).squeeze()
    #locs[all_cache_frames] = -1
    all_cache_frames = np.split(
        all_cache_frames, np.where(np.diff(all_cache_frames) != 1)[0]+1
        )
    assert(len(all_cache_frames) == len(cache_sites))
    all_noncache_frames = []
    all_noncache_locs = []
    for cache_site in np.unique(cache_sites):
        support_region = [(cache_site + i)%network.N_pl for i in range(-2, 3)]
        noncache_frames = np.argwhere(np.isin(locs, support_region)).squeeze()
        noncache_frames = np.split(
            noncache_frames, np.where(np.diff(noncache_frames) != 1)[0]+1
            )
        all_noncache_frames.extend(noncache_frames)
        all_noncache_locs.extend(
            [cache_site for _ in range(len(noncache_frames))]
            )
    return all_cache_frames, all_noncache_frames, all_noncache_locs

def get_cache_index_mat(f, inputs, cache_sites, cache_or_retriev, network, locs):
    all_cache_frames, all_noncache_frames, all_noncache_locs = \
        collect_cache_noncache(f, inputs, cache_sites, cache_or_retriev, network, locs)
    cache_idx_mat = np.zeros((network.num_units, len(cache_sites)))
    for idx, cache_site in enumerate(cache_sites):
        cache_frames = all_cache_frames[idx]
        noncache_frames = [
            arr for i, arr in enumerate(all_noncache_frames)\
            if all_noncache_locs[i] == cache_site
            ]
        cache_idx_col = calc_cache_index(f, cache_frames, noncache_frames)
        cache_idx_mat[:,idx] = cache_idx_col
    cache_idx_mean = np.mean(cache_idx_mat, axis=1)
    significant = np.zeros(network.num_units)
    shuff_means = []
    num_shuffs = 110
    for _ in range(num_shuffs): # Significance Test with Shuffles
        shuff_cache_idx_mat = np.zeros((network.num_units, len(cache_sites)))
        all_visits = list.copy(all_cache_frames)
        all_visits.extend(list.copy(all_noncache_frames))
        shuff_idxs = np.arange(len(all_visits)).astype(int)
        np.random.shuffle(shuff_idxs)
        all_visits = [all_visits[i] for i in shuff_idxs]
        shuff_cache_frames = all_visits[:len(all_cache_frames)]
        shuff_noncache_frames = all_visits[len(all_cache_frames):]
        for idx, cache_site in enumerate(cache_sites):
            cache_frames = shuff_cache_frames[idx]
            noncache_frames = [
                arr for i, arr in enumerate(shuff_noncache_frames)\
                if all_noncache_locs[i] == cache_site
                ]
            cache_idx_col = calc_cache_index(f, cache_frames, noncache_frames)
            shuff_cache_idx_mat[:,idx] = cache_idx_col
        shuff_cache_idx_mean = np.mean(shuff_cache_idx_mat, axis=1)
        shuff_means.append(shuff_cache_idx_mean)
        significant += (shuff_cache_idx_mean < cache_idx_mean)
    significant = significant > 0.90*num_shuffs
    return cache_idx_mat, significant

def calc_cache_index(fr, cache_frames, noncache_frames):
    cache_frs = np.mean(fr[:,cache_frames], axis=1)
    noncache_frs = []# (Noncache visits, neurs) array
    for noncache_frame in noncache_frames:
        noncache_fr = np.mean(fr[:,noncache_frame], axis=1)
        noncache_frs.append(noncache_fr)
    noncache_frs = np.array(noncache_frs)
    cache_frs = np.tile(cache_frs, (noncache_frs.shape[0], 1))
    cache_idx = np.sum(noncache_frs < cache_frs, axis=0)
    cache_idx = cache_idx/noncache_frs.shape[0]
    return cache_idx
