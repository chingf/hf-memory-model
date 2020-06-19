import pickle
import itertools
import time
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from math import pi
from math import sin
from math import cos
from Simulator import Simulator
from Input import NavigationInput, EpisodeInput
from Input import TestFPInput, AssocInput, TestNavFPInput
from utils import *
from sklearn.preprocessing import minmax_scale

class DataAnalysis(object):
    """
    Functions to run experiment-inspired data analysis given model behavior from
    caching, retrieval, and navigation simulations.

    Args:
        sim_results: A dictionary containing the results of caching, retrieval,
            and navigation simulations of the model.
        cache_repr_filename: String filepath to cache representation analysis;
            if None, representation will be calculated and saved in a pickle file
        spatial_repr_filename: String filepath to spatial representation analysis;
            if None, representation will be calculated and saved in a pickle file
    """

    def __init__(
        self, sim_results, cache_repr_filename=None, spatial_repr_filename=None
        ):

        self.sim_results = sim_results
        self.plot = plot
        if cache_repr_filename is None:
            cache_repr_filename = "cache_analysis.p"
            self._run_cache_analysis(cache_repr_filename)
        if spatial_repr_filename is None:
            spatial_repr_filename = "spatial_analysis.p"
            self._run_spatial_analysis(spatial_repr_filename)
        self.cache_repr_filename = cache_repr_filename
        self.spatial_repr_filename = spatial_repr_filename

    def plot_conjunction(self):
        network = self.network
        with open(self.cache_repr_filename, "rb") as pickle_f:
            results = pickle.load(pickle_f)
            cache_idx_mat = results["Matrix"]
            sig_cache = results["Sig"]
        with open(self.spatial_repr_filename, "rb") as pickle_f:
            results = pickle.load(pickle_f)
            norm_spatial_info = results["Norm Info"]
            sig_spatial = results["Sig"]
            spatial_f = results["Firing Fields"]
            sig_fields = results["Field Peaks"]
        place_info = []
        ep_info = []
        sigs = []
        ring_or_not = []
        for neur in range(network.num_units):
            if sig_spatial[neur] and sig_cache[neur]:
                sigs.append("Place and Cache")
                if neur in network.J_pl_indices: print(neur)
            elif sig_spatial[neur]:
                sigs.append("Place Only")
            elif sig_cache[neur]:
                sigs.append("Cache Only")
            else:
                sigs.append("Neither")
            if neur in network.J_pl_indices:
                ring_or_not.append("Ring Attractor")
            else:
                ring_or_not.append("Unspecified")
            place_info.append(norm_spatial_info[neur])
            ep_info.append(np.mean(cache_idx_mat[neur,:]))
        df = pd.DataFrame({
            "Place": place_info, "Ep": ep_info, "Significance": sigs,
            "Unit Type": ring_or_not
            })
        fig, ax = plt.subplots(figsize=(8,5))
        palette = sns.color_palette()
        color_dict = {
            "Place and Cache": palette[0], "Place Only": palette[1],
            "Cache Only": palette[2], "Neither": "grey"
            }
        sns.scatterplot(
            x="Place", y="Ep", hue="Significance", alpha=0.5,
            data=df, ax=ax, palette=color_dict,
            hue_order = ["Place and Cache", "Place Only", "Cache Only", "Neither"],
            style="Unit Type"
            )
        plt.xlabel("Place Information", fontsize=12)
        plt.ylabel("Cache Information", fontsize=12)
        plt.title("Joint Representation", fontsize=14)
        plt.savefig("conj.png")
        plt.show()
    
    def plot_cache_analysis(self):
        network = self.network
        with open(self.cache_repr_filename, "rb") as pickle_f:
            results = pickle.load(pickle_f)
            cache_idx_mat = results["Matrix"]
            sig_cache = results["Sig"]
        cache_locs = np.linspace(0, network.N_pl, 4, endpoint=False)
        locs = [
            [(l + i)%network.N_pl for i in range(-2, 3)] for l in cache_locs
            ]
        locs = np.concatenate(locs)
        #locs = np.concatenate([locs, cache_locs])
        cache_idx_mat = cache_idx_mat[:,cache_locs.size:]
        cache_idx_mat = cache_idx_mat[:, np.argsort(locs)]
        locs = np.sort(locs)
        ax = sns.heatmap(cache_idx_mat[sig_cache, :], cmap="viridis")
        ax.set_aspect("equal")
        plt.xlabel("Cache Location", fontsize=12)
        plt.ylabel("Unit", fontsize=12)
        plt.yticks([])
        plt.xticks(
            [0, locs.size//2, locs.size], ["0", "\u03C0", "2\u03C0"],
            rotation=0
            )
    #    plt.xticks(
    #        [0, locs.size//2, locs.size-1], ["0", "\u03C0", "2\u03C0"],
    #        rotation=0
    #        )
        plt.title("Sorted Retrieval Tuning", fontsize=14)
        plt.savefig("cache.png")
        plt.show()
    
    def plot_spatial_analysis(self):
        network = self.network
        with open(self.spatial_f, "rb") as pickle_f:
            results = pickle.load(pickle_f)
            norm_spatial_info = results["Norm Info"]
            spatial_f = results["Firing Fields"]
            sig_fields = results["Field Peaks"]
            sig_neurs = results["Sig"]
        sig_fields = minmax_scale(sig_fields.T).T
        #sig_neurs[network.J_pl_indices] = False
        spatial_f = spatial_f[sig_neurs]
        sig_fields = sig_fields[sig_neurs]
        spatial_f = spatial_f[sig_fields != -1]
        sig_fields = sig_fields[sig_fields != -1]
        sns.heatmap(spatial_f[np.argsort(sig_fields), :], cmap="viridis")
        print(np.sort(sig_fields))
        num_bins = network.N_pl
        plt.yticks([])
        plt.xticks(np.linspace(0,num_bins,3), ["0", "\u03C0", "2\u03C0"], rotation=0)
        plt.ylabel("Unit", fontsize=12)
        plt.xlabel("Location", fontsize=12)
        plt.title("Sorted Place Fields", fontsize=14)
        plt.savefig("spatial.png")
        plt.show()

    def _run_cache_analysis(self, filename):
        results = self.results
        network = self.network
        all_f = []
        all_inputs = []
        all_input_locs = []
        cache_or_retriev = []
        cache_sites = []
        cache_locations = np.linspace(0, network.N_pl, 4, endpoint=False)
        for idx, assoc_result in enumerate(results["Assoc"]):
            m, f, inputs, J, J_ext = assoc_result
            ep_inputs = inputs[network.J_ep_indices,:]
            caching = (np.sum(ep_inputs, axis=0) > 0).astype(int)
            cache_sites.append(cache_locations[idx])
            input_locs = inputs[network.J_pl_indices, :]
            input_locs = np.argmax(input_locs, axis=0).squeeze()
            input_locs[caching.astype(bool)] = cache_locations[idx]
            all_f.append(f[:, start_idx:])
            all_inputs.append(inputs[:, start_idx:])
            all_input_locs.append(input_locs[start_idx:])
            cache_or_retriev.append(caching[start_idx:])
            begin_cache = np.argwhere(caching > 0).squeeze()[0]
            end_cache = np.argwhere(caching > 0).squeeze()[-1]
        sim = Simulator(network)
        locs = cache_locations.copy()#np.arange(0, 100, 10)
        locs = [
            [(loc + i)%network.N_pl for i in range(-2, 3)] for loc in locs
            ]
        locs = np.concatenate(locs)
        locs = np.tile(locs, (1, 3)).squeeze()
        for loc in locs:
            inputgen = TestNavFPInput(recall_loc=loc, network=network)
            m, f, inputs = sim.simulate(inputgen)
            ep_inputs = inputs[network.J_ep_indices,:]
            retrieving = np.zeros(ep_inputs.shape[1]).astype(int)
            retrieving[inputgen.recall_start:inputgen.recall_end] = 2
            input_locs = inputs[network.J_pl_indices, :]
            input_locs = np.argmax(input_locs, axis=0).squeeze()
            input_locs[retrieving.astype(bool)] = loc
            cache_sites.append(loc)
            all_f.append(f[:, start_idx:])
            all_inputs.append(inputs[:, start_idx:])
            all_input_locs.append(input_locs[start_idx:])
            cache_or_retriev.append(retrieving[start_idx:])
        all_f = np.hstack(all_f)
        all_inputs = np.hstack(all_inputs)
        all_input_locs = np.concatenate(all_input_locs)
        cache_or_retriev = np.concatenate(cache_or_retriev)
        norm_spatial_info, sig_neurs = get_spatial_mi(
            all_f, all_inputs, network, all_input_locs
            )
        spatial_f, sig_fields = get_spatial_fields(
            all_f, all_inputs, network, all_input_locs
            )
        with open(filename, "wb") as pickle_f:
            results = {
                "Norm Info": norm_spatial_info, "Sig": sig_neurs,
                "Firing Fields": spatial_f, "Field Peaks": sig_fields
                }
            pickle.dump(results, pickle_f)
    
    def _run_spatial_analysis(self, filename):
        results = self.results
        network = self.network
        all_f = []
        all_inputs = []
        all_input_locs = []
        for idx, assoc_result in enumerate(results["Assoc"]):
            m, f, inputs, J, J_ext = assoc_result
            ep_inputs = inputs[network.J_ep_indices,:]
            caching = (np.sum(ep_inputs, axis=0) > 0).astype(bool)
            not_caching = np.logical_not(caching)
            not_caching[:start_idx] = False
            input_locs = inputs[network.J_pl_indices, :]
            input_locs = np.argmax(input_locs, axis=0).squeeze()
            all_f.append(f[:,not_caching])
            all_inputs.append(inputs[:,not_caching])
            all_input_locs.append(input_locs[not_caching])
        sim = Simulator(network)
        locs = np.linspace(0, network.N_pl, 4, endpoint=False)
        locs = [
            [(loc + i)%network.N_pl for i in range(-2, 3)] for loc in locs
            ]
        locs = np.concatenate(locs)
        #locs = np.tile(locs, (1, 3)).squeeze()
        for loc in locs:
            inputgen = TestNavFPInput(recall_loc=loc, network=network)
            m, f, inputs = sim.simulate(inputgen)
            not_retrieving = np.ones(inputs.shape[1]).astype(bool)
            not_retrieving[inputgen.recall_start:inputgen.recall_end] = False
            not_retrieving[:start_idx] = False
            input_locs = inputs[network.J_pl_indices, :]
            input_locs = np.argmax(input_locs, axis=0).squeeze()
            all_f.append(f[:, not_retrieving])
            all_inputs.append(inputs[:, not_retrieving])
            all_input_locs.append(input_locs[not_retrieving])
        all_f = np.hstack(all_f)
        all_inputs = np.hstack(all_inputs)
        all_input_locs = np.concatenate(all_input_locs)
        norm_spatial_info, sig_neurs = get_spatial_mi(
            all_f, all_inputs, network, all_input_locs
            )
        spatial_f, sig_fields = get_spatial_fields(all_f, all_inputs, network, all_input_locs)
        with open(filename, "wb") as pickle_f:
            results = {
                "Norm Info": norm_spatial_info, "Sig": sig_neurs,
                "Firing Fields": spatial_f, "Field Peaks": sig_fields
                }
            pickle.dump(results, pickle_f)
    
