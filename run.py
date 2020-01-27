import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from math import pi
from PlotMaker import PlotMaker
from GridSearch import GridSearch
from Network import RingNetwork, SimpleMixedNetwork, MixedNetwork, PlasticMixedNetwork

def gridsearch(overlap, name):
    scores, std = GridSearch(overlap=overlap).run_search()
    results = {}
    results['scores'] = scores
    results['std'] = std
    with open("gridsearch-" + name + ".p", "wb") as f:
        pickle.dump(results, f)

def main(network_type):
    if network_type == "overlap":
        print("Running overlap grid search")
        gridsearch(True, "overlap")
    else:
        print("Running non-overlapping grid search")
        gridsearch(True, "nonoverlap")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Which network?")
    parser.add_argument("t")
    args = parser.parse_args()
    main(args.t)
