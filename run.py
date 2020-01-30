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

def gridsearch(name):
    scores, std = GridSearch(overlap=True).run_search()
    results = {}
    results['scores'] = scores
    results['std'] = std
    with open("gridsearch-" + name + ".p", "wb") as f:
        pickle.dump(results, f)

def main(network_type):
    gridsearch("normalized")

if __name__ == "__main__":
    main()
