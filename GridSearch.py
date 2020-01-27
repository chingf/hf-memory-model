import pdb
import numpy as np
from math import pi

from InputGenerator import InputGenerator
from Network import *

class GridSearch(object):
    """Performs grid search over parameters for a network"""

    def __init__(self, Ncr_max=60, K_max=0.5, Jcr_max=0.09):
        self.N = 100
        self.N_c = 2
        self.target_indices = np.array([self.N//2, 0])
        self.T = 1250
        self.num_reps = 25
        self.C = 1
        Ncr_step = 5
        K_step = 0.025
        Jcr_step = 0.005
        self.Ncr_range = np.array([1, 2, 4, 8, 16, 32, 40])
        self.K_range = np.arange(0., K_max + K_step, K_step)
        self.Jcr_range = np.arange(0.01, Jcr_max + Jcr_step, Jcr_step)

    def run_search(self):
        """
        Looks at some parameters or something
        """

        # Define parameter range and input
        input_ext, input_c, alphas = InputGenerator().get_input(self.T, self.N_c)
        scores = np.zeros(
            (self.Ncr_range.size, self.K_range.size, self.Jcr_range.size)
            )*np.nan
        std = np.zeros(
            (self.Ncr_range.size, self.K_range.size, self.Jcr_range.size)
            )*np.nan

        # Define the variables needed to evaluate each parameter set
        target_end_indices = [np.argwhere(i > 0)[-1,0] for i in input_c.T]
        target_width = 2
        nav_end = np.argwhere(alphas == 0)[0, 0]
        nav_restart = np.argwhere(alphas == 0)[-1, 0] + 1

        for Ncr_idx, Ncr in enumerate(self.Ncr_range):
            print()
            print(Ncr)
            for K_idx, K in enumerate(self.K_range):
                print(K)
                for Jcr_idx, Jcr in enumerate(self.Jcr_range):
                    f_avg, f_std = self._get_network_score(
                        Ncr, K, Jcr, input_ext, input_c, alphas
                        )

                    # Record the average std dev during context situations 
                    std[Ncr_idx, K_idx, Jcr_idx] = np.mean(
                        f_std[:, nav_end:nav_restart]
                        )

                    # Parameter set is valid if the bump jumps to contexts
                    valid_bump_switch = True
                    for target_num, target_idx in enumerate(self.target_indices):
                        bump_location = np.argmax(
                            f_avg[:, target_end_indices[target_num]]
                            )
                        if not self._bump_loc_correct(
                            bump_location, target_idx, target_width
                            ):
                            valid_bump_switch = False
                    if not valid_bump_switch:
                        continue

                    # Evaluate how biased f is to contexts during navigation
                    bias_strength = []
                    for target_num, target_idx in enumerate(self.target_indices):
                        context0 = (target_idx + target_width) % self.N
                        noncontext0 = (context0 + 1) % self.N
                        bias_strength.append(np.linalg.norm(
                            f_avg[context0, :nav_end] - f_avg[noncontext0, :nav_end]
                            ))
                        context1 = (target_idx - target_width) % self.N
                        noncontext1 = (context1 - 1) % self.N
                        bias_strength.append(np.linalg.norm(
                            f_avg[context1, :nav_end] - f_avg[noncontext1, :nav_end]
                            ))

                    # Record the bias
                    scores[Ncr_idx, K_idx, Jcr_idx] = np.mean(bias_strength)
        return scores, std

    def _get_network_score(self, Ncr, K, Jcr, input_ext, input_c, alphas):
        """
        Randomly initializes the network and runs it multiple times.
        """

        fs = []

        # Run the network multiple times
        for _ in range(self.num_reps):
            network = PlasticMixedNetwork(
                self.N, self.N_c, self.C, K, Ncr, Jcr, self.target_indices
                )
            _, f, _ = network.simulate(input_ext, input_c, alphas)
            fs.append(f.reshape(self.N*self.T,))

        f_avg = np.mean(fs, axis=0).reshape(self.N, self.T)
        f_std = np.std(fs, axis=0).reshape(self.N, self.T)
        return f_avg, f_std

    def _bump_loc_correct(self, bump_loc, target_idx, target_width):
        """
        Checks if bump location is correct. Modulo
        """

        lower_bound = target_idx - target_width
        upper_bound = target_idx + target_width
        correct_locations = [
            i%self.N for i in np.arange(lower_bound, upper_bound + 1)
            ] 
        return bump_loc in correct_locations

