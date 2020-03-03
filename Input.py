import numpy as np
from math import pi

class Input(object):
    """
    An object that generates inputs to feed into a Network object.
    """

    def __init__(self):
        pass

    def set_network(self, network):
        self.network = network
        self.f = np.zeros(network.num_units)
        self.inputs = np.zeros((self.T, network.num_units))
        self.alphas = np.zeros(self.T)

    def set_current_activity(self, f):
        self.f = f

    def _get_sharp_cos(self, loc=pi):
        """ Returns a sharp sinusoidal curve that drops off rapidly """

        mu = 0
        kappa = self.network.kappa
        vonmises_gain = self.network.vonmises_gain
        loc_idx = int(loc/(2*pi)*self.network.N_pl)
        x = np.linspace(-pi, pi, self.network.N_pl, endpoint=False)
        curve = np.exp(kappa*np.cos(x-mu))/(2*pi*np.i0(kappa))
        curve -= np.max(curve)/2.
        curve *= vonmises_gain
        curve = np.roll(curve, loc_idx - self.network.N_pl//2)
        return curve

class NoisyInput(Input):
    """ Feeds in random noise into the episode network. """

    def __init__(self, T=200, noise_length=100):
        self.T = T
        self.noise_length = noise_length
        self.t = 0

    def get_inputs(self):
        if self.t < self.noise_length:
            input_t = np.zeros(self.network.num_units)
            input_t[self.network.J_episode_indices] = np.random.normal(
                0, 1, self.network.N_ep
                )
            alpha_t = 0.6
        elif self.t < self.T:
            input_t = np.zeros(self.network.num_units)
            alpha_t = 0
        else:
            raise StopIteration
        self.inputs[self.t,:] = input_t
        self.alphas[self.t] = alpha_t
        self.t += 1
        return input_t, alpha_t

class NavigationInput(Input):
    """ Feeds in navigation input to the place network. """

    def __init__(self, input_order=0, T=800):
        self.T = T
        self.input_order = input_order
        self.t = 0

    def get_inputs(self):
        if self.t < self.T:
            period = 800
            loc_t = ((self.t % period)/period)*(2*pi)
            input_t = np.zeros(self.network.num_units)
            input_t[self.network.J_place_indices] = self._get_sharp_cos(loc_t)
            alpha_t = 0.6
        else:
            raise StopIteration
        self.inputs[self.t,:] = input_t
        self.alphas[self.t] = alpha_t
        self.t += 1
        return input_t, alpha_t

class IndependentInput(Input):
    """ Feeds in random noise into episode network, then navigation input. """

    def __init__(self, input_order=0, T=300):
        self.T = T
        self.input_order = input_order
        self.t = 0

    def get_inputs(self):
        if self.t < self.T//2:
            loc_t = (self.t % 100)/(2*pi)
            input_t = np.zeros(self.network.num_units)
            input_t[self.network.J_place_indices] = self._get_sharp_cos(loc_t)
            alpha_t = 0.6
        elif self.t < self.T:
            input_t = np.zeros(self.network.num_units)
            input_t[self.network.J_episode_indices] = np.random.normal(
                0, 1, self.network.N_ep
                )
            alpha_t = 0.6
        else:
            raise StopIteration
        self.inputs[self.t,:] = input_t
        self.alphas[self.t] = alpha_t
        self.t += 1
        return input_t, alpha_t

class BehavioralInput(Input):
    """
    Creates input that is chickadee-like. Specifically:
        - Movement from 0 around the whole ring then to loc (10 sec)
        - 2s query for a seed
        - Move to seed location (4 sec)
    """

    def __init__(self, pre_seed_loc=3*pi/2):
        self.T = 300
        self.pre_seed_loc = pre_seed_loc 
        self.t = 0
        self.target_seed = np.nan
        self.event_times = [12, 16, 18, 27]
        self.event_times = [s + 10 for s in self.event_times]
        self.T = 400

    def set_current_activity(self, f):
        if self.t <= (self.event_times[1]*10) + 1:
            self.f = f

    def get_inputs(self):
        T1, T2, T3, T4 = self.event_times
        t = self.t
        if self.to_seconds(t) < T1: # Free navigation to loc
            loc_t = ((t/(T1*10)) * (2*pi + self.pre_seed_loc)) % (2*pi)
            input_t = np.zeros(self.network.num_units)
            input_t[self.network.J_place_indices] = self._get_sharp_cos(loc_t)
            alpha_t = 0.6
        elif self.to_seconds(t) < T3: # Query for seed
            input_t = np.zeros(self.network.num_units)
            input_t[self.network.J_episode_indices] = np.random.normal(
                0, 1, self.network.N_ep
                )
            alpha_t = 0.6 if t < (T2*10) else 0
        elif self.to_seconds(t) < T4: # Navigation to seed
            if np.isnan(self.target_seed):
                self.set_seed()
            nav_start = T3*10 
            nav_time = (T4 - T3) *10
            nav_distance = self.target_seed - self.pre_seed_loc
            loc_t = ((t - nav_start)/nav_time)*nav_distance + self.pre_seed_loc
            input_t = np.zeros(self.network.num_units)
            input_t[self.network.J_place_indices] = self._get_sharp_cos(loc_t)
            alpha_t = 0.6
        elif t < self.T: # End input, but let network evolve for some time
            input_t = np.zeros(self.network.num_units)
            alpha_t = 0
        else:
            raise StopIteration
        self.inputs[self.t,:] = input_t
        self.alphas[self.t] = alpha_t
        self.t += 1
        return input_t, alpha_t

    def to_seconds(self, t):
        return t/10.

    def set_seed(self):
        place_f = self.f[self.network.J_place_indices]
        self.target_seed = (np.argmax(place_f)/self.network.N_pl)*(2*pi)
