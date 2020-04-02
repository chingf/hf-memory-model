import numpy as np
from math import pi, cos, sin, radians, degrees, atan2

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

class NavigationInput(Input):
    """ Feeds in navigation input to the place network. """

    def __init__(self, T=1800):
        self.T = T
        self.t = 0

    def get_inputs(self):
        if self.t < self.T:
            period = 150
            loc_t = ((self.t % period)/period)*(2*pi)
            input_t = np.zeros(self.network.num_units)
            input_t[self.network.J_place_indices] = self._get_sharp_cos(loc_t)
            alpha_t = 1.
            input_t[input_t < 0] = 0
            self.inputs[self.t,:] = input_t
            self.alphas[self.t] = alpha_t
            self.t += 1
            return input_t, alpha_t, 1
        else:
            raise StopIteration

class MultiCacheInput(Input):
    """ Feeds in random noise into episode network and navigation input. """

    def __init__(self, K_ep):
        self.t = 0
        self.noise_length = 30
        self.query_length = 33
        self.fastlearn_length = 5
        self.cache_length = self.query_length + self.fastlearn_length
        self.navigation_length = self.cache_length*10
        self.module_length = self.cache_length + self.navigation_length
        self.cache_locs = [0, 2*pi/3, 4*pi/3]
        self.cache_idx = 0
        self.caching = True
        self.K_ep = K_ep
        self.T = self.module_length*2 + self.cache_length

    def get_inputs(self):
        if self.t < self.T:
            if self.caching:
                input_t, alpha_t, fastlearn = self._get_cache_inputs()
            else: # Navigating
                prev_loc = self.cache_locs[self.cache_idx - 1]
                cache_loc = self.cache_locs[self.cache_idx]
                nav_length = self.module_length
                loc_t = ((self.t % nav_length)/nav_length)*(cache_loc - prev_loc)
                loc_t += prev_loc
                input_t = np.zeros(self.network.num_units)
                input_t[self.network.J_place_indices] += self._get_sharp_cos(loc_t)
                input_t[self.network.J_episode_indices] += np.random.normal(
                    0, 0.5, self.network.N_ep
                    )
                input_t[input_t < 0] = 0
                alpha_t = 1. 
                fastlearn = False
                if (self.t + 1) % self.module_length == 0: self.caching = True
        else:
            raise StopIteration
        self.inputs[self.t,:] = input_t
        self.alphas[self.t] = alpha_t
        self.t += 1
        return input_t, alpha_t, fastlearn

    def _get_cache_inputs(self):
        t = self.t % self.cache_length
        cache_loc = self.cache_locs[self.cache_idx]
        fastlearn = False
        input_t = np.zeros(self.network.num_units)
        if t < self.query_length: # Activate episode network and let settle
            if t < self.noise_length:
                input_t[self.network.J_episode_indices] = np.random.normal(
                    0, 0.5, self.network.N_ep
                    ) + self.K_ep
            input_t[self.network.J_place_indices] += self._get_sharp_cos(cache_loc)
        elif self.query_length <= t < self.cache_length: # Fast learn
            fastlearn = True
        input_t[input_t < 0] = 0
        alpha_t = 1.
        if t == self.cache_length - 1:
            self.caching = False
            self.cache_idx += 1
        return input_t, alpha_t, fastlearn

class BehavioralInput(Input):
    """
    Creates input that is chickadee-like. Specifically:
        - Movement from 0 around the whole ring then to loc (10 sec)
        - 2s query for a seed
        - Move to seed location (4 sec)
    """

    def __init__(self, pre_seed_loc=3*pi/2, K_pl=0, K_ep=0):
        self.pre_seed_loc = pre_seed_loc 
        self.t = 0
        self.target_seed = np.nan
        self.event_times = [12, 16, 18, 27]
        self.event_times = [1, 1.2, 1.35, 2]
        self.event_times = [8, 9., 9.5, 12]
        self.T_sec = self.event_times[-1]
        self.K_pl = K_pl
        self.K_ep = K_ep

    def set_current_activity(self, f):
        if self.t <= self.to_frames(self.event_times[1]) + 1:
            self.f = f

    def get_inputs(self):
        T1, T2, T3, T4 = self.event_times
        t = self.t
        nav_scale=0.7
        if self.to_seconds(t) < T1: # Free navigation to loc
            loc_t = ((t/(self.to_frames(T1))) * (2*pi + self.pre_seed_loc)) % (2*pi)
            loc_t = loc_t//(2*pi/16) * (2*pi/16)
            input_t = np.zeros(self.network.num_units)
            input_t[self.network.J_place_indices] = self._get_sharp_cos(loc_t)*nav_scale
            input_t[self.network.J_episode_indices] += np.random.normal(
                0, 0.5, self.network.N_ep
                )
            input_t[input_t < 0] = 0
            alpha_t = 1.
        elif self.to_seconds(t) < T3: # Query for seed
            input_t = np.zeros(self.network.num_units)
            input_t[self.network.J_episode_indices] = np.random.normal(
                0, 0.5, self.network.N_ep
                ) + self.K_ep
            input_t[self.network.J_place_indices] += self._get_sharp_cos(
                self.pre_seed_loc
                )*nav_scale
            input_t[input_t < 0] = 0
            alpha_t = 1. if t < self.to_frames(T2) else 0
        elif self.to_seconds(t) < T4: # Navigation to seed
            if np.isnan(self.target_seed):
                self.set_seed()
            nav_start = self.to_frames(T3)
            nav_time = self.to_frames(T4 - T3)
            nav_distance = self.target_seed - self.pre_seed_loc
            loc_t = ((t - nav_start)/nav_time)*nav_distance + self.pre_seed_loc
            input_t = np.zeros(self.network.num_units)
            input_t[self.network.J_place_indices] = self._get_sharp_cos(loc_t)*nav_scale
            input_t[self.network.J_episode_indices] += np.random.normal(
                0, 0.5, self.network.N_ep
                )
            input_t[input_t < 0] = 0
            alpha_t = 1.
        elif t < self.T: # End input, but let network evolve
            input_t = np.zeros(self.network.num_units)
            alpha_t = 0
        else:
            raise StopIteration
        self.inputs[self.t,:] = input_t
        self.alphas[self.t] = alpha_t
        self.t += 1
        return input_t, alpha_t, 0

    def to_seconds(self, frame):
        return frame/self.network.steps_in_s

    def to_frames(self, sec):
        return sec*self.network.steps_in_s

    def set_seed(self):
        place_f = self.f[self.network.J_place_indices]
        place_locs = np.linspace(0, 2*pi, self.network.N_pl, endpoint=False)
        place_locs = place_locs[place_f > 0]
        place_weights = place_f[place_f > 0]
        if place_locs.size == 0:
            target_loc = pi
        else:
            target_loc = np.average(place_locs, weights=place_weights)
#        x = y = 0.
#        for angle, weight in zip(place_locs, place_weights):
#            x += cos(angle) * weight
#            y += sin(angle) * weight
#        target_loc = degrees(atan2(y, x))
        print(target_loc/(2*pi))
        self.target_seed = target_loc

    def set_network(self, network):
        self.network = network
        self.f = np.zeros(network.num_units)
        self.T = int(self.to_frames(self.T_sec))
        self.inputs = np.zeros((self.T, network.num_units))
        self.alphas = np.zeros(self.T)

