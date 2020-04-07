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
            input_t[self.network.J_episode_indices] += np.random.normal(
                0, 0.5, self.network.N_ep
                )
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
        alpha_t *= 2.
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

    def __init__(self, pre_seed_loc, K_pl, K_ep):
        self.pre_seed_loc = pre_seed_loc  # In perch
        self.t = 0
        self.K_pl = K_pl
        self.K_ep = K_ep
        self.query_noise_length = 1 # In sec
        self.query_settle_length = 0.5 # In sec
        self.velocity = 2 # Perches/sec
        self.event_end_times = self._set_event_times()
        self.T_sec = self.event_end_times[-1]

    def get_inputs(self):
        event_end_times = [self.to_frames(e) for e in self.event_end_times]
        T1, T2, T3 = event_end_times
        t = self.t
        if t < T1:
            input_t, alpha_t = self._get_navigation_input(t)
        elif t < T2:
            t -= T1
            input_t, alpha_t = self._get_query_input(t)
        elif t < T3:
            t -= (T2 - T1)
            input_t, alpha_t = self._get_navigation_input(t)
        else:
            raise StopIteration
        self.inputs[self.t,:] = input_t
        self.alphas[self.t] = alpha_t
        self.t += 1
        return input_t, alpha_t, 0

    def set_network(self, network):
        self.network = network
        self.f = np.zeros(network.num_units)
        self.T = int(self.to_frames(self.T_sec))
        self.inputs = np.zeros((self.T, network.num_units))
        self.alphas = np.zeros(self.T)

    def to_seconds(self, frame):
        return frame/self.network.steps_in_s

    def to_frames(self, sec):
        return int(sec*self.network.steps_in_s)

    def _set_event_times(self):
        query_length = self.query_noise_length + self.query_settle_length
        nav1_end_time = (16 + self.pre_seed_loc)/self.velocity
        query_end_time = nav1_end_time + query_length
        nav2_end_time = query_end_time + 8/self.velocity
        event_end_times = [nav1_end_time, query_end_time, nav2_end_time]
        return event_end_times

    def _get_navigation_input(self, t):
        nav_scale = 1.
        hop_length = self.to_frames(1./self.velocity)
        loc_t = (self.to_seconds(t)*self.velocity) % (2*pi)
        velocity_modulation = self._get_velocity_modulation(hop_length, t)
        place_input = self._get_sharp_cos(loc_t)*velocity_modulation*nav_scale
        input_t = np.zeros(self.network.num_units)
        input_t[self.network.J_place_indices] = place_input
        input_t[input_t < 0] = 0
        input_t[self.network.J_episode_indices] += np.random.normal(
            0, 0.5, self.network.N_ep
            )
        input_t[input_t < 0] = 0
        alpha_t = 1.
        return input_t, alpha_t

    def _get_query_input(self, t):
        input_t = np.zeros(self.network.num_units)
        if t < self.to_frames(self.query_noise_length):
            input_t[self.network.J_episode_indices] = np.random.normal(
                0, 0.5, self.network.N_ep
                ) + self.K_ep
            input_t[input_t < 0] = 0
        else:
            input_t[self.network.J_episode_indices] = self.K_ep/2
        alpha_t = 1.
        return input_t, alpha_t

    def _get_velocity_modulation(self,  hop_length, t):
        scale = 6.
        center = scale/2.
        x = ((t % hop_length)/hop_length)*scale - center
        return np.exp(-(x**2)/2)

