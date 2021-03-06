import numpy as np
from math import pi, cos, sin, radians, degrees, atan2

class Input(object):
    """
    An object that generates inputs to feed into a Network object.
    """

    J_samplerate = -1
    nav_scale = 1.2
    cache_scale = 1.2 #2 TODO
    cache_noise_std = 0.4
    noise_std = 0.02

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

    J_samplerate = 100

    def __init__(self, T=1800):
        self.T = T
        self.t = 0

    def get_inputs(self):
        if self.t < self.T:
            period = 2000
            loc_t = ((self.t % period)/period)*(2*pi)
            input_t = np.zeros(self.network.num_units)
            input_t[self.network.J_place_indices] = self._get_sharp_cos(loc_t)
            input_t[input_t < 0] = 0
            input_t[self.network.J_place_indices] *= self.nav_scale
            input_t += np.random.normal(0, self.noise_std, input_t.shape)
            alpha_t = 1.
            input_t[input_t < 0] = 0
            self.inputs[self.t,:] = input_t
            self.alphas[self.t] = alpha_t
            self.t += 1
            return input_t, alpha_t, 1
        else:
            raise StopIteration

class OneCacheInput(Input):
    """ Feeds in random noise into episode network and navigation input. """

    J_samplerate = 1

    def __init__(self, K_ep):
        self.t = 0
        self.cache_length = 100 #40 TODO
        self.fastlearn_length = 5 #6 TODO
        self.nav_speed = 1/2000. # revolution/timesteps
        self.cache_loc = pi/2
        self.prev_loc = 0
        self.caching = False
        self.K_ep = K_ep
        self.cache_start = int((1/self.nav_speed)*self.cache_loc/(2*pi))
        self.cache_start += int(1/self.nav_speed)
        self.T = int(2/self.nav_speed + self.cache_length)
        self.cache_noise_std = 0.3

    def get_inputs(self):
        t = self.t
        if t < self.T:
            if self.caching:
                input_t, alpha_t, fastlearn = self._get_cache_inputs()
            else: # Navigating
                if t == self.cache_start:
                    self.caching = True
                elif t > self.cache_start:
                    t -= self.cache_length
                loc_t = (t*self.nav_speed) * 2*pi
                input_t = np.zeros(self.network.num_units)
                input_t[self.network.J_place_indices] += self._get_sharp_cos(loc_t)
                input_t[self.network.J_place_indices] *= self.nav_scale
                input_t[input_t < 0] = 0
                input_t += np.random.normal(0, self.noise_std, input_t.shape)
                input_t[input_t < 0] = 0
                alpha_t = 1. 
                fastlearn = False
        else:
            raise StopIteration
        self.inputs[self.t,:] = input_t
        self.alphas[self.t] = alpha_t
        self.t += 1
        return input_t, alpha_t, fastlearn

    def _get_cache_inputs(self):
        t = (self.t - self.cache_start) % self.cache_length
        fastlearn = False
        input_t = np.zeros(self.network.num_units)
        input_t[self.network.J_episode_indices] = np.random.normal(
            0, self.cache_noise_std, self.network.N_ep
            ) + self.K_ep
        input_t[self.network.J_place_indices] += self._get_sharp_cos(self.cache_loc)
        input_t *= self.cache_scale
        input_t += np.random.normal(0, self.noise_std, input_t.shape)
        if t > self.cache_length - self.fastlearn_length: # Fast learn
            fastlearn = True
        input_t[input_t < 0] = 0
        if t == self.cache_length - 1:
            self.caching = False
        alpha_t = 1
        return input_t, alpha_t, fastlearn

class MultiCacheInput(Input):
    """ Feeds in random noise into episode network and navigation input. """

    def __init__(self, K_ep):
        self.t = 0
        self.cache_length = 100
        self.fastlearn_length = 5
        self.navigation_length = self.cache_length*20
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
                input_t[self.network.J_place_indices] *= self.nav_scale
                input_t[self.network.J_episode_indices] += np.random.normal(
                    0, self.cache_noise_std, self.network.N_ep
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
        input_t[self.network.J_episode_indices] = np.random.normal(
            0, self.cache_noise_std, self.network.N_ep
            ) + self.K_ep
        input_t[self.network.J_place_indices] += self._get_sharp_cos(cache_loc)
        if t > self.cache_length - self.fastlearn_length: # Fast learn
            fastlearn = True
        input_t[input_t < 0] = 0
        alpha_t = self.cache_scale
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
        self.query_noise_length = 1.5 # In sec
        self.query_settle_length = 0. # In sec
        self.velocity = 2 # Perches/sec
        self.event_end_times = self._set_event_times()
        self.T_sec = self.event_end_times[-1]

    def get_inputs(self):
        event_end_times = [self.to_frames(e) for e in self.event_end_times]
        T1, T2, T3 = event_end_times
        t = self.t
        if t < T1:
            input_t, alpha_t = self._get_navigation_input(t)
#            if T1 - t < 150:
#                input_t[self.network.J_episode_indices] += np.random.normal(
#                    0, 0.5, self.network.N_ep
#                    ) + self.network.K_ep
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
        hop_length = self.to_frames(1./self.velocity)
        loc_t = 2*pi*((self.to_seconds(t)*self.velocity) % (16))/16
        velocity_modulation = self._get_velocity_modulation(hop_length, t)
        place_input = self._get_sharp_cos(loc_t)*velocity_modulation*self.nav_scale
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
            input_t[self.network.J_episode_indices] = self.K_ep
        alpha_t = 1.
        return input_t, alpha_t

    def _get_velocity_modulation(self,  hop_length, t):
        scale = 6.
        center = scale/2.
        x = ((t % hop_length)/hop_length)*scale - center
        return np.exp(-(x**2)/2)

class PresentationInput(Input):
    """
    Creates input that is chickadee-like. Specifically:
        - Movement from 0 around the whole ring then to loc (10 sec)
        - 2s query for a seed
        - Move to seed location (4 sec)
    """

    def __init__(self, pre_seed_locs, K_pl, K_ep):
        self.pre_seed_locs = pre_seed_locs # In radians
        self.t = 0
        self.K_pl = K_pl
        self.K_ep = K_ep
        self.query_length = 1.5 # In sec
        self.velocity = 2*pi/4#8 # rad/sec
        self.event_end_times = self._set_event_times()
        self.T_sec = self.event_end_times[-1]
        self.T_sec = self.event_end_times[2]

    def get_inputs(self):
        event_end_times = [self.to_frames(e) for e in self.event_end_times]
        T1, T2, T3, T4, T5 = event_end_times
        t = self.t
        if t < T1:
            input_t, alpha_t = self._get_navigation_input(t)
        elif t < T2:
            t -= T1
            input_t, alpha_t = self._get_query_input(t)
        elif t < T3:
            t -= (T2 - T1)
            input_t, alpha_t = self._get_navigation_input(t)
        elif t < T4:
            raise StopIteration
            t -= T3
            input_t, alpha_t = self._get_query_input(t)
        elif t < T5:
            t -= (T2 - T1)
            t -= (T4 - T3)
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
        seed1, seed2 = self.pre_seed_locs
        nav1_dist = 4*2*pi + seed1
        nav1_end_time = nav1_dist/self.velocity
        query1_end_time = nav1_end_time + self.query_length
        nav2_dist = (2*pi - seed1) + seed2 + (4*2*pi)
        nav2_end_time = query1_end_time + nav2_dist/self.velocity
        query2_end_time = nav2_end_time + self.query_length
        nav3_dist = 2*pi
        nav3_end_time = query2_end_time + nav3_dist/self.velocity
        event_end_times = [
            nav1_end_time, query1_end_time, nav2_end_time,
            query2_end_time, nav3_end_time
            ]
        return event_end_times

    def _get_navigation_input(self, t):
        loc_t = (self.to_seconds(t)*self.velocity) % (2*pi)
        place_input = self._get_sharp_cos(loc_t)*self.nav_scale
        input_t = np.zeros(self.network.num_units)
        input_t[self.network.J_place_indices] = place_input
        input_t[input_t < 0] = 0
        input_t += np.random.normal(0, self.noise_std, input_t.shape)
        input_t[input_t < 0] = 0
        alpha_t = 1.
        return input_t, alpha_t

    def _get_query_input(self, t):
        input_t = np.zeros(self.network.num_units)
        input_t[self.network.J_episode_indices] = np.random.normal(
            0, self.cache_noise_std, self.network.N_ep
            ) + self.K_ep
        input_t[self.network.J_episode_indices] *= self.cache_scale
        input_t += np.random.normal(0, self.noise_std, input_t.shape)
        input_t[input_t < 0] = 0
        alpha_t = 1
        return input_t, alpha_t
