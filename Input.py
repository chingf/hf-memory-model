import numpy as np
from math import pi, cos, sin, radians, degrees, atan2

class Input(object):
    """
    An object that generates inputs to feed into a Network object.
    """

    noise_std = 0.2

    def __init__(self):
        pass

    def set_network(self, network):
        self.network = network
        self.f = np.zeros(network.num_units)

    def set_current_activity(self, f):
        self.f = f

    def _get_sharp_cos(
        self, loc, num_units, kappa=2.5, vonmises_gain=2, curve_offset=2
        ):
        """ Returns a sharp sinusoidal curve that drops off rapidly """

        mu = 0
        loc_idx = int(loc/(2*pi)*num_units)
        x = np.linspace(-pi, pi, num_units, endpoint=False)
        curve = np.exp(kappa*np.cos(x-mu))/(2*pi*np.i0(kappa))
        #curve -= np.max(curve)/curve_offset #TODO: was 2
        curve -= np.sum(curve)/curve.size
        curve *= 1.5#vonmises_gain
        curve = np.roll(curve, loc_idx - num_units//2)
        return curve

#import matplotlib.pyplot as plt
#i = Input()
#c = i._get_sharp_cos(loc=pi, num_units=100)
#plt.plot(c);plt.axhline(0, color="gray");plt.show()
#integral = np.sum(c) + 10
#c -= integral/c.size
#plt.plot(c);plt.axhline(0, color="gray");plt.show()
#print(np.sum(c>0))

class NavigationInput(Input):
    """ Feeds in navigation input to the place network. """

    def __init__(self, T=4000):
        self.T = T
        self.t = 0
        self.plasticity = 0.02
        self.plasticity_induction_p = 0.005

    def get_inputs(self):
        if self.t < self.T:
            period = 2000
            loc_t = ((self.t % period)/period)*(2*pi)
            input_t = np.zeros(self.network.num_units)
            input_t = self._get_sharp_cos(loc_t, self.network.num_units)
            input_t[input_t < 0] = 0
            #input_t += np.random.normal(0, self.noise_std, input_t.shape)
            #input_t[input_t < 0] = 0
            self.t += 1
            random_draw = np.random.uniform(size=1)[0]
            if random_draw < self.plasticity_induction_p:
                plasticity = self.plasticity
            else:
                plasticity = 0
            return input_t, plasticity 
        else:
            raise StopIteration

class EpisodeInput(Input):
    """ Feeds in uncorrelated, random input to the place network. """

    def __init__(self, T=80, plasticity=1., noise_mean=0, noise_std=0.75):
        self.T = T
        self.t = 0
        self.plasticity = plasticity
        self.plasticity_induction_p = 0.05
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def set_network(self, network):
        self.network = network
        self.f = np.zeros(network.num_units)
        input_t = np.random.normal(
            self.noise_mean, self.noise_std, self.network.num_units
            )
        input_t[input_t < 0] = 0
        self.input_t = input_t

    def get_inputs(self):
        if self.t < self.T:
            self.t += 1
            random_draw = np.random.uniform(size=1)[0]
            if self.t == 70:
#            if self.t > self.T//2 and random_draw < self.plasticity_induction_p:
                plasticity = self.plasticity
            else:
                plasticity = 0
            return self.input_t, plasticity 
        else:
            raise StopIteration

class TestFPInput(Input):
    """ Feeds in uncorrelated, random input to the place network. """

    def __init__(
        self, T=150, plasticity=1., noise_mean=0, noise_std=0.75,
        use_memory=False, memory_noise_std=0.2, recall_scale=2
        ):
        self.T = T
        self.t = 0
        self.plasticity = plasticity
        self.plasticity_induction_p = 0.05
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.use_memory = use_memory
        self.memory_noise_std = memory_noise_std
        self.recall_scale = recall_scale

    def set_network(self, network):
        self.network = network
        self.f = np.zeros(network.num_units)
        self.K_inhib = network.K_inhib
        input_t = (np.random.uniform(size=self.network.num_units) < 0.3).astype(float)
        input_t *= 0.8
        self.input_t = input_t

    def get_inputs(self):
        if self.t < self.T:
            self.t += 1
            inhib = True
            if self.use_memory is not False:
                if self.t < self.T/3:
                    input_t = self.use_memory.copy()
                else:
                    input_t = self.use_memory.copy() 
                    if self.t % 4 == 0:
                        input_t[self.network.J_ep_indices] = (
                            np.random.uniform(size=self.network.N_ep) < 0.15
                            ).astype(float) * 0.5
                    inhib = False
                input_t[input_t < 0] = 0
            else:
                input_t = np.zeros(self.network.num_units)
                if self.t < self.T/5:
                    input_t = self.input_t.copy()
                    input_t[self.network.J_pl_indices] = 0
                inhib = False
            plasticity = ext_plasticity = 0
            return input_t, plasticity, ext_plasticity, inhib
        else:
            raise StopIteration

class TestNavFPInput(Input):
    """ Feeds in uncorrelated, random input to the place network. """

    def __init__(self, recall_loc, network):
        super().__init__()
        self.recall_loc = (recall_loc/network.N_pl)*2*pi
        self.t = 0
        self._set_task_params()
        np.random.seed()
        self.input_t = (np.random.uniform(size=network.num_units) < 0.15).astype(float)
        np.random.seed(0)

    def _set_task_params(self):
        self.recall_length = 600
        self.nav_speed = 1/2000. # revolution/timesteps
        self.recall = False
        self.recall_start = int((1/self.nav_speed)*self.recall_loc/(2*pi))
        self.recall_start += int(1/self.nav_speed)
        self.T = int(self.recall_start + self.recall_length*2 + 4000)

    def get_inputs(self):
        if self.t < self.T:
            if self.recall:
                input_t, plasticity, ext_plasticity, inhib = self._get_recall_inputs()
            else:
                if self.t == self.recall_start:
                    self.recall = True
                input_t, plasticity, ext_plasticity, inhib = self._get_nav_inputs()
            self.t += 1
            return input_t, plasticity, ext_plasticity, inhib
        else:
            raise StopIteration

    def _get_recall_inputs(self):
        input_t = np.zeros(self.network.num_units)
#        if self.t % 4 == 0:
#            np.random.seed()
#            input_t[self.network.J_ep_indices] = (
#                np.random.uniform(size=self.network.N_ep) < 0.3
#                ).astype(float) * 0.5
#            np.random.seed(0)
        inhib = False
        plasticity = ext_plasticity = 0
        recall_end = self.recall_start + self.recall_length
        if self.t < self.recall_start + 100:
            input_t[self.network.J_pl_indices] += self._get_sharp_cos(
                self.recall_loc, self.network.N_pl,
                )*0.2
        if self.t < self.recall_start + 200:
            input_t[self.network.J_ep_indices] = self.input_t[
                self.network.J_ep_indices
                ]*0.3
        if self.t > recall_end - self.recall_length/5:
            inhib = 0.7
        if self.t == recall_end - 1:
            self.recall = False
        input_t[input_t < 0] = 0
        return input_t, plasticity, ext_plasticity, inhib

    def _get_nav_inputs(self):
        t = self.t
        if t > self.recall_start:
            t -= self.recall_length
        loc_t = (t*self.nav_speed) * 2*pi
        input_t = np.zeros(self.network.num_units)
        input_t[self.network.J_pl_indices] += self._get_sharp_cos(
            loc_t, self.network.N_pl
            )
        input_t[input_t < 0] = 0
        return input_t, 0, 0, True

class AssocInput(Input):
    """ Feeds in uncorrelated, random input to the place network. """

    def __init__(
        self, noise_mean=0, noise_std=0.3, cache_loc=pi,
        ext_plasticity=1., plasticity=1.
        ):

        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.cache_loc = cache_loc
        self.plasticity = plasticity
        self.ext_plasticity = ext_plasticity
        self.plasticity_induction_p = 0.05
        self.t = 0
        self._set_task_params()

    def _set_task_params(self):
        self.cache_length = 30
        self.nav_speed = 1/2000. # revolution/timesteps
        self.prev_loc = 0
        self.caching = False
        self.cache_start = int((1/self.nav_speed)*self.cache_loc/(2*pi))
        self.cache_start += 1/self.nav_speed
        self.T = int(self.cache_start + self.cache_length*2 + 4000)

    def set_network(self, network):
        self.network = network
        self.f = np.zeros(network.num_units)
        self.K_inhib = network.K_inhib
        input_t = (np.random.uniform(size=self.network.num_units) < 0.25).astype(float)
        input_t *= 0.8
        self.input_t = input_t

    def get_inputs(self):
        if self.t < self.T:
            if self.caching:
                input_t, plasticity, ext_plasticity, inhib = self._get_cache_inputs()
            else:
                if self.t == self.cache_start:
                    self.caching = True
                input_t, plasticity, ext_plasticity, inhib = self._get_nav_inputs()
            self.t += 1
            return input_t, plasticity, ext_plasticity, inhib
        else:
            raise StopIteration

    def _get_cache_inputs(self):
        t = (self.t - self.cache_start) % self.cache_length
        if t == self.cache_length - 1:
            self.caching = False
            plasticity = self.plasticity
            ext_plasticity = self.ext_plasticity
        else:
            plasticity = ext_plasticity = 0
        input_t = self.input_t.copy()
        input_t[self.network.J_pl_indices] = 0
        return input_t, plasticity, ext_plasticity, True

    def _get_nav_inputs(self):
        t = self.t
        if t > self.cache_start:
            t -= self.cache_length
        loc_t = (t*self.nav_speed) * 2*pi
        input_t = np.zeros(self.network.num_units)
        input_t[self.network.J_pl_indices] += self._get_sharp_cos(
            loc_t, self.network.N_pl
            )
        input_t[input_t < 0] = 0
        #input_t += np.random.normal(0, self.noise_std, input_t.shape)
        #input_t[input_t < 0] = 0
        return input_t, 0, 0, True
