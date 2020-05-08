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

    def _get_sharp_cos(self, loc=pi):
        """ Returns a sharp sinusoidal curve that drops off rapidly """

        mu = 0
        kappa = 3
        vonmises_gain = 3.2
        loc_idx = int(loc/(2*pi)*self.network.num_units)
        x = np.linspace(-pi, pi, self.network.num_units, endpoint=False)
        curve = np.exp(kappa*np.cos(x-mu))/(2*pi*np.i0(kappa))
        curve -= np.max(curve)/2.
        curve *= vonmises_gain
        curve = np.roll(curve, loc_idx - self.network.num_units//2)
        return curve

class NavigationInput(Input):
    """ Feeds in navigation input to the place network. """

    def __init__(self, T=4000):
        self.T = T
        self.t = 0
        self.btsp = 0.02
        self.btsp_induction_p = 0.005

    def get_inputs(self):
        if self.t < self.T:
            period = 2000
            loc_t = ((self.t % period)/period)*(2*pi)
            input_t = np.zeros(self.network.num_units)
            input_t = self._get_sharp_cos(loc_t)
            input_t[input_t < 0] = 0
            #input_t += np.random.normal(0, self.noise_std, input_t.shape)
            #input_t[input_t < 0] = 0
            self.t += 1
            random_draw = np.random.uniform(size=1)[0]
            if random_draw < self.btsp_induction_p:
                btsp = self.btsp
            else:
                btsp = 0
            return input_t, btsp 
        else:
            raise StopIteration

class EpisodeInput(Input):
    """ Feeds in uncorrelated, random input to the place network. """

    def __init__(self, T=80, btsp=1., noise_mean=0, noise_std=0.75):
        self.T = T
        self.t = 0
        self.btsp = btsp
        self.btsp_induction_p = 0.05
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
#            if self.t > self.T//2 and random_draw < self.btsp_induction_p:
                btsp = self.btsp
            else:
                btsp = 0
            return self.input_t, btsp 
        else:
            raise StopIteration

class TestFPInput(EpisodeInput):
    """ Feeds in uncorrelated, random input to the place network. """

    def __init__(
        self, T=80, btsp=1., noise_mean=0, noise_std=0.75,
        use_memory=0, memory_noise_std=0.2
        ):
        super().__init__(
            T=T, btsp=btsp, noise_mean=noise_mean, noise_std=noise_std
            )
        self.use_memory = use_memory
        self.memory_noise_std = memory_noise_std

    def get_inputs(self):
        if self.t < self.T:
            self.t += 1
            if self.use_memory:
                input_t = self.network.memories[self.use_memory-1].copy()
                input_t += np.random.normal(
                    0, self.memory_noise_std, input_t.shape
                    )
                input_t[input_t < 0] = 0
                input_t *= 4
            else:
                input_t = np.random.normal(
                    self.noise_mean, self.noise_std, self.network.num_units
                    )
                #input_t = self.input_t.copy()
                input_t[input_t < 0] = 0
                input_t *= 3
            btsp = 0
            return input_t, btsp
        else:
            raise StopIteration
