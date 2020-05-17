import numpy as np
from sklearn.preprocessing import normalize
from math import pi

class RingAttractorRNN(object):
    """ Ring attractor network parameterized by von Mises functions """

    base_J0 = 1.5
    base_J2 = 3.
    kappa = 5
    norm_scale = 8

    def __init__(self, num_units):
        self.num_units = num_units
        self.J0 = self.base_J0/num_units
        self.J2 = self.base_J2/num_units
        self._init_J()
        self.J = normalize(self.J, axis=1, norm="l1")*self.norm_scale

    def _init_J(self):
        """ Initializes the connectivity matrix J """

        J = np.zeros((self.num_units, self.num_units))
        for i in range(self.num_units):
            weights = self._get_vonmises(i)
            J[i] = weights
        self.J = J

    def _get_vonmises_weight(self, i ,center):
        curve = self._get_vonmises(center=center)
        return curve[i]

    def _get_vonmises(self, center):
        """ Returns a sharp sinusoidal curve that drops off rapidly """

        mu = 0
        x = np.linspace(-pi, pi, self.num_units, endpoint=False)
        curve = np.exp(self.kappa*np.cos(x-mu))/(2*pi*np.i0(self.kappa))
        curve[self.num_units//2] = 0
        curve = np.roll(curve, center - self.num_units//2)
        curve = -self.J0 + self.J2*curve
        #curve -= np.sum(curve)/curve.size
        return curve 
