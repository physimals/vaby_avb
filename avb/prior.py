"""
AVB - Priors for model and noise parameters
"""

import numpy as np
import tensorflow as tf

from svb.utils import LogBase

class NormalPrior(LogBase):
    """
    Fixed prior for a single model parameter
    
    :attr mean: Mean value [W] or [1]
    :attr variance: Variance [W] or [1]
    """
    
    def __init__(self, mean, variance):
        """
        :param mean: Prior mean value as float or Numpy array [W]
        :param variance: Prior mean value as float or Numpy array [W]
        """
        LogBase.__init__(self)

        # Mean/variance is usually specified globally so
        # make sure it has a nodewise dimension
        while np.array(mean).ndim < 1:
            mean = np.array(mean)[np.newaxis, ...]
        while np.array(variance).ndim < 1:
            variance = np.array(variance)[np.newaxis, ...]

        self.mean = tf.constant(mean, dtype=np.float32)
        self.variance = tf.constant(variance, dtype=np.float32)

class NoisePrior:
    """
    Prior for the noise distribution

    :attr noise_s: Noise gamma distribution prior scale parameter
    :attr noise_c: Noise gamma distribution prior shape parameter
    """
    def __init__(self, shape, scale):
        """
        :param shape Prior noise shape parameter [(W)]
        :param scale Prior noise scale parameter [(W)]
        """
        self.shape = shape
        self.scale = scale

class Prior(LogBase):
    """
    Combined prior for all parameters and noise

    :attr means: Parameter prior means [W, P] or [1, P]
    :attr variances: Parameter prior variances [W, P] or [1, P]
    :attr covar: Prior covariance matrix (diagonal) [W, P, P] or [1, P, P]
    :attr precs: Prior precision matrix (diagonal) [W, P, P] or [1, P, P]
    :attr noise_s: Noise gamma distribution prior scale parameter
    :attr noise_c: Noise gamma distribution prior shape parameter
    """
    def __init__(self, param_priors, noise_prior):
        """
        :param param_priors: Sequence of parameter priors
        :param noise_prior: Noise prior 
        """
        LogBase.__init__(self)

        self.means = tf.stack([p.mean for p in param_priors], axis=1)
        self.variances = tf.stack([p.variance for p in param_priors], axis=1)
        self.covar = tf.linalg.diag(self.variances)
        self.precs = tf.linalg.inv(self.covar)
        self.noise_s = tf.constant(noise_prior.scale, dtype=tf.float32)
        self.noise_c = tf.constant(noise_prior.shape, dtype=tf.float32)
