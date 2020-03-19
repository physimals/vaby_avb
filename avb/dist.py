"""
Distributions that can be applied to a model parameter
"""
import math

import numpy as np

from .utils import LogBase

def get_dist(prefix, **kwargs):
    """
    Factory method to return a distribution from options
    """
    dist = kwargs.get("%s_dist" % prefix, kwargs.get("dist", "Normal"))
    mean = kwargs.get("%s_mean" % prefix, kwargs.get("mean", 0.0))
    var = kwargs.get("%s_var" % prefix, kwargs.get("var", 1.0))

    dist_class = globals().get(dist, None)
    if dist_class is None:
        raise ValueError("Unrecognized distribution: %s" % dist)
    else:
        return dist_class(mean, var)

class Identity:
    """
    Base class for variable transformations which defines just a
    simple identity transformation
    """

    def int_values(self, ext_values, ns=np):
        """
        Convert external (model-visible) values to internal (inferred)
        values

        :param ext_values: Model-visible values
        :ns: Namespace containing maths functions (defaults to TensorFlow, 
             alternatives would be numpy or math)
        :return: Object compatible with ext_values containing internal values
        """
        return ext_values

    def int_moments(self, ext_mean, ext_var, ns=np):
        """
        Convert internal (inferred) mean/variance to external
        (model-visible) mean/variance

        :param ext_mean: Model-visible mean
        :param ext_var: Model-visible variance
        :ns: Namespace containing maths functions (defaults to TensorFlow, 
             alternatives would be numpy or math)
        :return: Tuple of objects compatible with ext_mean, ext_var containing internal values
        """
        return ext_mean, ext_var

    def ext_values(self, int_values, ns=np):
        """
        Convert external (model) values to internal
        (inferred) values

        :param int_values: Internal values
        :ns: Namespace containing maths functions (defaults to TensorFlow, 
             alternatives would be numpy or math)
        :return: Object compatible with int_values containing external (model) values
        """
        return int_values

    def ext_moments(self, int_mean, int_var, ns=np):
        """
        Convert the external (model) mean/variance to internal
        (inferred) mean/variance

        :param ext_mean: Internal (inferred) mean
        :param ext_var: Internal (inferred) variance
        :ns: Namespace containing maths functions (defaults to TensorFlow, 
             alternatives would be numpy or math)
        :return: Tuple of objects compatible with int_mean, int_var containing external 
                 (model) values
        """
        return int_mean, int_var

class Log(Identity):
    """
    Log-transform used for log-normal distribution
    """
    def __init__(self, geom=True):
        self._geom = geom

    def int_values(self, ext_values, ns=np):
        return ns.log(ext_values)

    def int_moments(self, ext_mean, ext_var, ns=np):
        if self._geom:
            return ns.log(ext_mean), ns.log(ext_var)
        else:
            # See https://uk.mathworks.com/help/stats/lognstat.html
            return ns.log(ext_mean**2/ns.sqrt(ext_var + ext_mean**2)), ns.log(ext_var/ext_mean**2 + 1)

    def ext_values(self, int_values, ns=np):
        return ns.exp(int_values)

    def ext_moments(self, int_mean, int_var, ns=np):
        if self._geom:
            return ns.exp(int_mean), ns.exp(int_var)
        else:
            # FIXME this is wrong...
            return ns.exp(int_mean), ns.exp(int_var)

class Abs(Identity):
    """
    Absolute value transform used for folded normal distribution
    """
    def ext_values(self, int_values, ns=np):
        return ns.abs(int_values)

    def ext_moments(self, int_mean, int_var, ns=np):
        return ns.abs(int_mean), int_var

class Dist(LogBase):
    """
    A parameter distribution
    """
    pass

class Normal(Dist):
    """
    Gaussian-based distribution

    The distribution of a parameter has an *underlying* Gaussian
    distribution but may apply a transformation on top of this
    to form the *model* distribution.

    We force subclasses to implement the required methods rather
    than providing a default implementation
    """
    def __init__(self, ext_mean, ext_var, transform=Identity()):
        """
        Constructor.

        Sets the distribution mean, variance and std.dev.

        Note that these are the mean/variance of the *model*
        distribution, not the underlying Gaussian - the latter are
        returned by the ``int_mean`` and ``int_var`` methods
        """
        Dist.__init__(self)
        self.transform = transform
        self.ext_mean, self.ext_var = ext_mean, ext_var
        self.mean, self.var = self.transform.int_moments(ext_mean, ext_var, ns=math)
        self.sd = math.sqrt(self.var)

    def __str__(self):
        return "Gaussian (%f, %f)" % (self.mean, self.var)

class LogNormal(Normal):
    """
    Log of the parameter is distributed as a Gaussian.

    This is one means of ensuring that a parameter is always > 0.
    """

    def __init__(self, mean, var, geom=True, **kwargs):
        Normal.__init__(self, mean, var, transform=Log(geom), **kwargs)

    def __str__(self):
        return "Log-Normal (%f, %f)" % (self.ext_mean, self.ext_var)

class FoldedNormal(Normal):
    """
    Distribution where the probability density
    is zero for negative values and the sum of Gaussian
    densities for the value and its negative otherwise

    This is a fancy way of saying we take the absolute
    value of the underlying distribution as the model
    distribution value.
    """

    def __init__(self, mean, var, **kwargs):
        Normal.__init__(self, mean, var, transform=Abs(), **kwargs)

    def __str__(self):
        return "Folded Normal (%f, %f)" % (self.ext_mean, self.ext_var)

