"""
Multi-exponential models
"""
import numpy as np

from avb import __version__
from avb.model import Model
from avb.parameter import get_parameter

class MultiExpModel(Model):
    """
    Exponential decay with multiple independent decay rates and amplitudes
    """

    def __init__(self, data_model, **options):
        Model.__init__(self, data_model, **options)
        self._num_exps = options.get("num_exps", 1)
        for idx in range(self._num_exps):
            self.params += [
                get_parameter("amp%i" % (idx+1), 
                              dist="LogNormal", mean=1.0, 
                              prior_var=1e6, post_var=1.5, 
                              post_init=self._init_amp,
                              **options),
                get_parameter("r%i" % (idx+1), 
                              dist="LogNormal", mean=1.0, 
                              prior_var=1e6, post_var=1.5,
                              **options),
            ]


    def _init_amp(self, _param, _t, data):
        return np.reduce_max(data, axis=1) / self._num_exps, None

    def evaluate(self, params, tpts):
        print("ev", np.array(params).shape, tpts.shape)
        ret = None
        for idx in range(self._num_exps):
            amp = params[2*idx]
            r = params[2*idx+1]
            contrib = amp * np.exp(-r * tpts)
            if ret is None:
                ret = contrib
            else:
                ret += contrib
        return ret

    def __str__(self):
        return "Multi exponential model with %i exponentials: %s" % (self._num_exps, __version__)

class ExpModel(MultiExpModel):
    """
    Simple exponential decay model
    """
    def __init__(self, data_model, **options):
        MultiExpModel.__init__(self, data_model, num_exps=1, **options)

    def __str__(self):
        return "Exponential model: %s" % __version__

class BiExpModel(MultiExpModel):
    """
    Exponential decay with two independent decay rates and amplitudes
    """
    @staticmethod
    def add_options(parser):
        group = parser.add_argument_group("Biexponential model options")
        group.add_argument("--dt", help="Time separation between volumes", type=float, default=1.0)

    def __init__(self, data_model, **options):
        MultiExpModel.__init__(self, data_model, num_exps=2, **options)

    def __str__(self):
        return "Bi-Exponential model: %s" % __version__
