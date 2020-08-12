"""
Base class for a forward model whose parameters are to be fitted
"""
import pkg_resources
import collections

import numpy as np

from .utils import LogBase, ValueList

MODELS = {
}

_models_loaded = False

def get_model_class(model_name):
    """
    Get a model class by name
    """
    global _models_loaded
    if not _models_loaded:
        for model in pkg_resources.iter_entry_points('avb.models'):
            MODELS[model.name] = model.load()
        _models_loaded = True

    model_class = MODELS.get(model_name, None)
    if model_class is None:
        raise ValueError("No such model: %s" % model_name)

    return model_class

class ModelOption:
    def __init__(self, attr_name, desc, **kwargs):
        self.attr_name = attr_name
        self.desc = desc
        self.clargs = kwargs.get("clargs", ["--%s" % attr_name.replace("_", "-")])
        self.default = kwargs.get("default", None)
        self.units = kwargs.get("units", None)
        self.type = kwargs.get("type", str)

class Model(LogBase):
    """
    A forward model

    :attr params: Sequence of ``Parameter`` objects
    :attr nparams: Number of model parameters
    """
    OPTIONS = [
        ModelOption("dt", "Time separation between volumes", type=float, default=1.0),
        ModelOption("t0", "Time offset for first volume", type=float, default=0.0),
    ]

    def __init__(self, data_model, **options):
        LogBase.__init__(self)
        self.data_model = data_model
        self.params = []
        for option in self.OPTIONS:
            setattr(self, option.attr_name, options.get(option.attr_name, option.default))

    @property
    def nparams(self):
        """
        Number of parameters in the model
        """
        return len(self.params)

    def param_idx(self, name):
        """
        :return: the index of a named parameter
        """
        for idx, param in enumerate(self.params):
            if param.name == name:
                return idx
        raise ValueError("Parameter not found in model: %s" % name)

    def inference_to_model(self, inference_params):
        """
        Transform inference engine parameters into model parameters

        :param params: Inference engine parameters in shape [P, V, 1]
        """
        model_params = []
        for idx, p in enumerate(inference_params):
            model_params.append(self.params[idx].post_dist.transform.ext_values(p))
        return model_params

    def model_to_inference(self, model_params):
        """
        Transform inference engine parameters into model parameters

        :param params: Inference engine parameters in shape [P, V, 1]
        """
        inference_params = []
        for idx, p in enumerate(model_params):
            inference_params.append(self.params[idx].post_dist.transform.int_values(p))
        return inference_params

    def tpts(self):
        """
        Get the full set of timeseries time values

        :param n_tpts: Number of time points required for the data to be fitted
        :param shape: Shape of source data which may affect the times assigned

        By default this is a linear space using the attributes ``t0`` and ``dt``.
        Some models may have time values fixed by some other configuration. If
        the number of time points is fixed by the model it must match the
        supplied value ``n_tpts``.

        :return: Either a Numpy array of shape [n_tpts] or a Numpy array of shape
                 shape + [n_tpts] for voxelwise timepoints
        """
        return np.linspace(self.t0, self.t0+self.data_model.n_tpts*self.dt, num=self.data_model.n_tpts, endpoint=False)

    def evaluate(self, params, tpts):
        """
        Evaluate the model

        :param params Sequence of parameter values arrays, one for each parameter.
                      Each array is [W,1] array where W is the number of parameter vertices
                      may be supplied as a [P,W,1] array where P is the number of
                      parameters.
        :param tpts: Time values to evaluate the model at, supplied as an array of shape
                     [1,B] (if time values at each node are identical) or [W,B]
                     otherwise.

        :return: [W, B] tensor containing model output at the specified time values
                 for each node
        """
        raise NotImplementedError("evaluate")

    def jacobian(self, params, tpts):
        """
        Numerical differentiation to calculate Jacobian matrix
        of partial derivatives of model prediction with respect to
        parameters

        :param params Sequence of parameter values arrays, one for each parameter.
                      Each array is [W,1] array where W is the number of parameter vertices
                      may be supplied as a [P,W,1] array where P is the number of
                      parameters.
        :param tpts: Time values to evaluate the model at, supplied as an array of shape
                     [1,B] (if time values at each node are identical) or [W,B]
                     otherwise.

        :return: Jacobian matrix [W, B, P]
        """
        nt = tpts.shape[1]
        nparams = len(params)
        nv = params[0].shape[0]
        J = np.zeros([nv, nt, nparams], dtype=np.float32)
        #print("centre\n", params, params.shape)
        for param_idx, param_value in enumerate(params):
            plus = params.copy()
            minus = params.copy()
            delta = param_value * 1e-5
            delta[delta < 0] = -delta[delta < 0]
            delta[delta < 1e-10] = 1e-10

            plus[param_idx] += delta
            minus[param_idx] -= delta
            #print("param idx %i, delta %s" % (param_idx, delta))
            #print(plus)
            #print(minus)
            
            diff = self.evaluate(plus, tpts) - self.evaluate(minus, tpts)
            J[..., param_idx] = diff / (2*delta)
        return J

    def log_config(self, log=None):
        """
        Write model configuration to a log stream
        
        :param: log Optional logger to use - defaults to class instance logger
        """
        if log is None:
            log = self.log
        log.info("Model: %s", str(self))
        for option in self.OPTIONS:
            log.info(" - %s: %s", option.desc, str(getattr(self, option.attr_name)))
