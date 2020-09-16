"""
Implementation of analytic Variational Bayes to infer a 
nonlinear forward model

This implements section 4 of the FMRIB Variational Bayes tutorial 1.
The code is intended to be compatible with Fabber, however numerical
differences are expected as the forward model is not identical.
"""
import math

import numpy as np
import scipy.special
import tensorflow as tf

from svb.utils import LogBase

from .prior import Prior, NormalPrior, NoisePrior
from .posterior import MVNPosterior

class Avb(LogBase):

    def __init__(self, tpts, data_model, fwd_model, **kwargs):
        """
        :param data: Data timeseries [(W), B]
        :param model: Forward model
        """
        LogBase.__init__(self)
        while tpts.ndim < 2:
            tpts = tpts[np.newaxis, ...]
        self._tpts = np.array(tpts, dtype=np.float32)
        self._data_model = data_model
        self._data = data_model.data_flattened
        self._nv, self._nt = tuple(self._data.shape)
        self.model = fwd_model
        self._debug = kwargs.get("debug", False)
        self._maxits = kwargs.get("max_iterations", 10)

        model_priors = [NormalPrior(p.prior_dist.mean, p.prior_dist.var) for p in self.model.params]
        noise_prior = NoisePrior(scale=kwargs.get("noise_s0", 1e6), shape=kwargs.get("noise_c0", 1e-6))
        self.prior = Prior(model_priors, noise_prior)
        self.post = MVNPosterior(self._data_model, self.model.params, self._tpts)

        self.sess = tf.Session()
        self.post.sess = self.sess # FIXME
        init = tf.initialize_all_variables()
        self.sess.run(init)
        
    def noise_mean_prec(self, dist):
        return dist.noise_c*dist.noise_s, 1/(dist.noise_s*dist.noise_s*dist.noise_c)

    def _debug_output(self, text, J=None):
        if self._debug:
            self.log.debug(text)
            self.log.debug("Prior mean\n", self.prior.means)
            self.log.debug("Prior precs\n", self.prior.precs)
            self.log.debug("Post mean\n", self.post.means)
            self.log.debug("Post precs\n", self.post.precs)
            noise_mean, noise_prec = self.noise_mean_prec(self.prior)
            self.log.debug("Noise prior mean\n", noise_mean)
            self.log.debug("Noise prior prec\n", noise_prec)
            noise_mean, noise_prec = self.noise_mean_prec(self.post)
            self.log.debug("Noise post mean\n", noise_mean)
            self.log.debug("Noise post prec\n", noise_prec)
            if J is not None:
                self.log.debug("Jacobian\n", J)

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
        nparams = tf.shape(params)[0]
        nv = tf.shape(params)[1]
        J = []
        for param_idx in range(len(self.model.params)):
            param_value = params[param_idx]
            delta = param_value * 1e-5
            delta = tf.math.abs(delta)
            delta = tf.where(delta < 1e-5, tf.fill(tf.shape(delta), 1e-5), delta)

            pick_param = tf.equal(tf.range(len(self.model.params)), param_idx)
            plus = tf.where(pick_param, params + delta, params)
            minus = tf.where(pick_param, params - delta, params)

            plus = self._inference_to_model(plus)
            minus = self._inference_to_model(minus)
            diff = self.model.evaluate(plus, tpts) - self.model.evaluate(minus, tpts)
            J.append(diff / (2*delta))
        return tf.stack(J, axis=-1)

    def _init_run(self):
        self.model_means = None
        self.model_vars = None
        self.noise_means = None
        self.noise_vars = None
        self.modelfit = None
        self.free_energy = -1e99
        self.history = {}

    def _evaluate(self, params, tpts):
        # Make means shape [P, V, 1] to enable the parameters
        # to be unpacked as a sequence
        self.means_trans = tf.transpose(params, (1, 0))
        self.model_means = self._inference_to_model(self.means_trans)
        self.model_vars = tf.stack([self.post.covar[:, v, v] for v in range(len(self.model.params))]) # fixme transform
        modelfit = self.model.evaluate(tf.expand_dims(self.model_means, axis=-1), tpts)
        J = self.jacobian(tf.expand_dims(self.means_trans, axis=-1), self._tpts)
        return modelfit, J

    def _inference_to_model(self, inference_params, idx=None):
        """
        Transform inference engine parameters into model parameters

        :param params: Inference engine parameters in shape [P, V]
        """
        if idx is None:
            model_params = []
            for idx in range(len(self.model.params)):
                model_params.append(self.model.params[idx].post_dist.transform.ext_values(inference_params[idx, :], ns=tf))
            return tf.stack(model_params)
        else:
            return self.model.params[idx].post_dist.transform.ext_values(inference_params, ns=tf)

    def _model_to_inference(self, model_params, idx=None):
        """
        Transform inference engine parameters into model parameters

        :param params: Inference engine parameters in shape [P, V]
        """
        if idx is None:
            inference_params = []
            for idx in range(len(self.model.params)):
                inference_params.append(self.model.params[idx].post_dist.transform.int_values(model_params[idx, :], ns=np))
            return tf.stack(inference_params)
        else:
            return self.model.params[idx].post_dist.transform.int_values(model_params, ns=tf)

    def run(self, history=False):
        self._init_run()
        # Get model prediction and Jacobian from current parameters 
        self.modelfit, self.J = self._evaluate(self.post.means, self._tpts)
        self.noise_means, self.noise_precs = self.noise_mean_prec(self.post)
        self.noise_vars = 1.0/self.noise_precs
        self.k = self._data - self.modelfit
        
        # Here we need to update spatial/ARD priors and capture
        # their contribution to the free energy
        
        # Calculate the free energy and update model parameters
        # using the AVB update equations
        self.free_energy = self.post.free_energy(self.k, self.J, self.prior)
        self.new_means, self.new_cov = self.post.update_model_params(self.k, self.J, self.prior)

        # Follow Fabber in updating residuals after params change before updating 
        # noise. Note we don't update J and nor does Fabber (until the
        # end of the parameter updates when it re-centres the linearized model)
        self.k_new = self.k + tf.einsum("ijk,ik->ij", self.J, self.post.means - self.new_means)
        self.noise_c_new, self.noise_s_new = self.post.update_noise(self.k_new, self.J, self.prior)
        
        self._log_iter(0, history)
        self._debug_output("Start", self.J)
        # Update model and noise parameters iteratively
        for idx in range(self._maxits):
            self.update_theta = [
                self.post.means.assign(self.sess.run(self.new_means)),
                self.post.covar.assign(self.sess.run(self.new_cov))
            ]
            self.sess.run(self.update_theta)
            self._debug_output("Updated theta", self.J) 
            self.update_noise = [
                self.post.noise_c.assign(self.sess.run(self.noise_c_new)),
                self.post.noise_s.assign(self.sess.run(self.noise_s_new))
            ]

            self.sess.run(self.update_noise)
            self._debug_output("Updated noise", self.J)
            self._log_iter(idx+1, history)

        if history:
            for item, history in self.history.items():
                # Reshape history items so history is in last axis not first
                trans_axes = list(range(1, history[0].ndim+1)) + [0,]
                self.history[item] = self.sess.run(history).transpose(trans_axes)

    def _log_iter(self, iter, history):
        fmt = {"iter" : iter}
        for attr in ("model_means", "model_vars", "noise_means", "noise_vars", "free_energy"):
            voxelwise_data = getattr(self, attr)
            mean_data = np.mean(self.sess.run(voxelwise_data), axis=-1)
            fmt[attr] = mean_data
            if history:
                if attr not in self.history: self.history[attr] = []
                self.history[attr].append(voxelwise_data)
        self.log.info("Iteration %(iter)i: params=%(model_means)s, vars=%(model_vars)s, noise mean=%(noise_means)e, var=%(noise_vars)e, F=%(free_energy)e" % fmt)
