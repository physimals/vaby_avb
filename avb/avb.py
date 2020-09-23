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
        self.tpts = np.array(tpts, dtype=np.float32)
        self.data_model = data_model
        self.model = fwd_model

        self.data = self.data_model.data_flattened
        self.nv, self.nt = tuple(self.data.shape)
        self.nparam = len(self.model.params)
        self._debug = kwargs.get("debug", False)
        self._maxits = kwargs.get("max_iterations", 10)

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
        J = []
        for param_idx in range(self.nparam):
            param_value = params[param_idx]
            delta = param_value * 1e-5
            delta = tf.math.abs(delta)
            delta = tf.where(delta < 1e-5, tf.fill(tf.shape(delta), 1e-5), delta)

            pick_param = tf.equal(tf.range(self.nparam), param_idx)
            plus = tf.where(pick_param, params + delta, params)
            minus = tf.where(pick_param, params - delta, params)

            plus = self._inference_to_model(plus)
            minus = self._inference_to_model(minus)
            diff = self.model.evaluate(plus, tpts) - self.model.evaluate(minus, tpts)
            J.append(diff / (2*delta))
        return tf.stack(J, axis=-1)

    def _evaluate(self):
        # Make means shape [P, V] to enable the parameters
        # to be unpacked as a sequence
        self.means_trans = tf.transpose(self.post.means, (1, 0))
        self.model_means = self._inference_to_model(self.means_trans)
        self.model_vars = tf.stack([self.post.covar[:, v, v] for v in range(len(self.model.params))]) # fixme transform
        modelfit = self.model.evaluate(tf.expand_dims(self.model_means, axis=-1), self.tpts)
        J = self.jacobian(tf.expand_dims(self.means_trans, axis=-1), self.tpts)
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

    def _update_model_params(self):
        """
        Update model parameters

        From section 4.2 of the FMRIB Variational Bayes Tutorial I

        :return Tuple of (New means, New precisions)
        """
        precs_new = tf.expand_dims(tf.expand_dims(self.post.noise_s, -1), -1) * tf.expand_dims(tf.expand_dims(self.post.noise_c, -1), -1) * self.JtJ + self.prior.precs
        covar_new = tf.linalg.inv(precs_new)

        t1 = tf.einsum("ijk,ik->ij", self.J, self.post.means)
        t15 = tf.einsum("ijk,ik->ij", self.Jt, (self.k + t1))
        t2 = tf.expand_dims(self.post.noise_s, -1) * tf.expand_dims(self.post.noise_c, -1) * t15
        t3 = tf.einsum("ijk,ik->ij", self.prior.precs, self.prior.means)
        means_new = tf.einsum("ijk,ik->ij", covar_new, (t2 + t3))

        return means_new, covar_new
 
    def _update_noise(self):
        """
        Update noise parameters

        From section 4.2 of the FMRIB Variational Bayes Tutorial I

        :return: Tuple of (New scale param, New shape param)
        """
        c_new = tf.fill((self.nv,), (tf.cast(self.nt, tf.float32) - 1)/2 + self.prior.noise_c)
        t1 = 0.5 * tf.reduce_sum(tf.square(self.k), axis=-1)
        t15 = tf.matmul(self.post.covar, self.JtJ)
        t2 = 0.5 * tf.linalg.trace(t15)
        t0 = 1/self.prior.noise_s
        s_new = 1/(t0 + t1 + t2)
        
        return c_new, s_new

    def _free_energy(self):
        Linv = self.post.covar

        # Calculate individual parts of the free energy
        # For clarify define the following:
        c = self.post.noise_c
        c0 = self.prior.noise_c
        s = self.post.noise_s
        s0 = self.prior.noise_s
        N = tf.cast(self.nt, tf.float32)
        NP = tf.cast(self.nparam, tf.float32)
        P = self.post.precs
        P0 = self.prior.precs
        C = self.post.covar
        C0 = self.prior.covar
        m = self.post.means
        m0 = self.prior.means
        from tensorflow.math import log, digamma, lgamma
        from tensorflow.linalg import matmul, trace, slogdet

        Fparts = []

        # Expected value of the log 
        #
        # -0.5sc(KtK + Tr(CJtJ))
        #
        # Fabber: 1st term in expectedLogPosteriorParts[2] but missing s*c factor
        #         2nd term in expectedLogPosteriorParts[2] missing same factor
        Fparts.append(
            -0.5*s*c*(tf.reduce_sum(tf.square(self.k), axis=-1) + trace(matmul(C, self.JtJ)))
        )

        # Fabber: 1st & 2nd terms in expectedLogPosteriorParts[0]
        #         3nd term in expectedLogPosteriorParts[3]
        Fparts.append(
            0.5*N*(log(s) + digamma(c) - log(2*math.pi))
        )

        # KL divergence of model posterior 
        #
        # Fabber: 1st term in expectedLogThetaDist
        #         2nd term in expectedLogPosteriorParts[5]
        #         3rd term in expectedLogPosteriorParts[3]
        #         4th term in expectedLogThetaDist
        Fparts.append(
            -0.5*(slogdet(P)[1] + trace(matmul(P0, C))) + 0.5*(slogdet(P0)[1] + NP)
        )

        # -0.5 (m-m0)T P0 (m-m0)
        # Fabber: in expectedLogPosteriorParts[4]
        Fparts.append(
            -0.5 * tf.reshape(
              matmul(
                matmul(
                    tf.reshape(m - m0, (self.nv, 1, self.nparam)), 
                    P0
                ),
                tf.reshape(m - m0, (self.nv, self.nparam, 1))
              ), 
              (self.nv,)
            )
        )

        # KL divergence of noise posterior from prior
        # Fabber: 1st term in expectedLogPhiDist
        #         2nd term in expectedLogPhiDist
        #         3rd term in expectedLogPhiDist
        #         4th term in expectedLogPhiDist
        #         5th term in expectedLogPosteriorParts[0]
        #         6th term in expectedLogPosteriorParts[9]
        #         7th term in expectedLogPosteriorParts[9]
        #         8th term in expectedLogPosteriorParts[9]
        Fparts.append(
            -(c-1)*(log(s) + digamma(c)) + c*log(s) + c + lgamma(c) + (c0-1)*(log(s) + digamma(c)) -lgamma(c0) - c0*log(s0) - s*c / s0
        )

        # Fabber: have extra factor of +0.5*Np*log 2PI in expectedLogThetaDist
        #         have extra factor of -0.5*Np*log 2PI in expectedLogPosteriorParts[3]
        # these cancel out!

        # Assemble the parts into F
        self.Fparts = self.log_tf(tf.stack(Fparts, axis=-1), shape=True, force=False)
        F = tf.reduce_sum(self.Fparts, axis=-1)

        return F

    def _build_graph(self, use_adam, **kwargs):
        # Set up prior and posterior
        model_priors = [NormalPrior(p.prior_dist.mean, p.prior_dist.var) for p in self.model.params]
        noise_prior = NoisePrior(scale=kwargs.get("noise_s0", 1e6), shape=kwargs.get("noise_c0", 1e-6))
        self.prior = Prior(model_priors, noise_prior)
        self.post = MVNPosterior(self.data_model, self.model.params, self.tpts, force_positive_vars=use_adam)

        # Get model prediction and Jacobian from current parameters 
        self.modelfit, self.J = self._evaluate()        
        self.Jt = tf.transpose(self.J, (0, 2, 1))
        self.JtJ = tf.matmul(self.Jt, self.J)
        self.noise_means, self.noise_precs = self.noise_mean_prec(self.post)
        self.noise_c, self.noise_s = self.post.noise_c, self.post.noise_s
        self.noise_vars = 1.0/self.noise_precs
        self.k = self.data - self.modelfit
        
        # Here we need to update spatial/ARD priors and capture
        # their contribution to the free energy
        
        # Calculate the free energy and update model parameters
        # using the AVB update equations
        self.free_energy = self._free_energy()
        self.new_means, self.new_cov = self._update_model_params()

        # Follow Fabber in updating residuals after params change before updating 
        # noise. Note we don't update J and nor does Fabber (until the
        # end of the parameter updates when it re-centres the linearized model)
        self.k_new = self.k + tf.einsum("ijk,ik->ij", self.J, self.post.means - self.new_means)
        self.noise_c_new, self.noise_s_new = self._update_noise()
       
        # Define adam optimizer to optimize F directly
        self.cost = -tf.reduce_mean(self.free_energy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        self.optimize = self.optimizer.minimize(self.cost)

        self.init = tf.global_variables_initializer()

    def run(self, record_history=False, use_adam=False, **kwargs):
        self._build_graph(use_adam)

        self.history = {}

        self.sess = tf.Session()
        self.sess.run(self.init)
        
        self._log_iter(0, record_history)
        self._debug_output("Start", self.J)

        if use_adam:
            for idx in range(self._maxits):
                self.sess.run(self.optimize)
                self._log_iter(idx+1, record_history)
        else:
            # Use analytic update equations to update model and noise parameters iteratively
            for idx in range(self._maxits):
                self.update_theta = [
                    self.post.means.assign(self.sess.run(self.new_means)),
                    self.post.covar_v.assign(self.sess.run(self.new_cov))
                ]
                self.sess.run(self.update_theta)
                self._debug_output("Updated theta", self.J) 
                self.update_noise = [
                    self.post.noise_c.assign(self.sess.run(self.noise_c_new)),
                    self.post.noise_s.assign(self.sess.run(self.noise_s_new))
                ]

                self.sess.run(self.update_noise)
                self._debug_output("Updated noise", self.J)
                self._log_iter(idx+1, record_history)

        if record_history:
            for item, item_history in self.history.items():
                # Reshape history items so history is in last axis not first
                trans_axes = list(range(1, item_history[0].ndim+1)) + [0,]
                self.history[item] = np.array(item_history).transpose(trans_axes)

        # Make final output into Numpy arrays
        for attr in ("model_means", "model_vars", "noise_means", "noise_vars", "free_energy", "modelfit"):
            setattr(self, attr, self.sess.run(getattr(self, attr)))

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
