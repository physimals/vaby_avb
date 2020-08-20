"""
Simple implementation of analytic Variational Bayes to infer a 
nonlinear forward model

This implements section 4 of the FMRIB Variational Bayes tutorial 1
"""
import math

import numpy as np
import scipy.special
import tensorflow as tf

from svb.utils import LogBase

class Prior(LogBase):

    def __init__(self, means, variances, noise_s, noise_c):
        """
        :param means: Prior mean values [(W), P]
        :param variances: Prior variances [(W), P]
        :param noise_s Prior noise scale parameter [(W)]
        :param noise_c Prior noise shape parameter [(W)]
        """
        LogBase.__init__(self)
        while np.array(means).ndim < 2:
            means = np.array(means)[np.newaxis, ...]
        while np.array(variances).ndim < 2:
            variances = np.array(variances)[np.newaxis, ...]
        self.means = np.array(means, dtype=np.float32)
        self.variances = np.array(variances, dtype=np.float32)
        self.covar = np.array([np.diag(v) for v in variances], dtype=np.float32)
        self.precs = np.linalg.inv(self.covar)
        self.noise_s = noise_s
        self.noise_c = noise_c

class Posterior(LogBase):

    def __init__(self, nv, init_means, init_variances, noise_s, noise_c):
        """
        :param nv: Number of nodes
        :param init_means: Initial mean values [(W), P]
        :param init_variances: Initial variances [(W), P]
        :param noise_s Initial noise scale parameter [(W)]
        :param noise_c Initial noise shape parameter [(W)]
        """
        LogBase.__init__(self)
        if np.array(init_means).ndim == 1:
            init_means = np.tile(np.array(init_means)[np.newaxis, ...], (nv, 1))
        if np.array(init_variances).ndim == 1:
            init_variances = np.tile(np.array(init_variances)[np.newaxis, ...], (nv, 1))

        if np.array(noise_s).ndim == 0:
            noise_s = np.tile(np.atleast_1d(noise_s), (nv, ))
        if np.array(noise_c).ndim == 0:
            noise_c = np.tile(np.atleast_1d(noise_c), (nv, ))

        self.means = np.array(init_means, dtype=np.float32)
        self.covar = np.array([np.diag(v) for v in init_variances], dtype=np.float32)
        self.precs = np.linalg.inv(self.covar)
        self.noise_s = np.array(noise_s, dtype=np.float32)
        self.noise_c = np.array(noise_c, dtype=np.float32)

    def update_model_params(self, k, J, prior):
        """
        Update model parameters

        From section 4.2 of the FMRIB Variational Bayes Tutorial I
        
        :param k: data - prediction
        :param means: means (prior = M0)
        :param precs: precisions (prior=P0)
        :param noise_s: noise scale (prior = s0)
        :param noise_c: noise shape (prior = c0)
        :param J: Jacobian matrix at parameter values

        :return Tuple of (New means, New precisions)
        """
        jt = J.transpose(0, 2, 1)
        jtd = np.matmul(jt, J)
        precs_new = self.noise_s[..., np.newaxis, np.newaxis]*self.noise_c[..., np.newaxis, np.newaxis]*jtd + prior.precs
        covar_new = np.linalg.inv(precs_new)

        t1 = np.einsum("ijk,ik->ij", J, self.means)
        t15 = np.einsum("ijk,ik->ij", jt, (k + t1))
        t2 = self.noise_s[..., np.newaxis] * self.noise_c[..., np.newaxis] * t15
        t3 = np.einsum("ijk,ik->ij", prior.precs, prior.means)
        means_new = np.einsum("ijk,ik->ij", covar_new, (t2 + t3))
        self.means = means_new
        self.precs = precs_new
        self.covar = np.linalg.inv(self.precs)

    def update_noise(self, k, J, prior):
        """
        Update noise parameters

        From section 4.2 of the FMRIB Variational Bayes Tutorial I
        
        :param precs: precisions (prior=P0)
        :param k: data - prediction
        :param J: Jacobian matrix at parameter values

        :return: Tuple of (New scale param, New shape param)
        """
        nt = J.shape[1]
        nv = J.shape[0]
        c_new = np.zeros((nv,), dtype=np.float32)
        c_new[:] = (nt-1)/2 + prior.noise_c
        jt = J.transpose(0, 2, 1)
        t1 = 0.5 * np.sum(np.square(k), axis=-1)
        t12 = np.matmul(jt, J)
        t15 = np.matmul(self.covar, np.matmul(jt, J))
        t2 = 0.5 * np.trace(t15, axis1=-1, axis2=-2)
        t0 = 1/prior.noise_s
        s_new = 1/(t0 + t1 + t2)
        self.noise_c = c_new
        self.noise_s = s_new

    def free_energy(self, k, J, prior):
        Linv = self.covar
        nv = J.shape[0]
        nt = J.shape[1]
        nparam = J.shape[2]
        jt = J.transpose(0, 2, 1)

        # Calculate individual parts of the free energy

        # Bits arising from the factorised posterior for theta
        expectedLogThetaDist = 0.5 * np.linalg.slogdet(self.precs)[1] - 0.5 * nparam * (np.log(2 * math.pi) + 1)

        # Bits arising fromt he factorised posterior for phi
        expectedLogPhiDist = -scipy.special.gammaln(self.noise_c) - self.noise_c * np.log(self.noise_s) - self.noise_c + (self.noise_c - 1) * (scipy.special.digamma(self.noise_c) + np.log(self.noise_s))

        # Bits arising from the likelihood
        expectedLogPosteriorParts = []

        # nTimes using phi_{i+1} = Qis[i].Trace()
        expectedLogPosteriorParts.append(
            (scipy.special.digamma(self.noise_c) + np.log(self.noise_s)) * (nt * 0.5 + prior.noise_c - 1)
        )

        expectedLogPosteriorParts.append(
            -scipy.special.gammaln(prior.noise_c) - prior.noise_c * np.log(prior.noise_s) - self.noise_s * self.noise_c / prior.noise_s
        )

        expectedLogPosteriorParts.append(
            -0.5 * np.sum(np.square(k), axis=-1) - 0.5 * np.trace(np.matmul(np.matmul(jt, J), self.covar), axis1=-1, axis2=-2)
        )

        expectedLogPosteriorParts.append(
            +0.5 * np.linalg.slogdet(prior.precs)[1]
            - 0.5 * nt * np.log(2 * math.pi) - 0.5 * nparam * np.log(2 * math.pi)
        )

        expectedLogPosteriorParts.append(
            -0.5
            * np.matmul(
                np.matmul(
                    np.reshape(self.means - prior.means, (nv, 1, nparam)), prior.precs),
                    np.reshape(self.means - prior.means, (nv, nparam, 1)
                )
            ).reshape((nv,))
        )

        expectedLogPosteriorParts.append(
            -0.5 * np.trace(self.covar * prior.precs, axis1=-1, axis2=-2)
        )

        # Assemble the parts into F
        F = -expectedLogThetaDist - expectedLogPhiDist + sum(expectedLogPosteriorParts)

        return F

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

        prior_means = [p.prior_dist.mean for p in self.model.params]
        prior_vars = [p.prior_dist.var for p in self.model.params]
        self.prior = Prior(prior_means, prior_vars,
                            kwargs.get("noise_s0", 1e6), kwargs.get("noise_c0", 1e-6))

        post_means = [p.post_dist.mean for p in self.model.params]
        post_vars = [p.post_dist.var for p in self.model.params]
        self.post = Posterior(self._nv, post_means, post_vars,
                               noise_s=kwargs.get("noise_s", 1e-8), noise_c=kwargs.get("noise_c", 50.0))

        for idx, p in enumerate(self.model.params):
            if p.post_init is not None:
                mean, var = p.post_init(idx, tpts, self._data)
                if mean is not None:
                    self.post.means[:, idx] = self._model_to_inference(mean, idx)
                if var is not None:
                    # FIXME transform
                    self.post.covar[:, idx, idx] = var
                    self.post.precs = np.linalg.inv(self.post.covar)
        
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
        nparams = len(params)
        nv = params[0].shape[0]
        J = np.zeros([nv, nt, nparams], dtype=np.float32)
        for param_idx, param_value in enumerate(params):
            plus = params.copy()
            minus = params.copy()
            delta = param_value * 1e-5
            delta[delta < 0] = -delta[delta < 0]
            delta[delta < 1e-5] = 1e-5

            plus[param_idx] += delta
            minus[param_idx] -= delta

            plus = self._inference_to_model(plus)
            minus = self._inference_to_model(minus)
            diff = self.model.evaluate(plus, tpts) - self.model.evaluate(minus, tpts)
            J[..., param_idx] = diff / (2*delta)
        return J

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
        self.means_trans = params.transpose((1, 0))
        self.model_means = self._inference_to_model(self.means_trans)
        self.model_vars = np.array([self.post.covar[:, v, v] for v in range(len(self.model.params))]) # fixme transform
        modelfit = self.model.evaluate(self.model_means[..., np.newaxis], tpts)
        J = self.jacobian(self.means_trans[..., np.newaxis], self._tpts)
        return modelfit, J

    def _inference_to_model(self, inference_params, idx=None):
        """
        Transform inference engine parameters into model parameters

        :param params: Inference engine parameters in shape [P, V, 1]
        """
        if idx is None:
            model_params = []
            for idx, p in enumerate(inference_params):
                model_params.append(self.model.params[idx].post_dist.transform.ext_values(p, ns=np))
            return tf.stack(model_params)
        else:
            return self.model.params[idx].post_dist.transform.ext_values(inference_params, ns=np)

    def _model_to_inference(self, model_params, idx=None):
        """
        Transform inference engine parameters into model parameters

        :param params: Inference engine parameters in shape [P, V, 1]
        """
        if idx is None:
            inference_params = []
            for idx, p in enumerate(model_params):
                inference_params.append(self.model.params[idx].post_dist.transform.int_values(p, ns=np))
            return tf.stack(model_params)
        else:
            return self.model.params[idx].post_dist.transform.int_values(model_params, ns=np)

    def run(self, history=False):
        self._init_run()
        # Update model and noise parameters iteratively
        for idx in range(self._maxits):
            orig_means = np.copy(self.post.means)

            #print(self.post.means.shape)
            #print(self._tpts.shape)
            #jac = jacobian(self._modelfit) 
            #print(jac)
            #print(jacobian(self._modelfit)(self.post.means, self._tpts))
            #print(jac(self.post.means[0, ...], self._tpts[0, ...]))
            #J = [np.squeeze(jac(self.post.means[v, ...][np.newaxis, ...], self._tpts[0, ...][np.newaxis, ...])) for v in range(self.post.means.shape[0])]
            #print(J[0].shape)
            #J = np.stack(J)
            #J = jacobian(self._modelfit)(self.post.means, self._tpts)
            #print(J.shape)
            self.modelfit, J = self._evaluate(self.post.means, self._tpts)
            self.noise_means, self.noise_precs = self.noise_mean_prec(self.post)
            self.noise_vars = 1.0/self.noise_precs
            k = self._data - self.modelfit
            #print(J.shape)
            self.free_energy = self.post.free_energy(k, J, self.prior)
            if idx == 0:
                self._log_iter(idx, history)
                self._debug_output("Start", J)
            self.post.update_model_params(k, J, self.prior)
            self._debug_output("Updated theta", J)

            # Follow Fabber in recalculating residuals after params change
            # Note we don't recalculate J and nor does Fabber (until the
            # end of the parameter updates when it re-centres the linearized model)
            k = k + np.einsum("ijk,ik->ij", J, orig_means - self.post.means)
            self.post.update_noise(k, J, self.prior)
            self._debug_output("Updated noise", J)
            self._log_iter(idx+1, history)

        if history:
            for item, history in self.history.items():
                # Reshape history items so history is in last axis not first
                trans_axes = list(range(1, history[0].ndim+1)) + [0,]
                self.history[item] = np.array(history).transpose(trans_axes)

    def _log_iter(self, iter, history):
        fmt = {"iter" : iter}
        for attr in ("model_means", "model_vars", "noise_means", "noise_vars", "free_energy"):
            voxelwise_data = getattr(self, attr)
            mean_data = np.mean(voxelwise_data, axis=-1)
            fmt[attr] = mean_data
            if attr not in self.history: self.history[attr] = []
            if history:
                self.history[attr].append(voxelwise_data)
        self.log.info("Iteration %(iter)i: params=%(model_means)s, vars=%(model_vars)s, noise mean=%(noise_means)e, var=%(noise_vars)e, F=%(free_energy)e" % fmt)
