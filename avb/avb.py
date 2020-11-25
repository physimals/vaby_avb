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
import tensorflow.compat.v1 as tf
from tensorflow.python.ops.parallel_for.gradients import jacobian
tf.disable_v2_behavior()

from svb.utils import LogBase

from .prior import MVNPrior, NoisePrior, get_prior
from .posterior import MVNPosterior, NoisePosterior, CombinedPosterior

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
        self.nn = self.data_model.n_nodes
        self.nparam = len(self.model.params)
        self._debug = kwargs.get("debug", False)
        self._maxits = kwargs.get("max_iterations", 10)

    def _debug_output(self, text, J=None):
        if self._debug:
            self.log.debug(text)
            self.log.debug("Prior mean\n", self.prior.mean)
            self.log.debug("Prior prec\n", self.prior.prec)
            self.log.debug("Post mean\n", self.post.mean)
            self.log.debug("Post prec\n", self.post.prec)
            noise_mean, noise_prec = self.noise_prior.mean_prec()
            self.log.debug("Noise prior mean\n", noise_mean)
            self.log.debug("Noise prior prec\n", noise_prec)
            noise_mean, noise_prec = self.noise_post.mean_prec()
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
            delta = tf.where(delta < 1e-10, tf.fill(tf.shape(delta), 1e-10), delta)

            pick_param = tf.equal(tf.range(self.nparam), param_idx)
            plus = tf.where(pick_param, params + delta, params)
            minus = tf.where(pick_param, params - delta, params)

            plus = self.log_tf(self._inference_to_model(plus), force=False, shape=True, name="plus")
            minus = self.log_tf(self._inference_to_model(minus), force=False, shape=True, name="minus")
            diff = self.log_tf(self.model.evaluate(plus, tpts), force=False, shape=True, name="plus") - self.log_tf(self.model.evaluate(minus, tpts), force=False, shape=True, name="minus")
            J.append(diff / (2*delta))
        return self.log_tf(tf.stack(J, axis=-1), shape=True, force=False)

    def jacobian2(self, modelfit, mean):
        """
        Experimental jacobian using Tensorflow. Doesn't batch over voxels currently
        that might need TF2

        :param modelfit: [W, B]
        :param mean: [W, P]
        """
        J = tf.stack([tf.reshape(jacobian(modelfit[0], mean), (self.nt, self.nparam))])
        #J = tf.stack([
        #    jacobian(modelfit[t], mean[t])
        #    for t in range(self.nv)
        #])
        return self.log_tf(J, shape=True, force=False)

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

    def _sum_sq_resid(self):
        cost = tf.reduce_sum(tf.square(self.k), axis=-1)
        return cost

    def _free_energy(self):
        Linv = self.post.cov

        # Calculate individual parts of the free energy
        # For clarity define the following:
        c = self.noise_post.c
        c0 = self.noise_prior.c
        s = self.noise_post.s
        s0 = self.noise_prior.s
        N = tf.cast(self.nt, tf.float32)
        NP = tf.cast(self.nparam, tf.float32)
        #P = self.post.prec
        P0 = self.prior.prec
        C = self.post.cov
        C0 = self.prior.cov
        m = self.post.mean
        m0 = self.prior.mean
        from tensorflow.math import log, digamma, lgamma
        from tensorflow.linalg import matmul, trace, slogdet
        to_voxels = self.data_model.nodes_to_voxels

        Fparts_vox = []
        Fparts_node = []

        # Expected value of the log likelihood
        #
        # noise and k (residuals) are voxelwise, C and JtJ are nodewise
        # since this is likelihood of data probably want everything voxelwise?
        #
        # -0.5sc(KtK + Tr(CJtJ))
        #
        # Fabber: 1st term in expectedLogPosteriorParts[2] but missing s*c factor
        #         2nd term in expectedLogPosteriorParts[2] missing same factor
        Fparts_vox.append(
            -0.5*s*c*(
                tf.reduce_sum(tf.square(self.k), axis=-1)
            )
        )

        Fparts_vox.append(
            -0.5*s*c*(
                tf.squeeze(to_voxels(tf.expand_dims(trace(matmul(C, self.JtJ)), 1)), 1)
            )
        )

        # Fabber: 1st & 2nd terms in expectedLogPosteriorParts[0]
        #         3nd term in expectedLogPosteriorParts[3]
        Fparts_vox.append(
            0.5*N*(log(s) + digamma(c) - log(2*math.pi)) # [V]
        )

        # KL divergence of model posterior 
        #
        # Fabber: 1st term in expectedLogThetaDist
        #         2nd term in expectedLogPosteriorParts[5]
        #         3rd term in expectedLogPosteriorParts[3]
        #         4th term in expectedLogThetaDist
        Fparts_node.append(
            -0.5*(-slogdet(C)[1] + trace(matmul(P0, C))) + 0.5*(slogdet(P0)[1] + NP) # [W]
        )

        # -0.5 (m-m0)T P0 (m-m0)
        # Fabber: in expectedLogPosteriorParts[4]
        Fparts_node.append(
            -0.5 * tf.reshape(
              matmul(
                matmul(
                    tf.reshape(m - m0, (self.nn, 1, self.nparam)), 
                    P0
                ),
                tf.reshape(m - m0, (self.nn, self.nparam, 1))
              ), 
              (self.nn,)
            )
        ) # [W]

        # KL divergence of noise posterior from prior
        # Fabber: 1st term in expectedLogPhiDist
        #         2nd term in expectedLogPhiDist
        #         3rd term in expectedLogPhiDist
        #         4th term in expectedLogPhiDist
        #         5th term in expectedLogPosteriorParts[0]
        #         6th term in expectedLogPosteriorParts[9]
        #         7th term in expectedLogPosteriorParts[9]
        #         8th term in expectedLogPosteriorParts[9]
        Fparts_vox.append(
            -(c-1)*(log(s) + digamma(c)) + c*log(s) + c + lgamma(c) + (c0-1)*(log(s) + digamma(c)) -lgamma(c0) - c0*log(s0) - s*c / s0 # [V]
        )

        # Fabber: have extra factor of +0.5*Np*log 2PI in expectedLogThetaDist
        #         have extra factor of -0.5*Np*log 2PI in expectedLogPosteriorParts[3]
        # these cancel out!

        # Assemble the parts into F
        #
        # FIXME we have some parts defined nodewise and some voxelwise. Need to keep them
        # separate until we sum the contributions when optimizing the total free energy
        self.Fparts_vox = self.log_tf(tf.stack(Fparts_vox, axis=-1), shape=True, force=False, name="F_vox")
        self.Fparts_node = self.log_tf(tf.stack(Fparts_node, axis=-1), shape=True, force=False)
        F_vox = tf.reduce_sum(self.Fparts_vox, axis=-1)
        F_node = tf.reduce_sum(self.Fparts_node, axis=-1)
        for prior in self.param_priors:
            F_node += prior.free_energy()

        return F_vox, F_node

    def _build_graph(self, **kwargs):
        # Set up prior and posterior
        self.tpts = tf.constant(self.tpts, dtype=tf.float32) # FIXME
        self.noise_post = NoisePosterior(self.data_model, force_positive=kwargs.get("use_adam", False))
        self.post = MVNPosterior(self.data_model, self.model.params, self.tpts, force_positive_var=kwargs.get("use_adam", False))

        self.noise_prior = NoisePrior(s=kwargs.get("noise_s0", 1e6), c=kwargs.get("noise_c0", 1e-6))
        self.param_priors = [
            get_prior(idx, p, self.data_model, self.post, **kwargs) 
            for idx, p in enumerate(self.model.params)
        ]
        self.prior = MVNPrior(self.param_priors, self.noise_prior)

        # Combined parameter/noise posterior for output only
        self.all_post = CombinedPosterior(self.post, self.noise_post)

        # Report summary of parameters
        for idx, param in enumerate(self.model.params):
            self.log.info(" - %s", param)
            self.log.info("   - Prior: %s %s", param.prior_dist, self.param_priors[idx])
            self.log.info("   - Posterior: %s", param.post_dist)

        # FIXME Hack for output
        idx = 1
        for p in self.param_priors:
            if hasattr(p, "ak"):
                setattr(self, "ak%i" % idx, p.ak)
                idx += 1

        # Get model prediction, Jacobian and residuals from current parameters
        self.mean_trans = tf.transpose(self.post.mean, (1, 0)) # [P, W]
        self.model_mean = self.log_tf(self._inference_to_model(self.mean_trans), force=False, shape=True, name="mean") # [P, W]
        self.model_var = tf.stack([self.post.cov[:, v, v] for v in range(len(self.model.params))]) # [P, W] FIXME transform
        self.modelfit_nodes = self.model.evaluate(tf.expand_dims(self.model_mean, axis=-1), self.tpts) # [W, B]
        self.modelfit_nodes = self.log_tf(self.modelfit_nodes, force=False, shape=True, name="modelfit") 
        self.J_nodes = self.jacobian(tf.expand_dims(self.mean_trans, axis=-1), self.tpts) # [W, B, P]

        # Convert model prediction and Jacobian to voxel space
        self.modelfit = tf.squeeze(self.data_model.nodes_to_voxels_ts(tf.expand_dims(self.modelfit_nodes, 1)), 1) # [V, B]
        self.J = self.log_tf(self.J_nodes, force=False, shape=True, name="J", summarize=1000) # FIXME? [V, B, P]
        self.Jt = tf.transpose(self.J, (0, 2, 1)) # [W, P, B]
        self.JtJ = self.log_tf(tf.matmul(self.Jt, self.J), shape=True, force=False, name="jtj") # [W, P, P]
        data = self.log_tf(tf.constant(self.data, dtype=tf.float32), name="data", force=False, shape=True)
        self.k = self.log_tf(data - self.modelfit, name="k", force=False, shape=True) # [V, B, P]

        # Convenience for outputing noise mean, variance
        self.noise_mean, self.noise_prec = self.noise_post.mean_prec() # [V], [V]
        self.noise_var = 1.0/self.noise_prec # [V]

        # Convenience for outputing combined posterior
        self.all_mean = self.all_post.mean
        self.all_cov = self.all_post.cov

        # Calculate the free energy and update model parameters
        # using the AVB update equations
        self.free_energy_vox, self.free_energy_node = self._free_energy()
        self.sum_sq_resid = self._sum_sq_resid()

        # Follow Fabber in updating residuals after params change before updating 
        # noise. Note we don't update J and nor does Fabber (until the
        # end of the parameter updates when it re-centres the linearized model)
        #self.k_new = self.k + tf.einsum("ijk,ik->ij", self.J, self.post.mean - self.new_mean)
       
        # Define adam optimizer to optimize F directly
        self.cost = -tf.reduce_mean(self.free_energy_vox) - tf.reduce_mean(self.free_energy_node)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=kwargs.get("learning_rate", 0.5))
        self.optimize_leastsq = self.optimizer.minimize(self.sum_sq_resid)
        self.optimize_f = self.optimizer.minimize(self.cost)
        self.init = tf.global_variables_initializer()

    def run_leastsq(self, record_history, **kwargs):
        for idx in range(self._maxits):
            self.sess.run(self.optimize_leastsq)
            self._log_iter(idx+1, record_history)

    def run_maxf(self, record_history, **kwargs):
        for idx in range(self._maxits):
            self.sess.run(self.optimize_f)
            self._log_iter(idx+1, record_history)

    def run_analytic(self, record_history, **kwargs):
        # Use analytic update equations to update model and noise parameters iteratively
        for idx in range(self._maxits):
            # Update model parameters
            param_updates = [
                var.assign(self.sess.run(value)) for var, value in self.post.get_updates(self)
            ]
            self.sess.run(param_updates)
            self._debug_output("Updated theta", self.J) 

            # Update noise parameters
            noise_updates = [
                var.assign(self.sess.run(value)) for var, value in self.noise_post.get_updates(self)
            ]
            self.sess.run(noise_updates)
            self._debug_output("Updated noise", self.J)

            # Update priors (e.g. spatial, ARD)
            self.prior_updates = []
            for prior in self.param_priors:
                for var, value in prior.get_updates(self.post):
                    self.prior_updates.append(var.assign(self.sess.run(value)))
            self.sess.run(self.prior_updates)

            self._log_iter(idx+1, record_history)

    def run(self, record_history=False, **kwargs):
        self.history = {}
        
        self._build_graph(**kwargs)
        self.sess = tf.Session()
        self.sess.run(self.init)

        self._log_iter(0, record_history)
        self._debug_output("Start", self.J)

        if kwargs.get("use_adam", False):
            if kwargs.get("init_leastsq", False):
                self.run_leastsq(record_history)
            self.run_maxf(record_history)
        else:
            self.run_analytic(record_history)
            
        if record_history:
            for item, item_history in self.history.items():
                # Reshape history items so history is in last axis not first
                trans_axes = list(range(1, item_history[0].ndim+1)) + [0,]
                self.history[item] = np.array(item_history).transpose(trans_axes)

        # Make final output into Numpy arrays
        for attr in ("model_mean", "model_var", "noise_mean", "noise_var", "free_energy_vox", "free_energy_node",
                     "modelfit", "all_mean", "all_cov"):
            setattr(self, attr, self.sess.run(getattr(self, attr)))

    def _log_iter(self, iter, history):
        iter_data = {"iter" : iter}
        attrs = ["model_mean", "model_var", "noise_mean", "noise_var", "free_energy_vox", "free_energy_node", "cost"]
        fmt = "Iteration %(iter)i: params=%(model_mean)s, var=%(model_var)s, noise mean=%(noise_mean)e, var=%(noise_var)e, F=%(free_energy_vox)e, %(free_energy_node)e, %(cost)e"

        # Pick up any spatial smoothing params to output
        # FIXME ugly and hacky
        idx = 1
        while hasattr(self, "ak%i" % idx):
            attrs.append("ak%i" % idx)
            fmt += ", ak%i=%%(ak%i)e" % (idx, idx)
            idx += 1

        for attr in attrs:
            data = self.sess.run(getattr(self, attr))
            if data.size > 1:
                # voxelwise data
                data = np.mean(data, axis=-1)
            iter_data[attr] = data
            if history:
                if attr not in self.history: self.history[attr] = []
                self.history[attr].append(voxelwise_data)
        self.log.info(fmt % iter_data)
