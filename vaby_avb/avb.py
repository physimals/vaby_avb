"""
VABY_AVB - Implementation of analytic Variational Bayes to infer a 
nonlinear forward model

This implements section 4 of the FMRIB Variational Bayes tutorial 1.
The code is intended to be compatible with Fabber, however numerical
differences are expected as the forward model is not identical.
"""
import math

import numpy as np
import tensorflow as tf

from vaby.utils import LogBase

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

        self.data = self.data_model.data_flat
        self.nv, self.nt = tuple(self.data.shape)
        self.nn = self.data_model.n_nodes
        self.nparam = len(self.model.params)

    def debug_output(self, text):
        self.log.debug(text)
        self.log.debug("Prior mean: %s\n", self.prior.mean)
        self.log.debug("Prior prec: %s\n", self.prior.prec)
        self.log.debug("Post mean: %s\n", self.post.mean)
        self.log.debug("Post prec: %s\n", self.post.prec)
        self.log.debug("Noise prior mean: %s\n", self.noise_prior.mean)
        self.log.debug("Noise prior prec: %s\n", self.noise_prior.prec)
        self.log.debug("Noise post mean: %s\n", self.noise_post.mean)
        self.log.debug("Noise post prec: %s\n", self.noise_post.prec)
        self.log.debug("Jacobian:\n%s\n", self.J)

    def jacobian(self):
        """
        Numerical differentiation to calculate Jacobian matrix
        of partial derivatives of model prediction with respect to
        parameters

        :return: Jacobian matrix [W, B, P]
        """
        J = []
        mean_trans = tf.transpose(self.post.mean, (1, 0)) # [P, W]
        params = tf.expand_dims(mean_trans, axis=-1)
        min_delta = tf.fill(tf.shape(params), 1e-5)
        delta = tf.math.abs(params * 1e-5) # [P, W, 1]
        delta = tf.where(delta < 1e-5, min_delta, delta)
        plus = params + delta # [P, W, 1]
        minus = params - delta # [P, W, 1]
        #print("Params\n", params)
        #print(plus, minus)
        #print(delta)
        for param_idx in range(self.nparam):
            pick_param = tf.equal(tf.range(self.nparam), param_idx) # [P]
            pick_param = tf.expand_dims(tf.expand_dims(pick_param, -1), -1) # [P, 1, 1]
            plus_param = tf.where(pick_param, plus, params) # [P, W, 1]
            minus_param = tf.where(pick_param, minus, params) # [P, W, 1]
            plus_model = self.inference_to_model(plus_param)
            minus_model = self.inference_to_model(minus_param)
            #print(param_idx, plus_model.numpy(), minus_model.numpy())
            # diff [W, B]
            diff = self.model.evaluate(plus_model, self.tpts) - self.model.evaluate(minus_model, self.tpts)
            #print("Delta\n", delta[param_idx])
            #print("Diff\n", diff)
            J.append(diff / (2*delta[param_idx]))

        J = tf.stack(J, axis=-1) # [W, B, P]
        #print("J", J.shape)
        #print(J.numpy())
        return J

    @tf.function
    def jacobian_autodiff(self):
        """
        Experimental jacobian using Tensorflow. Doesn't batch over voxels currently
        that might need TF2

        :param modelfit: [W, B]
        :param mean: [W, P]
        """
        with tf.GradientTape() as tape:
            mean_trans = tf.transpose(self.post.mean, (1, 0)) # [P, W]
            params_model = self.inference_to_model(tf.expand_dims(mean_trans, axis=-1))
            modelfit = self.model.evaluate(params_model, self.tpts)

        J = tape.batch_jacobian(modelfit, self.post.mean)
        #print(J)
        return J

    def inference_to_model(self, inference_params, idx=None):
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

    def model_to_inference(self, model_params, idx=None):
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

    def free_energy(self):
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
        m = self.post.mean
        m0 = self.prior.mean
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
                tf.squeeze(to_voxels(tf.expand_dims(tf.linalg.trace(tf.linalg.matmul(C, self.JtJ)), 1)), 1)
            )
        )

        # Fabber: 1st & 2nd terms in expectedLogPosteriorParts[0]
        #         3nd term in expectedLogPosteriorParts[3]
        Fparts_vox.append(
            0.5*N*(tf.math.log(s) + tf.math.digamma(c) - tf.math.log(2*math.pi)) # [V]
        )

        # KL divergence of model posterior 
        #
        # Fabber: 1st term in expectedLogThetaDist
        #         2nd term in expectedLogPosteriorParts[5]
        #         3rd term in expectedLogPosteriorParts[3]
        #         4th term in expectedLogThetaDist
        Fparts_node.append(
            -0.5*(-tf.linalg.slogdet(C)[1] + tf.linalg.trace(tf.linalg.matmul(P0, C))) + 0.5*(tf.linalg.slogdet(P0)[1] + NP) # [W]
        )

        # -0.5 (m-m0)T P0 (m-m0)
        # Fabber: in expectedLogPosteriorParts[4]
        Fparts_node.append(
            -0.5 * tf.reshape(
              tf.linalg.matmul(
                tf.linalg.matmul(
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
            -(c-1)*(tf.math.log(s) + tf.math.digamma(c)) + c*tf.math.log(s) + c + tf.math.lgamma(c) + (c0-1)*(tf.math.log(s) + tf.math.digamma(c)) -tf.math.lgamma(c0) - c0*tf.math.log(s0) - s*c / s0 # [V]
        )

        # Fabber: have extra factor of +0.5*Np*log 2PI in expectedLogThetaDist
        #         have extra factor of -0.5*Np*log 2PI in expectedLogPosteriorParts[3]
        # these cancel out!

        # Assemble the parts into F
        #
        # FIXME we have some parts defined nodewise and some voxelwise. Need to keep them
        # separate until we sum the contributions when optimizing the total free energy
        self.Fparts_vox = tf.stack(Fparts_vox, axis=-1)
        self.Fparts_node = tf.stack(Fparts_node, axis=-1)
        F_vox = tf.reduce_sum(self.Fparts_vox, axis=-1)
        F_node = tf.reduce_sum(self.Fparts_node, axis=-1)
        for prior in self.param_priors:
            F_node += prior.free_energy()

        return F_vox, F_node

    def create_prior_post(self, **kwargs):
        # Set up prior and posterior
        self.tpts = tf.constant(self.tpts, dtype=tf.float32) # FIXME
        self.noise_post = NoisePosterior(self.data_model, force_positive=kwargs.get("use_adam", False), init=self.data_model.post_init)
        self.post = MVNPosterior(self.data_model, self.model.params, self.tpts, init=self.data_model.post_init, **kwargs)

        self.noise_prior = NoisePrior(s=kwargs.get("noise_s0", 1e6), c=kwargs.get("noise_c0", 1e-6))
        self.param_priors = [
            get_prior(idx, p, self.data_model, self.post, **kwargs) 
            for idx, p in enumerate(self.model.params)
        ]
        self.prior = MVNPrior(self.param_priors)

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

    def evaluate(self):
        mean_trans = tf.transpose(self.post.mean, (1, 0)) # [P, W]
        self.model_mean = self.inference_to_model(mean_trans) # [P, W]
        self.model_var = tf.stack([self.post.cov[:, v, v] for v in range(len(self.model.params))]) # [P, W] FIXME transform
        self.noise_mean, self.noise_prec = self.noise_post.mean, self.noise_post.prec # [V], [V]
        self.noise_var = 1.0/self.noise_prec # [V]
        modelfit_nodes = self.model.evaluate(tf.expand_dims(self.model_mean, axis=-1), self.tpts) # [W, B]
        self.modelfit = tf.squeeze(self.data_model.nodes_to_voxels_ts(tf.expand_dims(modelfit_nodes, 1)), 1) # [V, B]
        self.k = self.data - self.modelfit # [V, B, P]

    def linearise(self):
        self.J = self.jacobian() # [W, B, P]
        #self.J = self.jacobian_autodiff() # [W, B, P]
        self.Jt = tf.transpose(self.J, (0, 2, 1)) # [W, P, B]
        self.JtJ = tf.linalg.matmul(self.Jt, self.J) # [W, P, P]

    def cost_free_energy(self):
        self.prior.build()
        self.all_post.build()
        self.linearise()
        self.evaluate()
        self.free_energy_vox, self.free_energy_node = self.free_energy()
        self.cost_fe = -tf.reduce_mean(self.free_energy_vox) - tf.reduce_mean(self.free_energy_node)
        return self.cost_fe

    def cost_leastsq(self):
        self.prior.build()
        self.all_post.build()
        self.linearise()
        self.evaluate()
        self.cost_free_energy()
        self.cost_lsq = tf.reduce_sum(tf.square(self.k), axis=-1)
        return self.cost_lsq

    def run_leastsq(self, max_iterations=10, record_history=False, **kwargs):
        self.log.info("Doing least squares optimization")
        optimizer = tf.keras.optimizers.Adam(learning_rate=kwargs.get("learning_rate", 0.05))
        for idx in range(max_iterations):
            with tf.GradientTape(persistent=False) as t:
                # Loss function
                self.cost = self.cost_leastsq()
            gradients = t.gradient(self.cost, self.all_post.vars)

            # Apply the gradient
            optimizer.apply_gradients(zip(gradients, self.all_post.vars))
            self.log_iter(idx+1, record_history)

    def run_maxf(self, max_iterations=10, record_history=False, **kwargs):
        self.log.info("Doing free energy maximisation")
        optimizer = tf.keras.optimizers.Adam(learning_rate=kwargs.get("learning_rate", 0.05))
        for idx in range(max_iterations):
            with tf.GradientTape(persistent=False) as t:
                # Loss function
                self.cost = self.cost_free_energy()
            gradients = t.gradient(self.cost, self.all_post.vars)

            # Apply the gradient
            optimizer.apply_gradients(zip(gradients, self.all_post.vars))
            self.log_iter(idx+1, record_history)

    def run_analytic(self, max_iterations=10, record_history=False, progress_cb=None, **kwargs):
        self.log.info("Doing analytic VB")
        # Use analytic update equations to update model and noise parameters iteratively
        self.cost_free_energy()
        self.log_iter(0, record_history)
        for idx in range(max_iterations):
            # Update model parameters
            self.orig_mean = self.post.mean.numpy()
            self.post.avb_update(self)
            self.debug_output("Updated theta")
    
            # Update noise parameters
            self.noise_post.avb_update(self)
            self.debug_output("Updated noise")

            # Update priors (e.g. spatial, ARD)
            self.prior_updates = []
            for prior in self.param_priors:
                prior.avb_update(self)

            self.cost_free_energy()
            self.log_iter(idx+1, record_history)
            if progress_cb is not None:
                progress_cb(float(idx)/float(max_iterations))

    def run(self, **kwargs):
        self.history = {}
        self.create_prior_post(**kwargs)

        method = kwargs.pop("method", "analytic")
        if method == "leastsq":
            self.run_leastsq(**kwargs)
        elif method == "maxf":
            self.run_maxf(**kwargs)
        elif method == "analytic":
            self.run_analytic(**kwargs)
        else:
            raise ValueError("Unknown optimization method: %s" % method)

        if kwargs.get("record_history", False):
            for item, item_history in self.history.items():
                # Reshape history items so history is in last axis not first
                trans_axes = list(range(1, item_history[0].ndim+1)) + [0,]
                self.history[item] = np.array(item_history).transpose(trans_axes)

        # Make final output into Numpy arrays
        for attr in ("model_mean", "model_var", "noise_mean", "noise_var", "free_energy_vox", "free_energy_node",
                     "modelfit"):
            setattr(self, attr, getattr(self, attr).numpy())

    def log_iter(self, iter, history):
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
            if not hasattr(self, attr):
                iter_data[attr] = 0
                continue
            else:
                data = getattr(self, attr).numpy()
                if data.size > 1:
                    # voxelwise data
                    iter_data[attr] = np.mean(data, axis=-1)
                else:
                    iter_data[attr] = data
            if history:
                if attr not in self.history: self.history[attr] = []
                self.history[attr].append(data)
        self.log.info(fmt % iter_data)
