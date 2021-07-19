"""
AVB - Posterior distribution
"""
import math

import numpy as np
import tensorflow as tf

from vaby.utils import LogBase

class Posterior(LogBase):
    """
    Posterior distribution for a parameter
    """

    def build(self):
        """
        Define dependency tensors

        Only tf.Variable tensors may be defined in the constructor. Dependent
        variables must be created in this method to allow gradient recording
        """
        pass

    def avb_update(self, avb):
        """
        Update variables from AVB state
        """
        pass

class NoisePosterior(Posterior):
    """
    Posterior distribution for noise parameter

    :attr s: Noise gamma distribution prior scale parameter [V]
    :attr c: Noise gamma distribution prior shape parameter [V]
    """

    def __init__(self, data_model, **kwargs):
        LogBase.__init__(self)
        nv = data_model.n_unmasked_voxels

        if kwargs.get("init", None) is not None:
            # Initial posterior provided
            # This will be a full set of means and covariances
            # and will include the noise component
            self.log.info("Initializing noise from previous run")
            init_mean, init_cov = kwargs["init"]
            init_mean = init_mean[:, -1]
            init_var = init_cov[:, -1, -1]
            init_s = init_var / init_mean
            init_c = init_mean / init_s
        else:
            init_s = kwargs.get("s", 1e-8)
            init_c = kwargs.get("c", 50.0)
            if np.array(init_s).ndim == 0:
                init_s = tf.fill((nv, ), init_s)
            if np.array(init_c).ndim == 0:
                init_c = tf.fill((nv, ), init_c)

        self.log_s = tf.Variable(tf.math.log(init_s), dtype=tf.float32, name="noise_s")
        self.log_c = tf.Variable(tf.math.log(init_c), dtype=tf.float32, name="noise_c")
        self.vars = [self.log_s, self.log_c]

    def build(self):
        self.s = tf.exp(self.log_s)
        self.c = tf.exp(self.log_c)
        self.mean = self.c*self.s # [V]
        self.var = self.s*self.s*self.c # [V]
        self.prec = 1/self.var # [V]

    def avb_update(self, avb):
        """
        Update noise parameters

        From section 4.2 of the FMRIB Variational Bayes Tutorial I

        :return: Sequence of tuples: (variable, new value)
        """
        c_new = tf.fill((avb.nv,), (tf.cast(avb.nt, tf.float32) - 1)/2 + avb.noise_prior.c)
        # FIXME need k (residuals) in voxel-space
        dmu = avb.orig_mean - avb.post.mean
        dk = tf.einsum("ijk,ik->ij", avb.J, dmu)
        k = avb.k + dk
        t1 = 0.5 * tf.reduce_sum(tf.square(k), axis=-1) # [V]
        # FIXME need CJtJ in voxel-space?
        t15 = tf.matmul(avb.post.cov, avb.JtJ) # [V, P, P]
        t2 = 0.5 * tf.linalg.trace(t15) # [V]
        t0 = 1/avb.noise_prior.s # [V]
        s_new = 1/(t0 + t1 + t2)

        self.log_s.assign(tf.math.log(s_new))
        self.log_c.assign(tf.math.log(c_new))
        self.build()

class MVNPosterior(Posterior):
    """
    MVN Posterior distribution for model parameters

    :attr mean: mean [W, P]
    :attr cov: covariance matrix [W, P, P]
    :attr prec: precision matrix [W, P, P]
    """

    def __init__(self, data_model, params, tpts, **kwargs):
        LogBase.__init__(self)
        self.num_nodes = data_model.n_nodes
        self.num_params = len(params)

        if kwargs.get("init", None) is not None:
            # Initial posterior provided
            # This will be a full set of means and covariances
            # and will include the noise component
            self.log.info("Initializing posterior from previous run")
            init_mean, init_cov = kwargs["init"]
            init_mean = init_mean[:, :-1]
            init_cov = init_cov[:, :-1, :-1]
            init_var = tf.linalg.diag_part(init_cov)
        else:
            # Parameters provide initial mean and variance
            init_mean = []
            init_var = []

            for idx, p in enumerate(params):
                mean, var = None, None
                if p.post_init is not None: # FIXME won't work if nodes != voxels
                    mean, var = p.post_init(idx, tpts, data_model.data_flattened)
                    if mean is not None:
                        mean = p.post_dist.transform.int_values(mean, ns=tf.math)
                    if var is not None:
                        # FIXME transform
                        pass

                if mean is None:
                    mean = tf.fill((self.num_nodes, ), p.post_dist.mean)
                if var is None:
                    var = tf.fill((self.num_nodes, ), p.post_dist.var)

                init_mean.append(tf.cast(mean, tf.float32))
                init_var.append(tf.cast(var, tf.float32))

            # Make shape [W, P]
            init_mean = tf.stack(init_mean, axis=-1)
            init_var = tf.stack(init_var, axis=-1)
            init_cov = tf.linalg.diag(init_var)

        self.mean = tf.Variable(init_mean, dtype=tf.float32, name="post_mean")
        self.vars = [self.mean,]

        self.force_positive_var = kwargs.get("force_positive_var", False)
        if self.force_positive_var:
            # If we want to optimize this using tensorflow we should build it up as in
            # SVB to ensure it is always positive definite. The analytic approach
            # guarantees this automatically (I think!)
            # FIXME Note that we are not initializing the off diag elements yet
            self.log_var = tf.Variable(tf.math.log(init_var), name="post_log_var")
            self.vars.append(self.log_var)

            init_cov_chol = tf.linalg.cholesky(init_cov)
            self.off_diag_vars = tf.Variable(init_cov_chol, name="post_off_diag_cov")
            self.vars.append(self.off_diag_vars)
        else:
            self.cov = tf.Variable(init_cov, dtype=tf.float32, name="post_cov")
            self.vars.append(self.cov)

    def build(self):
        if self.force_positive_var:
            self.var = tf.math.exp(self.log_var)
            self.std = tf.math.sqrt(self.var)
            self.std_diag = tf.linalg.diag(self.std)

            self.off_diag_cov_chol = tf.linalg.set_diag(tf.linalg.band_part(self.off_diag_vars, -1, 0),
                                                        tf.zeros([self.num_nodes, self.num_params]))
            self.off_diag_cov_chol = self.off_diag_cov_chol
            self.cov_chol = tf.add(self.std_diag, self.off_diag_cov_chol)
            self.cov = tf.matmul(tf.transpose(self.cov_chol, perm=(0, 2, 1)), self.cov_chol)

        self.prec = tf.linalg.inv(self.cov)

    def avb_update(self, avb):
        """
        Get AVB updates for model MVN parameters

        From section 4.2 of the FMRIB Variational Bayes Tutorial I

        :return: Sequence of tuples: (variable, new value)
        """
        prec_new = tf.expand_dims(tf.expand_dims(avb.noise_post.s, -1), -1) * tf.expand_dims(tf.expand_dims(avb.noise_post.c, -1), -1) * avb.JtJ + avb.prior.prec
        cov_new = tf.linalg.inv(prec_new)
       
        t1 = tf.einsum("ijk,ik->ij", avb.J, self.mean)
        t15 = tf.einsum("ijk,ik->ij", avb.Jt, (avb.k + t1))
        t2 = tf.expand_dims(avb.noise_post.s, -1) * tf.expand_dims(avb.noise_post.c, -1) * t15
        t3 = tf.einsum("ijk,ik->ij", avb.prior.prec, avb.prior.mean)
        mean_new = tf.einsum("ijk,ik->ij", cov_new, (t2 + t3))

        self.mean.assign(mean_new)
        self.cov.assign(cov_new)
        # FIXME broken if force_positive_var set
        self.build()

class CombinedPosterior(LogBase):
    """
    Represents the parameter posterior and the noise posterior as a single
    distribution for input/output purposes
    """
    def __init__(self, model_post, noise_post):
        self.model_post = model_post
        self.noise_post = noise_post
        self.vars = self.model_post.vars + self.noise_post.vars

    def build(self):
        self.model_post.build()
        self.noise_post.build()

        # FIXME in surface mode noise not on nodes and this will not work
        self.mean = tf.concat([self.model_post.mean, tf.reshape(self.noise_post.mean, (-1, 1))], axis=1)

        cov_model_padded = tf.pad(self.model_post.cov, tf.constant([[0, 0], [0, 1], [0, 1]]))
        cov_noise = tf.reshape(self.noise_post.var, (-1, 1, 1))
        cov_noise_padded = tf.pad(cov_noise, tf.constant([[0, 0], [self.model_post.num_params, 0], [self.model_post.num_params, 0]]))
        self.cov = cov_model_padded + cov_noise_padded

    def avb_update(self, avb):
        self.model_post.avb_update(avb)
        self.noise_post.avb_update(avb)