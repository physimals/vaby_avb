"""
AVB - Posterior distribution
"""
import math

import numpy as np
import tensorflow as tf

from svb.utils import LogBase

class NoisePosterior(LogBase):
    """
    Posterior distribution for noise parameter

    :attr s: Noise gamma distribution prior scale parameter [V]
    :attr c: Noise gamma distribution prior shape parameter [V]
    """

    def __init__(self, data_model, **kwargs):
        nv = data_model.n_unmasked_voxels
        init_s = kwargs.get("s", 1e-8)
        init_c = kwargs.get("c", 50.0)
        if np.array(init_s).ndim == 0:
            init_s = tf.fill((nv, ), init_s)
        if np.array(init_c).ndim == 0:
            init_c = tf.fill((nv, ), init_c)

        self.log_s = tf.Variable(tf.math.log(init_s), dtype=tf.float32, name="noise_s")
        self.log_c = tf.Variable(tf.math.log(init_c), dtype=tf.float32, name="noise_c")
        self.s = tf.exp(self.log_s)
        self.c = tf.exp(self.log_c)
        self.mean = self.c*self.s # [V]
        self.var = self.s*self.s*self.c # [V]
        self.prec = 1/self.var # [V]

    def mean_prec(self):
        return self.mean, self.prec # [V], [V]

    def get_updates(self, avb):
        """
        Get updates for noise parameters

        From section 4.2 of the FMRIB Variational Bayes Tutorial I

        :return: Sequence of tuples: (variable, new value)
        """
        c_new = tf.fill((avb.nv,), (tf.cast(avb.nt, tf.float32) - 1)/2 + avb.noise_prior.c)
        # FIXME need k (residuals) in voxel-space
        t1 = 0.5 * tf.reduce_sum(tf.square(avb.k), axis=-1) # [V]
        # FIXME need CJtJ in voxel-space?
        t15 = tf.matmul(avb.post.cov, avb.JtJ) # [V, P, P]
        t2 = 0.5 * tf.linalg.trace(t15) # [V]
        t0 = 1/avb.noise_prior.s # [V]
        s_new = 1/(t0 + t1 + t2)

        return [
            (self.log_s, tf.math.log(s_new)),
            (self.log_c, tf.math.log(c_new))
        ]

class MVNPosterior(LogBase):
    """
    MVN Posterior distribution for model parameters

    :attr mean: mean [W, P]
    :attr cov: covariance matrix [W, P, P]
    :attr prec: precision matrix [W, P, P]
    """

    def __init__(self, data_model, params, tpts, **kwargs):
        """
        """
        LogBase.__init__(self)
        self.num_nodes = data_model.n_nodes
        self.num_params = len(params)

        init_mean = []
        init_var = []

        for idx, p in enumerate(params):
            mean, var = None, None
            if p.post_init is not None: # FIXME won't work if nodes != voxels
                mean, var = p.post_init(idx, tpts, data_model.data_flattened)
                if mean is not None:
                    mean = p.post_dist.transform.int_values(mean, ns=tf)
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

        self.mean = tf.Variable(init_mean, dtype=tf.float32, name="post_mean")

        # If we want to optimize this using tensorflow we should build it up as in
        # SVB to ensure it is always positive definite. The analytic approach
        # guarantees this automatically (I think!)
        if kwargs.get("force_positive_var", False):
            self.log_var = tf.Variable(tf.math.log(init_var), name="post_log_var")
            self.var = tf.math.exp(self.log_var)
            self.std = tf.math.sqrt(self.var)
            self.std_diag = tf.linalg.diag(self.std)
            cov_init = tf.zeros([self.num_nodes, self.num_params, self.num_params], dtype=tf.float32)

            self.off_diag_vars = tf.Variable(cov_init, name="post_off_diag_cov")
            self.off_diag_cov_chol = tf.linalg.set_diag(tf.linalg.band_part(self.off_diag_vars, -1, 0),
                                                        tf.zeros([self.num_nodes, self.num_params]))
            self.off_diag_cov_chol = self.log_tf(self.off_diag_cov_chol, shape=True, force=False, name="offdiag")
            self.cov_chol = tf.add(self.std_diag, self.off_diag_cov_chol)
            self.cov = tf.matmul(tf.transpose(self.cov_chol, perm=(0, 2, 1)), self.cov_chol)
            self.cov = self.log_tf(self.cov, shape=True, force=False, name="cov")
        else:
            self.cov = tf.Variable(tf.linalg.diag(init_var), dtype=tf.float32, name="post_cov")

        self.prec = tf.linalg.inv(self.cov)

    def get_updates(self, avb):
        """
        Get updates for model MVN parameters

        From section 4.2 of the FMRIB Variational Bayes Tutorial I

        :return: Sequence of tuples: (variable, new value)
        """
        prec_new = tf.expand_dims(tf.expand_dims(avb.noise_post.s, -1), -1) * tf.expand_dims(tf.expand_dims(avb.noise_post.c, -1), -1) * avb.JtJ + avb.prior.prec
        cov_new = tf.linalg.inv(prec_new)

        t1 = self.log_tf(tf.einsum("ijk,ik->ij", avb.J, self.mean), force=False, shape=True, name="t1")
        t15 = tf.einsum("ijk,ik->ij", avb.Jt, (avb.k + t1))
        t2 = self.log_tf(tf.expand_dims(avb.noise_post.s, -1) * tf.expand_dims(avb.noise_post.c, -1) * t15, force=False, shape=True, name="t2")
        t3 = self.log_tf(tf.einsum("ijk,ik->ij", avb.prior.prec, avb.prior.mean), force=False, shape=True, name="t3")
        mean_new = tf.einsum("ijk,ik->ij", cov_new, (t2 + t3))

        return [
            (self.mean, mean_new),
            (self.cov, cov_new),
        ]

class CombinedPosterior(LogBase):
    """
    Represents the parameter posterior and the noise posterior as a single
    distribution for input/output purposes
    """
    def __init__(self, post, noise_post):
        # FIXME in surface mode noise not on nodes and this will not work
        self.mean = tf.concat([post.mean, tf.reshape(noise_post.mean, (-1, 1))], axis=1)

        cov_model_padded = tf.pad(post.cov, tf.constant([[0, 0], [0, 1], [0, 1]]))
        cov_noise = tf.reshape(noise_post.var, (-1, 1, 1))
        cov_noise_padded = tf.pad(cov_noise, tf.constant([[0, 0], [post.num_params, 0], [post.num_params, 0]]))
        self.cov = cov_model_padded + cov_noise_padded
