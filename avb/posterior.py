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

        self.log_s = tf.Variable(tf.math.log(init_s), dtype=tf.float32)
        self.log_c = tf.Variable(tf.math.log(init_c), dtype=tf.float32)
        self.s = tf.exp(self.log_s)
        self.c = tf.exp(self.log_c)

    def mean_prec(self):
        return self.c*self.s, 1/(self.s*self.s*self.c) # [V], [V]

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
        t15 = tf.matmul(avb.post.covar, avb.JtJ) # [V, P, P]
        t2 = 0.5 * tf.linalg.trace(t15) # [V]
        t0 = 1/avb.noise_prior.s # [V]
        s_new = 1/(t0 + t1 + t2)

        return [
            (self.log_s, tf.log(s_new)),
            (self.log_c, tf.log(c_new))
        ]

class MVNPosterior(LogBase):
    """
    MVN Posterior distribution for model parameters

    :attr means: Parameter prior means [W, P]
    :attr covar: Parameter covariance matrix [W, P, P]
    :attr precs: Parameter precision matrix [W, P, P]
    """

    def __init__(self, data_model, params, tpts, **kwargs):
        """
        """
        LogBase.__init__(self)
        nn = data_model.n_nodes

        init_means = []
        init_variances = []

        for idx, p in enumerate(params):
            mean, var = None, None
            if False and p.post_init is not None:
                mean, var = p.post_init(idx, tpts, data_model.data_flattened)
                if mean is not None:
                    mean = p.post_dist.transform.int_values(mean, ns=tf)
                if var is not None:
                    # FIXME transform
                    pass

            if mean is None:
                mean = tf.fill((nn, ), p.post_dist.mean)
            if var is None:
                var = tf.fill((nn, ), p.post_dist.var)

            init_means.append(tf.cast(mean, tf.float32))
            init_variances.append(tf.cast(var, tf.float32))

        # Make shape [W, P]
        init_means = tf.stack(init_means, axis=-1)
        init_variances = tf.stack(init_variances, axis=-1)

        self.means = tf.Variable(init_means, dtype=tf.float32)

        # If we want to optimize this using tensorflow we should build it up as in
        # SVB to ensure it is always positive definite. The analytic approach
        # guarantees this automatically (I think!)
        if kwargs.get("force_positive_vars", False):
            self.log_var = tf.Variable(tf.math.log(init_variances))
            self.var = tf.math.exp(self.log_var)
            self.std = tf.math.sqrt(self.var)
            self.std_diag = tf.matrix_diag(self.std)
            covar_init = tf.zeros([nn, len(params), len(params)], dtype=tf.float32)

            self.off_diag_vars = tf.Variable(covar_init)
            self.off_diag_cov_chol = tf.matrix_set_diag(tf.matrix_band_part(self.off_diag_vars, -1, 0),
                                                        tf.zeros([nn, len(params)]))
            self.off_diag_cov_chol = self.log_tf(self.off_diag_cov_chol, shape=True, force=False, name="offdiag")
            self.covar_chol = tf.add(self.std_diag, self.off_diag_cov_chol)
            self.covar = tf.matmul(tf.transpose(self.covar_chol, perm=(0, 2, 1)), self.covar_chol)
            self.covar = self.log_tf(self.covar, shape=True, force=False, name="covar")
        else:
            self.covar = tf.Variable(tf.linalg.diag(init_variances), dtype=tf.float32)

        self.precs = tf.linalg.inv(self.covar)

    def get_updates(self, avb):
        """
        Get updates for model MVN parameters

        From section 4.2 of the FMRIB Variational Bayes Tutorial I

        :return: Sequence of tuples: (variable, new value)
        """
        precs_new = tf.expand_dims(tf.expand_dims(avb.noise_post.s, -1), -1) * tf.expand_dims(tf.expand_dims(avb.noise_post.c, -1), -1) * avb.JtJ + avb.prior.precs
        covar_new = tf.linalg.inv(precs_new)

        t1 = tf.einsum("ijk,ik->ij", avb.J, avb.post.means)
        t15 = tf.einsum("ijk,ik->ij", avb.Jt, (avb.k + t1))
        t2 = tf.expand_dims(avb.noise_post.s, -1) * tf.expand_dims(avb.noise_post.c, -1) * t15
        t3 = tf.einsum("ijk,ik->ij", avb.prior.precs, avb.prior.means)
        means_new = tf.einsum("ijk,ik->ij", covar_new, (t2 + t3))

        return [
            (self.means, means_new),
            (self.covar, covar_new),
        ]
