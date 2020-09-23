"""
AVB - Posterior distribution
"""
import math

import numpy as np
import tensorflow as tf

from svb.utils import LogBase

class MVNPosterior(LogBase):
    """
    Posterior distribution comprising of MVN for model parameters
    and separate noise gamma distribution

    :attr means: Parameter prior means [W, P]
    :attr covar: Parameter covariance matrix [W, P, P]
    :attr precs: Parameter precision matrix [W, P, P]
    :param noise_s: Noise gamma distribution prior scale parameter [W]
    :param noise_c: Noise gamma distribution prior shape parameter [W]
    """

    def __init__(self, data_model, params, tpts, **kwargs):
        """
        :param nv: Number of nodes
        :param init_means: Initial mean values [(W), P]
        :param init_variances: Initial variances [(W), P]
        :param noise_s Initial noise scale parameter [(W)]
        :param noise_c Initial noise shape parameter [(W)]
        """
        LogBase.__init__(self)
        nv = data_model.n_unmasked_voxels

        init_means = []
        init_variances = []

        for idx, p in enumerate(params):
            mean, var = None, None
            if p.post_init is not None:
                mean, var = p.post_init(idx, tpts, data_model.data_flattened)
                if mean is not None:
                    mean = p.post_dist.transform.int_values(mean, ns=tf)
                if var is not None:
                    # FIXME transform
                    pass

            if mean is None:
                mean = tf.fill((nv, ), p.post_dist.mean)
            if var is None:
                var = tf.fill((nv, ), p.post_dist.var)

            init_means.append(tf.cast(mean, tf.float32))
            init_variances.append(tf.cast(var, tf.float32))

        # Make shape [W, P]
        init_means = tf.stack(init_means, axis=-1)
        init_variances = tf.stack(init_variances, axis=-1)
            
        init_noise_s = kwargs.get("noise_s", 1e-8)
        init_noise_c = kwargs.get("noise_c", 50.0)
        if np.array(init_noise_s).ndim == 0:
            init_noise_s = tf.fill((nv, ), init_noise_s)
        if np.array(init_noise_c).ndim == 0:
            init_noise_c = tf.fill((nv, ), init_noise_c)

        self.means = tf.Variable(init_means, dtype=tf.float32)

        # If we want to optimize this using tensorflow we should build it up as in
        # SVB to ensure it is always positive definite. The analytic approach
        # guarantees this automatically (I think!)
        if kwargs.get("force_positive_vars", False):
            self.log_var = tf.Variable(tf.math.log(init_variances))
            self.var = tf.math.exp(self.log_var)
            self.std = tf.math.sqrt(self.var)
            self.std_diag = tf.matrix_diag(self.std)
            covar_init = tf.zeros([nv, len(params), len(params)], dtype=tf.float32)

            self.off_diag_vars = tf.Variable(covar_init)
            self.off_diag_cov_chol = tf.matrix_set_diag(tf.matrix_band_part(self.off_diag_vars, -1, 0),
                                                        tf.zeros([nv, len(params)]))
            self.off_diag_cov_chol = self.log_tf(self.off_diag_cov_chol, shape=True, force=False, name="offdiag")
            self.covar_chol = tf.add(self.std_diag, self.off_diag_cov_chol)
            self.covar = tf.matmul(tf.transpose(self.covar_chol, perm=(0, 2, 1)), self.covar_chol)
            self.covar = self.log_tf(self.covar, shape=True, force=False, name="covar")

            self.log_noise_s = tf.Variable(tf.math.log(init_noise_s), dtype=tf.float32)
            self.log_noise_c = tf.Variable(tf.math.log(init_noise_c), dtype=tf.float32)
            self.noise_s = tf.math.exp(self.log_noise_s)
            self.noise_c = tf.math.exp(self.log_noise_c)
        else:
            self.covar_v = tf.Variable(tf.linalg.diag(init_variances), dtype=tf.float32)
            self.noise_s = tf.Variable(init_noise_s, dtype=tf.float32)
            self.noise_c = tf.Variable(init_noise_c, dtype=tf.float32)
            self.covar = self.log_tf(self.covar_v, shape=True, force=False, name="covar")

        self.precs = tf.linalg.inv(self.covar)
