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
            if False and p.post_init is not None: # FIXME initialization disabled
                mean, var = p.post_init(idx, tpts, data_model.data_flattened)
                if mean is not None:
                    mean = p.post_dist.transform.int_values(mean, ns=np)
                if var is not None:
                    # FIXME transform
                    pass
            if mean is None:
                mean = np.full((nv, ), p.post_dist.mean, dtype=np.float32)
            if var is None:
                var = np.full((nv, ), p.post_dist.var, dtype=np.float32)
            init_means.append(mean)
            init_variances.append(var)

        # Make shape [W, P]
        init_means = np.array(init_means).transpose()
        init_variances = np.array(init_variances).transpose()
            
        init_noise_s = kwargs.get("noise_s", 1e-8)
        init_noise_c = kwargs.get("noise_c", 50.0)
        if np.array(init_noise_s).ndim == 0:
            init_noise_s = np.full((nv, ), init_noise_s)
        if np.array(init_noise_c).ndim == 0:
            init_noise_c = np.full((nv, ), init_noise_c)

        self.means = tf.Variable(init_means, dtype=tf.float32)

        # If we want to optimize this using tensorflow we should build it up as in
        # SVB to ensure it is always positive definite. The analytic approach
        # guarantees this automatically (I think!)
        self.covar = tf.Variable(tf.linalg.diag(init_variances), dtype=tf.float32)
        self.precs = tf.linalg.inv(self.covar)
        self.noise_s = tf.Variable(init_noise_s, dtype=tf.float32)
        self.noise_c = tf.Variable(init_noise_c, dtype=tf.float32)

    def update_model_params(self, k, J, prior):
        """
        Update model parameters

        From section 4.2 of the FMRIB Variational Bayes Tutorial I
        
        :param k: data - prediction [W, N]
        :param J: Jacobian [W, N, P]
        :param precs: precisions (prior=P0)
        :param noise_s: noise scale (prior = s0)
        :param noise_c: noise shape (prior = c0)

        :return Tuple of (New means, New precisions)
        """
        jt = tf.transpose(J, (0, 2, 1))
        jtd = tf.matmul(jt, J)
        precs_new = tf.expand_dims(tf.expand_dims(self.noise_s, -1), -1) * tf.expand_dims(tf.expand_dims(self.noise_c, -1), -1) * jtd + prior.precs
        covar_new = tf.linalg.inv(precs_new)

        t1 = tf.einsum("ijk,ik->ij", J, self.means)
        t15 = tf.einsum("ijk,ik->ij", jt, (k + t1))
        t2 = tf.expand_dims(self.noise_s, -1) * tf.expand_dims(self.noise_c, -1) * t15
        t3 = tf.einsum("ijk,ik->ij", prior.precs, prior.means)
        means_new = tf.einsum("ijk,ik->ij", covar_new, (t2 + t3))

        #self.sess.run(self.means.assign(self.sess.run(means_new)))
        #self.sess.run(self.precs.assign(precs_new))
        #self.sess.run(self.covar.assign(self.sess.run(tf.linalg.inv(precs_new))))
        return means_new, covar_new

    def update_noise(self, k, J, prior):
        """
        Update noise parameters

        From section 4.2 of the FMRIB Variational Bayes Tutorial I
        
        :param precs: precisions (prior=P0)
        :param k: data - prediction
        :param J: Jacobian matrix at current parameter values [W, N, P]

        :return: Tuple of (New scale param, New shape param)
        """
        nt = tf.cast(tf.shape(J)[1], tf.float32)
        nv = tf.shape(J)[0]
        c_new = tf.fill((nv,), (nt-1)/2 + prior.noise_c)
        jt = tf.transpose(J, (0, 2, 1))
        t1 = 0.5 * tf.reduce_sum(tf.square(k), axis=-1)
        t12 = tf.matmul(jt, J)
        t15 = tf.matmul(self.covar, tf.matmul(jt, J))
        t2 = 0.5 * tf.linalg.trace(t15)
        t0 = 1/prior.noise_s
        s_new = 1/(t0 + t1 + t2)
        #self.sess.run(self.noise_c.assign(self.sess.run(c_new)))
        #self.sess.run(self.noise_s.assign(self.sess.run(s_new)))
        return c_new, s_new

    def free_energy(self, k, J, prior):
        Linv = self.covar
        J = self.log_tf(J, name="J", shape=True, force=False)
        nv = self.log_tf(tf.shape(J)[0], name="nv", force=False)
        nt = self.log_tf(tf.shape(J)[1], name="nt", force=False)
        nparam = self.log_tf(tf.shape(J)[2], name="np", force=False)
        jt = self.log_tf(tf.transpose(J, (0, 2, 1)), name="jt", shape=True, force=False)

        # Calculate individual parts of the free energy

        # Bits arising from the factorised posterior for theta
        expectedLogThetaDist = 0.5 * tf.linalg.slogdet(self.precs)[1] - 0.5 * tf.cast(nparam, tf.float32) * (tf.log(2 * math.pi) + 1)

        # Bits arising fromt he factorised posterior for phi
        expectedLogPhiDist = -tf.math.lgamma(self.noise_c) - self.noise_c * tf.log(self.noise_s) - self.noise_c + (self.noise_c - 1) * (tf.math.digamma(self.noise_c) + tf.log(self.noise_s))

        # Bits arising from the likelihood
        expectedLogPosteriorParts = []

        # nTimes using phi_{i+1} = Qis[i].Trace()
        expectedLogPosteriorParts.append(
            (tf.math.digamma(self.noise_c) + tf.log(self.noise_s)) * (tf.cast(nt, tf.float32) * 0.5 + prior.noise_c - 1)
        )

        expectedLogPosteriorParts.append(
            -tf.math.lgamma(prior.noise_c) - prior.noise_c * tf.log(prior.noise_s) - self.noise_s * self.noise_c / prior.noise_s
        )

        expectedLogPosteriorParts.append(
            -0.5 * tf.reduce_sum(tf.square(k), axis=-1) - 0.5 * tf.linalg.trace(tf.matmul(tf.matmul(jt, J), self.covar))
        )

        expectedLogPosteriorParts.append(
            +0.5 * tf.linalg.slogdet(prior.precs)[1]
            - 0.5 * tf.cast(nt, tf.float32) * tf.log(2 * math.pi) - 0.5 * tf.cast(nparam, tf.float32) * tf.log(2 * math.pi)
        )

        means = self.log_tf(self.means, name="means", shape=True, force=False)
        pmeans = self.log_tf(prior.means, name="pmeans", shape=True, force=False)
        pprecs = self.log_tf(prior.precs, name="pprecs", shape=True, force=False)
        
        expectedLogPosteriorParts.append(
            -0.5 * tf.reshape(
              tf.matmul(
                tf.matmul(
                    tf.reshape(means - pmeans, (nv, 1, nparam)), 
                    pprecs
                ),
                tf.reshape(means - pmeans, (nv, nparam, 1))
              ), 
              (nv,)
            )
        )

        expectedLogPosteriorParts.append(
            -0.5 * tf.linalg.trace(self.covar * prior.precs)
        )

        # Assemble the parts into F
        F = -expectedLogThetaDist - expectedLogPhiDist + sum(expectedLogPosteriorParts)

        return F

