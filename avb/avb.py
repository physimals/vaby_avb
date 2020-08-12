"""
Simple implementation of analytic Variational Bayes to infer a 
nonlinear forward model

This implements section 4 of the FMRIB Variational Bayes tutorial 1
"""
import math

import numpy as np
from scipy.special import digamma, gammaln

from .utils import LogBase

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
        self.means = means
        self.variances = variances
        self.covar = np.array([np.diag(v) for v in variances])
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

        self.means = np.array(init_means)
        self.covar = np.array([np.diag(v) for v in init_variances])
        self.precs = np.linalg.inv(self.covar)
        self.noise_s = noise_s
        self.noise_c = noise_c

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
        expectedLogPhiDist = -gammaln(self.noise_c) - self.noise_c * np.log(self.noise_s) - self.noise_c + (self.noise_c - 1) * (digamma(self.noise_c) + np.log(self.noise_s))

        # Bits arising from the likelihood
        expectedLogPosteriorParts = []

        # nTimes using phi_{i+1} = Qis[i].Trace()
        expectedLogPosteriorParts.append(
            (digamma(self.noise_c) + np.log(self.noise_s)) * (nt * 0.5 + prior.noise_c - 1)
        )

        expectedLogPosteriorParts.append(
            -gammaln(prior.noise_c) - prior.noise_c * np.log(prior.noise_s) - self.noise_s * self.noise_c / prior.noise_s
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
                np.matmul(np.reshape(self.means - prior.means, (nv, 1, nparam)), prior.precs),
                np.reshape(self.means - prior.means, (nv, nparam, 1))
            )
        )

        expectedLogPosteriorParts.append(
            -0.5 * np.trace(self.covar * prior.precs, axis1=-1, axis2=-2)
        )

        # Assemble the parts into F
        F = -expectedLogThetaDist - expectedLogPhiDist + sum(expectedLogPosteriorParts)

        return F

class Avb(LogBase):

    def __init__(self, tpts, data, model, **kwargs):
        """
        :param data: Data timeseries [(W), B]
        :param model: Forward model
        """
        LogBase.__init__(self)
        while data.ndim < 2:
            data = data[np.newaxis, ...]
        while tpts.ndim < 2:
            tpts = tpts[np.newaxis, ...]
        self._tpts = tpts
        self._data = data
        self._nv, self._nt = tuple(data.shape)
        self._model = model
        self._debug = kwargs.get("debug", False)

        prior_means = [p.prior_dist.mean for p in model.params]
        prior_vars = [p.prior_dist.var for p in model.params]
        self._prior = Prior(prior_means, prior_vars, 
                            kwargs.get("noise_s0", 1e6), kwargs.get("noise_c0", 1e-6))

        post_means = [p.post_dist.mean for p in model.params]
        post_vars = [p.post_dist.var for p in model.params]
        self._post = Posterior(self._nv, post_means, post_vars, 
                               noise_s=kwargs.get("noise_s", 1e-8), noise_c=kwargs.get("noise_c", 50.0))

    def noise_mean_prec(self, dist):
        return dist.noise_c*dist.noise_s, 1/(dist.noise_s*dist.noise_s*dist.noise_c)

    def _debug_output(self, text, J=None):
        if self._debug:
            print(text)
            print("Prior mean\n", self._prior.means)
            print("Prior precs\n", self._prior.precs)
            print("Post mean\n", self._post.means)
            print("Post precs\n", self._post.precs)
            noise_mean, noise_prec = self.noise_mean_prec(self._prior)
            print("Noise prior mean\n", noise_mean)
            print("Noise prior prec\n", noise_prec)
            noise_mean, noise_prec = self.noise_mean_prec(self._post)
            print("Noise post mean\n", noise_mean)
            print("Noise post prec\n", noise_prec)
            if J is not None:
                print("Jacobian\n", J)
            #print("data\n", self._data)
            #print("eval\n", self._model.evaluate(means_reshaped, self._tpts))
            #print("k\n", k)

    def run(self):
        self.log_iter(0)

        # Update model and noise parameters iteratively
        for idx in range(200):
            orig_means = np.copy(self._post.means)

            # Make means shape [P, V, 1] to enable the parameters
            # to be unpacked as a sequence 
            means_reshaped = self._post.means.transpose(1, 0)[..., np.newaxis]
            k = self._data - self._model.evaluate(means_reshaped, self._tpts)
            J = self._model.jacobian(means_reshaped, self._tpts)
            self._debug_output("Start", J)
            self._post.update_model_params(k, J, self._prior)
            self._debug_output("Updated theta", J)

            # Follow Fabber in recalculating residuals after params change
            # Note we don't recalculate J and nor does Fabber (until the
            # end of the parameter updates when it re-centres the
            # linearized model)
            k = k + np.einsum("ijk,ik->ij", J, orig_means - self._post.means)
            self._post.update_noise(k, J, self._prior)
            self._debug_output("Updated noise", J)
            self.log_iter(idx+1)
            print("f=%f" % self._post.free_energy(k, J, self._prior))

    def log_iter(self, idx):
        c = np.mean(self._post.noise_c)
        s = np.mean(self._post.noise_s)
        noise_mean, noise_prec = self.noise_mean_prec(self._post)
        noise_mean, noise_prec = np.mean(noise_mean), np.mean(noise_prec)
        print("Iteration %i: params=%s, noise: s=%e, c=%e, m=%e p=%e" % (idx, np.mean(self._post.means, axis=0), s, c, noise_mean, noise_prec))


# Priors - noninformative because of high variance
#
# Note that the noise posterior is a gamma distribution
# with shape and scale parameters s, c. The mean here is
# b*c and the variance is c * b^2. To make this more 
# intuitive we define a prior mean and variance for the 
# noise parameter BETA and express the prior scale
# and shape parameters in terms of these
#
# So long as the priors stay noninformative they should not 
# have a big impact on the inferred values - this is the 
# point of noninformative priors. However if you start to
# reduce the prior variances the inferred values will be
# drawn towards the prior values and away from the values
# suggested by the data
a0 = 1.0
a_var0 = 1000
lam0 = 1.0
lam_var0 = 10.0

noise_mean0 = 1
noise_var0 = 1000
# c=scale, s=shape parameters for Gamma distribution
noise_c0 = noise_var0 / noise_mean0
noise_s0 = noise_mean0**2 / noise_var0
