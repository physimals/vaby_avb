"""
VABY_AVB - Posterior distribution
"""
import numpy as np
import tensorflow as tf

from vaby.utils import LogBase, TF_DTYPE, NP_DTYPE
import vaby.dist as dist

def get_posterior(idx, param, data_model, **kwargs):
    """
    Factory method to return a posterior

    :param param: svb.parameter.Parameter instance
    """
    initial_mean, initial_var = None, None
    if param.post_init is not None:
        initial_mean, initial_var = param.post_init(param, data_model.data_space.srcdata.flat)

        if initial_mean is not None:
            initial_mean = data_model.data_to_model(initial_mean, pv_scale=param.pv_scale)
        if initial_var is not None:
            initial_var = data_model.data_to_model(initial_var, pv_scale=param.pv_scale)

    # The size of the posterior (number of positions at which it is 
    # estimated) is determined by the data_space it refers to, and 
    # in turn by the data model. If it is global, the reduction will
    # be applied when creating the GaussianGlobalPosterior later on 
    post_size = data_model.model_space.size

    if initial_mean is None:
        initial_mean = tf.fill([post_size], NP_DTYPE(param.post_dist.mean))
    else:
        initial_mean = param.post_dist.transform.int_values(NP_DTYPE(initial_mean))

    if initial_var is None:
        initial_var = tf.fill([post_size], NP_DTYPE(param.post_dist.var))
    else:
        # FIXME variance not value?
        initial_var = param.post_dist.transform.int_values(NP_DTYPE(initial_var))

    if isinstance(param.post_dist, dist.Normal):
        return NormalPosterior(idx, initial_mean, initial_var, **kwargs)

    #if isinstance(param.post_dist, dist.Normal):
    #    return GaussianGlobalPosterior(idx, initial_mean, initial_var, **kwargs)

    raise ValueError("Can't create posterior for distribution: %s" % param.post_dist)
        
class Posterior(LogBase):
    """
    Posterior distribution for a parameter

    Attributes:
     - ``nvars`` Number of variates for a multivariate distribution
     - ``nnodes`` Number of independent nodes at which the distribution is estimated
     - ``variables`` Sequence of tf.Variable objects containing posterior state
     - ``mean`` [nnodes, nvars] Mean value(s) at each node
     - ``var`` [nnodes, nvars] Variance(s) at each node
     - ``cov`` [nnodes, nvars, nvars] Covariance matrix at each node

    ``nvars`` (if > 1) and ``variables`` must be initialized in the constructor. Other
    attributes must be initialized either in the constructor (if they are constant
    tensors or tf.Variable) or in ``build`` (if they are dependent tensors). The
    constructor should call ``build`` after initializing constant and tf.Variable
    tensors.
    """
    def __init__(self, idx, data_model, **kwargs):
        LogBase.__init__(self, **kwargs)
        self.idx = idx
        self.data_model = data_model
        self.nnodes = data_model.model_space.size
        self.nvars = 1

    def build(self):
        """
        Define tensors that depend on Variables in the posterior
        
        Only constant tensors and tf.Variables should be defined in the constructor.
        Any dependent variables must be created in this method to allow gradient 
        recording
        """
        pass

    def avb_update(self, avb):
        """
        Update variables from AVB state
        """
        raise NotImplementedError()

    def sample(self, nsamples):
        """
        :param nsamples: Number of samples to return per parameter vertex / parameter

        :return: A tensor of shape [W, P, S] where W is the number
                 of parameter nodes, P is the number of parameters in the distribution
                 (possibly 1) and S is the number of samples
        """
        raise NotImplementedError()

    def entropy(self, samples=None):
        """
        :param samples: A tensor of shape [W, P, S] where W is the number
                        of parameter nodes, P is the number of parameters in the prior
                        (possibly 1) and S is the number of samples.
                        This parameter may or may not be used in the calculation.
                        If it is required, the implementation class must check
                        that it is provided

        :return Tensor of shape [W] containing vertexwise distribution entropy
        """
        raise NotImplementedError()

    def log_det_cov(self):
        """
        :return: Log of the determinant of the covariance matrix
        """
        raise NotImplementedError()

class NoisePosterior(Posterior):
    """
    Gamma posterior for noise parameter.

    Currently only used in AVB - methods required to support SVB not
    yet implemented.

    :attr s: Noise gamma distribution prior scale parameter [V]
    :attr c: Noise gamma distribution prior shape parameter [V]
    """

    def __init__(self, data_model, **kwargs):
        LogBase.__init__(self)
        nv = data_model.data_space.size

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

        self.log_s = tf.Variable(tf.math.log(init_s), dtype=TF_DTYPE)
        self.log_c = tf.Variable(tf.math.log(init_c), dtype=TF_DTYPE)
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
        c_new = tf.fill((avb.n_voxels,), (NP_DTYPE(avb.n_tpts) - 1)/2 + avb.noise_prior.c)
        # FIXME need k (residuals) in voxel-space
        dmu = avb.orig_mean - avb.post.mean
        dk = tf.einsum("ijk,ik->ij", avb.J, dmu)
        dk_data = avb.data_model.model_to_data(dk) # FIXME pv_scale?
        k = avb.k + dk_data # [V, T]
        t1 = 0.5 * tf.reduce_sum(tf.square(k), axis=-1) # [V]
        # FIXME need CJtJ in voxel-space?
        t15 = tf.matmul(avb.post.cov, avb.JtJ) # [W, P, P]
        t2 = 0.5 * tf.linalg.trace(t15) # [W]
        t2_data = avb.data_model.model_to_data(t2) # FIXME pv_scale
        t0 = 1/avb.noise_prior.s # [V]
        s_new = 1/(t0 + t1 + t2_data)

        self.log_s.assign(tf.math.log(s_new))
        self.log_c.assign(tf.math.log(c_new))

class MVNPosterior(Posterior):
    """
    MVN Posterior distribution for model parameters

    :attr mean: mean [W, P]
    :attr cov: covariance matrix [W, P, P]
    :attr prec: precision matrix [W, P, P]
    """

    def __init__(self, data_model, params, tpts, **kwargs):
        Posterior.__init__(self, 0, data_model)
        self.nvars = len(params)

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
                    mean, var = p.post_init(idx, data_model.data_space.srcdata.flat)
                    if mean is not None:
                        mean = p.post_dist.transform.int_values(mean, ns=tf.math)
                        mean = data_model.data_to_model(mean, pv_scale=p.pv_scale)
                    if var is not None:
                        # FIXME transform
                        pass

                if mean is None:
                    mean = tf.fill((self.nnodes, ), p.post_dist.mean)
                if var is None:
                    var = tf.fill((self.nnodes, ), p.post_dist.var)

                init_mean.append(tf.cast(mean, TF_DTYPE))
                init_var.append(tf.cast(var, TF_DTYPE))

            # Make shape [W, P]
            init_mean = tf.stack(init_mean, axis=-1)
            init_var = tf.stack(init_var, axis=-1)
            init_cov = tf.linalg.diag(init_var)

        self.mean = tf.Variable(init_mean, dtype=TF_DTYPE)
        self.vars = [self.mean,]

        self.force_positive_var = kwargs.get("force_positive_var", False)
        if self.force_positive_var:
            # If we want to optimize this using tensorflow we should build it up as in
            # SVB to ensure it is always positive definite. The analytic approach
            # guarantees this automatically (I think!)
            # FIXME Note that we are not initializing the off diag elements yet
            self.log_var = tf.Variable(tf.math.log(init_var))
            self.vars.append(self.log_var)

            init_cov_chol = tf.linalg.cholesky(init_cov)
            self.off_diag_vars = tf.Variable(init_cov_chol)
            self.vars.append(self.off_diag_vars)
        else:
            self.cov = tf.Variable(init_cov, dtype=TF_DTYPE)
            self.vars.append(self.cov)

    def build(self):
        if self.force_positive_var:
            self.var = tf.math.exp(self.log_var)
            std_diag = tf.linalg.diag(tf.math.sqrt(self.var))

            off_diag_cov_chol = tf.linalg.set_diag(tf.linalg.band_part(self.off_diag_vars, -1, 0),
                                                   tf.zeros([self.nnodes, self.nvars]))
            off_diag_cov_chol = off_diag_cov_chol
            cov_chol = tf.add(std_diag, off_diag_cov_chol)
            self.cov = tf.matmul(tf.transpose(cov_chol, perm=(0, 2, 1)), cov_chol)

        self.prec = tf.linalg.inv(self.cov)

    def avb_update(self, avb):
        """
        Get AVB updates for model MVN parameters

        From section 4.2 of the FMRIB Variational Bayes Tutorial I

        FIXME use of pv_scale?

        :return: Sequence of tuples: (variable, new value)
        """
        sc = avb.noise_post.s * avb.noise_post.c # [V]
        sc_nodes = avb.data_model.data_to_model(sc, pv_scale=True) # [W]
        k_nodes = avb.data_model.data_to_model(avb.k, pv_scale=True)
        prec_new = tf.expand_dims(tf.expand_dims(sc_nodes, -1), -1) * avb.JtJ + avb.prior.prec # [W, P, P]
        cov_new = tf.linalg.inv(prec_new)

        t1 = tf.einsum("ijk,ik->ij", avb.J, self.mean) # [W]
        t15 = tf.einsum("ijk,ik->ij", avb.Jt, (k_nodes + t1)) # [W, T]
        t2 = tf.expand_dims(sc_nodes, -1) * t15 # [W, T]
        t3 = tf.einsum("ijk,ik->ij", avb.prior.prec, avb.prior.mean)
        mean_new = tf.einsum("ijk,ik->ij", cov_new, (t2 + t3))

        self.mean.assign(mean_new)
        self.cov.assign(cov_new)

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
        #self.mean = tf.concat([self.model_post.mean, tf.reshape(self.noise_post.mean, (-1, 1))], axis=1)

        #cov_model_padded = tf.pad(self.model_post.cov, tf.constant([[0, 0], [0, 1], [0, 1]]))
        #cov_noise = tf.reshape(self.noise_post.var, (-1, 1, 1))
        #cov_noise_padded = tf.pad(cov_noise, tf.constant([[0, 0], [self.model_post.nvars, 0], [self.model_post.nvars, 0]]))
        #self.cov = cov_model_padded + cov_noise_padded

    def avb_update(self, avb):
        self.model_post.avb_update(avb)
        self.noise_post.avb_update(avb)
