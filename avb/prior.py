"""
AVB - Priors for model and noise parameters
"""

import numpy as np
import tensorflow as tf

from svb.utils import LogBase
from svb.dist import Normal

def get_prior(idx, param, data_model, post, **kwargs):
    """
    Factory method to return a vertexwise prior
    """
    prior = None
    if isinstance(param.prior_dist, Normal):
        if param.prior_type == "N":
            prior = NormalPrior(data_model, param.prior_dist.mean, param.prior_dist.var, **kwargs)
        elif param.prior_type == "M":
            prior = MRFSpatialPrior(data_model, post, idx, param.prior_dist.var, **kwargs)
        #elif param.prior_type == "M2":
        #    prior = MRF2SpatialPrior(data_model.n_vertices, param.prior_dist.mean, param.prior_dist.var, **kwargs)
        #elif param.prior_type == "Mfab":
        #    prior = FabberMRFSpatialPrior(data_model.n_vertices, param.prior_dist.mean, param.prior_dist.var, **kwargs)
        elif param.prior_type == "A":
            prior = ARDPrior(data_model, idx, param.prior_dist.var, **kwargs)

    if prior is not None:
        return prior
    else:
        raise ValueError("Can't create prior type %s for distribution %s - unrecognized combination" % (param.prior_type, param.prior_dist))

class ParameterPrior(LogBase):
    """
    Gaussian prior for a single model parameter
    
    :attr mean: Mean value [W] or [1]
    :attr variance: Variance [W] or [1]
    """

    def free_energy(self):
        """
        :return: Free energy contribution associated with this prior, if required. [W] or constant
        """
        return 0

    def get_updates(self, post):
        """
        Update prior in analytic update mode

        Required for adaptive priors, e.g. spatial, ARD

        :param post: Current posterior
        :return: Sequence of tuples: (variable to update, tensor to update from)
        """
        return []

class NormalPrior(ParameterPrior):
    """
    Fixed prior for a single model parameter
    """
    
    def __init__(self, data_model, mean, variance, **kwargs):
        """
        :param mean: Prior mean value as float or Numpy array [W]
        :param variance: Prior mean value as float or Numpy array [W]
        """
        LogBase.__init__(self)

        # Mean/variance is usually specified globally so
        # make sure it has a nodewise dimension
        nn = data_model.n_nodes
        while np.array(mean).ndim < 1:
            mean = np.full((nn,), mean)
        while np.array(variance).ndim < 1:
            variance = np.full((nn,), variance)

        self.mean = tf.constant(mean, dtype=np.float32)
        self.variance = tf.constant(variance, dtype=np.float32)

class MRFSpatialPrior(ParameterPrior):
    """
    Spatial prior based on a Markov Random Field
    """

    def __init__(self, data_model, post, idx, init_variance, **kwargs):
        """
        """
        LogBase.__init__(self)

        # Laplacian matrix with diagonal zeroed
        offdiag_mask = data_model.laplacian.row != data_model.laplacian.col
        indices=np.array([
                data_model.laplacian.row[offdiag_mask], 
                data_model.laplacian.col[offdiag_mask]
        ]).T
        laplacian_nodiag = tf.SparseTensor(
            indices=indices,
            values=data_model.laplacian.data[offdiag_mask], 
            dense_shape=[data_model.n_nodes, data_model.n_nodes]
        ) # [W, W] sparse

        # Set up spatial smoothing parameter - infer the log so always positive
        ak_init = kwargs.get("ak", 1e-5)
        if  kwargs.get("infer_ak", True):
            self.log_ak = tf.Variable(np.log(ak_init), name="log_ak", dtype=tf.float32)
        else:
            self.log_ak = tf.constant(np.log(ak_init), name="log_ak", dtype=tf.float32)
        self.ak = self.log_tf(tf.exp(self.log_ak, name="ak"))

        # For the spatial mean we essentially need the (weighted) average of 
        # nearest neighbour mean values. This does not involve the current posterior
        # mean at the voxel itself!
        # This is the equivalent of ApplyToMVN in Fabber
        node_mean = self.log_tf(tf.expand_dims(post.mean[:, idx], 1), name="node_mean", force=False, shape=True) # [W]
        node_nn_total_weight = self.log_tf(tf.sparse_reduce_sum(laplacian_nodiag, axis=1), name="node_weight", force=False, shape=True) # [W]
        spatial_mean = self.log_tf(tf.sparse_tensor_dense_matmul(laplacian_nodiag, node_mean), name="matmul", force=False, shape=True)
        spatial_mean = self.log_tf(tf.squeeze(spatial_mean, 1), name="matmul2", force=False, shape=True)
        spatial_mean = self.log_tf(spatial_mean / node_nn_total_weight, name="spmean", force=False, shape=True)
        spatial_prec = node_nn_total_weight * self.ak

        self.variance = self.log_tf(1 / (1/init_variance + spatial_prec), name="spvar", force=False, shape=True)
        self.mean = self.log_tf(self.variance * spatial_prec * spatial_mean, name="spmean2", force=False, shape=True)

    def free_energy(self):
        # Gamma prior if we care
        q1, q2 = 1, 100
        F = ((q1-1) * self.log_ak - self.ak / q2)
        print("F=", F)
        return F

class ARDPrior(ParameterPrior):
    """
    Automatic Relevence Detection prior
    """

    def __init__(self, data_model, idx, init_variance, **kwargs):
        """
        """
        LogBase.__init__(self)
        self.fixed_var = init_variance
        self.idx = idx
        nn = data_model.n_nodes
        default = np.full((nn, ), np.log(1e-12))

        # Set up inferred precision parameter phi - infer the log so always positive
        self.log_phi = tf.Variable(default, name="log_phi", dtype=tf.float32)
        self.phi = self.log_tf(tf.exp(self.log_phi, name="phi"))

        # FIXME hack to make phi get printed after each iteration using code written for
        # the MRF spatial prior
        self.ak = tf.reduce_mean(self.phi)

        self.variance = 1/self.phi
        self.mean = tf.fill((nn,), tf.constant(0.0, dtype=tf.float32))

    def get_updates(self, post):
        """
        Update the ARD precision when using analytic update mode

        See Appendix 4 of Fabber paper

        :param post: Current posterior
        :return: Sequence of tuples: (variable to update, tensor to update from)
        """
        new_var = tf.square(post.mean[:, self.idx]) + post.cov[:, self.idx, self.idx]
        return [(self.log_phi, tf.log(1/new_var),)]

    def free_energy(self):
        # See appendix D of Fabber paper
        # Tends to force phi -> 0 - but this means high prior variance which has cost?
        b = 2 * self.phi
        F = -1.5 * (tf.math.log(b) + tf.math.digamma(0.5)) - 0.5*(1 + tf.math.log(b)) - tf.math.lgamma(0.5)
        return F

class NoisePrior(LogBase):
    """
    Prior for the noise distribution

    :attr s: Noise gamma distribution prior scale parameter [V] or constant
    :attr c: Noise gamma distribution prior shape parameter [V] or constant
    """
    def __init__(self, c, s):
        """
        :param shape Prior noise shape parameter [V] or constant
        :param scale Prior noise scale parameter [V] or constant
        """
        LogBase.__init__(self)
        self.s = tf.constant(s, dtype=tf.float32)
        self.c = tf.constant(c, dtype=tf.float32)

    def mean_prec(self):
        return self.c*self.s, 1/(self.s*self.s*self.c)

class MVNPrior(LogBase):
    """
    Combined MVN prior for all parameters

    Note that this prior does not include prior covariance between parameters
    - the covariance matrix is always diagonal. There is no theoretical reason
    why we can't have prior covariance but in practice most prior knowledge
    is in terms of normal ranges for individual parameters rather than 
    relationships between them.

    :attr mean: Prior mean [W, P] or [1, P]
    :attr var: Prior variances [W, P] or [1, P]
    :attr cov: Prior covariance matrix (always diagonal) [W, P, P] or [1, P, P]
    :attr prec: Prior precision matrix (always diagonal) [W, P, P] or [1, P, P]
    """
    def __init__(self, param_priors, noise_prior):
        """
        :param param_priors: Sequence of parameter priors
        :param noise_prior: Noise prior 
        """
        LogBase.__init__(self)

        self.mean = tf.stack([p.mean for p in param_priors], axis=1)
        self.var = tf.stack([p.variance for p in param_priors], axis=1)
        self.cov = tf.linalg.diag(self.var)
        self.prec = tf.linalg.inv(self.cov)
