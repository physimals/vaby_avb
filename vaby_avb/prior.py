"""
AVB - Priors for model and noise parameters
"""

import numpy as np
import tensorflow as tf

from vaby.utils import LogBase
from vaby.dist import Normal

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
    :attr var: Variance [W] or [1]
    """

    def free_energy(self):
        """
        :return: Free energy contribution associated with this prior, if required. [W] or constant
        """
        return 0

    def update(self, avb):
        """
        Update prior in analytic update mode

        Required for adaptive priors, e.g. spatial, ARD. This method should
        use tf.assign to update any tf.Variable instances that control
        it's operation, and also update any dependency tensors.

        :param avb: Avb object
        """
        pass

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
        self.var = tf.constant(variance, dtype=np.float32)

class MRFSpatialPrior(ParameterPrior):
    """
    Spatial prior based on a Markov Random Field
    """

    def __init__(self, data_model, post, idx, init_variance, **kwargs):
        """
        """
        LogBase.__init__(self)
        self.idx = idx
        self.n_nodes = data_model.n_nodes

        # Laplacian matrix with diagonal zeroed
        indices=np.array([
                data_model.laplacian.row, 
                data_model.laplacian.col
        ]).T
        self.laplacian = tf.SparseTensor(
            indices=indices,
            values=data_model.laplacian.data, 
            dense_shape=[data_model.n_nodes, data_model.n_nodes]
        ) # [W, W] sparse

        # Laplacian matrix with diagonal zeroed
        offdiag_mask = data_model.laplacian.row != data_model.laplacian.col
        indices=np.array([
                data_model.laplacian.row[offdiag_mask], 
                data_model.laplacian.col[offdiag_mask]
        ]).T
        self.laplacian_nodiag = tf.SparseTensor(
            indices=indices,
            values=data_model.laplacian.data[offdiag_mask], 
            dense_shape=[data_model.n_nodes, data_model.n_nodes]
        ) # [W, W] sparse

        # Diagonal of Laplacian matrix [W]
        self.laplacian_diagonal = tf.constant(-data_model.laplacian.diagonal(), dtype=tf.float32)

        # Set up spatial smoothing parameter - infer the log so always positive
        ak_init = kwargs.get("ak", 1e-5)
        if kwargs.get("infer_ak", True):
            self.log_ak = tf.Variable(np.log(ak_init), name="log_ak", dtype=tf.float32)
        else:
            self.log_ak = tf.constant(np.log(ak_init), name="log_ak", dtype=tf.float32)

    def _update_deps(self):
        self.ak = tf.exp(self.log_ak)

        # For the spatial mean we essentially need the (weighted) average of 
        # nearest neighbour mean values. This does not involve the current posterior
        # mean at the voxel itself!
        # This is the equivalent of ApplyToMVN in Fabber
        node_mean = tf.expand_dims(post.mean[:, idx], 1) # [W]
        node_nn_total_weight = tf.sparse.reduce_sum(self.laplacian_nodiag, axis=1) # [W]
        spatial_mean = tf.sparse.sparse_dense_matmul(self.laplacian_nodiag, node_mean)
        spatial_mean = tf.squeeze(spatial_mean, 1)
        spatial_mean = spatial_mean / node_nn_total_weight
        spatial_prec = node_nn_total_weight * self.ak

        self.var = 1 / (1/init_variance + spatial_prec)
        self.mean = self.var * spatial_prec * spatial_mean

    def update(self, AVB):
        """
        Update the global spatial precision when using analytic update mode

        Penny et al 2005 Fig 4 (Spatial Precisions)

        This is the equivalent of Fabber's CalculateAk

        :param post: Current posterior
        :return: Sequence of tuples: (variable to update, tensor to update from)
        """
        # Posterior variance for parameter [W]
        sigmaK = avb.post.cov[:, self.idx, self.idx]

        # Posterior mean for parameter [W]
        wK = tf.expand_dims(avb.post.mean[:, self.idx], 1)

        # First term for gk:   Tr(sigmaK*S'*S) (note our Laplacian is S'*S directly)
        trace_term = tf.reduce_sum(sigmaK * self.laplacian_diagonal)

        # Contribution from nearest neighbours - sum of differences
        # between voxel mean and neighbour mean multipled by the
        # neighbour weighting [W]
        SwK = -tf.sparse_tensor_dense_matmul(self.laplacian, wK)

        # For MRF spatial prior the spatial precision matrix S'S is handled
        # directly so we are effectively calculating wK * D * wK where
        # D is the spatial matrix.
        # This is second term for gk:  wK'S'SwK using elementwise multiplication
        term2 = tf.reduce_sum(SwK * wK)

        #self.log.warn("MRFSpatialPrior::Calculate aK %i: trace_term=%e, term2=%e", self.idx, trace_term, term2)

        # Fig 4 in Penny (2005) update equations for gK, hK and aK
        # Following Penny, prior on aK is a relatively uninformative gamma distribution with 
        # q1 = 10 (1/q1 = 0.1) and q2 = 1.0
        gk = 1 / (0.5 * trace_term + 0.5 * term2 + 0.1)
        hK = self.n_nodes * 0.5 + 1.0
        aK = gk * hK

        # Checks below are from Fabber - unsure if required
        #
        #if aK < 1e-50:
        #    // Don't let aK get too small
        #    LOG << "SpatialPrior::Calculate aK " << m_idx << ": was " << aK << endl;
        #    WARN_ONCE("SpatialPrior::Calculate aK - value was tiny - fixing to 1e-50");
        #    aK = 1e-50;

        # Controls the speed of changes to aK - unsure whether this is useful or not but
        # it is only used if m_spatial_speed is given as an option

        # FIXME default for m_spatial_speed is -1 so code below will always be executed - 
        # harmless but potentially confusing
        #aKMax = aK * m_spatial_speed;
        #if aKMax < 0.5:
        #    # totally the wrong scaling.. oh well
        #    aKMax = 0.5

        #if m_spatial_speed > 0 and aK > aKMax:
        #    self.log.warn("MRFSpatialPrior::Calculate aK %i: Rate-limiting the increase on aK: was %e, now %e", self.idx, ak, akMax)
        #    aK = aKMax

        #self.log.info("MRFSpatialPrior::Calculate aK %i: New aK: %e", self.idx, ak)
        self.log_ak.assign(tf.log(aK))
        self._update_deps()

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

    def _update_deps(self):
        self.phi = tf.clip_by_value(tf.exp(self.log_phi), 0, 1e6)

        # FIXME hack to make phi get printed after each iteration using code written for
        # the MRF spatial prior
        self.ak = tf.reduce_mean(self.phi)

        self.var = 1/self.phi
        self.mean = tf.fill((nn,), tf.constant(0.0, dtype=tf.float32))

    def update(self, avb):
        """
        Update the ARD precision when using analytic update mode

        See Appendix 4 of Fabber paper

        :param post: Current posterior
        :return: Sequence of tuples: (variable to update, tensor to update from)
        """
        mean = post.mean[:, self.idx]
        var = post.cov[:, self.idx, self.idx]
        new_var = tf.square(mean) + var
        self.log_phi.assign(tf.log(1/new_var))
        self._update_deps()

    def free_energy(self):
        # See appendix D of Fabber paper
        # Tends to force phi -> 0 - but this means high prior variance which has cost?
        b = 2 * self.phi
        t1 = -1.5 * (tf.math.log(b) + tf.math.digamma(0.5)) + 0.5*(1 + tf.math.log(b))
        t2 = -tf.math.lgamma(0.5)
        return - (t1 + t2)

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
        self.mean = self.c*self.s
        self.var = self.s*self.s*self.c
        self.prec = 1/self.var

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
        self.var = tf.stack([p.var for p in param_priors], axis=1)
        self.cov = tf.linalg.diag(self.var)
        self.prec = tf.linalg.inv(self.cov)
