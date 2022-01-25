"""
VABY_AVB - Priors for model and noise parameters
"""

import numpy as np
import tensorflow as tf

from vaby.utils import LogBase, TF_DTYPE, NP_DTYPE
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

    def build(self, avb):
        pass

    def avb_update(self, avb):
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
        nn = data_model.model_space.size
        while np.array(mean).ndim < 1:
            mean = np.full((nn,), mean)
        while np.array(variance).ndim < 1:
            variance = np.full((nn,), variance)

        self.mean = tf.constant(mean, dtype=NP_DTYPE)
        self.var = tf.constant(variance, dtype=NP_DTYPE)
        self.vars = []

class MRFSpatialPrior(ParameterPrior):
    """
    Spatial prior based on a Markov Random Field
    """

    def __init__(self, data_model, post, idx, init_variance, **kwargs):
        """
        """
        LogBase.__init__(self)
        self.idx = idx
        self.n_nodes = data_model.model_space.size

        # Laplacian matrix
        self.laplacian = data_model.model_space.laplacian

        # Laplacian matrix with diagonal zeroed
        offdiag_mask = self.laplacian.indices[:, 0] != self.laplacian.indices[:, 1]
        self.laplacian_nodiag = tf.SparseTensor(
            indices=self.laplacian.indices[offdiag_mask],
            values=self.laplacian.values[offdiag_mask], 
            dense_shape=[self.n_nodes, self.n_nodes]
        ) # [W, W] sparse

        # Diagonal of Laplacian matrix [W]
        diag_mask = self.laplacian.indices[:, 0] == self.laplacian.indices[:, 1]
        self.laplacian_diagonal = -self.laplacian.values[diag_mask]

        # Set up spatial smoothing parameter - infer the log so always positive
        self.num_nodes = data_model.model_space.size
        self.num_aks = data_model.model_space.num_strucs
        self.sub_strucs = data_model.model_space.parts
        self.slices = data_model.model_space.slices
        ak_init = tf.fill([self.num_aks], kwargs.get("ak", 1e-5))
        if kwargs.get("infer_ak", True):
            self.log_ak = tf.Variable(np.log(ak_init), name="log_ak", dtype=TF_DTYPE)
            self.vars = [self.log_ak]
        else:
            self.log_ak = tf.constant(np.log(ak_init), name="log_ak", dtype=TF_DTYPE)
            self.vars = []

        self.mean = tf.fill((self.n_nodes,), 0.0)
        self.var = tf.fill((self.n_nodes,), 1/ak_init)

    def build(self, avb):
        aks_nodewise = []
        for struc_idx, struc in enumerate(self.sub_strucs):
            aks_nodewise.append(tf.fill([struc.size], tf.exp(self.log_ak[struc_idx])))
        self.ak = tf.concat(aks_nodewise, 0) # [W]

        # For the spatial mean we essentially need the (weighted) average of 
        # nearest neighbour mean values. This does not involve the current posterior
        # mean at the voxel itself!
        # This is the equivalent of ApplyToMVN in Fabber
        node_mean = tf.expand_dims(avb.post.mean[:, self.idx], 1) # [W]
        node_nn_total_weight = tf.sparse.reduce_sum(self.laplacian_nodiag, axis=1) # [W]
        spatial_mean = tf.sparse.sparse_dense_matmul(self.laplacian_nodiag, node_mean) # [W]
        spatial_mean = tf.squeeze(spatial_mean, 1)
        spatial_mean = spatial_mean / node_nn_total_weight # [W]
        spatial_prec = node_nn_total_weight * self.ak # [W]

        #self.var = 1 / (1/init_variance + spatial_prec)
        self.var = 1 / spatial_prec # [W]
        self.mean = self.var * spatial_prec * spatial_mean # [W]

    def avb_update(self, avb):
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

        # Contribution from nearest neighbours - sum of differences
        # between voxel mean and neighbour mean multipled by the
        # neighbour weighting [W]
        SwK = -tf.sparse.sparse_dense_matmul(self.laplacian, wK) # [W]

        # First term for gk:   Tr(sigmaK*S'*S) (note our Laplacian is S'*S directly)

        # For MRF spatial prior the spatial precision matrix S'S is handled
        # directly so we are effectively calculating wK * D * wK where
        # D is the spatial matrix.
        # This is second term for gk:  wK'S'SwK using elementwise multiplication
        trace_matrix = sigmaK * self.laplacian_diagonal # [W]
        term2_matrix = SwK * wK # [W]
        trace_term, term2 = [], []
        for struc_idx, struc in enumerate(self.sub_strucs):
            trace_term.append(tf.reduce_sum(trace_matrix[self.slices[struc_idx]])) # scalar
            term2.append(tf.reduce_sum(term2_matrix[self.slices[struc_idx]])) # scalar
        trace_term = tf.concat(trace_term, 0)
        term2 = tf.concat(term2, 0)

        # Fig 4 in Penny (2005) update equations for gK, hK and aK
        # Following Penny, prior on aK is a relatively uninformative gamma distribution with 
        # q1 = 10 (1/q1 = 0.1) and q2 = 1.0
        # FIXME for multiple aks, gk, hk shape [T]
        gk = 1 / (0.5 * trace_term + 0.5 * term2 + 0.1)
        hK = self.n_nodes * 0.5 + 1.0
        aK = gk * hK # [T]

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
        self.log_ak.assign(tf.reshape(tf.math.log(aK), [-1]))
        self.build(avb)

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
        self.mean = tf.fill((data_model.model_space.size,), tf.constant(0.0, dtype=TF_DTYPE))

        # Set up inferred precision parameter phi - infer the log so always positive
        default_phi = np.full((data_model.model_space.size, ), np.log(1e-12))
        self.log_phi = tf.Variable(default_phi, name="log_phi", dtype=TF_DTYPE)
        self.vars = [self.log_phi]

    def build(self, avb):
        self.phi = tf.clip_by_value(tf.exp(self.log_phi), 0, 1e6)

        # FIXME hack to make phi get printed after each iteration using code written for
        # the MRF spatial prior
        self.ak = tf.reduce_mean(self.phi)
        self.var = 1/self.phi

    def avb_update(self, avb):
        """
        Update the ARD precision when using analytic update mode

        See Appendix 4 of Fabber paper

        :param post: Current posterior
        :return: Sequence of tuples: (variable to update, tensor to update from)
        """
        mean = avb.post.mean[:, self.idx]
        var = avb.post.cov[:, self.idx, self.idx]
        new_var = tf.square(mean) + var
        self.log_phi.assign(tf.math.log(1/new_var))
        self.build(avb)

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
        self.s = tf.constant(s, dtype=TF_DTYPE)
        self.c = tf.constant(c, dtype=TF_DTYPE)
        self.mean = self.c*self.s
        self.var = self.s*self.s*self.c
        self.prec = 1/self.var
        self.vars = []

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
    def __init__(self, param_priors):
        """
        :param param_priors: Sequence of parameter priors
        :param noise_prior: Noise prior 
        """
        LogBase.__init__(self)
        self.param_priors = param_priors

    def build(self, avb):
        for prior in self.param_priors:
            prior.build(avb)
        self.mean = tf.stack([p.mean for p in self.param_priors], axis=1)
        self.var = tf.stack([p.var for p in self.param_priors], axis=1)
        self.cov = tf.linalg.diag(self.var)
        self.prec = tf.linalg.inv(self.cov)
        self.vars = sum([p.vars for p in self.param_priors], [])
