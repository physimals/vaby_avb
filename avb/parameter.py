"""
SVB - Model parameters

This module defines a set of classes of model parameters.

The factory methods which create priors/posteriors can
make use of the instance class to create the appropriate
type of vertexwise prior/posterior
"""
from .utils import LogBase
from . import dist

def get_parameter(name, **kwargs):
    """
    Factory method to create an instance of a parameter
    """
    custom_kwargs = kwargs.pop("param_overrides", {}).get(name, {})
    kwargs.update(custom_kwargs)

    desc = kwargs.get("desc", "No description given")
    prior_dist = dist.get_dist(prefix="prior", **kwargs)
    prior_type = kwargs.get("prior_type", "N")
    post_dist = dist.get_dist(prefix="prior", **kwargs)
    post_init = kwargs.get("post_init", None)

    return Parameter(name, desc=desc, prior_dist=prior_dist, prior_type=prior_type, post_dist=post_dist, post_init=post_init)

class Parameter(LogBase):
    """
    A standard model parameter
    """

    def __init__(self, name, **kwargs):
        """
        Constructor

        :param name: Parameter name
        :param prior_dist: Dist instance giving the parameter's prior distribution
        :param post_dist: Dist instance giving the parameter's initial posterio distribution
        :param desc: Optional parameter description

        Keyword arguments (optional):
         - ``mean_init`` Initial value for the posterior mean either as a numeric
                         value or a callable which takes the parameters t, data, param_name
         - ``log_var_init`` Initial value for the posterior log variance either as a numeric
                            value or a callable which takes the parameters t, data, param_name
         - ``param_overrides`` Dictionary keyed by parameter name. Value should be dictionary
                               of keyword arguments which will override those defined as
                               existing keyword arguments
        """
        LogBase.__init__(self)

        custom_kwargs = kwargs.pop("param_overrides", {}).get(name, {})
        kwargs.update(custom_kwargs)

        self.name = name
        self.desc = kwargs.get("desc", "No description given")
        self.prior_dist = kwargs.get("prior_dist")
        self.prior_type = kwargs.get("prior_type", "N")
        self.post_dist = kwargs.get("post_dist")
        self.post_init = kwargs.get("post_init", None)

    def __str__(self):
        return "Parameter: %s" % self.name
