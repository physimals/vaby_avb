"""
General utility functions
"""
import logging

def ValueList(value_type):
    """
    Class used with argparse for options which can be given as a comma separated list
    """
    def _call(value):
        return [value_type(v) for v in value.replace(",", " ").split()]
    return _call

class LogBase(object):
    """
    Base class that provides a named log
    """
    def __init__(self, **kwargs):
        self.log = logging.getLogger(type(self).__name__)

    def log_tf(self, tensor, *args, **kwargs):
        self.log.debug("%s: shape=%s", (kwargs.get("name", "tensor"), tensor.shape))
        return tensor
