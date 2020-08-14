try:
    from ._version import __version__, __timestamp__
except ImportError:
    __version__ = "Unknown version"
    __timestamp__ = "Unknown timestamp"

from .avb import Avb
from .data_model import DataModel
from .model import Model, get_model_class

__all__ = [   "__version__",
    "__timestamp__",
    "Avb",
    "DataModel",
    "Model",
    "get_model_class",
]
