"""
VABY_AVB
"""
try:
    from ._version import __version__, __timestamp__
except ImportError:
    __version__ = "Unknown version"
    __timestamp__ = "Unknown timestamp"

from .avb import Avb

__all__ = [
    "__version__",
    "__timestamp__",
    "Avb",
]
