from importlib.metadata import version

try:
    __version__ = version("divisualisation")
except Exception:
    __version__ = "unknown"

from .errors import add_edge_error_tracks
from .lift import SpacetimeLift

__all__ = ["SpacetimeLift", "add_edge_error_tracks"]
