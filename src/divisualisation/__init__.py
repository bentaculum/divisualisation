from importlib.metadata import version

try:
    __version__ = version("divisualisation")
except Exception:
    __version__ = "unknown"
