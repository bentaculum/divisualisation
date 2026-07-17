"""Back-compat shim.

The spacetime renderer moved to ``divisualisation.spacetime``. Import
``Divisualisation`` from there directly; this module re-exports it so existing
code and scripts keep working.
"""

# _animate_with_gl_context is re-exported (not in __all__) for back-compat with
# any code that imported it from this module before the move to spacetime.py.
from .spacetime import Divisualisation, _animate_with_gl_context  # noqa: F401

__all__ = ["Divisualisation"]
