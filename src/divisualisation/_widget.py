"""Optional napari dock widget to toggle divisualisation error layers at once.

The functional API (:func:`divisualisation.add_edge_error_tracks`) works without
this widget, and every error layer it adds already has napari's built-in
per-layer eye-icon toggle. This widget only adds a one-click bulk toggle for all
error layers together.
"""

import napari
from magicgui.widgets import CheckBox, Container

from .errors import DEFAULT_ERROR_GRAPHS

ERROR_LAYER_NAMES = frozenset(str(flag.value) for flag in DEFAULT_ERROR_GRAPHS)


class ErrorToggleWidget(Container):
    """Show/hide all divisualisation edge-error track layers at once."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        self._show_errors = CheckBox(value=True, text="Show edge errors")
        self._show_errors.changed.connect(self._apply_visibility)
        self.append(self._show_errors)

        # Keep newly added error layers in sync with the checkbox.
        self._viewer.layers.events.inserted.connect(self._apply_visibility)

    def _apply_visibility(self, *_):
        visible = bool(self._show_errors.value)
        for layer in self._viewer.layers:
            if layer.name in ERROR_LAYER_NAMES:
                layer.visible = visible
