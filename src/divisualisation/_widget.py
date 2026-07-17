"""Optional napari dock widgets for divisualisation.

- ``ErrorToggleWidget`` bulk-shows/hides the edge-error track layers.
- ``SpacetimeWidget`` lifts selected tracks layers into the 3D time->z
  "spacetime" view on demand, with a live lift-amount slider, and reverts back
  to the flat 2D view when toggled off.

Both are additive: the functional API works without them.
"""

import napari
from magicgui.widgets import CheckBox, Container, FloatSlider, Select

from .errors import DEFAULT_ERROR_GRAPHS
from .lift import SpacetimeLift, _is_tracks

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


class SpacetimeWidget(Container):
    """Toggle the 3D time->z spacetime lift on selected tracks layers."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._lift = SpacetimeLift(viewer)

        self._enabled = CheckBox(value=False, text="Spacetime lift")
        self._lift_amount = FloatSlider(value=12, min=0, max=40, label="lift")
        self._layers = Select(label="tracks", choices=self._track_layer_names)

        self._enabled.changed.connect(self._on_toggle)
        self._lift_amount.changed.connect(self._on_lift_amount)
        self.extend([self._enabled, self._lift_amount, self._layers])

        # Refresh the selectable tracks layers as layers come and go.
        self._viewer.layers.events.inserted.connect(self._refresh_choices)
        self._viewer.layers.events.removed.connect(self._refresh_choices)
        self._refresh_choices()

    def _track_layer_names(self, _widget=None):
        return [layer.name for layer in self._viewer.layers if _is_tracks(layer)]

    def _refresh_choices(self, *_):
        names = self._track_layer_names()
        self._layers.choices = names
        # Default to all tracks layers selected; the user can deselect some.
        if not self._enabled.value:
            self._layers.value = names

    def _on_toggle(self, *_):
        if self._enabled.value:
            self._lift.time_scale = self._lift_amount.value
            self._lift.apply(self._layers.value)
            self._layers.enabled = False
        else:
            self._lift.revert()
            self._layers.enabled = True
            self._refresh_choices()

    def _on_lift_amount(self, *_):
        self._lift.time_scale = self._lift_amount.value
