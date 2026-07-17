"""Optional napari dock widgets for divisualisation.

- ``ErrorToggleWidget`` bulk-shows/hides the edge-error track layers.
- ``SpacetimeWidget`` lifts selected tracks layers into the 3D time->z
  "spacetime" view on demand, with a live lift-amount slider, and reverts back
  to the flat 2D view when toggled off.

Both are additive: the functional API works without them.
"""

import napari
from magicgui.widgets import CheckBox, ComboBox, Container, FloatSlider

from .errors import DEFAULT_ERROR_GRAPHS
from .lift import ROLES, SpacetimeLift, _is_tracks

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


# One dropdown per lift role, with a human label and substrings used to guess
# which track layer fills the role from its name.
_ROLE_LABELS = {
    "gt": "GT tracks",
    "pred": "predicted tracks",
    "fn_edges": "FN edges",
    "fp_edges": "FP edges",
}
_ROLE_NAME_HINTS = {
    "gt": ("gt", "ground"),
    "pred": ("pred",),
    "fn_edges": ("fn", "false_neg", "false neg", "ctc_fn"),
    "fp_edges": ("fp", "false_pos", "false pos", "ctc_fp"),
}
_NONE_CHOICE = "—"  # blank / skip this role


class SpacetimeWidget(Container):
    """Toggle the 3D time->z spacetime lift, declaring the tracks layers by role.

    Each of the four roles (GT / predicted / FN edges / FP edges) has its own
    dropdown, prefilled by name-based guessing and set to blank when no match is
    found. Every role is optional; only the ones pointing at a real tracks layer
    are lifted. On toggle-on each declared layer takes the "error view" look for
    its role (colormap, tail length/width, blending, opacity); on toggle-off the
    layers' original settings are restored.
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._lift = SpacetimeLift(viewer)

        self._enabled = CheckBox(value=False, text="Spacetime lift")
        self._lift_amount = FloatSlider(value=12, min=0, max=40, label="lift")
        self._role_combos = {
            role: ComboBox(label=_ROLE_LABELS[role], choices=[_NONE_CHOICE])
            for role in ROLES
        }

        self._enabled.changed.connect(self._on_toggle)
        self._lift_amount.changed.connect(self._on_lift_amount)
        self.extend([self._enabled, self._lift_amount, *self._role_combos.values()])

        # Keep the dropdowns' choices in sync as layers come and go.
        self._viewer.layers.events.inserted.connect(self._refresh_choices)
        self._viewer.layers.events.removed.connect(self._refresh_choices)
        self._refresh_choices()

    def _track_layer_names(self):
        return [layer.name for layer in self._viewer.layers if _is_tracks(layer)]

    def _guess(self, role, names, already):
        """Guess the layer name for a role from name substrings; else blank."""
        for name in names:
            if name in already:
                continue
            low = name.lower()
            if any(hint in low for hint in _ROLE_NAME_HINTS[role]):
                return name
        return _NONE_CHOICE

    def _refresh_choices(self, *_):
        if self._enabled.value:
            return  # don't reshuffle while a lift is active
        names = self._track_layer_names()
        choices = [_NONE_CHOICE, *names]
        assigned: set[str] = set()
        for role in ROLES:
            combo = self._role_combos[role]
            combo.choices = choices
            guess = self._guess(role, names, assigned)
            combo.value = guess
            if guess != _NONE_CHOICE:
                assigned.add(guess)

    def _layer_roles(self):
        return {
            role: combo.value
            for role, combo in self._role_combos.items()
            if combo.value and combo.value != _NONE_CHOICE
        }

    def _set_combos_enabled(self, enabled):
        for combo in self._role_combos.values():
            combo.enabled = enabled

    def _on_toggle(self, *_):
        if self._enabled.value:
            self._lift.time_scale = self._lift_amount.value
            self._lift.apply(self._layer_roles())
            self._set_combos_enabled(False)
        else:
            self._lift.revert()
            self._set_combos_enabled(True)
            self._refresh_choices()

    def _on_lift_amount(self, *_):
        self._lift.time_scale = self._lift_amount.value
