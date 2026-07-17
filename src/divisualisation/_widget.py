"""Optional napari dock widgets for divisualisation.

- ``ErrorToggleWidget`` bulk-shows/hides the edge-error track layers.
- ``SpacetimeWidget`` lifts selected tracks layers into the 3D time->z
  "spacetime" view on demand, with a live lift-amount slider, and reverts back
  to the flat 2D view when toggled off.

Both are additive: the functional API works without them.
"""

import napari
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FloatSlider,
    PushButton,
    RadioButtons,
)

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


_MODE_TRACKS = "Visualize tracks"
_MODE_ERRORS = "GT / pred errors"

# Name hints for the two segmentation-labels dropdowns used to compute errors.
_LABELS_HINTS = {"gt": ("gt", "ground"), "pred": ("pred", "res")}


class SpacetimeWidget(Container):
    """Two-workflow spacetime plugin.

    Mode "Visualize tracks": lift one or more tracks layers into the 3D
    time->z view, keeping their own (e.g. random per-track) coloring.

    Mode "GT / pred errors": declare the GT / predicted / FN-edge / FP-edge
    tracks layers by role (prefilled by name-guessing, all optional), optionally
    compute the CTC edge errors from the GT/pred tracks + segmentation labels,
    and lift with the traditional error-view look (GT->Greens, pred->Wistia,
    errors->cool). Toggling the lift off restores every layer's own settings.
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._lift = SpacetimeLift(viewer)

        self._mode = RadioButtons(
            choices=[_MODE_TRACKS, _MODE_ERRORS], value=_MODE_TRACKS, label="mode"
        )
        self._enabled = CheckBox(value=False, text="Spacetime lift")
        self._lift_amount = FloatSlider(value=12, min=0, max=40, label="lift")
        # Role dropdowns (tracks) for the error workflow.
        self._role_combos = {
            role: ComboBox(label=_ROLE_LABELS[role], choices=[_NONE_CHOICE])
            for role in ROLES
        }
        # A single tracks dropdown for the simple visualize workflow.
        self._viz_tracks = ComboBox(label="tracks", choices=[_NONE_CHOICE])
        # Segmentation-labels dropdowns + button for computing errors.
        self._gt_labels = ComboBox(label="GT labels", choices=[_NONE_CHOICE])
        self._pred_labels = ComboBox(label="pred labels", choices=[_NONE_CHOICE])
        self._compute_btn = PushButton(text="Compute errors")

        self._mode.changed.connect(self._on_mode)
        self._enabled.changed.connect(self._on_toggle)
        self._lift_amount.changed.connect(self._on_lift_amount)
        self._compute_btn.changed.connect(self._on_compute)
        self.extend([
            self._mode,
            self._enabled,
            self._lift_amount,
            self._viz_tracks,
            *self._role_combos.values(),
            self._gt_labels,
            self._pred_labels,
            self._compute_btn,
        ])

        self._viewer.layers.events.inserted.connect(self._refresh_choices)
        self._viewer.layers.events.removed.connect(self._refresh_choices)
        self._refresh_choices()
        self._on_mode()

    # --- layer discovery ----------------------------------------------------

    def _track_layer_names(self):
        return [layer.name for layer in self._viewer.layers if _is_tracks(layer)]

    def _labels_layer_names(self):
        return [
            layer.name
            for layer in self._viewer.layers
            if type(layer).__name__ == "Labels"
        ]

    @staticmethod
    def _guess(names, hints, already):
        for name in names:
            if name in already:
                continue
            if any(hint in name.lower() for hint in hints):
                return name
        return _NONE_CHOICE

    def _refresh_choices(self, *_):
        if self._enabled.value:
            return  # don't reshuffle while a lift is active
        track_names = self._track_layer_names()
        track_choices = [_NONE_CHOICE, *track_names]
        # Track roles: keep any explicit choice that is still valid, and only
        # fill blanks by name-guessing. (add_dock_widget resets combo values, so
        # we cannot rely on guesses made in __init__ sticking.)
        assigned = {
            c.value
            for c in self._role_combos.values()
            if c.value not in (None, _NONE_CHOICE)
        }
        for role in ROLES:
            combo = self._role_combos[role]
            keep = combo.value if combo.value in track_names else _NONE_CHOICE
            combo.choices = track_choices
            if keep != _NONE_CHOICE:
                combo.value = keep
                continue
            guess = self._guess(track_names, _ROLE_NAME_HINTS[role], assigned)
            combo.value = guess
            if guess != _NONE_CHOICE:
                assigned.add(guess)

        keep_viz = self._viz_tracks.value
        self._viz_tracks.choices = track_choices
        if keep_viz in track_names:
            self._viz_tracks.value = keep_viz
        elif track_names:
            self._viz_tracks.value = track_names[0]

        label_names = self._labels_layer_names()
        label_choices = [_NONE_CHOICE, *label_names]
        used: set[str] = set()
        for combo, key in (
            (self._gt_labels, "gt"),
            (self._pred_labels, "pred"),
        ):
            keep = combo.value if combo.value in label_names else _NONE_CHOICE
            combo.choices = label_choices
            if keep != _NONE_CHOICE:
                combo.value = keep
                used.add(keep)
                continue
            guess = self._guess(label_names, _LABELS_HINTS[key], used)
            combo.value = guess
            if guess != _NONE_CHOICE:
                used.add(guess)

    # --- mode switching -----------------------------------------------------

    def _errors_mode(self):
        return self._mode.value == _MODE_ERRORS

    def _on_mode(self, *_):
        self._refresh_choices()
        errors = self._errors_mode()
        for combo in self._role_combos.values():
            combo.visible = errors
        self._gt_labels.visible = errors
        self._pred_labels.visible = errors
        self._compute_btn.visible = errors
        self._viz_tracks.visible = not errors

    # --- lift ---------------------------------------------------------------

    def _layer_roles(self):
        """Roles -> layer names for the active mode (drives the lift)."""
        if self._errors_mode():
            return {
                role: combo.value
                for role, combo in self._role_combos.items()
                if combo.value and combo.value != _NONE_CHOICE
            }
        # Visualize mode: lift the one chosen tracks layer with no role look, so
        # its own (e.g. random per-track) coloring is kept.
        name = self._viz_tracks.value
        if name and name != _NONE_CHOICE:
            return [name]
        return []

    def _set_inputs_enabled(self, enabled):
        self._mode.enabled = enabled
        self._viz_tracks.enabled = enabled
        for combo in self._role_combos.values():
            combo.enabled = enabled
        self._gt_labels.enabled = enabled
        self._pred_labels.enabled = enabled
        self._compute_btn.enabled = enabled

    def _on_toggle(self, *_):
        if self._enabled.value:
            self._lift.time_scale = self._lift_amount.value
            self._lift.apply(self._layer_roles())
            self._set_inputs_enabled(False)
        else:
            self._lift.revert()
            self._set_inputs_enabled(True)
            self._refresh_choices()

    def _on_lift_amount(self, *_):
        self._lift.time_scale = self._lift_amount.value

    # --- compute errors -----------------------------------------------------

    def _on_compute(self, *_):
        from .errors import compute_edge_errors_from_layers

        layers = self._viewer.layers
        gt_tracks = self._role_combos["gt"].value
        pred_tracks = self._role_combos["pred"].value
        gt_labels = self._gt_labels.value
        pred_labels = self._pred_labels.value
        missing = [
            label
            for label, value in (
                ("GT tracks", gt_tracks),
                ("predicted tracks", pred_tracks),
                ("GT labels", gt_labels),
                ("pred labels", pred_labels),
            )
            if not value or value == _NONE_CHOICE
        ]
        if missing:
            raise ValueError(
                "Computing errors needs " + ", ".join(missing) + " to be set."
            )
        compute_edge_errors_from_layers(
            self._viewer,
            layers[gt_tracks],
            layers[gt_labels],
            layers[pred_tracks],
            layers[pred_labels],
        )
        self._refresh_choices()
