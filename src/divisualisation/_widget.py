"""Optional napari dock widget for divisualisation.

``SpacetimeWidget`` (Plugins -> divisualisation) offers two mutually exclusive
workflows via two toggle switches sharing one lift-amount slider:

- **Lift all tracks layers**: lift every tracks layer into the 3D time->z
  "spacetime" view, keeping each layer's own coloring.
- **Divisualisation**: declare the GT / predicted / FN-edge / FP-edge tracks
  layers by role (name-guessed), optionally compute the CTC edge errors from the
  GT/pred tracks + segmentation labels, and lift with the error-view look.

Additive: the functional API works without it.
"""

import napari
from magicgui.backends._qtpy.widgets import QBaseValueWidget
from magicgui.widgets import ComboBox, Container, FloatSlider, PushButton
from magicgui.widgets.bases import ValueWidget
from qtpy.QtCore import QTimer
from superqt import QToggleSwitch

from .lift import ROLES, SpacetimeLift, _is_tracks


class _ToggleSwitchBackend(QBaseValueWidget):
    def __init__(self, **kwargs):
        super().__init__(QToggleSwitch, "isChecked", "setChecked", "toggled", **kwargs)


class ToggleSwitch(ValueWidget):
    """A magicgui on/off value widget rendered as a sliding toggle switch."""

    def __init__(self, **kwargs):
        super().__init__(widget_type=_ToggleSwitchBackend, **kwargs)


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
_LABELS_HINTS = {"gt": ("gt", "ground"), "pred": ("pred", "res")}
_NONE_CHOICE = "—"  # blank / skip this role


class SpacetimeWidget(Container):
    """Two mutually exclusive lift workflows sharing one lift-amount slider."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._lift = SpacetimeLift(viewer)

        # Two mutually exclusive toggles + one shared lift slider.
        self._lift_all = ToggleSwitch(value=False, label="Lift all tracks layers")
        self._lift_errors = ToggleSwitch(value=False, label="Divisualisation")
        self._lift_amount = FloatSlider(value=12, min=0, max=99, label="lift")

        # Error-view controls (shown only while the errors toggle is on).
        self._role_combos = {
            role: ComboBox(label=_ROLE_LABELS[role], choices=[_NONE_CHOICE])
            for role in ROLES
        }
        self._gt_labels = ComboBox(label="GT labels", choices=[_NONE_CHOICE])
        self._pred_labels = ComboBox(label="pred labels", choices=[_NONE_CHOICE])
        self._compute_btn = PushButton(text="Compute errors")
        self._error_controls = [
            *self._role_combos.values(),
            self._gt_labels,
            self._pred_labels,
            self._compute_btn,
        ]

        # Guard so programmatic combo updates (name-guessing) don't trigger the
        # re-lift handler.
        self._refreshing = False

        self._lift_all.changed.connect(self._on_toggle_all)
        self._lift_errors.changed.connect(self._on_toggle_errors)
        self._lift_amount.changed.connect(self._on_lift_amount)
        self._compute_btn.changed.connect(self._on_compute)
        for combo in (*self._role_combos.values(), self._gt_labels, self._pred_labels):
            combo.changed.connect(self._on_role_changed)

        self.extend([
            self._lift_all,
            self._lift_errors,
            self._lift_amount,
            *self._error_controls,
        ])

        self._viewer.layers.events.inserted.connect(self._refresh_choices)
        self._viewer.layers.events.removed.connect(self._refresh_choices)
        self._refresh_choices()
        self._update_error_controls_visibility()
        # add_dock_widget resets combo values right after __init__; re-guess on
        # the next event-loop tick so the dropdowns are usable immediately.
        QTimer.singleShot(0, self._refresh_choices)

    # --- toggles ------------------------------------------------------------

    def _on_toggle_all(self, *_):
        if self._lift_all.value:
            if self._lift_errors.value:  # enforce mutual exclusivity
                self._lift_errors.value = False
            self._apply_lift(self._all_tracks_target())
        elif not self._lift_errors.value:
            self._revert_lift()

    def _on_toggle_errors(self, *_):
        if self._lift_errors.value:
            if self._lift_all.value:  # enforce mutual exclusivity
                self._lift_all.value = False
            self._apply_lift(self._roles_target())
        elif not self._lift_all.value:
            self._revert_lift()
        self._update_error_controls_visibility()

    def _apply_lift(self, target):
        # Revert any active lift first so switching modes rebuilds cleanly.
        if self._lift.applied:
            self._lift.revert()
        self._lift.time_scale = self._lift_amount.value
        self._lift.apply(target)
        self._update_error_controls_visibility()

    def _revert_lift(self):
        if self._lift.applied:
            self._lift.revert()
        self._refresh_choices()
        self._update_error_controls_visibility()

    def _on_lift_amount(self, *_):
        self._lift.time_scale = self._lift_amount.value

    def _on_role_changed(self, *_):
        # A role/label dropdown changed. If the Divisualisation lift is active,
        # re-apply it live with the new selection. Ignore programmatic updates
        # from name-guessing (_refresh_choices).
        if self._refreshing:
            return
        if self._lift_errors.value and self._lift.applied:
            self._apply_lift(self._roles_target())

    def _update_error_controls_visibility(self):
        # Error controls are visible and editable whenever the Divisualisation
        # workflow is on; changing a role/label dropdown re-applies the lift
        # live (see _on_role_changed). Compute is likewise always clickable.
        visible = self._lift_errors.value
        for w in self._error_controls:
            w.visible = visible
            w.enabled = visible

    # --- layer discovery ----------------------------------------------------

    def _all_tracks_target(self):
        # Every tracks layer; a plain list keeps each layer's own coloring.
        return [layer.name for layer in self._viewer.layers if _is_tracks(layer)]

    def _roles_target(self):
        return {
            role: combo.value
            for role, combo in self._role_combos.items()
            if combo.value and combo.value != _NONE_CHOICE
        }

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
        if self._lift.applied:
            return  # don't reshuffle while a lift is active
        self._refreshing = True
        try:
            self._refresh_choices_impl()
        finally:
            self._refreshing = False

    def _refresh_choices_impl(self):
        track_names = self._track_layer_names()
        track_choices = [_NONE_CHOICE, *track_names]
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

        label_names = self._labels_layer_names()
        label_choices = [_NONE_CHOICE, *label_names]
        used: set[str] = set()
        for combo, key in ((self._gt_labels, "gt"), (self._pred_labels, "pred")):
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
