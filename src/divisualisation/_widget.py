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
        # One lift engine per toggle view, so each view keeps its own
        # layer-control settings. They share a camera store, so the camera
        # (center/zoom/angles/perspective) is shared across the two views.
        camera_store: dict = {}
        self._lift_all_engine = SpacetimeLift(viewer, camera_store=camera_store)
        self._lift_errors_engine = SpacetimeLift(viewer, camera_store=camera_store)
        # Points at the engine for the currently active toggle.
        self._lift = self._lift_all_engine

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
        for role, combo in self._role_combos.items():
            combo.changed.connect(lambda *_, r=role: self._on_role_changed(r))
        for combo in (self._gt_labels, self._pred_labels):
            combo.changed.connect(lambda *_: self._on_role_changed(None))

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
            self._apply_lift(self._lift_all_engine, self._all_tracks_target())
        elif not self._lift_errors.value:
            self._revert_lift()

    def _on_toggle_errors(self, *_):
        if self._lift_errors.value:
            if self._lift_all.value:  # enforce mutual exclusivity
                self._lift_all.value = False
            self._apply_lift(self._lift_errors_engine, self._roles_target())
            # Default the Divisualisation view to hiding the predicted-tracks
            # layer (its errors are shown by the FN/FP overlays); remember its
            # prior visibility to restore on toggle-off.
            self._hide_predicted()
        else:
            self._restore_predicted()
            if not self._lift_all.value:
                self._revert_lift()
        self._update_error_controls_visibility()

    def _hide_predicted(self):
        name = self._role_combos["pred"].value
        if name and name != _NONE_CHOICE and name in self._viewer.layers:
            layer = self._viewer.layers[name]
            self._pred_prior_visible = (name, layer.visible)
            layer.visible = False

    def _restore_predicted(self):
        prior = getattr(self, "_pred_prior_visible", None)
        if prior is not None:
            name, was_visible = prior
            if name in self._viewer.layers:
                self._viewer.layers[name].visible = was_visible
            self._pred_prior_visible = None

    def _apply_lift(self, engine, target):
        # Revert whichever engine is currently active so switching modes (and
        # engines) rebuilds cleanly; the shared camera carries over.
        for e in (self._lift_all_engine, self._lift_errors_engine):
            if e.applied:
                e.revert()
        self._lift = engine
        engine.time_scale = self._lift_amount.value
        # Always lift EVERY tracks layer. In the errors workflow ``target`` is a
        # role mapping (those get error-view colors); every other tracks layer
        # (incl. hidden, non-role ones) is lifted too, keeping its own coloring.
        engine.apply(target, extra_layers=self._all_tracks_target())
        self._update_error_controls_visibility()

    def _revert_lift(self):
        for e in (self._lift_all_engine, self._lift_errors_engine):
            if e.applied:
                e.revert()
        self._refresh_choices()
        self._update_error_controls_visibility()

    def _on_lift_amount(self, *_):
        for e in (self._lift_all_engine, self._lift_errors_engine):
            e.time_scale = self._lift_amount.value

    def _on_role_changed(self, changed_role=None):
        # A role/label dropdown changed. Ignore programmatic updates from
        # name-guessing (_refresh_choices).
        if self._refreshing:
            return
        # A layer may fill only one role: if the changed role now points at a
        # layer another role already uses, clear that other role (the just-set
        # role wins). Guard so these programmatic clears don't recurse.
        if changed_role is not None:
            value = self._role_combos[changed_role].value
            if value and value != _NONE_CHOICE:
                self._refreshing = True
                try:
                    for role, combo in self._role_combos.items():
                        if role != changed_role and combo.value == value:
                            combo.value = _NONE_CHOICE
                finally:
                    self._refreshing = False
        # Re-apply live if the Divisualisation lift is active.
        if self._lift_errors.value and self._lift_errors_engine.applied:
            self._apply_lift(self._lift_errors_engine, self._roles_target())

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
        if self._lift_all_engine.applied or self._lift_errors_engine.applied:
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
        from traccuracy import EdgeFlag

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
        # If a lift is active, the tracks layers currently hold lifted (folded)
        # coordinates. Revert so error computation runs on the original data,
        # then re-apply afterwards so the new error layers get lifted too.
        was_lifted = self._lift_errors_engine.applied
        if was_lifted:
            self._lift_errors_engine.revert()

        error_layers = compute_edge_errors_from_layers(
            self._viewer,
            layers[gt_tracks],
            layers[gt_labels],
            layers[pred_tracks],
            layers[pred_labels],
        )
        # Refresh dropdown choices (the new error layers must be present as
        # options) then point the FN/FP roles at them. Guard with _refreshing so
        # these programmatic combo updates don't each trigger a re-lift.
        self._refresh_choices()
        role_by_flag = {
            "fn_edges": EdgeFlag.CTC_FALSE_NEG,
            "fp_edges": EdgeFlag.CTC_FALSE_POS,
        }
        self._refreshing = True
        try:
            for role, flag in role_by_flag.items():
                layer = error_layers.get(flag)
                if layer is not None and layer.name in self._role_combos[role].choices:
                    self._role_combos[role].value = layer.name
        finally:
            self._refreshing = False

        if was_lifted:
            self._apply_lift(self._lift_errors_engine, self._roles_target())
