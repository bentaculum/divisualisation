"""Optional napari dock widget for divisualisation.

``SpacetimeWidget`` (Plugins -> divisualisation) offers two mutually exclusive
workflows via two toggle switches sharing one lift-amount slider:

- **Lift all tracks layers**: lift every tracks layer into the 3D time->z
  "spacetime" view, keeping each layer's own coloring.
- **Divisualisation**: declare the GT / predicted / FN-edge / FP-edge tracks
  layers by role (name-guessed), optionally compute the CTC edge errors from the
  GT/pred tracks + segmentation labels, and lift with the error-view look. Only
  the tracks layers picked in the role dropdowns stay visible (the predicted
  tracks are hidden by default too); every other TRACKS layer is hidden while
  lifted and restored on toggle-off. Image / labels layers are left untouched.

The role / labels dropdowns and Compute button are always shown, even before the
Divisualisation toggle is on, so layers can be picked up front.

Additive: the functional API works without it.
"""

from contextlib import contextmanager

import napari
from magicgui.backends._qtpy.widgets import QBaseValueWidget
from magicgui.widgets import ComboBox, Container, FloatSlider, PushButton
from magicgui.widgets.bases import ValueWidget
from qtpy.QtCore import QTimer  # type: ignore[attr-defined]
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

        # Error-view controls. Choices are CALLABLES (not static lists): napari's
        # add_dock_widget auto-connects layer inserted/removed/reordered/renamed
        # to each magicgui widget's reset_choices, which re-derives choices from
        # these callables. So the option lists stay correct on any layer change,
        # and a selection is preserved as long as its layer still exists -- which
        # is exactly what we want. (A static list would instead be wiped back to
        # just "—" on every layer event, emptying the dropdowns.)
        self._role_combos = {
            role: ComboBox(label=_ROLE_LABELS[role], choices=self._track_choices)
            for role in ROLES
        }
        self._gt_labels = ComboBox(label="GT labels", choices=self._label_choices)
        self._pred_labels = ComboBox(label="pred labels", choices=self._label_choices)
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
        # Layers are auto-guessed into roles ONCE, when the widget first sees
        # them. After that the dropdowns are the user's to drive: later layer
        # inserts/removals only refresh the choice lists, never re-guess or
        # disturb an existing selection.
        self._guessed = False

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

        # napari's add_dock_widget keeps the choice LISTS in sync (via the
        # auto-connected reset_choices). We only need to fire the one-time
        # auto-guess when layers first appear -- it's a no-op once guessed and
        # never disturbs a selection thereafter.
        self._viewer.layers.events.inserted.connect(self._guess_once)
        self._show_error_controls()
        # Defer the initial guess to the next event-loop tick: add_dock_widget
        # resets combo values right after __init__, so guessing synchronously
        # here would be clobbered.
        QTimer.singleShot(0, self._guess_once)

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
            # _apply_lift hides every non-selected layer for the errors engine.
            self._apply_lift(self._lift_errors_engine, self._roles_target())
        else:
            self._restore_hidden()
            if not self._lift_all.value:
                self._revert_lift()

    def _selected_layer_names(self):
        """Names picked in any role or labels dropdown (the layers the
        Divisualisation view keeps visible).
        """
        selected = set()
        for combo in (*self._role_combos.values(), self._gt_labels, self._pred_labels):
            name = combo.value
            if name and name != _NONE_CHOICE:
                selected.add(name)
        return selected

    def _hide_unselected(self):
        """Hide every TRACKS layer not picked in a dropdown, remembering prior
        visibility so toggle-off can restore it.

        Non-tracks layers (images, labels) are left untouched -- the error view
        only manages which tracks are shown. The predicted-tracks layer is
        hidden even when it fills the ``pred`` role: its errors are shown by the
        FN/FP overlays, so the error view defaults to hiding it.
        """
        keep = self._selected_layer_names()
        pred = self._role_combos["pred"].value
        if pred and pred != _NONE_CHOICE:
            keep.discard(pred)  # predicted tracks stay hidden even as a role
        # Snapshot once per hide session; don't clobber an existing snapshot on
        # a live re-apply (that would record the already-hidden state).
        if not hasattr(self, "_prior_visible") or self._prior_visible is None:
            self._prior_visible = {}
        for layer in self._viewer.layers:
            if not _is_tracks(layer):
                continue  # never hide image / labels layers
            if layer.name in keep:
                # A previously hidden layer that is now selected: show it and
                # forget its snapshot so we don't re-hide/restore it wrongly.
                if layer.name in self._prior_visible:
                    layer.visible = self._prior_visible.pop(layer.name)
                continue
            if layer.name not in self._prior_visible:
                self._prior_visible[layer.name] = layer.visible
            layer.visible = False

    def _restore_hidden(self):
        prior = getattr(self, "_prior_visible", None)
        if not prior:
            self._prior_visible = None
            return
        for name, was_visible in prior.items():
            if name in self._viewer.layers:
                self._viewer.layers[name].visible = was_visible
        self._prior_visible = None

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
        # In the Divisualisation view, keep only the selected layers visible.
        # Re-run on every apply so changing a dropdown updates what's hidden.
        if engine is self._lift_errors_engine:
            self._hide_unselected()

    def _revert_lift(self):
        for e in (self._lift_all_engine, self._lift_errors_engine):
            if e.applied:
                e.revert()

    def _on_lift_amount(self, *_):
        for e in (self._lift_all_engine, self._lift_errors_engine):
            e.time_scale = self._lift_amount.value

    @contextmanager
    def _suspend_role_events(self):
        # Suppress the _on_role_changed handler while we programmatically set
        # combo choices/values.
        #
        # We do two things. (1) Block each combo's ``changed`` signal at the
        # source, so setting .choices (which transiently resets the value) or
        # .value emits nothing -- a flag alone is not enough because the signal
        # can be delivered asynchronously, after the flag is cleared, and then
        # leak into _on_role_changed and wipe selections. (2) Also raise the
        # _refreshing flag and save/restore it (rather than force it False) so
        # nested suspensions -- e.g. a layer-insert refresh firing while another
        # is mid-flight -- don't unguard the outer one.
        combos = [*self._role_combos.values(), self._gt_labels, self._pred_labels]
        prev = self._refreshing
        self._refreshing = True
        blockers = [c.changed.blocked() for c in combos]
        for b in blockers:
            b.__enter__()
        try:
            yield
        finally:
            for b in blockers:
                b.__exit__(None, None, None)
            self._refreshing = prev

    def _on_role_changed(self, changed_role=None):
        # A role/label dropdown changed. Ignore programmatic updates from
        # name-guessing / choice re-derivation (see _suspend_role_events).
        if self._refreshing:
            return
        # A layer may fill only one role: if the changed role now points at a
        # layer another role already uses, clear that other role (the just-set
        # role wins). Guard so these programmatic clears don't recurse.
        if changed_role is not None:
            value = self._role_combos[changed_role].value
            if value and value != _NONE_CHOICE:
                with self._suspend_role_events():
                    for role, combo in self._role_combos.items():
                        if role != changed_role and combo.value == value:
                            combo.value = _NONE_CHOICE
        # Re-apply live if the Divisualisation lift is active.
        if self._lift_errors.value and self._lift_errors_engine.applied:
            self._apply_lift(self._lift_errors_engine, self._roles_target())

    def _show_error_controls(self):
        # The role / labels dropdowns and Compute button are always visible and
        # editable, even before the Divisualisation toggle is on, so the user can
        # pick layers up front. Changing a role/label dropdown re-applies the
        # lift live only while the Divisualisation workflow is active (see
        # _on_role_changed); Compute works regardless.
        for w in self._error_controls:
            w.visible = True
            w.enabled = True

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

    # magicgui ``choices`` callables -- receive the ComboBox and return its
    # current options. Used so napari's reset_choices re-derives live choices on
    # every layer event (see __init__).
    def _track_choices(self, *_):
        return [_NONE_CHOICE, *self._track_layer_names()]

    def _label_choices(self, *_):
        return [_NONE_CHOICE, *self._labels_layer_names()]

    @staticmethod
    def _guess(names, hints, already):
        for name in names:
            if name in already:
                continue
            if any(hint in name.lower() for hint in hints):
                return name
        return _NONE_CHOICE

    def _guess_once(self):
        # Auto-guess roles / labels from layer names, but only the first time and
        # only while nothing is lifted. After this the dropdowns belong to the
        # user: napari's reset_choices keeps the option lists current on later
        # layer events, and this never re-guesses or disturbs a selection again.
        if self._guessed:
            return
        if self._lift_all_engine.applied or self._lift_errors_engine.applied:
            return
        track_names = self._track_layer_names()
        label_names = self._labels_layer_names()
        if not (track_names or label_names):
            return  # nothing to guess from yet; try again on the next insert
        self._guessed = True

        with self._suspend_role_events():
            # Make sure the combos list the current layers before we assign
            # guesses -- our inserted handler may run before napari's own
            # reset_choices for the same event.
            self.reset_choices()
            assigned: set[str] = set()
            for role in ROLES:
                combo = self._role_combos[role]
                guess = self._guess(track_names, _ROLE_NAME_HINTS[role], assigned)
                combo.value = guess
                if guess != _NONE_CHOICE:
                    assigned.add(guess)

            used: set[str] = set()
            for combo, key in ((self._gt_labels, "gt"), (self._pred_labels, "pred")):
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
        # Make sure the new error layers are present as options, then point the
        # FN/FP roles at them. Suspend role events so re-deriving choices and
        # setting these values don't each trigger a re-lift.
        role_by_flag = {
            "fn_edges": EdgeFlag.CTC_FALSE_NEG,
            "fp_edges": EdgeFlag.CTC_FALSE_POS,
        }
        with self._suspend_role_events():
            self.reset_choices()
            for role, flag in role_by_flag.items():
                layer = error_layers.get(flag)
                if layer is not None and layer.name in self._role_combos[role].choices:
                    self._role_combos[role].value = layer.name

        if was_lifted:
            self._apply_lift(self._lift_errors_engine, self._roles_target())
