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

A "Color division edges" checkbox (under the dropdowns, default off) applies only
in the Divisualisation workflow and only to the tracks layers picked in the role
dropdowns. napari draws a Tracks layer's parent->daughter division edges in a
hardcoded, uncolorable white. When on, each selected role layer is edited IN PLACE
-- a vertex per division edge is appended to its data so the division draws as
part of the layer's OWN (colored) tail -- and the layer's native white graph edges
are turned off. The original data / coloring / graph toggle are restored on
toggle-off / uncheck.

Additive: the functional API works without it.
"""

import logging
from contextlib import contextmanager

import napari
from magicgui.backends._qtpy.widgets import QBaseValueWidget
from magicgui.widgets import CheckBox, ComboBox, Container, FloatSlider, PushButton
from magicgui.widgets.bases import ValueWidget
from qtpy.QtCore import QTimer  # type: ignore[attr-defined]
from qtpy.QtWidgets import QGroupBox, QVBoxLayout  # type: ignore[attr-defined]
from superqt import QToggleSwitch

from .division_edges import ColoredDivisionEdges
from .lift import ROLES, SpacetimeLift, _is_labels, _is_tracks

logger = logging.getLogger(__name__)


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
    "pred": "Predicted tracks",
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
        # Draws the selected role layers' division edges as coloured tail (see
        # division_edges.py); the widget picks which layers, this owns the
        # in-place augmentation + restore.
        self._div_edges = ColoredDivisionEdges()
        # One lift engine per toggle view, so each view keeps its own
        # layer-control settings. They share a camera store, so the camera
        # (center/zoom/angles/perspective) is shared across the two views.
        camera_store: dict = {}
        self._lift_all_engine = SpacetimeLift(viewer, camera_store=camera_store)
        self._lift_errors_engine = SpacetimeLift(viewer, camera_store=camera_store)
        # Points at the engine for the currently active toggle.
        self._lift = self._lift_all_engine

        # Two mutually exclusive toggles, each shown in its own box below. The
        # lift amount is one shared value but has a slider in each box; the two
        # sliders mirror each other. ``_lift_scale`` (errors box) is canonical.
        # Both toggles read "Lift tracks" -- which workflow each drives is given
        # by its enclosing box title ("Lift all tracks layers" / "Divisualisation").
        self._lift_all = ToggleSwitch(value=False, label="Lift tracks")
        self._lift_errors = ToggleSwitch(value=False, label="Lift tracks")
        self._lift_scale = FloatSlider(value=12, min=1, max=50, label="Lift scale")
        self._lift_scale_all = FloatSlider(value=12, min=1, max=50, label="Lift scale")

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
        self._pred_labels = ComboBox(
            label="Predicted labels", choices=self._label_choices
        )
        # Under the 6 role/labels dropdowns: opt-in colored division edges (see
        # module docstring). Only acts in the Divisualisation workflow.
        self._division_edges = CheckBox(value=False, label="Color division edges")
        self._compute_btn = PushButton(text="Compute edge errors")
        # The 6 role/labels dropdowns, ordered GT / pred tracks, GT / pred labels,
        # then FN / FP edges last. _error_controls keeps every control that is
        # only meaningful in the Divisualisation workflow (used by
        # _show_error_controls).
        self._role_labels_order = [
            self._role_combos["gt"],
            self._role_combos["pred"],
            self._gt_labels,
            self._pred_labels,
            self._role_combos["fn_edges"],
            self._role_combos["fp_edges"],
        ]
        self._error_controls = [
            *self._role_labels_order,
            self._division_edges,
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
        self._lift_scale.changed.connect(
            lambda *_: self._on_lift_scale(self._lift_scale)
        )
        self._lift_scale_all.changed.connect(
            lambda *_: self._on_lift_scale(self._lift_scale_all)
        )
        self._division_edges.changed.connect(self._on_division_edges_changed)
        self._compute_btn.changed.connect(self._on_compute)
        for role, combo in self._role_combos.items():
            combo.changed.connect(lambda *_, r=role: self._on_role_changed(r))
        for combo in (self._gt_labels, self._pred_labels):
            combo.changed.connect(lambda *_: self._on_role_changed(None))

        # Lay the controls out as two titled boxes: "Lift all tracks layers" and
        # "Divisualisation", each with its own (synced) lift slider. Each group
        # is a magicgui Container wrapped in a QGroupBox added to our native
        # layout; the widgets keep working as normal magicgui widgets.
        self._lift_all_box = Container(
            widgets=[self._lift_all, self._lift_scale_all], labels=True
        )
        self._divis_box = Container(
            widgets=[self._lift_errors, self._lift_scale, *self._error_controls],
            labels=True,
        )
        for title, box in (
            ("Lift all tracks layers", self._lift_all_box),
            ("Divisualisation", self._divis_box),
        ):
            group = QGroupBox(title)
            layout = QVBoxLayout()
            layout.setContentsMargins(4, 4, 4, 4)
            layout.addWidget(box.native)
            group.setLayout(layout)
            self.native.layout().addWidget(group)

        # Keep the role/labels dropdown CHOICES in sync with the layer list.
        # napari's add_dock_widget auto-connects these events to the docked
        # widget's reset_choices, but our combos live inside nested box
        # Containers that the top-level reset_choices doesn't reach, so wire the
        # same events straight to our own per-combo reset instead.
        for event in (
            self._viewer.layers.events.inserted,
            self._viewer.layers.events.removed,
            self._viewer.layers.events.reordered,
            self._viewer.layers.events.renamed,
        ):
            event.connect(lambda *_: self._reset_role_choices())
        # Also fire the one-time auto-guess when layers first appear -- a no-op
        # once guessed and never disturbs a selection thereafter.
        self._viewer.layers.events.inserted.connect(self._guess_once)
        self._show_error_controls()
        # When the dock widget is torn down, restore any role layers we augmented
        # for colored division edges so they don't stay altered.
        self.native.destroyed.connect(self._on_native_destroyed)
        # Defer the initial guess to the next event-loop tick: add_dock_widget
        # resets combo values right after __init__, so guessing synchronously
        # here would be clobbered.
        QTimer.singleShot(0, self._guess_once)
        # napari titles the dock "<widget> (<plugin>)" -- e.g. "Lift tracks &
        # Divisualisation (Divisualisation)". Override it to just the widget name.
        # Deferred: the QDockWidget parent only exists after add_dock_widget runs.
        QTimer.singleShot(0, self._set_dock_title)

    def _set_dock_title(self):
        # Walk up to the enclosing QDockWidget and drop napari's " (plugin)"
        # suffix from its title. Best-effort: if the widget isn't docked (e.g.
        # constructed standalone in a test), there's nothing to retitle.
        from qtpy.QtWidgets import QDockWidget

        parent = self.native.parent()
        while parent is not None and not isinstance(parent, QDockWidget):
            parent = parent.parent()
        if isinstance(parent, QDockWidget):
            parent.setWindowTitle("Lift tracks & Divisualisation")

    # --- toggles ------------------------------------------------------------

    def _on_toggle_all(self, *_):
        if self._lift_all.value:
            if self._lift_errors.value:  # enforce mutual exclusivity
                self._lift_errors.value = False
            self._apply_lift(self._lift_all_engine, self._all_tracks_target())
        elif not self._lift_errors.value:
            self._revert_lift()

    def _on_toggle_errors(self, *_):
        logger.debug(
            "[divedges] Divisualisation toggled -> %s (color-edges checkbox=%s)",
            self._lift_errors.value,
            self._division_edges.value,
        )
        if self._lift_errors.value:
            if self._lift_all.value:  # enforce mutual exclusivity
                self._lift_all.value = False
            # _apply_lift hides every non-selected layer for the errors engine.
            self._apply_lift(self._lift_errors_engine, self._roles_target())
        else:
            self._restore_hidden()
            if not self._lift_all.value:
                self._revert_lift()
            # Revert (above) restores each source layer's own graph via the
            # engine snapshot; drop our edge layers and suppression bookkeeping.
            self._div_edges.teardown()

    def _on_division_edges_changed(self, *_):
        # Only meaningful in an active Divisualisation view. Re-run the standard
        # apply transaction so edges are (re)built or torn down with the correct
        # revert -> build-from-flat -> apply ordering.
        active = self._lift_errors.value and self._lift_errors_engine.applied
        logger.debug(
            "[divedges] checkbox changed -> %s (divisualisation active=%s)",
            self._division_edges.value,
            active,
        )
        if active:
            # A coloring change must not alter layer visibility (e.g. re-hide the
            # predicted layer) the way the Divisualisation toggle does.
            self._apply_lift(
                self._lift_errors_engine,
                self._roles_target(),
                preserve_visibility=True,
            )

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

    def _apply_lift(self, engine, target, preserve_visibility=False):
        """Revert any active lift, (re)build division edges, and apply ``engine``.

        ``preserve_visibility``: when True, keep each tracks layer's current
        visibility across the revert/apply churn and skip the ``_hide_unselected``
        pass. Used when a re-apply is driven by a *coloring* change (the "Color
        division edges" checkbox), which must not re-hide layers the way a
        Divisualisation toggle or role change does.
        """
        # Snapshot current visibility BEFORE the revert/apply cycle, which resets
        # layer data (and can disturb visibility), so a coloring-only re-apply
        # leaves what the user is looking at untouched.
        visible_before = (
            {ly.name: ly.visible for ly in self._viewer.layers if _is_tracks(ly)}
            if preserve_visibility
            else None
        )
        # Revert whichever engine is currently active so switching modes (and
        # engines) rebuilds cleanly; the shared camera carries over.
        for e in (self._lift_all_engine, self._lift_errors_engine):
            if e.applied:
                e.revert()
        self._lift = engine
        engine.lift_scale = self._lift_scale.value
        # Colored division edges (Divisualisation only). Augment the selected role
        # layers' data AFTER the revert above (so we edit FLAT data, never doubly
        # folded) and BEFORE engine.apply below (so the engine folds the augmented
        # data with everything else). Switching to Lift-all tears it down instead.
        if engine is self._lift_errors_engine:
            self._rebuild_division_edges()
        else:
            self._div_edges.teardown()
        # Always lift EVERY tracks layer. In the errors workflow ``target`` is a
        # role mapping (those get error-view colors); every other tracks layer
        # (incl. hidden, non-role ones) is lifted too, keeping its own coloring.
        engine.apply(target, extra_layers=self._all_tracks_target())
        # After apply, sync the layer-controls "graph" checkbox to our
        # display_graph change and force a redraw so re-augmented layers render at
        # their folded (lifted) positions rather than the flat z=0 plane.
        self._div_edges.finalize(self._viewer)
        if visible_before is not None:
            # Coloring-only re-apply: restore exactly the visibility we had,
            # don't re-run the hide policy.
            for ly in self._viewer.layers:
                if ly.name in visible_before:
                    ly.visible = visible_before[ly.name]
        elif engine is self._lift_errors_engine:
            # In the Divisualisation view, keep only the selected layers visible.
            # Re-run on every apply so changing a dropdown updates what's hidden.
            self._hide_unselected()

    def _revert_lift(self):
        for e in (self._lift_all_engine, self._lift_errors_engine):
            if e.applied:
                e.revert()

    def _on_lift_scale(self, source):
        # The two boxes each have a lift slider for one shared value; mirror the
        # one the user moved onto the other (signals blocked to avoid a loop),
        # then push the value to both engines.
        value = source.value
        other = self._lift_scale_all if source is self._lift_scale else self._lift_scale
        if other.value != value:
            with other.changed.blocked():
                other.value = value
        for e in (self._lift_all_engine, self._lift_errors_engine):
            e.lift_scale = value

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

    # --- colored division edges ---------------------------------------------

    def _selected_role_layers(self):
        """(role, layer) for each role dropdown pointing at a real tracks layer.

        Includes ``pred``: the predicted-tracks layer is hidden by default in the
        Divisualisation view, but its division edges are still colored in place so
        they show as soon as it's made visible.
        """
        layer_names = [ly.name for ly in self._viewer.layers]
        for role, combo in self._role_combos.items():
            name = combo.value
            if not name or name == _NONE_CHOICE:
                continue
            if name not in self._viewer.layers:
                logger.debug(
                    "[divedges] role %s=%r not found in viewer layers %s",
                    role,
                    name,
                    layer_names,
                )
                continue
            layer = self._viewer.layers[name]
            if not _is_tracks(layer):
                logger.debug(
                    "[divedges] role %s=%r is not a Tracks layer (type=%s)",
                    role,
                    name,
                    type(layer).__name__,
                )
                continue
            yield role, layer

    def _rebuild_division_edges(self):
        """Colour the selected role layers' division edges (if the box is on).

        Owns the widget-side policy -- checkbox gate, which layers, and the
        user-facing warnings -- and delegates the in-place augmentation to
        ``self._div_edges``. MUST run while the lift is reverted and before
        ``engine.apply`` (see ``_apply_lift``); ``_div_edges.apply`` tears down
        any previous augmentation first.
        """
        if not self._division_edges.value:
            self._div_edges.teardown()
            return
        selected = list(self._selected_role_layers())
        logger.debug(
            "[divedges] rebuild: selected layers=%s",
            [layer.name for _r, layer in selected],
        )
        if not selected:
            # The feature only acts on layers picked in the role dropdowns; with
            # none selected it would silently do nothing. Say so.
            self._div_edges.teardown()
            napari.utils.notifications.show_warning(
                "Color division edges: no role layer selected -- pick a tracks "
                "layer in the GT (or FN/FP) dropdown."
            )
            return
        augmented_any = self._div_edges.apply([layer for _r, layer in selected])
        if not augmented_any:
            napari.utils.notifications.show_warning(
                "Color division edges: the selected tracks layer(s) have no "
                "divisions (empty graph)."
            )

    def _on_native_destroyed(self, *_):
        # The Qt widget is being destroyed; the magicgui container may already be
        # partially torn down (attribute access raising via its __getattr__), so
        # guard the restore.
        try:
            self._div_edges.teardown()
        except (AttributeError, RuntimeError):
            pass

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
        return [layer.name for layer in self._viewer.layers if _is_labels(layer)]

    # magicgui ``choices`` callables -- receive the ComboBox and return its
    # current options. Used so napari's reset_choices re-derives live choices on
    # every layer event (see __init__).
    def _track_choices(self, *_):
        return [_NONE_CHOICE, *self._track_layer_names()]

    def _label_choices(self, *_):
        return [_NONE_CHOICE, *self._labels_layer_names()]

    def _reset_role_choices(self):
        """Re-derive every role/labels combo's choices from the current layers.

        Reset each combo directly rather than via ``self.reset_choices()``: the
        combos are nested inside the per-workflow box Containers, so the
        top-level container's reset_choices does not reach them.
        """
        for combo in (*self._role_combos.values(), self._gt_labels, self._pred_labels):
            combo.reset_choices()

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
            self._reset_role_choices()
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

    def _show_activity_dock(self, state=True):
        # Toggle napari's activity dock so progress bars are visible. Best-effort:
        # the status bar / dock API is private and only present on a real GUI
        # viewer, so guard it (a headless ViewerModel or API change is a no-op).
        try:
            self._viewer.window._status_bar._toggle_activity_dock(state)
        except (AttributeError, RuntimeError):
            pass

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
                ("Predicted tracks", pred_tracks),
                ("GT labels", gt_labels),
                ("Predicted labels", pred_labels),
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
        # Restore any role layers we augmented for colored division edges back to
        # their original data before CTC matching reads them, so the added
        # division-connection vertices don't leak into the computation. The
        # trailing _apply_lift re-augments them.
        self._div_edges.teardown()

        # Give the error overlays the same SPATIAL scale as the GT tracks layer
        # (e.g. an anisotropic z shown 10x), so they align with the image/tracks
        # instead of rendering at unit scale in a wrong z plane. A Tracks layer's
        # scale is (t, (z,) y, x); add_edge_error_tracks wants a spatial-only
        # ((z,) y, x) scale, so drop the leading time entry. FN edges come from
        # the GT graph and FP from the predicted graph -- they should carry the
        # same scale, so warn if the two tracks layers disagree.
        error_scale = tuple(float(s) for s in layers[gt_tracks].scale[1:])
        pred_scale = tuple(float(s) for s in layers[pred_tracks].scale[1:])
        if pred_scale != error_scale:
            napari.utils.notifications.show_warning(
                "GT and predicted tracks layers have different scales "
                f"({error_scale} vs {pred_scale}); using the GT scale for both "
                "error overlays. Align the layer scales to avoid a mismatch."
            )
        # Show napari's activity dock so the loader's mask-matching progress bar
        # (routed through napari.utils.progress) is visible during the compute.
        self._show_activity_dock(True)
        try:
            error_layers = compute_edge_errors_from_layers(
                self._viewer,
                layers[gt_tracks],
                layers[gt_labels],
                layers[pred_tracks],
                layers[pred_labels],
                scale=error_scale,
            )
        finally:
            self._show_activity_dock(False)
        # Make sure the new error layers are present as options, then point the
        # FN/FP roles at them. Suspend role events so re-deriving choices and
        # setting these values don't each trigger a re-lift.
        role_by_flag = {
            "fn_edges": EdgeFlag.CTC_FALSE_NEG,
            "fp_edges": EdgeFlag.CTC_FALSE_POS,
        }
        with self._suspend_role_events():
            self._reset_role_choices()
            for role, flag in role_by_flag.items():
                layer = error_layers.get(flag)
                if layer is not None and layer.name in self._role_combos[role].choices:
                    self._role_combos[role].value = layer.name

        if was_lifted:
            self._apply_lift(self._lift_errors_engine, self._roles_target())
