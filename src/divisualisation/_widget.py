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
import numpy as np
from magicgui.backends._qtpy.widgets import QBaseValueWidget
from magicgui.widgets import CheckBox, ComboBox, Container, FloatSlider, PushButton
from magicgui.widgets.bases import ValueWidget
from qtpy.QtCore import QTimer  # type: ignore[attr-defined]
from superqt import QToggleSwitch

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
        # Colored-division-edge state. Role layers we augmented in place (added
        # division-connection vertices so the divisions draw as the layer's own
        # colored tail), keyed by the layer OBJECT -> its original state (data,
        # display_graph, properties, color_by) to restore on teardown.
        self._suppressed_graphs: dict = {}
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
        # Under the 6 role/labels dropdowns: opt-in colored division edges (see
        # module docstring). Only acts in the Divisualisation workflow.
        self._division_edges = CheckBox(value=False, label="Color division edges")
        self._compute_btn = PushButton(text="Compute errors")
        self._error_controls = [
            *self._role_combos.values(),
            self._gt_labels,
            self._pred_labels,
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
        self._lift_amount.changed.connect(self._on_lift_amount)
        self._division_edges.changed.connect(self._on_division_edges_changed)
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
        # When the dock widget is torn down, restore any role layers we augmented
        # for colored division edges so they don't stay altered.
        self.native.destroyed.connect(lambda *_: self._teardown_division_edges())
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
        logger.info(
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
            self._teardown_division_edges()

    def _on_division_edges_changed(self, *_):
        # Only meaningful in an active Divisualisation view. Re-run the standard
        # apply transaction so edges are (re)built or torn down with the correct
        # revert -> build-from-flat -> apply ordering.
        active = self._lift_errors.value and self._lift_errors_engine.applied
        logger.info(
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
        engine.time_scale = self._lift_amount.value
        # Colored division edges (Divisualisation only). Augment the selected role
        # layers' data AFTER the revert above (so we edit FLAT data, never doubly
        # folded) and BEFORE engine.apply below (so the engine folds the augmented
        # data with everything else). Switching to Lift-all tears it down instead.
        if engine is self._lift_errors_engine:
            self._rebuild_division_edges()
        else:
            self._teardown_division_edges()
        # Always lift EVERY tracks layer. In the errors workflow ``target`` is a
        # role mapping (those get error-view colors); every other tracks layer
        # (incl. hidden, non-role ones) is lifted too, keeping its own coloring.
        engine.apply(target, extra_layers=self._all_tracks_target())
        # After apply, sync the layer-controls "graph" checkbox to our
        # display_graph change and force a redraw so re-augmented layers render at
        # their folded (lifted) positions rather than the flat z=0 plane.
        self._finalize_division_edges()
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
                logger.warning(
                    "[divedges] role %s=%r not found in viewer layers %s",
                    role,
                    name,
                    layer_names,
                )
                continue
            layer = self._viewer.layers[name]
            if not _is_tracks(layer):
                logger.warning(
                    "[divedges] role %s=%r is not a Tracks layer (type=%s)",
                    role,
                    name,
                    type(layer).__name__,
                )
                continue
            yield role, layer

    @staticmethod
    def _division_connection_rows(layer):
        """Rows to append to a layer's data so its division edges draw as tail.

        napari's Tracks ``graph`` maps ``child_track_id -> [parent_track_ids]``
        and draws those parent->daughter edges in a fixed, uncolorable white. To
        get them in the layer's OWN (colored) tail instead, we extend each
        daughter track back to the division point: add a vertex carrying the
        DAUGHTER's track id at the PARENT's last position (max time). The daughter
        tail then starts at the division node, so the edge is drawn as tail.

        Returns an ``(N, cols)`` array of ``[track_id, t, (z,) y, x]`` rows (one
        per division edge, matching the layer's column count), or ``None`` if the
        layer has no divisions.
        """
        graph = dict(layer.graph)
        if not graph:
            return None
        data = np.asarray(layer.data, dtype=float)  # [track_id, t, (z,) y, x]
        track_ids = data[:, 0]

        def last_vertex(track_id):
            rows = data[track_ids == track_id]
            return None if len(rows) == 0 else rows[np.argmax(rows[:, 1])]

        rows = []
        for child, parents in graph.items():
            for parent in np.atleast_1d(parents):
                parent_last = last_vertex(int(parent))
                if parent_last is None:
                    continue
                # Daughter id, at the parent's last position -> extends the
                # daughter's tail back to the division point.
                rows.append([float(child), *parent_last[1:]])
        if not rows:
            return None
        return np.asarray(rows, dtype=float)

    def _rebuild_division_edges(self):
        """Fold division edges into the selected role layers' own colored tails.

        MUST be called while the lift is reverted (edits flat source data) and
        before ``engine.apply`` (which then folds the augmented data with the
        rest). For each selected role layer with divisions, appends a vertex per
        division edge so it draws as the layer's own tail (see
        ``_division_connection_rows``), and turns off the layer's native white
        graph edges. The original data + ``display_graph`` are stashed and put
        back by ``_teardown_division_edges``.
        """
        self._teardown_division_edges()
        logger.info(
            "[divedges] rebuild: checkbox=%s errors_toggle=%s engine_applied=%s",
            self._division_edges.value,
            self._lift_errors.value,
            self._lift_errors_engine.applied,
        )
        if not self._division_edges.value:
            logger.info("[divedges] checkbox off -> nothing to do")
            return
        selected = list(self._selected_role_layers())
        logger.info(
            "[divedges] role values=%s -> selected layers=%s",
            {r: c.value for r, c in self._role_combos.items()},
            [layer.name for _r, layer in selected],
        )
        if not selected:
            # The feature only acts on layers picked in the role dropdowns; with
            # none selected it would silently do nothing. Say so.
            logger.warning("[divedges] no role layer selected -> nothing drawn")
            napari.utils.notifications.show_warning(
                "Color division edges: no role layer selected -- pick a tracks "
                "layer in the GT (or FN/FP) dropdown."
            )
            return
        augmented_any = False
        for _role, layer in selected:
            rows = self._division_connection_rows(layer)
            logger.info(
                "[divedges] %r (role=%s): graph_size=%d, ndata=%d, connection_rows=%s",
                layer.name,
                _role,
                len(dict(layer.graph)),
                len(layer.data),
                None if rows is None else len(rows),
            )
            if rows is None:
                continue
            # Stash originals to restore on teardown. The engine snapshots data
            # AFTER we edit it here (its snapshot runs at apply time), so revert
            # alone would restore the augmented data -- we own this restore.
            self._suppressed_graphs[layer] = {
                "data": np.asarray(layer.data).copy(),
                "graph": dict(layer.graph),
                "display_graph": layer.display_graph,
                "properties": {k: v.copy() for k, v in layer.properties.items()},
                "color_by": layer.color_by,
            }
            augmented = np.vstack([np.asarray(layer.data, dtype=float), rows])
            # Augmenting drops the graph (setting .data resets it) -- fine, the
            # divisions live in the tail now; the original graph is stashed above
            # and restored on teardown so the native edges work again.
            self._set_tracks_data(
                layer, augmented, graph={}, prior=self._suppressed_graphs[layer]
            )
            # Divisions are in the colored tail now; hide the native white edges.
            layer.display_graph = False
            augmented_any = True
            logger.info(
                "[divedges] %r augmented -> ndata=%d, display_graph=%s, color_by=%s",
                layer.name,
                len(layer.data),
                layer.display_graph,
                layer.color_by,
            )
        if not augmented_any:
            # Roles are selected but none has a division graph to draw.
            logger.warning("[divedges] selected layers had no divisions to draw")
            napari.utils.notifications.show_warning(
                "Color division edges: the selected tracks layer(s) have no "
                "divisions (empty graph)."
            )

    def _finalize_division_edges(self):
        """Post-apply fixups for augmented layers.

        Run AFTER ``engine.apply``, where we mutated ``display_graph`` / ``data``
        / ``graph`` inside the engine's blocked-events context. Re-emit
        ``display_graph`` so the vispy layer hides the native white graph edges
        (its ``_on_appearance_change`` listens to that event), and ``refresh()``
        forces a redraw at the folded positions rather than the flat z=0 plane.

        Note: this does NOT update the layer-controls "graph" checkbox, which
        stays visually stale -- napari's QtGraphCheckBoxControl only binds
        checkbox->layer, not layer->checkbox (see its own source comment), so a
        programmatic display_graph change can't drive the widget. Cosmetic only;
        the edges render correctly.
        """
        for layer in self._suppressed_graphs:
            if layer not in self._viewer.layers:
                continue
            layer.events.display_graph()
            layer.refresh()

    @staticmethod
    def _set_tracks_data(layer, data, graph, prior):
        """Set a Tracks layer's data + graph, preserving its coloring.

        Setting ``.data`` clears the graph and resets properties to just
        ``track_id``; re-apply ``graph`` and the prior properties (padded to the
        new length with the column's last value) and ``color_by`` so the layer
        keeps its coloring for the added vertices.
        """
        prior_props = prior["properties"]
        prior_color_by = prior["color_by"]
        layer.color_by = "track_id"  # always-present; avoids a transient warning
        # napari skips updating a HIDDEN layer's ndim/extent when its data
        # changes, so augmenting/restoring a hidden layer leaves a stale extent
        # and it renders unlifted (flat) once shown. Set the data while
        # momentarily visible so the extent updates, then restore visibility.
        was_visible = layer.visible
        layer.visible = True
        layer.data = data
        layer.visible = was_visible
        layer.graph = graph
        n = len(layer.data)
        rebuilt = {}
        for key, values in prior_props.items():
            values = np.asarray(values)
            if len(values) == n:
                rebuilt[key] = values
            elif len(values):
                pad = np.full(n - len(values), values[-1])
                rebuilt[key] = np.concatenate([values, pad])
        if rebuilt:
            layer.properties = rebuilt
        if prior_color_by in layer.properties or not layer.properties:
            layer.color_by = prior_color_by

    def _teardown_division_edges(self):
        """Restore any role layers we augmented back to their original state.

        Safe to call whether or not a lift is active. Puts back each layer's
        original data, graph, coloring and ``display_graph``. This is the
        authoritative restore path: the engine snapshots data AFTER our edit, so
        its own revert would otherwise keep the augmented data.
        """
        if self._suppressed_graphs:
            logger.info(
                "[divedges] teardown: restoring %d layer(s): %s",
                len(self._suppressed_graphs),
                [layer.name for layer in self._suppressed_graphs],
            )
        for layer, prior in list(self._suppressed_graphs.items()):
            if layer in self._viewer.layers:
                self._set_tracks_data(layer, prior["data"], prior["graph"], prior)
                layer.display_graph = prior["display_graph"]
        self._suppressed_graphs.clear()

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
        # Restore any role layers we augmented for colored division edges back to
        # their original data before CTC matching reads them, so the added
        # division-connection vertices don't leak into the computation. The
        # trailing _apply_lift re-augments them.
        self._teardown_division_edges()

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
