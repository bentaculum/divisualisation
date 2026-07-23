"""Draw a tracks layer's parent->daughter division edges as coloured tail.

napari draws a Tracks layer's ``graph`` (division/lineage) edges in a fixed,
uncolourable white. :class:`ColoredDivisionEdges` works around that by editing
the layer's data IN PLACE: it appends, per division edge, a vertex carrying the
DAUGHTER's track id at the PARENT's last position, so the daughter's own
(coloured) tail starts at the division point. The layer's native white graph
edges are then turned off. Every mutated layer's original state is stashed and
restored on :meth:`teardown`.

This lives outside the widget so the mechanism is a self-contained, testable
unit; the widget owns one instance and drives it around the spacetime lift:
``apply`` (before ``SpacetimeLift.apply``, on flat data), ``finalize`` (after),
and ``teardown`` (on toggle-off / recompute / disposal).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ColoredDivisionEdges:
    """Augment tracks layers so their division edges draw as coloured tail.

    Stateful: ``apply`` records each mutated layer's original data / graph /
    coloring / ``display_graph`` so ``teardown`` restores it exactly.
    """

    def __init__(self):
        # Layer object -> its pre-augmentation state, to restore on teardown.
        self._suppressed: dict = {}

    def apply(self, layers) -> bool:
        """Augment each layer in ``layers`` that has divisions; return if any did.

        MUST be called while the lift is reverted (edits FLAT data) and before
        ``SpacetimeLift.apply`` (which then folds the augmented data with the
        rest). Idempotent: tears down any previous augmentation first.
        """
        self.teardown()
        augmented_any = False
        for layer in layers:
            rows = self._connection_rows(layer)
            logger.debug(
                "[divedges] %r: graph_size=%d, ndata=%d, connection_rows=%s",
                layer.name,
                len(dict(layer.graph)),
                len(layer.data),
                None if rows is None else len(rows),
            )
            if rows is None:
                continue
            # Stash originals to restore on teardown. The lift engine snapshots
            # data AFTER we edit it here (its snapshot runs at apply time), so
            # engine.revert alone would keep the augmented data -- we own this.
            self._suppressed[layer] = {
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
                layer, augmented, graph={}, prior=self._suppressed[layer]
            )
            # Divisions are in the colored tail now; hide the native white edges.
            layer.display_graph = False
            augmented_any = True
            logger.debug(
                "[divedges] %r augmented -> ndata=%d, display_graph=%s, color_by=%s",
                layer.name,
                len(layer.data),
                layer.display_graph,
                layer.color_by,
            )
        return augmented_any

    def finalize(self, viewer) -> None:
        """Post-lift fixups for augmented layers.

        Run AFTER ``SpacetimeLift.apply``, where ``display_graph`` / ``data`` /
        ``graph`` were mutated inside the engine's blocked-events context.
        Re-emit ``display_graph`` so the vispy layer hides the native white graph
        edges (its ``_on_appearance_change`` listens to that event), and
        ``refresh()`` forces a redraw at the folded positions rather than the
        flat z=0 plane.

        Note: this does NOT update the layer-controls "graph" checkbox, which
        stays visually stale -- napari's QtGraphCheckBoxControl only binds
        checkbox->layer, not layer->checkbox, so a programmatic display_graph
        change can't drive the widget. Cosmetic only; the edges render correctly.
        """
        for layer in self._suppressed:
            if layer not in viewer.layers:
                continue
            layer.events.display_graph()
            layer.refresh()

    def teardown(self) -> None:
        """Restore every augmented layer to its original state.

        Safe to call whether or not a lift is active. This is the authoritative
        restore path: the lift engine snapshots data AFTER our edit, so its own
        revert would otherwise keep the augmented data.
        """
        if self._suppressed:
            logger.debug(
                "[divedges] teardown: restoring %d layer(s): %s",
                len(self._suppressed),
                [layer.name for layer in self._suppressed],
            )
        for layer, prior in list(self._suppressed.items()):
            # A layer may have been removed from the viewer since augmentation;
            # napari raises when setting data on a detached layer, so guard.
            try:
                self._set_tracks_data(layer, prior["data"], prior["graph"], prior)
                layer.display_graph = prior["display_graph"]
            except (ValueError, RuntimeError, KeyError):
                pass
        self._suppressed.clear()

    @staticmethod
    def _connection_rows(layer):
        """Rows to append to a layer's data so its division edges draw as tail.

        napari's Tracks ``graph`` maps ``child_track_id -> [parent_track_ids]``.
        Extend each daughter track back to the division point: add a vertex
        carrying the DAUGHTER's track id at the PARENT's last position (max
        time). The daughter tail then starts at the division node.

        Returns an ``(N, cols)`` array of ``[track_id, t, (z,) y, x]`` rows (one
        per division edge, matching the layer's column count), or ``None`` if the
        layer has no divisions.
        """
        graph = dict(layer.graph)
        if not graph:
            return None
        data = np.asarray(layer.data, dtype=float)  # [track_id, t, (z,) y, x]
        if len(data) == 0:
            return None

        # Precompute each track's LAST vertex (max time) in one vectorized pass,
        # rather than re-scanning the whole array per division (which was
        # O(divisions x vertices)). Sort by (track_id, t); the last row of each
        # track_id group is its last vertex.
        order = np.lexsort((data[:, 1], data[:, 0]))  # by track_id, then t
        sorted_data = data[order]
        sorted_ids = sorted_data[:, 0]
        # A row is the last of its group iff the next row has a different id.
        is_last = np.empty(len(sorted_data), dtype=bool)
        is_last[-1] = True
        is_last[:-1] = sorted_ids[1:] != sorted_ids[:-1]
        last_rows = sorted_data[is_last]
        last_by_track = {int(r[0]): r for r in last_rows}

        rows = []
        for child, parents in graph.items():
            for parent in np.atleast_1d(parents):
                parent_last = last_by_track.get(int(parent))
                if parent_last is None:
                    continue
                # Daughter id, at the parent's last position -> extends the
                # daughter's tail back to the division point.
                rows.append([float(child), *parent_last[1:]])
        if not rows:
            return None
        return np.asarray(rows, dtype=float)

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
