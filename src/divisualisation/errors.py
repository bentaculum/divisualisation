"""Functional API to overlay cell-tracking edge errors on an existing napari viewer.

This module implements the modular use case requested in
https://github.com/bentaculum/divisualisation/issues/2: call a single function
on a viewer that already holds your data (points / labels / tracks layers) and
get edge-error track layers added on top, without a dummy ``z`` dimension for
2D data.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import napari
import numpy as np
from napari.utils.colormaps.colormap_utils import vispy_or_mpl_colormap
from traccuracy import EdgeFlag, TrackingGraph

if TYPE_CHECKING:
    # Only needed for type hints; importing napari.layers eagerly would pull in
    # Qt, which the functional API does not otherwise require.
    from napari.layers import Tracks

logger = logging.getLogger(__name__)

# napari Tracks layer property used to drive the per-error-type colormap.
_PROPERTY_KEY = "error"

# Value written to the color property per error type, matching the original
# Divisualisation renderer (the enumerate index: false negatives 0, false
# positives 1). Passed via colormaps_dict below, these map RAW through the
# colormap -- the Tracks layer only 0-1 normalizes when using layer.colormap,
# not colormaps_dict -- so 0 and 1 hit the colormap's actual endpoints.
DEFAULT_ERROR_VALUES: dict[EdgeFlag, float] = {
    EdgeFlag.CTC_FALSE_NEG: 0.0,
    EdgeFlag.CTC_FALSE_POS: 1.0,
}

# CTC false negatives are missing edges, so their coordinates live in the
# ground-truth graph. CTC false positives are spurious predicted edges, so
# their coordinates live in the prediction graph. This mapping is the single
# source of truth for which graph to read for a given error flag.
DEFAULT_ERROR_GRAPHS: dict[EdgeFlag, str] = {
    EdgeFlag.CTC_FALSE_NEG: "gt",
    EdgeFlag.CTC_FALSE_POS: "pred",
}

# Colormap per error type. Matches the original Divisualisation renderer, which
# uses a single "cool" map for both error types and relies on them being
# separate named layers (not color) to tell false negatives from false
# positives.
DEFAULT_ERROR_COLORMAPS: dict[EdgeFlag, str] = {
    EdgeFlag.CTC_FALSE_NEG: "cool",
    EdgeFlag.CTC_FALSE_POS: "cool",
}


def _detect_ndim(graph: TrackingGraph) -> int:
    """Return the number of spatial dimensions (2 or 3) of a tracking graph.

    Decided from a single node and validated to be consistent across the graph,
    so a graph that mixes 2D and 3D nodes fails loudly rather than silently
    producing geometrically wrong tracks.
    """
    nodes = graph.graph.nodes
    if len(nodes) == 0:
        raise ValueError("Cannot detect dimensionality of an empty graph.")

    first_attrs = next(iter(nodes.values()))
    has_z = "z" in first_attrs
    for node_id, attrs in nodes.items():
        if ("z" in attrs) != has_z:
            raise ValueError(
                f"Graph mixes 2D and 3D nodes: node {node_id} disagrees with the "
                "first node on whether it has a 'z' attribute."
            )
    return 3 if has_z else 2


def _edge_error_tracks(
    graph: TrackingGraph,
    flag: EdgeFlag,
    ndim: int,
) -> np.ndarray:
    """Build a napari tracks array for all edges carrying ``flag``.

    Each errored edge becomes a two-point tracklet (its endpoints), so it shows
    up as a short segment. Returns an ``(N, ndim + 2)`` array with columns
    ``[track_id, t, (z), y, x]``, or an empty ``(0, ndim + 2)`` array.
    """
    n_columns = ndim + 2
    rows = []
    for edge_id, (u_id, v_id) in enumerate(graph.get_edges_with_flag(flag), start=1):
        for node_id in (u_id, v_id):
            node = graph.graph.nodes[node_id]
            if ndim == 3:
                rows.append([edge_id, node["t"], node["z"], node["y"], node["x"]])
            else:
                rows.append([edge_id, node["t"], node["y"], node["x"]])

    if not rows:
        return np.empty((0, n_columns))
    return np.asarray(rows, dtype=float)


def add_edge_error_tracks(
    viewer: napari.Viewer,
    gt_graph: TrackingGraph,
    pred_graph: TrackingGraph,
    *,
    error_flags: Sequence[EdgeFlag] = (
        EdgeFlag.CTC_FALSE_NEG,
        EdgeFlag.CTC_FALSE_POS,
    ),
    ndim: int | None = None,
    scale: Sequence[float] | None = None,
    translate: Sequence[float] | None = None,
    tail_width: int = 4,
    colormaps: Mapping[EdgeFlag, str] | None = None,
) -> dict[EdgeFlag, Tracks | None]:
    """Add edge-error track layers to an existing napari viewer.

    Works for 2D+time and 3D+time data. Unlike the ``Divisualisation`` spacetime
    renderer, this does not fold time into the ``z`` axis, does not add a dummy
    ``z`` dimension for 2D data, and does not change ``viewer.dims.ndisplay`` or
    install any callbacks. The layers it adds compose with whatever layers are
    already in the viewer (e.g. points / labels / tracks from motile-tracker),
    and get napari's built-in eye-icon visibility toggles for free.

    Both graphs are accepted so a single call can draw both error types. Only
    the graph a requested flag reads from is actually used, so to visualize just
    one error type you may pass an empty graph for the other (and set ``ndim``
    explicitly if that leaves no non-empty graph to auto-detect from).

    Args:
        viewer: An existing napari viewer to add layers to.
        gt_graph: Matched ground-truth tracking graph (source of false negatives).
        pred_graph: Matched prediction tracking graph (source of false positives).
        error_flags: Which edge error types to visualize.
        ndim: Number of spatial dimensions (2 or 3). If ``None``, auto-detected
            from the graphs (2D iff nodes have no ``z`` attribute).
        scale: Spatial-only scale ``(y, x)`` for 2D or ``(z, y, x)`` for 3D,
            matching napari's Tracks layer convention (no leading time entry).
            Pass the same scale as your existing image/labels layers so the
            overlay aligns. ``None`` uses unit scale.
        translate: Spatial-only translation, same convention as ``scale``.
        tail_width: Width of the error segments, in pixels.
        colormaps: Optional mapping from error flag to a matplotlib/vispy
            colormap name. Missing entries fall back to sensible defaults.

    Returns:
        A dict from each requested error flag to its Tracks layer, or ``None``
        for flags that had no errors (so callers can iterate over every
        requested flag without a ``KeyError``).
    """
    graphs = {"gt": gt_graph, "pred": pred_graph}
    colormaps = {**DEFAULT_ERROR_COLORMAPS, **(colormaps or {})}

    # Resolve which graph each requested flag reads from, rejecting unknown flags
    # before touching any data.
    flag_graphs: dict[EdgeFlag, TrackingGraph] = {}
    for flag in error_flags:
        graph_key = DEFAULT_ERROR_GRAPHS.get(flag)
        if graph_key is None:
            raise ValueError(
                f"Unsupported error flag {flag!r}. Supported flags: "
                f"{list(DEFAULT_ERROR_GRAPHS)}."
            )
        graph = graphs.get(graph_key)
        if graph is None:
            raise ValueError(
                f"Flag {flag!r} maps to unknown graph key {graph_key!r}; "
                "DEFAULT_ERROR_GRAPHS and the gt/pred arguments are out of sync."
            )
        flag_graphs[flag] = graph

    if ndim is None:
        # Detect from the graphs we actually read (a graph the user does not
        # need may legitimately be empty), and require them to agree.
        detected = {_detect_ndim(g) for g in flag_graphs.values() if len(g.nodes) > 0}
        if not detected:
            raise ValueError(
                "Cannot auto-detect dimensionality: all requested graphs are "
                "empty. Pass ndim=2 or ndim=3 explicitly."
            )
        if len(detected) > 1:
            raise ValueError(
                f"Ground-truth and prediction graphs disagree on dimensionality: "
                f"{sorted(detected)}. Pass ndim explicitly if this is intended."
            )
        ndim = detected.pop()

    for name, value in (("scale", scale), ("translate", translate)):
        if value is not None and len(value) != ndim:
            raise ValueError(
                f"{name} must have {ndim} spatial entries (napari Tracks layers "
                f"take a spatial-only {name} with no leading time entry), got "
                f"{len(value)}."
            )

    layers: dict[EdgeFlag, Tracks | None] = {}
    for flag, graph in flag_graphs.items():
        tracks = _edge_error_tracks(graph, flag, ndim)
        if len(tracks) == 0:
            logger.info("No edge errors of type %s", flag)
            layers[flag] = None
            continue

        # One flat value per error type (main's enumerate index), mapped RAW via
        # colormaps_dict so it bypasses the Tracks layer's 0-1 normalization and
        # hits the colormap's true value (matches the original renderer).
        error_value = DEFAULT_ERROR_VALUES[flag]
        properties = {_PROPERTY_KEY: np.full(len(tracks), error_value)}
        layer = viewer.add_tracks(
            data=tracks,
            name=str(flag.value),
            properties=properties,
            color_by=_PROPERTY_KEY,
            colormaps_dict={_PROPERTY_KEY: vispy_or_mpl_colormap(colormaps[flag])},
            tail_width=tail_width,
            # Long tail so the whole error segment stays drawn, matching the
            # original Divisualisation renderer.
            tail_length=1000,
            head_length=1,
            blending="translucent_no_depth",
            opacity=1.0,
            scale=None if scale is None else tuple(scale),
            translate=None if translate is None else tuple(translate),
        )
        layers[flag] = layer

    return layers


def compute_edge_errors_from_layers(
    viewer,
    gt_tracks,
    gt_labels,
    pred_tracks,
    pred_labels,
    **kwargs,
):
    """Compute CTC edge errors from napari layers and overlay them.

    Runs traccuracy CTC matching on the GT and prediction tracks (using their
    segmentation label layers), then adds the false-negative / false-positive
    edge overlays via :func:`add_edge_error_tracks`. Use this when the errors
    are not already present in the viewer.

    Detections are matched to the segmentation implicitly: each detection's
    label is read from the pixel under its ``(t, (z), y, x)`` position, so the
    tracks layers need no extra per-detection property. This assumes each track
    point sits inside its own mask (true for tracks produced from the
    segmentation, e.g. via :func:`~divisualisation.utils.graph_to_napari_tracks`).

    Args:
        viewer: The napari viewer to add error layers to.
        gt_tracks: Ground-truth napari Tracks layer.
        gt_labels: Ground-truth napari Labels layer (segmentation).
        pred_tracks: Prediction napari Tracks layer.
        pred_labels: Prediction napari Labels layer (segmentation).
        **kwargs: Forwarded to :func:`add_edge_error_tracks`.

    Returns:
        The dict returned by :func:`add_edge_error_tracks`.
    """
    # Imported lazily so the module has no hard traccuracy-loader / heavy deps
    # at import time.
    from traccuracy import run_metrics
    from traccuracy.matchers import CTCMatcher
    from traccuracy.metrics import CTCMetrics

    # load_napari_data is vendored (traccuracy PR #358 is not released yet); once
    # it lands in a traccuracy release, switch back to
    # ``from traccuracy.loaders import load_napari_data``.
    from ._napari_loader import load_napari_data

    def to_graph(tracks, labels, name):
        # Implicit matching: load_napari_data assigns each detection a mask per
        # frame by bipartite matching. Route its progress bar through napari's
        # activity dock (progbar_class=napari.utils.progress) so the matching
        # shows up in the GUI, not just the terminal.
        return load_napari_data(
            np.asarray(tracks.data),
            graph=tracks.graph,
            segmentation=np.asarray(labels.data),
            name=name,
            progbar_class=napari.utils.progress,
        )

    gt_graph = to_graph(gt_tracks, gt_labels, "gt")
    pred_graph = to_graph(pred_tracks, pred_labels, "pred")
    _, matched = run_metrics(
        gt_data=gt_graph,
        pred_data=pred_graph,
        matcher=CTCMatcher(),
        metrics=[CTCMetrics()],
    )
    return add_edge_error_tracks(viewer, matched.gt_graph, matched.pred_graph, **kwargs)
