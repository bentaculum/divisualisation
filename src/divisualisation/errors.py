"""Functional API to overlay cell-tracking edge errors on an existing napari viewer.

This module implements the modular use case requested in
https://github.com/bentaculum/divisualisation/issues/2: call a single function
on a viewer that already holds your data (points / labels / tracks layers) and
get edge-error track layers added on top, without a dummy ``z`` dimension for
2D data.
"""

import logging
from collections.abc import Mapping, Sequence

import napari
import numpy as np
from napari.layers import Tracks
from napari.utils.colormaps.colormap_utils import vispy_or_mpl_colormap
from traccuracy import EdgeFlag, TrackingGraph

logger = logging.getLogger(__name__)

# CTC false negatives are missing edges, so their coordinates live in the
# ground-truth graph. CTC false positives are spurious predicted edges, so
# their coordinates live in the prediction graph. This mapping is the single
# source of truth for which graph to read for a given error flag.
DEFAULT_ERROR_GRAPHS: dict[EdgeFlag, str] = {
    EdgeFlag.CTC_FALSE_NEG: "gt",
    EdgeFlag.CTC_FALSE_POS: "pred",
}

# Distinct colormaps per error type so false negatives and false positives are
# easy to tell apart when both are overlaid (the spacetime renderer uses a
# single "cool" map for both, since there it relies on layer separation).
DEFAULT_ERROR_COLORMAPS: dict[EdgeFlag, str] = {
    EdgeFlag.CTC_FALSE_NEG: "spring",
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
        flag_graphs[flag] = graphs[graph_key]

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

        properties = {"error": np.full(len(tracks), 0.5)}
        layer = viewer.add_tracks(
            data=tracks,
            name=str(flag.value),
            properties=properties,
            colormaps_dict={"error": vispy_or_mpl_colormap(colormaps[flag])},
            tail_width=tail_width,
            tail_length=1,
            head_length=1,
            blending="translucent_no_depth",
            opacity=1.0,
            scale=None if scale is None else tuple(scale),
            translate=None if translate is None else tuple(translate),
        )
        # Set color_by after construction to avoid a transient napari warning
        # about the feature not being present yet during __init__.
        layer.color_by = "error"
        layers[flag] = layer

    return layers
