"""Fully programmatic 2D example: build the spacetime view and render a video.

Unlike ``example_2d.py`` (which docks the interactive plugin), this scripts the
whole pipeline end to end -- load data, build the napari layers, fold time into
z, add the edge-error overlays, and render an ``.mp4`` -- with no GUI
interaction. It drives the SAME machinery the plugin uses (``SpacetimeLift`` for
the lift, ``add_edge_error_tracks`` for the FN/FP overlays), so there is no
separate scripted rendering path in the core.

Run in ipython:  %run example_programmatic_2d.py
"""

import logging
import pprint
from pathlib import Path

import napari
import numpy as np
from napari_animation import Animation
from tqdm import tqdm
from traccuracy import EdgeFlag, run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics

from divisualisation import SpacetimeLift, add_edge_error_tracks
from divisualisation.utils import (
    graph_to_napari_tracks,
    load_tiff_timeseries,
    rescale_intensity,
)

logging.basicConfig(level=logging.INFO)
pp = pprint.PrettyPrinter(indent=4)


gt = load_ctc_data("data/bacteria/TRA", "data/bacteria/TRA/man_track.txt", name="gt")
pred = load_ctc_data("data/bacteria/RES", "data/bacteria/RES/man_track.txt", name="res")

img = load_tiff_timeseries(Path("data/bacteria/img"))
img = np.stack([
    rescale_intensity(_x, pmin=5, pmax=99.9, clip=False, subsample=1)
    for _x in tqdm(img, desc="Rescale intensity")
])

ctc_results, ctc_matched = run_metrics(
    gt_data=gt,
    pred_data=pred,
    matcher=CTCMatcher(),
    metrics=[CTCMetrics()],
)
pp.pprint(ctc_results)

gt_graph = ctc_matched.gt_graph
pred_graph = ctc_matched.pred_graph

# Build a plain 2D viewer with image, masks and GT/pred tracks -- exactly the
# layers example_2d.py sets up, just without docking the plugin.
viewer = napari.Viewer()
viewer.theme = "dark"
viewer.add_image(img, name="raw", colormap="gray")
viewer.add_labels(pred.segmentation, name="pred masks", opacity=0.3)

for graph, name in ((gt_graph, "GT tracks"), (pred_graph, "predicted tracks")):
    tracks, tracks_graph, _ = graph_to_napari_tracks(
        graph.graph,
        include_z=False,
        drop_division_duplicates=True,
    )
    viewer.add_tracks(tracks, graph=tracks_graph, name=name, tail_length=5)

# Add the false-negative / false-positive edge overlays (each its own Tracks
# layer, named after its EdgeFlag, e.g. "ctc_fn" / "ctc_fp").
error_layers = add_edge_error_tracks(viewer, gt_graph, pred_graph)

# Fold time into z with the same engine the plugin uses. Map each layer to its
# role so the lift applies the error-view coloring (GT / pred / FN / FP); the
# predicted-tracks layer is hidden, since its errors are shown by the overlays.
viewer.layers["predicted tracks"].visible = False
role_names = {"gt": "GT tracks", "pred": "predicted tracks"}
for role, flag in (
    ("fn_edges", EdgeFlag.CTC_FALSE_NEG),
    ("fp_edges", EdgeFlag.CTC_FALSE_POS),
):
    layer = error_layers[flag]
    if layer is not None:
        role_names[role] = layer.name

lift = SpacetimeLift(viewer, time_scale=12)
lift.apply(role_names)  # goes 3D, folds time into z, applies the error-view look

# Render a video by scrubbing the time slider from first to last frame. This is
# the whole render: napari_animation captures a keyframe per slider end and
# tweens between them.
animation = Animation(viewer)
viewer.dims.set_current_step(0, 0)
animation.capture_keyframe()
viewer.dims.set_current_step(0, viewer.dims.nsteps[0] - 1)
animation.capture_keyframe(steps=60)
animation.animate("divisualisation_2d.mp4", fps=12, canvas_only=True)
print("Saved divisualisation_2d.mp4")
