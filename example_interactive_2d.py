"""Interactive 2D example: overlay edge errors on an existing napari viewer.

This is the modular use case from
https://github.com/bentaculum/divisualisation/issues/2. You already have your
2D data in a napari viewer (image, masks, tracks — e.g. from motile-tracker),
and you just want the false-negative / false-positive edges drawn on top,
without a dummy z dimension and without switching the viewer into 3D.

Run in ipython:  %run example_interactive_2d.py
"""

import logging
import pprint
from pathlib import Path

import napari
import numpy as np
from tqdm import tqdm
from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics

from divisualisation import add_edge_error_tracks
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

# A normal 2D napari viewer holding your own data. Nothing 3D, no dummy z.
viewer = napari.Viewer()
viewer.add_image(img, name="raw", colormap="gray")
viewer.add_labels(pred.segmentation, name="predicted masks", opacity=0.3)

# Predicted tracks as plain 2D tracks (4 columns: [id, t, y, x]).
pred_tracks, pred_tracks_graph, _ = graph_to_napari_tracks(
    pred_graph.graph, include_z=False
)
viewer.add_tracks(
    pred_tracks, graph=pred_tracks_graph, name="predicted tracks", tail_length=5
)

# The one call this refactor is about: drop error overlays onto the viewer.
# Each error type becomes its own named Tracks layer with a free eye-icon toggle.
error_layers = add_edge_error_tracks(viewer, gt_graph, pred_graph)
print("Added error layers:", {str(k.value): v for k, v in error_layers.items()})

# Optional: pop the bulk-toggle dock widget (also available via Plugins menu).
viewer.window.add_plugin_dock_widget("divisualisation", "Edge error toggle")

napari.run()
