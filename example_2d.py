"""Interactive 2D example.

Loads the bacteria dataset flat (image, masks, GT/pred tracks, and edge-error
overlays) into a normal 2D napari viewer, and docks the divisualisation plugin
widgets. Nothing is scripted: use **Plugins -> divisualisation -> Spacetime
lift** to fold time into z on demand and the napari time slider to play through
it; toggle it back off to return to flat 2D.

Run in ipython:  %run example_2d.py
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

# A normal 2D viewer with everything loaded flat. No dummy z, no 3D.
viewer = napari.Viewer()
viewer.theme = "dark"
viewer.add_image(img, name="raw", colormap="gray")
viewer.add_labels(pred.segmentation, name="predicted masks", opacity=0.3)

# GT and predicted tracks with the same tail settings the app defaults to.
for graph, name in ((gt_graph, "GT tracks"), (pred_graph, "predicted tracks")):
    tracks, tracks_graph, _ = graph_to_napari_tracks(graph.graph, include_z=False)
    viewer.add_tracks(tracks, graph=tracks_graph, name=name, tail_length=5)

# Flat edge-error overlays (false negatives / false positives).
add_edge_error_tracks(viewer, gt_graph, pred_graph, tail_width=4)

# Dock the plugin widgets so you can toggle + play interactively.
viewer.window.add_plugin_dock_widget("divisualisation", "Edge error toggle")
viewer.window.add_plugin_dock_widget("divisualisation", "Spacetime lift")

napari.run()
