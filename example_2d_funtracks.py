"""Interactive 2D example using funtracks + the motile-tracker tracks viewer.

Per Caroline Malin-Mayor's suggestion: load the GT tracking graph into a
funtracks ``Tracks`` object and display it through motile-tracker's
``TracksViewer`` (which gives the lineage tree view), then add divisualisation's
edge-error overlays on top. This is the "errors on top of the tools I already
use" workflow from https://github.com/bentaculum/divisualisation/issues/2.

Run in ipython:  %run example_2d_funtracks.py
"""

import logging
import pprint
from pathlib import Path

import napari
import numpy as np
from funtracks.data_model import SolutionTracks
from motile_tracker.data_views import TracksViewer
from tqdm import tqdm
from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics
from tracksdata.graph import RustWorkXGraph

from divisualisation import add_edge_error_tracks
from divisualisation.utils import load_tiff_timeseries, rescale_intensity

logging.basicConfig(level=logging.INFO)
pp = pprint.PrettyPrinter(indent=4)


def ctc_to_funtracks(ctc_dir: str, ndim: int = 3) -> SolutionTracks:
    """Load a CTC track directory into a funtracks Tracks object.

    funtracks is built on tracksdata, so we read the CTC folder into a
    tracksdata graph, take a full GraphView of it, and wrap that in
    SolutionTracks. ndim is 3 for 2D+time (t, y, x).
    """
    graph = RustWorkXGraph.from_ctc(ctc_dir)
    view = graph.filter(node_ids=list(graph.node_ids())).subgraph()
    return SolutionTracks(view, time_attr="t", pos_attr=("y", "x"), ndim=ndim)


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

# A normal 2D viewer with the raw image loaded flat.
viewer = napari.Viewer()
viewer.theme = "dark"
viewer.add_image(img, name="raw", colormap="gray")

# Load the GT graph into a funtracks Tracks object and display it through the
# motile-tracker tracks viewer (adds the lineage tree view + napari layers).
tracks_viewer = TracksViewer.get_instance(viewer)
gt_tracks = ctc_to_funtracks("data/bacteria/TRA")
tracks_viewer.tracks_list.add_tracks(gt_tracks, name="GT tracks")

# Divisualisation edge-error overlays on top, plus the interactive plugin
# widgets (Edge error toggle, Spacetime lift).
add_edge_error_tracks(viewer, gt_graph, pred_graph, tail_width=4)
viewer.window.add_plugin_dock_widget("divisualisation", "Edge error toggle")
viewer.window.add_plugin_dock_widget("divisualisation", "Spacetime lift")

napari.run()
