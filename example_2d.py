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

from divisualisation import Divisualisation
from divisualisation.utils import load_tiff_timeseries, rescale_intensity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter(indent=4)


gt = load_ctc_data(
    "data/bacteria/TRA",
    "data/bacteria/TRA/man_track.txt",
    name="gt",
)
pred = load_ctc_data(
    "data/bacteria/RES",
    "data/bacteria/RES/man_track.txt",
    name="res",
)

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

v = napari.current_viewer()
if v is not None:
    v.close()
v = napari.Viewer()
for layer in v.layers:
    v.layers.remove(layer)
v.theme = "dark"

divis = Divisualisation(
    z_scale=1,
    time_scale=12,
    tracks_width=2,
)

v = divis.visualize_gt(
    v,
    x=img,
    masks=pred.segmentation,
    # networkx graph at traccuracy.TrackingGraph.graph
    graph=gt_graph.graph,
)

v = divis.visualize_edge_errors(
    viewer=v,
    gt_graph=gt_graph,
    pred_graph=pred_graph,
    masks_original=gt.segmentation,
    masks_tracked=pred.segmentation,
)

v.dims.set_current_step(0, 190)
v.camera.angles = (27.919484296382873, -49.86671510905139, -35.8190766165135)
v.camera.perspective = 27

divis.render(v, name="divisualisation_2d")
