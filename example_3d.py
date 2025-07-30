import logging
import pickle
import pprint
from pathlib import Path

import napari

# from trackastra.tracking import load_ctc_graph, graph_to_napari_tracks
from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics

from divisualisation import Divisualisation
from divisualisation.utils import (
    load_tiff_timeseries,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter(indent=4)

# gt = load_ctc_data(
#     "/Users/bgallusser/data/ctc/Fluo-C3DL-MDA231/01_GT/TRA",
#     "/Users/bgallusser/data/ctc/Fluo-C3DL-MDA231/01_GT/TRA/man_track.txt",
#     name="gt",
# )
# pred = load_ctc_data(
#     "/Users/bgallusser/data/ctc/Fluo-C3DL-MDA231/01_RES_trackastra",
#     "/Users/bgallusser/data/ctc/Fluo-C3DL-MDA231/01_RES_trackastra/man_track.txt",
#     name="res",
# )
if "gt" not in locals():
    gt = load_ctc_data(
        "/Users/bgallusser/code/traccuracy/downloads/Fluo-N3DH-CE/01_GT/TRA",
        "/Users/bgallusser/code/traccuracy/downloads/Fluo-N3DH-CE/01_GT/TRA/man_track.txt",
        run_checks=False,
        name="gt",
    )
    pred = load_ctc_data(
        "/Users/bgallusser/code/traccuracy/downloads/Fluo-N3DH-CE/01_GT/TRA",
        "/Users/bgallusser/code/traccuracy/downloads/Fluo-N3DH-CE/01_GT/TRA/man_track.txt",
        run_checks=False,
        name="gt",
    )

    img = load_tiff_timeseries(
        Path("/Users/bgallusser/code/traccuracy/downloads/Fluo-N3DH-CE/01")
    )
    # img = np.stack(
    #     [
    #         rescale_intensity(_x, pmin=1, pmax=99.8, clip=False, subsample=16)
    #         for _x in tqdm(img, desc="Rescale intensity")
    #     ]
    # )

    matched_path = "3d_matched.pkl"
    try:
        ctc_matched = pickle.load(open(matched_path, "rb"))
    except FileNotFoundError:
        ctc_results, ctc_matched = run_metrics(
            gt_data=gt,
            pred_data=pred,
            matcher=CTCMatcher(),
            metrics=[CTCMetrics()],
        )
        pp.pprint(ctc_results)
        pickle.dump(ctc_matched, open(matched_path, "wb"))

    gt_graph = ctc_matched.gt_graph
    pred_graph = ctc_matched.pred_graph
    # networkx graph at traccuracy.TrackingGraph.graph

v = napari.current_viewer()
if v is not None:
    v.close()
v = napari.Viewer()
for layer in v.layers:
    v.layers.remove(layer)
v.theme = "dark"
div = Divisualisation(
    z_scale=5,
    time_scale=10,
)

v = div.visualize_gt(
    v,
    x=img,
    masks=gt.segmentation,
    graph=gt_graph.graph,
)

v.dims.ndisplay = 3
v.camera.angles = (27.919484296382873, -49.86671510905139, -35.8190766165135)
v.camera.perspective = 27


# errors_layer, errors_data = visualize_edge_errors(
#     viewer=v,
#     gt_graph=gt_graph,
#     pred_graph=pred_graph,
#     masks_original=gt.segmentation,
#     masks_tracked=pred.segmentation,
#     scale=scale,
# )
