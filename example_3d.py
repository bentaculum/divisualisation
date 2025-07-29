import traccuracy

import os
import pprint
import urllib.request
import zipfile

from tqdm import tqdm
import pickle

from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatcher, IOUMatcher
from traccuracy.metrics import CTCMetrics, DivisionMetrics

import sys
import yaml
import logging
from pathlib import Path
import napari
from napari.utils.colormaps import label_colormap
from napari.utils.colormaps.colormap_utils import vispy_or_mpl_colormap
from utils import (
    crop_borders,
    rescale_intensity,
    str2bool,
    str2path,
    load_tiff_timeseries,
    graph_to_napari_tracks,
)
from create_layers import visualize_gt, visualize_edge_errors

# from trackastra.tracking import load_ctc_graph, graph_to_napari_tracks
from tifffile import imread
import skimage
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import vispy
import pandas as pd
from copy import deepcopy
from napari_animation import Animation

# from visualize_2d_tracking_errors import visualize_edge_errors

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
if not "gt" in locals():
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
    img = np.stack(
        [
            rescale_intensity(_x, pmin=1, pmax=99.8, clip=False, subsample=16)
            for _x in tqdm(img, desc="Rescale intensity")
        ]
    )

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

    gt_tracks, gt_tracks_graph, gt_properties = graph_to_napari_tracks(
        gt_graph.graph,
        properties=["t"],
    )

v = napari.current_viewer()
if v is not None:
    v.close()
v = napari.Viewer()
for layer in v.layers:
    v.layers.remove(layer)
# v.window._qt_window.showFullScreen()

v.theme = "dark"

scale = (1, 5, 1, 1)  # TODO remove hardcoded params
image_layer, labels_layer, _, gt_tracks_layer, gt_tracks_data = visualize_gt(
    v,
    img,
    gt.segmentation,
    gt_tracks,
    gt_tracks_graph,
    gt_properties,
    # TODO remove hardcoded params
    frame=10,
    scale=scale,
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


image_layer.affine.translate = [0, -scale[1] * img.shape[1] / 2 + scale[1], 0, 0]


# Update clipping plane based on time (axis 0)
def update_clipping_plane(event=None):
    t = v.dims.point[0]
    # Move clipping plane along Z axis to t (or adjust axis as needed)
    clipping_planes_tracks = [
        {
            "position": (t - 1, 0, 0),
            "normal": (1, 0, 0),
            "enabled": False,
        },
        {
            "position": ((t + 1) * 10, 0, 0),
            "normal": (-1, 0, 0),
            "enabled": True,
        },
    ]
    gt_tracks_layer.experimental_clipping_planes = clipping_planes_tracks


def update_translate(event=None):
    t = v.dims.point[0]
    gt_tracks_layer.translate = [0, -10 * (t + 1), 0, 0]


# Connect events
v.dims.events.point.connect(update_clipping_plane)
v.dims.events.point.connect(update_translate)
v.dims.set_current_step(0, image_layer.data.shape[0])
