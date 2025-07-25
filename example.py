import traccuracy

import os
import pprint
import urllib.request
import zipfile

from tqdm import tqdm

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


ctc_results, ctc_matched = run_metrics(
    gt_data=gt,
    pred_data=pred,
    matcher=CTCMatcher(),
    metrics=[CTCMetrics()],
)
pp.pprint(ctc_results)

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
# v.window._qt_window.showFullScreen()

v.theme = "light"

scale = (7, 7, 1, 1)  # TODO remove hardcoded params
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


errors_layer, errors_data = visualize_edge_errors(
    viewer=v,
    gt_graph=gt_graph,
    pred_graph=pred_graph,
    masks_original=gt.segmentation,
    masks_tracked=pred.segmentation,
    scale=scale,
)


# Update clipping plane based on time (axis 0)
def update_clipping_plane(event=None):
    t = v.dims.point[0]
    # Move clipping plane along Z axis to t (or adjust axis as needed)
    clipping_planes_img = [
        {
            "position": (t - 1, 0, 0),
            "normal": (1, 0, 0),
            "enabled": True,
        },
        {
            "position": (t, 0, 0),
            "normal": (-1, 0, 0),
            "enabled": True,
        },
    ]
    clipping_planes_tracks = [
        {
            "position": (t - 1, 0, 0),
            "normal": (1, 0, 0),
            "enabled": False,
        },
        {
            "position": (t, 0, 0),
            "normal": (-1, 0, 0),
            "enabled": True,
        },
    ]
    image_layer.experimental_clipping_planes = clipping_planes_img
    labels_layer.experimental_clipping_planes = clipping_planes_img
    gt_tracks_layer.experimental_clipping_planes = clipping_planes_tracks
    # for name, layer in errors_layer.items():
    # layer.experimental_clipping_planes = clipping_planes


# Connect event
v.dims.events.point.connect(update_clipping_plane)
v.dims.set_current_step(0, image_layer.data.shape[0])
