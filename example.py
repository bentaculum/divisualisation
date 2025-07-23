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
from .utils import crop_borders, rescale_intensity, str2bool, str2path
from trackastra.tracking import load_ctc_graph, graph_to_napari_tracks
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

gt_data = load_ctc_data(
    "downloads/Fluo-N2DL-HeLa/01_GT/TRA",
    "downloads/Fluo-N2DL-HeLa/01_GT/TRA/man_track.txt",
    name="gt",
)
pred_data = load_ctc_data(
    "sample-data/Fluo-N2DL-HeLa/01_RES",
    "sample-data/Fluo-N2DL-HeLa/01_RES/res_track.txt",
    name="pred",
)

ctc_results, ctc_matched = run_metrics(
    gt_data=gt_data,
    pred_data=pred_data,
    matcher=CTCMatcher(),
    metrics=[CTCMetrics()],
)
pp.pprint(ctc_results)


def visualize_edge_errors(
    viewer,
    masks_tracked,
    masks_original,
    scale,
    df_edge_errors=None,
    feats="",
    translate=[0, 0, 0],
):
    errors_layer = {}
    errors_data = {}
    for i, (error, reference_masks, cmap) in enumerate(
        zip(
            ("is_fn", "is_fp"),
            (masks_original, masks_tracked),
            # ("hsv", "gray_r"),
            # ("PiYG", "PiYG"),
            ("cool", "cool"),
        )
    ):
        np.zeros_like(masks_original)
        edge_error_tracks = []
        edge_error_props = {"error_type": []}
        for edge_id, row in df_edge_errors[df_edge_errors[error]].iterrows():
            if row.t_u >= len(reference_masks) or row.t_v >= len(reference_masks):
                continue

            edge_id += 1  # avoid track id 0

            c_u = np.argwhere(reference_masks[row.t_u] == int(row.u)).mean(axis=0)
            edge_error_tracks.append([edge_id, row.t_u, *c_u])
            edge_error_props["error_type"].append(i)

            c_v = np.argwhere(reference_masks[row.t_v] == int(row.v)).mean(axis=0)
            edge_error_tracks.append([edge_id, row.t_v, *c_v])
            edge_error_props["error_type"].append(i)

        edge_error_props = {k: np.array(v) for k, v in edge_error_props.items()}

        if len(edge_error_tracks) > 0:
            layer = viewer.add_tracks(
                data=np.stack(edge_error_tracks),
                properties=edge_error_props,
                color_by="error_type",
                # colormap=cmap,
                colormaps_dict={
                    "error_type": vispy_or_mpl_colormap(cmap),
                },
                scale=scale,
                # tail_width=8,
                tail_width=5,
                head_length=1,
                tail_length=1,
                visible=True,
                blending="translucent_no_depth",
                opacity=1.0,
                translate=translate,
                name=error,
            )
            errors_layer[error] = layer
            errors_data[error] = {
                "tracks": np.stack(edge_error_tracks),
                "properties": edge_error_props,
            }
        else:
            logger.info(f"No edge errors of type {error}")

    return errors_layer, errors_data
