import traccuracy
from traccuracy import EdgeFlag

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

logger = logging.getLogger(__name__)


def visualize_gt(
    v,
    x,
    masks,
    tracks,
    tracks_graph,
    properties,
    frame,
    scale,
    crop=[0, 0, 0, 0],
):
    assert x.ndim == 3
    # Fix cropping
    crop[1] = None if crop[1] == 0 else -crop[1]
    crop[3] = None if crop[3] == 0 else -crop[3]

    xc = x[:, crop[0] : crop[1], crop[2] : crop[3]]
    mc = masks[:, crop[0] : crop[1], crop[2] : crop[3]]

    xc = np.stack(
        [
            rescale_intensity(_x, pmin=1, pmax=99.8, clip=False)
            for _x in tqdm(xc, desc="Rescale intensity")
        ]
    )

    img_layer = v.add_image(
        # np.expand_dims(xc, 0),
        xc,
        scale=scale,
        # Somehow rotation around a self-chosen origin does not work
        # Error msg: Non-orthogonal slicing is being requested, but is not fully supported. Data is displayed without applying an out-of-slice rotation or shear component.
        translate=np.array([frame, crop[0], crop[2]]) * scale,
        colormap="gray",
        rendering="translucent",
        depiction="plane",
        contrast_limits=[0.0, 0.4],
    )
    labels_layer = v.add_labels(
        # np.expand_dims(mc, 0),
        mc,
        scale=scale,
        translate=np.array([frame, crop[0], crop[2]]) * scale,
        # rendering="iso_categorical",
        rendering="translucent",
        opacity=1.0,
        # seed=0.12,
        depiction="plane",
    )
    labels_layer = None

    # Box around the image
    # v.layers[0].bounding_box.visible = True
    # v.layers[0].bounding_box.opacity = 0.25
    # v.layers[0].bounding_box.line_thickness = 4
    # v.layers[0].bounding_box.points = False
    # v.layers[0].bounding_box.blending = "translucent_no_depth"
    # # v.layers[0].bounding_box.line_color = np.array([0, 0, 0, 0])
    # v.layers[0].bounding_box.line_color = np.array([1.0, 1.0, 1.0, 1.0])

    # TODO remove hardcoded dep on specific property
    properties["gt"] = np.ones_like(properties["t"]) * 0.5
    # lin = np.array(properties["lineage"])

    # # shuffle lineage ids
    # rng = np.random.default_rng(0)
    # perm = rng.permutation(int(lin.max()) + 1)
    # lin_perm = perm[lin.astype(int)]
    # properties["lineage_permuted"] = 0.0 + 1.0 * (
    #     lin_perm.astype(float) / max(lin_perm.max(), 1)
    # )

    track_ids = np.array(tracks[:, 0])
    # properties["track_id_normalized"] = 0.3 + 0.7 * (track_ids / track_ids.max())
    # properties["lineage_normalized"] = lin.astype(float) / lin.max()

    properties = {"gt": properties["gt"]}

    logger.info("Adding gt tracks")
    gt_tracks_layer = v.add_tracks(
        data=tracks,
        # graph=tracks_graph,
        name="gt_tracks",
        properties=properties,
        scale=scale,
        color_by="gt",
        blending="translucent_no_depth",
        colormaps_dict={
            # "lineage_permuted": vispy_or_mpl_colormap("Greens"),
            # "lineage_permuted": label_colormap(seed=0.12),
            # "lineage_normalized": label_colormap(seed=0.12),
            # "track_id_normalized": vispy_or_mpl_colormap("gray_r"),
            # "gt": vispy_or_mpl_colormap("greens"),
            # For white bg
            "gt": vispy_or_mpl_colormap("Greens"),
        },
        # tail_width=2.0,
        tail_width=0.5,
        opacity=1.0,
    )
    return (
        img_layer,
        labels_layer,
        mc,
        gt_tracks_layer,
        {
            "tracks": deepcopy(tracks),
            "properties": deepcopy(properties),
        },
    )


def visualize_edge_errors(
    viewer: napari.Viewer,
    gt_graph: traccuracy.TrackingGraph,
    pred_graph: traccuracy.TrackingGraph,
    masks_original: np.ndarray,
    masks_tracked: np.ndarray,
    scale: tuple,
    feats: str = "",
    translate: list = [0, 0, 0],
):
    errors_layer = {}
    errors_data = {}
    for i, (error, graph, reference_masks, cmap) in enumerate(
        zip(
            (EdgeFlag.CTC_FALSE_NEG, EdgeFlag.CTC_FALSE_POS),
            (gt_graph, pred_graph),
            (masks_original, masks_tracked),
            # ("hsv", "gray_r"),
            # ("PiYG", "PiYG"),
            ("cool", "cool"),
        )
    ):
        np.zeros_like(masks_original)
        edge_error_tracks = []
        edge_error_props = {"error_type": []}
        for edge_id, (u_id, v_id) in enumerate(graph.get_edges_with_flag(error)):
            # Skip edges that are not in the reference masks
            # if row.t_u >= len(reference_masks) or row.t_v >= len(reference_masks):
            # continue

            edge_id += 1  # avoid edge id 0
            u = graph.nodes[u_id]
            v = graph.nodes[v_id]
            # Get positions of the edge endpoints
            # c_u = np.argwhere(reference_masks[u["t"]] == u["segmentation_id"]).mean(
            # axis=0
            # )
            edge_error_tracks.append([edge_id, u["t"], u["y"], u["x"]])
            edge_error_props["error_type"].append(i)

            # c_v = np.argwhere(reference_masks[v["t"]] == v["segmentation_id"]).mean(
            # axis=0
            # )
            edge_error_tracks.append([edge_id, v["t"], v["y"], v["x"]])
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
