"""Interactive 3D example (C. elegans nuclei).

Downloads the dataset if needed, then loads it into a normal napari viewer with
the real 3D volume, tracks, and edge-error overlays, and docks the
divisualisation plugin widgets. Nothing is scripted: use **Plugins ->
divisualisation -> Visualize tracks** to fold time into the z axis on demand (on
top of the real nuclei depth) and the time slider to play through it; toggle it
back off to return to the plain volume.

Run in ipython:  %run example_3d.py
"""

import logging
import os
import pickle
import pprint
import urllib
import urllib.request
import zipfile
from pathlib import Path

import napari
import numpy as np
from tqdm import tqdm
from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics

from divisualisation.utils import (
    graph_to_napari_tracks,
    load_tiff_timeseries,
    rescale_intensity,
)

logging.basicConfig(level=logging.INFO)
pp = pprint.PrettyPrinter(indent=4)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


if "gt" not in locals():
    url = "http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-CE.zip"
    data_dir = "data/celegans/downloads/"

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    filename = url.split("/")[-1]
    file_path = os.path.join(data_dir, filename)
    ds_name = filename.split(".")[0]
    if not os.path.exists(file_path):
        print(f"Downloading {ds_name} data from the CTC website")
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(url, file_path, reporthook=t.update_to)
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

    gt_path = Path("data/celegans/TRA")
    if not os.path.exists(gt_path):
        with zipfile.ZipFile(f"{gt_path}.zip", "r") as zip_ref:
            zip_ref.extractall(gt_path.parent)
    gt = load_ctc_data(
        str(gt_path), str(gt_path / "man_track.txt"), run_checks=False, name="gt"
    )

    pred_path = Path("data/celegans/RES")
    if not os.path.exists(pred_path):
        with zipfile.ZipFile(f"{pred_path}.zip", "r") as zip_ref:
            zip_ref.extractall(pred_path.parent)
    pred = load_ctc_data(
        str(pred_path),
        str(pred_path / "man_track.txt"),
        run_checks=False,
        name="trackatra_prediction",
    )

    img = load_tiff_timeseries(Path("data/celegans/downloads/Fluo-N3DH-CE/01"))[:195]
    img = np.stack([
        rescale_intensity(_x, pmin=5, pmax=99.9, clip=False, subsample=16)
        for _x in tqdm(img, desc="Rescale intensity")
    ])

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

# A normal viewer with the real 3D volume and tracks loaded flat. No time->z fold.
viewer = napari.Viewer()
viewer.theme = "dark"
viewer.add_image(img, name="raw", colormap="gray", rendering="mip")
# Both GT and predicted segmentation labels, so the plugin's "Compute errors"
# workflow can match them.
viewer.add_labels(gt.segmentation, name="gt masks", opacity=0.3, visible=False)
viewer.add_labels(pred.segmentation, name="pred masks", opacity=0.3)

# GT and predicted tracks, keeping their real z (include_z=True). Carry each
# detection's segmentation label id so the plugin can compute edge errors, and
# drop division-node duplicates so the tracks round-trip cleanly back to a graph.
for graph, name in ((gt_graph, "GT tracks"), (pred_graph, "predicted tracks")):
    tracks, tracks_graph, props = graph_to_napari_tracks(
        graph.graph,
        properties=["segmentation_id"],
        include_z=True,
        drop_division_duplicates=True,
    )
    viewer.add_tracks(
        tracks,
        graph=tracks_graph,
        name=name,
        properties={"segmentation_id": np.asarray(props["segmentation_id"])},
        tail_length=5,
    )

# Errors are NOT precomputed: open Plugins -> divisualisation ->
# "Divisualisation" and click "Compute errors" to add the
# false-negative / false-positive overlays. (Or call add_edge_error_tracks
# directly, as the flat functional API.)

# Dock the plugin widget so you can toggle + play interactively.
viewer.window.add_plugin_dock_widget("divisualisation", "Lift tracks")

napari.run()
