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

from divisualisation import Divisualisation
from divisualisation.utils import load_tiff_timeseries, rescale_intensity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        # Downloading data
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(url, file_path, reporthook=t.update_to)
        # Unzip the data
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

    gt_path = Path("data/celegans/TRA")
    if not os.path.exists(gt_path):
        with zipfile.ZipFile(f"{gt_path}.zip", "r") as zip_ref:
            zip_ref.extractall(gt_path.parent)
    gt = load_ctc_data(
        str(gt_path),
        str(gt_path / "man_track.txt"),
        run_checks=False,
        name="gt",
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

    if True:
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

v = napari.current_viewer()
if v is not None:
    v.close()
v = napari.Viewer()
for layer in v.layers:
    v.layers.remove(layer)
v.theme = "dark"

divis = Divisualisation(
    z_scale=5,
    time_scale=5,
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
