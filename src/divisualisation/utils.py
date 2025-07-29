import logging
from pathlib import Path

import networkx as nx
import numpy as np
import tifffile
from skimage.util import img_as_ubyte
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_tiff_timeseries(
    dir: Path,
    dtype: str | type | None = None,
    downscale: tuple[int, ...] | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> np.ndarray:
    """Loads a folder of `.tif` or `.tiff` files into a numpy array.
    Each file is interpreted as a frame of a time series.

    Args:
        folder:
        dtype:
        downscale: One int for each dimension of the data. Avoids memory overhead.
        start_frame: The first frame to load.
        end_frame: The last frame to load.

    Returns:
        np.ndarray: The loaded data.
    """
    # TODO make safe for label arrays
    logger.debug(f"Loading tiffs from {dir} as {dtype}")
    files = sorted(list(dir.glob("*.tif")) + list(dir.glob("*.tiff")))[
        start_frame:end_frame
    ]
    shape = tifffile.imread(files[0]).shape
    if downscale:
        assert len(downscale) == len(shape)
    else:
        downscale = (1,) * len(shape)

    files = files[:: downscale[0]]

    x = []
    for f in tqdm(
        files,
        leave=False,
        desc=f"Loading [{start_frame}:{end_frame}:{downscale[0]}]",
    ):
        _x = tifffile.imread(f)
        if dtype:
            _x = _x.astype(dtype)
        assert _x.shape == shape
        slices = tuple(slice(None, None, d) for d in downscale[1:])
        _x = _x[slices]
        x.append(_x)

    x = np.stack(x)
    logger.debug(f"Loaded array of shape {x.shape} from {dir}")
    return x


def crop_borders(x):
    """Crop black and/or white borders of 2D image.

    Parameters
    ----------
    x : np.ndarray
        2d array, or 3d array (RGB, RGBA) with trailing channels dimension.
    """
    if x.ndim < 2 or x.ndim > 3:
        raise ValueError("Input must be 2D image.")
    img = img_as_ubyte(x)
    mask_black = img != 0
    mask_white = img != 255
    if img.ndim == 3:
        if x.shape[2] not in (3, 4):
            raise ValueError("Input image must have 3 (RGB) or 4 (RGBA) channels.")
        mask_black = mask_black[..., :3].all(2)
        mask_white = mask_white[..., :3].all(2)

    mask = mask_white & mask_black
    m, n = mask.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()

    return x[row_start:row_end, col_start:col_end]


def rescale_intensity(
    x, pmin=0.2, pmax=99.8, clip=False, eps=1e-10, axis=None, subsample=1
):
    """Rescale intensity values with percentile-based min and max.

    Parameters
    ----------
    x : np.ndarray
        N-dimensional image.
    pmin : float
        This percentile value gets mapped to 0.
    pmax : float
        This percentile value gets mapped to 1.
    clip : bool
        If True, output is clipped to (0,1).
    eps : float
        Additive term for denominator to avoid division by 0.
    axis : int or tuple of ints
        Axes along which the intensities are rescaled.
        The default is to rescale along a flattened version of the array.
    subsample : int
        If subsample > 1, use a fraction for percentile calculation (faster).

    Returns:
    -------
    image: np.ndarray
        float32
    """
    x = np.asarray(x, dtype=np.float32)

    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, int):
        axis = (np.arange(x.ndim)[axis],)

    subslice = tuple(
        slice(0, None, subsample) if i in axis else slice(None)
        for i in tuple(range(x.ndim))
    )

    mi, ma = np.percentile(x[subslice], (pmin, pmax), axis=axis, keepdims=True)
    logger.debug(f"Min intensity (at p={pmin / 100}) = {mi}")
    logger.debug(f"Max intensity (at p={pmax / 100}) = {ma}")

    x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x.astype(np.float32, copy=False)


def str2bool(x: str) -> bool:
    """Cast string to boolean.

    Useful for parsing command line arguments.
    """
    if not isinstance(x, str):
        raise TypeError("String expected.")
    elif x.lower() in ("true", "t", "1"):
        return True
    elif x.lower() in ("false", "f", "0"):
        return False
    else:
        raise ValueError(f"'{x}' does not seem to be boolean.")


def str2path(x: str) -> Path:
    """Cast string to resolved absolute path.

    Useful for parsing command line arguments.
    """
    if not isinstance(x, str):
        raise TypeError("String expected.")
    else:
        return Path(x).expanduser().resolve()


def linear_chains(G: nx.DiGraph):
    """Find all linear chains in a tree/graph, i.e. paths that.

    i) either start/end at a node with out_degree>in_degree or and have no internal branches, or
    ii) consists of a single node or a single splitting node

    Note that each chain includes its start/end node, i.e. they can be appear in multiple chains.
    """
    # get all nodes with out_degree>in_degree (i.e. start of chain)
    nodes = tuple(n for n in G.nodes if G.out_degree[n] > G.in_degree[n])
    # single nodes are those that are not starting a linear chain
    # single_nodes = tuple(n for n in G.nodes if G.out_degree[n] == G.in_degree[n] == 0)
    single_nodes = tuple(
        n for n in G.nodes if G.in_degree[n] == 0 and G.out_degree[n] != 1
    )

    for ni in single_nodes:
        yield [ni]

    for ni in nodes:
        neighs = tuple(G.neighbors(ni))
        for child in neighs:
            path = [ni, child]
            while len(childs := tuple(G.neighbors(path[-1]))) == 1:
                path.append(childs[0])
            yield path


def graph_to_napari_tracks(
    graph: nx.DiGraph,
    properties: list[str] = [],
):
    """Convert a track graph to napari tracks."""
    # each tracklet is a linear chain in the graph
    chains = tuple(linear_chains(graph))

    track_end_to_track_id = dict()
    labels = []
    for i, cs in enumerate(chains):
        label = i + 1
        labels.append(label)
        # if len(cs) == 1:
        #     print(cs)
        #     # Non-connected node
        #     continue
        end = cs[-1]
        track_end_to_track_id[end] = label

    tracks = []
    tracks_graph = dict()
    tracks_props = {p: [] for p in properties}

    for label, cs in tqdm(zip(labels, chains), total=len(chains)):
        start = cs[0]
        if start in track_end_to_track_id and len(cs) > 1:
            tracks_graph[label] = track_end_to_track_id[start]
            # nodes = cs[1:]
            # Include division edges as first edge of chain
            nodes = cs
        else:
            nodes = cs

        for c in nodes:
            node = graph.nodes[c]
            t = node["t"]
            # TODO replace hardcoded node attrs
            coord = (node["z"] + 10 * node["t"], node["y"], node["x"])
            tracks.append([label, t, *list(coord)])

            for p in properties:
                tracks_props[p].append(node[p])

    tracks = np.array(tracks)
    return tracks, tracks_graph, tracks_props
