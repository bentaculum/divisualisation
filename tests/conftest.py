import networkx as nx
import pytest
from napari.components import ViewerModel
from traccuracy import EdgeFlag, TrackingGraph


def _node(t, y, x, z=None):
    attrs = {"t": t, "y": y, "x": x}
    if z is not None:
        attrs["z"] = z
    return attrs


def make_tracking_graph(edges, ndim=2):
    """Build a traccuracy TrackingGraph from a list of edge specs.

    Each edge spec is ((u_coord, v_coord), flag), where a coord is a tuple
    (t, y, x) for 2D or (t, z, y, x) for 3D.
    """
    graph = nx.DiGraph()
    next_id = 0
    for (u_coord, v_coord), flag in edges:
        ids = []
        for coord in (u_coord, v_coord):
            if ndim == 3:
                t, z, y, x = coord
                graph.add_node(next_id, **_node(t, y, x, z=z))
            else:
                t, y, x = coord
                graph.add_node(next_id, **_node(t, y, x))
            ids.append(next_id)
            next_id += 1
        graph.add_edge(ids[0], ids[1], **{flag.value: True})
    return TrackingGraph(graph)


@pytest.fixture
def make_graph():
    """Factory fixture returning :func:`make_tracking_graph`."""
    return make_tracking_graph


@pytest.fixture
def graphs_2d():
    """A GT graph with one false negative and a pred graph with one false positive."""
    gt = make_tracking_graph(
        [(((0, 10.0, 20.0), (1, 11.0, 21.0)), EdgeFlag.CTC_FALSE_NEG)],
        ndim=2,
    )
    pred = make_tracking_graph(
        [(((0, 30.0, 40.0), (1, 31.0, 41.0)), EdgeFlag.CTC_FALSE_POS)],
        ndim=2,
    )
    return gt, pred


@pytest.fixture
def graphs_3d():
    gt = make_tracking_graph(
        [(((0, 5.0, 10.0, 20.0), (1, 5.0, 11.0, 21.0)), EdgeFlag.CTC_FALSE_NEG)],
        ndim=3,
    )
    pred = make_tracking_graph(
        [(((0, 6.0, 30.0, 40.0), (1, 6.0, 31.0, 41.0)), EdgeFlag.CTC_FALSE_POS)],
        ndim=3,
    )
    return gt, pred


@pytest.fixture
def viewer_model():
    return ViewerModel()
