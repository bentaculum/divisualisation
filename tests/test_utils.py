import networkx as nx

from divisualisation.utils import graph_to_napari_tracks, linear_chains


def _linear_2d_graph():
    graph = nx.DiGraph()
    graph.add_node(0, t=0, y=10.0, x=20.0)
    graph.add_node(1, t=1, y=11.0, x=21.0)
    graph.add_node(2, t=2, y=12.0, x=22.0)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    return graph


def _linear_3d_graph():
    graph = nx.DiGraph()
    graph.add_node(0, t=0, z=5.0, y=10.0, x=20.0)
    graph.add_node(1, t=1, z=5.0, y=11.0, x=21.0)
    graph.add_edge(0, 1)
    return graph


def test_include_z_true_yields_five_columns():
    tracks, _, _ = graph_to_napari_tracks(_linear_2d_graph(), include_z=True)
    # [id, t, z, y, x], with z=1 as the pseudo dimension for 2D nodes.
    assert tracks.shape[1] == 5


def test_include_z_false_yields_four_columns():
    tracks, _, _ = graph_to_napari_tracks(_linear_2d_graph(), include_z=False)
    # [id, t, y, x] for genuinely 2D data.
    assert tracks.shape[1] == 4


def test_3d_graph_five_columns_with_real_z():
    tracks, _, _ = graph_to_napari_tracks(_linear_3d_graph(), include_z=True)
    assert tracks.shape[1] == 5
    # z column preserves the real value, not the pseudo 1.
    assert set(tracks[:, 2]) == {5.0}


def test_linear_chains_on_dividing_track():
    # A single cell (0 -> 1) that divides into two daughters (2, 3) at node 1.
    graph = nx.DiGraph()
    for n in range(4):
        graph.add_node(n, t=0, y=0.0, x=0.0)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)  # daughter A
    graph.add_edge(1, 3)  # daughter B
    chains = {tuple(chain) for chain in linear_chains(graph)}
    # The mother chain 0->1, then one chain per daughter starting at the
    # division node 1 (which is shared, as documented in linear_chains).
    assert chains == {(0, 1), (1, 2), (1, 3)}


def _dividing_2d_graph():
    # node 0 (t0) -> node 1 (t1) divides into node 2 and node 3 (t2).
    graph = nx.DiGraph()
    graph.add_node(0, t=0, y=0.0, x=0.0)
    graph.add_node(1, t=1, y=1.0, x=1.0)
    graph.add_node(2, t=2, y=2.0, x=2.0)
    graph.add_node(3, t=2, y=3.0, x=3.0)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(1, 3)
    return graph


def test_drop_division_duplicates():
    graph = _dividing_2d_graph()
    # Default keeps the division node in each child chain, so the divider node
    # (id 1) is emitted once per child -> more rows than graph nodes.
    kept, _, _ = graph_to_napari_tracks(graph, include_z=False)
    dropped, _, _ = graph_to_napari_tracks(
        graph, include_z=False, drop_division_duplicates=True
    )
    assert len(dropped) == graph.number_of_nodes()  # 4, one row per detection
    assert len(kept) > len(dropped)  # duplicates present by default


def test_division_connection_rows_extend_daughters_to_division():
    # The in-place augmentation rows need no GUI viewer: a bare Tracks layer
    # carries the .data + .graph the staticmethod reads. Each row extends a
    # daughter track back to the division point (daughter id at parent's last
    # position), so the division draws as the daughter's own colored tail.
    import numpy as np
    from napari.layers import Tracks

    from divisualisation.division_edges import ColoredDivisionEdges

    graph = _dividing_2d_graph()
    # Layers are built by the examples with the division node dropped, so each
    # daughter starts one step after the parent -- the realistic input.
    tracks, tracks_graph, _ = graph_to_napari_tracks(
        graph, include_z=False, drop_division_duplicates=True
    )
    layer = Tracks(tracks, graph=tracks_graph)

    rows = ColoredDivisionEdges._connection_rows(layer)
    # Two divisions (1->2, 1->3): one appended vertex each.
    assert rows.shape == (2, 4)  # [track_id, t, y, x]
    # Each row carries a DAUGHTER track id at the PARENT's last position (node 1
    # at t1, y1, x1 -- the division point).
    assert set(rows[:, 0]) == {2, 3}  # daughter track ids
    for row in rows:
        np.testing.assert_allclose(row[1:], [1.0, 1.0, 1.0])  # parent's last vertex


def test_division_connection_rows_none_without_divisions():
    from napari.layers import Tracks

    from divisualisation.division_edges import ColoredDivisionEdges

    # A single linear track has an empty graph -> no division connections.
    tracks, tracks_graph, _ = graph_to_napari_tracks(
        _linear_2d_graph(), include_z=False
    )
    layer = Tracks(tracks, graph=tracks_graph)
    assert ColoredDivisionEdges._connection_rows(layer) is None
