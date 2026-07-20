import networkx as nx
import numpy as np
import pytest
from traccuracy import EdgeFlag, TrackingGraph

from divisualisation import add_edge_error_tracks
from divisualisation.errors import _detect_ndim


def test_2d_tracks_have_four_columns_no_dummy_z(viewer_model, graphs_2d):
    gt, pred = graphs_2d
    layers = add_edge_error_tracks(viewer_model, gt, pred)

    fn = layers[EdgeFlag.CTC_FALSE_NEG]
    assert fn is not None
    # 2D data: [track_id, t, y, x] -> 4 columns, layer.ndim == 3 (t, y, x)
    assert fn.data.shape[1] == 4
    assert fn.ndim == 3
    # The false-negative coordinates come from the GT graph.
    np.testing.assert_allclose(sorted(fn.data[:, 2]), [10.0, 11.0])
    np.testing.assert_allclose(sorted(fn.data[:, 3]), [20.0, 21.0])


def test_2d_does_not_change_ndisplay(viewer_model, graphs_2d):
    gt, pred = graphs_2d
    assert viewer_model.dims.ndisplay == 2
    add_edge_error_tracks(viewer_model, gt, pred)
    # The functional API must not flip the viewer into 3D like the spacetime path.
    assert viewer_model.dims.ndisplay == 2


def test_3d_tracks_have_five_columns(viewer_model, graphs_3d):
    gt, pred = graphs_3d
    layers = add_edge_error_tracks(viewer_model, gt, pred)
    fp = layers[EdgeFlag.CTC_FALSE_POS]
    assert fp is not None
    assert fp.data.shape[1] == 5
    assert fp.ndim == 4


def test_false_positive_read_from_pred_graph(viewer_model, graphs_2d):
    gt, pred = graphs_2d
    layers = add_edge_error_tracks(viewer_model, gt, pred)
    fp = layers[EdgeFlag.CTC_FALSE_POS]
    # FP coordinates must come from the prediction graph (30/40), not GT (10/20).
    np.testing.assert_allclose(sorted(fp.data[:, 2]), [30.0, 31.0])
    np.testing.assert_allclose(sorted(fp.data[:, 3]), [40.0, 41.0])


def test_one_layer_per_requested_flag(viewer_model, graphs_2d):
    gt, pred = graphs_2d
    layers = add_edge_error_tracks(viewer_model, gt, pred)
    assert set(layers) == {EdgeFlag.CTC_FALSE_NEG, EdgeFlag.CTC_FALSE_POS}
    assert len(viewer_model.layers) == 2


def test_empty_error_type_returns_none_not_missing_key(viewer_model, make_graph):
    # GT graph has a false negative but pred graph has no false positives.
    gt = make_graph(
        [(((0, 10.0, 20.0), (1, 11.0, 21.0)), EdgeFlag.CTC_FALSE_NEG)],
        ndim=2,
    )
    empty_pred = make_graph([], ndim=2)
    # Detection cannot use the empty pred graph, so pass ndim explicitly.
    layers = add_edge_error_tracks(viewer_model, gt, empty_pred, ndim=2)
    # Both requested flags are keys; the empty one maps to None (no KeyError trap).
    assert EdgeFlag.CTC_FALSE_POS in layers
    assert layers[EdgeFlag.CTC_FALSE_POS] is None
    assert layers[EdgeFlag.CTC_FALSE_NEG] is not None
    assert len(viewer_model.layers) == 1


def test_scale_is_propagated_to_layers(viewer_model, graphs_3d):
    gt, pred = graphs_3d
    layers = add_edge_error_tracks(viewer_model, gt, pred, scale=(3.0, 2.0, 4.0))
    fn = layers[EdgeFlag.CTC_FALSE_NEG]
    # napari prepends a unit time scale, so the full 3D+t scale is (1, z, y, x).
    assert tuple(fn.scale) == (1.0, 3.0, 2.0, 4.0)


def test_translate_is_propagated_to_layers(viewer_model, graphs_2d):
    gt, pred = graphs_2d
    layers = add_edge_error_tracks(viewer_model, gt, pred, translate=(5.0, 10.0))
    fn = layers[EdgeFlag.CTC_FALSE_NEG]
    # 2D+t: (t, y, x) with a zero time offset prepended.
    assert tuple(fn.translate) == (0.0, 5.0, 10.0)


def test_scale_wrong_length_raises(viewer_model, graphs_2d):
    gt, pred = graphs_2d
    with pytest.raises(ValueError, match="spatial entries"):
        # 2D expects 2 spatial entries, not 3.
        add_edge_error_tracks(viewer_model, gt, pred, scale=(1.0, 1.0, 1.0))


def test_translate_wrong_length_raises(viewer_model, graphs_2d):
    gt, pred = graphs_2d
    with pytest.raises(ValueError, match="spatial entries"):
        add_edge_error_tracks(viewer_model, gt, pred, translate=(1.0, 1.0, 1.0))


def test_tail_width_is_propagated(viewer_model, graphs_2d):
    gt, pred = graphs_2d
    layers = add_edge_error_tracks(viewer_model, gt, pred, tail_width=8)
    assert layers[EdgeFlag.CTC_FALSE_NEG].tail_width == 8


def test_color_by_and_default_colormaps(viewer_model, graphs_2d):
    gt, pred = graphs_2d
    layers = add_edge_error_tracks(viewer_model, gt, pred)
    fn = layers[EdgeFlag.CTC_FALSE_NEG]
    fp = layers[EdgeFlag.CTC_FALSE_POS]
    assert fn.color_by == "error"
    # Both error types default to "cool", passed via colormaps_dict so the value
    # maps raw (bypassing the Tracks 0-1 normalization), matching the original
    # renderer. Values are the enumerate index: FN 0, FP 1.
    assert fn.colormaps_dict["error"].name == "cool"
    assert fp.colormaps_dict["error"].name == "cool"
    import numpy as np

    assert np.unique(fn.properties["error"]).tolist() == [0.0]
    assert np.unique(fp.properties["error"]).tolist() == [1.0]


def test_colormaps_override(viewer_model, graphs_2d):
    gt, pred = graphs_2d
    layers = add_edge_error_tracks(
        viewer_model, gt, pred, colormaps={EdgeFlag.CTC_FALSE_NEG: "hot"}
    )
    assert layers[EdgeFlag.CTC_FALSE_NEG].colormaps_dict["error"].name == "hot"


def test_custom_error_flags_subset(viewer_model, graphs_2d):
    gt, pred = graphs_2d
    layers = add_edge_error_tracks(
        viewer_model, gt, pred, error_flags=(EdgeFlag.CTC_FALSE_NEG,)
    )
    assert set(layers) == {EdgeFlag.CTC_FALSE_NEG}
    assert len(viewer_model.layers) == 1


def test_multiple_edges_of_same_type(viewer_model, make_graph):
    gt = make_graph(
        [
            (((0, 10.0, 20.0), (1, 11.0, 21.0)), EdgeFlag.CTC_FALSE_NEG),
            (((0, 50.0, 60.0), (1, 51.0, 61.0)), EdgeFlag.CTC_FALSE_NEG),
        ],
        ndim=2,
    )
    empty_pred = make_graph([], ndim=2)
    layers = add_edge_error_tracks(viewer_model, gt, empty_pred, ndim=2)
    fn = layers[EdgeFlag.CTC_FALSE_NEG]
    # Two edges -> two 2-point tracklets -> 4 rows, 2 distinct track ids.
    assert fn.data.shape[0] == 4
    assert len(set(fn.data[:, 0])) == 2


def test_unsupported_flag_raises(viewer_model, graphs_2d):
    gt, pred = graphs_2d
    with pytest.raises(ValueError, match="Unsupported error flag"):
        add_edge_error_tracks(viewer_model, gt, pred, error_flags=(EdgeFlag.TRUE_POS,))


def test_detect_ndim_2d_and_3d(graphs_2d, graphs_3d):
    assert _detect_ndim(graphs_2d[0]) == 2
    assert _detect_ndim(graphs_3d[0]) == 3


def test_detect_ndim_rejects_mixed_graph():
    graph = nx.DiGraph()
    graph.add_node(0, t=0, y=1.0, x=2.0)  # no z
    graph.add_node(1, t=1, z=3.0, y=1.0, x=2.0)  # has z
    graph.add_edge(0, 1)
    with pytest.raises(ValueError, match="mixes 2D and 3D"):
        _detect_ndim(TrackingGraph(graph))


def test_detect_ndim_rejects_empty_graph(make_graph):
    with pytest.raises(ValueError, match="empty graph"):
        _detect_ndim(make_graph([], ndim=2))


def test_autodetect_from_pred_when_gt_empty(viewer_model, make_graph):
    # Only false positives wanted; the GT graph is empty but detection should
    # still succeed from the (non-empty) prediction graph.
    empty_gt = make_graph([], ndim=2)
    pred = make_graph(
        [(((0, 30.0, 40.0), (1, 31.0, 41.0)), EdgeFlag.CTC_FALSE_POS)],
        ndim=2,
    )
    layers = add_edge_error_tracks(
        viewer_model, empty_gt, pred, error_flags=(EdgeFlag.CTC_FALSE_POS,)
    )
    assert layers[EdgeFlag.CTC_FALSE_POS].data.shape[1] == 4


def test_autodetect_all_empty_raises(viewer_model, make_graph):
    with pytest.raises(ValueError, match="all requested graphs are empty"):
        add_edge_error_tracks(
            viewer_model, make_graph([], ndim=2), make_graph([], ndim=2)
        )
