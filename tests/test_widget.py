"""Widget tests. Require a real Qt viewer, so they skip without pytest-qt.

These build a full napari GUI viewer, which needs a working OpenGL context.
That segfaults under offscreen Qt on macOS, so the module is skipped there; it
runs in CI on Linux with a virtual display (``xvfb-run``).
"""

import os
import sys

import numpy as np
import pytest

# pytest-qt installs as the module "pytestqt" (no underscore); the wrong name
# here would silently skip the test even when pytest-qt is available.
pytest.importorskip("pytestqt")

if sys.platform == "darwin" and os.environ.get("QT_QPA_PLATFORM") == "offscreen":
    pytest.skip(
        "napari GUI viewer segfaults under offscreen Qt on macOS",
        allow_module_level=True,
    )


def test_lift_all_keeps_coloring(make_napari_viewer):
    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.random.rand(4, 8, 8), name="img")
    for name in ("tracks a", "tracks b"):
        viewer.add_tracks(
            np.array([[1, 0, 2, 3], [1, 1, 2, 3], [1, 2, 2, 3]], float),
            name=name,
            tail_length=5,
        )

    w = SpacetimeWidget(viewer)
    w._lift_amount.value = 15
    w._lift_all.value = True

    # Both tracks layers lifted, keeping their own coloring.
    for name in ("tracks a", "tracks b"):
        layer = viewer.layers[name]
        assert layer.data.shape[1] == 5
        np.testing.assert_allclose(layer.data[:, 2], -15 * layer.data[:, 1])
        assert layer.color_by == "track_id"
    assert viewer.dims.ndisplay == 3

    w._lift_all.value = False
    for name in ("tracks a", "tracks b"):
        assert viewer.layers[name].data.shape[1] == 4
        assert viewer.layers[name].color_by == "track_id"
    assert viewer.dims.ndisplay == 2


def test_lift_all_sync_matches_main(make_napari_viewer):
    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((5, 8, 8)), name="img")
    viewer.add_tracks(np.array([[1, t, 4, 4] for t in range(5)], float), name="tracks")
    w = SpacetimeWidget(viewer)
    w._lift_amount.value = 10
    w._lift_all.value = True

    layer = viewer.layers["tracks"]
    viewer.dims.set_current_step(0, 3)
    # Flipped-axis coupling: clip at -t*scale=-30, translate +t*scale=30.
    enabled = [p for p in layer.experimental_clipping_planes if p.enabled]
    assert enabled and enabled[0].position[0] == pytest.approx(-30)
    assert list(layer.translate) == pytest.approx([0, 30, 0, 0])


def test_errors_toggle_applies_role_look(make_napari_viewer):
    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.random.rand(4, 8, 8), name="img")
    viewer.add_tracks(
        np.array([[1, 0, 2, 3], [1, 1, 2, 3], [1, 2, 2, 3]], float),
        name="GT tracks",
        tail_length=5,
    )

    w = SpacetimeWidget(viewer)
    assert w._role_combos["gt"].value == "GT tracks"  # name-guessed

    w._lift_amount.value = 15
    w._lift_errors.value = True
    layer = viewer.layers["GT tracks"]
    assert layer.data.shape[1] == 5
    assert viewer.dims.ndisplay == 3
    assert layer.tail_length == 1000  # error-view look
    assert layer.color_by == "_lift_gt"

    w._lift_errors.value = False
    layer = viewer.layers["GT tracks"]
    assert layer.data.shape[1] == 4
    assert viewer.dims.ndisplay == 2
    assert layer.tail_length == 5  # restored
    assert layer.color_by == "track_id"


def test_toggles_are_mutually_exclusive(make_napari_viewer):
    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 8, 8)), name="img")
    viewer.add_tracks(
        np.array([[1, t, 2, 3] for t in range(4)], float), name="GT tracks"
    )
    w = SpacetimeWidget(viewer)
    w._lift_all.value = True
    assert w._lift_all.value and not w._lift_errors.value
    # Turning on errors turns off lift-all.
    w._lift_errors.value = True
    assert w._lift_errors.value and not w._lift_all.value
    assert viewer.dims.ndisplay == 3
    w._lift_errors.value = False
    assert viewer.dims.ndisplay == 2


def test_errors_controls_hidden_until_errors_toggle(make_napari_viewer):
    from qtpy.QtWidgets import QApplication

    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((2, 8, 8)), name="raw")
    viewer.add_tracks(np.array([[1, 0, 2, 3], [1, 1, 6, 6]], float), name="GT tracks")
    viewer.add_tracks(
        np.array([[2, 0, 5, 5], [2, 1, 3, 3]], float), name="predicted tracks"
    )
    viewer.add_labels(np.zeros((2, 8, 8), int), name="gt masks")
    viewer.add_labels(np.zeros((2, 8, 8), int), name="pred masks")

    w = SpacetimeWidget(viewer)
    w.show()  # show the container so child .visible reflects the set value
    QApplication.processEvents()
    QApplication.processEvents()

    # Error controls hidden until the errors toggle is on.
    assert not w._compute_btn.visible
    w._lift_errors.value = True
    assert w._compute_btn.visible
    # Roles/labels are name-guessed and usable immediately (before any lift).
    assert w._role_combos["gt"].value == "GT tracks"
    assert w._gt_labels.value == "gt masks"


def test_toggles_are_toggle_switches(make_napari_viewer):
    from superqt import QToggleSwitch

    from divisualisation._widget import SpacetimeWidget

    w = SpacetimeWidget(make_napari_viewer())
    assert isinstance(w._lift_all.native, QToggleSwitch)
    assert isinstance(w._lift_errors.native, QToggleSwitch)


def test_role_change_relifts_live(make_napari_viewer):
    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((6, 20, 20)), name="raw")
    viewer.add_tracks(
        np.array([[1, t, 5, 5] for t in range(6)], float), name="GT tracks"
    )
    viewer.add_tracks(
        np.array([[2, t, 8, 8] for t in range(6)], float), name="other tracks"
    )
    w = SpacetimeWidget(viewer)
    w._lift_errors.value = True

    # Dropdowns are editable while the Divisualisation toggle is on.
    assert w._role_combos["gt"].enabled
    assert viewer.layers["GT tracks"] in w._lift._track_bases

    # Changing the GT role re-applies the lift with the new selection.
    w._role_combos["gt"].value = "other tracks"
    assert viewer.layers["other tracks"] in w._lift._track_bases
    assert w._lift.applied


def test_compute_during_lift_reapplies_and_keeps_dropdowns(
    make_napari_viewer, monkeypatch
):
    import numpy as np
    from traccuracy import EdgeFlag

    import divisualisation.errors as errors_mod
    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 8, 8)), name="raw")
    for nm in ("GT tracks", "predicted tracks"):
        viewer.add_tracks(
            np.array([[1, t, 2, 3] for t in range(4)], float),
            name=nm,
        )
    viewer.add_labels(np.zeros((4, 8, 8), int), name="gt masks")
    viewer.add_labels(np.zeros((4, 8, 8), int), name="pred masks")

    # Stub the heavy CTC computation: just add two flat error tracks layers.
    def fake_compute(v, gt_t, gt_l, pred_t, pred_l, **kw):
        out = {}
        for flag in (EdgeFlag.CTC_FALSE_NEG, EdgeFlag.CTC_FALSE_POS):
            layer = v.add_tracks(
                np.array([[1, 0, 2, 3], [1, 1, 2, 3]], float),
                name=str(flag.value),
            )
            out[flag] = layer
        return out

    monkeypatch.setattr(errors_mod, "compute_edge_errors_from_layers", fake_compute)

    w = SpacetimeWidget(viewer)
    w._lift_errors.value = True  # lift GT/pred
    assert viewer.layers["GT tracks"] in w._lift._track_bases

    w._on_compute()  # compute (stubbed) while lifted

    # GT/pred stay lifted; the new error layers are lifted + role-colored.
    for nm in ("GT tracks", "predicted tracks"):
        assert viewer.layers[nm] in w._lift._track_bases
        assert viewer.layers[nm].data.shape[1] == 5
    fn = str(EdgeFlag.CTC_FALSE_NEG.value)
    assert viewer.layers[fn] in w._lift._track_bases
    assert viewer.layers[fn].color_by == "_lift_fn_edges"
    # Dropdowns still work and include the new error layers.
    assert fn in w._role_combos["fn_edges"].choices
    assert w._role_combos["gt"].enabled


def test_compute_with_division_edges_sees_real_graph(make_napari_viewer, monkeypatch):
    # Regression: with colored division edges on, computing errors must run on
    # the GT layer's REAL graph, not the zeroed one we swap in for the overlay.
    import numpy as np
    from traccuracy import EdgeFlag

    import divisualisation.errors as errors_mod
    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 20, 20)), name="raw")
    _add_dividing_gt(viewer, name="GT tracks")
    _add_dividing_gt(viewer, name="predicted tracks")
    viewer.add_labels(np.zeros((4, 20, 20), int), name="gt masks")
    viewer.add_labels(np.zeros((4, 20, 20), int), name="pred masks")

    seen = {}

    def fake_compute(v, gt_t, gt_l, pred_t, pred_l, **kw):
        seen["gt_graph"] = dict(gt_t.graph)  # graph visible at compute time
        out = {}
        for flag in (EdgeFlag.CTC_FALSE_NEG, EdgeFlag.CTC_FALSE_POS):
            out[flag] = v.add_tracks(
                np.array([[1, 0, 2, 3], [1, 1, 2, 3]], float), name=str(flag.value)
            )
        return out

    monkeypatch.setattr(errors_mod, "compute_edge_errors_from_layers", fake_compute)

    w = SpacetimeWidget(viewer)
    w._division_edges.value = True
    w._lift_errors.value = True
    # While active the GT layer's graph is zeroed for the colored overlay...
    assert not dict(viewer.layers["GT tracks"].graph)

    w._on_compute()
    # ...but compute saw the real division graph, not the empty one.
    assert seen["gt_graph"]  # non-empty: divisions present
    # Edges are rebuilt after compute and GT is re-suppressed.
    assert any(ly.name.endswith("division edges") for ly in viewer.layers)


def test_camera_shared_display_per_view(make_napari_viewer):
    import numpy as np

    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((6, 20, 20)), name="raw")
    viewer.add_tracks(
        np.array([[1, t, 5, 5] for t in range(6)], float),
        name="GT tracks",
        tail_width=2,
    )

    w = SpacetimeWidget(viewer)
    # Lift-all view: tweak the shared camera and a per-view display param.
    w._lift_all.value = True
    viewer.camera.zoom = 3.3
    viewer.layers["GT tracks"].tail_width = 9

    # Switch to the errors view.
    w._lift_errors.value = True
    # Camera is shared across views.
    assert viewer.camera.zoom == pytest.approx(3.3)
    # Display is NOT shared: the errors view has its own tail_width.
    assert viewer.layers["GT tracks"].tail_width != 9

    # Back to lift-all: its own display tweak is remembered; camera still shared.
    w._lift_all.value = True
    assert viewer.layers["GT tracks"].tail_width == 9
    assert viewer.camera.zoom == pytest.approx(3.3)


def test_role_reassign_dedups_and_lifts_all(make_napari_viewer):
    import numpy as np

    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((6, 20, 20)), name="raw")
    for nm in ("GT tracks", "predicted tracks", "ctc_fp"):
        viewer.add_tracks(np.array([[1, t, 5, 5] for t in range(6)], float), name=nm)

    w = SpacetimeWidget(viewer)
    # gt -> GT tracks, fp_edges -> ctc_fp (name-guessed)
    w._lift_errors.value = True
    assert w._role_combos["fp_edges"].value == "ctc_fp"

    # Reassign ctc_fp to the gt role: it must leave fp_edges (a layer fills one
    # role), and every declared role layer must be lifted (none left flat).
    w._role_combos["gt"].value = "ctc_fp"
    assert w._role_combos["gt"].value == "ctc_fp"
    assert w._role_combos["fp_edges"].value == "—"  # cleared
    roles = w._roles_target()
    for name in roles.values():
        assert viewer.layers[name].data.shape[1] == 5  # lifted
    # Every tracks layer lifts in the Divisualisation view; a layer dropped from
    # all roles stays lifted but reverts to its own coloring (no role color).
    gt = viewer.layers["GT tracks"]
    assert gt.data.shape[1] == 5
    assert gt.color_by == "track_id"


def test_divisualisation_lifts_all_tracks_incl_non_role(make_napari_viewer):
    import numpy as np

    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((6, 20, 20)), name="raw")
    viewer.add_tracks(
        np.array([[1, t, 5, 5] for t in range(6)], float), name="GT tracks"
    )
    extra = viewer.add_tracks(
        np.array([[2, t, 9, 9] for t in range(6)], float), name="extra tracks"
    )
    extra.visible = False  # hidden, and not a named role

    w = SpacetimeWidget(viewer)
    w._lift_errors.value = True  # Divisualisation workflow

    # Role layer lifts with error-view color.
    assert viewer.layers["GT tracks"].color_by == "_lift_gt"
    # The hidden, non-role tracks layer is lifted too, keeping its own coloring.
    ex = viewer.layers["extra tracks"]
    assert ex.data.shape[1] == 5
    assert ex.color_by == "track_id"
    # Still lifted after being shown.
    ex.visible = True
    assert viewer.layers["extra tracks"].data.shape[1] == 5


def test_layer_hidden_at_lift_time_folds_when_shown(make_napari_viewer):
    # THE recurring bug: a tracks layer hidden when the lift is applied, then
    # unhidden in the lifted (3D) view, must be properly folded (5-col, real
    # extent) -- not left flat -- and must not crash the 3D draw.
    import numpy as np
    from qtpy.QtWidgets import QApplication

    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((10, 30, 30)), name="raw")
    viewer.add_tracks(
        np.array([[1, t, 5, 5] for t in range(10)], float), name="GT tracks"
    )
    hidden = viewer.add_tracks(
        np.array([[2, t, 20, 20] for t in range(10)], float), name="extra tracks"
    )
    hidden.visible = False  # hidden in the flat view

    w = SpacetimeWidget(viewer)
    w._lift_amount.value = 12
    w._lift_errors.value = True  # -> ndisplay 3, lifts all incl. the hidden one
    QApplication.processEvents()

    # Unhide it in the lifted view -> must be folded, with a 4-col (ndim=4)
    # extent, and the 3D draw must not raise.
    hidden.visible = True
    QApplication.processEvents()
    layer = viewer.layers["extra tracks"]
    assert layer.data.shape[1] == 5  # folded, not flat
    np.testing.assert_allclose(layer.data[:, 2], -12 * layer.data[:, 1])
    assert layer.extent.data.shape[1] == 4  # ndim=4 extent (not stale 3-col)
    # Force a canvas draw: without the fix the hidden layer keeps a stale 3-col
    # extent and the 3D draw raises IndexError (reproducible with a real GL
    # context, i.e. Linux CI; a harmless no-op where the canvas can't render).
    canvas = viewer.window._qt_viewer.canvas._scene_canvas
    canvas.update()
    canvas.events.draw()
    QApplication.processEvents()


def test_divisualisation_hides_predicted_by_default(make_napari_viewer):
    import numpy as np

    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((6, 20, 20)), name="raw")
    viewer.add_tracks(
        np.array([[1, t, 5, 5] for t in range(6)], float), name="GT tracks"
    )
    viewer.add_tracks(
        np.array([[2, t, 8, 8] for t in range(6)], float), name="predicted tracks"
    )

    w = SpacetimeWidget(viewer)
    assert viewer.layers["predicted tracks"].visible
    w._lift_errors.value = True
    # Predicted is hidden by default in the Divisualisation view...
    assert not viewer.layers["predicted tracks"].visible
    w._lift_errors.value = False
    # ...and restored to its prior visibility on toggle-off.
    assert viewer.layers["predicted tracks"].visible


def _add_dividing_gt(viewer, name="GT tracks"):
    """Add a GT tracks layer whose graph has one division (built like the
    examples, with the shared division node dropped)."""
    import networkx as nx

    from divisualisation.utils import graph_to_napari_tracks

    g = nx.DiGraph()
    # 0 (t0) -> 1 (t1) divides into 2 and 3 (t2), each continues to t3.
    coords = {
        0: (0, 5.0, 5.0),
        1: (1, 6.0, 6.0),
        2: (2, 7.0, 4.0),
        3: (2, 7.0, 8.0),
        4: (3, 8.0, 3.0),
        5: (3, 8.0, 9.0),
    }
    for n, (t, y, x) in coords.items():
        g.add_node(n, t=t, y=y, x=x)
    for u, v in [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5)]:
        g.add_edge(u, v)
    tracks, tracks_graph, _ = graph_to_napari_tracks(
        g, include_z=False, drop_division_duplicates=True
    )
    return viewer.add_tracks(tracks, graph=tracks_graph, name=name, tail_length=5)


def _edge_layers(viewer):
    return [ly for ly in viewer.layers if ly.name.endswith("division edges")]


def test_division_edges_off_by_default(make_napari_viewer):
    import numpy as np

    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 20, 20)), name="raw")
    _add_dividing_gt(viewer)

    w = SpacetimeWidget(viewer)
    assert w._division_edges.value is False
    # Toggling Divisualisation without checking the box adds no edge layers.
    w._lift_errors.value = True
    assert _edge_layers(viewer) == []
    # The GT layer keeps its own (white) graph edges -- graph not suppressed.
    assert dict(viewer.layers["GT tracks"].graph)


def test_division_edges_build_and_teardown(make_napari_viewer):
    import numpy as np

    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 20, 20)), name="raw")
    _add_dividing_gt(viewer)

    w = SpacetimeWidget(viewer)
    w._division_edges.value = True
    w._lift_errors.value = True

    # One colored edge layer for the GT role; it is lifted (5 columns) with the
    # rest, and the source layer's white graph is suppressed while it stands in.
    edges = _edge_layers(viewer)
    assert len(edges) == 1
    edge_layer = edges[0]
    assert edge_layer.data.shape[1] == 5  # lifted alongside everything else
    assert edge_layer.color_by == "_div"
    assert not dict(viewer.layers["GT tracks"].graph)  # native edges suppressed

    # Toggle Divisualisation off: edge layer gone, GT graph restored, flat again.
    w._lift_errors.value = False
    assert _edge_layers(viewer) == []
    assert dict(viewer.layers["GT tracks"].graph)  # restored
    assert viewer.layers["GT tracks"].data.shape[1] == 4


def test_division_edges_checkbox_live_toggle(make_napari_viewer):
    import numpy as np

    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 20, 20)), name="raw")
    _add_dividing_gt(viewer)

    w = SpacetimeWidget(viewer)
    w._lift_errors.value = True
    assert _edge_layers(viewer) == []  # box off

    # Checking the box while Divisualisation is active builds the edges live...
    w._division_edges.value = True
    assert len(_edge_layers(viewer)) == 1
    assert not dict(viewer.layers["GT tracks"].graph)

    # ...and unchecking removes them and restores the source graph, still lifted.
    w._division_edges.value = False
    assert _edge_layers(viewer) == []
    assert dict(viewer.layers["GT tracks"].graph)
    assert viewer.layers["GT tracks"].data.shape[1] == 5  # still lifted


def test_division_edges_not_in_dropdowns(make_napari_viewer):
    import numpy as np

    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 20, 20)), name="raw")
    _add_dividing_gt(viewer)

    w = SpacetimeWidget(viewer)
    w._division_edges.value = True
    w._lift_errors.value = True

    # The widget-owned edge overlay must never appear as a selectable role.
    assert len(_edge_layers(viewer)) == 1
    assert not any(name.endswith("division edges") for name in w._track_layer_names())
    # It also stays visible (not swept into the hidden set) under the errors view.
    assert _edge_layers(viewer)[0].visible
