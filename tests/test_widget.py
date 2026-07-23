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
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
    w._lift_scale.value = 15
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
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
    w._lift_scale.value = 10
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
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
    assert w._role_combos["gt"].value == "GT tracks"  # name-guessed

    w._lift_scale.value = 15
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
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
    w._lift_all.value = True
    assert w._lift_all.value and not w._lift_errors.value
    # Turning on errors turns off lift-all.
    w._lift_errors.value = True
    assert w._lift_errors.value and not w._lift_all.value
    assert viewer.dims.ndisplay == 3
    w._lift_errors.value = False
    assert viewer.dims.ndisplay == 2


def test_errors_controls_visible_before_toggle(make_napari_viewer):
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
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
    w.show()  # show the container so child .visible reflects the set value
    QApplication.processEvents()
    QApplication.processEvents()

    # The error controls (Compute button, role/labels dropdowns) are shown and
    # usable from the start -- before the Divisualisation toggle is on -- so
    # layers can be picked up front. They stay visible once the toggle goes on.
    assert w._compute_btn.visible
    assert w._role_combos["gt"].value == "GT tracks"  # name-guessed
    assert w._gt_labels.value == "gt masks"
    w._lift_errors.value = True
    assert w._compute_btn.visible


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
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
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
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
    w._lift_errors.value = True  # lift GT/pred
    assert viewer.layers["GT tracks"] in w._lift._track_bases

    w._on_compute()  # compute (stubbed) while lifted

    # GT/pred stay lifted; the new error layers are lifted + role-colored.
    for nm in ("GT tracks", "predicted tracks"):
        assert viewer.layers[nm] in w._lift._track_bases
        assert viewer.layers[nm].data.shape[1] == 5
    fn = str(EdgeFlag.CTC_FALSE_NEG.value)
    fp = str(EdgeFlag.CTC_FALSE_POS.value)
    assert viewer.layers[fn] in w._lift._track_bases
    assert viewer.layers[fn].color_by == "_lift_fn_edges"
    # Dropdowns still work and include the new error layers.
    assert fn in w._role_combos["fn_edges"].choices
    assert w._role_combos["gt"].enabled
    # Compute assigns the new layers to the FN/FP role dropdowns (the combos are
    # nested in the per-workflow box, so this relies on resetting their choices
    # directly, not via the top-level container).
    assert w._role_combos["fn_edges"].value == fn
    assert w._role_combos["fp_edges"].value == fp
    # ... and shows them by default (not swept into the hidden set).
    assert viewer.layers[fn].visible
    assert viewer.layers[fp].visible


def test_compute_passes_gt_spatial_scale_to_error_overlays(
    make_napari_viewer, monkeypatch
):
    # The error overlays must inherit the GT tracks layer's SPATIAL scale (e.g.
    # an anisotropic z shown 10x) so they align with the data instead of
    # rendering at unit scale in the wrong z plane.
    import numpy as np
    from traccuracy import EdgeFlag

    import divisualisation.errors as errors_mod
    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 3, 8, 8), np.uint8), name="raw", scale=(1, 10, 1, 1))
    for nm in ("GT tracks", "predicted tracks"):
        viewer.add_tracks(
            np.array([[1, t, 1, 2, 3] for t in range(4)], float),
            name=nm,
            scale=(1, 10, 1, 1),  # (t, z, y, x)
        )
    viewer.add_labels(np.zeros((4, 3, 8, 8), int), name="gt masks", scale=(1, 10, 1, 1))
    viewer.add_labels(
        np.zeros((4, 3, 8, 8), int), name="pred masks", scale=(1, 10, 1, 1)
    )

    seen = {}

    def fake_compute(v, gt_t, gt_l, pred_t, pred_l, **kw):
        seen["scale"] = kw.get("scale")
        return {f: None for f in (EdgeFlag.CTC_FALSE_NEG, EdgeFlag.CTC_FALSE_POS)}

    monkeypatch.setattr(errors_mod, "compute_edge_errors_from_layers", fake_compute)

    w = SpacetimeWidget(viewer)
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
    w._role_combos["gt"].value = "GT tracks"
    w._role_combos["pred"].value = "predicted tracks"
    w._gt_labels.value = "gt masks"
    w._pred_labels.value = "pred masks"
    w._on_compute()

    # Spatial part of (t, z, y, x) -> (z, y, x); z carries the 10x.
    assert seen["scale"] == (10.0, 1.0, 1.0)


def test_compute_with_division_edges_sees_real_graph(make_napari_viewer, monkeypatch):
    # Regression: with colored division edges on, the GT layer is augmented in
    # place (its graph moves into the tail). Computing errors must see the layer
    # restored to its ORIGINAL graph + data, not the augmented state.
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
    n0 = len(viewer.layers["GT tracks"].data)

    seen = {}

    def fake_compute(v, gt_t, gt_l, pred_t, pred_l, **kw):
        seen["gt_graph"] = dict(gt_t.graph)  # graph visible at compute time
        seen["gt_rows"] = len(gt_t.data)  # data visible at compute time
        out = {}
        for flag in (EdgeFlag.CTC_FALSE_NEG, EdgeFlag.CTC_FALSE_POS):
            out[flag] = v.add_tracks(
                np.array([[1, 0, 2, 3], [1, 1, 2, 3]], float), name=str(flag.value)
            )
        return out

    monkeypatch.setattr(errors_mod, "compute_edge_errors_from_layers", fake_compute)

    w = SpacetimeWidget(viewer)
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
    w._division_edges.value = True
    w._lift_errors.value = True
    assert viewer.layers["GT tracks"].display_graph is False  # native edges off

    w._on_compute()
    assert seen["gt_graph"]  # compute saw the real division graph
    assert seen["gt_rows"] == n0  # and the un-augmented data
    # After compute the layer is re-augmented and native edges stay off.
    assert viewer.layers["GT tracks"].display_graph is False


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
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
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
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
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
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
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
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
    w._lift_scale.value = 12
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
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
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


def _n_division_rows(viewer, name="GT tracks"):
    # The dividing GT graph has two divisions (1->2, 1->3), so augmentation adds
    # exactly two vertices to the layer's data.
    return 2


def test_division_edges_off_by_default(make_napari_viewer):
    import numpy as np

    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 20, 20)), name="raw")
    _add_dividing_gt(viewer)
    n0 = len(viewer.layers["GT tracks"].data)

    w = SpacetimeWidget(viewer)
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
    assert w._division_edges.value is False
    # Toggling Divisualisation without checking the box leaves the layer alone.
    w._lift_errors.value = True
    assert len(viewer.layers["GT tracks"].data) == n0  # no vertices added
    # The GT layer still shows its own (white) graph edges -- not turned off.
    assert viewer.layers["GT tracks"].display_graph is True


def test_division_edges_build_and_teardown(make_napari_viewer):
    import numpy as np

    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 20, 20)), name="raw")
    _add_dividing_gt(viewer)
    n0 = len(viewer.layers["GT tracks"].data)

    w = SpacetimeWidget(viewer)
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
    w._division_edges.value = True
    w._lift_errors.value = True

    # The GT layer is edited IN PLACE (no separate overlay): a vertex per
    # division is appended so the divisions draw as its own tail, its native
    # white graph edges are turned off, and it lifts (5 cols) with everything.
    gt = viewer.layers["GT tracks"]
    assert not any(ly.name.endswith("division edges") for ly in viewer.layers)
    assert len(gt.data) == n0 + _n_division_rows(viewer)
    assert gt.data.shape[1] == 5  # lifted
    assert gt.display_graph is False  # native edges hidden

    # Toggle Divisualisation off: data + native edges restored, flat again.
    w._lift_errors.value = False
    gt = viewer.layers["GT tracks"]
    assert len(gt.data) == n0
    assert gt.display_graph is True  # restored
    assert dict(gt.graph)  # graph restored
    assert gt.data.shape[1] == 4


def test_division_edges_augment_gt_and_pred(make_napari_viewer):
    import numpy as np

    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 20, 20)), name="raw")
    _add_dividing_gt(viewer, name="GT tracks")
    _add_dividing_gt(viewer, name="predicted tracks")
    n_gt = len(viewer.layers["GT tracks"].data)
    n_pred = len(viewer.layers["predicted tracks"].data)

    w = SpacetimeWidget(viewer)
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
    w._role_combos["gt"].value = "GT tracks"
    w._role_combos["pred"].value = "predicted tracks"
    w._division_edges.value = True
    w._lift_errors.value = True

    # Both GT and predicted role layers get their division edges colored in place
    # (predicted too, even though the view hides it by default).
    gt = viewer.layers["GT tracks"]
    pred = viewer.layers["predicted tracks"]
    assert len(gt.data) == n_gt + _n_division_rows(viewer)
    assert gt.display_graph is False
    assert len(pred.data) == n_pred + _n_division_rows(viewer)
    assert pred.display_graph is False

    # Toggle off restores both.
    w._lift_errors.value = False
    assert len(viewer.layers["GT tracks"].data) == n_gt
    assert len(viewer.layers["predicted tracks"].data) == n_pred
    assert viewer.layers["predicted tracks"].display_graph is True


def test_division_edges_checkbox_live_toggle(make_napari_viewer):
    import numpy as np

    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 20, 20)), name="raw")
    _add_dividing_gt(viewer)
    n0 = len(viewer.layers["GT tracks"].data)

    w = SpacetimeWidget(viewer)
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
    w._lift_errors.value = True
    assert len(viewer.layers["GT tracks"].data) == n0  # box off, untouched

    # Checking the box while Divisualisation is active augments the layer live...
    w._division_edges.value = True
    assert len(viewer.layers["GT tracks"].data) == n0 + _n_division_rows(viewer)
    assert viewer.layers["GT tracks"].display_graph is False

    # ...and unchecking restores the data + native edges, still lifted.
    w._division_edges.value = False
    gt = viewer.layers["GT tracks"]
    assert len(gt.data) == n0
    assert gt.display_graph is True
    assert gt.data.shape[1] == 5  # still lifted


def test_division_edges_checkbox_preserves_visibility(make_napari_viewer):
    # Toggling the color-edges checkbox must NOT change layer visibility (it must
    # not re-hide the predicted layer the way the Divisualisation toggle does).
    import numpy as np

    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 20, 20)), name="raw")
    _add_dividing_gt(viewer, name="GT tracks")
    _add_dividing_gt(viewer, name="predicted tracks")

    w = SpacetimeWidget(viewer)
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
    w._role_combos["gt"].value = "GT tracks"
    w._role_combos["pred"].value = "predicted tracks"
    w._division_edges.value = True
    w._lift_errors.value = True
    # Divisualisation hides pred by default; the user then shows it.
    assert viewer.layers["predicted tracks"].visible is False
    viewer.layers["predicted tracks"].visible = True

    # Toggling the coloring checkbox leaves visibility exactly as-is.
    w._division_edges.value = False
    assert viewer.layers["predicted tracks"].visible is True
    assert viewer.layers["GT tracks"].visible is True
    w._division_edges.value = True
    assert viewer.layers["predicted tracks"].visible is True
    assert viewer.layers["GT tracks"].visible is True


def test_division_edges_hidden_layer_folds_when_shown(make_napari_viewer):
    # Augmenting a HIDDEN layer must still leave it correctly lifted: when later
    # shown it should render folded (5-col data, z<0, 4D extent), not flat at z=0.
    import numpy as np

    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 20, 20)), name="raw")
    _add_dividing_gt(viewer, name="GT tracks")
    _add_dividing_gt(viewer, name="predicted tracks")

    w = SpacetimeWidget(viewer)
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
    w._role_combos["gt"].value = "GT tracks"
    w._role_combos["pred"].value = "predicted tracks"
    w._lift_errors.value = True  # pred hidden by default
    # Check the box while pred is hidden, so its augmentation happens hidden.
    w._division_edges.value = True
    pred = viewer.layers["predicted tracks"]
    assert pred.visible is False

    pred.visible = True
    data = np.asarray(pred.data)
    assert data.shape[1] == 5  # lifted
    assert data[:, 2].min() < 0  # folded into z, not flat at 0
    assert pred.extent.data.shape[1] == 4  # 4D extent (not a stale 3D one)


def test_division_edges_keep_layer_coloring(make_napari_viewer):
    import numpy as np

    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 20, 20)), name="raw")
    _add_dividing_gt(viewer)

    w = SpacetimeWidget(viewer)
    w._guess_once()  # deferred in the widget; run now so name-guessed roles are set
    w._division_edges.value = True
    w._lift_errors.value = True

    # The appended division vertices are colored like the rest of the GT layer:
    # the color property spans every row, so no vertex is left uncolored.
    gt = viewer.layers["GT tracks"]
    assert len(gt.properties[gt.color_by]) == len(gt.data)
    assert gt.track_colors is not None and len(gt.track_colors) == len(gt.data)
