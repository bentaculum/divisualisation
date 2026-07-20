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
    assert "GT tracks" in w._lift._track_bases

    # Changing the GT role re-applies the lift with the new selection.
    w._role_combos["gt"].value = "other tracks"
    assert "other tracks" in w._lift._track_bases
    assert w._lift.applied
