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


def test_lift_all_widget_lifts_all_tracks(make_napari_viewer):
    from divisualisation._widget import LiftAllTracksWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.random.rand(4, 8, 8), name="img")
    for name in ("tracks a", "tracks b"):
        viewer.add_tracks(
            np.array([[1, 0, 2, 3], [1, 1, 2, 3], [1, 2, 2, 3]], float),
            name=name,
            tail_length=5,
        )

    widget = LiftAllTracksWidget(viewer)
    widget._lift_amount.value = 15
    widget._enabled.value = True

    # Both tracks layers lifted, keeping their own coloring (no error-view look).
    for name in ("tracks a", "tracks b"):
        layer = viewer.layers[name]
        assert layer.data.shape[1] == 5
        np.testing.assert_allclose(layer.data[:, 2], 15 * layer.data[:, 1])
        assert layer.color_by == "track_id"  # own coloring kept
    assert viewer.dims.ndisplay == 3

    widget._enabled.value = False
    for name in ("tracks a", "tracks b"):
        assert viewer.layers[name].data.shape[1] == 4
        assert viewer.layers[name].color_by == "track_id"  # restored
    assert viewer.dims.ndisplay == 2


def test_lift_all_widget_sync_matches_main(make_napari_viewer):
    from divisualisation._widget import LiftAllTracksWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((5, 8, 8)), name="img")
    viewer.add_tracks(np.array([[1, t, 4, 4] for t in range(5)], float), name="tracks")
    widget = LiftAllTracksWidget(viewer)
    widget._lift_amount.value = 10
    widget._enabled.value = True

    layer = viewer.layers["tracks"]
    viewer.dims.set_current_step(0, 3)
    # Verbatim main coupling: clip at t*scale=30, translate -t*scale=-30.
    enabled = [p for p in layer.experimental_clipping_planes if p.enabled]
    assert enabled and enabled[0].position[0] == pytest.approx(30)
    assert list(layer.translate) == pytest.approx([0, -30, 0, 0])


def test_errors_widget_applies_role_look(make_napari_viewer):
    from divisualisation._widget import ErrorsWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.random.rand(4, 8, 8), name="img")
    viewer.add_tracks(
        np.array([[1, 0, 2, 3], [1, 1, 2, 3], [1, 2, 2, 3]], float),
        name="GT tracks",
        tail_length=5,
    )

    widget = ErrorsWidget(viewer)
    assert widget._role_combos["gt"].value == "GT tracks"  # name-guessed

    widget._lift_amount.value = 15
    widget._enabled.value = True
    layer = viewer.layers["GT tracks"]
    assert layer.data.shape[1] == 5
    assert viewer.dims.ndisplay == 3
    assert layer.tail_length == 1000  # error-view look
    assert layer.color_by == "_lift_gt"

    widget._enabled.value = False
    layer = viewer.layers["GT tracks"]
    assert layer.data.shape[1] == 4
    assert viewer.dims.ndisplay == 2
    assert layer.tail_length == 5  # restored
    assert layer.color_by == "track_id"


def test_errors_widget_dropdowns_populate_before_lift(make_napari_viewer):
    import numpy as np

    from divisualisation._widget import ErrorsWidget

    viewer = make_napari_viewer()
    viewer.add_tracks(np.array([[1, 0, 2, 3], [1, 1, 2, 3]], float), name="GT tracks")
    viewer.add_tracks(
        np.array([[2, 0, 5, 5], [2, 1, 5, 5]], float), name="predicted tracks"
    )
    viewer.add_labels(np.zeros((2, 8, 8), int), name="gt masks")
    viewer.add_labels(np.zeros((2, 8, 8), int), name="pred masks")

    widget = ErrorsWidget(viewer)
    viewer.window.add_dock_widget(widget)
    # Let the deferred (QTimer.singleShot) re-guess fire after the dock reset.
    from qtpy.QtWidgets import QApplication

    QApplication.processEvents()
    QApplication.processEvents()

    # Roles/labels are usable immediately, without toggling the lift first.
    assert widget._role_combos["gt"].value == "GT tracks"
    assert widget._role_combos["pred"].value == "predicted tracks"
    assert widget._gt_labels.value == "gt masks"
    assert widget._pred_labels.value == "pred masks"


def test_lift_toggle_is_a_toggle_switch(make_napari_viewer):
    from superqt import QToggleSwitch

    from divisualisation._widget import LiftAllTracksWidget

    widget = LiftAllTracksWidget(make_napari_viewer())
    assert isinstance(widget._enabled.native, QToggleSwitch)
