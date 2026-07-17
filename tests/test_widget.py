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


def test_visualize_widget_lifts_all_tracks(make_napari_viewer):
    from divisualisation._widget import VisualizeTracksWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.random.rand(4, 8, 8), name="img")
    for name in ("tracks a", "tracks b"):
        viewer.add_tracks(
            np.array([[1, 0, 2, 3], [1, 1, 2, 3], [1, 2, 2, 3]], float),
            name=name,
            tail_length=5,
        )

    widget = VisualizeTracksWidget(viewer)
    widget._lift_amount.value = 15
    widget._enabled.value = True

    # Both tracks layers lifted, keeping their own coloring (no error-view look).
    for name in ("tracks a", "tracks b"):
        layer = viewer.layers[name]
        assert layer.data.shape[1] == 5
        np.testing.assert_allclose(layer.data[:, 2], 15 * layer.data[:, 1])
        assert layer.color_by == "track_id"
        assert layer.tail_length == 5
    assert viewer.dims.ndisplay == 3

    widget._enabled.value = False
    for name in ("tracks a", "tracks b"):
        assert viewer.layers[name].data.shape[1] == 4
    assert viewer.dims.ndisplay == 2


def test_visualize_widget_sync_matches_main(make_napari_viewer):
    from divisualisation._widget import VisualizeTracksWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((5, 8, 8)), name="img")
    viewer.add_tracks(np.array([[1, t, 4, 4] for t in range(5)], float), name="tracks")
    widget = VisualizeTracksWidget(viewer)
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
