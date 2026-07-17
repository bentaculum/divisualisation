"""Widget tests. Require a real Qt viewer, so they skip without pytest-qt.

These build a full napari GUI viewer, which needs a working OpenGL context.
That segfaults under offscreen Qt on macOS, so the module is skipped there; it
runs in CI on Linux with a virtual display (``xvfb-run``).
"""

import os
import sys

import pytest
from traccuracy import EdgeFlag

from divisualisation import add_edge_error_tracks

# pytest-qt installs as the module "pytestqt" (no underscore); the wrong name
# here would silently skip the test even when pytest-qt is available.
pytest.importorskip("pytestqt")

if sys.platform == "darwin" and os.environ.get("QT_QPA_PLATFORM") == "offscreen":
    pytest.skip(
        "napari GUI viewer segfaults under offscreen Qt on macOS",
        allow_module_level=True,
    )


def test_toggle_hides_and_shows_error_layers(make_napari_viewer, graphs_2d):
    from divisualisation._widget import ErrorToggleWidget

    viewer = make_napari_viewer()
    gt, pred = graphs_2d
    add_edge_error_tracks(viewer, gt, pred)

    widget = ErrorToggleWidget(viewer)
    fn_name = str(EdgeFlag.CTC_FALSE_NEG.value)
    fp_name = str(EdgeFlag.CTC_FALSE_POS.value)

    assert viewer.layers[fn_name].visible
    assert viewer.layers[fp_name].visible

    widget._show_errors.value = False
    assert not viewer.layers[fn_name].visible
    assert not viewer.layers[fp_name].visible

    widget._show_errors.value = True
    assert viewer.layers[fn_name].visible
    assert viewer.layers[fp_name].visible


def test_new_error_layer_added_after_widget_respects_toggle(
    make_napari_viewer, graphs_2d
):
    from divisualisation._widget import ErrorToggleWidget

    viewer = make_napari_viewer()
    widget = ErrorToggleWidget(viewer)
    widget._show_errors.value = False

    gt, pred = graphs_2d
    add_edge_error_tracks(viewer, gt, pred)

    # Layers added after the toggle was turned off must come in hidden.
    fn_name = str(EdgeFlag.CTC_FALSE_NEG.value)
    assert not viewer.layers[fn_name].visible


def test_spacetime_widget_toggles_lift(make_napari_viewer):
    import numpy as np

    from divisualisation._widget import SpacetimeWidget

    viewer = make_napari_viewer()
    viewer.add_image(np.random.rand(4, 8, 8), name="img")
    viewer.add_tracks(
        np.array([[1, 0, 2, 3], [1, 1, 2, 3], [1, 2, 2, 3]], float),
        name="GT tracks",
        tail_length=5,
    )

    widget = SpacetimeWidget(viewer)
    # The GT role dropdown is name-guessed to the "GT tracks" layer.
    assert widget._role_combos["gt"].value == "GT tracks"

    widget._lift_amount.value = 15
    widget._enabled.value = True
    layer = viewer.layers["GT tracks"]
    assert layer.data.shape[1] == 5
    np.testing.assert_allclose(layer.data[:, 2], 15 * layer.data[:, 1])
    assert viewer.dims.ndisplay == 3
    # GT role look applied on toggle-on.
    assert layer.tail_length == 1000

    widget._enabled.value = False
    layer = viewer.layers["GT tracks"]
    assert layer.data.shape[1] == 4
    assert viewer.dims.ndisplay == 2
    # Original display settings restored on toggle-off.
    assert layer.tail_length == 5
