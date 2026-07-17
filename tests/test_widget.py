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
