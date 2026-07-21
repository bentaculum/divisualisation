"""Visibility / control-layout tests for the Divisualisation widget.

These drive the widget's visibility bookkeeping and control layout directly on a
``ViewerModel`` (no GUI canvas, no 3D lift), so unlike ``test_widget.py`` they
run under offscreen Qt on macOS too. Toggling the lift itself needs a real GL
context (see ``test_widget.py``); here we call the hide/restore helpers directly.
"""

import numpy as np
import pytest
from napari.components import ViewerModel

pytest.importorskip("magicgui")

from divisualisation._widget import SpacetimeWidget


def _make_viewer():
    v = ViewerModel()
    v.add_image(np.zeros((5, 8, 8)), name="raw")
    v.add_labels(np.zeros((5, 8, 8), int), name="gt masks")
    v.add_labels(np.zeros((5, 8, 8), int), name="pred masks")
    v.add_tracks(np.array([[1, t, 5, 5] for t in range(5)], float), name="gt tracks")
    v.add_tracks(np.array([[2, t, 6, 6] for t in range(5)], float), name="pred tracks")
    v.add_tracks(np.array([[3, t, 7, 7] for t in range(5)], float), name="other tracks")
    return v


def _visible(viewer):
    return {layer.name: layer.visible for layer in viewer.layers}


def test_error_controls_visible_before_toggle():
    # The six role/labels dropdowns and the Compute button must be present and
    # enabled from the start, before the Divisualisation toggle is turned on
    # (previously they were hidden until the toggle went on).
    v = _make_viewer()
    w = SpacetimeWidget(v)
    w._guess_once()  # run the one-time auto-guess (normally a deferred tick)
    assert not w._lift_errors.value  # toggle off
    controls = (*w._role_combos.values(), w._gt_labels, w._pred_labels, w._compute_btn)
    for widget in controls:
        # magicgui reports .visible == False until the parent container is shown,
        # so check enabled + membership here and .visible after show() below.
        assert widget.enabled
        assert widget in list(w)

    # Once the container is shown (as when docked), the controls are visible even
    # with the toggle still off.
    w.show()
    try:
        assert not w._lift_errors.value
        for widget in controls:
            assert widget.visible
    finally:
        w.hide()


def test_hide_unselected_hides_only_unselected_tracks():
    v = _make_viewer()
    w = SpacetimeWidget(v)
    w._guess_once()  # run the one-time auto-guess (normally a deferred tick)
    # Role-guessing picks gt/pred tracks + gt/pred masks; "other tracks" is an
    # unselected tracks layer; "raw" is an (always-untouched) image layer.
    w._hide_unselected()
    vis = _visible(v)
    # Selected tracks stay visible.
    assert vis["gt tracks"]
    # Non-tracks layers are never hidden, selected or not.
    assert vis["raw"]  # image, untouched
    assert vis["gt masks"] and vis["pred masks"]  # labels, untouched
    # Unselected TRACKS layers are hidden.
    assert not vis["other tracks"]
    # Predicted tracks are hidden by default even though they fill a role.
    assert not vis["pred tracks"]


def test_restore_hidden_round_trips_visibility():
    v = _make_viewer()
    # Start with a non-default visibility so we prove exact restoration.
    v.layers["other tracks"].visible = False
    before = _visible(v)
    w = SpacetimeWidget(v)
    w._guess_once()  # run the one-time auto-guess (normally a deferred tick)
    w._hide_unselected()
    w._restore_hidden()
    assert _visible(v) == before


def test_changing_role_updates_hidden_set():
    # Emulate a live re-hide after the user changes a dropdown while divisualised:
    # a newly selected layer must reappear, a newly deselected one must hide, and
    # a full restore must still return to the true original visibility.
    v = _make_viewer()
    w = SpacetimeWidget(v)
    w._guess_once()  # run the one-time auto-guess (normally a deferred tick)
    original = _visible(v)

    w._hide_unselected()
    assert not v.layers["other tracks"].visible  # unselected -> hidden

    # Select "other tracks" as the fp role, then re-hide (what _apply_lift does).
    w._refreshing = True  # suppress the live re-lift handler
    w._role_combos["fp_edges"].value = "other tracks"
    w._refreshing = False
    w._hide_unselected()
    assert v.layers["other tracks"].visible  # now selected -> visible again

    # Toggling off restores every layer to its pre-divisualisation visibility.
    w._restore_hidden()
    assert _visible(v) == original


def test_selected_layer_names_covers_all_six_dropdowns():
    v = _make_viewer()
    w = SpacetimeWidget(v)
    w._guess_once()  # run the one-time auto-guess (normally a deferred tick)
    names = w._selected_layer_names()
    # gt/pred tracks (roles) and gt/pred masks (labels) are all guessed.
    assert {"gt tracks", "pred tracks", "gt masks", "pred masks"} <= names
