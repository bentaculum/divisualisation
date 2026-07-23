"""Dropdown-choice tests for the Divisualisation widget.

Model: layers are auto-guessed into roles ONCE when the widget opens; after that
the dropdowns are the user's. The combos use *callable* choices, so napari's
add_dock_widget keeps the option lists current by calling ``reset_choices`` on
every layer inserted/removed/reordered/renamed event -- preserving a selection
as long as its layer still exists, and never re-guessing.

These run on a ``ViewerModel`` under offscreen Qt (no GL canvas). A bare
ViewerModel does NOT auto-connect reset_choices (only add_dock_widget does), so
the tests call ``w._reset_role_choices()`` / ``w._guess_once()`` explicitly to stand
in for what the docked widget gets for free.
"""

import numpy as np
import pytest
from napari.components import ViewerModel

pytest.importorskip("magicgui")

from divisualisation._widget import _NONE_CHOICE, SpacetimeWidget


def _make_viewer():
    v = ViewerModel()
    v.add_labels(np.zeros((5, 8, 8), int), name="gt masks")
    v.add_labels(np.zeros((5, 8, 8), int), name="pred masks")
    v.add_tracks(np.array([[1, t, 5, 5] for t in range(5)], float), name="gt tracks")
    v.add_tracks(np.array([[2, t, 6, 6] for t in range(5)], float), name="pred tracks")
    return v


def _make_widget(v):
    # Build the widget and run the one-time auto-guess (normally a deferred tick).
    w = SpacetimeWidget(v)
    w._guess_once()
    return w


def _roles(w):
    return {r: c.value for r, c in w._role_combos.items()} | {
        "gt_labels": w._gt_labels.value,
        "pred_labels": w._pred_labels.value,
    }


def test_auto_guess_happens_once_on_open():
    v = _make_viewer()
    w = SpacetimeWidget(v)
    # Guess is deferred: nothing selected until the tick runs.
    assert all(c.value == _NONE_CHOICE for c in w._role_combos.values())
    w._guess_once()
    guessed = _roles(w)
    assert guessed["gt"] == "gt tracks" and guessed["pred"] == "pred tracks"
    assert guessed["gt_labels"] == "gt masks" and guessed["pred_labels"] == "pred masks"


def test_adding_layer_keeps_selections_and_lists_options():
    # Reported bug: adding/duplicating a layer emptied the dropdowns (options
    # gone, values reset to "—"). reset_choices must instead keep selections and
    # just extend the option lists.
    v = _make_viewer()
    w = _make_widget(v)
    before = _roles(w)

    v.add_tracks(
        np.array([[1, t, 5, 5] for t in range(5)], float), name="gt tracks copy"
    )
    w._reset_role_choices()  # what napari's add_dock_widget does automatically

    assert _roles(w) == before  # every selection preserved
    # Options are NOT emptied: the new layer is present, existing ones remain.
    for combo in w._role_combos.values():
        assert "gt tracks copy" in list(combo.choices)
        assert "gt tracks" in list(combo.choices)


def test_deleting_layer_keeps_options_and_other_selections():
    v = _make_viewer()
    w = _make_widget(v)
    v.add_tracks(np.array([[3, t, 1, 1] for t in range(5)], float), name="extra")
    w._reset_role_choices()
    before = _roles(w)

    v.layers.remove("extra")  # a non-selected layer
    w._reset_role_choices()

    # Options still list the real layers (not emptied), selections untouched.
    for combo in w._role_combos.values():
        assert "gt tracks" in list(combo.choices)
        assert "extra" not in list(combo.choices)
    assert _roles(w) == before


def test_deleting_selected_layer_clears_only_that_dropdown():
    v = _make_viewer()
    w = _make_widget(v)
    before = _roles(w)

    v.layers.remove("pred tracks")  # the layer in the pred role
    w._reset_role_choices()

    after = _roles(w)
    assert after["pred"] == _NONE_CHOICE  # its layer is gone
    for key in ("gt", "fn_edges", "fp_edges", "gt_labels", "pred_labels"):
        assert after[key] == before[key]  # everything else preserved


def test_layer_event_never_reguesses_after_manual_change():
    v = _make_viewer()
    w = _make_widget(v)
    # User overrides gt to a non-name-matching layer, then a layer is added.
    w._role_combos["gt"].value = "pred tracks"
    v.add_tracks(np.array([[3, t, 1, 1] for t in range(5)], float), name="extra")
    w._reset_role_choices()
    w._guess_once()  # even if fired again, it's a no-op after the first guess
    # The manual pick stands; no re-guess reclaims "gt tracks" for the gt role.
    assert w._role_combos["gt"].value == "pred tracks"


def test_guess_from_layers_added_after_empty_open():
    # Widget docked before any layers exist: the one-time guess should fire when
    # the first layers arrive (we connect _guess_once to the inserted event).
    v = ViewerModel()
    w = SpacetimeWidget(v)
    w._guess_once()  # no layers yet -> no-op, still unguessed
    assert not w._guessed
    v.add_tracks(np.array([[1, t, 5, 5] for t in range(5)], float), name="gt tracks")
    # The insert event fires the one-time guess (connected in __init__).
    assert w._guessed
    assert w._role_combos["gt"].value == "gt tracks"


def test_combo_choices_are_callables():
    # The fix hinges on choices being callables (so napari's reset_choices
    # re-derives live options) rather than a static list that gets wiped.
    v = _make_viewer()
    w = _make_widget(v)
    assert callable(w._track_choices)
    assert callable(w._label_choices)
    assert w._track_choices() == [_NONE_CHOICE, "gt tracks", "pred tracks"]
    assert w._label_choices() == [_NONE_CHOICE, "gt masks", "pred masks"]


def test_suspend_role_events_is_reentrant():
    v = _make_viewer()
    w = _make_widget(v)
    assert w._refreshing is False
    with w._suspend_role_events():
        assert w._refreshing is True
        with w._suspend_role_events():
            assert w._refreshing is True
        assert w._refreshing is True  # inner exit must not clear outer suspension
    assert w._refreshing is False
