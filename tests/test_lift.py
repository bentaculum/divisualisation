import numpy as np
import pytest
from napari.components import ViewerModel

from divisualisation.lift import SpacetimeLift


@pytest.fixture
def viewer_with_layers():
    v = ViewerModel()
    v.add_image(np.random.rand(5, 8, 8), name="img")
    v.add_labels(np.zeros((5, 8, 8), int), name="masks")
    v.add_tracks(
        np.array([[1, 0, 10, 20], [1, 1, 11, 21], [1, 2, 12, 22]], float),
        name="tracks",
    )
    return v


def test_apply_lifts_tracks_to_five_columns(viewer_with_layers):
    v = viewer_with_layers
    lift = SpacetimeLift(v, time_scale=10)
    lift.apply(["tracks"])
    tracks = v.layers["tracks"].data
    assert tracks.shape[1] == 5
    # z == time_scale * t
    np.testing.assert_allclose(tracks[:, 2], 10 * tracks[:, 1])
    assert v.dims.ndisplay == 3


def test_apply_expands_image_and_labels_to_volume(viewer_with_layers):
    v = viewer_with_layers
    SpacetimeLift(v).apply(["tracks"])
    # image/labels gain a singleton z so they share the tracks' 4D dims.
    assert v.layers["img"].data.shape == (5, 1, 8, 8)
    assert v.layers["masks"].data.shape == (5, 1, 8, 8)


def test_revert_restores_exact_state(viewer_with_layers):
    v = viewer_with_layers
    orig_tracks = v.layers["tracks"].data.copy()
    orig_img_shape = v.layers["img"].data.shape

    lift = SpacetimeLift(v, time_scale=10)
    lift.apply(["tracks"])
    lift.revert()

    assert np.array_equal(v.layers["tracks"].data, orig_tracks)
    assert v.layers["img"].data.shape == orig_img_shape
    assert v.dims.ndisplay == 2
    assert not lift.applied


def test_round_trip_is_stable(viewer_with_layers):
    v = viewer_with_layers
    orig = v.layers["tracks"].data.copy()
    lift = SpacetimeLift(v, time_scale=7)
    for _ in range(3):
        lift.apply(["tracks"])
        lift.revert()
    assert np.array_equal(v.layers["tracks"].data, orig)


def test_time_scale_slider_refolds_live(viewer_with_layers):
    v = viewer_with_layers
    lift = SpacetimeLift(v, time_scale=10)
    lift.apply(["tracks"])
    lift.time_scale = 25
    tracks = v.layers["tracks"].data
    np.testing.assert_allclose(tracks[:, 2], 25 * tracks[:, 1])


def test_deselected_tracks_stay_flat(viewer_with_layers):
    v = viewer_with_layers
    v.add_tracks(np.array([[2, 0, 5, 5], [2, 1, 6, 6]], float), name="other")
    lift = SpacetimeLift(v)
    lift.apply(["tracks"])  # only lift "tracks", not "other"
    assert v.layers["tracks"].data.shape[1] == 5
    # "other" is not lifted; it gets expanded like image/labels do not apply to
    # tracks, so it keeps its original 4 columns.
    assert v.layers["other"].data.shape[1] == 4


def test_double_apply_is_noop(viewer_with_layers):
    v = viewer_with_layers
    lift = SpacetimeLift(v, time_scale=10)
    lift.apply(["tracks"])
    tracks_after_first = v.layers["tracks"].data.copy()
    lift.apply(["tracks"])  # should do nothing
    assert np.array_equal(v.layers["tracks"].data, tracks_after_first)


def test_revert_without_apply_is_safe(viewer_with_layers):
    lift = SpacetimeLift(viewer_with_layers)
    lift.revert()  # no error
    assert not lift.applied


def test_sweep_callback_updates_on_time_change(viewer_with_layers):
    v = viewer_with_layers
    lift = SpacetimeLift(v, time_scale=10)
    lift.apply(["tracks"])
    v.dims.set_current_step(0, 2)  # scrub to t=2
    planes = v.layers["tracks"].experimental_clipping_planes
    # The sweep clips just past the current frame: (t + 1) * time_scale = 30,
    # so the t=2 slice stays visible.
    enabled = [p for p in planes if p.enabled]
    assert enabled and enabled[0].position[0] == pytest.approx(30)
