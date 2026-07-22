import warnings

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
    np.testing.assert_allclose(tracks[:, 2], -10 * tracks[:, 1])
    assert v.dims.ndisplay == 3


def test_fold_accounts_for_z_display_scale():
    # A tracks layer with an anisotropic z display scale (e.g. 10x for the
    # C. elegans volume) must fold in WORLD units: the data-space z offset is
    # divided by the z scale, so the cone's world steepness stays == time_scale
    # regardless of the scale (not scale-times too steep).
    v = ViewerModel()
    # 3D+t tracks: [id, t, z, y, x] -> 4D layer, scale (t, z, y, x).
    data = np.array([[1, t, 0, 5, 5] for t in range(4)], float)
    v.add_tracks(data, name="tracks", scale=(1, 10, 1, 1))
    lift = SpacetimeLift(v, time_scale=10)
    lift.apply(["tracks"])
    folded = v.layers["tracks"].data
    z_scale = float(v.layers["tracks"].scale[1])
    # data-space offset is time_scale / z_scale per frame ...
    np.testing.assert_allclose(folded[:, 2], -(10 / z_scale) * folded[:, 1])
    # ... so in WORLD units (z_scale * data) it is exactly time_scale per frame.
    world_z = z_scale * folded[:, 2]
    np.testing.assert_allclose(world_z, -10 * folded[:, 1])


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
    np.testing.assert_allclose(tracks[:, 2], -25 * tracks[:, 1])


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
    v.dims.set_current_step(0, 2)  # scrub to t=2, time_scale=10
    layer = v.layers["tracks"]
    planes = layer.experimental_clipping_planes
    # Verbatim main coupling: clip at t*time_scale=20, translate -t*scale=-20.
    enabled = [p for p in planes if p.enabled]
    assert enabled and enabled[0].position[0] == pytest.approx(-20)
    assert list(layer.translate) == pytest.approx([0, 20, 0, 0])


def test_lift_preserves_real_z_for_3d_tracks():
    # 3D tracks already have a real z; lifting adds the time offset on top of it
    # (matching the original Divisualisation: z = z_real - time_scale * t).
    v = ViewerModel()
    v.add_image(np.random.rand(3, 2, 8, 8), name="vol")  # t, z, y, x
    v.add_tracks(
        np.array([[1, 0, 4.0, 5, 5], [1, 1, 4.0, 6, 6], [1, 2, 4.0, 7, 7]], float),
        name="tracks",
    )
    lift = SpacetimeLift(v, time_scale=5)
    lift.apply(["tracks"])
    # z_real=4, time_scale=5, t=[0,1,2] -> 4 - 5*t = [4, -1, -6]
    np.testing.assert_allclose(v.layers["tracks"].data[:, 2], [4, -1, -6])
    assert v.layers["tracks"].data.shape[1] == 5  # stays 5-col

    lift.revert()
    np.testing.assert_allclose(v.layers["tracks"].data[:, 2], [4, 4, 4])


def test_role_mapping_applies_and_restores_display():
    # Declaring layers by role applies each role's "error view" look on lift and
    # restores the layers' original display settings on revert.
    from divisualisation.lift import ROLE_DISPLAY

    v = ViewerModel()
    gt = v.add_tracks(
        np.array([[1, 0, 5, 5], [1, 1, 6, 6]], float),
        name="GT tracks",
        tail_length=7,
        tail_width=3,
    )
    fn = v.add_tracks(
        np.array([[2, 0, 1, 1], [2, 1, 2, 2]], float),
        name="is_ctc_fn",
        tail_length=9,
        tail_width=1,
    )
    orig = {ly.name: (ly.tail_length, ly.tail_width, ly.color_by) for ly in (gt, fn)}

    lift = SpacetimeLift(v, time_scale=10)
    lift.apply({"gt": "GT tracks", "fn_edges": "is_ctc_fn", "pred": "", "fp_edges": ""})

    # Common "error view" look on both; error role gets the doubled tail width.
    assert gt.tail_length == 1000 and fn.tail_length == 1000
    assert gt.tail_width == 2  # gt width_factor 1 * base 2
    assert fn.tail_width == 2 * ROLE_DISPLAY["fn_edges"]["width_factor"]  # 4
    assert gt.color_by == "_lift_gt" and fn.color_by == "_lift_fn_edges"

    lift.revert()
    for layer in (v.layers["GT tracks"], v.layers["is_ctc_fn"]):
        tl, tw, cb = orig[layer.name]
        assert layer.tail_length == tl
        assert layer.tail_width == tw
        assert layer.color_by == cb
        assert layer.data.shape[1] == 4


def test_missing_roles_are_skipped():
    # All roles optional: a role pointing at no layer (blank) is simply skipped.
    v = ViewerModel()
    v.add_tracks(np.array([[1, 0, 5, 5], [1, 1, 6, 6]], float), name="GT tracks")
    lift = SpacetimeLift(v, time_scale=8)
    # Only gt is declared; pred/fn/fp are blank.
    lift.apply({"gt": "GT tracks", "pred": "", "fn_edges": "", "fp_edges": ""})
    assert v.layers["GT tracks"].data.shape[1] == 5
    lift.revert()
    assert v.layers["GT tracks"].data.shape[1] == 4


def test_lift_restores_all_display_params():
    # Any display attribute the user changed (not just the ones the lift sets)
    # is snapshotted generically and restored on revert.
    v = ViewerModel()
    v.add_image(np.zeros((6, 8, 8)), name="raw")
    layer = v.add_tracks(
        np.array([[1, t, 5, 5] for t in range(6)], float), name="tracks"
    )
    layer.tail_width = 7
    layer.opacity = 0.42
    layer.head_length = 3
    layer.tail_length = 55
    layer.blending = "opaque"
    before = {
        a: getattr(layer, a)
        for a in ("tail_width", "opacity", "head_length", "tail_length", "blending")
    }

    lift = SpacetimeLift(v, time_scale=10)
    lift.apply(["tracks"])  # lift-all overwrites some of these
    lift.revert()

    layer = v.layers["tracks"]
    after = {a: getattr(layer, a) for a in before}
    assert after == before


def test_lifted_display_tweaks_persist_across_toggle():
    # A display param changed WHILE lifted persists on the next lift, separately
    # from the layer's own non-lifted value.
    v = ViewerModel()
    v.add_image(np.zeros((6, 8, 8)), name="raw")
    v.add_tracks(
        np.array([[1, t, 5, 5] for t in range(6)], float),
        name="tracks",
        tail_width=2,
    )
    lift = SpacetimeLift(v, time_scale=10)
    lift.apply(["tracks"])
    v.layers["tracks"].tail_width = 12  # widen while lifted
    lift.revert()
    # Non-lifted value restored.
    assert v.layers["tracks"].tail_width == 2
    # Re-lift restores the lifted tweak.
    lift.apply(["tracks"])
    assert v.layers["tracks"].tail_width == 12


def test_all_lifted_layers_share_head_length():
    # Every lifted tracks layer gets head_length=0 so the clip plane cuts them
    # identically (error layers are created with head_length=1, which offset
    # them past the plane relative to gt/pred).
    v = ViewerModel()
    v.add_image(np.zeros((6, 8, 8)), name="raw")
    v.add_tracks(np.array([[1, t, 5, 5] for t in range(6)], float), name="gt")
    err = v.add_tracks(np.array([[2, t, 8, 8] for t in range(6)], float), name="err")
    err.head_length = 1  # mimic add_edge_error_tracks
    lift = SpacetimeLift(v, time_scale=10)
    lift.apply({"gt": "gt", "fn_edges": "err", "pred": "", "fp_edges": ""})
    assert v.layers["gt"].head_length == 0
    assert v.layers["err"].head_length == 0


def test_slider_change_keeps_error_colors():
    # Changing the lift amount re-folds the data; the role colormap/color_by
    # must survive (setting layer.data resets properties, which would drop the
    # error-view coloring).
    v = ViewerModel()
    v.add_image(np.zeros((6, 8, 8)), name="raw")
    v.add_tracks(np.array([[1, t, 5, 5] for t in range(6)], float), name="gt")
    lift = SpacetimeLift(v, time_scale=10)
    lift.apply({"gt": "gt", "pred": "", "fn_edges": "", "fp_edges": ""})
    layer = v.layers["gt"]
    color_by = layer.color_by
    assert color_by == "_lift_gt"
    # move the lift slider
    lift.time_scale = 30
    assert v.layers["gt"].color_by == color_by  # unchanged
    np.testing.assert_allclose(
        v.layers["gt"].data[:, 2], -30 * v.layers["gt"].data[:, 1]
    )


def test_slider_refold_emits_no_color_by_warning():
    # Re-folding sets layer.data, which momentarily clears the layer's features
    # to just {track_id}. napari warns ("... not present in features. Falling
    # back to track_id") if the active color_by (here "_lift_gt") is gone at
    # that instant, and silently drops the coloring. Re-folding must not warn.
    v = ViewerModel()
    v.add_image(np.zeros((6, 8, 8)), name="raw")
    v.add_tracks(np.array([[1, t, 5, 5] for t in range(6)], float), name="gt")
    lift = SpacetimeLift(v, time_scale=10)
    lift.apply({"gt": "gt", "pred": "", "fn_edges": "", "fp_edges": ""})
    assert v.layers["gt"].color_by == "_lift_gt"

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*not present in features.*")
        lift.time_scale = 30  # slider move -> _refold_tracks
    # And the real coloring is still active afterwards, not reset to track_id.
    assert v.layers["gt"].color_by == "_lift_gt"


def test_lift_emits_no_invalid_divide_warning():
    # The disabled placeholder clipping plane must carry a unit normal: napari
    # normalizes a plane's normal on construction, so a zero normal raises a
    # numpy "invalid value encountered in divide" (0/0) warning. Applying the
    # lift and sweeping the clip plane must stay silent.
    v = ViewerModel()
    v.add_image(np.zeros((6, 8, 8)), name="raw")
    v.add_tracks(np.array([[1, t, 5, 5] for t in range(6)], float), name="gt")
    lift = SpacetimeLift(v, time_scale=10)

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="invalid value encountered in divide")
        lift.apply(["gt"])
        lift._update_sweep()
