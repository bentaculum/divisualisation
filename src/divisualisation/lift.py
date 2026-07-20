"""Reversible spacetime lift for an existing napari viewer.

Turns a flat 2D+time setup into the 3D "spacetime" view on demand: selected
tracks layers fold time into a z axis (a cone rising out of the image plane),
while image/labels stay planar but get a per-timepoint clipping plane that
sweeps through as you scrub time. Everything is snapshotted on apply and
restored exactly on revert, so it can be toggled on and off from the GUI.
"""

import copy
import logging
from contextlib import contextmanager

import napari
import numpy as np
from napari.utils.colormaps.colormap_utils import vispy_or_mpl_colormap

logger = logging.getLogger(__name__)


def _clipping_planes(cut_at: float):
    """Clip everything ahead of ``cut_at`` along the first (folded-time) axis."""
    return [
        {"position": (0, 0, 0), "normal": (0, 0, 0), "enabled": False},
        {"position": (cut_at, 0, 0), "normal": (-1, 0, 0), "enabled": True},
    ]


# Every lifted layer gets a single constant color property, so each renders as
# one flat color; the actual color is the layer's active colormap (chosen in the
# GUI dropdown). The Tracks layer min-max normalizes the property before mapping,
# so a constant value collapses to one color regardless of the value used --
# ``color_value`` is thus bookkeeping only (0.5 for gt/pred, 1.0 for errors).
_DEFAULT_COLOR_VALUE = 0.5

# The four track roles the lift understands and the "error view" look each one
# takes on toggle-on (all reverted on toggle-off). Matches the original
# Divisualisation renderer: GT -> Greens, predicted -> Wistia, edge errors ->
# cool with a doubled tail width. ``colormap`` is a matplotlib/vispy name;
# ``color_value`` is the raw value mapped through the colormap (fn 0 / fp 1
# match main, giving cool's cyan / magenta endpoints).
ROLES = ("gt", "pred", "fn_edges", "fp_edges")
ROLE_DISPLAY: dict[str, dict] = {
    "gt": {"colormap": "Greens", "width_factor": 1, "color_value": 0.5},
    "pred": {"colormap": "Wistia", "width_factor": 1, "color_value": 0.5},
    "fn_edges": {"colormap": "cool", "width_factor": 2, "color_value": 0.0},
    "fp_edges": {"colormap": "cool", "width_factor": 2, "color_value": 1.0},
}
# Shared look applied to every lifted track layer regardless of role.
_COMMON_DISPLAY = {
    "tail_length": 1000,
    "blending": "translucent_no_depth",
    "opacity": 1.0,
}
# Base tail width; error roles get ``width_factor`` x this.
_BASE_TAIL_WIDTH = 2

# Tracks-layer attributes NOT snapshotted generically as "display state": data
# and geometry (restored explicitly), color (color_by/colormap/properties, which
# are coupled and order-sensitive), and internal/interaction/identity emitters.
# Everything else the layer exposes as an event (tail_width, opacity,
# head_length, tail_length, blending, display_tail, ...) is captured
# generically, so new napari display params are picked up without naming them.
_NON_DISPLAY_ATTRS = frozenset({
    "data",
    "set_data",
    "reload",
    "loaded",
    "refresh",
    "rebuild_graph",
    "rebuild_tracks",
    "scale",
    "translate",
    "affine",
    "rotate",
    "shear",
    "extent",
    "_extent_augmented",
    "properties",
    "color_by",
    "colormap",
    "name",
    "metadata",
    "thumbnail",
    "status",
    "help",
    "cursor",
    "cursor_size",
    "_overlays",
    "mode",
    "editable",
    "locked",
    "mouse_pan",
    "mouse_zoom",
    "scale_factor",
    "units",
    "axis_labels",
    "projection_mode",
    "display_id",
    "visible",
})


def _display_attrs(layer):
    """The Tracks layer's user-facing display attributes, derived from its event
    emitters minus the non-display set. Generic, so it tracks napari changes.
    """
    return sorted(set(layer.events.emitters) - _NON_DISPLAY_ATTRS)


# Default camera rotation for the first lift: a near-orthogonal 3D view of the
# image plane. Tuned for napari >= 0.7, which overhauled the camera-angle
# convention (0.7.0, a breaking change: default angles (0,0,90) -> (0,0,0) and
# intuitive right-handed rotations). Pre-0.7 angles were different.
_DEFAULT_LIFT_ANGLES = (-15, -2, -65)
_DEFAULT_LIFT_PERSPECTIVE = 35
_DEFAULT_LIFT_ZOOM = 1


class SpacetimeLift:
    """Apply and revert the time->z spacetime lift on an existing viewer.

    Args:
        viewer: The napari viewer to transform.
        time_scale: How far tracks lift per unit time. Higher = steeper cone.
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        time_scale: float = 12,
        camera_store: dict | None = None,
    ):
        self._viewer = viewer
        self._time_scale = time_scale
        self._applied = False
        self._snapshots: dict[str, dict] = {}
        self._viewer_snapshot: dict = {}
        # 5-column base tracks (z zeroed) per lifted layer, so changing the lift
        # amount is a cheap recompute from the original time column.
        self._track_bases: dict[str, np.ndarray] = {}
        # Remembered lifted-view camera. Held in a shared dict so several lift
        # instances (e.g. the two plugin toggles) share ONE camera across views;
        # toggling off then on returns to the same 3D view. Pass the same
        # ``camera_store`` to share it. None until the first lift.
        self._camera_store = {} if camera_store is None else camera_store
        # Remembered lifted-view display params per layer (layer name -> {attr:
        # value}), so tweaks made while lifted (e.g. a wider tail) persist across
        # toggling the lift off and on. Per-instance (NOT shared), so each view
        # keeps its own layer-control settings.
        self._lift_display: dict[str, dict] = {}

    @property
    def applied(self) -> bool:
        return self._applied

    @property
    def _lift_camera(self):
        return self._camera_store.get("camera")

    @_lift_camera.setter
    def _lift_camera(self, value):
        self._camera_store["camera"] = value

    def _camera_state(self) -> dict:
        cam = self._viewer.camera
        return {
            "center": tuple(cam.center),
            "zoom": cam.zoom,
            "angles": tuple(cam.angles),
            "perspective": cam.perspective,
        }

    def _set_camera_state(self, state: dict) -> None:
        cam = self._viewer.camera
        cam.center = state["center"]
        cam.zoom = state["zoom"]
        cam.angles = state["angles"]
        cam.perspective = state["perspective"]

    @property
    def time_scale(self) -> float:
        return self._time_scale

    @time_scale.setter
    def time_scale(self, value: float):
        self._time_scale = value
        if self._applied:
            self._refold_tracks()
            self._update_sweep()

    def apply(self, layer_roles):
        """Lift the declared track layers; sweep-clip image/labels; go 3D.

        Args:
            layer_roles: Either a mapping ``{role: layer_name}`` (role is one of
                ``ROLES``) or a plain iterable of layer names. A mapping applies
                each role's "error view" look (including its colormap); a plain
                iterable applies only the shared spacetime look (tail length /
                width / blending / opacity), keeping each layer's own coloring.
                Prior display settings are snapshotted and restored on revert.
                All roles are optional; unknown or missing names are skipped.

        Idempotent: calling apply while already applied is a no-op.
        """
        if self._applied:
            return
        # Normalize to {layer_name: role|None}.
        if isinstance(layer_roles, dict):
            name_to_role = {name: role for role, name in layer_roles.items() if name}
        else:
            name_to_role = {name: None for name in layer_roles}

        # Capture the current timepoint BEFORE mutating: lifting layer data
        # resets it, and we want the slider to stay put across the toggle.
        current_time = self._viewer.dims.current_step[0]

        for layer in self._viewer.layers:
            self._snapshots[layer.name] = self._snapshot_layer(layer)

        # Mutate all layers to a consistent ndim before any render, so napari
        # never draws a mix of 3-ndim and 4-ndim layers (which raises an
        # IndexError on the not-yet-lifted layer's extent).
        with self._block_layer_events():
            for layer in self._viewer.layers:
                if layer.name in name_to_role and _is_tracks(layer):
                    self._lift_tracks_layer(layer)
                    role = name_to_role[layer.name]
                    if role is not None:
                        self._apply_display(layer, role)
                    else:
                        # Lift-all: shared look only, keep the layer's coloring.
                        self._apply_common_display(layer)
                        self._restore_lift_display(layer)
                else:
                    self._expand_to_volume(layer)

        self._viewer_snapshot = {
            "ndisplay": self._viewer.dims.ndisplay,
            "camera": self._camera_state(),
            "current_time": current_time,
        }
        self._viewer.dims.ndisplay = 3
        self._viewer.dims.events.point.connect(self._update_sweep)
        self._applied = True
        if self._lift_camera is not None:
            # Return to the lifted view the user last had.
            self._set_camera_state(self._lift_camera)
        else:
            # First lift: frame the extent, then take a near-orthogonal 3D angle.
            self._viewer.reset_view()
            self._viewer.camera.angles = _DEFAULT_LIFT_ANGLES
            self._viewer.camera.perspective = _DEFAULT_LIFT_PERSPECTIVE
            self._viewer.camera.zoom = _DEFAULT_LIFT_ZOOM
            # Pull the camera center back along depth (axis 0) to half the
            # lifted cone's height, keeping the framed y/x.
            n_timepoints = self._viewer.dims.nsteps[0]
            center = list(self._viewer.camera.center)
            center[0] = 0.5 * self._time_scale * n_timepoints
            self._viewer.camera.center = center
        # Restore the timepoint (reset_view / data changes reset it); this also
        # drives _update_sweep to the right slice via the point event. Set only
        # the time axis, since ndim grew from 3 to 4.
        self._viewer.dims.set_current_step(0, current_time)
        self._update_sweep()

    def revert(self):
        """Restore every layer and the viewer to their pre-apply state."""
        if not self._applied:
            return
        # Disconnect before restoring so the callback cannot fire mid-revert.
        self._viewer.dims.events.point.disconnect(self._update_sweep)

        # Keep the slider where it is across the toggle (restoring data resets it).
        current_time = self._viewer.dims.current_step[0]

        # Remember the lifted view (camera + per-layer display params) BEFORE
        # restoring layers, so toggling back on returns to it. Restoring layer
        # data changes the scene extent and makes napari re-zoom / reset the
        # display, so capturing after would lose these.
        self._lift_camera = self._camera_state()
        self._capture_lift_display()

        for layer in self._viewer.layers:
            snap = self._snapshots.get(layer.name)
            if snap is not None:
                self._restore_layer(layer, snap)

        self._viewer.dims.ndisplay = self._viewer_snapshot["ndisplay"]
        self._set_camera_state(self._viewer_snapshot["camera"])
        self._viewer.dims.set_current_step(0, current_time)
        self._applied = False
        self._snapshots.clear()
        self._track_bases.clear()
        self._viewer_snapshot = {}

    # --- snapshot / restore -------------------------------------------------

    @staticmethod
    def _snapshot_layer(layer) -> dict:
        snap = {
            "data": copy.deepcopy(layer.data),
            "scale": tuple(layer.scale),
            "translate": tuple(layer.translate),
            "clipping_planes": copy.deepcopy([
                p.dict() for p in layer.experimental_clipping_planes
            ]),
        }
        # Display settings only exist on Tracks layers; snapshot them so the
        # "error view" look can be fully reverted on toggle-off. All exposed
        # display attributes are captured generically (tail_width, opacity,
        # head_length, ...); color (color_by/colormap/properties) is kept
        # separate because those are coupled and must be restored in order.
        if _is_tracks(layer):
            snap["graph"] = dict(layer.graph)
            snap["display"] = {a: getattr(layer, a) for a in _display_attrs(layer)}
            snap["color"] = {
                "colormap": layer.colormap,
                "color_by": layer.color_by,
                "properties": {k: v.copy() for k, v in layer.properties.items()},
            }
        return snap

    @staticmethod
    def _restore_layer(layer, snap: dict):
        # Restore data (and graph) FIRST: setting a Tracks layer's .data resets
        # its properties/graph, so display state must be restored afterwards.
        if _is_labels(layer):
            _set_labels_data(layer, snap["data"])
        else:
            layer.data = snap["data"]
            if "graph" in snap:
                layer.graph = snap["graph"]
        color = snap.get("color")
        if color is not None:
            # Restore properties before color_by so the key exists.
            layer.properties = color["properties"]
            if color["color_by"] in layer.properties or not layer.properties:
                layer.color_by = color["color_by"]
            layer.colormap = color["colormap"]
        display = snap.get("display")
        if display is not None:
            for attr, value in display.items():
                setattr(layer, attr, value)
        layer.scale = snap["scale"]
        layer.translate = snap["translate"]
        layer.experimental_clipping_planes = snap["clipping_planes"]

    @contextmanager
    def _block_layer_events(self):
        """Block every layer's events so the ndim swap does not trigger a render
        until all layers are consistently 4D.
        """
        blockers = [layer.events.blocker() for layer in self._viewer.layers]
        for b in blockers:
            b.__enter__()
        try:
            yield
        finally:
            for b in blockers:
                b.__exit__(None, None, None)

    @staticmethod
    def _apply_common_display(layer):
        """Apply the shared spacetime look from the original renderer
        (tail_length, blending, opacity, base tail width) to a lifted track
        layer, leaving its coloring untouched. Prior settings are already
        snapshotted so toggle-off restores them.
        """
        for attr, value in _COMMON_DISPLAY.items():
            setattr(layer, attr, value)
        layer.tail_width = _BASE_TAIL_WIDTH

    @staticmethod
    def _apply_colormap(layer, colormap, key, value=_DEFAULT_COLOR_VALUE):
        """Color a lifted track layer flat: a constant property (all edges =
        ``value``) under ``key``, mapped through ``colormap`` via colormaps_dict.
        Using colormaps_dict maps ``value`` RAW (the Tracks layer only 0-1
        normalizes when using layer.colormap, not colormaps_dict), matching the
        original renderer. Prior coloring is snapshotted so revert restores it.
        """
        layer.properties = {
            **dict(layer.properties),
            key: np.full(len(layer.data), value),
        }
        layer.colormaps_dict = {
            **dict(layer.colormaps_dict),
            key: vispy_or_mpl_colormap(colormap),
        }
        layer.color_by = key

    def _apply_display(self, layer, role):
        """Give a lifted track layer the "error view" look for its role.

        Prior display settings are already snapshotted.
        """
        spec = ROLE_DISPLAY.get(role)
        if spec is None:
            return
        self._apply_common_display(layer)
        self._apply_colormap(
            layer, spec["colormap"], f"_lift_{role}", spec["color_value"]
        )
        # Error roles get a wider tail than the shared base.
        layer.tail_width = _BASE_TAIL_WIDTH * spec["width_factor"]
        # Re-apply any display tweaks the user made in a previous lifted view so
        # they persist across toggling (e.g. a wider tail set while lifted).
        self._restore_lift_display(layer)

    def _restore_lift_display(self, layer):
        """Overlay this layer's remembered lifted-view display params (if any)
        onto the freshly applied role/common look.
        """
        for attr, value in self._lift_display.get(layer.name, {}).items():
            setattr(layer, attr, value)

    def _capture_lift_display(self):
        """Remember each lifted layer's current display params, so re-lifting
        restores them (independent of the layer's non-lifted settings).
        """
        for name in self._track_bases:
            if name in self._viewer.layers:
                layer = self._viewer.layers[name]
                self._lift_display[name] = {
                    a: getattr(layer, a) for a in _display_attrs(layer)
                }

    # --- transforms ---------------------------------------------------------

    def _lift_tracks_layer(self, layer):
        """Fold time into z for one tracks layer, matching the original render."""
        data = np.asarray(layer.data, dtype=float)
        if data.shape[1] == 4:
            # 2D + t: [id, t, y, x] -> [id, t, z=0, y, x]. The inserted z is 0,
            # so lifting invents a z purely from time.
            base = np.insert(data, 2, 0.0, axis=1)
        else:
            # 3D + t: keep the real z; lifting adds the time offset on top of it.
            base = data.copy()
        self._track_bases[layer.name] = base
        # Reassigning .data resets the layer's graph and properties; restore both
        # so division/lineage edges keep drawing and per-detection properties
        # (e.g. segmentation_id, used to compute errors) survive the lift.
        graph = dict(layer.graph)
        properties = {k: v.copy() for k, v in layer.properties.items()}
        layer.data = self._folded(base)
        layer.graph = graph
        layer.properties = properties

    def _folded(self, base: np.ndarray) -> np.ndarray:
        data = base.copy()
        # z <- z - time_scale * t, so tracks rise out of the plane over time.
        # The depth axis points towards the viewer since napari 0.6, so the time
        # term is subtracted to lift upward. Preserves any real z.
        data[:, 2] = base[:, 2] - self._time_scale * base[:, 1]
        return data

    def _refold_tracks(self):
        for name, base in self._track_bases.items():
            layer = self._viewer.layers[name]
            graph = dict(layer.graph)
            layer.data = self._folded(base)
            layer.graph = graph

    @staticmethod
    def _expand_to_volume(layer):
        """Give a 2D+t image/labels layer a singleton z so it shares the 3D dims.

        The new z axis is inserted into scale/translate too, so the layer's
        transform stays consistent with the expanded (t, z, y, x) data.
        """
        data = layer.data
        if getattr(data, "ndim", 0) != 3:
            return
        scale = list(layer.scale)
        translate = list(layer.translate)
        if _is_labels(layer):
            _set_labels_data(layer, np.expand_dims(data, 1))
        else:
            layer.data = np.expand_dims(data, 1)
        # Insert the z entry after time (index 1). Growing the data reset the
        # transforms to 4D defaults; overwrite with the intended values.
        scale.insert(1, 1.0)
        translate.insert(1, 0.0)
        layer.scale = scale
        layer.translate = translate

    def _update_sweep(self, event=None):
        """Sync each lifted tracks layer to the current timepoint.

        Clip the tracks at ``-t * time_scale`` along the folded-time (z) axis
        and translate them by ``+t * time_scale`` there, so the current
        timepoint's slice lands on the fixed image plane and later frames recede
        above it as you scrub. Signs follow the fold (``z = z - time_scale * t``,
        upward under napari 0.6+ axis directions). Image/labels are untouched.
        """
        t = self._viewer.dims.point[0]
        clipping_planes = [
            {"position": (0, 0, 0), "normal": (0, 0, 0), "enabled": False},
            {
                "position": (-t * self._time_scale, 0, 0),
                "normal": (1, 0, 0),
                "enabled": True,
            },
        ]
        for name in self._track_bases:
            if name not in self._viewer.layers:
                continue  # layer removed (e.g. viewer teardown); skip
            layer = self._viewer.layers[name]
            layer.experimental_clipping_planes = clipping_planes
            layer.translate = [0, self._time_scale * t, 0, 0]


def _is_tracks(layer) -> bool:
    return type(layer).__name__ == "Tracks"


def _is_labels(layer) -> bool:
    return type(layer).__name__ == "Labels"


def _set_labels_data(layer, data: np.ndarray) -> None:
    """Set a Labels layer's data, bypassing a napari ndim-change bug.

    napari's ``Labels.data`` setter pre-sets ``_ndim`` to the new value before
    ``_update_dims()``, so a 3D<->4D change is not reflected in the layer's
    transforms and the vispy render path hits a matmul shape error. Setting
    ``_data`` directly keeps ``_ndim`` at the old value until ``_update_dims``
    runs, so the transforms grow/shrink correctly.
    """
    layer._data = layer._ensure_int_labels(data)
    layer._update_dims()
    layer.events.data(value=layer.data)
    layer._reset_editable()
