"""Reversible spacetime lift for an existing napari viewer.

Turns a flat 2D+time setup into the 3D "spacetime" view on demand: selected
tracks layers fold time into a z axis (a cone rising out of the image plane),
while image/labels stay planar but get a per-timepoint clipping plane that
sweeps through as you scrub time. Everything is snapshotted on apply and
restored exactly on revert, so it can be toggled on and off from the GUI.
"""

import copy
import logging

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


# The four track roles the lift understands and the "error view" look each one
# takes on toggle-on (all reverted on toggle-off). Matches the original
# Divisualisation renderer: GT -> Greens, predicted -> Wistia, edge errors ->
# cool with a doubled tail width. ``colormap`` is a matplotlib/vispy name.
ROLES = ("gt", "pred", "fn_edges", "fp_edges")
ROLE_DISPLAY: dict[str, dict] = {
    "gt": {"colormap": "Greens", "width_factor": 1},
    "pred": {"colormap": "Wistia", "width_factor": 1},
    "fn_edges": {"colormap": "cool", "width_factor": 2},
    "fp_edges": {"colormap": "cool", "width_factor": 2},
}
# Shared look applied to every lifted track layer regardless of role.
_COMMON_DISPLAY = {
    "tail_length": 1000,
    "blending": "translucent_no_depth",
    "opacity": 1.0,
}
# Base tail width; error roles get ``width_factor`` x this.
_BASE_TAIL_WIDTH = 2


class SpacetimeLift:
    """Apply and revert the time->z spacetime lift on an existing viewer.

    Args:
        viewer: The napari viewer to transform.
        time_scale: How far tracks lift per unit time. Higher = steeper cone.
    """

    def __init__(self, viewer: napari.Viewer, time_scale: float = 12):
        self._viewer = viewer
        self._time_scale = time_scale
        self._applied = False
        self._snapshots: dict[str, dict] = {}
        self._viewer_snapshot: dict = {}
        # 5-column base tracks (z zeroed) per lifted layer, so changing the lift
        # amount is a cheap recompute from the original time column.
        self._track_bases: dict[str, np.ndarray] = {}

    @property
    def applied(self) -> bool:
        return self._applied

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
                ``ROLES``) or a plain iterable of layer names. A mapping also
                applies each role's "error view" display look (colormap, tail
                length/width, blending, opacity), snapshotting and restoring the
                layers' prior display settings. All roles are optional; unknown
                or missing layer names are skipped.

        Idempotent: calling apply while already applied is a no-op.
        """
        if self._applied:
            return
        # Normalize to {layer_name: role|None}.
        if isinstance(layer_roles, dict):
            name_to_role = {name: role for role, name in layer_roles.items() if name}
        else:
            name_to_role = {name: None for name in layer_roles}

        for layer in self._viewer.layers:
            self._snapshots[layer.name] = self._snapshot_layer(layer)

        for layer in self._viewer.layers:
            if layer.name in name_to_role and _is_tracks(layer):
                self._lift_tracks_layer(layer)
                self._apply_display(layer, name_to_role[layer.name])
            else:
                self._expand_to_volume(layer)

        self._viewer_snapshot = {
            "ndisplay": self._viewer.dims.ndisplay,
            "camera_center": tuple(self._viewer.camera.center),
            "camera_zoom": self._viewer.camera.zoom,
        }
        self._viewer.dims.ndisplay = 3
        self._viewer.dims.events.point.connect(self._update_sweep)
        self._applied = True
        self._update_sweep()
        # Frame the camera on the new 3D extent so the lifted cone is visible.
        self._viewer.reset_view()

    def revert(self):
        """Restore every layer and the viewer to their pre-apply state."""
        if not self._applied:
            return
        # Disconnect before restoring so the callback cannot fire mid-revert.
        self._viewer.dims.events.point.disconnect(self._update_sweep)

        for layer in self._viewer.layers:
            snap = self._snapshots.get(layer.name)
            if snap is not None:
                self._restore_layer(layer, snap)

        self._viewer.dims.ndisplay = self._viewer_snapshot["ndisplay"]
        self._viewer.camera.center = self._viewer_snapshot["camera_center"]
        self._viewer.camera.zoom = self._viewer_snapshot["camera_zoom"]
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
        # "error view" look can be fully reverted on toggle-off.
        if _is_tracks(layer):
            snap["display"] = {
                "colormaps_dict": dict(layer.colormaps_dict),
                "color_by": layer.color_by,
                "properties": {k: v.copy() for k, v in layer.properties.items()},
                "tail_length": layer.tail_length,
                "tail_width": layer.tail_width,
                "blending": layer.blending,
                "opacity": layer.opacity,
            }
        return snap

    @staticmethod
    def _restore_layer(layer, snap: dict):
        display = snap.get("display")
        if display is not None:
            # Restore properties/colormap before color_by so the key exists.
            layer.properties = display["properties"]
            layer.colormaps_dict = display["colormaps_dict"]
            if display["color_by"] in layer.properties or not layer.properties:
                layer.color_by = display["color_by"]
            layer.tail_length = display["tail_length"]
            layer.tail_width = display["tail_width"]
            layer.blending = display["blending"]
            layer.opacity = display["opacity"]
        if _is_labels(layer):
            _set_labels_data(layer, snap["data"])
        else:
            layer.data = snap["data"]
        layer.scale = snap["scale"]
        layer.translate = snap["translate"]
        layer.experimental_clipping_planes = snap["clipping_planes"]

    def _apply_display(self, layer, role):
        """Give a lifted track layer the "error view" look for its role.

        role is None for layers lifted without a declared role (geometry only,
        no display change). Prior display settings are already snapshotted.
        """
        if role is None:
            return
        spec = ROLE_DISPLAY.get(role)
        if spec is None:
            return
        for key, value in _COMMON_DISPLAY.items():
            setattr(layer, key, value)
        layer.tail_width = _BASE_TAIL_WIDTH * spec["width_factor"]
        # Flat single-color look: constant property driving the role colormap.
        key = f"_lift_{role}"
        layer.properties = {
            **dict(layer.properties),
            key: np.full(len(layer.data), 0.5),
        }
        layer.colormaps_dict = {
            **dict(layer.colormaps_dict),
            key: vispy_or_mpl_colormap(spec["colormap"]),
        }
        layer.color_by = key

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
        layer.data = self._folded(base)

    def _folded(self, base: np.ndarray) -> np.ndarray:
        data = base.copy()
        # z <- z + time_scale * t, so tracks rise out of the plane over time
        # while preserving any real z (matches the original Divisualisation).
        data[:, 2] = base[:, 2] + self._time_scale * base[:, 1]
        return data

    def _refold_tracks(self):
        for name, base in self._track_bases.items():
            self._viewer.layers[name].data = self._folded(base)

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
        """Move the clipping plane / lifted-track translate to the current time."""
        t = self._viewer.dims.point[0]
        # Clip just past the current timepoint so its slice is always visible
        # (a bare t * time_scale cut hides everything at t=0).
        cut_at = (t + 1) * self._time_scale
        translate_z = t * self._time_scale
        for layer in self._viewer.layers:
            layer.experimental_clipping_planes = _clipping_planes(cut_at)
            if layer.name in self._track_bases:
                translate = list(layer.translate)
                translate[-3] = -translate_z
                layer.translate = translate


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
