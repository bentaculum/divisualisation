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

logger = logging.getLogger(__name__)


def _clipping_planes(cut_at: float):
    """Clip everything ahead of ``cut_at`` along the first (folded-time) axis."""
    return [
        {"position": (0, 0, 0), "normal": (0, 0, 0), "enabled": False},
        {"position": (cut_at, 0, 0), "normal": (-1, 0, 0), "enabled": True},
    ]


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

    def apply(self, track_layer_names):
        """Lift the named tracks layers; sweep-clip image/labels; go 3D.

        Idempotent: calling apply while already applied is a no-op.
        """
        if self._applied:
            return
        track_layer_names = set(track_layer_names)

        for layer in self._viewer.layers:
            self._snapshots[layer.name] = self._snapshot_layer(layer)

        for layer in self._viewer.layers:
            if layer.name in track_layer_names and _is_tracks(layer):
                self._lift_tracks_layer(layer)
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
        return {
            "data": copy.deepcopy(layer.data),
            "scale": tuple(layer.scale),
            "translate": tuple(layer.translate),
            "clipping_planes": copy.deepcopy([
                p.dict() for p in layer.experimental_clipping_planes
            ]),
        }

    @staticmethod
    def _restore_layer(layer, snap: dict):
        layer.data = snap["data"]
        layer.scale = snap["scale"]
        layer.translate = snap["translate"]
        layer.experimental_clipping_planes = snap["clipping_planes"]

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
        """Give a 2D+t image/labels layer a singleton z so it shares the 3D dims."""
        data = layer.data
        if getattr(data, "ndim", 0) == 3:
            layer.data = np.expand_dims(data, 1)

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
