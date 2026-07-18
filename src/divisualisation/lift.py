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

# Default camera for the first lift: a near-orthogonal 3D view of the image
# plane, taken from the original example_2d on main.
_DEFAULT_LIFT_ANGLES = (27.919484296382873, -49.86671510905139, -35.8190766165135)
_DEFAULT_LIFT_PERSPECTIVE = 27


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
        # Remembered lifted-view camera, so toggling off then on returns to the
        # same 3D view. None until the first lift.
        self._lift_camera: dict | None = None

    @property
    def applied(self) -> bool:
        return self._applied

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

        for layer in self._viewer.layers:
            snap = self._snapshots.get(layer.name)
            if snap is not None:
                self._restore_layer(layer, snap)

        # Remember where the lifted view was, so toggling back on returns to it.
        self._lift_camera = self._camera_state()

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
        # "error view" look can be fully reverted on toggle-off.
        if _is_tracks(layer):
            snap["graph"] = dict(layer.graph)
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
            if "graph" in snap:
                layer.graph = snap["graph"]
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
    def _apply_colormap(layer, colormap, key):
        """Color a lifted track layer flat with ``colormap`` via a constant
        property stored under ``key`` (unique per layer so overlaid layers don't
        clash). Prior coloring is already snapshotted so toggle-off restores it.
        """
        layer.properties = {
            **dict(layer.properties),
            key: np.full(len(layer.data), 0.5),
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
        self._apply_colormap(layer, spec["colormap"], f"_lift_{role}")
        # Error roles get a wider tail than the shared base.
        layer.tail_width = _BASE_TAIL_WIDTH * spec["width_factor"]

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
        # Reassigning .data resets the layer's division graph; restore it so the
        # division/lineage edges keep drawing (lifted with the node coordinates).
        graph = dict(layer.graph)
        layer.data = self._folded(base)
        layer.graph = graph

    def _folded(self, base: np.ndarray) -> np.ndarray:
        data = base.copy()
        # z <- z + time_scale * t, so tracks rise out of the plane over time
        # while preserving any real z (matches the original Divisualisation).
        data[:, 2] = base[:, 2] + self._time_scale * base[:, 1]
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

        Verbatim port of the original Divisualisation renderer: clip the tracks
        at ``t * time_scale`` along the folded-time (z) axis and translate them
        by ``-t * time_scale`` there, so the current timepoint's slice lands on
        the fixed image plane and later frames recede above it as you scrub.
        Image/labels planes are left untouched.
        """
        t = self._viewer.dims.point[0]
        clipping_planes = [
            {"position": (0, 0, 0), "normal": (0, 0, 0), "enabled": False},
            {
                "position": (t * self._time_scale, 0, 0),
                "normal": (-1, 0, 0),
                "enabled": True,
            },
        ]
        for name in self._track_bases:
            if name not in self._viewer.layers:
                continue  # layer removed (e.g. viewer teardown); skip
            layer = self._viewer.layers[name]
            layer.experimental_clipping_planes = clipping_planes
            layer.translate = [0, -self._time_scale * t, 0, 0]


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
