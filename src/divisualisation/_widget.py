"""Optional napari dock widgets for divisualisation.

Two fixed widgets (each additive; the functional API works without them):

- ``LiftAllTracksWidget`` lifts every tracks layer in the viewer into the 3D
  time->z "spacetime" view, keeping each layer's own coloring, with a live
  lift-amount slider. Toggling off restores the flat 2D view.
- ``ErrorsWidget`` declares the GT / predicted / FN-edge / FP-edge tracks
  layers by role, can compute the CTC edge errors from the GT/pred tracks +
  segmentation labels, and lifts with the traditional error-view look.
"""

import napari
from magicgui.widgets import CheckBox, ComboBox, Container, FloatSlider, PushButton

from .lift import ROLES, SpacetimeLift, _is_tracks


class LiftAllTracksWidget(Container):
    """Lift all tracks layers into the spacetime view, keeping their coloring.

    A single lift-amount slider drives every tracks layer in the viewer; the
    checkbox toggles the 3D time->z lift on and off. The cutting plane stays
    synchronized to the tracks exactly as in the original renderer.
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._lift = SpacetimeLift(viewer)

        self._enabled = CheckBox(value=False, text="Spacetime lift")
        self._lift_amount = FloatSlider(value=12, min=0, max=40, label="lift")
        self._enabled.changed.connect(self._on_toggle)
        self._lift_amount.changed.connect(self._on_lift_amount)
        self.extend([self._enabled, self._lift_amount])

    def _all_tracks_layer_names(self):
        return [layer.name for layer in self._viewer.layers if _is_tracks(layer)]

    def _on_toggle(self, *_):
        if self._enabled.value:
            self._lift.time_scale = self._lift_amount.value
            # Lift every tracks layer, coloring them all green (like main's GT).
            self._lift.apply(self._all_tracks_layer_names(), default_colormap="Greens")
        else:
            self._lift.revert()

    def _on_lift_amount(self, *_):
        self._lift.time_scale = self._lift_amount.value


# One dropdown per lift role, with a human label and substrings used to guess
# which track layer fills the role from its name.
_ROLE_LABELS = {
    "gt": "GT tracks",
    "pred": "predicted tracks",
    "fn_edges": "FN edges",
    "fp_edges": "FP edges",
}
_ROLE_NAME_HINTS = {
    "gt": ("gt", "ground"),
    "pred": ("pred",),
    "fn_edges": ("fn", "false_neg", "false neg", "ctc_fn"),
    "fp_edges": ("fp", "false_pos", "false pos", "ctc_fp"),
}
_LABELS_HINTS = {"gt": ("gt", "ground"), "pred": ("pred", "res")}
_NONE_CHOICE = "—"  # blank / skip this role


class ErrorsWidget(Container):
    """Lift GT/pred tracks with the error-view look; optionally compute errors.

    Declare each role's tracks layer (prefilled by name-guessing, all optional).
    "Compute errors" runs CTC matching from the GT/pred tracks + segmentation
    label layers and adds the FN/FP overlays. The lift applies the traditional
    look (GT->Greens, pred->Wistia, errors->cool) and restores each layer's own
    settings on toggle-off.
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._lift = SpacetimeLift(viewer)

        self._enabled = CheckBox(value=False, text="Spacetime lift")
        self._lift_amount = FloatSlider(value=12, min=0, max=40, label="lift")
        self._role_combos = {
            role: ComboBox(label=_ROLE_LABELS[role], choices=[_NONE_CHOICE])
            for role in ROLES
        }
        self._gt_labels = ComboBox(label="GT labels", choices=[_NONE_CHOICE])
        self._pred_labels = ComboBox(label="pred labels", choices=[_NONE_CHOICE])
        self._compute_btn = PushButton(text="Compute errors")

        self._enabled.changed.connect(self._on_toggle)
        self._lift_amount.changed.connect(self._on_lift_amount)
        self._compute_btn.changed.connect(self._on_compute)
        self.extend([
            self._enabled,
            self._lift_amount,
            *self._role_combos.values(),
            self._gt_labels,
            self._pred_labels,
            self._compute_btn,
        ])

        self._viewer.layers.events.inserted.connect(self._refresh_choices)
        self._viewer.layers.events.removed.connect(self._refresh_choices)
        self._refresh_choices()

    # --- layer discovery ----------------------------------------------------

    def _track_layer_names(self):
        return [layer.name for layer in self._viewer.layers if _is_tracks(layer)]

    def _labels_layer_names(self):
        return [
            layer.name
            for layer in self._viewer.layers
            if type(layer).__name__ == "Labels"
        ]

    @staticmethod
    def _guess(names, hints, already):
        for name in names:
            if name in already:
                continue
            if any(hint in name.lower() for hint in hints):
                return name
        return _NONE_CHOICE

    def _refresh_choices(self, *_):
        if self._enabled.value:
            return  # don't reshuffle while a lift is active
        track_names = self._track_layer_names()
        track_choices = [_NONE_CHOICE, *track_names]
        assigned = {
            c.value
            for c in self._role_combos.values()
            if c.value not in (None, _NONE_CHOICE)
        }
        for role in ROLES:
            combo = self._role_combos[role]
            keep = combo.value if combo.value in track_names else _NONE_CHOICE
            combo.choices = track_choices
            if keep != _NONE_CHOICE:
                combo.value = keep
                continue
            guess = self._guess(track_names, _ROLE_NAME_HINTS[role], assigned)
            combo.value = guess
            if guess != _NONE_CHOICE:
                assigned.add(guess)

        label_names = self._labels_layer_names()
        label_choices = [_NONE_CHOICE, *label_names]
        used: set[str] = set()
        for combo, key in ((self._gt_labels, "gt"), (self._pred_labels, "pred")):
            keep = combo.value if combo.value in label_names else _NONE_CHOICE
            combo.choices = label_choices
            if keep != _NONE_CHOICE:
                combo.value = keep
                used.add(keep)
                continue
            guess = self._guess(label_names, _LABELS_HINTS[key], used)
            combo.value = guess
            if guess != _NONE_CHOICE:
                used.add(guess)

    # --- lift ---------------------------------------------------------------

    def _layer_roles(self):
        return {
            role: combo.value
            for role, combo in self._role_combos.items()
            if combo.value and combo.value != _NONE_CHOICE
        }

    def _set_inputs_enabled(self, enabled):
        for combo in self._role_combos.values():
            combo.enabled = enabled
        self._gt_labels.enabled = enabled
        self._pred_labels.enabled = enabled
        self._compute_btn.enabled = enabled

    def _on_toggle(self, *_):
        if self._enabled.value:
            self._lift.time_scale = self._lift_amount.value
            self._lift.apply(self._layer_roles())
            self._set_inputs_enabled(False)
        else:
            self._lift.revert()
            self._set_inputs_enabled(True)
            self._refresh_choices()

    def _on_lift_amount(self, *_):
        self._lift.time_scale = self._lift_amount.value

    # --- compute errors -----------------------------------------------------

    def _on_compute(self, *_):
        from .errors import compute_edge_errors_from_layers

        layers = self._viewer.layers
        gt_tracks = self._role_combos["gt"].value
        pred_tracks = self._role_combos["pred"].value
        gt_labels = self._gt_labels.value
        pred_labels = self._pred_labels.value
        missing = [
            label
            for label, value in (
                ("GT tracks", gt_tracks),
                ("predicted tracks", pred_tracks),
                ("GT labels", gt_labels),
                ("pred labels", pred_labels),
            )
            if not value or value == _NONE_CHOICE
        ]
        if missing:
            raise ValueError(
                "Computing errors needs " + ", ".join(missing) + " to be set."
            )
        compute_edge_errors_from_layers(
            self._viewer,
            layers[gt_tracks],
            layers[gt_labels],
            layers[pred_tracks],
            layers[pred_labels],
        )
        self._refresh_choices()
