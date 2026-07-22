# Divisualisation

Visualize cell tracking edge errors computed via [traccuracy](https://github.com/live-image-tracking-tools/traccuracy/) in napari.

2D tracking (bacteria) | 3D tracking (C. elegans nuclei)
:-: | :-:
<video src='https://github.com/user-attachments/assets/38d047b1-bc7b-4315-a192-97886a1bf906' width=180></video> | <video src='https://github.com/user-attachments/assets/3724faf2-9d24-428a-8b4f-84ee68646424' width=180/></video>

We originally introduced these visualisations to compare our results in [_Trackastra: Transformer-based cell tracking for live-cell microscopy_](https://github.com/weigertlab/trackastra) to other cell tracking algorithms.



<video src='https://github.com/user-attachments/assets/99ac7295-cab5-43a0-9899-4fa007b110f7' width=60></video>


### Installation

Divisualisation is a [napari](https://napari.org) plugin.

- Install [napari](https://napari.org/stable/tutorials/fundamentals/installation.html).
- Install Divisualisation, either:
  - from within napari via **Plugins → Install/Uninstall Plugins…** (search for
    "divisualisation"), or
  - from PyPI:
    ```
    pip install divisualisation
    ```
  - the latest development version from GitHub:
    ```
    pip install git+https://github.com/bentaculum/divisualisation.git
    ```

### Usage

There are two ways to use Divisualisation.

#### Interactive: overlay errors on an existing viewer

Call a single function on a viewer that already holds your data (image, masks,
tracks — e.g. from motile-tracker) to add the edge errors on top. This works
for genuine 2D data without a dummy `z` dimension and does not switch the viewer
into 3D:

```python
import napari
from divisualisation import add_edge_error_tracks

viewer = napari.Viewer()          # your existing 2D viewer with your own layers
# ... add_image / add_labels / add_tracks as usual ...

error_layers = add_edge_error_tracks(viewer, gt_graph, pred_graph)
```

`gt_graph` and `pred_graph` are matched `traccuracy.TrackingGraph` objects (the
`.gt_graph` / `.pred_graph` returned by `traccuracy.run_metrics`). Each error
type becomes its own named `Tracks` layer, so you get napari's built-in eye-icon
visibility toggles for free — no plugin required.

2D vs 3D is auto-detected from the graph nodes (2D iff nodes have no `z`
attribute). Pass `scale=(y_scale, x_scale)` (spatial only, no leading time
entry) to align the overlay with your image/labels layers. See
`example_programmatic_2d.py`, which uses `add_edge_error_tracks` as part of a
fully scripted render.

Divisualisation also ships an optional napari dock widget,
**Plugins → Divisualisation → Lift tracks & Divisualisation**, with two mutually
exclusive workflows (each in its own box, with its own lift-amount slider):

- **Lift all tracks layers** — fold time into a `z` axis so every tracks layer
  rises out of the moving image plane into a 3D "spacetime" cone, keeping each
  layer's own coloring. Scrub the time slider to sweep through the cone; toggle
  off to restore the flat 2D view exactly. This is the original `Divisualisation`
  render effect, made interactive and reversible.
- **Divisualisation** — pick your ground-truth / predicted / FN-edge / FP-edge
  tracks layers via the role dropdowns (auto-guessed from layer names),
  optionally **Compute errors** from the GT/predicted tracks plus their
  segmentation labels, and lift with the error-view coloring. A **Color division
  edges** checkbox draws each selected layer's parent→daughter division edges as
  coloured track tails (napari otherwise draws them in a fixed, uncolourable
  white).

#### Examples

Run `example_2d.py` (bacteria) or `example_3d.py` (C. elegans nuclei) in ipython.
Each loads its data into a napari viewer with the tracks and edge-error overlays
and docks the widget, so you can toggle the lift and play the time slider
yourself.

#### Rendering an animation

To render the spacetime mp4 animations shown above without any GUI interaction,
run `example_programmatic_2d.py`. It scripts the whole pipeline -- build the
layers, fold time into `z` with `SpacetimeLift`, add the edge-error overlays
with `add_edge_error_tracks`, then capture a `napari_animation` keyframe render
-- driving the same machinery the interactive plugin uses.
