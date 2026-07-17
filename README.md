# Divisualisation

Visualize cell tracking edge errors computed via [traccuracy](https://github.com/live-image-tracking-tools/traccuracy/) in napari.

2D tracking (bacteria) | 3D tracking (C. elegans nuclei)
:-: | :-:
<video src='https://github.com/user-attachments/assets/38d047b1-bc7b-4315-a192-97886a1bf906' width=180></video> | <video src='https://github.com/user-attachments/assets/3724faf2-9d24-428a-8b4f-84ee68646424' width=180/></video>

We originally introduced these visualisations to compare our results in [_Trackastra: Transformer-based cell tracking for live-cell microscopy_](https://github.com/weigertlab/trackastra) to other cell tracking algorithms.



<video src='https://github.com/user-attachments/assets/99ac7295-cab5-43a0-9899-4fa007b110f7' width=60></video>


### Installation

- Install napari
- Install this repo
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
`example_interactive_2d.py`.

Divisualisation also ships two optional napari dock widgets (**Plugins → Divisualisation**):

- **Edge error toggle** — show/hide all error layers at once.
- **Spacetime lift** — interactively fold time into a `z` axis so the selected
  tracks layers rise out of the moving image plane into a 3D "spacetime" cone,
  with a live lift-amount slider. Tick which tracks layers to lift (all by
  default), turn the toggle on, and scrub the time slider to sweep through the
  cone; turn it off to restore the flat 2D view exactly. This is the original
  `Divisualisation` render effect, made interactive and reversible.

#### Rendering: the 3D spacetime animation

Run `example_2d.py` or `example_3d.py` in ipython to render the spacetime mp4
animations shown above. These use the `Divisualisation` class, which folds time
into the `z` axis and drives a scripted animation render.
