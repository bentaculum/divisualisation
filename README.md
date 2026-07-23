# divisualisation

[![PyPI](https://img.shields.io/pypi/v/divisualisation.svg?color=green)](https://pypi.org/project/divisualisation)
[![tests](https://github.com/bentaculum/divisualisation/workflows/Tests/badge.svg)](https://github.com/bentaculum/divisualisation/actions)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/divisualisation)](https://napari-hub.org/plugins/divisualisation)

A napari plugin to visualise cell-tracking errors, computed via [`traccuracy`](https://github.com/live-image-tracking-tools/traccuracy/), by lifting 2D/3D+time tracks into an interactive 3D "spacetime" view.

> **🆕 divisualisation is now a fully fledged napari plugin, with a stateful spacetime lifted view that integrates with regular napari workflows.**

2D tracking (bacteria) | 3D tracking (C. elegans nuclei)
:-: | :-:
<video src='https://github.com/user-attachments/assets/38d047b1-bc7b-4315-a192-97886a1bf906' width=180></video> | <video src='https://github.com/user-attachments/assets/3724faf2-9d24-428a-8b4f-84ee68646424' width=180/></video>

We originally introduced these visualisations to compare our results in [_Trackastra: Transformer-based cell tracking for live-cell microscopy_](https://github.com/weigertlab/trackastra) to other cell tracking algorithms.



<video src='https://github.com/user-attachments/assets/99ac7295-cab5-43a0-9899-4fa007b110f7' width=60></video>


## Installation

1. Please install napari as outlined [here](https://napari.org/stable/tutorials/fundamentals/installation.html).

2. After that, install divisualisation, either:
    - from within napari via **Plugins → Install/Uninstall Plugins…** (search for "divisualisation"),
    - or from PyPI:
      ```
      pip install divisualisation
      ```
    - or the latest development version from GitHub:
      ```
      pip install git+https://github.com/bentaculum/divisualisation.git
      ```

Note: requires Python ≥ 3.11 and napari ≥ 0.8.

## Usage

Open **Plugins → divisualisation → Lift tracks & Divisualisation**. The widget has two independent workflows, each in its own box:

- **Lift all tracks layers** — fold time into a `z` axis so every tracks layer rises out of the image plane into a 3D "spacetime" cone. Scrub the time slider to sweep through the cone; toggle off to restore the flat view exactly.
- **Divisualisation** — assign ground-truth / predicted / FN-edge / FP-edge tracks layers via the role dropdowns (auto-guessed from layer names), **Compute edge errors** from the GT/predicted tracks plus their labels, and lift with the error colouring. **Color division edges** draws each layer's parent→daughter edges as coloured tails (napari otherwise draws them in uncolourable white).

## Examples

Run in ipython — each loads data into a viewer, adds the tracks and edge-error overlays, and docks the widget:

- `example_2d.py` — bacteria (2D+t).
- `example_3d.py` — C. elegans nuclei (3D+t, `z` scaled ×10).
- `example_programmatic_2d.py` — fully scripted render (no GUI): build layers, lift with `SpacetimeLift`, overlay errors with `add_edge_error_tracks`, capture a `napari_animation` keyframe video.