import logging

import napari
import networkx as nx
import numpy as np
import traccuracy
from napari.utils.colormaps.colormap_utils import vispy_or_mpl_colormap
from traccuracy import EdgeFlag

from .utils import graph_to_napari_tracks

logger = logging.getLogger(__name__)


class Divisualisation:
    def __init__(
        self,
        time_scale: int = 10,
        z_scale: int | None = 1,
        tracks_width: int = 1,
    ):
        self.time_scale = time_scale
        self.z_scale = z_scale
        # TODO adapt for 2D. Fake 3rd dimension that does not do anything
        self.scale = (self.time_scale, self.z_scale, 1, 1)
        self.tracks_width = tracks_width

    def visualize_gt(
        self,
        v: napari.Viewer,
        x: np.ndarray,
        masks: np.ndarray,
        graph: nx.DiGraph,
        time_attr: str = "t",
    ):
        tracks, _, properties = graph_to_napari_tracks(
            graph,
            properties=[time_attr],
        )

        # TODO figure out broadcasting for 2D case
        # xc_broadcast = np.broadcast_to(xc[None, ...], (len(xc),) + xc.shape)
        translate_to_tracks = [
            0,
            -self.z_scale * (x.shape[1] / 2) + self.z_scale,
            0,
            0,
        ]

        v.add_image(
            # np.expand_dims(xc, 0),
            x,
            scale=(1, self.z_scale, 1, 1),
            colormap="gray",
            rendering="mip",
            translate=translate_to_tracks,
            # depiction="plane",
        )

        v.add_labels(
            masks,
            name="masks",
            scale=(1, self.z_scale, 1, 1),
            # rendering="iso_categorical",
            rendering="translucent",
            translate=translate_to_tracks,
            opacity=0.2,
            # depiction="plane",
        )

        properties["gt"] = np.ones_like(properties[time_attr]) * 0.5

        properties = {"gt": properties["gt"]}
        logger.info("Adding gt tracks")

        # Scale tracks z dim by time
        assert tracks.shape[1] == 5
        tracks[:, -3] = tracks[:, -3] + self.time_scale * tracks[:, -4]

        tracks_layer = v.add_tracks(
            data=tracks,
            name="tracks",
            properties=properties,
            color_by="gt",
            blending="translucent_no_depth",
            colormaps_dict={
                "gt": vispy_or_mpl_colormap("Greens"),
            },
            tail_width=self.tracks_width,
            tail_length=1000,
            opacity=1.0,
        )

        def update_gt_state(event=None):
            t = v.dims.point[0]
            clipping_planes_tracks = [
                {
                    "position": (0, 0, 0),
                    "normal": (0, 0, 0),
                    "enabled": False,
                },
                {
                    "position": ((t + 1) * self.time_scale, 0, 0),
                    "normal": (-1, 0, 0),
                    "enabled": True,
                },
            ]
            tracks_layer.experimental_clipping_planes = clipping_planes_tracks
            tracks_layer.translate = [0, -self.time_scale * (t + 1), 0, 0]

        v.dims.events.point.connect(update_gt_state)
        v.dims.ndisplay = 3
        # v.dims.set_current_step(0, img_layer.data.shape[0])
        v.camera.center = (
            -self.time_scale * x.shape[0] / 2,
            v.camera.center[1],
            v.camera.center[2],
        )
        v.camera.zoom = 0.5

        return v

    def visualize_edge_errors(
        self,
        viewer: napari.Viewer,
        gt_graph: traccuracy.TrackingGraph,
        pred_graph: traccuracy.TrackingGraph,
        masks_original: np.ndarray,
        masks_tracked: np.ndarray,
    ):
        errors_layer = {}
        # errors_data = {}
        for i, (error, graph, reference_masks, cmap) in enumerate(
            zip(
                (EdgeFlag.CTC_FALSE_NEG, EdgeFlag.CTC_FALSE_POS),
                (gt_graph, pred_graph),
                (masks_original, masks_tracked),
                # ("hsv", "gray_r"),
                # ("PiYG", "PiYG"),
                ("cool", "cool"),
            )
        ):
            np.zeros_like(masks_original)
            edge_error_tracks = []
            edge_error_props = {"error_type": []}
            for edge_id, (u_id, v_id) in enumerate(graph.get_edges_with_flag(error)):
                # Skip edges that are not in the reference masks
                # if row.t_u >= len(reference_masks) or row.t_v >= len(reference_masks):
                # continue

                edge_id += 1  # avoid edge id 0
                u = graph.nodes[u_id]
                v = graph.nodes[v_id]
                # Get positions of the edge endpoints
                # c_u = np.argwhere(reference_masks[u["t"]] == u["segmentation_id"]).mean(
                # axis=0
                # )
                edge_error_tracks.append([edge_id, u["t"], u["z"], u["y"], u["x"]])
                edge_error_props["error_type"].append(i)

                # c_v = np.argwhere(reference_masks[v["t"]] == v["segmentation_id"]).mean(
                # axis=0
                # )
                edge_error_tracks.append([edge_id, v["t"], v["z"], v["y"], v["x"]])
                edge_error_props["error_type"].append(i)

            edge_error_props = {k: np.array(v) for k, v in edge_error_props.items()}

            if len(edge_error_tracks) > 0:
                tracks = np.array(edge_error_tracks)

                # Scale tracks z dim by time
                assert tracks.shape[1] == 5
                tracks[:, -3] = tracks[:, -3] + self.time_scale * tracks[:, -4]
                layer = viewer.add_tracks(
                    data=tracks,
                    properties=edge_error_props,
                    color_by="error_type",
                    # colormap=cmap,
                    colormaps_dict={
                        "error_type": vispy_or_mpl_colormap(cmap),
                    },
                    tail_width=self.tracks_width * 2,
                    head_length=1,
                    tail_length=1000,
                    visible=True,
                    blending="translucent_no_depth",
                    opacity=1.0,
                    name=error,
                )
                errors_layer[error] = layer

            else:
                logger.info(f"No edge errors of type {error}")

        def update_errors_state(event=None):
            t = viewer.dims.point[0]
            clipping_planes_tracks = [
                {
                    "position": (0, 0, 0),
                    "normal": (0, 0, 0),
                    "enabled": False,
                },
                {
                    "position": ((t + 1) * self.time_scale, 0, 0),
                    "normal": (-1, 0, 0),
                    "enabled": True,
                },
            ]
            for error, layer in errors_layer.items():
                layer.experimental_clipping_planes = clipping_planes_tracks
                layer.translate = [0, -self.time_scale * (t + 1), 0, 0]

        viewer.dims.events.point.connect(update_errors_state)
        viewer.dims.ndisplay = 3

        return viewer
