import logging

import napari
import networkx as nx
import numpy as np
import traccuracy
from napari.utils.colormaps.colormap_utils import vispy_or_mpl_colormap

# from trackastra.tracking import load_ctc_graph, graph_to_napari_tracks
from traccuracy import EdgeFlag

from .utils import graph_to_napari_tracks

# from visualize_2d_tracking_errors import visualize_edge_errors

logger = logging.getLogger(__name__)


class Divisualisation:
    def __init__(
        self,
        time_scale: int = 10,
        z_scale: int | None = 1,
    ):
        self.time_scale = time_scale
        self.z_scale = z_scale
        # TODO adapt for 2D. Fake 3rd dimension that does not do anything
        self.scale = (self.time_scale, self.z_scale, 1, 1)

    # def render_error_video(self, viewer, path):
    #     Animation = napari.Animation()
    #     self.take_keyframe(viewer)
    #     self.move_slider(viewer)

    def visualize_gt(
        self,
        v: napari.Viewer,
        x: np.ndarray,
        masks: np.ndarray,
        graph: nx.DiGraph,
    ):
        tracks, _, properties = graph_to_napari_tracks(
            graph,
            properties=["t"],
        )

        # TODO figure out broadcasting for 2D case
        # xc_broadcast = np.broadcast_to(xc[None, ...], (len(xc),) + xc.shape)
        img_layer = v.add_image(
            # np.expand_dims(xc, 0),
            x,
            scale=(1, self.z_scale, 1, 1),
            # translate = [0, -scale[1] * img.shape[1] / 2 + scale[1], 0, 0]
            colormap="gray",
            rendering="mip",
            # depiction="plane",
        )

        labels_layer = v.add_labels(
            # np.expand_dims(mc, 0),
            masks,
            name="masks",
            scale=(1, self.z_scale, 1, 1),
            # rendering="iso_categorical",
            rendering="translucent",
            opacity=1.0,
            # depiction="plane",
        )

        # TODO remove hardcoded dep on specific property
        properties["gt"] = np.ones_like(properties["t"]) * 0.5
        # lin = np.array(properties["lineage"])

        # # shuffle lineage ids
        # rng = np.random.default_rng(0)
        # perm = rng.permutation(int(lin.max()) + 1)
        # lin_perm = perm[lin.astype(int)]
        # properties["lineage_permuted"] = 0.0 + 1.0 * (
        #     lin_perm.astype(float) / max(lin_perm.max(), 1)
        # )

        # properties["track_id_normalized"] = 0.3 + 0.7 * (track_ids / track_ids.max())
        # properties["lineage_normalized"] = lin.astype(float) / lin.max()

        properties = {"gt": properties["gt"]}

        # tracks[:, 1] = tracks[:, 1] * 2  # Double time points to match image
        # tracks = np.concat([tracks[:, 0:1], tracks[:, 1:2], tracks[:, 1:]], axis=1)

        logger.info("Adding gt tracks")

        # Scale tracks z dim by time
        assert tracks.shape[1] == 5
        tracks[:, -3] = tracks[:, -3] + self.time_scale * tracks[:, -4]

        tracks_layer = v.add_tracks(
            data=tracks,
            # graph=tracks_graph,
            name="tracks",
            properties=properties,
            # TODO can I just scale adaptively here?
            # scale=(self.time_scale, 1, 1, 1),
            color_by="gt",
            blending="translucent_no_depth",
            colormaps_dict={
                # "lineage_permuted": vispy_or_mpl_colormap("Greens"),
                # "lineage_permuted": label_colormap(seed=0.12),
                # "lineage_normalized": label_colormap(seed=0.12),
                # "track_id_normalized": vispy_or_mpl_colormap("gray_r"),
                # "gt": vispy_or_mpl_colormap("greens"),
                # For white bg
                "gt": vispy_or_mpl_colormap("Greens"),
            },
            # tail_width=2.0,
            tail_width=2,
            tail_length=1000,
            opacity=1.0,
        )

        def update_gt_state(event=None):
            t = v.dims.point[0]
            # Move clipping plane along Z axis to t (or adjust axis as needed)
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
        translate_to_tracks = [
            0,
            -self.z_scale * (x.shape[1] / 2) + self.z_scale,
            0,
            0,
        ]
        img_layer.affine.translate = translate_to_tracks
        labels_layer.affine.translate = translate_to_tracks
        v.dims.set_current_step(0, img_layer.data.shape[0])

        return v


def visualize_edge_errors(
    viewer: napari.Viewer,
    gt_graph: traccuracy.TrackingGraph,
    pred_graph: traccuracy.TrackingGraph,
    masks_original: np.ndarray,
    masks_tracked: np.ndarray,
    scale: tuple,
    feats: str = "",
    translate: list = [0, 0, 0],
):
    errors_layer = {}
    errors_data = {}
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
            edge_error_tracks.append([edge_id, u["t"], u["y"], u["x"]])
            edge_error_props["error_type"].append(i)

            # c_v = np.argwhere(reference_masks[v["t"]] == v["segmentation_id"]).mean(
            # axis=0
            # )
            edge_error_tracks.append([edge_id, v["t"], v["y"], v["x"]])
            edge_error_props["error_type"].append(i)

        edge_error_props = {k: np.array(v) for k, v in edge_error_props.items()}

        if len(edge_error_tracks) > 0:
            tracks = np.array(edge_error_tracks)
            # tracks[:, 1] = tracks[:, 1] * 2  # Double time points to match image
            tracks = np.concat([tracks[:, 0:1], tracks[:, 1:2], tracks[:, 1:]], axis=1)
            layer = viewer.add_tracks(
                data=tracks,
                properties=edge_error_props,
                color_by="error_type",
                # colormap=cmap,
                colormaps_dict={
                    "error_type": vispy_or_mpl_colormap(cmap),
                },
                scale=scale,
                # tail_width=8,
                tail_width=5,
                head_length=1,
                tail_length=1000,
                visible=True,
                blending="translucent_no_depth",
                opacity=1.0,
                translate=translate,
                name=error,
            )
            errors_layer[error] = layer
            errors_data[error] = {
                "tracks": tracks,
                "properties": edge_error_props,
            }
        else:
            logger.info(f"No edge errors of type {error}")

    return errors_layer, errors_data
