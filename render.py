from napari_animation import Animation


def video(v, args, name):
    animation = Animation(v)

    def replace_gt_tracks_data():
        # TODO write in cropping
        z_cutoff = int(image_layer.plane.position[0])
        print(f"{z_cutoff=}")
        idx = gt_tracks_data["tracks"][:, 1] <= z_cutoff
        print(f"{idx.sum()=}")

        v.layers.remove("tracks")
        properties = {k: v[idx] for k, v in gt_tracks_data["properties"].items()}
        tracks_layer = v.add_tracks(
            data=gt_tracks_data["tracks"][idx],
            properties=properties,
            scale=args.scale,
            color_by="gt",
            blending="translucent_no_depth",
            colormaps_dict={
                "gt": vispy_or_mpl_colormap("Greens"),
            },
            # tail_width=2.0,
            tail_width=0.5,
            opacity=1.0,
            name="tracks",
        )

        for error, data in errors_data.items():
            idx = data["tracks"][:, 1] <= z_cutoff
            if idx.sum() == 0:
                # dummy
                idx[0] = True

            v.layers.remove(error)
            properties = {k: v[idx] for k, v in data["properties"].items()}

            v.add_tracks(
                data=data["tracks"][idx],
                properties=properties,
                color_by="error_type",
                # colormap=cmap,
                colormaps_dict={
                    "error_type": vispy_or_mpl_colormap("cool"),
                },
                scale=args.scale,
                # tail_width=8,
                tail_width=2,
                head_length=1,
                tail_length=1,
                blending="translucent_no_depth",
                opacity=1.0,
                name=error,
            )

    image_layer.plane.events.position.connect(replace_gt_tracks_data)
    # labels_layer.visible = True
    # gt_tracks_layer.experimental_clipping_planes = [
    #     {
    #         "position": (0, 0, 0),
    #         "normal": (1, 0, 0),  # point up in z (i.e: show stuff above plane)
    #     }
    # ]

    steps = image_layer.data.shape[0] // 2 // 2

    image_layer.plane.position = (0, 0, 0)
    # v.camera.center = (158, 516, 216)
    # Start
    animation.capture_keyframe(steps=steps)

    # # gt_tracks_layer.experimental_clipping_planes[0].position = (0, 0, 0)
    # v.camera.angles = (177, 56, 177)
    image_layer.plane.position = (image_layer.data.shape[0] - 1, 0, 0)
    # v.camera.angles = (-17, 49, 160)
    # v.camera.center = args.center

    # Render out errors
    animation.capture_keyframe(steps=steps)
    # labels_layer.plane.position = (labels_layer.data.shape[0] - 1, 0, 0)

    # image_layer.plane.position = (image_layer.data.shape[0] - 1 + 10, 0, 0)
    # v.camera.angles = (160, 56, 155)
    # animation.capture_keyframe(steps=steps)

    # v.camera.angles = (177, 56, 177)

    # TODO expose rotation to config

    # v.camera.angles = (3, -31.133148879520093, -178.41736087980567)
    v.camera.angles = (8.15823293225907, -32.9710370784703, 166.09842227043492)
    v.layers[1].opacity = 0.3
    # v.camera.angles = (0, 49, 180)
    # Tilt
    animation.capture_keyframe(steps=int(steps // 2.2))

    # image_layer.plane.position = (image_layer.data.shape[0] - 1, 0, 0)
    v.camera.angles = args.angles

    # Tilt back
    v.layers[1].opacity = 1.0
    animation.capture_keyframe(steps=int(steps // 2.2))

    image_layer.plane.position = (0, 0, 0)
    # v.camera.angles = args.angles
    # v.camera.center = (158, 516, 216)
    # Hack to get this freaking color set

    # Roll up to orig
    animation.capture_keyframe(steps=steps)

    # image_layer.plane.position = (0, 0, 0)

    # animation.capture_keyframe(steps=steps)

    # image_layer.plane.events.position.connect(replace_labels_data)
    # labels_layer.visible = True
    # labels_layer.experimental_clipping_planes = [{
    #     "position": (0, 0, 0),
    #     "normal": (-1, 0, 0),  # point up in z (i.e: show stuff above plane)
    # }]

    # image_layer.plane.position = (59, 0, 0)
    # # access first plane, since it's a list
    # labels_layer.experimental_clipping_planes[0].position = (59, 0, 0)
    # animation.capture_keyframe(steps=steps)

    # image_layer.plane.position = (0, 0, 0)
    # animation.capture_keyframe(steps=30)

    print("Saving animation")
    animation.animate(
        filename=f"{name}.mp4",
        canvas_only=True,
        quality=5,
        fps=12,
    )
