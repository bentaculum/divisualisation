import napari
import numpy as np


if __name__ == "__main__":
    # Create sample volume
    x = np.random.randint(0, 100, (100, 100, 100))

    x_broadcast = np.broadcast_to(x[None, ...], (len(x),) + x.shape)

    viewer = napari.Viewer()
    layer = viewer.add_image(x_broadcast, rendering="mip", name="volume")

    viewer.dims.ndisplay = 3

    # Update clipping plane based on time (axis 0)
    def update_clipping_plane(event=None):
        t = viewer.dims.point[0]
        # Move clipping plane along Z axis to t (or adjust axis as needed)
        layer.experimental_clipping_planes = [
            {
                "position": (t, 0, 0),
                "normal": (1, 0, 0),
                "enabled": True,
            },
            {
                "position": (t, 0, 0),
                "normal": (-1, 0, 0),
                "enabled": True,
            },
        ]

    # Connect event
    viewer.dims.events.point.connect(update_clipping_plane)
