gt = load_from(file_gt)
pred = load_from(file_pred)
img = load_from(file_img)

metrics, matched = compute_metrics()

import utils

class Divisualisation():
    def __init__(self, matched, scale, )

    def visualize_tracks(self, viewer)
        register_slider_for_tracks()    


    def visualize_errors(
        self,
        viewer,
        matched,
    ):
        napari_tracks = convert(matched.gt)
        v = Viewer()
        visualize_gt()
        visualize_errors(matched)

        go_to_last_frame()

        register_event_listener_for_slider()

        return v

    def render_error_video(viewer, path):
        Animation = napari.Animation()
        take_keyframe()
        move_slider()


    def register_slider()
        pass


# Fast 3d playable video
# In 2d, copy the video, instead of broadcasting?

# Example script
d = Divisualisation(matched, scale=(1, 5, 1, 1))

v = d.visualize_graph(v, graph)
v = d.visualize_errors(v, matched)

# Maybe a one liner
d.render_video()


