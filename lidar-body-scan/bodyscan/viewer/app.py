"""The interactive viewer (Viser, in the web browser), replacing the review videos.

    python -m bodyscan view --results C:\\lidar\\cluster_results_v160\\results --segments C:\\lidar\\open
        --avatar C:\\lidar\\s02_avatar.npz --model SMPLX_MALE.npz --evaluations C:\\lidar\\evaluation_v160

then open http://localhost:8080. The panel on the right:

    Take       motion type (walk01, jump02, ...) and repetition; Load
    Playback   frame slider, play, speed, previous or next flagged frame
    Show       the person's points (coloured by sensor or by distance to the body),
               the body (plain, guessed limbs in violet, or a heatmap: this frame's
               error, the take's median or p95 error, violation rate, coverage),
               the skeleton, the sensor
    Numbers    the frame (time, flags, guessed limbs, pixel classes) and the take
               (evaluation summary, worst parts); curves over the take

The viewer runs on the machine that holds the data; on a cluster node, forward
the port (ssh -L 8080:<node>:8080 ...) and open it locally.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from bodyscan.evaluation.heatmap import colours
from bodyscan.viewer.data import MODE_SCALES, VERTEX_MODES, TakeData, catalogue

BODY_COLOUR = (196, 228, 222)
GUESS_COLOUR = np.array([176, 150, 232], dtype=np.uint8)
POINT_COLOUR = np.array([40, 170, 255], dtype=np.uint8)
BONES = [(j, p) for j, p in enumerate((-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19))
         if p >= 0]


class ViewerSession:
    """The Viser server, its controls and the take on display."""

    def __init__(self, groups: dict, avatar_path, model_path=None, port: int = 8080, host: str = "0.0.0.0"):
        import viser
        self.groups = groups
        if not groups:
            raise SystemExit("no tracked take found (a results folder holds one subfolder per take with motion.npz)")
        self.avatar_path, self.model_path = avatar_path, model_path
        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.set_up_direction("+z")
        self.take: TakeData | None = None
        self.frame = 0
        self.handles: dict = {}
        self._build_controls()

    # -- controls ------------------------------------------------------------------------------
    def _build_controls(self) -> None:
        gui = self.server.gui
        kinds = list(self.groups)
        with gui.add_folder("Take"):
            self.kind = gui.add_dropdown("Motion", kinds, initial_value=kinds[0])
            self.repetition = gui.add_dropdown("Take", [e.name for e in self.groups[kinds[0]]])
            self.load_button = gui.add_button("Load")
        with gui.add_folder("Playback"):
            self.slider = gui.add_slider("Frame", 0, 1, 1, 0)
            self.playing = gui.add_checkbox("Play", False)
            self.speed = gui.add_slider("Speed", 0.1, 2.0, 0.1, 1.0)
            self.previous_flag = gui.add_button("Previous flagged frame")
            self.next_flag = gui.add_button("Next flagged frame")
        with gui.add_folder("Show"):
            self.show_points = gui.add_checkbox("Points", True)
            self.point_mode = gui.add_dropdown("Point colour", ("by sensor", "distance to body"),
                                               initial_value="distance to body")
            self.show_body = gui.add_checkbox("Body", True)
            self.body_mode = gui.add_dropdown("Body colour", VERTEX_MODES, initial_value="guessed limbs")
            self.opacity = gui.add_slider("Body opacity", 0.2, 1.0, 0.1, 1.0)
            self.show_skeleton = gui.add_checkbox("Skeleton", False)
            self.show_sensor = gui.add_checkbox("Sensor", True)
            self.legend = gui.add_markdown("")
        with gui.add_folder("Numbers"):
            self.frame_text = gui.add_markdown("Load a take.")
            self.take_text = gui.add_markdown("")
        self.plot_folder = gui.add_folder("Curves")
        self.plot = None

        @self.kind.on_update
        def _(_event) -> None:
            self.repetition.options = [e.name for e in self.groups[self.kind.value]]
            self.repetition.value = self.repetition.options[0]

        @self.load_button.on_click
        def _(_event) -> None:
            self.load(self.kind.value, self.repetition.value)

        @self.slider.on_update
        def _(_event) -> None:
            self.show(int(self.slider.value))

        for control in (self.show_points, self.point_mode, self.show_body, self.body_mode, self.opacity,
                        self.show_skeleton, self.show_sensor):
            control.on_update(lambda _event: self.show(self.frame))

        @self.previous_flag.on_click
        def _(_event) -> None:
            self._jump_to_flag(-1)

        @self.next_flag.on_click
        def _(_event) -> None:
            self._jump_to_flag(+1)

    def _jump_to_flag(self, direction: int) -> None:
        if self.take is None:
            return
        flagged = self.take.flagged_frames()
        if direction > 0:
            later = flagged[flagged > self.frame]
            target = int(later[0]) if len(later) else self.frame
        else:
            earlier = flagged[flagged < self.frame]
            target = int(earlier[-1]) if len(earlier) else self.frame
        self.slider.value = target

    # -- loading and drawing ---------------------------------------------------------------------
    def entry(self, kind: str, name: str):
        for e in self.groups[kind]:
            if e.name == name:
                return e
        raise KeyError(name)

    def load(self, kind: str, name: str) -> None:
        e = self.entry(kind, name)
        self.frame_text.content = f"Loading {name} ..."
        self.take = TakeData(e.motion, e.segments, self.avatar_path, self.model_path, e.evaluation)
        self.slider.max = max(len(self.take) - 1, 1)
        self.slider.value = 0
        self.take_text.content = self.take.take_text()
        self._plot_curves()
        self.show(0)

    def _plot_curves(self) -> None:
        if self.plot is not None:
            self.plot.remove()
            self.plot = None
        series = self.take.series()
        names = [key for key in ("violation_share", "unexplained_share") if key in series]
        if not names:
            return
        data = (np.asarray(series["time"], dtype=np.float64),) + tuple(
            np.nan_to_num(np.asarray(series[key], dtype=np.float64), nan=0.0) for key in names)
        colour = {"violation_share": "#be141e", "unexplained_share": "#f46d43"}
        with self.plot_folder:
            self.plot = self.server.gui.add_uplot(
                data, ({"label": "time [s]"},) + tuple({"label": key.replace("_", " "), "stroke": colour[key],
                                                        "width": 1.5} for key in names),
                title="share of the ranged pixels per frame", aspect=1.6)

    def show(self, k: int) -> None:
        if self.take is None:
            return
        take = self.take
        k = int(np.clip(k, 0, len(take) - 1))
        self.frame = k
        scene = self.server.scene
        vertices = take.vertices(k)
        # the body
        mode = self.body_mode.value
        if self.show_body.value:
            values = take.vertex_values(k, mode)
            if mode == "plain" or (mode != "guessed limbs" and values is None):
                self.handles["/body"] = scene.add_mesh_simple(
                    "/body", vertices, take.faces, color=BODY_COLOUR,
                    opacity=self.opacity.value if self.opacity.value < 1.0 else None)
                self.legend.content = ("No evaluation for this take: run evaluate-take for heatmaps."
                                       if mode not in ("plain", "guessed limbs") else "")
            else:
                rgb = np.tile(np.array(BODY_COLOUR, dtype=np.uint8), (len(vertices), 1))
                if mode == "guessed limbs":
                    rgb[take.guessed_mask(k)] = GUESS_COLOUR
                    self.legend.content = "Violet: limbs the tracker guessed (no sensor saw them)."
                else:
                    _, vmin, vmax, unit = MODE_SCALES[mode]
                    if mode == "coverage":
                        rgb = colours(1.0 - values, 0.0, 1.0)
                        self.legend.content = "Coverage: dark green = observed in every frame, red = never."
                    else:
                        rgb = colours(values, vmin, vmax)
                        scale = f"{1000 * vmax:.0f} mm" if unit == "m" else f"{vmax:.0%}"
                        self.legend.content = (f"{mode}: dark green 0, yellow at 40 %, red at {scale} and above; "
                                               "grey: no sensor observed it.")
                self._coloured_body(vertices, take.faces, rgb)
        else:
            self._hide("/body")
        # the points
        points = take.points(k)
        if self.show_points.value and len(points):
            if self.point_mode.value == "distance to body":
                distances = take.point_distances(k, points)
                rgb = colours(distances, 0.0, 0.05)
            else:
                rgb = np.tile(POINT_COLOUR, (len(points), 1))
            self.handles["/points"] = scene.add_point_cloud("/points", points.astype(np.float32), rgb,
                                                            point_size=0.012, point_shape="circle")
        else:
            self._hide("/points")
        # the skeleton
        joints = take.joints(k)
        if self.show_skeleton.value and joints is not None:
            segments = np.stack([[joints[j], joints[p]] for j, p in BONES])
            self.handles["/skeleton"] = scene.add_line_segments("/skeleton", segments, (255, 200, 40),
                                                                thickness=0.012)
        else:
            self._hide("/skeleton")
        # the sensor
        sensor = take.sensor_position
        if self.show_sensor.value and sensor is not None:
            self.handles["/sensor"] = scene.add_frame("/sensor", axes_length=0.25, axes_radius=0.01,
                                                      position=tuple(float(v) for v in sensor))
        else:
            self._hide("/sensor")
        self.frame_text.content = take.frame_text(k)

    def _coloured_body(self, vertices, faces, rgb) -> None:
        import trimesh
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        alpha = int(round(255 * self.opacity.value))
        mesh.visual.vertex_colors = np.concatenate([rgb, np.full((len(rgb), 1), alpha, dtype=np.uint8)], axis=1)
        self.handles["/body"] = self.server.scene.add_mesh_trimesh("/body", mesh)

    def _hide(self, name: str) -> None:
        handle = self.handles.get(name)
        if handle is not None:
            handle.visible = False

    # -- running -------------------------------------------------------------------------------
    def run(self, frame_rate: float = 20.0) -> None:
        """Serve until interrupted; while 'Play' is ticked the frames advance at the take's speed."""
        last = time.time()
        while True:
            time.sleep(1.0 / 60.0)
            if self.take is None or not self.playing.value:
                last = time.time()
                continue
            elapsed = (time.time() - last) * self.speed.value
            step = int(elapsed * frame_rate)
            if step >= 1:
                last = time.time()
                self.slider.value = (self.frame + step) % len(self.take)

    def stop(self) -> None:
        self.server.stop()


def run_viewer(results, segments=None, evaluations=None, avatar=None, model=None, port: int = 8080,
               host: str = "0.0.0.0", take: str | None = None) -> None:
    from bodyscan.log import info
    groups = catalogue(Path(results), segments, evaluations)
    session = ViewerSession(groups, avatar, model, port, host)
    count = sum(len(v) for v in groups.values())
    info(f"viewer: {count} takes in {len(groups)} motion types; open http://localhost:{port} in a browser "
         "(Ctrl+C to stop)")
    if take is not None:
        for kind, entries in groups.items():
            if any(e.name == take for e in entries):
                session.kind.value = kind
                session.repetition.value = take
                session.load(kind, take)
    try:
        session.run()
    except KeyboardInterrupt:
        session.stop()
