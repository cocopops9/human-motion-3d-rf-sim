"""Tools: field-of-view check, sensor network check, conversion of recordings,
synthetic recordings, parameter reference."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from bodyscan.commands.base import Command
from bodyscan.log import info, set_quiet


class CheckView(Command):
    """Check that the LiDAR sees the whole person on the platform (feet AND head), and how finely.

    Run it on a SHORT capture made right after moving or tilting the sensor,
    with the person on the platform:

        python -m bodyscan capture-turntable C:\\lidar\\check1 --duration 20
        python -m bodyscan check-view C:\\lidar\\check1 --person-height 1.81

    Reports the sensor height and tilt, the platform centre and distance, the
    highest and lowest beam towards the platform, the height range of the
    person actually seen, the tilt that fits head and feet with a margin, and
    the sample spacing and smallest visible gap at the person's distance."""
    name = "check-view"
    help = "does the sensor see the whole person on the platform? (short capture)"

    def add_arguments(self, parser):
        parser.add_argument("run", help="capture directory (short capture, person on the platform)")
        parser.add_argument("--person-height", type=float, default=1.85, help="stature of the person [m]")
        parser.add_argument("--center", type=float, nargs=2, default=None, metavar=("X", "Y"),
                            help="platform centre in the floor frame, if it is not found automatically")
        parser.add_argument("--frames", type=int, default=20, help="frames used for the person")
        parser.add_argument("--margin", type=float, default=3.0, help="wanted margin at both ends [deg]")

    def run(self, args):
        from bodyscan.io import NpzRecording, median_range
        from bodyscan.pipelines.common import locate_person
        from bodyscan.scene import RansacFloor, RingPlatform
        from bodyscan.scene.coverage import beam_limits, sampling_at

        source = NpzRecording(args.run)
        # Floor from the empty scene, or from the frames (the floor is visible in both).
        if source.background_count():
            scene_range = source.background_range(0.5)
        else:
            scene_range = median_range([source.load(k).range_m for k in range(min(10, len(source)))], 0.5)
        points = source.sensor.xyz(scene_range)
        points = points[np.isfinite(points[..., 0])]
        floor = RansacFloor().estimate(points)
        world = floor.to_world(points)
        height = floor.sensor_height
        info(f"sensor: {height:.3f} m above the floor, its axis tilted {floor.tilt_deg:.1f} deg from the vertical")

        if args.center is not None:
            center = np.array(args.center, dtype=np.float64)
            info(f"platform centre (given): ({center[0]:.3f}, {center[1]:.3f}) m")
        else:
            start = None
            if source.background_count():
                start, _ = locate_person(source, floor, source.background_range(0.5))
            ring = RingPlatform().find(world, start if start is not None else np.array([1.1, 0.0]))
            if ring is not None and ring.plausible:
                center = np.asarray(ring.center)
                info(f"platform ring: centre ({center[0]:.3f}, {center[1]:.3f}) m, radius {ring.radius:.3f} m")
            elif start is not None:
                center = start
                info(f"platform ring not found; person at ({center[0]:.3f}, {center[1]:.3f}) m")
            else:
                raise SystemExit("neither the platform ring nor a person found; pass --center X Y")
        offset = center - floor.sensor_position[:2]
        distance = float(np.hypot(*offset))
        info(f"platform centre {distance:.2f} m from the sensor (horizontal)")

        top, bottom = beam_limits(source.sensor.directions, floor.world_from_sensor, center)
        feet_distance, head_distance, feet_height = distance - 0.15, distance - 0.10, 0.03
        need_bottom = -np.degrees(np.arctan2(height - feet_height, feet_distance))
        need_top = np.degrees(np.arctan2(args.person_height - height, head_distance))
        info(f"towards the platform: highest beam {top:+.1f} deg, lowest beam {bottom:+.1f} deg")
        info(f"  highest beam reaches {height + np.tan(np.radians(top)) * head_distance:.2f} m at the front of "
             f"the head ({head_distance:.2f} m); needed {args.person_height:.2f} m")
        info(f"  lowest beam reaches {height + np.tan(np.radians(bottom)) * feet_distance:.2f} m at the front of "
             f"the feet ({feet_distance:.2f} m); needed {feet_height:.2f} m")

        picks = np.linspace(0, len(source) - 1, min(args.frames, len(source))).astype(int)
        tops, lows = [], []
        for k in picks:
            seen = source.points(source.load(int(k)))
            seen = floor.to_world(seen)
            on = (np.hypot(*(seen[:, :2] - center).T) < 0.5) & (seen[:, 2] > 0.02) & (seen[:, 2] < 2.5)
            if on.sum() > 200:
                tops.append(np.percentile(seen[on, 2], 99.9))
                lows.append(np.percentile(seen[on, 2], 0.1))
        if tops:
            info(f"person seen from {np.median(lows):.2f} m to {np.max(tops):.2f} m (over {len(tops)} frames)")

        horizontal, vertical = source.sensor.angular_steps()
        near = sampling_at(distance - 0.15, horizontal, vertical)
        info(f"sampling at the front of the body ({near.distance:.2f} m): columns every "
             f"{1000 * near.horizontal:.1f} mm, beams every {1000 * near.vertical:.1f} mm, beam diameter about "
             f"{1000 * near.footprint:.0f} mm; smallest visible gap about {1000 * near.smallest_gap():.0f} mm "
             f"across, {1000 * near.smallest_gap('vertical'):.0f} mm along the height")

        low_delta = need_top + args.margin - top            # smallest upward turn of the fan that shows the head
        high_delta = need_bottom - args.margin - bottom      # largest upward turn that still shows the feet
        info()
        if low_delta > high_delta:
            info(f"the person does not fit in the field of view with {args.margin:g} deg margins at this "
                 "distance and height: move the sensor back or raise it")
        elif low_delta <= 0 <= high_delta:
            info(f"OK: head and feet are both inside the field of view (margins {top - need_top:.1f} deg at the "
                 f"head, {need_bottom - bottom:.1f} deg at the feet)")
        else:
            delta = 0.5 * (low_delta + high_delta)
            direction = "UP (less downward tilt)" if delta > 0 else "DOWN"
            info(f"tilt the sensor about {abs(delta):.0f} deg {direction}; any value between "
                 f"{min(abs(low_delta), abs(high_delta)):.0f} and {max(abs(low_delta), abs(high_delta)):.0f} deg works")
        return 0


class CheckSensor(Command):
    """Diagnose why no LiDAR frames arrive, one layer at a time.

        python -m bodyscan check-sensor --host os-122542000054.local

    Step 1 reads the sensor configuration over HTTP (where it sends its data,
    on which ports, whether it streams). Step 2 binds plain UDP sockets on
    those ports without the SDK and counts the packets. Cannot bind: another
    program owns the port (Ouster Studio, an old Python process). Binds, no
    packets: wrong udp_dest, standby mode, cable or switch, or the firewall.
    Packets arrive: the network is fine, the problem is in the SDK.
    Only --set-ports writes to the sensor."""
    name = "check-sensor"
    help = "network diagnosis of the Ouster sensor"

    def add_arguments(self, parser):
        parser.add_argument("--host", default="os-122542000054.local")
        parser.add_argument("--seconds", type=float, default=3.0)
        parser.add_argument("--set-ports", type=int, nargs=2, metavar=("LIDAR", "IMU"),
                            help="write new UDP ports to the sensor, e.g. --set-ports 7602 7603")
        parser.add_argument("--dest", default="192.168.33.30", help="destination used with --set-ports")

    def run(self, args):
        from bodyscan.capture.diagnostics import set_ports, show_configuration, test_sockets
        if args.set_ports:
            set_ports(args.host, args.set_ports[0], args.set_ports[1], args.dest)
            return 0
        test_sockets(show_configuration(args.host), args.seconds)
        return 0


class Convert(Command):
    """Ouster recording (.osf, or .pcap with --meta) -> capture directory usable by every command.

        python -m bodyscan convert C:\\lidar\\scene.osf C:\\lidar\\scene_run --background-seconds 3

    The first --background-seconds of the recording become the empty-scene
    frames (0: none); the rest become the frames."""
    name = "convert"
    help = "Ouster recording -> capture directory"

    def add_arguments(self, parser):
        parser.add_argument("recording")
        parser.add_argument("out", help="output directory (new or empty)")
        parser.add_argument("--meta", default=None, help="metadata json of a .pcap recording")
        parser.add_argument("--background-seconds", type=float, default=0.0)
        parser.add_argument("--quiet", action="store_true")

    def run(self, args):
        from bodyscan.capture.sensor import OusterSource
        from bodyscan.capture.session import Background, Record, Session
        from bodyscan.capture.writer import RunDirectory
        set_quiet(args.quiet)
        source = OusterSource(args.recording, meta=args.meta)
        run_directory = RunDirectory(args.out)
        log = {}
        try:
            run_directory.write_sensor(source)
            phases = ([Background(args.background_seconds, "empty-scene frames")]
                      if args.background_seconds > 0 else []) + [Record(None, float("inf"), "frames")]
            log = Session(source, run_directory, time.monotonic(), interactive=False).run(phases)
        finally:
            source.close()
            run_directory.close(log)
        info(f"background frames {log.get('background', 0)}, frames {log.get('frames', 0)}")
        return 0


class Simulate(Command):
    """Synthetic recording of a scenario, to try the commands without the sensor.

        python -m bodyscan simulate detect C:\\lidar\\sim_detect
        python -m bodyscan detect C:\\lidar\\sim_detect --rotation --human

    Scenarios: turntable (person on a turning platform), inplace (person
    turning by steps), detect (two platforms, a person and a chair on them, a
    person standing still, furniture, a stand). truth.json holds the truth."""
    name = "simulate"
    help = "write a synthetic recording (turntable, inplace, detect)"

    def add_arguments(self, parser):
        parser.add_argument("scenario", choices=["turntable", "inplace", "detect"])
        parser.add_argument("out", help="output directory")
        parser.add_argument("--frames", type=int, default=None, help="number of frames (default per scenario)")
        parser.add_argument("--rows", type=int, default=128)
        parser.add_argument("--columns", type=int, default=2048)
        parser.add_argument("--noise", type=float, default=0.005, help="range noise sigma [m]")
        parser.add_argument("--height", type=float, default=1.15, help="sensor height [m]")
        parser.add_argument("--tilt", type=float, default=0.0, help="downward tilt of the sensor [deg]")
        parser.add_argument("--quiet", action="store_true")

    def run(self, args):
        from bodyscan.synthetic import SyntheticSensor, scenario
        set_quiet(args.quiet)
        sensor = SyntheticSensor(rows=args.rows, columns=args.columns, height=args.height, tilt_deg=args.tilt,
                                 noise=args.noise)
        out = scenario(args.scenario, args.out, args.frames, sensor)
        info(f"wrote {out}")
        return 0


class Params(Command):
    """Parameter reference of every command (the source of docs/parameters.md).

        python -m bodyscan params --out docs/parameters.md"""
    name = "params"
    help = "print the parameter reference (markdown)"

    def add_arguments(self, parser):
        parser.add_argument("--out", default=None, help="write to this file instead of the console")

    def run(self, args):
        from bodyscan import config as cfg
        from bodyscan.capture.protocols import InPlaceCaptureConfig, TurntableCaptureConfig
        from bodyscan.pipelines.detect import DetectConfig
        from bodyscan.pipelines.inplace import InPlaceConfig
        from bodyscan.pipelines.mesh import MeshConfig
        from bodyscan.pipelines.smooth import SmoothMeshConfig
        from bodyscan.pipelines.turntable import TurntableConfig
        parts = ["# Parameter reference", "",
                 "Generated by `python -m bodyscan params` from the declarations in the code: every row is an "
                 "option of the command (`--name VALUE`), a key of a TOML configuration file (`[section]` "
                 "then `name = value`), and `--set section.name=VALUE`. See docs/tuning.md for how the main "
                 "parameters change the results.", ""]
        for title, config_class in (("bodyscan fuse (turntable)", TurntableConfig),
                                    ("bodyscan fuse-inplace", InPlaceConfig),
                                    ("bodyscan mesh", MeshConfig),
                                    ("bodyscan smooth", SmoothMeshConfig),
                                    ("bodyscan detect", DetectConfig),
                                    ("bodyscan capture-turntable", TurntableCaptureConfig),
                                    ("bodyscan capture-inplace", InPlaceCaptureConfig)):
            parts.append(cfg.to_markdown(config_class, title))
        text = "\n".join(parts) + "\n"
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            info(f"wrote {args.out}")
        else:
            print(text)
        return 0
