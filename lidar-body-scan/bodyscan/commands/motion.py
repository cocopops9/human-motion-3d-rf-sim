"""Commands for moving people: capture, segmentation, avatar, tracking, export, review, simulation, bench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bodyscan.capture.protocols import CountdownConfig, SensorConfig
from bodyscan.commands.base import Command, ConfiguredCommand
from bodyscan.commands.capture import CaptureCommand
from bodyscan.config import param, section
from bodyscan.log import info, set_quiet, warning
from bodyscan.pipelines.common import FrameConfig
from bodyscan.scene import FloorConfig


# ----------------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------------

@dataclass
class MotionTimingConfig(CountdownConfig):
    """Time to leave the room, empty room, countdown, recording with sound cues."""
    leave_seconds: float = param(10.0, "time to walk out of the sensor's view after starting the command, before "
                                       "the empty room is recorded (low ticks every second; nothing stored)",
                                 unit="s")
    background_seconds: float = param(10.0, "empty-room recording (nobody in the sensor's view): starts with two "
                                            "low beeps", unit="s")
    delay: float = param(10.0, "countdown to walk to the floor mark and take the A-pose (a beep every second)",
                         unit="s")
    duration: float | None = param(None, "record this long; not set: until ENTER (or the safety stop)", unit="s")
    hold: float = param(3.0, "standing still in the A-pose at the start (one long beep, then two short high "
                             "beeps: move) and at the end (one long beep)", unit="s")
    back_seconds: float = param(8.0, "three medium beeps this long before the final hold: go back to the mark "
                                     "(0: no such cue)", unit="s")
    cue_every: float = param(0.0, "one short beep every this many seconds while moving: change movement "
                                  "(0: none)", unit="s")
    max_seconds: float = param(1800.0, "safety stop of a recording that waits for ENTER", unit="s")
    expected_mode: str = param("1024x20", "warn if the sensor runs another lidar mode ('' = no check)")


@dataclass
class MotionCaptureConfig:
    sensor: SensorConfig = section(SensorConfig)
    timing: MotionTimingConfig = section(MotionTimingConfig)


def _segmentation_config():
    from bodyscan.dynamic.segment import MotionSegmentationConfig

    @dataclass
    class SegmentMotionConfig:
        frames: FrameConfig = section(FrameConfig)
        floor: FloorConfig = section(FloorConfig)
        segmentation: MotionSegmentationConfig = section(MotionSegmentationConfig)

    return SegmentMotionConfig


def _avatar_config():
    from bodyscan.dynamic.avatar import AvatarConfig

    @dataclass
    class AvatarCommandConfig:
        avatar: AvatarConfig = section(AvatarConfig)

    return AvatarCommandConfig


def _export_config():
    from bodyscan.dynamic.export import ExportConfig

    @dataclass
    class ExportCommandConfig:
        export: ExportConfig = section(ExportConfig)

    return ExportCommandConfig


# The configuration classes are built from modules that do not import PyTorch,
# so the command line works on the capture PC.
SegmentMotionConfig = _segmentation_config()
AvatarCommandConfig = _avatar_config()
ExportCommandConfig = _export_config()


# ----------------------------------------------------------------------------
# Capture
# ----------------------------------------------------------------------------

class CaptureMotion(CaptureCommand):
    """Record a person moving around the LiDAR (walking, jumping, ...).

        python -m bodyscan capture-motion C:\\lidar\\walk01 --duration 60

        leave (--leave-seconds, default 10 s: walk out of the sensor's view;
               one long low beep, then low ticks) ->
        empty room (--background-seconds, default 10 s: two low beeps, then
               silence; nobody in view) ->
        countdown (--delay: a beep every second; walk to the floor mark,
               take the A-pose) ->
        recording (--duration seconds, or until ENTER), with beeps:
            1 long            stand still in the A-pose (--hold, 3 s)
            2 short high      move
            1 short           change movement (every --cue-every s, if set)
            3 medium          go back to the mark (--back-seconds before the end)
            1 long            A-pose, stand still (the last --hold s)
            2 (end)           done

    Every take should start and end with 3 s standing still in the A-pose on
    the floor mark (the tracker starts from it). Use the sensor in 1024x20
    mode (more frames per second; the vertical spacing limits the detail
    anyway): set it in the sensor's web page or with the Ouster SDK."""
    name = "capture-motion"
    help = "record a person moving around the LiDAR (Ouster sensor)"
    config_class = MotionCaptureConfig

    def phases(self, config, devices):
        from bodyscan.capture.protocols import leave_phases
        from bodyscan.capture.session import Background, Countdown, Motion
        t = config.timing
        return leave_phases(t) + [
            Background(t.background_seconds, f"{t.background_seconds:g} s: empty room, stay out of the sensor's "
                                             "view"),
            Countdown(t.delay, f"{t.delay:g} s: walk to the floor mark and take the A-pose"),
            Motion(t.duration, t.max_seconds, t.hold, t.back_seconds, t.cue_every,
                   (f"{t.duration:g} s: " if t.duration else "until ENTER: ") +
                   f"{t.hold:g} s A-pose, the motion, {t.hold:g} s A-pose")]

    def extra_log(self, config, devices):
        return {"protocol": "motion", "duration": config.timing.duration}

    def check_source(self, config, source):
        expected = config.timing.expected_mode
        if not expected:
            return
        try:
            mode = source.lidar_mode()
        except AttributeError:                                          # metadata without the mode
            warning("could not read the lidar mode of the sensor")
            return
        if mode != expected:
            warning(f"the sensor runs in {mode} mode, not {expected}: moving people are better captured at "
                    "20 frames per second (1024x20). Change it in the sensor's web page, or pass "
                    "--expected-mode '' to silence this check")


class CheckTake(Command):
    """Quality report of a take, right after capture-motion and segment-motion.

        python -m bodyscan capture-motion C:\\lidar\\open\\s01_check --duration 20
        python -m bodyscan segment-motion C:\\lidar\\open\\s01_check --out C:\\lidar\\open\\s01_check_seg
        python -m bodyscan check-take C:\\lidar\\open\\s01_check C:\\lidar\\open\\s01_check_seg

    Reports: the columns received per frame (a lost UDP packet is 16 columns:
    how many frames lose one, which packets are lost, how wide the hole is at
    the person's distance), the frames without a person, and the distance of
    the person between the beeps of the take (1.5 to 4 m is the working range)."""
    name = "check-take"
    help = "quality report of a take: lost columns, person found, distance over time"

    def add_arguments(self, parser):
        parser.add_argument("take", help="capture directory (capture-motion)")
        parser.add_argument("segments", help="segment-motion output folder")
        parser.add_argument("--near", type=float, default=1.5, help="nearest good distance [m]")
        parser.add_argument("--far", type=float, default=4.0, help="farthest good distance [m]")

    def run(self, args):
        import glob
        import json
        import numpy as np
        take, seg = Path(args.take), Path(args.segments)
        paths = sorted(glob.glob(str(take / "frames" / "frame_*.npz")))
        if not paths:
            raise SystemExit(f"{take}: no frames (a capture directory of capture-motion is expected)")
        ok, times, masks = [], [], []
        for path in paths:
            data = np.load(path)
            ok.append(float(data["columns_ok"]) if "columns_ok" in data.files else 1.0)
            times.append(float(data["time"]) if "time" in data.files else np.nan)
            if "column_ok" in data.files:
                masks.append(data["column_ok"])
        ok, times = np.array(ok), np.array(times)
        print(f"{take}: {len(ok)} frames")

        print("columns received (a lost UDP packet is 16 columns)")
        print(f"  min {ok.min():.3f}   5th percentile {np.percentile(ok, 5):.3f}   median {np.median(ok):.3f}")
        for limit in (0.99, 0.95, 0.90):
            print(f"  frames below {limit:.2f}: {int((ok < limit).sum())} ({100 * (ok < limit).mean():.0f} %)")
        print("  segment-motion drops frames below 0.90")

        if len(masks) == len(ok):
            masks = np.array(masks)
            columns = masks.shape[1]
            lost = (masks == 0)
            per_frame = lost.sum(axis=1)
            packets = lost.reshape(len(ok), columns // 16, 16).any(axis=2)
            print(f"  lost packets per frame: mean {packets.sum(axis=1).mean():.2f}, "
                  f"max {int(packets.sum(axis=1).max())}")
            share = packets.mean(axis=0)
            top = np.argsort(share)[::-1][:5]
            print("  packets lost most often (packet number 0 to "
                  f"{packets.shape[1] - 1}, share of frames): "
                  + ", ".join(f"{int(k)}: {100 * share[k]:.0f} %" for k in top if share[k] > 0))
            if len(ok) > 1:
                again = (packets[1:] & packets[:-1]).sum() / max(packets[:-1].sum(), 1)
                print(f"  a packet lost in one frame is also lost in the next one in {100 * again:.0f} % of the cases "
                      "(near 0: random; high: a fixed cause)")
            span = 360.0 / columns * 16
            print(f"  one lost packet is a hole {span:.1f} deg wide: {0.0175 * span * 2.5:.2f} m at 2.5 m")
        else:
            print("  (this take has no per-column record: it was made with an older version)")

        summary = json.load(open(seg / "segments.json"))
        frames = summary["frames"]
        missing = summary.get("missing", [])
        print(f"person found in {len(frames)} of {len(frames) + len(missing)} frames"
              + (f"; frames without a person: {missing[:20]}{' ...' if len(missing) > 20 else ''}" if missing else ""))
        if not frames:
            return 1
        # the segmented times are sensor times; the cues and record_start are on the recording PC clock,
        # which every frame file also stores: the window of a frame comes from its own file
        index = np.array([f["index"] for f in frames], dtype=np.int64)
        t = times[np.clip(index, 0, len(times) - 1)]
        distance = np.array([f["distance_m"] for f in frames])
        points = np.array([f["points"] for f in frames])

        try:
            log = json.load(open(take / "capture.json"))
        except OSError:
            log = {}
        origin = float(log.get("record_start", np.nanmin(times)))
        cues = log.get("cues") or []
        edges, labels = [0.0], ["start"]
        for cue in cues:
            at, label = (cue[0], cue[1]) if isinstance(cue, (list, tuple)) else (cue, "beep")
            edges.append(float(at)), labels.append(label)
        end = float(np.nanmax(t) - origin) + 0.1
        edges.append(end)
        if len(edges) == 2:
            edges = list(np.arange(0.0, end + 4.0, 4.0))
            labels = [""] * len(edges)
        print("window (s after the start of the take)   frames   distance [m]      points (median)")
        rel = t - origin
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            m = (rel >= a) & (rel < b)
            if m.any():
                print(f"  {a:6.1f} to {b:6.1f}  {labels[k]:<20} {int(m.sum()):5d}   "
                      f"{distance[m].min():5.2f} to {distance[m].max():5.2f}     {int(np.median(points[m])):6d}")
        outside = int(((distance < args.near) | (distance > args.far)).sum())
        print(f"distance of the person: {distance.min():.2f} to {distance.max():.2f} m; "
              f"{outside} frames outside {args.near:g} to {args.far:g} m")
        return 0


# ----------------------------------------------------------------------------
# Processing
# ----------------------------------------------------------------------------

class SegmentMotion(ConfiguredCommand):
    """Cut the moving person out of every frame of a recording (numpy and Open3D only).

        python -m bodyscan segment-motion C:\\lidar\\walk01 --out C:\\lidar\\walk01_seg

    Needs the empty-room frames of the recording (capture-motion records them
    first). Writes scene.npz, frames/person_XXXXX.npz and segments.json."""
    name = "segment-motion"
    help = "moving person in every frame of a recording"
    config_class = SegmentMotionConfig

    def add_arguments(self, parser):
        parser.add_argument("run", help="capture directory (lut.npz, background/, frames/)")
        parser.add_argument("--out", required=True, help="output folder")
        parser.add_argument("--start", type=int, default=0, help="first frame")
        parser.add_argument("--stop", type=int, default=None, help="last frame (excluded)")
        self.add_config_arguments(parser)

    def run(self, args):
        config = self.prepare(args)
        if config is None:
            return 0
        from bodyscan.dynamic.segment import MotionSegmenter
        from bodyscan.io import NpzRecording
        source = NpzRecording(args.run, config.frames.min_range, config.frames.max_range)
        MotionSegmenter(source, config.segmentation, config.floor).run(args.out, args.start, args.stop)
        return 0


class FitAvatar(ConfiguredCommand):
    """Fit the body model (SMPL-X) to a person's turntable scan: the avatar used for tracking.

        python -m bodyscan avatar person_tt11_mesh.ply --model SMPLX_NEUTRAL.npz
            --out person_tt11_avatar

    The scan is the output of 'fuse' or 'mesh' (z up, feet on z = 0, A-pose,
    palms forward). Writes <out>.npz (the avatar), <out>_rest.ply (T-pose with
    the scan's detail), <out>_scan_pose.ply (as fitted, to compare with the
    scan) and <out>.json (fit report). Needs PyTorch."""
    name = "avatar"
    help = "fit the body model to a turntable scan (the avatar)"
    config_class = AvatarCommandConfig

    def add_arguments(self, parser):
        parser.add_argument("scan", help="fused point cloud or mesh of the person (.ply)")
        parser.add_argument("--model", required=True, help="SMPL-X model .npz (or the test body)")
        parser.add_argument("--out", required=True, help="output base name")
        self.add_config_arguments(parser)

    def run(self, args):
        config = self.prepare(args)
        if config is None:
            return 0
        import torch
        from bodyscan.body.model import BodyModel
        from bodyscan.dynamic.avatar import AvatarFitter, load_scan, report_text, write_meshes
        from bodyscan.jsonio import write_json
        c = config.avatar
        device = ("cuda" if torch.cuda.is_available() else "cpu") if c.device == "auto" else c.device
        model = BodyModel(args.model, c.num_betas, device, torch.float32 if device.startswith("cuda") else torch.float64)
        points, normals = load_scan(args.scan, max_points=400000)
        info(f"scan: {len(points)} points")
        avatar = AvatarFitter(model, c).fit(points, normals)
        base = Path(args.out)
        if base.suffix == ".npz":
            base = base.with_suffix("")
        avatar.save(str(base) + ".npz")
        write_meshes(avatar, model, base)
        write_json(str(base) + ".json", avatar.report)
        info(report_text(avatar))
        info(f"wrote {base}.npz, {base}_rest.ply, {base}_scan_pose.ply")
        return 0


class Track(ConfiguredCommand):
    """Track a moving person: the avatar fitted to every frame, then the whole sequence refined.

        python -m bodyscan track C:\\lidar\\walk01_seg --avatar person_tt11_avatar.npz --out walk01_motion

    The input is a segment-motion folder, or a capture directory (it is then
    segmented first into <out>_segments). Writes <out>.npz (the motion: times,
    root, joint rotations, joints, foot contacts, observed fraction per body
    part, quality per frame) and <out>.json (summary). Needs PyTorch;
    uses the GPU when there is one."""
    name = "track"
    help = "fit the avatar to every frame of a moving person"

    @property
    def config_class(self):
        from bodyscan.dynamic.track import TrackPipelineConfig
        return TrackPipelineConfig

    def add_arguments(self, parser):
        parser.add_argument("input", help="segment-motion folder, or capture directory")
        parser.add_argument("--avatar", required=True, help="avatar .npz (bodyscan avatar)")
        parser.add_argument("--out", required=True, help="output base name (.npz)")
        parser.add_argument("--model", default=None, help="model file, if it moved since the avatar was fitted")
        parser.add_argument("--start", type=int, default=0, help="first segmented frame")
        parser.add_argument("--stop", type=int, default=None, help="last segmented frame (excluded)")
        self.add_config_arguments(parser)

    def run(self, args):
        config = self.prepare(args)
        if config is None:
            return 0
        from bodyscan.dynamic.track import run_tracking
        folder = Path(args.input)
        if (folder / "lut.npz").exists():
            from bodyscan.dynamic.segment import MotionSegmenter
            from bodyscan.io import NpzRecording
            segments = Path(str(Path(args.out).with_suffix("")) + "_segments")
            info(f"segmenting {folder} into {segments}")
            MotionSegmenter(NpzRecording(folder)).run(segments)
            folder = segments
        run_tracking(folder, args.avatar, args.out, config, args.model, args.start, args.stop)
        return 0


class ExportMotion(ConfiguredCommand):
    """Animated meshes of a tracked motion for the radio simulation, and body tracks.

        python -m bodyscan export-motion walk01_motion.npz --avatar person_tt11_avatar.npz
            --out walk01_meshes --rate 200

    Writes frames/mesh_XXXXXX.ply (one mesh per time step), velocity_XXXXXX.npy
    (per-vertex velocity [m/s]), sequence.npz (times, faces, part labels,
    joints, per-part velocities), joints.csv (body tracks) and export.json.
    Coordinates: metres, the floor frame of the recording (z up)."""
    name = "export-motion"
    help = "animated meshes and body tracks of a tracked motion"
    config_class = ExportCommandConfig

    def add_arguments(self, parser):
        parser.add_argument("motion", help="motion .npz (bodyscan track)")
        parser.add_argument("--avatar", required=True, help="avatar .npz used for the tracking")
        parser.add_argument("--out", required=True, help="output folder")
        parser.add_argument("--model", default=None, help="model file, if it moved since the avatar was fitted")
        self.add_config_arguments(parser)

    def run(self, args):
        config = self.prepare(args)
        if config is None:
            return 0
        from bodyscan.dynamic.export import export_motion
        export_motion(args.motion, args.avatar, args.out, config.export, args.model)
        return 0


class ReviewMotion(Command):
    """Pictures of a tracked motion over the LiDAR points, for human review.

        python -m bodyscan review-motion walk01_motion.npz --segments C:\\lidar\\walk01_seg
            --avatar person_tt11_avatar.npz --out walk01_review

    Writes review.gif (every --every-th frame) and review_XXXXX.png for the
    frames flagged by the tracker (and for --frames)."""
    name = "review-motion"
    help = "pictures of a tracked motion for review"

    def add_arguments(self, parser):
        parser.add_argument("motion", help="motion .npz")
        parser.add_argument("--segments", required=True, help="segment-motion folder of the recording")
        parser.add_argument("--avatar", required=True, help="avatar .npz")
        parser.add_argument("--out", required=True, help="output folder")
        parser.add_argument("--every", type=int, default=2, help="frame step of the GIF")
        parser.add_argument("--frames", type=int, nargs="*", default=None, help="tracked frames to draw as PNG")
        parser.add_argument("--model", default=None)
        parser.add_argument("--quiet", action="store_true")

    def run(self, args):
        set_quiet(args.quiet)
        from bodyscan.dynamic.review import review
        review(args.motion, args.segments, args.avatar, args.out, args.every, args.frames, model_path=args.model)
        return 0


# ----------------------------------------------------------------------------
# Synthetic test bench
# ----------------------------------------------------------------------------

class SimulateMotion(ConfiguredCommand):
    """Synthetic recording of a body moving near a simulated LiDAR, with the truth.

        python -m bodyscan simulate-motion C:\\lidar\\sim_walk --model SMPLX_NEUTRAL.npz --motion walk
            --distance 2.5 --mode 1024x20

    Writes a capture directory (readable by every command) plus truth.npz,
    truth_avatar.npz (the exact body), labels.npz, scenario.json and scan.ply
    (a simulated turntable scan of the body, the input of 'avatar'). Motions:
    walk, walk-circle, jump, jump-forward, stand, or an AMASS .npz. Needs
    PyTorch."""
    name = "simulate-motion"
    help = "synthetic recording of a moving body (with truth)"

    @property
    def config_class(self):
        from bodyscan.dynamic.bench import SimulateMotionConfig
        return SimulateMotionConfig

    def add_arguments(self, parser):
        parser.add_argument("out", help="output directory (new)")
        parser.add_argument("--model", required=True, help="SMPL-X model .npz (or the test body)")
        self.add_config_arguments(parser)

    def run(self, args):
        config = self.prepare(args)
        if config is None:
            return 0
        from bodyscan.dynamic.bench import make_take
        make_take(args.out, args.model, config)
        return 0


class EvaluateMotion(Command):
    """Accuracy of a tracked motion against the truth of a synthetic recording.

        python -m bodyscan evaluate-motion sim_walk_motion.npz --truth C:\\lidar\\sim_walk
            --avatar person_avatar.npz"""
    name = "evaluate-motion"
    help = "compare a tracked motion with the truth of a synthetic recording"

    def add_arguments(self, parser):
        parser.add_argument("motion", help="motion .npz")
        parser.add_argument("--truth", required=True, help="synthetic run directory (truth.npz, truth_avatar.npz)")
        parser.add_argument("--avatar", required=True, help="avatar .npz used for the tracking")
        parser.add_argument("--model", default=None)
        parser.add_argument("--frequency-ghz", type=float, default=60.0, help="for the Doppler equivalent")
        parser.add_argument("--json", default=None, help="also write the result here")

    def run(self, args):
        import torch
        from bodyscan.dynamic import evaluate
        from bodyscan.dynamic.avatar import Avatar
        estimate = Avatar.load(args.avatar)
        model = estimate.model(args.model, "cpu", torch.float64)
        truth = Avatar.load(Path(args.truth) / "truth_avatar.npz")
        result = evaluate.evaluate(args.motion, args.truth, model, estimate.shaped(model, 0), truth.shaped(model, 0),
                                   args.frequency_ghz)
        info(evaluate.report_text(result))
        if args.json:
            evaluate.save(result, args.json)
        return 0


class BenchMotion(ConfiguredCommand):
    """Synthetic test bench: simulated takes over a grid, tracked and compared with the truth.

        python -m bodyscan bench-motion C:\\lidar\\bench --model SMPLX_NEUTRAL.npz
            --motions walk jump --modes 1024x20 2048x10 --distances 2 3.5

    For every take: simulate-motion, segment-motion, track, evaluate-motion.
    Writes bench.md (table) and bench.json; the takes stay in sub-folders
    (a second run reuses them). Needs PyTorch; a take of 4 s takes a few
    minutes on a CPU."""
    name = "bench-motion"
    help = "synthetic test bench: error tables over motions, sensor modes, distances"

    @property
    def config_class(self):
        from bodyscan.dynamic.bench import BenchConfig
        from bodyscan.config import section as make_section

        @dataclass
        class BenchCommandConfig:
            bench: BenchConfig = make_section(BenchConfig)

        return BenchCommandConfig

    def add_arguments(self, parser):
        parser.add_argument("out", help="output folder")
        parser.add_argument("--model", required=True, help="SMPL-X model .npz (or the test body)")
        self.add_config_arguments(parser)

    def run(self, args):
        config = self.prepare(args)
        if config is None:
            return 0
        from bodyscan.dynamic.bench import run_bench
        run_bench(args.out, args.model, config.bench)
        return 0


class MakeTestBody(Command):
    """Write the procedural test body (SMPL-X file format) for tests and demonstrations.

        python -m bodyscan make-test-body testbody_smplx.npz

    A 1.76 m body built from capsules, with the 55 SMPL-X joints, skinning
    weights, 10 shape directions and relaxed hands. Not a person: use the
    real SMPL-X model for real data."""
    name = "make-test-body"
    help = "write the procedural test body (SMPL-X format)"

    def add_arguments(self, parser):
        parser.add_argument("out", help="output .npz")
        parser.add_argument("--quiet", action="store_true")

    def run(self, args):
        set_quiet(args.quiet)
        from bodyscan.body.testbody import build
        info(f"wrote {build(args.out)}")
        return 0
