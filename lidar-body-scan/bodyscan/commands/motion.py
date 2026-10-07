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
    """Empty room, countdown, recording."""
    background_seconds: float = param(10.0, "empty-room recording at the start (nobody in the room)", unit="s")
    delay: float = param(10.0, "countdown to walk to the floor mark and take the A-pose", unit="s")
    duration: float | None = param(None, "record this long; not set: until ENTER (or the safety stop)", unit="s")
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

        empty room (--background-seconds, default 10 s: nobody in the room) ->
        countdown (--delay: walk to the floor mark, take the A-pose) ->
        recording (--duration seconds, or until ENTER).

    Every take should start and end with 3 s standing still in the A-pose on
    the floor mark (the tracker starts from it). Use the sensor in 1024x20
    mode (more frames per second; the vertical spacing limits the detail
    anyway): set it in the sensor's web page or with the Ouster SDK."""
    name = "capture-motion"
    help = "record a person moving around the LiDAR (Ouster sensor)"
    config_class = MotionCaptureConfig

    def phases(self, config, devices):
        from bodyscan.capture.session import Background, Countdown, Record
        t = config.timing
        return [Background(t.background_seconds, f"{t.background_seconds:g} s: empty room, stay out of the "
                                                 "sensor's view"),
                Countdown(t.delay, f"{t.delay:g} s: walk to the floor mark and take the A-pose"),
                Record(t.duration, t.max_seconds,
                       (f"{t.duration:g} s: " if t.duration else "until ENTER: ") +
                       "3 s A-pose, the motion, 3 s A-pose")]

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

        python -m bodyscan avatar person_tt11_mesh.ply --model C:\\models\\smplx\\SMPLX_NEUTRAL.npz
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
