"""The synthetic test bench: simulated takes with truth, and error tables.

A take: a body (SMPL-X or the test body, with chosen shape coefficients)
performs a motion near the simulated LiDAR in a furnished room; the
recording is written like a real capture (bodyscan.dynamic.simulate) with
truth.npz and truth_avatar.npz (the exact body). Scenarios, with the LiDAR
at the origin and the motion centred in the direction 'azimuth' (90 deg: the
sensor's +y axis; at 0 deg, its +x axis, every frame starts and ends on the
person, the hardest case for the rolling shutter):

    walk          a straight pass across the line of sight at 'distance' metres
    walk-circle   a circle of radius 'distance' around the LiDAR
    jump          jumps in place at 'distance' metres, facing the LiDAR
    jump-forward  forward jumps across the line of sight at 'distance' metres
    <file>.npz    an AMASS or PoseSequence file, placed at 'distance' metres

The bench runs takes over a grid (motions x sensor modes x distances),
segments and tracks each one, and compares with the truth
(bodyscan.dynamic.evaluate). With avatar_from_scan the avatar is fitted to a
simulated turntable scan of the body (as with a real person) instead of
being the exact body, so the table includes the avatar error.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bodyscan.config import param, section
from bodyscan.log import info, set_quiet

MODES = ("512x10", "512x20", "1024x10", "1024x20", "2048x10")


@dataclass
class MotionScenarioConfig:
    """The simulated take."""
    motion: str = param("walk", "walk, walk-circle, jump, jump-forward, stand, or an AMASS / PoseSequence .npz")
    distance: float = param(2.5, "distance of the motion from the LiDAR (walk: of the straight pass; circle: "
                                 "radius)", unit="m")
    azimuth: float = param(90.0, "direction of the motion seen from the LiDAR (0: the sensor's x axis, where "
                                 "the synthetic frames start and end)", unit="deg")
    duration: float = param(5.0, "walking time (walks)", unit="s")
    speed: float = param(1.15, "walking speed", unit="m/s")
    jumps: int = param(2, "number of jumps")
    jump_height: float = param(0.25, "rise of the pelvis above its take-off height in a jump", unit="m")
    betas: list[float] = param([], "shape coefficients of the simulated person (empty: the average body)")
    seed: int = param(0, "random seed of the sensor noise")
    scan: bool = param(True, "also write scan.ply: a simulated turntable scan of the body (input of 'avatar')")


@dataclass
class SimulatedSensorConfig:
    """The simulated LiDAR (an Ouster OS0-128 by default)."""
    mode: str = param("1024x20", "columns x frames per second", choices=MODES)
    height: float = param(1.0, "lens height above the floor", unit="m")
    noise: float = param(0.008, "range noise (one sigma)", unit="m")
    body_dropout: float = param(0.02, "probability that a beam on the body returns nothing")
    grazing_dropout: float = param(0.3, "additional dropout on the body at grazing incidence (x (1 - cos)^2)")
    footprint_rays: int = param(4, "sub-rays per beam (1: no beam footprint, no mixed pixels)")
    mixed_fraction: float = param(0.3, "probability of a mixed range at an edge between person and background")
    background_frames: int = param(10, "frames of the empty room")


@dataclass
class SimulateMotionConfig:
    scenario: MotionScenarioConfig = section(MotionScenarioConfig)
    sensor: SimulatedSensorConfig = section(SimulatedSensorConfig)


def build_motion(shaped, scenario: MotionScenarioConfig):
    """The PoseSequence of a scenario (LiDAR at the origin)."""
    from bodyscan.dynamic import motions
    d = scenario.distance
    name = scenario.motion
    # built along the +x axis, then turned to the azimuth
    if name == "walk":
        length = scenario.speed * scenario.duration
        sequence = motions.walk(shaped, duration=scenario.duration, speed=scenario.speed, heading_deg=90.0,
                                start_xy=(d, -0.5 * length))
    elif name == "walk-circle":
        sequence = motions.walk(shaped, duration=scenario.duration, speed=scenario.speed, heading_deg=90.0,
                                turn_rate_deg=float(np.degrees(scenario.speed / max(d, 0.5))), start_xy=(d, 0.0))
    elif name == "jump":
        sequence = motions.jump(shaped, count=scenario.jumps, height=scenario.jump_height, heading_deg=180.0,
                                start_xy=(d, 0.0))
    elif name == "jump-forward":
        sequence = motions.jump(shaped, count=scenario.jumps, height=scenario.jump_height, forward=0.5,
                                heading_deg=90.0, start_xy=(d, -0.25 * scenario.jumps))
    elif name == "stand":
        sequence = motions.standing(shaped, duration=scenario.duration, xy=(d, 0.0), heading_deg=180.0)
    elif Path(name).suffix == ".npz" and Path(name).exists():
        sequence = motions.load_motion(name, shaped)
        sequence.root_position[:, :2] += np.array([d, 0.0]) - sequence.root_position[0, :2]
    else:
        raise SystemExit(f"unknown motion {name!r}")
    return sequence.turned(np.radians(scenario.azimuth))


def make_take(out, model_path, config: SimulateMotionConfig, quiet: bool = False) -> Path:
    """Simulate one take into the run directory 'out' (plus truth_avatar.npz and, if asked, scan.ply)."""
    import torch
    from bodyscan.body.model import BodyModel
    from bodyscan.dynamic.avatar import Avatar
    from bodyscan.dynamic.simulate import MotionSensor, Room, default_furniture, record
    s, c = config.scenario, config.sensor
    betas = np.asarray(s.betas, dtype=np.float64) if s.betas else None
    model = BodyModel(model_path, num_betas=max(len(s.betas), 10), device="cpu", dtype=torch.float64)
    shaped = model.shaped(betas)
    motion = build_motion(shaped, s)
    columns, rate = (int(v) for v in c.mode.split("x"))
    sensor = MotionSensor(columns=columns, frame_rate=float(rate), height=c.height, noise=c.noise,
                          body_dropout=c.body_dropout, grazing_dropout=c.grazing_dropout,
                          footprint_rays=c.footprint_rays, mixed_fraction=c.mixed_fraction, seed=s.seed)
    extent = np.abs(motion.root_position[:, :2]).max() + 2.5
    furniture = [item for item in default_furniture() if _clearance(item, motion.root_position[:, :2]) > 1.0]
    room = Room(size=(max(12.0, 2 * extent + 2.0), max(9.0, 2 * extent)), center=(1.0, 0.0), furniture=furniture)
    out = Path(out)
    record(out, shaped, motion, sensor, room, c.background_frames, model_path=model_path, betas=betas, quiet=quiet)
    Avatar.from_model(model, betas).save(out / "truth_avatar.npz")
    if s.scan:
        write_scan(out / "scan.ply", *turntable_scan(shaped))
    (out / "scenario.json").write_text(json.dumps({"scenario": asdict(s), "sensor": asdict(c)}, indent=2))
    return out


def _clearance(mesh, path_xy: np.ndarray) -> float:
    """Smallest horizontal distance [m] between a piece of furniture and the pelvis path."""
    box = mesh.get_axis_aligned_bounding_box()
    low, high = np.asarray(box.min_bound)[:2], np.asarray(box.max_bound)[:2]
    outside = np.maximum(np.maximum(low - path_xy, path_xy - high), 0.0)
    return float(np.linalg.norm(outside, axis=1).min())


def turntable_scan(shaped, yaw_deg: float = 20.0):
    """Points and normals of the body standing in the A-pose on the floor, as a
    turntable fusion gives them (bodyscan.dynamic.simulate.static_scan)."""
    from bodyscan.dynamic import motions
    from bodyscan.dynamic.simulate import static_scan
    pose = motions.angles_to_body(motions.a_pose(1))[0]
    root = np.array([0.0, 0.0, np.radians(yaw_deg)])
    sequence = motions.PoseSequence(np.zeros(1), root[None], np.zeros((1, 3)), pose[None])
    soles, _ = sequence.posed(shaped, subset=motions.foot_vertices(shaped))
    position = np.array([0.0, 0.0, -float(soles[0, :, 2].min())])
    return static_scan(shaped, pose, root, position)


def write_scan(path, points: np.ndarray, normals: np.ndarray) -> None:
    import open3d as o3d
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(str(path), cloud)


def fitted_avatar(model_path, betas, out_path, level: int = 1) -> Path:
    """An avatar fitted to a simulated turntable scan of the body (as for a real person)."""
    import torch
    from bodyscan.body.model import BodyModel
    from bodyscan.dynamic.avatar import AvatarConfig, AvatarFitter
    model = BodyModel(model_path, num_betas=max(len(betas), 10), device="cpu", dtype=torch.float64)
    shaped = model.shaped(np.asarray(betas) if len(betas) else None)
    points, normals = turntable_scan(shaped)
    avatar = AvatarFitter(model, AvatarConfig(level=level)).fit(points, normals)
    avatar.save(out_path)
    return Path(out_path)


@dataclass
class BenchConfig:
    """The grid of the test bench."""
    motions: list[str] = param(["walk", "jump"], "motions (see simulate-motion)")
    modes: list[str] = param(["1024x20", "2048x10"], "sensor modes")
    distances: list[float] = param([2.0, 3.5], "distances from the LiDAR", unit="m")
    duration: float = param(4.0, "walking time of the walks", unit="s")
    jumps: int = param(2, "jumps per take")
    avatar_from_scan: bool = param(False, "fit the avatar to a simulated turntable scan (adds the avatar error)")
    refine: bool = param(True, "refine the whole sequence (stage 2) when tracking")
    reuse: bool = param(False, "keep the result of takes already tracked and evaluated (to resume a bench that "
                               "stopped; the simulated recordings are always reused)")


def run_bench(out, model_path, bench: BenchConfig, quiet: bool = True) -> list[dict]:
    """Every take of the grid: simulate, segment, track, evaluate; writes bench.json and bench.md."""
    import torch
    from bodyscan.body.model import BodyModel
    from bodyscan.dynamic import evaluate
    from bodyscan.dynamic.avatar import Avatar
    from bodyscan.dynamic.segment import MotionSegmenter
    from bodyscan.dynamic.track import TrackPipelineConfig, run_tracking
    from bodyscan.io import NpzRecording
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    avatar_path = None
    if bench.avatar_from_scan:
        avatar_path = out / "fitted_avatar.npz"
        if not avatar_path.exists():
            info("fitting the avatar to a simulated turntable scan")
            fitted_avatar(model_path, [], avatar_path)
    for motion in bench.motions:
        for mode in bench.modes:
            for distance in bench.distances:
                name = f"{Path(motion).stem}_{mode}_{distance:g}m"
                take = out / name
                info(f"take {name}")
                config = SimulateMotionConfig()
                config.scenario.motion, config.scenario.distance = motion, distance
                config.scenario.duration, config.scenario.jumps = bench.duration, bench.jumps
                config.sensor.mode = mode
                set_quiet(quiet)
                try:
                    if not (take / "truth.npz").exists():
                        if take.exists():                      # a take interrupted while it was simulated
                            shutil.rmtree(take)
                        make_take(take, model_path, config, quiet=True)
                    segments = take / "segments"
                    if not (segments / "segments.json").exists():
                        MotionSegmenter(NpzRecording(take)).run(segments)
                    tracking = TrackPipelineConfig()
                    tracking.refine.enabled = bench.refine
                    use_avatar = avatar_path or (take / "truth_avatar.npz")
                    motion_file = take / "motion.npz"
                    done = take / "evaluation.json"
                    if bench.reuse and motion_file.exists() and done.exists():
                        result = json.loads(done.read_text())
                    else:
                        run_tracking(segments, use_avatar, motion_file, tracking)
                        model = BodyModel(model_path, device="cpu", dtype=torch.float64)
                        truth_shaped = Avatar.load(take / "truth_avatar.npz").shaped(model, 0)
                        estimate_shaped = Avatar.load(use_avatar).shaped(model, 0)
                        result = evaluate.evaluate(motion_file, take, model, estimate_shaped, truth_shaped)
                        evaluate.save(result, done)
                finally:
                    set_quiet(False)
                row = {"take": name, "motion": motion, "mode": mode, "distance_m": distance, **_row(result)}
                rows.append(row)
                info(f"  MPJPE {row['mpjpe_mm']:.1f} mm, hidden joints {row['hidden_joint_mm']}, velocity "
                     f"{row['velocity_rms_m_s']:.3f} m/s")
    (out / "bench.json").write_text(json.dumps(rows, indent=2))
    (out / "bench.md").write_text(table(rows))
    info(table(rows))
    return rows


def _row(result: dict) -> dict:
    hidden = result.get("hidden_joint_mm")
    return {"mpjpe_mm": result["mpjpe_mm"]["mean"], "mpjpe_p90_mm": result["mpjpe_mm"]["p90"],
            "seen_joint_mm": result.get("seen_joint_mm"), "hidden_joint_mm": None if hidden is None else round(hidden, 1),
            "hidden_fraction": result.get("hidden_fraction"), "vertex_mm": result.get("vertex_error_mm"),
            "velocity_rms_m_s": result.get("velocity_error_rms_m_s", float("nan")),
            "acceleration_error_m_s2": result.get("acceleration_error_m_s2", float("nan")),
            "skating_m_s": result.get("foot_skating_m_s", {}).get("median"), "yaw_deg": result.get("yaw_deg")}


def table(rows: list[dict]) -> str:
    lines = ["| take | joints (mean) | joints p90 | seen | hidden | vertices | part velocity | acceleration error | "
             "standing feet |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        def mm(value):
            return "n/a" if value is None else f"{value:.0f} mm"
        lines.append(f"| {r['take']} | {mm(r['mpjpe_mm'])} | {mm(r['mpjpe_p90_mm'])} | {mm(r['seen_joint_mm'])} | "
                     f"{mm(r['hidden_joint_mm'])} | {mm(r['vertex_mm'])} | {r['velocity_rms_m_s']:.2f} m/s | "
                     f"{r['acceleration_error_m_s2']:.1f} m/s2 | "
                     + ("n/a" if r["skating_m_s"] is None else f"{r['skating_m_s']:.3f} m/s") + " |")
    return "\n".join(lines) + "\n"


