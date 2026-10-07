"""Synthetic recordings of a moving person, with ground truth (the test bench).

A body (SMPL-X model, or an avatar) plays a motion (PoseSequence) in a room;
a simulated spinning LiDAR scans it frame by frame and the result is written
as a run directory of the capture commands (lut.npz, background/, frames/,
capture.json), so every command reads it like a real recording, plus
truth.npz (the motion, the body, the sensor pose) and labels.npz (which
pixels hit the person).

What the simulation reproduces:

    rolling shutter   the sensor turns during a frame: column c of frame k is
                      measured at t_k + c / (W f), and the body is posed at the
                      time of every block of block_columns columns
    beam footprint    footprint_rays sub-rays spread over the beam divergence;
                      the return is the nearest surface that fills at least a
                      quarter of the footprint, and at an edge between the
                      person and the background a mixed (veil) range with
                      probability mixed_fraction
    range noise       Gaussian, noise metres
    dropouts          on the body: body_dropout plus grazing_dropout x
                      (1 - |cos(incidence)|)^2 (dark fabric at grazing angles
                      returns little light)

Coordinates: the motion and the room are in the floor frame of the
simulation (z up, z = 0 on the floor). The sensor stands at 'position'
(x, y) and 'height', turned by yaw_deg about z. The processing estimates
the floor frame from the empty-scene frames (origin below the sensor, axes of
the sensor); truth.npz holds the sensor pose, and to_sensor_floor_frame()
maps truth into that frame for the evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d

from bodyscan.dynamic.motions import PoseSequence
from bodyscan.log import Progress, info

SENSOR_CLOCK_ORIGIN = 1000.0        # sensor time of t = 0 [s]


@dataclass
class MotionSensor:
    """A spinning multi-beam LiDAR (Ouster OS0-128 like), in the simulation floor frame."""
    rows: int = 128
    columns: int = 1024
    frame_rate: float = 20.0
    vertical_fov_deg: float = 90.0
    position: tuple = (0.0, 0.0)
    height: float = 1.0
    yaw_deg: float = 0.0
    noise: float = 0.008
    body_dropout: float = 0.02
    grazing_dropout: float = 0.3
    footprint_rays: int = 4
    divergence_deg: float = 0.35
    mixed_fraction: float = 0.3
    block_columns: int = 8
    max_range: float = 60.0
    seed: int = 0

    def __post_init__(self):
        half = self.vertical_fov_deg / 2.0
        elevation = np.radians(np.linspace(half, -half, self.rows))
        azimuth = np.radians(np.linspace(0.0, 360.0, self.columns, endpoint=False))
        el, az = np.meshgrid(elevation, azimuth, indexing="ij")
        self.directions = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)
        yaw = np.radians(self.yaw_deg)
        self.world_rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0],
                                        [0.0, 0.0, 1.0]])
        self.origin = np.array([self.position[0], self.position[1], self.height], dtype=np.float64)
        self.rng = np.random.default_rng(self.seed)

    @property
    def mode(self) -> str:
        return f"{self.columns}x{int(round(self.frame_rate))}"

    def world_directions(self, columns=None) -> np.ndarray:
        d = self.directions if columns is None else self.directions[:, columns]
        return d @ self.world_rotation.T

    def column_time(self, column) -> np.ndarray:
        return np.asarray(column, dtype=np.float64) / (self.columns * self.frame_rate)

    def column_of_azimuth(self, azimuth_world: np.ndarray) -> np.ndarray:
        """Column index (float) looking towards a world azimuth."""
        local = np.mod(azimuth_world - np.radians(self.yaw_deg), 2 * np.pi)
        return local / (2 * np.pi) * self.columns


def _cast(scene, origin, directions):
    rays = np.concatenate([np.broadcast_to(origin, directions.shape), directions], axis=-1).reshape(-1, 6)
    answer = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
    distance = answer["t_hit"].numpy().reshape(directions.shape[:-1])
    normals = answer["primitive_normals"].numpy().reshape(directions.shape)
    return distance.astype(np.float64), normals.astype(np.float64)


def _scene_of(meshes) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    for mesh in meshes:
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


@dataclass
class Room:
    """Floor, walls and furniture of the simulated scene (floor frame)."""
    size: tuple = (12.0, 9.0)
    center: tuple = (1.0, 0.0)
    height: float = 2.8
    furniture: list = field(default_factory=list)          # open3d meshes

    def meshes(self) -> list:
        from bodyscan.synthetic import room
        return [item.mesh for item in room(self.size, self.center, self.height)] + list(self.furniture)


def default_furniture() -> list:
    from bodyscan.synthetic import box, chair
    desk = box((1.4, 0.7, 0.75), (-2.8, 3.4))
    cabinet = box((0.5, 0.6, 1.9), (5.6, -3.6))
    seat = chair().translate((-3.5, -3.0, 0.0))
    return [desk, cabinet, seat]


class MotionScanner:
    """Scans a body playing a motion (see the module docstring)."""

    def __init__(self, shaped, motion: PoseSequence, sensor: MotionSensor, room: Room | None = None,
                 body_radius: float = 1.1):
        self.shaped, self.motion, self.sensor = shaped, motion, sensor
        self.room = room or Room()
        self.body_radius = body_radius
        self.faces = o3d.utility.Vector3iVector(shaped.faces_np.astype(np.int32))
        self.static_scene = _scene_of(self.room.meshes())
        self._static = None
        footprint = np.radians(sensor.divergence_deg) / 2.355               # FWHM -> sigma
        rays = max(int(sensor.footprint_rays), 1)
        self.jitter = (np.zeros((1, 2)) if rays == 1 else
                       sensor.rng.normal(0.0, footprint, (rays, 2)) * np.array([1.0, 1.0]))

    # -- geometry -----------------------------------------------------------------------------
    def _sub_directions(self, directions: np.ndarray) -> np.ndarray:
        """(..., 3) beam directions -> (..., S, 3) sub-rays of the footprint."""
        if len(self.jitter) == 1:
            return directions[..., None, :]
        up = np.array([0.0, 0.0, 1.0])
        side = np.cross(directions, up)
        side /= np.maximum(np.linalg.norm(side, axis=-1, keepdims=True), 1e-9)
        lift = np.cross(side, directions)
        rays = (directions[..., None, :] + self.jitter[:, 0, None] * side[..., None, :]
                + self.jitter[:, 1, None] * lift[..., None, :])
        return rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    def static_ranges(self):
        """Range of the empty room for every pixel and sub-ray, (H, W, S)."""
        if self._static is None:
            directions = self._sub_directions(self.sensor.world_directions())
            distance, _ = _cast(self.static_scene, self.sensor.origin, directions)
            self._static = distance
        return self._static

    def person_columns(self, t0: float, t1: float) -> np.ndarray:
        """Columns that may see the person between t0 and t1."""
        s = self.sensor
        inside = (self.motion.times >= t0 - 0.05) & (self.motion.times <= t1 + 0.05)
        positions = self.motion.root_position[inside] if inside.any() else self.motion.sample([t0]).root_position
        offset = positions[:, :2] - s.origin[:2]
        distance = np.linalg.norm(offset, axis=1)
        if np.any(distance <= self.body_radius * 1.05):
            return np.arange(s.columns)
        center = s.column_of_azimuth(np.arctan2(offset[:, 1], offset[:, 0]))
        half = np.degrees(np.arcsin(np.clip(self.body_radius / distance, 0.0, 1.0))) / 360.0 * s.columns + 2
        columns = set()
        for c, h in zip(center, half):
            low, high = int(np.floor(c - h)), int(np.ceil(c + h))
            columns.update(np.mod(np.arange(low, high + 1), s.columns).tolist())
        return np.array(sorted(columns), dtype=np.int64)

    # -- one frame ------------------------------------------------------------------------------
    def frame(self, t0: float):
        """Range image [m] (NaN: no return) and person mask of the frame starting at t0."""
        s = self.sensor
        static = self.static_ranges()
        best = static.copy()
        body = np.zeros(best.shape, dtype=bool)
        cos_incidence = np.ones(best.shape)
        columns = self.person_columns(t0, t0 + 1.0 / s.frame_rate)
        if len(columns):
            # blocks of neighbouring columns; a set that wraps around column 0 is two runs, measured at
            # the start and at the end of the frame
            runs = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
            blocks = [run[k:k + s.block_columns] for run in runs for k in range(0, len(run), s.block_columns)]
            block_times = np.array([t0 + float(np.mean(s.column_time(b))) for b in blocks])
            inside = (block_times >= self.motion.times[0]) & (block_times <= self.motion.times[-1])
            if inside.any():
                poses = self.motion.sample(block_times[inside])
                vertices, _ = poses.posed(self.shaped)
                for index, block in zip(np.flatnonzero(inside), [b for b, ok in zip(blocks, inside) if ok]):
                    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(
                        vertices[int(np.sum(inside[:index]))].astype(np.float64)), self.faces)
                    directions = self._sub_directions(s.world_directions(block))
                    distance, normals = _cast(_scene_of([mesh]), s.origin, directions)
                    nearer = distance < best[:, block]
                    best[:, block] = np.where(nearer, distance, best[:, block])
                    body[:, block] |= nearer
                    cos_incidence[:, block] = np.where(nearer, np.abs(np.einsum("...k,...k->...", normals,
                                                                                directions)), 1.0)
        return self._combine(best, body, cos_incidence)

    def _combine(self, sub_range, sub_body, cos_incidence):
        """Sub-rays to one return per pixel, then noise and dropouts."""
        s = self.sensor
        rng = s.rng
        h, w, n = sub_range.shape
        valid = np.isfinite(sub_range) & (sub_range < s.max_range)
        r = np.where(valid, sub_range, np.inf)
        order = np.argsort(r, axis=2)
        r_sorted = np.take_along_axis(r, order, axis=2)
        body_sorted = np.take_along_axis(sub_body, order, axis=2)
        cos_sorted = np.take_along_axis(cos_incidence, order, axis=2)
        nearest = r_sorted[..., 0]
        with np.errstate(invalid="ignore"):
            same = np.abs(r_sorted - nearest[..., None]) < 0.10
        fraction = same.sum(axis=2) / n
        # the nearest surface returns if it fills a quarter of the footprint, else the next one
        pick = np.where(fraction >= 0.25, 0, np.argmax(~same, axis=2))
        result = np.take_along_axis(r_sorted, pick[..., None], axis=2)[..., 0]
        is_body = np.take_along_axis(body_sorted, pick[..., None], axis=2)[..., 0]
        cosine = np.take_along_axis(cos_sorted, pick[..., None], axis=2)[..., 0]
        if n > 1 and s.mixed_fraction > 0:
            other = np.where(np.isfinite(r_sorted), r_sorted, -np.inf)
            far = np.max(other, axis=2)
            far = np.where(np.isfinite(far), far, np.nan)
            with np.errstate(invalid="ignore"):
                edge = (fraction >= 0.25) & (fraction <= 0.75) & np.isfinite(far) & (far - nearest > 0.10)
            mixed = edge & (rng.random((h, w)) < s.mixed_fraction)
            weight = rng.random((h, w))
            result = np.where(mixed, weight * nearest + (1 - weight) * far, result)
        returned = np.isfinite(result)
        drop = s.body_dropout + s.grazing_dropout * (1.0 - np.clip(cosine, 0.0, 1.0)) ** 2
        returned &= ~(is_body & (rng.random((h, w)) < drop))
        result = np.where(returned, result + rng.normal(0.0, s.noise, (h, w)), np.nan)
        return result, is_body & returned


# ----------------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------------

def to_sensor_floor_frame(points: np.ndarray, sensor_pose: dict) -> np.ndarray:
    """Simulation floor frame -> floor frame of the processing (origin below the
    sensor, x along the sensor's x axis)."""
    yaw = np.radians(sensor_pose["yaw_deg"])
    rotation = np.array([[np.cos(yaw), np.sin(yaw), 0.0], [-np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    offset = np.array([sensor_pose["position"][0], sensor_pose["position"][1], 0.0])
    return (np.asarray(points) - offset) @ rotation.T


def record(out, shaped, motion: PoseSequence, sensor: MotionSensor | None = None, room: Room | None = None,
           background_frames: int = 10, start: float | None = None, stop: float | None = None,
           model_path=None, betas=None, normal_displacement=None, level: int = 0, quiet: bool = False) -> Path:
    """Scan 'motion' and write a run directory with truth.npz and labels.npz."""
    sensor = sensor or MotionSensor()
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    (out / "background").mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    scanner = MotionScanner(shaped, motion, sensor, room)
    np.savez(out / "lut.npz", direction=sensor.directions.astype(np.float32),
             offset=np.zeros_like(sensor.directions, dtype=np.float32),
             pixel_shift=np.zeros(sensor.rows, dtype=np.int64))
    period = 1.0 / sensor.frame_rate
    start = motion.times[0] if start is None else start
    stop = motion.times[-1] if stop is None else stop
    count = int(np.floor((stop - start) / period))
    columns = np.arange(sensor.columns)

    def save(path, range_m, index, t0):
        range_mm = np.where(np.isfinite(range_m), np.round(np.clip(range_m, 0.0, 65.0) * 1000.0), 0)
        stamps = np.round((SENSOR_CLOCK_ORIGIN + t0 + sensor.column_time(columns)) * 1e9).astype(np.uint64)
        np.savez(path, range=range_mm.astype(np.uint16), frame_id=np.int64(index), time=np.float64(t0),
                 timestamps=stamps, phase=np.int64(0), columns_ok=np.float64(1.0))

    static = scanner.static_ranges()
    for k in range(background_frames):
        empty, _ = scanner._combine(static, np.zeros(static.shape, dtype=bool), np.ones(static.shape))
        save(out / "background" / f"bg_{k:05d}.npz", empty, k, start - period * (background_frames - k) - 1.0)
    labels = []
    progress = Progress("simulated frames", count) if not quiet else None
    for k in range(count):
        t0 = start + k * period
        range_m, person = scanner.frame(t0)
        save(out / "frames" / f"frame_{k:05d}.npz", range_m, k, t0)
        labels.append(np.packbits(person, axis=1))
        if progress is not None:
            progress.maybe(k + 1, 50)
    np.savez_compressed(out / "labels.npz", person=np.array(labels), width=np.int64(sensor.columns))
    sensor_pose = {"position": list(sensor.position), "height": sensor.height, "yaw_deg": sensor.yaw_deg}
    motion.shifted(SENSOR_CLOCK_ORIGIN).save(
        out / "truth.npz", model=str(model_path or ""), model_sha1=getattr(shaped.model, "sha1", ""),
        betas=np.zeros(0) if betas is None else np.asarray(betas, dtype=np.float64),
        normal_displacement=np.zeros(0) if normal_displacement is None else np.asarray(normal_displacement),
        level=np.int64(level), sensor_pose=json.dumps(sensor_pose), clock_origin=SENSOR_CLOCK_ORIGIN)
    capture = {"version": "synthetic-motion", "mode": sensor.mode, "frame_rate": sensor.frame_rate,
               "frames": count, "background": background_frames,
               "sensor": {"rows": sensor.rows, "columns": sensor.columns, "vertical_fov_deg": sensor.vertical_fov_deg,
                          "noise_m": sensor.noise, "body_dropout": sensor.body_dropout,
                          "grazing_dropout": sensor.grazing_dropout, "footprint_rays": sensor.footprint_rays,
                          "mixed_fraction": sensor.mixed_fraction, **sensor_pose}}
    (out / "capture.json").write_text(json.dumps(capture, indent=2))
    if not quiet:
        info(f"wrote {count} frames ({sensor.mode}) and {background_frames} empty-scene frames to {out}")
    return out


def load_truth(run_dir):
    """(PoseSequence on the sensor clock, metadata dict) of a synthetic recording."""
    path = Path(run_dir) / "truth.npz"
    data = np.load(path, allow_pickle=False)
    meta = {"model": str(data["model"]), "model_sha1": str(data["model_sha1"]), "betas": data["betas"],
            "normal_displacement": data["normal_displacement"], "level": int(data["level"]),
            "sensor_pose": json.loads(str(data["sensor_pose"]))}
    return PoseSequence.load(path), meta


def load_labels(run_dir) -> np.ndarray:
    """(frames, H, W) boolean: pixels that hit the person."""
    data = np.load(Path(run_dir) / "labels.npz")
    return np.unpackbits(data["person"], axis=2, count=int(data["width"])).astype(bool)


def static_scan(shaped, body_pose, root_rotation, root_position, count: int = 200000, noise: float = 0.002,
                hole_top: float = 0.02, hole_soles: float = 0.015, seed: int = 0):
    """Points and normals of a body standing still, as a turntable fusion would
    give them: uniform on the surface, noise along the normal, and no points on
    the soles nor on the top of the head (the sensor does not see them)."""
    import torch
    from bodyscan.body.rotations import axis_angle_to_matrix_np
    model = shaped.model
    with torch.no_grad():
        posed = shaped.pose(model.rotations(body=np.asarray(body_pose)[None]),
                            model.tensor(axis_angle_to_matrix_np(np.asarray(root_rotation))[None]),
                            model.tensor(np.asarray(root_position)[None]))
    from bodyscan.dynamic.fitting import sample_surface
    vertices = posed.vertices[0].cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(seed)
    points, normals = sample_surface(vertices, shaped.faces_np, count, rng)
    points = points + rng.normal(0.0, noise, (len(points), 1)) * normals
    top = points[:, 2].max()
    keep = (points[:, 2] > hole_soles) & (points[:, 2] < top - hole_top)
    return points[keep], normals[keep]
