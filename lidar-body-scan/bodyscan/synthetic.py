"""Synthetic recordings: try the commands without the sensor, and test them.

A Scene is a list of items placed in the floor frame (z up, z = 0 on the
floor). An item is a mesh (body, chair, box, platform) that may turn about a
vertical axis with an angle given as a function of time. A SyntheticSensor
(uniform beam fan like an Ouster OS0, at a height, optionally tilted) ray
casts the scene with Open3D, adds range noise, and writes the run directory
of the capture commands:

    lut.npz, capture.json, background/ (empty scene), frames/, truth.json

Simplifications: every frame is rendered at one instant (no rolling
shutter: all the columns of a frame carry the same timestamp), no mixed
pixels, no reflectivity, Gaussian range noise.

Scenarios (python -m bodyscan simulate NAME OUT):

    turntable   a person standing on a rotating platform, a desk and a cabinet
    inplace     a person turning on the spot by steps, holding still in between
    detect      a person and a chair on two rotating platforms, a person standing
                still, a desk, a cabinet and a stand: for bodyscan detect
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d

from bodyscan.log import Progress, info


# ----------------------------------------------------------------------------
# Shapes
# ----------------------------------------------------------------------------

def box(size, center_xy, z0=0.0) -> o3d.geometry.TriangleMesh:
    """Box of size (sx, sy, sz) standing on height z0, centred on center_xy."""
    sx, sy, sz = size
    mesh = o3d.geometry.TriangleMesh.create_box(sx, sy, sz)
    mesh.translate((center_xy[0] - sx / 2, center_xy[1] - sy / 2, z0))
    return mesh


def ellipsoid(a, b, c, center) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_sphere(1.0, resolution=30)
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * np.array([a, b, c]) + center)
    return mesh


def limb(p0, p1, radius) -> o3d.geometry.TriangleMesh:
    """Cylinder from p0 to p1 with a rounded end at p1."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    length = np.linalg.norm(p1 - p0)
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius, length, resolution=24, split=4)
    direction = (p1 - p0) / length
    axis = np.cross([0.0, 0.0, 1.0], direction)
    s = np.linalg.norm(axis)
    if s > 1e-9:
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis / s * np.arctan2(s, direction[2])),
                    center=(0, 0, 0))
    mesh.translate((p0 + p1) / 2)
    return mesh + o3d.geometry.TriangleMesh.create_sphere(radius, resolution=16).translate(p1)


def merge(meshes) -> o3d.geometry.TriangleMesh:
    result = o3d.geometry.TriangleMesh()
    for mesh in meshes:
        result += mesh
    return result


def person(arm_deg: float = 25.0, breath: float = 1.0, height: float = 1.75) -> o3d.geometry.TriangleMesh:
    """Standing adult in an A-pose ('arm_deg' between the arms and the body),
    facing +x, feet on z = 0, vertical axis through the origin; built for a
    1.75 m stature and scaled to 'height'."""
    parts = [ellipsoid(0.11 * breath, 0.17, 0.28, (0, 0, 1.22)),          # torso
             ellipsoid(0.12, 0.17, 0.14, (0, 0, 0.95)),                   # pelvis
             ellipsoid(0.10, 0.085, 0.115, (0.01, 0, 1.635)),             # head
             ellipsoid(0.03, 0.02, 0.02, (0.10, 0, 1.635)),               # nose
             limb((0, 0, 1.45), (0, 0, 1.54), 0.05)]                      # neck
    for side in (1, -1):
        shoulder = np.array([0, side * 0.20, 1.45])
        angle = np.radians(arm_deg)
        parts.append(limb(shoulder, shoulder + 0.62 * np.array([0.0, side * np.sin(angle), -np.cos(angle)]), 0.045))
        hip, ankle = np.array([0, side * 0.09, 0.90]), np.array([0, side * 0.11, 0.08])
        parts.append(limb(hip, ankle, 0.07))
        parts.append(box((0.24, 0.09, 0.07), (ankle[0] + 0.05, ankle[1])))
    body = merge(parts)
    body.scale(height / 1.75, center=(0, 0, 0))
    return body


def chair() -> o3d.geometry.TriangleMesh:
    """Seat 0.45 m high, backrest to 0.95 m, four legs; centred on the origin."""
    parts = [box((0.45, 0.45, 0.04), (0, 0), 0.43), box((0.04, 0.45, 0.50), (-0.205, 0), 0.47)]
    for x in (-0.2, 0.2):
        for y in (-0.2, 0.2):
            parts.append(box((0.035, 0.035, 0.43), (x, y)))
    return merge(parts)


def stand() -> o3d.geometry.TriangleMesh:
    """Lamp or camera stand: a 1.6 m pole on three legs."""
    parts = [limb((0, 0, 0.3), (0, 0, 1.6), 0.015)]
    for angle in np.radians([0, 120, 240]):
        parts.append(limb((0, 0, 0.35), (0.35 * np.cos(angle), 0.35 * np.sin(angle), 0.01), 0.012))
    return merge(parts)


def platform(radius: float = 0.582, top: float = 0.03, rim: float = 0.03,
             segments: int = 96) -> o3d.geometry.TriangleMesh:
    """The turntable as the sensor sees it: the rim (a ring 'rim' wide, 'top' high);
    the dark foam top returns almost nothing and is left out."""
    parts = []
    arc = 2 * np.pi * (radius - rim / 2) / segments * 1.05
    for angle in np.linspace(0.0, 2 * np.pi, segments, endpoint=False):
        piece = o3d.geometry.TriangleMesh.create_box(rim, arc, top).translate((-rim / 2, -arc / 2, 0.0))
        piece.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, angle)), center=(0, 0, 0))
        piece.translate(((radius - rim / 2) * np.cos(angle), (radius - rim / 2) * np.sin(angle), 0.0))
        parts.append(piece)
    return merge(parts)


# ----------------------------------------------------------------------------
# Scene
# ----------------------------------------------------------------------------

def constant_turn(speed_deg_s: float, start_s: float = 0.0, turn_deg: float | None = None):
    """Angle [deg] of a platform that starts at start_s and turns at a constant speed, up to turn_deg."""
    def angle(t):
        value = speed_deg_s * max(t - start_s, 0.0)
        return value if turn_deg is None else float(np.clip(value, -abs(turn_deg), abs(turn_deg)))
    return angle


def stepping_turn(step_deg: float = 25.0, period_s: float = 4.0, step_s: float = 1.0):
    """Angle [deg] of a person turning by steps: step_deg in step_s, every period_s, still in between."""
    def angle(t):
        steps, phase = divmod(max(t, 0.0), period_s)
        return step_deg * (steps + min(phase / step_s, 1.0))
    return angle


@dataclass
class Item:
    """A mesh placed at 'position' (floor frame), turned by angle(t) [deg] about its own vertical axis.
    'present' tells when it is in the scene (people are not there during the empty-scene frames)."""
    name: str
    mesh: o3d.geometry.TriangleMesh
    position: tuple = (0.0, 0.0)
    angle: Callable[[float], float] = field(default=lambda t: 0.0)
    height: float = 0.0
    background: bool = True                   # present in the empty-scene frames
    offset: tuple = (0.0, 0.0)                # position on its platform (off the axis)

    def at(self, t: float) -> o3d.geometry.TriangleMesh:
        mesh = o3d.geometry.TriangleMesh(self.mesh)
        mesh.translate((self.offset[0], self.offset[1], self.height))
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.radians(self.angle(t)))), center=(0, 0, 0))
        mesh.translate((self.position[0], self.position[1], 0.0))
        return mesh


def room(size=(8.0, 7.0), center=(1.0, 0.0), height=2.8) -> list[Item]:
    """Floor and four walls around 'center' (floor frame)."""
    sx, sy = size
    cx, cy = center
    meshes = [box((sx, sy, 0.02), (cx, cy), -0.02),
              box((0.1, sy, height), (cx + sx / 2, cy)), box((0.1, sy, height), (cx - sx / 2, cy)),
              box((sx, 0.1, height), (cx, cy + sy / 2)), box((sx, 0.1, height), (cx, cy - sy / 2))]
    return [Item(f"room {k}", mesh) for k, mesh in enumerate(meshes)]


# ----------------------------------------------------------------------------
# Sensor
# ----------------------------------------------------------------------------

@dataclass
class SyntheticSensor:
    """Beam fan of an Ouster-like sensor: 'rows' beams evenly spread over the
    vertical field of view, 'columns' azimuths, at 'height' above the floor,
    tilted by 'tilt_deg' about the y axis (positive: looking down at +x)."""
    rows: int = 128
    columns: int = 2048
    vertical_fov_deg: float = 90.0
    height: float = 1.15
    tilt_deg: float = 0.0
    noise: float = 0.005
    frame_rate: float = 10.0
    seed: int = 0

    def __post_init__(self):
        half = self.vertical_fov_deg / 2
        elevation = np.radians(np.linspace(half, -half, self.rows))
        azimuth = np.radians(np.linspace(0.0, 360.0, self.columns, endpoint=False))
        el, az = np.meshgrid(elevation, azimuth, indexing="ij")
        self.directions = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)
        tilt = np.radians(self.tilt_deg)
        self.world_from_sensor = np.array([[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0],
                                           [-np.sin(tilt), 0, np.cos(tilt)]])
        self.rng = np.random.default_rng(self.seed)

    def scan(self, meshes) -> np.ndarray:
        """Range image [mm] (uint16, 0 = no return)."""
        scene = o3d.t.geometry.RaycastingScene()
        for mesh in meshes:
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        directions = self.directions.reshape(-1, 3) @ self.world_from_sensor.T
        origins = np.tile([0.0, 0.0, self.height], (len(directions), 1))
        rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
        distance = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(self.rows, self.columns)
        hit = np.isfinite(distance) & (distance < 60.0)
        distance = np.where(hit, distance + self.rng.normal(0.0, self.noise, distance.shape), 0.0)
        return np.round(np.clip(distance, 0.0, 65.0) * 1000.0).astype(np.uint16)


# ----------------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------------

def write_recording(out, sensor: SyntheticSensor, items: list[Item], frames: int, background_frames: int = 5,
                    capture: dict | None = None, truth: dict | None = None) -> Path:
    """Render the scene and write a run directory (see the module docstring)."""
    out = Path(out)
    (out / "background").mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    np.savez(out / "lut.npz", direction=sensor.directions.astype(np.float32),
             offset=np.zeros_like(sensor.directions, dtype=np.float32))
    period = 1.0 / sensor.frame_rate
    sensor_origin = 1000.0                                   # sensor clock at the first frame [s]

    def save(path, range_mm, index, t):
        stamps = np.full(sensor.columns, int(round((sensor_origin + t) * 1e9)), dtype=np.uint64)
        np.savez(path, range=range_mm, frame_id=np.int64(index), time=np.float64(t), timestamps=stamps,
                 phase=np.int64(0), columns_ok=np.float64(1.0))

    empty = [item.at(0.0) for item in items if item.background]
    for k in range(background_frames):
        save(out / "background" / f"bg_{k:05d}.npz", sensor.scan(empty), k, -period * (background_frames - k))
    progress = Progress("frames", frames)
    for k in range(frames):
        t = k * period
        save(out / "frames" / f"frame_{k:05d}.npz", sensor.scan([item.at(t) for item in items]), k, t)
        progress.maybe(k + 1, 50)
    (out / "capture.json").write_text(json.dumps({"version": "synthetic", **(capture or {})}, indent=2))
    (out / "truth.json").write_text(json.dumps(truth or {}, indent=2))
    return out


def scenario(name: str, out, frames: int | None = None, sensor: SyntheticSensor | None = None) -> Path:
    """Write one of the scenarios of the module docstring; returns the run directory."""
    sensor = sensor or SyntheticSensor()
    furniture = [Item("desk", box((1.2, 0.6, 0.75), (-1.2, 1.4))), Item("cabinet", box((0.5, 0.5, 1.0), (2.6, 1.8)))]
    if name == "turntable":
        center, speed = (1.2, 0.1), 10.0
        frames = frames or 380                               # a little more than one lap at 10 deg/s
        items = room() + furniture + [
            Item("platform", platform(), center, constant_turn(speed, 1.0)),
            Item("person", person(), center, constant_turn(speed, 1.0), height=0.03, background=False,
                 offset=(0.02, -0.01))]
        truth = {"axis": list(center), "speed_deg_s": speed, "start_s": 1.0, "platform_top": 0.03}
        capture = {"motor": False}
    elif name == "inplace":
        center = (1.6, 0.2)
        frames = frames or 600
        items = room() + furniture + [Item("person", person(), center, stepping_turn(25.0, 4.0, 1.0),
                                           background=False)]
        truth = {"position": list(center), "step_deg": 25.0, "period_s": 4.0, "step_s": 1.0}
        capture = {"cue_every": 4.0}
    elif name == "detect":
        frames = frames or 80
        items = room() + furniture + [
            Item("platform A", platform(), (1.2, 0.1), constant_turn(6.0)),
            Item("person A", person(), (1.2, 0.1), constant_turn(6.0), height=0.03, background=False,
                 offset=(0.02, 0.0)),
            Item("platform B", platform(0.4), (0.0, -2.0), constant_turn(-9.0)),
            Item("chair B", chair(), (0.0, -2.0), constant_turn(-9.0), height=0.03, background=False,
                 offset=(0.05, 0.03)),
            Item("person C", person(arm_deg=8.0, height=1.68), (-1.0, 1.0), lambda t: 140.0, background=False),
            Item("stand", stand(), (2.4, -1.2))]
        truth = {"rotating": {"person A": [1.2, 0.1], "chair B": [0.0, -2.0]}, "still person": [-1.0, 1.0],
                 "humans": ["person A", "person C"]}
        capture = {}
    else:
        raise SystemExit(f"unknown scenario {name!r}: turntable, inplace or detect")
    info(f"scenario {name}: {frames} frames of {sensor.rows}x{sensor.columns} at {sensor.frame_rate:g} Hz")
    return write_recording(out, sensor, items, frames, capture=capture, truth=truth)
