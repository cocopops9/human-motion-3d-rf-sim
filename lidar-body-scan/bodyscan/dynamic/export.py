"""Animated meshes for the radio simulation, and body tracks.

From a tracked motion (motion.npz) and the avatar, the body is posed at any
rate along the smooth (C1) interpolation of the tracked frames
(bodyscan.dynamic.motions.PoseSequence.sample) and written as:

    frames/mesh_000000.ply         binary PLY per time step (vertices in metres, triangles)
    frames/velocity_000000.npy     per-vertex velocity (V, 3) float32 [m/s] of that step
    frames/vertices_000000.npy     (option positions) the vertices (V, 3) float32 [m] of that step,
                                   for programs that update a mesh in place (triangles in sequence.npz)
    sequence.npz                   times, faces, part labels and names, joints (T, 55, 3),
                                   per-part mean velocity (T, P, 3)
    joints.csv                     body tracks: time, then x, y, z of the 55 joints
    parts/<part>/mesh_000000.ply   (option split_parts) one mesh per body part, for
                                   simulators that give one velocity per object
    export.json                    rate, level, frame, units, how the velocities were obtained

The frame is the floor frame of the recording: z up, z = 0 on the floor,
origin on the floor below the LiDAR, x and y along the LiDAR's own axes.
Velocities are the derivative of the interpolated motion (central difference
over 1 ms of the smooth curve, one-sided within 0.5 ms of its ends, computed
in the precision of the model), not differences between exported steps, so
they hold for the interpolated motion at any export rate.

For Sionna RT (radio ray tracing): load each step's PLY as the person (one
object, or one object per part with split_parts), set the object's velocity
from the per-part mean velocities if Doppler is computed from velocities,
or recompute the paths at every step and take the phase change between steps.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bodyscan.body import skeleton
from bodyscan.config import param
from bodyscan.log import Progress, info, warning


@dataclass
class ExportConfig:
    """Writing the animated mesh."""
    rate: float = param(120.0, "time steps per second of the export", unit="Hz",
                        effect="higher: more files; velocities are exact at any rate")
    start: float | None = param(None, "first time, seconds from the start of the motion (not set: the start)",
                                unit="s")
    stop: float | None = param(None, "last time, seconds from the start of the motion (not set: the end)",
                               unit="s")
    level: int | None = param(None, "subdivision level of the mesh (not set: the avatar's, with its detail)")
    ply: bool = param(True, "write one PLY mesh per time step")
    velocities: bool = param(True, "write the per-vertex velocity of every step")
    positions: bool = param(False, "also write the vertices of every step as .npy (a third of the PLY size)")
    split_parts: bool = param(False, "also write one mesh per body part per step")
    max_steps: int = param(100000, "refuse to write more steps than this (disk safety)")


def write_ply(path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Minimal binary little-endian PLY: float32 x, y, z and int32 triangles."""
    vertices = np.ascontiguousarray(vertices, dtype="<f4")
    faces = np.ascontiguousarray(faces, dtype="<i4")
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n").encode("ascii")
    records = np.empty(len(faces), dtype=[("count", "u1"), ("indices", "<i4", 3)])
    records["count"] = 3
    records["indices"] = faces
    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(vertices.tobytes())
        handle.write(records.tobytes())


def export_motion(motion_path, avatar_path, out, config: ExportConfig | None = None, model_path=None,
                  device: str = "auto") -> dict:
    import torch
    from bodyscan.dynamic.avatar import Avatar
    from bodyscan.dynamic.motions import PoseSequence
    config = config or ExportConfig()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    avatar = Avatar.load(avatar_path)
    model = avatar.model(model_path, device, torch.float32 if device.startswith("cuda") else torch.float64)
    shaped = avatar.shaped(model, config.level)
    motion = PoseSequence.load(motion_path)
    t0 = motion.times[0]
    start = t0 + (config.start or 0.0)
    stop = t0 + config.stop if config.stop is not None else motion.times[-1]
    times = np.arange(start, stop + 1e-9, 1.0 / config.rate)
    if len(times) > config.max_steps:
        raise SystemExit(f"{len(times)} steps exceed max_steps {config.max_steps}: lower the rate or the time range")
    out = Path(out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    labels = skeleton.part_labels(shaped.weights.detach().cpu().numpy())
    parts = len(skeleton.PART_NAMES)
    present = np.bincount(labels, minlength=parts) > 0
    faces = shaped.faces_np.astype(np.int64)
    joints_all = np.zeros((len(times), skeleton.NUM_JOINTS, 3), dtype=np.float32)
    part_velocity = np.zeros((len(times), parts, 3), dtype=np.float32)
    h = 5e-4                                                            # half step of the central difference [s]
    batch = 32
    progress = Progress("exported steps", len(times))
    part_faces = {}
    if config.split_parts:
        face_part = labels[faces].max(axis=1)                          # a face belongs to its highest-labelled corner
        for k, name in enumerate(skeleton.PART_NAMES):
            members = np.flatnonzero(face_part == k)
            if len(members):
                used = np.unique(faces[members])
                remap = -np.ones(len(labels), dtype=np.int64)
                remap[used] = np.arange(len(used))
                part_faces[name] = (used, remap[faces[members]])
                (out / "parts" / name).mkdir(parents=True, exist_ok=True)
    for first in range(0, len(times), batch):
        span = times[first:first + batch]
        now = motion.sample(span)
        vertices, joints = now.posed(shaped)
        joints_all[first:first + len(span)] = joints
        # velocities always (their part means go into sequence.npz); per-vertex files only if asked.
        # The stencil stays inside the motion: outside it the motion is held, which would halve the
        # velocity at the ends.
        later = np.minimum(span + h, motion.times[-1])
        earlier = np.maximum(span - h, motion.times[0])
        interval = np.maximum(later - earlier, 1e-9)[:, None, None]
        ahead, _ = motion.sample(later).posed(shaped, dtype=np.float64)
        behind, _ = motion.sample(earlier).posed(shaped, dtype=np.float64)
        velocity = ((ahead - behind) / interval).astype(np.float32)
        for k in np.flatnonzero(present):
            part_velocity[first:first + len(span), k] = velocity[:, labels == k].mean(axis=1)
        for i in range(len(span)):
            step = first + i
            if config.ply:
                write_ply(out / "frames" / f"mesh_{step:06d}.ply", vertices[i], faces)
            if config.velocities:
                np.save(out / "frames" / f"velocity_{step:06d}.npy", velocity[i])
            if config.positions:
                np.save(out / "frames" / f"vertices_{step:06d}.npy", vertices[i].astype(np.float32))
            for name, (used, local_faces) in part_faces.items():
                write_ply(out / "parts" / name / f"mesh_{step:06d}.ply", vertices[i][used], local_faces)
        progress.maybe(min(first + batch, len(times)), 10 * batch)
    np.savez_compressed(out / "sequence.npz", times=times, faces=faces, part_labels=labels,
                        part_names=np.array(skeleton.PART_NAMES), joints=joints_all, part_velocity=part_velocity,
                        joint_names=np.array(skeleton.JOINT_NAMES))
    header = "time," + ",".join(f"{name}_{axis}" for name in skeleton.JOINT_NAMES for axis in "xyz")
    table = np.column_stack([times, joints_all.reshape(len(times), -1)])
    np.savetxt(out / "joints.csv", table, delimiter=",", header=header, comments="", fmt="%.5f")
    info_json = {"motion": str(Path(motion_path).resolve()), "avatar": str(Path(avatar_path).resolve()),
                 "steps": int(len(times)), "rate_hz": config.rate, "start_s": float(start), "stop_s": float(stop),
                 "level": shaped.level, "vertices": int(shaped.vertex_count), "triangles": int(len(faces)),
                 "units": "metres, seconds (sensor clock), m/s",
                 "frame": "floor frame of the recording: z up, z = 0 on the floor, origin below the LiDAR",
                 "velocities": "derivative of the C1 interpolated motion (central difference over 1 ms, "
                               "one-sided at the ends)",
                 "config": asdict(config)}
    (out / "export.json").write_text(json.dumps(info_json, indent=2))
    if len(times) * shaped.vertex_count > 2e9 / 12:
        warning("large export: consider a lower rate or level")
    info(f"exported {len(times)} steps at {config.rate:g} Hz ({shaped.vertex_count} vertices, level {shaped.level}) "
         f"to {out}")
    return info_json
