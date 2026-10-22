"""Coverage of a moving body by the sensors of a rig: the placement study before any tripod moves.

For every sampled time of a motion the body is posed and, for every sensor:

    visible   a vertex is visible when it is inside the sensor's field of view
              and range limits, faces the sensor, and the first surface along
              the ray from the sensor to it is the vertex itself (the body
              occludes itself; nothing else is in the scene)
    samples   the sensor's own rays that point at the body are cast against it;
              the hits per body part give the sampling density, and the mean
              spacing between samples on the visible surface is
              sqrt(visible area / hits)

Per body part, over the frames: the visible share of the part's area for each
sensor and for the union of the sensors, the share of frames in which at least
half of the part is visible to some sensor, and the sample spacing per sensor.
This measures geometry only (what the sensors can see), not tracking error;
the tracking error of a layout comes from the synthetic bench once the
multi-sensor tracker exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import open3d as o3d

from bodyscan.config import param
from bodyscan.sensors.image import cast
from bodyscan.sensors.rig import Rig


@dataclass
class CoverageConfig:
    """The coverage study of a rig."""
    rate: float = param(10.0, "poses sampled per second of the motion", unit="Hz")
    occlusion_margin: float = param(0.01, "a vertex is visible if the first surface along its ray lies within "
                                          "this of it", unit="m")
    half_visible: float = param(0.5, "a part counts as seen in a frame when at least this share of its area is "
                                     "visible to some sensor")


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    face_normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])     # area-weighted
    normals = np.zeros_like(v)
    for k in range(3):
        np.add.at(normals, f[:, k], face_normals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def _body_scene(vertices: np.ndarray, faces: np.ndarray) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(np.asarray(vertices, dtype=np.float32))
    mesh.triangle.indices = o3d.core.Tensor(np.asarray(faces, dtype=np.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene


def visible_vertices(placement, vertices: np.ndarray, normals: np.ndarray, scene, margin: float = 0.01) -> np.ndarray:
    """(V,) True for the vertices a sensor sees (field of view, facing, not occluded by the body)."""
    origin = placement.position
    in_view = placement.sensor.in_view(placement.to_sensor(vertices))
    toward = vertices - origin
    distance = np.linalg.norm(toward, axis=1)
    directions = toward / np.maximum(distance[:, None], 1e-12)
    facing = np.sum(normals * -directions, axis=1) > 0.0
    candidates = np.flatnonzero(in_view & facing)
    visible = np.zeros(len(vertices), dtype=bool)
    if len(candidates):
        hits = cast(scene, np.broadcast_to(origin, (len(candidates), 3)), directions[candidates])
        visible[candidates] = hits["distance"] >= distance[candidates] - margin
    return visible


def world_directions(placement) -> np.ndarray:
    """(H x W, 3) the sensor's ray directions in the world frame."""
    return placement.sensor.directions().reshape(-1, 3) @ np.asarray(placement.world_from_sensor)[:3, :3].T


def sample_hits(placement, vertices: np.ndarray, scene, directions: np.ndarray | None = None) -> np.ndarray:
    """(T,) triangle index of every sensor ray that hits the body (the rays pointing at its bounding sphere).
    directions: the sensor's world ray directions (world_directions), computed once per rig."""
    origin = placement.position
    centre = vertices.mean(axis=0)
    radius = float(np.linalg.norm(vertices - centre, axis=1).max())
    toward = centre - origin
    distance = float(np.linalg.norm(toward))
    if distance <= radius:
        cone = -1.0
    else:
        cone = float(np.cos(np.arcsin(radius / distance) + np.radians(1.0)))
    if directions is None:
        directions = world_directions(placement)
    pointing = directions @ (toward / max(distance, 1e-12)) >= cone
    if not pointing.any():
        return np.zeros(0, dtype=np.int64)
    rays = directions[pointing]
    hits = cast(scene, np.broadcast_to(origin, rays.shape), rays)
    sensor = placement.sensor
    usable = np.isfinite(hits["distance"]) & (hits["distance"] >= sensor.min_range)
    if hasattr(sensor, "depth_of"):
        local = rays[usable] @ np.asarray(placement.world_from_sensor)[:3, :3]
        depth = hits["distance"][usable] * local[:, 0]
        keep = depth <= sensor.max_range
    else:
        keep = hits["distance"][usable] <= sensor.max_range
    return hits["triangle"][usable][keep]


def rig_coverage(rig: Rig, shaped, motion, config: CoverageConfig | None = None, labels=None) -> dict:
    """Coverage of the body playing 'motion' (PoseSequence, floor frame of the rig) by every sensor of
    the rig and by their union. labels: per-vertex part index (default from the skinning weights)."""
    from bodyscan.body import skeleton
    from bodyscan.evaluation.fit import vertex_areas
    c = config or CoverageConfig()
    faces = np.asarray(shaped.faces_np, dtype=np.int64)
    if labels is None:
        labels = skeleton.part_labels(shaped.weights.detach().cpu().numpy())
    parts = skeleton.PART_NAMES
    times = np.arange(motion.times[0], motion.times[-1], 1.0 / c.rate)
    vertices_all, _ = motion.sample(times).posed(shaped, dtype=np.float64)
    names = rig.names
    frames, part_count = len(times), len(parts)
    visible_share = np.zeros((len(names) + 1, frames, part_count))          # last row: the union
    hits = np.zeros((len(names), frames, part_count))
    visible_area = np.zeros((len(names), frames, part_count))
    total_share = np.zeros((len(names) + 1, frames))
    triangle_part = labels[faces[:, 0]]
    rays = {name: world_directions(rig.placement(name)) for name in names}
    for t in range(frames):
        vertices = vertices_all[t]
        normals = vertex_normals(vertices, faces)
        areas = vertex_areas(vertices, faces)
        part_area = np.bincount(labels, weights=areas, minlength=part_count)
        scene = _body_scene(vertices, faces)
        union = np.zeros(len(vertices), dtype=bool)
        for s, name in enumerate(names):
            placement = rig.placement(name)
            seen = visible_vertices(placement, vertices, normals, scene, c.occlusion_margin)
            union |= seen
            area = np.bincount(labels[seen], weights=areas[seen], minlength=part_count)
            visible_area[s, t] = area
            with np.errstate(invalid="ignore", divide="ignore"):
                visible_share[s, t] = np.where(part_area > 0, area / part_area, np.nan)
            total_share[s, t] = areas[seen].sum() / areas.sum()
            triangles = sample_hits(placement, vertices, scene, rays[name])
            if len(triangles):
                hits[s, t] = np.bincount(triangle_part[triangles], minlength=part_count)
        area = np.bincount(labels[union], weights=areas[union], minlength=part_count)
        with np.errstate(invalid="ignore", divide="ignore"):
            visible_share[-1, t] = np.where(part_area > 0, area / part_area, np.nan)
        total_share[-1, t] = areas[union].sum() / areas.sum()
    sources = names + ["union"]
    result = {"sensors": names, "frames": int(frames), "rate_hz": c.rate, "settings": asdict(c),
              "body_visible_share": {src: {"mean": float(total_share[k].mean()), "min": float(total_share[k].min())}
                                     for k, src in enumerate(sources)},
              "parts": {}}
    for p, part in enumerate(parts):
        row = {}
        for k, src in enumerate(sources):
            values = visible_share[k, :, p]
            row[f"visible_{src}"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        row["frames_half_visible"] = float(np.mean(visible_share[-1, :, p] >= c.half_visible))
        for s, name in enumerate(names):
            count = hits[s, :, p].sum()
            area = visible_area[s, :, p].sum()
            row[f"spacing_mm_{name}"] = float(1000.0 * np.sqrt(area / count)) if count > 0 else None
            row[f"samples_per_frame_{name}"] = float(hits[s, :, p].mean())
        result["parts"][part] = row
    ranked = [(name, row) for name, row in result["parts"].items() if np.isfinite(row["visible_union"])]
    weakest = sorted(ranked, key=lambda item: item[1]["visible_union"])[:5]
    result["least_visible_parts"] = [[name, row["visible_union"]] for name, row in weakest]
    return result


# ----------------------------------------------------------------------------
# The bench: layouts x motions
# ----------------------------------------------------------------------------

BENCH_MOTIONS = ("walk-across", "walk-radial", "jump-facing", "jump-side")


def bench_motion(shaped, name: str, zone_distance: float = 2.3):
    """Short motions centred on the capture zone (the Ouster at the origin, the zone along +x)."""
    from bodyscan.dynamic import motions
    d = zone_distance
    if name == "walk-across":                       # side-on to the Ouster, as walk01 and walk02
        return motions.walk(shaped, duration=2.0, speed=1.0, heading_deg=90.0, start_xy=(d, -1.0))
    if name == "walk-radial":                       # towards the Ouster, as walk03
        return motions.walk(shaped, duration=2.0, speed=0.8, heading_deg=180.0, start_xy=(d + 0.8, 0.0))
    if name == "jump-facing":                       # facing the Ouster, as jump01
        return motions.jump(shaped, count=2, height=0.25, heading_deg=180.0, start_xy=(d, 0.0))
    if name == "jump-side":                         # side-on, as jump02
        return motions.jump(shaped, count=2, height=0.25, heading_deg=90.0, start_xy=(d, 0.0))
    raise SystemExit(f"unknown bench motion {name!r}: {', '.join(BENCH_MOTIONS)}")


def bench_table(results: dict) -> str:
    """Markdown table of a bench: rows layouts, columns motions, cells the mean visible share of the body
    (union of the sensors), then the parts least visible per layout."""
    layouts = list(results)
    motions = list(next(iter(results.values())))
    lines = ["| layout | " + " | ".join(motions) + " |", "|---" * (len(motions) + 1) + "|"]
    for layout in layouts:
        cells = [f"{results[layout][m]['body_visible_share']['union']['mean']:.0%}" for m in motions]
        lines.append(f"| {layout} | " + " | ".join(cells) + " |")
    lines += ["", "Least visible parts (union of the sensors, mean share of the part's area over all motions):", ""]
    for layout in layouts:
        parts = {}
        for m in motions:
            for name, row in results[layout][m]["parts"].items():
                parts.setdefault(name, []).append(row["visible_union"])
        worst = sorted(((float(np.mean(v)), k) for k, v in parts.items()))[:4]
        lines.append(f"* {layout}: " + ", ".join(f"{k} {v:.0%}" for v, k in worst))
    return "\n".join(lines) + "\n"


def bench_rig(layout_names, motion_names, out=None, model_path=None, avatar_path=None,
              config: CoverageConfig | None = None, zone_distance: float = 2.3) -> dict:
    """Coverage of every layout (bodyscan.sensors.layouts) for every bench motion; writes
    bench_rig.json and bench_rig.md into 'out' if given. The body is the avatar (with its model file)
    when avatar_path is given, else the model at model_path, else the procedural test body."""
    import json
    from pathlib import Path
    import torch
    from bodyscan.body.model import BodyModel
    from bodyscan.log import info
    from bodyscan.sensors.layouts import LAYOUTS, build_rig
    if avatar_path:
        from bodyscan.dynamic.avatar import Avatar
        avatar = Avatar.load(avatar_path)
        shaped = avatar.shaped(avatar.model(model_path, "cpu", torch.float64), 0)
        body = f"avatar {Path(avatar_path).name}"
    else:
        from bodyscan.body.testbody import cached
        path = model_path or cached()
        shaped = BodyModel(path, device="cpu", dtype=torch.float64).shaped()
        body = f"model {Path(path).name}"
    unknown = [name for name in layout_names if name not in LAYOUTS]
    if unknown:
        raise SystemExit(f"unknown layout {unknown[0]!r}: {', '.join(LAYOUTS)}")
    info(f"building the bench motions ({', '.join(motion_names)})")
    sequences = {motion: bench_motion(shaped, motion, zone_distance) for motion in motion_names}   # slow: once
    results = {}
    for layout in layout_names:
        rig = build_rig(LAYOUTS[layout], zone_distance=zone_distance)
        results[layout] = {}
        for motion in motion_names:
            results[layout][motion] = rig_coverage(rig, shaped, sequences[motion], config)
            share = results[layout][motion]["body_visible_share"]["union"]["mean"]
            info(f"  {layout:24s} {motion:12s} body visible to some sensor: {share:.0%} (mean over the frames)")
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        layouts = {name: {"description": LAYOUTS[name].description,
                          "sensors": [{"name": p.name, "model": p.sensor.model, "position_m": p.position.round(3).tolist()}
                                      for p in build_rig(LAYOUTS[name], zone_distance=zone_distance).placements]}
                   for name in layout_names}
        (out / "bench_rig.json").write_text(json.dumps({"body": body, "zone_distance_m": zone_distance,
                                                        "layouts": layouts, "results": results}, indent=2))
        header = (f"# Coverage of the body by candidate layouts\n\nBody: {body}; capture zone centre "
                  f"{zone_distance} m from the Ouster. Cells: mean share of the body surface visible to at least "
                  "one sensor (geometry only: field of view, facing, self-occlusion).\n\n")
        (out / "bench_rig.md").write_text(header + bench_table(results))
        info(f"wrote {out / 'bench_rig.md'}")
    return results
