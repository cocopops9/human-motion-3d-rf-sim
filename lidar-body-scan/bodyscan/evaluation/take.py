"""Evaluation of a tracked take of the single-LiDAR pipeline (segment-motion, then track).

For every evaluated frame, the silhouette crop of segment-motion (the pixels
around the person: measured range, person / dark labels, pixel times) becomes
a RayBatch in the floor frame. The crop columns are split into blocks of
block_columns columns; the body is posed at the mean time of each block (the
motion interpolated between the tracked frames), so the rolling shutter of
the sensor is respected to within one block (8 columns at 1024x20 are 0.4 ms).
Each block's rays are cast against its posed body and classified
(bodyscan.evaluation.fit), and the person points are measured against the same
body.

Results (evaluate_take returns them; write_take_results writes them):

    metrics.json   summary: pixel class shares, point inlier and outlier rates, per-part table,
                   kinematic and plausibility measures, tracker flag shares, settings
    metrics.npz    frame_index, times, class counts per frame (F, 6), point rates per frame,
                   per-vertex coverage, median, p95, violation and unexplained rates, part labels,
                   faces, the display pose of the body, and the per-frame per-vertex error (F, V)
                   in float16 millimetres (NaN: not observed in that frame)
    heatmap_*.ply  the body (pose of the first evaluated frame, moved to the origin) coloured
                   by median error, p95 error, violation rate and coverage
    report.md      the summary as text, with the heatmap legend
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bodyscan.config import param
from bodyscan.evaluation.fit import (CLASS_NAMES, DARK, NONE, OTHER, PERSON, ClassifyConfig, RayBatch, VertexMap,
                                     class_counts, classify, part_table, point_distances, point_rates, vertex_areas)
from bodyscan.log import Progress, info


@dataclass
class TakeEvaluationConfig:
    """Evaluating a tracked take against its measurements."""
    every: int = param(1, "evaluate every n-th tracked frame")
    sigma: float = param(0.01, "range noise of the LiDAR used for the tolerance (one sigma)", unit="m",
                         effect="larger: more pixels count as fitted, fewer as violations or unexplained")
    sigma_factor: float = param(3.0, "tolerance = max(sigma_factor x sigma, min_tolerance)")
    min_tolerance: float = param(0.02, "smallest tolerance between a measured range and the body", unit="m")
    max_error: float = param(0.2, "errors above this are capped in the per-vertex map", unit="m")
    block_columns: int = param(8, "crop columns posed together (the body is posed at their mean time)",
                               effect="smaller: closer to the rolling shutter, slower")
    level: int = param(0, "subdivision level of the avatar for the evaluation (0: as tracked by default)")
    outlier: float = param(0.05, "a person point farther than this (and than the tolerance) from the body is "
                                 "an outlier", unit="m")
    clutter: float = param(0.25, "a person point farther than this from the body belongs to nothing on it",
                           unit="m")
    min_coverage: float = param(0.05, "vertices observed in fewer frames than this share get no error value in "
                                      "the per-part table and the heatmaps")
    cutoff_hz: float = param(6.0, "velocity power above this frequency counts as high-frequency (jitter)",
                             unit="Hz")
    frame_errors: bool = param(True, "store the per-frame per-vertex error in metrics.npz (for the viewer)")


def crop_rays(segmented, person, sigma: float) -> tuple[RayBatch, np.ndarray]:
    """RayBatch of the silhouette crop of one PersonFrame (floor frame) and the crop column of every ray."""
    geometry = segmented.geometry
    r0, c0 = person.crop_origin
    mask = np.asarray(person.crop_mask)
    rows_n, cols_n = mask.shape
    rows = np.arange(r0, r0 + rows_n)
    columns = np.mod(np.arange(c0, c0 + cols_n), geometry.width)
    rr, cc = np.meshgrid(rows, columns, indexing="ij")
    directions = geometry.direction[rr, cc].reshape(-1, 3)
    offsets = geometry.offset[rr, cc].reshape(-1, 3)
    rotation, translation = geometry.rotation, geometry.translation
    origins = offsets @ rotation.T + translation
    world_directions = directions @ rotation.T
    world_directions /= np.maximum(np.linalg.norm(world_directions, axis=1, keepdims=True), 1e-12)
    measured = np.asarray(person.crop_range, dtype=np.float64).reshape(-1)
    labels = mask.reshape(-1)
    kind = np.where(labels == 1, PERSON, np.where(labels == 2, DARK, np.where(np.isfinite(measured), OTHER, NONE)))
    times = (np.asarray(person.crop_times, dtype=np.float64).reshape(-1) if person.crop_times is not None
             else np.full(len(measured), person.time))
    crop_column = np.broadcast_to(np.arange(cols_n), (rows_n, cols_n)).reshape(-1)
    batch = RayBatch(origins, world_directions, measured, kind.astype(np.uint8), np.full(len(measured), sigma),
                     times, "ouster")
    return batch, crop_column


def _blocks(batch: RayBatch, crop_column: np.ndarray, block_columns: int):
    """[(ray indices, block time)] of the crop, block by block of columns."""
    block = crop_column // max(int(block_columns), 1)
    out = []
    for b in np.unique(block):
        members = np.flatnonzero(block == b)
        times = batch.times[members]
        finite = np.isfinite(times) & (times > 0)
        out.append((members, float(np.mean(times[finite])) if finite.any() else np.nan))
    return out


def evaluate_take(motion_path, segment_folder, avatar_path, model_path=None,
                  config: TakeEvaluationConfig | None = None) -> dict:
    """Fit quality of a tracked take; returns the summary (dict) and the arrays (key 'arrays')."""
    import torch
    import open3d as o3d
    from bodyscan.body import skeleton
    from bodyscan.dynamic.avatar import Avatar
    from bodyscan.dynamic.motions import PoseSequence
    from bodyscan.dynamic.segment import SegmentedRecording
    from bodyscan.evaluation.kinematics import flag_shares, kinematic_metrics
    from bodyscan.sensors.image import cast
    c = config or TakeEvaluationConfig()
    classify_config = ClassifyConfig(c.sigma_factor, c.min_tolerance, c.max_error)
    # Copy the arrays and close the file: an open npz would lock it on Windows for the whole evaluation.
    with np.load(motion_path, allow_pickle=False) as npz:
        data = {key: npz[key] for key in npz.files}
    motion = PoseSequence.load(motion_path)
    frame_index = data["frame_index"] if "frame_index" in data else np.arange(len(motion))
    avatar = Avatar.load(avatar_path)
    model = avatar.model(model_path, "cpu", torch.float64)
    shaped = avatar.shaped(model, c.level)
    faces = np.asarray(shaped.faces_np, dtype=np.int64)
    labels = skeleton.part_labels(shaped.weights.detach().cpu().numpy())
    segmented = SegmentedRecording(segment_folder)
    by_index = {int(path.stem.split("_")[-1]): k for k, path in enumerate(segmented.paths)}
    evaluated = [k for k in range(0, len(motion), max(int(c.every), 1)) if int(frame_index[k]) in by_index]
    if not evaluated:
        raise SystemExit("no tracked frame of the motion has a segmented frame in the segment folder")
    vertex_count = int(model.subdivision(c.level).counts[-1]) if c.level > 0 else int(model.vertex_count)
    vertex_map = VertexMap(vertex_count, len(evaluated))
    counts = np.zeros((len(evaluated), len(CLASS_NAMES)), dtype=np.int64)
    rates = np.full((len(evaluated), 3), np.nan)                        # inlier, outlier, clutter
    point_counts = np.zeros(len(evaluated), dtype=np.int64)
    all_distances = []
    progress = Progress("evaluated frames", len(evaluated))
    chunk = 16
    for start in range(0, len(evaluated), chunk):
        slots = list(range(start, min(start + chunk, len(evaluated))))
        prepared = []
        times = []
        for slot in slots:
            k = evaluated[slot]
            person = segmented.load(by_index[int(frame_index[k])])
            batch, crop_column = crop_rays(segmented, person, c.sigma)
            blocks = _blocks(batch, crop_column, c.block_columns)
            for members, t in blocks:
                times.append(t if np.isfinite(t) else float(motion.times[k]))
            prepared.append((slot, batch, blocks))
        unique, inverse = np.unique(np.asarray(times), return_inverse=True)
        vertices, _ = motion.sample(unique).posed(shaped, dtype=np.float64)
        position = 0
        for slot, batch, blocks in prepared:
            frame_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
            distances, sigmas = [], []
            for members, _ in blocks:
                body = vertices[inverse[position]]
                position += 1
                part = batch.subset(members)
                scene = o3d.t.geometry.RaycastingScene()
                mesh = o3d.t.geometry.TriangleMesh()
                mesh.vertex.positions = o3d.core.Tensor(body.astype(np.float32))
                mesh.triangle.indices = o3d.core.Tensor(faces.astype(np.int32))
                scene.add_triangles(mesh)
                hits = cast(scene, part.origins, part.directions)
                result = classify(part, hits["distance"], classify_config)
                vertex_map.add(slot, faces, hits["triangle"], hits["uv"], result["classes"], result["error"],
                               c.max_error)
                frame_counts += np.bincount(result["classes"], minlength=len(CLASS_NAMES))
                person_rays = (part.kind == PERSON) & np.isfinite(part.range)
                if person_rays.any():
                    points = part.origins[person_rays] + part.directions[person_rays] * part.range[person_rays, None]
                    distances.append(point_distances(points, body, faces))
                    sigmas.append(part.sigma[person_rays])
            counts[slot] = frame_counts
            if distances:
                d = np.concatenate(distances)
                s = np.concatenate(sigmas)
                r = point_rates(d, s, classify_config, c.clutter, c.outlier)
                rates[slot] = (r["inlier"], r["outlier"], r["clutter"])
                point_counts[slot] = r["points"]
                all_distances.append(d)
        progress.maybe(slots[-1] + 1, 50)
    summary_arrays = vertex_map.summary()
    rest = motion.subset(slice(evaluated[0], evaluated[0] + 1))
    display, _ = rest.posed(shaped, dtype=np.float64)
    display = display[0] - np.array([rest.root_position[0, 0], rest.root_position[0, 1], 0.0])
    areas = vertex_areas(display, faces)
    table = part_table(summary_arrays, labels, areas, skeleton.PART_NAMES, c.min_coverage)
    total = counts.sum(axis=0)
    ranged = total[0] + total[1] + total[2]
    distances = np.concatenate(all_distances) if all_distances else np.zeros(0)
    evaluated_index = np.asarray(evaluated)
    times_eval = motion.times[evaluated_index]
    joints = data["joints"][evaluated_index] if "joints" in data else None
    kinematics = {}
    if joints is not None and len(evaluated) >= 3:
        kinematics = kinematic_metrics(times_eval, joints, motion.root_rotation[evaluated_index],
                                       motion.body[evaluated_index], c.cutoff_hz)
    summary = {
        "frames_evaluated": int(len(evaluated)), "frames_tracked": int(len(motion)),
        "pixels": {name: int(total[k]) for k, name in enumerate(CLASS_NAMES)},
        "shares_of_ranged_pixels": ({"fitted": float(total[0] / ranged), "violation": float(total[1] / ranged),
                                     "unexplained": float(total[2] / ranged)} if ranged else {}),
        "frames_with_violation_over_5pct": float(np.mean(counts[:, 1] > 0.05 * np.maximum(counts[:, :3].sum(axis=1), 1))),
        "frames_with_unexplained_over_5pct": float(np.mean(counts[:, 2] > 0.05 * np.maximum(counts[:, :3].sum(axis=1), 1))),
        "points": {"count": int(len(distances)),
                   "distance_median_mm": float(1000 * np.median(distances)) if len(distances) else None,
                   "distance_p95_mm": float(1000 * np.percentile(distances, 95)) if len(distances) else None,
                   "inlier": float(np.nanmean(rates[:, 0])), "outlier": float(np.nanmean(rates[:, 1])),
                   "clutter": float(np.nanmean(rates[:, 2]))},
        "body_coverage": float(np.sum(summary_arrays["coverage"] * areas) / areas.sum()),
        "parts": table,
        "kinematics": kinematics,
        "tracker_flags": flag_shares(data),
        "settings": asdict(c),
        "inputs": {"motion": str(motion_path), "segments": str(segment_folder), "avatar": str(avatar_path)},
    }
    arrays = {"frame_index": np.asarray(frame_index)[evaluated_index], "times": times_eval,
              "class_counts": counts, "class_names": np.array(CLASS_NAMES), "point_rates": rates,
              "point_rate_names": np.array(["inlier", "outlier", "clutter"]), "point_counts": point_counts,
              "coverage": summary_arrays["coverage"], "median": summary_arrays["median"],
              "p95": summary_arrays["p95"], "violation_rate": summary_arrays["violation_rate"],
              "unexplained_rate": summary_arrays["unexplained_rate"], "part_labels": labels,
              "part_names": np.array(skeleton.PART_NAMES), "faces": faces, "display_vertices": display}
    if c.frame_errors:
        arrays["frame_error_mm"] = (1000.0 * vertex_map.error).astype(np.float16)
    summary["arrays"] = arrays
    return summary


def report_text(summary: dict) -> str:
    from bodyscan.evaluation.heatmap import legend_text
    shares = summary.get("shares_of_ranged_pixels", {})
    points = summary["points"]
    lines = [f"# Fit quality of {Path(summary['inputs']['motion']).name}", "",
             f"Frames evaluated: {summary['frames_evaluated']} of {summary['frames_tracked']}.", ""]
    if shares:
        lines.append(f"Ranged pixels near the person: fitted {shares['fitted']:.1%}, violation (body where the "
                     f"sensor saw through) {shares['violation']:.1%}, unexplained (measured person without body "
                     f"surface) {shares['unexplained']:.1%}.")
    lines.append(f"Frames with more than 5 % violations: {summary['frames_with_violation_over_5pct']:.1%}; with "
                 f"more than 5 % unexplained: {summary['frames_with_unexplained_over_5pct']:.1%}.")
    if points["count"]:
        lines.append(f"Person points to body: median {points['distance_median_mm']:.1f} mm, p95 "
                     f"{points['distance_p95_mm']:.1f} mm; inliers {points['inlier']:.1%}, outliers "
                     f"{points['outlier']:.1%}, clutter {points['clutter']:.1%}.")
    lines.append(f"Body surface observed (area-weighted share of frames): {summary['body_coverage']:.1%}.")
    lines += ["", "| part | coverage | median mm | p95 mm | violation | unexplained |", "|---|---|---|---|---|---|"]
    for name, row in summary["parts"].items():
        if "median_mm" in row:
            lines.append(f"| {name} | {row['coverage']:.0%} | {row['median_mm']:.1f} | {row['p95_mm']:.1f} | "
                         f"{row['violation_rate']:.1%} | {row['unexplained_rate']:.1%} |")
        else:
            lines.append(f"| {name} | {row['coverage']:.0%} | not observed | | | |")
    kin = summary.get("kinematics", {})
    if kin:
        lines += ["", f"High-frequency share of the joint velocities above {kin['cutoff_hz']:.0f} Hz (mean): "
                      f"{kin['high_frequency_share_mean']:.1%}."]
        if "turns" in kin:
            lines.append(f"Root turns faster than {kin['turns']['limit_deg_s']:.0f} deg/s: "
                         f"{kin['turns']['frames_over_limit']} frames (max {kin['turns']['max_deg_s']:.0f} deg/s).")
        if "hinges_out_of_plane" in kin:
            lines.append("Hinges out of plane: " + ", ".join(f"{k} {v:.1%}" for k, v in kin["hinges_out_of_plane"].items()))
            lines.append(f"Joint speeds over the anatomical limits (worst joint): {kin['joint_speed_over_limit_any']:.1%} "
                         "of frames.")
    flags = summary.get("tracker_flags", {})
    if flags:
        lines += ["", "Tracker flags (share of frames): " + ", ".join(f"{k} {v:.1%}" for k, v in flags.items())]
    lines += ["", "Every error here is measured against the sensor, along its rays: regions no sensor saw have no "
                  "value (grey in the heatmaps), and are not evidence of accuracy.", "", legend_text()]
    return "\n".join(lines) + "\n"


def write_take_results(summary: dict, out_folder, min_coverage: float = 0.05) -> dict:
    """metrics.json, metrics.npz, heatmap_*.ply and report.md in out_folder; returns their paths."""
    from bodyscan.evaluation.heatmap import write_heatmaps
    out = Path(out_folder)
    out.mkdir(parents=True, exist_ok=True)
    arrays = summary.pop("arrays")
    try:
        (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
        np.savez_compressed(out / "metrics.npz", **arrays)
        heatmaps = write_heatmaps(out, arrays["display_vertices"], arrays["faces"],
                                  {key: arrays[key] for key in ("coverage", "median", "p95", "violation_rate")},
                                  min_coverage)
        (out / "report.md").write_text(report_text(summary), encoding="utf-8")
    finally:
        summary["arrays"] = arrays
    info(f"fit quality written to {out}")
    return {"json": out / "metrics.json", "npz": out / "metrics.npz", "report": out / "report.md", **heatmaps}


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")
