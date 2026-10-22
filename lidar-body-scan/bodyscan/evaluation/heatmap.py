"""Body heatmaps: per-vertex values as colours on the avatar, written as coloured PLY meshes.

Colours follow the review pictures: dark green for no error, through yellow at
40 % of the scale, to red at the top of the scale and above. Vertices without
a value (never observed, or observed in fewer frames than min_coverage) are
grey, so an unobserved region never looks accurate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

UNOBSERVED_GREY = np.array([128, 128, 128], dtype=np.uint8)

# (position on the scale, RGB)
_STOPS = np.array([0.0, 0.2, 0.4, 0.7, 1.0])
_COLOURS = np.array([[0, 104, 55], [102, 189, 99], [255, 221, 72], [244, 109, 67], [190, 20, 30]], dtype=np.float64)


def colours(values, vmin: float, vmax: float, valid=None) -> np.ndarray:
    """(N, 3) uint8 colours of 'values' on the scale [vmin, vmax]; invalid entries are grey."""
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(values) if valid is None else (np.asarray(valid, dtype=bool) & np.isfinite(values))
    span = max(vmax - vmin, 1e-12)
    position = np.clip((np.where(ok, values, vmin) - vmin) / span, 0.0, 1.0)
    rgb = np.stack([np.interp(position, _STOPS, _COLOURS[:, k]) for k in range(3)], axis=1)
    out = np.round(rgb).astype(np.uint8)
    out[~ok] = UNOBSERVED_GREY
    return out


def write_coloured_ply(path, vertices: np.ndarray, faces: np.ndarray, rgb: np.ndarray) -> Path:
    """Binary little-endian PLY with float32 positions, uchar colours and int32 triangles."""
    path = Path(path)
    vertices = np.ascontiguousarray(vertices, dtype="<f4")
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    faces = np.ascontiguousarray(faces, dtype="<i4")
    records = np.empty(len(vertices), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    records["xyz"] = vertices
    records["rgb"] = rgb
    triangles = np.empty(len(faces), dtype=[("count", "u1"), ("indices", "<i4", 3)])
    triangles["count"] = 3
    triangles["indices"] = faces
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n").encode("ascii")
    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(records.tobytes())
        handle.write(triangles.tobytes())
    return path


# The heatmaps written for a take: summary key, scale, unit for the legend
HEATMAPS = {
    "median": ("median", 0.0, 0.05, "m"),
    "p95": ("p95", 0.0, 0.08, "m"),
    "violation": ("violation_rate", 0.0, 0.5, "share"),
    "coverage": ("coverage", 0.0, 1.0, "share"),
}


def write_heatmaps(out_folder, vertices: np.ndarray, faces: np.ndarray, summary: dict,
                   min_coverage: float = 0.05) -> dict:
    """heatmap_<name>.ply for every entry of HEATMAPS; returns {name: path}. The error and violation
    maps show only vertices observed in at least min_coverage of the frames; the coverage map shows
    every vertex (0 is dark green there: coverage is a quantity, not an error, so its scale runs
    from red at 0 to dark green at 1)."""
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    written = {}
    seen = np.asarray(summary["coverage"]) >= min_coverage
    for name, (key, vmin, vmax, _) in HEATMAPS.items():
        values = np.asarray(summary[key], dtype=np.float64)
        if name == "coverage":
            rgb = colours(1.0 - values, 0.0, 1.0)
        else:
            rgb = colours(values, vmin, vmax, valid=seen)
        written[name] = write_coloured_ply(out_folder / f"heatmap_{name}.ply", vertices, faces, rgb)
    return written


def legend_text() -> str:
    lines = ["Heatmaps (PLY, per-vertex colours):"]
    for name, (_, vmin, vmax, unit) in HEATMAPS.items():
        if name == "coverage":
            lines.append(f"  heatmap_coverage.ply: share of frames in which a sensor observed the vertex; "
                         f"dark green 1 (always), red 0 (never)")
        else:
            scale = f"{1000 * vmin:.0f} to {1000 * vmax:.0f} mm" if unit == "m" else f"{vmin:.0%} to {vmax:.0%}"
            lines.append(f"  heatmap_{name}.ply: dark green {scale.split(' to ')[0]}, yellow at 40 %, red at "
                         f"{scale.split(' to ')[1]} and above; grey: observed in too few frames to tell")
    return "\n".join(lines)
