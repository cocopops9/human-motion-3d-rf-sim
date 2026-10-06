"""Writers for point clouds with extra per-point fields."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def view_colors(count: int) -> np.ndarray:
    """Distinct colours along the hue circle, one per view."""
    hues = np.linspace(0.0, 1.0, count, endpoint=False)
    k = (np.array([5.0, 3.0, 1.0])[None, :] + hues[:, None] * 6.0) % 6.0
    return 1.0 - np.clip(np.minimum(k, 4.0 - k), 0.0, 1.0)


def write_cloud_with_fields(path, points: np.ndarray, normals: np.ndarray | None, fields: dict,
                            colors: np.ndarray | None = None) -> bool:
    """PLY with scalar fields (CloudCompare and MeshLab show them)."""
    cloud = o3d.t.geometry.PointCloud()
    cloud.point.positions = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
    if normals is not None:
        cloud.point.normals = o3d.core.Tensor(np.asarray(normals, dtype=np.float32))
    if colors is not None:
        cloud.point.colors = o3d.core.Tensor(np.asarray(colors, dtype=np.float32))
    for name, values in fields.items():
        cloud.point[name] = o3d.core.Tensor(np.asarray(values, dtype=np.float32)[:, None])
    return o3d.t.io.write_point_cloud(str(path), cloud)


def write_confidence(path, fused: o3d.geometry.PointCloud, confidence: dict) -> bool:
    """PLY with scalar fields 'views', 'points', 'spread_mm', coloured by the
    spread: green 0 mm to red 5 mm and more."""
    level = np.clip(confidence["spread_mm"] / 5.0, 0.0, 1.0)
    colors = np.column_stack([level, 1.0 - level, np.zeros_like(level)])
    return write_cloud_with_fields(path, np.asarray(fused.points), np.asarray(fused.normals),
                                   {name: confidence[name] for name in ("views", "points", "spread_mm")}, colors)
