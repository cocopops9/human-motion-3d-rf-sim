"""Least-squares and RANSAC fits: planes, circles, robust spread."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


def robust_spread(values: np.ndarray) -> float:
    """1.4826 x median absolute deviation (the standard deviation for Gaussian data)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def fit_circle(xy: np.ndarray, iterations: int = 6, tolerance: float = 0.02):
    """Circle through 2D points, refitted on the inliers.
    Returns (centre, radius, rms of the inliers, number of inliers)."""
    keep = np.ones(len(xy), dtype=bool)
    center, radius, residual = None, None, None
    for _ in range(iterations):
        a = np.column_stack([2 * xy[keep], np.ones(keep.sum())])
        b = (xy[keep] ** 2).sum(axis=1)
        solution = np.linalg.lstsq(a, b, rcond=None)[0]
        center = solution[:2]
        radius = float(np.sqrt(max(solution[2] + center @ center, 0.0)))
        residual = np.abs(np.linalg.norm(xy - center, axis=1) - radius)
        keep = residual < max(tolerance, 2.5 * np.median(residual[keep]))
    return center, radius, float(np.sqrt(np.mean(residual[keep] ** 2))), int(keep.sum())


@dataclass
class Plane:
    """n . x + d = 0 with |n| = 1."""
    normal: np.ndarray
    offset: float
    inliers: int

    def distance(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.offset


def ransac_planes(points: np.ndarray, count: int = 6, threshold: float = 0.02, min_inliers: int = 500,
                  sample: int = 40000, seed: int = 0, stop_below: int = 1000) -> list[Plane]:
    """Largest planes found one after the other (RANSAC on a random subsample;
    the inliers of each plane are removed before the next search, which stops
    when fewer than 'stop_below' points remain)."""
    rng = np.random.default_rng(seed)
    o3d.utility.random.seed(seed)
    points = points[rng.choice(len(points), min(len(points), sample), replace=False)]
    remaining = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    planes = []
    for _ in range(count):
        if len(remaining.points) < max(stop_below, 3):
            break
        model, inliers = remaining.segment_plane(threshold, 3, 1000)
        normal, offset = np.array(model[:3], dtype=np.float64), float(model[3])
        scale = np.linalg.norm(normal)
        if len(inliers) >= min_inliers:
            planes.append(Plane(normal / scale, offset / scale, len(inliers)))
        remaining = remaining.select_by_index(inliers, invert=True)
    return planes


def refit_plane(points: np.ndarray, plane: Plane, band: float = 0.015, min_points: int = 1000) -> Plane:
    """Least-squares refit on all the points within 'band' of a plane; the
    normal keeps the side of the original."""
    near = points[np.abs(plane.distance(points)) < band]
    if len(near) <= min_points:
        return plane
    center = near.mean(axis=0)
    normal = np.linalg.svd(near - center, full_matrices=False)[2][2]
    if normal @ plane.normal < 0:
        normal = -normal
    return Plane(normal, -float(normal @ center), len(near))
