"""Fusion of the corrected views into one cloud with a per-point confidence."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import open3d as o3d

from bodyscan.config import param
from bodyscan.geometry import neighbourhood, support_filter
from bodyscan.io import view_colors


@dataclass
class FusionConfig:
    """Fusion of the views: support filter, voxel averaging, robust surface fit."""
    min_views: int = param(3, "support filter: a point needs neighbours from this many views (turntable: per lap "
                                 "of views; in place: keyframes)",
                           effect="larger removes more ghosts (and thin parts seen by few views: hands, fingers)")
    support_radius: float = param(0.02, "neighbourhood of the support filter", unit="m",
                                  effect="smaller keeps thin parts separate (fingers) but rejects more points")
    voxel: float = param(0.005, "voxel of the fused cloud", unit="m", effect="smaller: denser output, slower")
    confidence_radius: float = param(0.01, "neighbourhood of the surface fit and of the per-point confidence",
                                     unit="m", effect="larger: smoother surface, fine details flattened")
    surface_fit: bool = param(True, "move every point onto the median surface of the view points around it "
                                    "(off: keep the noisier voxel averages)")


def colored_union(clouds):
    """All views in one cloud, one colour per view, and the view label of every point."""
    views = o3d.geometry.PointCloud()
    labels = []
    for k, (cloud, color) in enumerate(zip(clouds, view_colors(len(clouds)))):
        colored = copy.deepcopy(cloud)
        colored.paint_uniform_color(color)
        views += colored
        labels.append(np.full(len(colored.points), k))
    return views, np.concatenate(labels) if labels else np.zeros(0, dtype=np.int64)


class SurfaceFusion:
    """Support filter, voxel averaging, robust surface fit, and a confidence per output point.

    Surface fit: the voxel averages of the union of the views hold one or two
    points each and keep the range noise. Every output point is moved along
    its normal onto the median of the supported view points around it, first
    within 2 x confidence_radius, then within confidence_radius, and points
    that end up on the same spot are merged. The starting points and the
    coarse pass use a random subset of one lap of views (more data would put
    more starting points further out in the noise tails, beyond the reach of
    the fit); the final pass uses every point of every lap, so more laps make
    the surface more precise.

    Confidence, from the final neighbourhood (confidence_radius): the number
    of distinct views, of points, and their spread along the normal
    (1.4826 x median absolute deviation, in mm; it also contains the surface
    curvature within the radius, about 1 mm on an arm for 1 cm). Standard
    error of the fitted surface about 1.25 x spread / sqrt(points)."""

    def __init__(self, config: FusionConfig):
        self.config = config

    def fuse(self, clouds, min_views: int, reference_views=None):
        """Returns (fused cloud, all views, fraction removed by the support filter, confidence dict)."""
        c = self.config
        views, labels = colored_union(clouds)
        points = np.asarray(views.points)
        keep = support_filter(points, labels, c.support_radius, min_views)
        supported = views.select_by_index(np.flatnonzero(keep))
        labels = labels[keep]
        source = np.asarray(supported.points)
        index = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(source))
        index.hybrid_index(2 * c.confidence_radius)
        budget = len(source)
        if reference_views:
            budget = min(budget, int(len(source) / max(len(clouds), 1) * reference_views))
        subset = np.sort(np.random.default_rng(1).choice(len(source), budget, replace=False)) \
            if budget < len(source) else np.arange(len(source))
        seeds = supported.select_by_index(subset)
        coarse_index = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(source[subset]))
        coarse_index.hybrid_index(2 * c.confidence_radius)

        fused = seeds.voxel_down_sample(c.voxel) if c.voxel > 0 else copy.deepcopy(seeds)
        fused.normalize_normals()
        if len(fused.points) > 30:
            fused, _ = fused.remove_statistical_outlier(20, 2.0)
        fused.colors = o3d.utility.Vector3dVector()
        if c.surface_fit:
            positions, normals = np.asarray(fused.points).copy(), np.asarray(fused.normals)
            passes = ((coarse_index, source[subset], labels[subset], 2 * c.confidence_radius),
                      (index, source, labels, c.confidence_radius))
            for search, pass_points, pass_labels, radius in passes:
                count, _, offset, _ = neighbourhood(search, pass_points, pass_labels, positions, normals, radius)
                positions += np.where(count >= 5, offset, 0.0)[:, None] * normals
            fused.points = o3d.utility.Vector3dVector(positions)
            fused = fused.voxel_down_sample(c.voxel)                  # merge points that met on the surface
            fused.normalize_normals()
        query, query_normals = np.asarray(fused.points), np.asarray(fused.normals)
        count, distinct, offset, spread = neighbourhood(index, source, labels, query, query_normals,
                                                        c.confidence_radius)
        confidence = {"views": distinct, "points": count, "spread_mm": 1000 * spread, "offset_mm": 1000 * offset}
        if len(views.points) > 2_000_000:              # keep the _views.ply file a manageable size
            views = views.voxel_down_sample(0.004)
        return fused, views, float(1.0 - keep.mean()), confidence


def simple_fusion(clouds, min_views: int, support_radius: float, voxel: float):
    """Support filter and voxel averaging only (in-place pipeline).
    Returns (fused cloud, all views, fraction removed)."""
    views, labels = colored_union(clouds)
    keep = support_filter(np.asarray(views.points), labels, support_radius, min_views)
    supported = views.select_by_index(np.flatnonzero(keep))
    fused = supported.voxel_down_sample(voxel) if voxel > 0 else copy.deepcopy(supported)
    fused.normalize_normals()
    if len(fused.points) > 30:
        fused, _ = fused.remove_statistical_outlier(20, 2.0)
    fused.colors = o3d.utility.Vector3dVector()
    return fused, views, float(1.0 - keep.mean())
