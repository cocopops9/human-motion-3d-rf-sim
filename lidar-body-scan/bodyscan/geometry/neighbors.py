"""Nearest-neighbour structures and neighbourhood statistics (Open3D tensor search)."""

from __future__ import annotations

import warnings

import numpy as np
import open3d as o3d

from bodyscan.geometry.transforms import transform_points


class Target:
    """Registration target: points, normals and a nearest-neighbour index."""

    def __init__(self, cloud: o3d.geometry.PointCloud):
        self.points = np.asarray(cloud.points)
        self.normals = np.asarray(cloud.normals)
        self.index = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(self.points))
        self.index.knn_index()

    def nearest(self, query: np.ndarray):
        """Index of the nearest target point and its distance, for every query point."""
        indices, squared = self.index.knn_search(o3d.core.Tensor(query), 1)
        return indices.numpy()[:, 0], np.sqrt(squared.numpy()[:, 0])


class RadiusSearch:
    """All the points within a radius of each query.

    The Open3D hybrid search returns at most a fixed number of neighbours,
    the nearest ones. With a fixed cap a neighbourhood defined by a radius
    silently shrinks where the points are dense (64 faces of a 2 mm mesh
    reach only 5 mm), so a parameter given in millimetres would act at a
    scale set by the tessellation instead. Here the cap doubles until almost
    no query reaches it (at most 'saturated' of a chunk), up to 'limit'."""

    def __init__(self, points: np.ndarray, radius: float, cap: int = 64, limit: int = 4096,
                 saturated: float = 0.002):
        self.radius = float(radius)
        self.cap = int(cap)
        self.limit = int(limit)
        self.saturated = float(saturated)
        self.index = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(np.ascontiguousarray(points)))
        self.index.hybrid_index(self.radius)

    def query(self, queries: np.ndarray):
        """Indices (-1 = none) and squared distances, nearest first, one row per query."""
        tensor = o3d.core.Tensor(np.ascontiguousarray(queries))
        while True:
            found, squared, counts = self.index.hybrid_search(tensor, self.radius, self.cap)
            counts = counts.numpy()
            full = float(np.mean(counts >= self.cap)) if len(counts) else 0.0
            if full <= self.saturated or self.cap >= self.limit:
                width = max(int(counts.max()), 1) if len(counts) else 1
                return found.numpy()[:, :width].astype(np.int64), squared.numpy()[:, :width]
            self.cap = min(2 * self.cap, self.limit)

    def chunks(self, queries: np.ndarray, entries: int = 4_000_000):
        """(slice, indices, squared distances) over the queries, in chunks of
        about 'entries' neighbour entries (bounded memory whatever the density).
        A sample of the queries sets the cap first, so that the first chunk
        is not sized for a cap that is then doubled several times."""
        if len(queries) > 2000:
            self.query(queries[np.linspace(0, len(queries) - 1, 2000).astype(np.int64)])
        begin = 0
        while begin < len(queries):
            part = slice(begin, min(begin + max(256, entries // self.cap), len(queries)))
            found, squared = self.query(queries[part])
            yield part, found, squared
            begin = part.stop


def evaluate(source: np.ndarray, target: Target, transform: np.ndarray, distance: float):
    """Overlap (fraction of source points within 'distance') and point-to-plane RMSE [m]."""
    moved = transform_points(transform, source)
    index, gap = target.nearest(moved)
    use = gap < distance
    if not use.any():
        return 0.0, float("inf")
    residual = np.einsum("ij,ij->i", target.normals[index[use]], moved[use] - target.points[index[use]])
    return float(use.mean()), float(np.sqrt(np.mean(residual ** 2)))


def distinct_labels(neighbour_labels: np.ndarray) -> np.ndarray:
    """Number of distinct non-negative labels per row (-1 = no neighbour)."""
    ordered = np.sort(neighbour_labels, axis=1)
    return (ordered[:, :1] >= 0).astype(int)[:, 0] + np.sum(
        (ordered[:, 1:] != ordered[:, :-1]) & (ordered[:, 1:] >= 0), axis=1)


def support_filter(points: np.ndarray, labels: np.ndarray, radius: float, min_views: int) -> np.ndarray:
    """Points that have neighbours from at least 'min_views' different views
    (the point's own view included) within 'radius'."""
    if min_views <= 1:
        return np.ones(len(points), dtype=bool)
    index = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(points))
    index.hybrid_index(radius)
    keep = np.zeros(len(points), dtype=bool)
    chunk = 250000                                     # bounded memory for runs with many views
    for begin in range(0, len(points), chunk):
        query = points[begin:begin + chunk]
        neighbours, _, _ = index.hybrid_search(o3d.core.Tensor(query), radius, 48)
        neighbours = neighbours.numpy()
        neighbour_labels = np.where(neighbours >= 0, labels[np.clip(neighbours, 0, None)], -1)
        keep[begin:begin + chunk] = distinct_labels(neighbour_labels) >= min_views
    return keep


def neighbourhood(index, source: np.ndarray, labels: np.ndarray, query: np.ndarray, normals: np.ndarray,
                  radius: float):
    """Statistics of the source points within 'radius' of every query point:
    count, distinct views, median offset along the query normal, spread along
    the normal (1.4826 x MAD). 'index' is a hybrid NearestNeighborSearch of 'source'."""
    count = np.zeros(len(query), dtype=np.int64)
    distinct = np.zeros(len(query), dtype=np.int64)
    offset = np.zeros(len(query))
    spread = np.zeros(len(query))
    chunk = 40000
    for begin in range(0, len(query), chunk):
        part = query[begin:begin + chunk]
        neighbours, _, _ = index.hybrid_search(o3d.core.Tensor(part), radius, 256)
        neighbours = neighbours.numpy()
        valid = neighbours >= 0
        safe = np.clip(neighbours, 0, None)
        along = np.einsum("nkj,nj->nk", source[safe] - part[:, None, :], normals[begin:begin + chunk])
        along = np.where(valid, along, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # empty neighbourhoods give NaN
            median = np.nanmedian(along, axis=1)
            spread[begin:begin + chunk] = 1.4826 * np.nanmedian(np.abs(along - median[:, None]), axis=1)
        offset[begin:begin + chunk] = median
        count[begin:begin + chunk] = valid.sum(axis=1)
        view_ids = np.where(valid, labels[safe], -1)
        distinct[begin:begin + chunk] = distinct_labels(view_ids)
    return count, distinct, np.nan_to_num(offset), np.nan_to_num(spread)


def reference_cloud(cloud: o3d.geometry.PointCloud, voxel: float = 0.008, max_points: int = 400000,
                    seed: int = 0) -> o3d.geometry.PointCloud:
    """Registration reference from the union of many noisy views, with a fixed point budget.

    A voxel subsampling keeps every voxel that any point reaches, so the
    reference is a shell as thick as the range noise, and the more points go
    in, the further out into the noise tails it reaches; on a convex body the
    outer tail has more voxels, so views aligned to a reference built from
    several laps were pulled outward (2 to 3 mm in simulation). A random
    subset of at most 'max_points' (about one lap of views) keeps the
    distribution and the shell the same whatever the number of laps."""
    if len(cloud.points) > max_points:
        keep = np.random.default_rng(seed).choice(len(cloud.points), max_points, replace=False)
        cloud = cloud.select_by_index(np.sort(keep))
    return cloud.voxel_down_sample(voxel)
