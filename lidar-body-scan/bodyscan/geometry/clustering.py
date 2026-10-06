"""Connected components of point sets."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def components_2d(xy: np.ndarray, radius: float) -> np.ndarray:
    """Connected components of 2D points (neighbours closer than 'radius'):
    DBSCAN with one point per cluster is single linkage. Label per point."""
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.column_stack([xy, np.zeros(len(xy))])))
    return np.asarray(cloud.cluster_dbscan(radius, 1, print_progress=False))


def clusters_3d(points: np.ndarray, eps: float, min_points: int = 10) -> np.ndarray:
    """DBSCAN labels of 3D points (-1 = noise)."""
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return np.asarray(cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))


def largest_cluster(points: np.ndarray, eps: float, min_points: int = 10) -> np.ndarray:
    """Indices of the points of the largest DBSCAN cluster (all points if there is none)."""
    if len(points) < 20:
        return np.arange(len(points))
    labels = clusters_3d(points, eps, min_points)
    if labels.size == 0 or labels.max() < 0:
        return np.arange(len(points))
    return np.flatnonzero(labels == np.bincount(labels[labels >= 0]).argmax())


def voxel_components(points: np.ndarray, voxel: float, min_points: int = 1) -> np.ndarray:
    """Connected components of the occupied voxels (26-neighbourhood), one
    label per point; fast for hundreds of thousands of points (numpy only).
    Components with fewer than 'min_points' points get label -1."""
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    cells = np.floor(points / voxel).astype(np.int64)
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    parent = np.arange(len(unique))

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    lookup = {tuple(c): k for k, c in enumerate(unique.tolist())}
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if (dx, dy, dz) > (0, 0, 0)]
    for k, cell in enumerate(unique.tolist()):
        for dx, dy, dz in offsets:
            other = lookup.get((cell[0] + dx, cell[1] + dy, cell[2] + dz))
            if other is not None:
                a, b = find(k), find(other)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    roots = np.array([find(k) for k in range(len(unique))])
    _, compact = np.unique(roots, return_inverse=True)
    labels = compact.reshape(-1)[inverse]
    if min_points > 1:
        counts = np.bincount(labels)
        labels = np.where(counts[labels] >= min_points, labels, -1)
    return labels
