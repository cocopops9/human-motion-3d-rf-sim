"""Surface reconstruction from a point cloud.

Mesher is the interface (build(cloud) -> mesh); implementations:

    GridMesher         organized range image: neighbouring pixels become triangles
    PoissonMesher      screened Poisson; needs oriented normals
    BallPivotingMesher interpolates the points; holes where sampling is sparse
    AlphaMesher        alpha shape
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from bodyscan.meshing.cleanup import orient_towards_sensor, remove_small_fragments


@dataclass
class CloudInput:
    """Point cloud plus, when available, its organized (H, W) structure."""
    points: np.ndarray
    grid: np.ndarray | None = None
    valid: np.ndarray | None = None
    normals: np.ndarray | None = None

    @property
    def is_organized(self) -> bool:
        return self.grid is not None


def load_cloud_input(path) -> CloudInput:
    """.npz (organized 'xyz' (H, W, 3) or 'points' (N, 3)), or .ply / .pcd / .xyz (with normals if stored)."""
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "xyz" in data.files and data["xyz"].ndim == 3:
            grid = data["xyz"].astype(np.float64)
            valid = np.linalg.norm(grid, axis=2) > 1e-6
            return CloudInput(grid[valid], grid, valid)
        if "points" in data.files:
            return CloudInput(data["points"].astype(np.float64))
        raise ValueError(f"{path} has neither an 'xyz' grid nor a 'points' array")
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"could not read any point from {path}")
    normals = np.asarray(cloud.normals) if cloud.has_normals() else None
    return CloudInput(np.asarray(cloud.points), normals=normals)


def crop_input(cloud_input: CloudInput, box_min, box_max) -> CloudInput:
    box_min = np.asarray(box_min, dtype=np.float64)
    box_max = np.asarray(box_max, dtype=np.float64)

    def inside(points):
        return np.all((points >= box_min) & (points <= box_max), axis=-1)

    if cloud_input.is_organized:
        valid = cloud_input.valid & inside(cloud_input.grid)
        return CloudInput(cloud_input.grid[valid], cloud_input.grid, valid)
    keep = inside(cloud_input.points)
    normals = cloud_input.normals[keep] if cloud_input.normals is not None else None
    return CloudInput(cloud_input.points[keep], normals=normals)


def mean_neighbor_spacing(cloud) -> float:
    return float(np.mean(np.asarray(cloud.compute_nearest_neighbor_distance())))


def prepare_cloud(points, normals=None, voxel=0.0, outlier_neighbors=20, outlier_std=2.0, remove_plane=False,
                  plane_threshold=0.01):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)   # kept by the filters below
    if voxel > 0:
        cloud = cloud.voxel_down_sample(voxel)
    if outlier_neighbors > 0:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=outlier_neighbors, std_ratio=outlier_std)
    if remove_plane:
        _, plane_indices = cloud.segment_plane(distance_threshold=plane_threshold, ransac_n=3, num_iterations=1000)
        cloud = cloud.select_by_index(plane_indices, invert=True)
    return cloud


def estimate_normals(cloud, orient="auto", normal_radius=0.0, sensor_origin=(0.0, 0.0, 0.0)):
    """Normals and their orientation.

    input       normals stored in the file (the fusion writes them outward);
    consistent  propagation over a neighbourhood graph, then away from the centroid
                (objects scanned all around);
    sensor      towards the sensor origin (single views only);
    auto        input when the file has normals, else consistent."""
    if orient == "auto":
        orient = "input" if cloud.has_normals() else "consistent"
    if orient == "input":
        if not cloud.has_normals():
            raise ValueError("orient 'input' needs a point cloud file with normals")
        cloud.normalize_normals()
        return cloud
    spacing = mean_neighbor_spacing(cloud)
    radius = normal_radius if normal_radius > 0 else 4.0 * spacing
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    if orient == "sensor":
        cloud.orient_normals_towards_camera_location(np.asarray(sensor_origin))
        return cloud
    cloud.orient_normals_consistent_tangent_plane(k=15)
    points, normals = np.asarray(cloud.points), np.asarray(cloud.normals)
    if np.mean(np.einsum("ij,ij->i", normals, points - points.mean(axis=0))) < 0:
        cloud.normals = o3d.utility.Vector3dVector(-normals)
    return cloud


def denoise_by_plane_projection(cloud, radius, iterations=2):
    """Move every point onto the plane fitted to its neighbours within 'radius'
    (Gaussian weights): removes scan-row layering and range noise, and
    flattens features smaller than about the radius. Normals are kept."""
    points = np.asarray(cloud.points).copy()
    original_normals = np.asarray(cloud.normals).copy() if cloud.has_normals() else None
    for _ in range(iterations):
        index = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(points))
        index.hybrid_index(radius)
        neighbours, squared, _ = index.hybrid_search(o3d.core.Tensor(points), radius, 64)
        neighbours, squared = neighbours.numpy(), squared.numpy()
        valid = neighbours >= 0
        weight = np.where(valid, np.exp(-squared / (0.5 * radius) ** 2), 0.0)
        gathered = points[np.clip(neighbours, 0, None)]
        total = weight.sum(axis=1, keepdims=True)
        centroid = (weight[..., None] * gathered).sum(axis=1) / total
        offset = gathered - centroid[:, None, :]
        covariance = np.einsum("nk,nki,nkj->nij", weight, offset, offset) / total[..., None]
        _, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, :, 0]                               # smallest eigenvalue
        enough = valid.sum(axis=1) >= 8
        distance = np.einsum("ni,ni->n", points - centroid, normal)
        points[enough] -= distance[enough, None] * normal[enough]
    result = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if original_normals is not None:
        result.normals = o3d.utility.Vector3dVector(original_normals)
    return result


class Mesher(ABC):
    @abstractmethod
    def build(self, cloud):
        """Triangle mesh from an Open3D point cloud (with normals where needed)."""


class PoissonMesher(Mesher):
    """Screened Poisson. trim_distance > 0 removes vertices farther than that
    many mean point spacings from the data (extrapolated surface) and clips to
    the cloud bounds; 0 keeps the closed surface (unseen parts invented)."""

    def __init__(self, depth=9, trim_distance=0.0, density_quantile=0.0):
        self.depth, self.trim_distance, self.density_quantile = depth, trim_distance, density_quantile

    def build(self, cloud):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=self.depth)
        densities = np.asarray(densities)
        if self.trim_distance > 0:
            limit = self.trim_distance * mean_neighbor_spacing(cloud)
            distance = np.asarray(o3d.geometry.PointCloud(mesh.vertices).compute_point_cloud_distance(cloud))
            mesh.remove_vertices_by_mask(distance > limit)
            remove_small_fragments(mesh, fraction=0.005)
        if self.density_quantile > 0:
            mesh.remove_vertices_by_mask(densities < np.quantile(densities, self.density_quantile))
        if self.trim_distance <= 0:
            return mesh
        bounds = cloud.get_axis_aligned_bounding_box()
        margin = 2.0 * mean_neighbor_spacing(cloud)
        return mesh.crop(o3d.geometry.AxisAlignedBoundingBox(bounds.min_bound - margin, bounds.max_bound + margin))


class BallPivotingMesher(Mesher):
    def __init__(self, radii=(1.5, 3.0, 6.0)):
        self.radii = radii

    def build(self, cloud):
        spacing = mean_neighbor_spacing(cloud)
        return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            cloud, o3d.utility.DoubleVector([factor * spacing for factor in self.radii]))


class AlphaMesher(Mesher):
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def build(self, cloud):
        alpha = self.alpha if self.alpha > 0 else 3.0 * mean_neighbor_spacing(cloud)
        return o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, alpha)


class GridMesher:
    """Triangulate an organized range image: each 2x2 block of neighbouring
    pixels gives two triangles, kept only if their three vertices are valid
    and the spread of their ranges is small (no 'curtains' between a
    foreground object and the background)."""

    def __init__(self, max_jump=0.05, max_relative_jump=0.03, wrap=True, sensor_origin=(0.0, 0.0, 0.0)):
        self.max_jump, self.max_relative_jump, self.wrap = max_jump, max_relative_jump, wrap
        self.sensor_origin = np.asarray(sensor_origin, dtype=np.float64)

    def build_grid(self, grid, valid):
        rows, cols = valid.shape
        ranges = np.linalg.norm(grid, axis=2).ravel()
        valid_flat = valid.ravel()
        index = np.arange(rows * cols).reshape(rows, cols)
        left_cols = np.arange(cols) if self.wrap else np.arange(cols - 1)   # 360 deg: last column next to first
        right_cols = (left_cols + 1) % cols
        top_left = index[:-1][:, left_cols].ravel()
        top_right = index[:-1][:, right_cols].ravel()
        bottom_left = index[1:][:, left_cols].ravel()
        bottom_right = index[1:][:, right_cols].ravel()
        triangles = np.concatenate([np.stack([top_left, bottom_left, top_right], axis=1),
                                    np.stack([top_right, bottom_left, bottom_right], axis=1)])
        vertex_ok = valid_flat[triangles].all(axis=1)
        triangle_ranges = ranges[triangles]
        spread = triangle_ranges.max(axis=1) - triangle_ranges.min(axis=1)
        allowed = np.maximum(self.max_jump, self.max_relative_jump * triangle_ranges.min(axis=1))
        triangles = triangles[vertex_ok & (spread <= allowed)]
        vertices = grid.reshape(-1, 3)
        triangles = orient_towards_sensor(vertices, triangles, self.sensor_origin)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                         o3d.utility.Vector3iVector(triangles.astype(np.int32)))
        mesh.remove_unreferenced_vertices()
        return mesh
