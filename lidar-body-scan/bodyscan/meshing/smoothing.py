"""Smoothing algorithms for radio ray tracing (used by 'bodyscan smooth', meshing.smoother).

Sionna RT reflects every ray on the plane of the triangle it hits, using the
triangle's own (face) normal: smooth vertex normals change nothing. A
triangle of edge L whose vertices carry a random error sigma along the
surface normal is tilted by about 2 sigma / L radians: 1 mm of noise on
5 mm triangles is 20 degrees of random tilt, so reflections scatter like
glitter instead of following the curvature of the body. Two levers:

    1. lower the noise (bilateral normal filtering, below);
    2. larger triangles where the body is flat and finer ones only where it
       curves (quadric decimation to a target edge length).

Bilateral normal filtering (Zheng et al., IEEE TVCG 2011): every face normal
becomes the average of the normals of the faces around it, weighted by area,
by distance (Gaussian, sigma = scale) and by the difference of the normals
(Gaussian, sigma = normal_sigma). With a small normal_sigma (0.35, about
20 deg) creases (between fingers, at the chin) are averaged only with their
own side; with a large one (1.5) the filter is nearly isotropic. The
vertices are then moved to agree with the filtered normals (Sun et al., IEEE
TVCG 2007). Vertices only move: a closed mesh stays closed.

Both steps act at a scale in millimetres, whatever the tessellation: every
face averages all the faces within 2 x scale (RadiusSearch; on fine meshes
the faces are grouped, FilterSources, to bound the cost). The vertex update
spreads a change by about one ring of triangles per pass, so its automatic
pass count grows as (reference edge / edge)^2 on finer grids, and as
(scale / 12 mm)^2 for scales above 12 mm, so that the vertices can follow a
normal field filtered over a wider area.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from bodyscan.geometry.neighbors import RadiusSearch
from bodyscan.log import info

SPEED_OF_LIGHT = 299_792_458.0
REFERENCE_EDGE_MM = 3.6             # median edge of the default 4 mm grid after the relaxation
REFERENCE_VERTEX_PASSES = 15        # vertex passes on that grid, for scales up to REFERENCE_SCALE_MM
REFERENCE_SCALE_MM = 12.0


def wavelength_mm(frequency_ghz: float) -> float:
    return SPEED_OF_LIGHT / (frequency_ghz * 1e9) * 1000.0


def median_edge_mm(vertices: np.ndarray, triangles: np.ndarray) -> float:
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    return 1000.0 * float(np.median(np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)))


def vertex_passes(requested: int, vertices: np.ndarray, triangles: np.ndarray, scale_mm: float = 6.0) -> int:
    """The requested number of vertex passes, or the automatic one: 15 on
    the default 4 mm grid (median edge 3.6 mm), times (3.6 mm / edge)^2 on
    other grids, times (scale / 12 mm)^2 for scales above 12 mm (measured on
    tt11: at 24 mm, 15 passes leave the surface in small dimples, facet noise
    p90 4.9 deg, while 60 passes follow the filtered normals)."""
    if requested > 0:
        return int(requested)
    ratio = REFERENCE_EDGE_MM / max(median_edge_mm(vertices, triangles), 1e-6)
    widening = max(1.0, scale_mm / REFERENCE_SCALE_MM) ** 2
    return int(np.clip(round(REFERENCE_VERTEX_PASSES * ratio ** 2 * widening), 3, 400))


def face_geometry(vertices: np.ndarray, triangles: np.ndarray):
    """Centroids, areas and unit normals of the faces."""
    v0, v1, v2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    length = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(length, 1e-30)[:, None]
    return (v0 + v1 + v2) / 3.0, 0.5 * length, normals


def unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-30)


class FilterSources:
    """What every face averages: the faces themselves, or, when the support
    of the filter holds more than 'max_sources' faces (fine meshes, large
    scales), groups of faces: the faces that share a cell of a grid sized so
    that the support holds about that many cells, and whose normals fall in
    the same bin of width 'normal_bin' (so the two sides of a crease or of a
    narrow gap stay separate sources). A group stands for its faces with the
    sum of their area-weighted normals. The cost per face is then bounded
    whatever the tessellation, and the average still covers the whole
    support with every face's contribution."""

    def __init__(self, centroids: np.ndarray, normals: np.ndarray, areas: np.ndarray, support: float,
                 normal_bin: float, max_sources: int = 96):
        expected = np.pi * support ** 2 / max(float(np.mean(areas)), 1e-30)
        self.cell = 0.0
        self.group_of_face = None
        self.groups = len(centroids)
        self.centroids = centroids
        if expected > max_sources:
            # a plane crosses on average 1.5 cells of side s per s^2 of its area
            self.cell = support * np.sqrt(1.5 * np.pi / max_sources)
            position = np.floor((centroids - centroids.min(axis=0)) / self.cell).astype(np.int64)
            direction = np.floor((normals + 1.0) / max(normal_bin, 0.05)).astype(np.int64)
            _, inverse = np.unique(np.hstack([position, direction]), axis=0, return_inverse=True)
            self.group_of_face = inverse.ravel()
            self.groups = int(self.group_of_face.max()) + 1
            area = self.sums(areas)
            self.centroids = np.stack([self.sums(areas * centroids[:, k]) for k in range(3)], axis=1) \
                / np.maximum(area, 1e-30)[:, None]

    def sums(self, values: np.ndarray) -> np.ndarray:
        return np.bincount(self.group_of_face, weights=values, minlength=self.groups)

    def vectors(self, normals: np.ndarray, areas: np.ndarray) -> np.ndarray:
        """Area-weighted normal of every source (a sum over its faces for a group)."""
        weighted = normals * areas[:, None]
        if self.group_of_face is None:
            return weighted
        return np.stack([self.sums(weighted[:, k]) for k in range(3)], axis=1)


def spatial_neighbourhoods(search: RadiusSearch, centroids: np.ndarray, scale: float):
    """Chunks of faces with their neighbourhoods in compact form: (faces,
    row of every entry, source index of every entry, spatial weight)."""
    for part, found, squared in search.chunks(centroids):
        valid = found >= 0
        rows = np.nonzero(valid)[0].astype(np.int32)
        spatial = np.exp(-squared[valid] / (2.0 * scale ** 2)).astype(np.float32)
        yield part, rows, found[valid].astype(np.int32), spatial


def bilateral_normals(centroids, normals, areas, scale, normal_sigma, iterations, max_sources=96,
                      max_cached=40_000_000):
    """Filtered face normals (see the module docstring): every face averages
    the area-weighted normals of all the sources within 2 x scale, weighted
    by distance (sigma = scale) and by the difference of the normals. The
    neighbourhoods are kept between passes when they hold at most
    'max_cached' entries (about 400 MB), and searched again otherwise."""
    support = 2.0 * scale
    sources = FilterSources(centroids, normals, areas, support, normal_sigma, max_sources)
    search = RadiusSearch(sources.centroids, support)
    cache, room = [], max_cached
    current = normals.copy()
    for iteration in range(iterations):
        vectors = sources.vectors(current, areas).astype(np.float32)
        directions = unit(vectors)
        filtered = current.copy()                        # a face without neighbours keeps its normal
        chunks = cache if iteration > 0 and cache is not None else spatial_neighbourhoods(search, centroids, scale)
        for part, rows, index, spatial in chunks:
            if iteration == 0 and cache is not None:
                room -= len(index)
                if room >= 0:
                    cache.append((part, rows, index, spatial))
                else:
                    cache = None                         # too large to keep: searched again at every pass
            own = current[part].astype(np.float32)
            difference = np.sum((directions[index] - own[rows]) ** 2, axis=1)
            weight = spatial * np.exp(-difference / np.float32(2.0 * normal_sigma ** 2))
            size = part.stop - part.start
            total = np.stack([np.bincount(rows, weights=weight * vectors[index, k], minlength=size) for k in range(3)],
                             axis=1)
            length = np.linalg.norm(total, axis=1)
            found = length > 1e-30
            filtered[part][found] = total[found] / length[found, None]
        current = filtered
    return current


def update_vertices(vertices, triangles, target_normals, iterations):
    """Move the vertices so that every face becomes perpendicular to its
    target normal: v += area-weighted mean over its faces of n (n . (c - v)).
    The area weights keep tiny (sliver) triangles, whose normals are poorly
    defined, from folding the surface."""
    vertices = vertices.copy()
    for _ in range(iterations):
        centroids, areas, _ = face_geometry(vertices, triangles)
        weight_sum = np.zeros(len(vertices))
        moves = np.zeros_like(vertices)
        for corner in range(3):
            v = vertices[triangles[:, corner]]
            step = target_normals * (areas * np.einsum("ij,ij->i", target_normals, centroids - v))[:, None]
            weight_sum += np.bincount(triangles[:, corner], weights=areas, minlength=len(vertices))
            for axis in range(3):
                moves[:, axis] += np.bincount(triangles[:, corner], weights=step[:, axis], minlength=len(vertices))
        vertices += moves / np.maximum(weight_sum, 1e-30)[:, None]
    return vertices


def vertex_normals(vertices, triangles):
    """Area-weighted vertex normals."""
    _, areas, normals = face_geometry(vertices, triangles)
    result = np.zeros_like(vertices)
    for corner in range(3):
        for axis in range(3):
            result[:, axis] += np.bincount(triangles[:, corner], weights=areas * normals[:, axis],
                                           minlength=len(vertices))
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-30)


def tangential_relaxation(vertices, triangles, iterations=5, factor=0.5):
    """Move every vertex towards the mean of its neighbours, within its
    tangent plane only: the triangles become better shaped (marching cubes
    leaves slivers whose normals are noise) while the surface stays where it is."""
    vertices = vertices.copy()
    edges = np.unique(np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]),
                              axis=1), axis=0)
    degree = np.bincount(edges.ravel(), minlength=len(vertices)).astype(np.float64)
    degree[degree == 0] = 1.0
    for _ in range(iterations):
        normals = vertex_normals(vertices, triangles)
        total = np.zeros_like(vertices)
        for axis in range(3):
            total[:, axis] = (np.bincount(edges[:, 0], weights=vertices[edges[:, 1], axis], minlength=len(vertices))
                              + np.bincount(edges[:, 1], weights=vertices[edges[:, 0], axis], minlength=len(vertices)))
        delta = total / degree[:, None] - vertices
        delta -= np.einsum("ij,ij->i", delta, normals)[:, None] * normals
        vertices += factor * delta
    return vertices


def bilateral_smooth(mesh, scale_m, normal_sigma=0.35, normal_iterations=4, vertex_iterations=0,
                     relax_iterations=5, verbose=True):
    """Tangential relaxation, bilateral normal filtering and vertex update of
    an Open3D mesh (returns a new mesh with the same triangles).
    vertex_iterations 0 = automatic (vertex_passes)."""
    vertices = np.asarray(mesh.vertices).copy()
    triangles = np.asarray(mesh.triangles).astype(np.int64)
    if relax_iterations > 0:
        vertices = tangential_relaxation(vertices, triangles, relax_iterations)
    centroids, areas, normals = face_geometry(vertices, triangles)
    filtered = bilateral_normals(centroids, normals, areas, scale_m, normal_sigma, normal_iterations)
    passes = vertex_passes(vertex_iterations, vertices, triangles, 1000.0 * scale_m)
    if vertex_iterations <= 0 and verbose:
        info(f"  {passes} vertex passes (automatic: median edge {median_edge_mm(vertices, triangles):.2f} mm, "
             f"scale {1000.0 * scale_m:g} mm)")
    vertices = update_vertices(vertices, triangles, filtered, passes)
    result = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), mesh.triangles)
    result.compute_vertex_normals()
    return result


def decimate_to_edge(mesh, target_edge_m):
    """Quadric decimation to the triangle count of a mesh with the target mean
    edge length (equilateral triangles); kept as it is if already coarser."""
    area = mesh.get_surface_area()
    target = int(area / (np.sqrt(3.0) / 4.0 * target_edge_m ** 2))
    if target <= 0 or target >= len(mesh.triangles):
        return mesh
    return mesh.simplify_quadric_decimation(target)
