"""Pieces shared by the avatar fit and the tracker: robust losses, closest
points on a mesh, visibility from the sensor, the silhouette distance map,
anatomical limits (PyTorch where a gradient is needed, numpy and Open3D
for the searches, which are recomputed between optimisation rounds)."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from bodyscan.body import skeleton


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------

def geman_mcclure(residual, scale: float):
    """Robust loss sigma^2 r^2 / (r^2 + sigma^2): quadratic for |r| << sigma,
    bounded by sigma^2 for outliers (a point that belongs to nothing on the
    body cannot pull it far)."""
    squared = residual * residual
    return scale * scale * squared / (squared + scale * scale)


def charbonnier(residual, scale: float):
    """sqrt(r^2 + s^2) - s: quadratic near zero, linear far away (keeps sharp
    events such as a heel strike while removing small jitter)."""
    import torch
    return torch.sqrt(residual * residual + scale * scale) - scale


def limit_penalty(body_pose):
    """Squared excess [rad^2] of the body joint rotations (..., 21, 3) beyond
    the anatomical limits of bodyscan.body.skeleton.LIMITS_DEG."""
    import torch
    lower, upper = skeleton.limit_arrays()
    lower = torch.as_tensor(lower[skeleton.BODY], dtype=body_pose.dtype, device=body_pose.device)
    upper = torch.as_tensor(upper[skeleton.BODY], dtype=body_pose.dtype, device=body_pose.device)
    return (torch.relu(lower - body_pose) ** 2 + torch.relu(body_pose - upper) ** 2).sum()


# ----------------------------------------------------------------------------
# Closest points on a triangle mesh
# ----------------------------------------------------------------------------

def closest_points(vertices: np.ndarray, faces: np.ndarray, queries: np.ndarray, face_subset=None):
    """For every query point the closest point on the mesh (or on the faces
    of face_subset): face index (into 'faces'), barycentric weights (N, 3)
    of its corners, and distance."""
    faces = np.asarray(faces)
    chosen = np.arange(len(faces)) if face_subset is None else np.asarray(face_subset)
    if len(chosen) == 0:
        n = len(queries)
        return np.zeros(n, dtype=np.int64), np.full((n, 3), 1 / 3), np.full(n, np.inf)
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(np.asarray(vertices, dtype=np.float32))
    mesh.triangle.indices = o3d.core.Tensor(faces[chosen].astype(np.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    answer = scene.compute_closest_points(o3d.core.Tensor(np.asarray(queries, dtype=np.float32)))
    local = answer["primitive_ids"].numpy().astype(np.int64)
    uv = answer["primitive_uvs"].numpy().astype(np.float64)
    bary = np.stack([1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], axis=1)
    points = answer["points"].numpy().astype(np.float64)
    distance = np.linalg.norm(points - queries, axis=1)
    return chosen[local], bary, distance


def surface_points(vertices, faces, face_ids, bary):
    """(..., N, 3) points on the faces 'face_ids' at barycentric 'bary' (torch, differentiable)."""
    corners = faces[face_ids]                                             # (N, 3) vertex indices
    return (vertices[..., corners[:, 0], :] * bary[:, 0:1] + vertices[..., corners[:, 1], :] * bary[:, 1:2]
            + vertices[..., corners[:, 2], :] * bary[:, 2:3])


def face_normals(vertices, faces, face_ids):
    """(..., N, 3) unit normals of the faces 'face_ids' (torch, differentiable)."""
    import torch
    corners = faces[face_ids]
    a = vertices[..., corners[:, 0], :]
    b = vertices[..., corners[:, 1], :]
    c = vertices[..., corners[:, 2], :]
    return torch.nn.functional.normalize(torch.cross(b - a, c - a, dim=-1), dim=-1, eps=1e-12)


def sample_surface(vertices: np.ndarray, faces: np.ndarray, count: int, rng=None):
    """'count' points uniformly on a triangle mesh (area-weighted), with the
    normals of their triangles; deterministic for a given generator."""
    rng = np.random.default_rng(0) if rng is None else rng
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    face = rng.choice(len(faces), count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    points = a[face] + u[:, None] * (b[face] - a[face]) + v[:, None] * (c[face] - a[face])
    normals = cross[face] / np.maximum(np.linalg.norm(cross[face], axis=1, keepdims=True), 1e-12)
    return points, normals


# ----------------------------------------------------------------------------
# Visibility from the sensor
# ----------------------------------------------------------------------------

def visible_vertices(vertices: np.ndarray, normals: np.ndarray, geometry, sensor_position: np.ndarray,
                     tolerance: float = 0.03, supersample: int = 2) -> np.ndarray:
    """Vertices the sensor can see: facing it, and not behind another part of
    the body. Occlusion: z-buffer of the vertex ranges on the range-image grid
    refined 'supersample' times (a vertex is hidden when another vertex of the
    same cell is more than 'tolerance' closer)."""
    towards = sensor_position - vertices
    facing = np.einsum("ij,ij->i", normals, towards) > 0.0
    row, column, distance = geometry.project(vertices)
    width = geometry.width * supersample
    r = np.round(row * supersample).astype(np.int64)
    c = np.mod(np.round(column * supersample).astype(np.int64), width)
    inside = (r >= 0) & (r < geometry.height * supersample)
    key = r * width + c
    order = np.lexsort((distance, key))
    first = np.ones(len(order), dtype=bool)
    first[1:] = key[order][1:] != key[order][:-1]
    nearest = np.empty(len(order))
    group = np.cumsum(first) - 1
    nearest_of_group = distance[order][first]
    nearest[order] = nearest_of_group[group]
    return facing & inside & (distance <= nearest + tolerance)


def visible_faces(faces: np.ndarray, visible: np.ndarray, minimum: int = 2) -> np.ndarray:
    """Indices of the faces with at least 'minimum' visible corners."""
    return np.flatnonzero(visible[faces].sum(axis=1) >= minimum)


# ----------------------------------------------------------------------------
# Silhouette distance map
# ----------------------------------------------------------------------------

def distance_map(mask: np.ndarray, row_step: float, column_step: float) -> np.ndarray:
    """Angular distance [rad] from every pixel to the nearest pixel of 'mask'
    (two-pass chamfer transform with the anisotropic steps of the image; the
    diagonal step is exact, the result within a few percent of Euclidean)."""
    big = 1e6
    d = np.where(mask, 0.0, big)
    h, w = d.shape
    diagonal = np.hypot(row_step, column_step)
    for r in range(h):                                                     # forward pass
        if r > 0:
            d[r] = np.minimum(d[r], d[r - 1] + row_step)
            d[r, 1:] = np.minimum(d[r, 1:], d[r - 1, :-1] + diagonal)
            d[r, :-1] = np.minimum(d[r, :-1], d[r - 1, 1:] + diagonal)
        for c in range(1, w):
            d[r, c] = min(d[r, c], d[r, c - 1] + column_step)
    for r in range(h - 1, -1, -1):                                         # backward pass
        if r < h - 1:
            d[r] = np.minimum(d[r], d[r + 1] + row_step)
            d[r, 1:] = np.minimum(d[r, 1:], d[r + 1, :-1] + diagonal)
            d[r, :-1] = np.minimum(d[r, :-1], d[r + 1, 1:] + diagonal)
        for c in range(w - 2, -1, -1):
            d[r, c] = min(d[r, c], d[r, c + 1] + column_step)
    return d


def sample_map(values, rows, columns):
    """Bilinear lookup of a (h, w) torch map at fractional (rows, columns),
    clamped to the map (the caller adds the distance outside it)."""
    import torch
    h, w = values.shape
    r = rows.clamp(0.0, float(h - 1))
    c = columns.clamp(0.0, float(w - 1))
    r0 = torch.floor(r).long().clamp(0, max(h - 2, 0))
    c0 = torch.floor(c).long().clamp(0, max(w - 2, 0))
    fr = (r - r0.to(r.dtype)).clamp(0.0, 1.0)
    fc = (c - c0.to(c.dtype)).clamp(0.0, 1.0)
    r1 = (r0 + 1).clamp(max=h - 1)
    c1 = (c0 + 1).clamp(max=w - 1)
    return (values[r0, c0] * (1 - fr) * (1 - fc) + values[r0, c1] * (1 - fr) * fc
            + values[r1, c0] * fr * (1 - fc) + values[r1, c1] * fr * fc)
