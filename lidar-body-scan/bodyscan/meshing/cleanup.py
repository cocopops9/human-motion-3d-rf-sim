"""Mesh cleaning, orientation and hole capping."""

from __future__ import annotations

from collections import deque

import numpy as np
import open3d as o3d

from bodyscan.log import info


def remove_small_fragments(mesh, fraction: float) -> None:
    """Drop connected pieces smaller than a fraction of the whole mesh (in place)."""
    if len(mesh.triangles) == 0:
        return
    labels, sizes, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    sizes = np.asarray(sizes)
    mesh.remove_triangles_by_mask(sizes[labels] < fraction * len(mesh.triangles))
    mesh.remove_unreferenced_vertices()


def clean_mesh(mesh, largest_component=False, min_component=0, target_triangles=0, drop_fragments=False):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if drop_fragments:                       # removing non-manifold edges can split off stray triangles
        remove_small_fragments(mesh, fraction=0.005)
    if largest_component or min_component > 0:
        labels, cluster_sizes, _ = mesh.cluster_connected_triangles()
        labels = np.asarray(labels)
        cluster_sizes = np.asarray(cluster_sizes)
        if largest_component:
            remove = labels != int(np.argmax(cluster_sizes))
        else:
            remove = cluster_sizes[labels] < min_component
        mesh.remove_triangles_by_mask(remove)
    mesh.remove_unreferenced_vertices()
    if target_triangles > 0 and len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    mesh.compute_vertex_normals()
    return mesh


def orient_towards_sensor(vertices, triangles, sensor_origin):
    """Flip triangles whose normal points away from the sensor."""
    v0, v1, v2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    to_sensor = sensor_origin - (v0 + v1 + v2) / 3.0
    facing_away = np.einsum("ij,ij->i", normals, to_sensor) < 0
    oriented = triangles.copy()
    oriented[facing_away, 1], oriented[facing_away, 2] = triangles[facing_away, 2], triangles[facing_away, 1]
    return oriented


def orient_consistently(triangles: np.ndarray) -> tuple[np.ndarray, int]:
    """Make the winding consistent across every edge shared by two triangles,
    piece by piece (breadth-first from the first triangle of each piece).
    Returns the re-wound triangles and the number of conflicts (edges where the
    surface cannot be oriented, e.g. where it touches itself); edges shared by
    more than two triangles are not crossed."""
    triangles = np.asarray(triangles, dtype=np.int64)
    count = len(triangles)
    if count == 0:
        return triangles.copy(), 0
    start = triangles.ravel()
    end = triangles[:, [1, 2, 0]].ravel()
    face = np.repeat(np.arange(count), 3)
    key = np.minimum(start, end) * (int(triangles.max()) + 1) + np.maximum(start, end)
    order = np.argsort(key, kind="stable")
    key, start, face = key[order], start[order], face[order]
    boundaries = np.flatnonzero(np.diff(key)) + 1
    first = np.concatenate([[0], boundaries])
    size = np.diff(np.concatenate([first, [len(key)]]))
    pairs = first[size == 2]
    a, b = face[pairs], face[pairs + 1]
    # Two triangles agree when they run along their shared edge in opposite directions.
    same_direction = start[pairs] == start[pairs + 1]
    neighbours = [[] for _ in range(count)]
    for f, g, flip in zip(a.tolist(), b.tolist(), same_direction.tolist()):
        neighbours[f].append((g, flip))
        neighbours[g].append((f, flip))
    flipped = np.zeros(count, dtype=bool)
    visited = np.zeros(count, dtype=bool)
    conflicts = 0
    for seed in range(count):
        if visited[seed]:
            continue
        visited[seed] = True
        queue = deque([seed])
        while queue:
            f = queue.popleft()
            for g, flip in neighbours[f]:
                wanted = flipped[f] ^ flip
                if not visited[g]:
                    visited[g] = True
                    flipped[g] = wanted
                    queue.append(g)
                elif flipped[g] != wanted:
                    conflicts += 1
    result = triangles.copy()
    result[flipped] = result[flipped][:, [0, 2, 1]]
    return result, conflicts // 2


def orient_by_cloud_normals(mesh, cloud):
    """Consistent winding over each connected piece, then each piece turned so
    that most of its triangles agree with the nearest point normals (Sionna RT
    uses the face normal to tell the two sides of a surface apart). Triangles
    are never flipped one by one: where the point normals are unreliable (thin
    parts, fingers) that would leave the winding inconsistent."""
    vertices = np.asarray(mesh.vertices)
    triangles, conflicts = orient_consistently(np.asarray(mesh.triangles))
    if len(triangles) == 0:
        return mesh
    if conflicts:
        info(f"orientation: {conflicts} edge(s) where the surface cannot be oriented consistently")
    v0, v1, v2 = (vertices[triangles[:, i]] for i in range(3))
    centroids = (v0 + v1 + v2) / 3.0
    index = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(np.asarray(cloud.points)))
    index.knn_index()
    nearest = index.knn_search(o3d.core.Tensor(centroids), 1)[0].numpy()[:, 0]
    cloud_normals = np.asarray(cloud.normals)
    agree = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), cloud_normals[nearest]) >= 0
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    labels = np.asarray(mesh.cluster_connected_triangles()[0])
    for label in np.unique(labels):
        piece = labels == label
        if np.mean(agree[piece]) < 0.5:
            triangles[piece] = triangles[piece][:, [0, 2, 1]]
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    return mesh


def boundary_loops(triangles):
    """Ordered boundary loops following the winding of the adjacent triangles;
    loops through a vertex with several outgoing boundary edges are skipped."""
    directed = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    edge_set = set(map(tuple, directed))
    boundary = [(a, b) for a, b in directed if (b, a) not in edge_set]
    following, ambiguous = {}, set()
    for a, b in boundary:
        if a in following:
            ambiguous.add(a)
        following[a] = b
    loops, visited = [], set()
    for start in following:
        if start in visited:
            continue
        loop, vertex, closed = [], start, False
        while vertex in following and vertex not in visited:
            visited.add(vertex)
            loop.append(vertex)
            vertex = following[vertex]
            if vertex == start:
                closed = True
                break
        if closed and len(loop) >= 3 and not ambiguous.intersection(loop):
            loops.append(loop)
    return loops


def cap_holes(mesh, max_size: float):
    """Close boundary loops no larger than max_size with a fan from their
    centroid (invented surface: flat for a planar hole, a shallow cone otherwise)."""
    vertices = np.asarray(mesh.vertices).copy()
    triangles = np.asarray(mesh.triangles)
    new_vertices, new_triangles = [], []
    for loop in boundary_loops(triangles):
        points = vertices[loop]
        if np.ptp(points, axis=0).max() > max_size:
            continue
        center_index = len(vertices) + len(new_vertices)
        new_vertices.append(points.mean(axis=0))
        for a, b in zip(loop, loop[1:] + loop[:1]):
            new_triangles.append([b, a, center_index])   # opposite winding to the boundary edge
    if new_vertices:
        mesh.vertices = o3d.utility.Vector3dVector(np.vstack([vertices, new_vertices]))
        mesh.triangles = o3d.utility.Vector3iVector(np.vstack([triangles, new_triangles]).astype(np.int32))
        info(f"capped {len(new_vertices)} hole(s) up to {max_size:g} m")
    return mesh
