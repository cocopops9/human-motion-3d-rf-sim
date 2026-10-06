"""
Convert a LiDAR point cloud into a triangle mesh.

Input formats
-------------
    .npz  from ouster_extract.py:
          frame_XXXXX.npz   -> organized cloud, key "xyz" with shape (H, W, 3)
          accumulated.npz   -> unorganized cloud, key "points" with shape (N, 3)
    .ply / .pcd / .xyz      -> unorganized cloud read with Open3D

Reconstruction methods
----------------------
    grid     Triangulates neighbouring pixels of the organized range image.
             Only valid for single frames (or per-pixel averaged frames) that
             keep the (H, W) sensor structure. Fast, needs no normals, and does
             not invent surface where the sensor saw nothing. Triangles that
             span a depth discontinuity are discarded.
    poisson  Screened Poisson reconstruction; needs well oriented normals.
             Surface farther than --trim-distance mean point spacings from the
             data is removed. A cloud that encloses the object therefore gives
             a closed mesh, and a partial cloud gives only the observed surface.
    bpa      Ball pivoting. Interpolates the points exactly; leaves holes where
             sampling is sparse.
    alpha    Alpha shape. Simple, but sensitive to the choice of alpha.

The default is "grid" for organized input and "poisson" otherwise.

Units are meters throughout, which is also what Sionna RT expects.

Examples
--------
    Mesh one frame, keep only a box around the object:
        python pointcloud_to_mesh.py frames/frame_00010.npz --out object.ply \
            --crop-min 1.0 -0.5 -1.0 --crop-max 2.5 0.5 1.0

    Poisson on a cloud fused by fuse_views.py (its normals are reused), with
    the unseen base capped to get a closed mesh:
        python pointcloud_to_mesh.py fused.ply --method poisson --fill-holes 0.3 --out object.ply

    Closed manifold mesh for Sionna RT from fuse_person.py output (floor at
    z = 0): untrimmed Poisson, rebuilt by marching cubes on a 4 mm signed
    distance grid, cut flat at the floor, decimated:
        python pointcloud_to_mesh.py person.ply --method poisson --depth 9 --trim-distance 0 \
            --watertight 0.004 --clip-below 0 --target-triangles 100000 --out person_mesh.ply
    Everything the sensor never saw (top of the head, soles, armpits) is
    Poisson's smooth guess, not a measurement.
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import open3d as o3d

VERSION = "2026-10-02a (generic: any point cloud in, mesh out; no scipy or scikit-image)"


# ----------------------------------------------------------------------------
# Loading and cropping
# ----------------------------------------------------------------------------

class PointCloudInput:
    """Point cloud plus, when available, its organized (H, W) structure."""

    def __init__(self, points, grid=None, valid=None, normals=None):
        self.points = points      # (N, 3) valid points only
        self.grid = grid          # (H, W, 3) or None
        self.valid = valid        # (H, W) bool or None
        self.normals = normals    # (N, 3) or None, e.g. from fuse_views.py

    @property
    def is_organized(self):
        return self.grid is not None


def load_point_cloud(path):
    path = Path(path)

    if path.suffix == ".npz":
        data = np.load(path)
        if "xyz" in data.files and data["xyz"].ndim == 3:
            grid = data["xyz"].astype(np.float64)
            valid = np.linalg.norm(grid, axis=2) > 1e-6
            return PointCloudInput(grid[valid], grid, valid)
        if "points" in data.files:
            return PointCloudInput(data["points"].astype(np.float64))
        raise ValueError(f"{path} has neither an 'xyz' grid nor a 'points' array")

    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"could not read any point from {path}")
    normals = np.asarray(cloud.normals) if cloud.has_normals() else None
    return PointCloudInput(np.asarray(cloud.points), normals=normals)


def inside_box(points, box_min, box_max):
    return np.all((points >= box_min) & (points <= box_max), axis=-1)


def crop_input(cloud_input, box_min, box_max):
    box_min = np.asarray(box_min, dtype=np.float64)
    box_max = np.asarray(box_max, dtype=np.float64)

    if cloud_input.is_organized:
        valid = cloud_input.valid & inside_box(cloud_input.grid, box_min, box_max)
        return PointCloudInput(cloud_input.grid[valid], cloud_input.grid, valid)

    keep = inside_box(cloud_input.points, box_min, box_max)
    normals = cloud_input.normals[keep] if cloud_input.normals is not None else None
    return PointCloudInput(cloud_input.points[keep], normals=normals)


# ----------------------------------------------------------------------------
# Preprocessing for unorganized methods
# ----------------------------------------------------------------------------

def preprocess_cloud(points, args, normals=None):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)   # kept by the filters below

    if args.voxel > 0:
        cloud = cloud.voxel_down_sample(args.voxel)

    if args.outlier_neighbors > 0:
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=args.outlier_neighbors, std_ratio=args.outlier_std)

    if args.remove_plane:
        _, plane_indices = cloud.segment_plane(
            distance_threshold=args.plane_threshold, ransac_n=3, num_iterations=1000)
        cloud = cloud.select_by_index(plane_indices, invert=True)

    return cloud


def mean_neighbor_spacing(cloud):
    distances = np.asarray(cloud.compute_nearest_neighbor_distance())
    return float(np.mean(distances))


def estimate_normals(cloud, args):
    """Estimate normals and orient them.

    "consistent" (default) propagates orientation over a neighbourhood graph
    and then makes the normals point away from the cloud centroid. This is
    right for an object scanned from all around, e.g. on the turntable.
    "sensor" points every normal towards the sensor origin; use it only for a
    single view, where it is exact. On a fused cloud it flips the far side.
    "input" keeps the normals stored in the file; fuse_views.py writes them
    oriented towards the sensor in each view, which is exact for every view.
    "auto" (default) is "input" when the file has normals, else "consistent".
    """
    orient = args.orient
    if orient == "auto":
        orient = "input" if cloud.has_normals() else "consistent"
    if orient == "input":
        if not cloud.has_normals():
            raise ValueError("--orient input needs a point cloud file with normals")
        cloud.normalize_normals()
        return cloud

    spacing = mean_neighbor_spacing(cloud)
    radius = args.normal_radius if args.normal_radius > 0 else 4.0 * spacing
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    if orient == "sensor":
        cloud.orient_normals_towards_camera_location(np.asarray(args.sensor_origin))
        return cloud

    cloud.orient_normals_consistent_tangent_plane(k=15)
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    outward = np.einsum("ij,ij->i", normals, points - points.mean(axis=0))
    if np.mean(outward) < 0:
        cloud.normals = o3d.utility.Vector3dVector(-normals)
    return cloud


def orient_by_cloud_normals(mesh, cloud):
    """Flip each triangle so that it agrees with the nearest point normal.

    Alpha shapes and, occasionally, ball pivoting return triangles with
    arbitrary winding. Sionna RT uses the face normal to tell the two sides of
    a surface apart, so the winding must be consistent with the estimated
    outward normals.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        return mesh

    v0, v1, v2 = (vertices[triangles[:, i]] for i in range(3))
    face_normals = np.cross(v1 - v0, v2 - v0)
    centroids = (v0 + v1 + v2) / 3.0

    tree = o3d.geometry.KDTreeFlann(cloud)
    cloud_normals = np.asarray(cloud.normals)
    nearest = np.array([tree.search_knn_vector_3d(c, 1)[1][0] for c in centroids])
    disagree = np.einsum("ij,ij->i", face_normals, cloud_normals[nearest]) < 0

    oriented = triangles.copy()
    oriented[disagree, 1], oriented[disagree, 2] = triangles[disagree, 2], triangles[disagree, 1]
    mesh.triangles = o3d.utility.Vector3iVector(oriented)

    # Per-face decisions can disagree with their neighbours where the point
    # normals are noisy. Make the winding consistent over each connected
    # piece, then flip whole pieces that mostly disagree with the points.
    mesh.orient_triangles()
    triangles = np.asarray(mesh.triangles).copy()
    v0, v1, v2 = (vertices[triangles[:, i]] for i in range(3))
    agree = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), cloud_normals[nearest]) >= 0
    labels = np.asarray(mesh.cluster_connected_triangles()[0])
    for label in np.unique(labels):
        piece = labels == label
        if np.mean(agree[piece]) < 0.5:
            triangles[piece] = triangles[piece][:, [0, 2, 1]]
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


# ----------------------------------------------------------------------------
# Reconstruction methods
# ----------------------------------------------------------------------------

def mesh_from_grid(grid, valid, args):
    """Triangulate an organized range image.

    Each 2x2 block of neighbouring pixels gives two triangles. A triangle is
    kept only if its three vertices are valid and the spread of their ranges
    is small, which removes the "curtains" that would otherwise connect a
    foreground object to the background behind it.
    """
    rows, cols = valid.shape
    ranges = np.linalg.norm(grid, axis=2).ravel()
    valid_flat = valid.ravel()
    index = np.arange(rows * cols).reshape(rows, cols)

    # Ouster columns cover 360 degrees, so the last column neighbours the first.
    left_cols = np.arange(cols) if args.wrap else np.arange(cols - 1)
    right_cols = (left_cols + 1) % cols

    top_left = index[:-1][:, left_cols].ravel()
    top_right = index[:-1][:, right_cols].ravel()
    bottom_left = index[1:][:, left_cols].ravel()
    bottom_right = index[1:][:, right_cols].ravel()

    triangles = np.concatenate([
        np.stack([top_left, bottom_left, top_right], axis=1),
        np.stack([top_right, bottom_left, bottom_right], axis=1),
    ])

    vertex_ok = valid_flat[triangles].all(axis=1)
    triangle_ranges = ranges[triangles]
    spread = triangle_ranges.max(axis=1) - triangle_ranges.min(axis=1)
    allowed = np.maximum(args.max_jump, args.max_relative_jump * triangle_ranges.min(axis=1))
    triangles = triangles[vertex_ok & (spread <= allowed)]

    vertices = grid.reshape(-1, 3)
    triangles = orient_towards_sensor(vertices, triangles, np.asarray(args.sensor_origin))

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles.astype(np.int32)))
    mesh.remove_unreferenced_vertices()
    return mesh


def orient_towards_sensor(vertices, triangles, sensor_origin):
    """Flip triangles whose normal points away from the sensor."""
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    to_sensor = sensor_origin - (v0 + v1 + v2) / 3.0
    facing_away = np.einsum("ij,ij->i", normals, to_sensor) < 0

    oriented = triangles.copy()
    oriented[facing_away, 1], oriented[facing_away, 2] = (
        triangles[facing_away, 2], triangles[facing_away, 1])
    return oriented


def remove_small_fragments(mesh, fraction):
    """Drop connected pieces smaller than a fraction of the whole mesh.

    Trimming can leave tiny islands (a few triangles) that each add boundary
    edges without representing anything measured.
    """
    if len(mesh.triangles) == 0:
        return
    labels, sizes, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    sizes = np.asarray(sizes)
    mesh.remove_triangles_by_mask(sizes[labels] < fraction * len(mesh.triangles))
    mesh.remove_unreferenced_vertices()


def mesh_poisson(cloud, args):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=args.depth)
    densities = np.asarray(densities)

    # Poisson closes the surface everywhere, including where nothing was
    # measured. Vertices far from every input point are extrapolation, so they
    # are removed. If the cloud encloses the object no vertex is far from the
    # data and the mesh stays closed; if the cloud covers only part of the
    # surface, the invented remainder is cut away and an open surface is left.
    if args.trim_distance > 0:
        limit = args.trim_distance * mean_neighbor_spacing(cloud)
        mesh_vertices = o3d.geometry.PointCloud(mesh.vertices)
        distance_to_data = np.asarray(mesh_vertices.compute_point_cloud_distance(cloud))
        mesh.remove_vertices_by_mask(distance_to_data > limit)
        remove_small_fragments(mesh, fraction=0.005)

    # Optional, less principled criterion based on the Poisson density.
    if args.density_quantile > 0:
        threshold = np.quantile(densities, args.density_quantile)
        mesh.remove_vertices_by_mask(densities < threshold)

    # With --trim-distance 0 the closed Poisson surface is kept as it is
    # (watertight, unseen parts invented). Otherwise also clip to the cloud's
    # bounds, since Poisson can extend beyond the data.
    if args.trim_distance <= 0:
        return mesh
    bounds = cloud.get_axis_aligned_bounding_box()
    margin = 2.0 * mean_neighbor_spacing(cloud)
    bounds = o3d.geometry.AxisAlignedBoundingBox(
        bounds.min_bound - margin, bounds.max_bound + margin)
    return mesh.crop(bounds)


def mesh_ball_pivoting(cloud, args):
    spacing = mean_neighbor_spacing(cloud)
    radii = [factor * spacing for factor in args.bpa_radii]
    return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        cloud, o3d.utility.DoubleVector(radii))


def mesh_alpha_shape(cloud, args):
    alpha = args.alpha if args.alpha > 0 else 3.0 * mean_neighbor_spacing(cloud)
    return o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, alpha)


# ----------------------------------------------------------------------------
# Cleaning and evaluation
# ----------------------------------------------------------------------------

def clean_mesh(mesh, args, drop_fragments=False):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Removing non-manifold edges can split off a few stray triangles.
    if drop_fragments:
        remove_small_fragments(mesh, fraction=0.005)

    if args.largest_component or args.min_component > 0:
        labels, cluster_sizes, _ = mesh.cluster_connected_triangles()
        labels = np.asarray(labels)
        cluster_sizes = np.asarray(cluster_sizes)
        if args.largest_component:
            remove = labels != int(np.argmax(cluster_sizes))
        else:
            remove = cluster_sizes[labels] < args.min_component
        mesh.remove_triangles_by_mask(remove)

    mesh.remove_unreferenced_vertices()

    if args.target_triangles > 0 and len(mesh.triangles) > args.target_triangles:
        mesh = mesh.simplify_quadric_decimation(args.target_triangles)

    mesh.compute_vertex_normals()
    return mesh


def dilate(mask, iterations):
    """Binary dilation with the 6-neighbourhood (numpy only)."""
    mask = mask.copy()
    for _ in range(iterations):
        grown = mask.copy()
        grown[1:] |= mask[:-1]
        grown[:-1] |= mask[1:]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        grown[:, :, 1:] |= mask[:, :, :-1]
        grown[:, :, :-1] |= mask[:, :, 1:]
        mask = grown
    return mask


# Cube corners, edges and faces (corners of a face in cyclic order).
CUBE_CORNERS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
CUBE_EDGES = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                       (0, 4), (1, 5), (2, 6), (3, 7)])
CUBE_FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (3, 2, 6, 7), (0, 3, 7, 4), (1, 2, 6, 5)]
FACE_NORMALS = np.array([[0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]], dtype=np.float64)


def midpoint_of(edge):
    return (CUBE_CORNERS[CUBE_EDGES[edge][0]] + CUBE_CORNERS[CUBE_EDGES[edge][1]]) / 2.0


def cube_cases():
    """Iso-surface triangles in a cube for each of the 256 inside/outside
    patterns of its corners. Vertex ids 0 to 11 are the crossing points on
    the cube edges, 12 and up the centres of the loops that need one.

    Built rather than tabulated: on every face the crossing points are
    joined into segments; a face with all four edges crossed (inside corners
    on a diagonal) is resolved by cutting off each inside corner, a rule that
    depends on the face alone, so the two cubes sharing a face agree and the
    surface is closed. The segments form closed loops (every crossing lies on
    two faces of the cube). A loop is triangulated as a fan unless two of its
    non-consecutive points lie on one face: the neighbouring cube may then
    use the same chord, and the chord would belong to four triangles; such a
    loop gets a centre point and a star of triangles instead.
    Returns, per pattern, (triangles, loops that have a centre)."""
    faces_of_edge = [{f for f, face in enumerate(CUBE_FACES)
                      if set(edge) <= set(face)} for edge in CUBE_EDGES.tolist()]
    edge_of = {frozenset(e): k for k, e in enumerate(CUBE_EDGES.tolist())}
    cases = []
    for pattern in range(256):
        inside = [bool(pattern >> c & 1) for c in range(8)]
        neighbours, outward_of = {}, {}
        for f, face in enumerate(CUBE_FACES):
            edges = [edge_of[frozenset((face[k], face[(k + 1) % 4]))] for k in range(4)]
            crossed = [k for k in range(4) if inside[face[k]] != inside[face[(k + 1) % 4]]]
            segments = []
            if len(crossed) == 2:
                inner = [CUBE_CORNERS[c] for c in face if inside[c]]
                outer = [CUBE_CORNERS[c] for c in face if not inside[c]]
                towards = np.mean(outer, axis=0) - np.mean(inner, axis=0)
                segments = [(edges[crossed[0]], edges[crossed[1]], towards)]
            elif len(crossed) == 4:
                for k in range(4):
                    if inside[face[k]]:
                        a, b = edges[(k - 1) % 4], edges[k]
                        towards = (midpoint_of(a) + midpoint_of(b)) / 2 - CUBE_CORNERS[face[k]]
                        segments.append((a, b, towards))
            for a, b, towards in segments:
                neighbours.setdefault(a, []).append(b)
                neighbours.setdefault(b, []).append(a)
                outward_of[frozenset((a, b))] = (f, towards)
        triangles, centred, seen = [], [], set()
        for start in sorted(neighbours):
            if start in seen:
                continue
            loop, previous, current = [start], None, start
            seen.add(start)
            while True:
                following = [n for n in neighbours[current] if n != previous][0]
                if following == start:
                    break
                loop.append(following)
                seen.add(following)
                previous, current = current, following
            # Direction: seen from outside, the loop runs counterclockwise.
            # On the face of its first segment, the inside of the surface
            # patch lies towards the cube (-face normal), and the outside of
            # the body towards 'towards'; the face's other cube makes the
            # opposite choice, so the orientation is consistent.
            f, towards = outward_of[frozenset((loop[0], loop[1]))]
            step = midpoint_of(loop[1]) - midpoint_of(loop[0])
            if np.cross(towards, step) @ (-FACE_NORMALS[f]) < 0:
                loop = [loop[0]] + loop[1:][::-1]
            count = len(loop)
            chord_on_face = any(faces_of_edge[loop[a]] & faces_of_edge[loop[b]]
                                for a in range(count) for b in range(a + 2, count)
                                if not (a == 0 and b == count - 1))
            if count == 3 or not chord_on_face:
                triangles += [(loop[0], loop[k], loop[k + 1]) for k in range(1, count - 1)]
            else:
                centre = 12 + len(centred)
                centred.append(loop)
                triangles += [(centre, loop[k], loop[(k + 1) % count]) for k in range(count)]
        cases.append((triangles, centred))
    return cases


CASES = cube_cases()
MAX_TRIANGLES = max(len(t) for t, _ in CASES)
MAX_CENTRES = max(len(c) for _, c in CASES)


def marching_cubes(sdf, voxel):
    """Zero level set of a sampled field as a closed, edge-manifold triangle
    mesh (numpy only), every triangle oriented from the negative (inside) to
    the positive side (the orientation is part of the case table). Returns vertices (grid index times voxel) and triangles."""
    shape = np.array(sdf.shape)
    negative = sdf < 0
    pattern = np.zeros(shape - 1, dtype=np.int32)
    for c, corner in enumerate(CUBE_CORNERS):
        pattern |= negative[corner[0]:shape[0] - 1 + corner[0], corner[1]:shape[1] - 1 + corner[1],
                            corner[2]:shape[2] - 1 + corner[2]].astype(np.int32) << c
    cubes = np.argwhere((pattern > 0) & (pattern < 255))
    case = pattern[cubes[:, 0], cubes[:, 1], cubes[:, 2]]
    strides = np.array([shape[1] * shape[2], shape[2], 1], dtype=np.int64)
    corner_index = (cubes[:, None, :] + CUBE_CORNERS[None, :, :]) @ strides           # (cubes, 8)
    flat = sdf.ravel()

    # Crossing points on the 12 edges of every active cube, shared between
    # cubes through the key of the grid edge.
    ends_a, ends_b = corner_index[:, CUBE_EDGES[:, 0]], corner_index[:, CUBE_EDGES[:, 1]]
    crossed = negative.ravel()[ends_a] != negative.ravel()[ends_b]
    keys = np.where(crossed, np.minimum(ends_a, ends_b) * flat.size + np.maximum(ends_a, ends_b), -1)
    unique_keys, inverse = np.unique(keys[crossed], return_inverse=True)
    a, b = unique_keys // flat.size, unique_keys % flat.size
    t = flat[a] / (flat[a] - flat[b])
    pa = np.stack(np.unravel_index(a, sdf.shape), axis=-1).astype(np.float64)
    pb = np.stack(np.unravel_index(b, sdf.shape), axis=-1).astype(np.float64)
    vertices = [pa + t[:, None] * (pb - pa)]
    vertex_of = np.full((len(cubes), 12 + MAX_CENTRES), -1, dtype=np.int64)
    vertex_of[:, :12][crossed] = inverse

    # Loop centres, where a case needs them: the mean of the loop's points.
    centre_count = len(unique_keys)
    for k, (_, centred) in enumerate(CASES):
        if not centred:
            continue
        members = np.flatnonzero(case == k)
        for c, loop in enumerate(centred):
            centres = vertices[0][vertex_of[members][:, loop]].mean(axis=1)
            vertex_of[members, 12 + c] = centre_count + np.arange(len(members))
            centre_count += len(members)
            vertices.append(centres)
    vertices = np.concatenate(vertices)

    counts = np.array([len(t) for t, _ in CASES])
    table = np.zeros((256, MAX_TRIANGLES, 3), dtype=np.int64)
    for k, (triangles, _) in enumerate(CASES):
        if triangles:
            table[k, :len(triangles)] = triangles
    owners = np.repeat(np.arange(len(cubes)), counts[case])
    slot = np.arange(len(owners)) - np.repeat(np.cumsum(counts[case]) - counts[case], counts[case])
    local = table[case[owners], slot]
    faces = vertex_of[owners[:, None], local]

    return vertices * voxel, faces


def watertight_remesh(mesh, voxel, clip_below=None, min_fraction=0.01):
    """Rebuild the mesh as the zero level set of a signed distance field.

    Marching cubes on a sampled field, padded on every side, yields a
    closed, edge-manifold, consistently oriented surface; Poisson meshes from
    Open3D can contain non-manifold edges and self-intersections where two
    parts nearly touch (arm against torso). The sign of the field comes from
    a flood fill of the free space from the grid border; near the surface,
    where the flood fill cannot decide, from the normal of the closest
    triangle. Holes narrower than about 1.5 voxels are sealed; details
    smaller than a voxel are lost. Components smaller than 'min_fraction' of
    the surface (bubbles) are dropped.
    """
    mesh = copy.deepcopy(mesh)
    mesh.compute_triangle_normals()
    triangle_normals = np.asarray(mesh.triangle_normals)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    lower = np.asarray(mesh.get_min_bound()) - 4 * voxel
    upper = np.asarray(mesh.get_max_bound()) + 4 * voxel
    shape = np.ceil((upper - lower) / voxel).astype(int) + 1

    # Narrow band: voxels touched by a dense sampling of the surface, dilated.
    # Exact distances are needed only there; elsewhere only the sign matters.
    samples = int(mesh.get_surface_area() / (0.4 * voxel) ** 2) + 1000
    surface = np.asarray(mesh.sample_points_uniformly(samples).points)
    cells = np.clip(np.round((surface - lower) / voxel).astype(int), 0, shape - 1)
    band = np.zeros(shape, dtype=bool)
    band[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    band = dilate(band, 2)

    index = np.argwhere(band)
    query = (lower + voxel * index).astype(np.float32)
    answer = scene.compute_closest_points(o3d.core.Tensor(query))
    offset = query - answer["points"].numpy()
    normals = triangle_normals[answer["primitive_ids"].numpy()]
    band_distance = np.linalg.norm(offset, axis=1)
    band_side = np.sign(np.einsum("ij,ij->i", offset, normals))

    # Away from the surface only the sign matters. Ray parity decides it, by
    # majority over several random rays, which tolerates small defects such
    # as an open sole or a few non-manifold edges; a flood fill would leak
    # through them. Evaluated once per connected block of voxels would be
    # faster but equally leaky, hence per voxel, in chunks.
    far = np.argwhere(~band)
    sdf = np.empty(shape, dtype=np.float32)
    for start in range(0, len(far), 2_000_000):
        chunk = far[start:start + 2_000_000]
        points = (lower + voxel * chunk).astype(np.float32)
        occupancy = scene.compute_occupancy(o3d.core.Tensor(points), nsamples=5).numpy() > 0.5
        sdf[chunk[:, 0], chunk[:, 1], chunk[:, 2]] = np.where(occupancy, -3.0 * voxel, 3.0 * voxel)
    sdf[index[:, 0], index[:, 1], index[:, 2]] = band_side * band_distance
    if not np.any(sdf < 0):
        raise SystemExit("watertight remesh: no interior found (the surface has large holes); "
                         "use --trim-distance 0 or --fill-holes first")

    # A sample exactly on the level set would put several vertices at one
    # point and make degenerate triangles; keep every sample off zero.
    if clip_below is not None:
        # Intersection with the half-space z >= clip_below: a flat, closed cut
        # (e.g. the floor under a standing person, where Poisson invents a bulge).
        heights = lower[2] + voxel * np.arange(shape[2])
        sdf = np.maximum(sdf, (clip_below - heights)[None, None, :]).astype(np.float32)
    sdf[np.abs(sdf) < 1e-7 * voxel] = 1e-7 * voxel
    # The grid border is outside by definition, so the surface cannot run
    # into it (an open Poisson sole would otherwise leave a hole there).
    for axis in range(3):
        sdf[(slice(None),) * axis + (0,)] = 3.0 * voxel
        sdf[(slice(None),) * axis + (-1,)] = 3.0 * voxel
    vertices, faces = marching_cubes(sdf, voxel)
    result = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices + lower),
                                       o3d.utility.Vector3iVector(faces))
    labels, sizes, _ = result.cluster_connected_triangles()
    labels, sizes = np.asarray(labels), np.asarray(sizes)
    result.remove_triangles_by_mask(sizes[labels] < min_fraction * len(result.triangles))
    result.remove_unreferenced_vertices()

    # Outward orientation: the signed volume of a closed surface is positive.
    v, f = np.asarray(result.vertices), np.asarray(result.triangles)
    if np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() < 0:
        result.triangles = o3d.utility.Vector3iVector(f[:, ::-1].copy())
    return result


def boundary_loops(triangles):
    """Ordered boundary loops, following the winding of the adjacent triangles.

    A directed edge (a, b) of a triangle is on the boundary if no triangle
    contains (b, a). Loops through a vertex with several outgoing boundary
    edges are ambiguous and skipped.
    """
    directed = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    edge_set = set(map(tuple, directed))
    boundary = [(a, b) for a, b in directed if (b, a) not in edge_set]

    following = {}
    ambiguous = set()
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


def cap_holes(mesh, max_size):
    """Close boundary loops no larger than max_size with a fan from their centroid.

    Meant for holes where the object cannot be measured, such as the base
    standing on the turntable. Each cap is invented surface: flat for a
    planar hole, a shallow cone otherwise. The fan is correct for holes that
    are star-shaped around their centroid, which covers typical bases.
    """
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
        print(f"capped {len(new_vertices)} hole(s) up to {max_size:g} m")
    return mesh


def point_to_mesh_distances(points, mesh, max_points=200_000, seed=0):
    """Unsigned distance from each input point to the mesh surface, in meters."""
    if len(points) > max_points:
        chosen = np.random.default_rng(seed).choice(len(points), max_points, replace=False)
        points = points[chosen]

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query = o3d.core.Tensor(points.astype(np.float32))
    return scene.compute_distance(query).numpy()


def report(mesh, input_points, coverage_threshold_mm, check_self_intersection=False):
    """Print mesh topology and how well the mesh covers the input points.

    The distance statistics measure agreement with the data, not accuracy:
    grid, ball pivoting and alpha meshes use the input points as vertices, so
    their median distance is zero by construction. The useful number is the
    fraction of points farther than the threshold, i.e. data the mesh does
    not represent (dropped silhouettes, trimmed regions, removed outliers).
    Accuracy needs a reference, such as an object of known dimensions.
    """
    # "Closed" means no boundary edges, i.e. the surface encloses a volume.
    # Open3D's is_watertight() additionally rejects any self-intersection;
    # Poisson meshes often contain a few tiny numerical folds, and that test
    # is slow (tens of seconds for 1e5 triangles), so it is optional here.
    boundary_edges = len(mesh.get_non_manifold_edges(allow_boundary_edges=False))
    print(f"vertices:        {len(mesh.vertices)}")
    print(f"triangles:       {len(mesh.triangles)}")
    print(f"closed:          {boundary_edges == 0}  ({boundary_edges} boundary or non-manifold edges)")
    print(f"vertex manifold: {mesh.is_vertex_manifold()}")
    if check_self_intersection:
        pairs = len(mesh.get_self_intersecting_triangles())
        print(f"self-intersecting triangle pairs: {pairs}")

    if len(mesh.triangles) == 0:
        return

    distances_mm = 1000.0 * point_to_mesh_distances(input_points, mesh)
    uncovered = 100.0 * np.mean(distances_mm > coverage_threshold_mm)
    print(f"points farther than {coverage_threshold_mm:g} mm from the mesh: {uncovered:.2f}%")
    print("point to surface distance [mm]: "
          f"mean {distances_mm.mean():.2f}  "
          f"rms {np.sqrt(np.mean(distances_mm ** 2)):.2f}  "
          f"p95 {np.percentile(distances_mm, 95):.2f}")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def denoise_by_plane_projection(cloud, radius, iterations=2):
    """Move every point onto the plane fitted to its neighbours within 'radius'.

    Clouds fused from many LiDAR views are layered: each view samples the
    surface on its own scan rows (2 cm apart vertically at 1.8 m), and small
    residual misalignments leave those rows at slightly different depths.
    Poisson then reproduces them as horizontal ridges. Projecting each point
    onto the local least-squares plane (Gaussian weights) removes the
    layering and the random range noise; features smaller than about the
    radius are flattened as well. Normals keep their original orientation.
    """
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


def build_mesh(cloud_input, args):
    method = args.method
    if method == "auto":
        method = "grid" if cloud_input.is_organized else "poisson"

    if method == "grid":
        if not cloud_input.is_organized:
            raise ValueError("the grid method needs an organized frame (.npz with 'xyz')")
        return method, mesh_from_grid(cloud_input.grid, cloud_input.valid, args), None

    cloud = preprocess_cloud(cloud_input.points, args, cloud_input.normals)
    cloud = estimate_normals(cloud, args)
    if args.denoise > 0:
        cloud = denoise_by_plane_projection(cloud, args.denoise)
    builders = {"poisson": mesh_poisson, "bpa": mesh_ball_pivoting, "alpha": mesh_alpha_shape}
    mesh = builders[method](cloud, args)
    return method, mesh, cloud


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert a LiDAR point cloud into a triangle mesh.")
    parser.add_argument("input", help=".npz from ouster_extract.py, or .ply / .pcd / .xyz")
    parser.add_argument("--out", default="mesh.ply", help="output mesh (.ply or .obj)")
    parser.add_argument("--method", default="auto", choices=["auto", "grid", "poisson", "bpa", "alpha"])

    region = parser.add_argument_group("region of interest")
    region.add_argument("--crop-min", type=float, nargs=3, metavar=("X", "Y", "Z"))
    region.add_argument("--crop-max", type=float, nargs=3, metavar=("X", "Y", "Z"))
    region.add_argument("--sensor-origin", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar=("X", "Y", "Z"), help="sensor position in the cloud's frame")

    grid = parser.add_argument_group("grid method")
    grid.add_argument("--max-jump", type=float, default=0.05,
                      help="absolute range spread allowed inside a triangle [m]")
    grid.add_argument("--max-relative-jump", type=float, default=0.03,
                      help="range spread allowed as a fraction of the range")
    grid.add_argument("--no-wrap", dest="wrap", action="store_false",
                      help="do not connect the last column to the first")

    cloud = parser.add_argument_group("preprocessing for poisson / bpa / alpha")
    cloud.add_argument("--voxel", type=float, default=0.0, help="voxel downsampling size [m], 0 = off")
    cloud.add_argument("--outlier-neighbors", type=int, default=20, help="0 disables outlier removal")
    cloud.add_argument("--outlier-std", type=float, default=2.0)
    cloud.add_argument("--remove-plane", action="store_true", help="remove the dominant plane (floor)")
    cloud.add_argument("--plane-threshold", type=float, default=0.01, help="[m]")
    cloud.add_argument("--normal-radius", type=float, default=0.0, help="[m], 0 = automatic")
    cloud.add_argument("--orient", default="auto", choices=["auto", "input", "consistent", "sensor"],
                       help="'input': normals stored in the file (fuse_views.py output); "
                            "'consistent': clouds fused from several views without normals; "
                            "'sensor': a single view; 'auto': input if available, else consistent")

    method = parser.add_argument_group("method parameters")
    method.add_argument("--depth", type=int, default=9, help="Poisson octree depth")
    method.add_argument("--watertight", type=float, default=0.0, metavar="VOXEL",
                        help="rebuild the final mesh as a closed manifold with marching cubes on a "
                             "signed distance grid of this voxel size [m], e.g. 0.004 (numpy and Open3D only); "
                             "use with --trim-distance 0 for a surface closed everywhere")
    method.add_argument("--denoise", type=float, default=0.0, metavar="RADIUS",
                        help="project each point onto its local plane (neighbours within RADIUS [m], "
                             "e.g. 0.02) before meshing: removes scan-row layering of fused clouds")
    method.add_argument("--smooth", type=int, default=0, metavar="N",
                        help="with --watertight: N Taubin smoothing iterations (volume preserving), e.g. 10")
    method.add_argument("--clip-below", type=float, default=None, metavar="Z",
                        help="with --watertight: cut the closed mesh flat at height Z (fuse_person.py "
                             "output has the floor at z = 0, so --clip-below 0 gives flat soles)")
    method.add_argument("--trim-distance", type=float, default=6.0,
                        help="Poisson: remove mesh vertices farther than this many mean point "
                             "spacings from the data (removes extrapolated surface), 0 = off. "
                             "LiDAR samples along rows, so the gaps between rows are several "
                             "times the mean spacing; much below 6 opens holes between rows")
    method.add_argument("--density-quantile", type=float, default=0.0,
                        help="Poisson: also trim vertices below this density quantile, 0 = off")
    method.add_argument("--bpa-radii", type=float, nargs="+", default=[1.5, 3.0, 6.0],
                        help="ball radii as multiples of the mean point spacing")
    method.add_argument("--alpha", type=float, default=0.0, help="[m], 0 = automatic")

    post = parser.add_argument_group("postprocessing")
    post.add_argument("--largest-component", action="store_true", help="keep only the largest piece")
    post.add_argument("--min-component", type=int, default=0,
                      help="drop pieces with fewer triangles than this")
    post.add_argument("--target-triangles", type=int, default=0, help="decimate to this count, 0 = off")
    post.add_argument("--fill-holes", type=float, default=0.0,
                      help="cap holes up to this size [m], e.g. the unseen base of an object "
                           "standing on the table; the caps are invented surface. 0 = off")
    post.add_argument("--coverage-threshold", type=float, default=5.0,
                      help="report the share of input points farther than this from the mesh [mm]")
    post.add_argument("--check-self-intersection", action="store_true",
                      help="count self-intersecting triangles (slow on large meshes)")

    return parser.parse_args()


def main():
    args = parse_arguments()
    print(f"pointcloud_to_mesh.py version {VERSION}")

    cloud_input = load_point_cloud(args.input)
    if args.crop_min is not None and args.crop_max is not None:
        cloud_input = crop_input(cloud_input, args.crop_min, args.crop_max)
    if len(cloud_input.points) < 10:
        raise SystemExit("fewer than 10 points left after cropping")
    print(f"input points: {len(cloud_input.points)}  organized: {cloud_input.is_organized}")

    method, mesh, cloud = build_mesh(cloud_input, args)
    mesh = clean_mesh(mesh, args, drop_fragments=(method == "poisson"))
    # Orientation needs an edge-manifold mesh, hence after cleaning; the grid
    # method is already oriented towards the sensor.
    if cloud is not None:
        mesh = orient_by_cloud_normals(mesh, cloud)
    if args.fill_holes > 0:
        mesh = cap_holes(mesh, args.fill_holes)
    if args.watertight > 0:
        mesh = watertight_remesh(mesh, args.watertight, args.clip_below)
        if args.smooth > 0:
            # Taubin smoothing moves vertices only, so the mesh stays closed;
            # unlike plain Laplacian smoothing it does not shrink the body.
            mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smooth)
        if args.target_triangles > 0 and len(mesh.triangles) > args.target_triangles:
            mesh = mesh.simplify_quadric_decimation(args.target_triangles)
    mesh.compute_vertex_normals()
    print(f"method: {method}")
    report(mesh, cloud_input.points, args.coverage_threshold, args.check_self_intersection)

    o3d.io.write_triangle_mesh(args.out, mesh)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
