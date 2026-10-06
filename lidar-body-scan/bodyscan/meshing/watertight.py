"""Closed, edge-manifold remeshing through a signed distance field."""

from __future__ import annotations

import copy

import numpy as np
import open3d as o3d

from bodyscan.log import info
from bodyscan.meshing.marching import dilate, marching_cubes


def _occupancy(scene, points: np.ndarray) -> np.ndarray:
    """Inside test by ray parity (majority of 5 rays), in chunks."""
    inside = np.empty(len(points), dtype=bool)
    for start in range(0, len(points), 2_000_000):
        chunk = o3d.core.Tensor(points[start:start + 2_000_000].astype(np.float32))
        inside[start:start + 2_000_000] = scene.compute_occupancy(chunk, nsamples=5).numpy() > 0.5
    return inside


def far_signs(scene, band: np.ndarray, lower: np.ndarray, voxel: float, sdf: np.ndarray, slab: int = 16) -> None:
    """Fill sdf outside the band with -3 voxel (inside) or +3 voxel (outside).

    Every voxel outside the band is more than about 1.2 voxels from the
    surface, so the 8 voxels of a 2 x 2 x 2 block that lies entirely outside
    the band share the sign of the block centre (0.87 voxel away): one ray
    test per block instead of eight. The other far voxels are tested one by
    one. The grid is processed in slabs of 'slab' blocks along z, so that a
    fine grid (2 mm for a person: about 2 x 10^8 voxels) fits in memory."""
    shape = np.array(band.shape)
    blocks = (shape + 1) // 2
    outside_value, inside_value = np.float32(3.0 * voxel), np.float32(-3.0 * voxel)
    for z0 in range(0, blocks[2], slab):
        z1 = min(z0 + slab, blocks[2])
        part = band[:, :, 2 * z0:2 * z1]
        padded = np.ones((2 * blocks[0], 2 * blocks[1], 2 * (z1 - z0)), dtype=bool)   # padding: not in the band
        padded[:part.shape[0], :part.shape[1], :part.shape[2]] = part
        padded = padded.reshape(blocks[0], 2, blocks[1], 2, z1 - z0, 2)
        all_far = ~padded.any(axis=(1, 3, 5))
        # Pad cells are 'True' in 'padded', so a block that sticks out of the grid is
        # never all-far; its real voxels are handled one by one below.
        whole = np.argwhere(all_far)
        if len(whole):
            centres = lower + voxel * (2.0 * whole + [0.5, 0.5, 0.5 + 2 * z0])
            values = np.where(_occupancy(scene, centres), inside_value, outside_value)
            for di in (0, 1):
                for dj in (0, 1):
                    for dk in (0, 1):
                        sdf[2 * whole[:, 0] + di, 2 * whole[:, 1] + dj, 2 * (whole[:, 2] + z0) + dk] = values
        rest = ~part & ~np.repeat(np.repeat(np.repeat(all_far, 2, 0), 2, 1), 2, 2)[:part.shape[0], :part.shape[1],
                                                                                     :part.shape[2]]
        single = np.argwhere(rest)
        if len(single):
            points = lower + voxel * (single + [0, 0, 2 * z0])
            sdf[single[:, 0], single[:, 1], single[:, 2] + 2 * z0] = np.where(_occupancy(scene, points),
                                                                              inside_value, outside_value)


def watertight_remesh(mesh, voxel: float, clip_below=None, min_fraction: float = 0.01):
    """Rebuild the mesh as the zero level set of a signed distance field.

    Marching cubes on a sampled field, padded on every side, yields a
    closed, edge-manifold, consistently oriented surface; Poisson meshes from
    Open3D can contain non-manifold edges and self-intersections where two
    parts nearly touch (arm against torso). The sign (inside or outside) comes
    from ray parity (majority over several rays, tolerant to small defects),
    except within half a voxel of the surface where the closest point lies
    inside a triangle: there the triangle's normal decides (it also handles
    thin overlaps of two parts). The winding is checked per connected piece
    against parity: a piece wound inside out is turned, a piece with an
    inconsistent winding takes the parity sign. A wrongly wound patch cannot
    add a second, jagged sheet to the surface. Holes narrower than about 1.5
    voxels are sealed; details smaller than a voxel are lost.
    Components smaller than 'min_fraction' of the surface (bubbles) are dropped.
    clip_below: intersect with the half-space z >= clip_below (a flat, closed
    cut, e.g. the platform under the feet)."""
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
    primitive = answer["primitive_ids"].numpy()
    offset = query - answer["points"].numpy()
    band_distance = np.linalg.norm(offset, axis=1)
    normal_side = np.sign(np.einsum("ij,ij->i", offset, triangle_normals[primitive]))
    parity_side = np.where(_occupancy(scene, query), -1.0, 1.0)
    # The normal of one triangle tells the side only when the closest point lies
    # inside that triangle: at an edge or a vertex a neighbour may disagree.
    uv = answer["primitive_uvs"].numpy()
    inside_triangle = (uv[:, 0] > 0.02) & (uv[:, 1] > 0.02) & (uv[:, 0] + uv[:, 1] < 0.98)
    use_parity = ~inside_triangle | (band_distance >= 0.5 * voxel)
    # The winding is checked piece by piece (connected parts of the mesh): a
    # piece wound inside out has its normals turned; a piece whose normals
    # disagree with parity on more than 1 % of the voxels near it, or that is
    # too small to check, takes the parity sign everywhere.
    piece = np.asarray(mesh.cluster_connected_triangles()[0])[primitive]
    away = band_distance >= voxel
    reversed_pieces, parity_pieces = 0, 0
    for label in np.unique(piece):
        members = piece == label
        checked = members & away
        if not checked.any():
            use_parity |= members
            continue
        disagreement = float(np.mean(normal_side[checked] != parity_side[checked]))
        if disagreement > 0.5:                                   # wound inside out
            normal_side[members] *= -1.0
            disagreement = 1.0 - disagreement
            reversed_pieces += 1
        if disagreement > 0.01:
            use_parity |= members
            parity_pieces += 1
    if reversed_pieces or parity_pieces:
        info(f"watertight remesh: {reversed_pieces} piece(s) wound inside out, {parity_pieces} with an "
             "inconsistent winding (their sign comes from ray parity)")
    band_side = np.where(use_parity, parity_side, normal_side)

    sdf = np.empty(shape, dtype=np.float32)
    far_signs(scene, band, lower, voxel, sdf)
    sdf[index[:, 0], index[:, 1], index[:, 2]] = band_side * band_distance
    if not np.any(sdf < 0):
        raise SystemExit("watertight remesh: no interior found (the surface has large holes); "
                         "use trim_distance 0 or fill_holes first")
    if clip_below is not None:
        heights = (lower[2] + voxel * np.arange(shape[2])).astype(np.float32)
        np.maximum(sdf, (np.float32(clip_below) - heights)[None, None, :], out=sdf)     # in place: float32 grid
    # A sample exactly on the level set would put several vertices at one
    # point and make degenerate triangles; keep every sample off zero.
    sdf[np.abs(sdf) < 1e-7 * voxel] = 1e-7 * voxel
    # The grid border is outside by definition, so the surface cannot run into it.
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
