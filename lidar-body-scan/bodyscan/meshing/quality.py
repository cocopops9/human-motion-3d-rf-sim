"""Mesh quality numbers for ray tracing: topology, triangle size, smoothness, fidelity.

The smoothness numbers that matter with face normals:

    facet_noise_deg   angle between each triangle's normal and the area-weighted
                      mean normal of the triangles within 'radius' around it.
                      On a smooth surface it is close to 0 whatever the
                      curvature (the curvature averages out symmetrically); on
                      a noisy one it is the random tilt that sends rays astray.
    dihedral_deg      angle between the normals of neighbouring triangles: the
                      curvature of the body (edge / radius of curvature, e.g.
                      8 mm on a 4 cm arm: 11 deg) plus the noise. It shrinks
                      with the triangles, so it compares meshes of the same
                      grid only.
    height_rms_mm     bump height: residual of the surface around a local
                      quadric fit (which absorbs the curvature of the body),
                      compared with lambda / 8 (Rayleigh: a surface with
                      smaller bumps reflects like a mirror) and lambda / 32
                      (Fraunhofer, stricter).
    fidelity          distance from the fused point cloud to the mesh: how much
                      the smoothing moved the surface away from the measurement.

Every neighbourhood holds all the triangles (or vertices) within 'radius',
however fine the mesh (RadiusSearch), so meshes of different grids are
measured over the same patches. Facet noise and bump height are computed at
a fixed random sample of at most 100 000 triangles and 20 000 vertices.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from bodyscan.geometry.neighbors import RadiusSearch
from bodyscan.meshing.smoothing import face_geometry, unit, wavelength_mm


def _edges(triangles):
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    faces = np.tile(np.arange(len(triangles)), 3)
    return edges, faces


def dihedral_angles(vertices, triangles) -> np.ndarray:
    """Angle [deg] between the normals of the two triangles of every interior edge."""
    _, _, normals = face_geometry(vertices, triangles)
    edges, faces = _edges(triangles)
    key = edges[:, 0] * (len(vertices) + 1) + edges[:, 1]
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    same = sorted_key[1:] == sorted_key[:-1]
    a, b = faces[order][:-1][same], faces[order][1:][same]
    return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", normals[a], normals[b]), -1.0, 1.0)))


def sample_indices(count: int, sample: int, seed: int = 0) -> np.ndarray:
    """All indices, or a fixed random subset of 'sample' of them (sorted)."""
    if count <= sample:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(count, sample, replace=False))


def facet_noise(vertices, triangles, radius, sample=100000, seed=0) -> np.ndarray:
    """Angle [deg] between the normal of every face (of a random sample) and
    the area-weighted mean normal of all the faces within 'radius' of it."""
    centroids, areas, normals = face_geometry(vertices, triangles)
    chosen = sample_indices(len(triangles), sample, seed)
    search = RadiusSearch(centroids, radius)
    result = np.empty(len(chosen))
    for part, found, _ in search.chunks(centroids[chosen]):
        valid = found >= 0
        safe = np.clip(found, 0, None)
        mean = unit(np.einsum("nk,nkj->nj", np.where(valid, areas[safe], 0.0), normals[safe]))
        result[part] = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", mean, normals[chosen[part]]), -1.0, 1.0)))
    return result


def quadric_roughness(mesh, radius, sample=20000, seed=0) -> np.ndarray:
    """Bump height [m] at a random sample of vertices: rms residual of all
    the vertices within 'radius' around a weighted quadric fitted to them
    (z = a x^2 + b xy + c y^2 + d x + e y + f in the tangent frame). The
    quadric absorbs the curvature of the body, so what is left is the
    irregularity that a smooth body does not have."""
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.compute_vertex_normals()
    vertices, normals = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
    chosen = sample_indices(len(vertices), sample, seed)
    search = RadiusSearch(vertices, radius)
    result = []
    for part, found, squared in search.chunks(vertices[chosen], entries=1_000_000):
        centre, n = vertices[chosen[part]], normals[chosen[part]]
        valid = found >= 0
        helper = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
        t1 = unit(np.cross(n, helper))
        t2 = np.cross(n, t1)
        relative = vertices[np.clip(found, 0, None)] - centre[:, None, :]
        x = np.einsum("nkj,nj->nk", relative, t1)
        y = np.einsum("nkj,nj->nk", relative, t2)
        z = np.einsum("nkj,nj->nk", relative, n)
        weight = np.where(valid, np.exp(-squared / (2.0 * (radius / 2.0) ** 2)), 0.0)
        design = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], axis=2)          # (n, k, 6)
        normal_matrix = np.einsum("nk,nki,nkj->nij", weight, design, design) + 1e-12 * np.eye(6)
        right = np.einsum("nk,nki,nk->ni", weight, design, z)
        enough = valid.sum(axis=1) >= 10
        coefficients = np.zeros((len(centre), 6))
        coefficients[enough] = np.linalg.solve(normal_matrix[enough], right[enough][..., None])[..., 0]
        residual = z - np.einsum("nki,ni->nk", design, coefficients)
        rms = np.sqrt(np.sum(weight * residual ** 2, axis=1) / np.maximum(weight.sum(axis=1), 1e-30))
        result.append(rms[enough])
    return np.concatenate(result) if result else np.zeros(0)


def summary(values, digits=2) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    return {"median": round(float(np.median(values)), digits), "p90": round(float(np.percentile(values, 90)), digits),
            "p99": round(float(np.percentile(values, 99)), digits)}


def mesh_quality(mesh, frequency_ghz=60.0, radius=0.01, cloud_points=None) -> dict:
    """Quality report of a triangle mesh (see the module docstring)."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles).astype(np.int64)
    report = {"vertices": len(vertices), "triangles": len(triangles)}
    if len(triangles) == 0:
        return report
    open_edges = len(mesh.get_non_manifold_edges(allow_boundary_edges=False))
    report.update({"closed": open_edges == 0, "boundary_or_non_manifold_edges": open_edges,
                   "vertex_manifold": bool(mesh.is_vertex_manifold()),
                   "area_m2": round(float(mesh.get_surface_area()), 4)})
    if open_edges == 0:
        v0, v1, v2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
        report["volume_l"] = round(float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0 * 1000.0), 3)
    edges, _ = _edges(triangles)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1) * 1000.0
    report["edge_mm"] = summary(lengths)
    report["dihedral_deg"] = summary(dihedral_angles(vertices, triangles))
    report["facet_noise_deg"] = summary(facet_noise(vertices, triangles, radius))
    heights = quadric_roughness(mesh, radius) * 1000.0
    wavelength = wavelength_mm(frequency_ghz)
    rms = float(np.sqrt(np.mean(heights ** 2)))
    report["height_rms_mm"] = round(rms, 3)
    report["height_mm"] = summary(heights, 3)
    report["rf"] = {"frequency_ghz": frequency_ghz, "wavelength_mm": round(wavelength, 3),
                    "rayleigh_limit_mm": round(wavelength / 8.0, 3), "fraunhofer_limit_mm": round(wavelength / 32.0, 3),
                    "smooth_by_rayleigh": rms < wavelength / 8.0, "smooth_by_fraunhofer": rms < wavelength / 32.0}
    if cloud_points is not None and len(cloud_points):
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        points = np.asarray(cloud_points, dtype=np.float32)
        if len(points) > 200000:
            points = points[np.random.default_rng(0).choice(len(points), 200000, replace=False)]
        distance = scene.compute_distance(o3d.core.Tensor(points)).numpy() * 1000.0
        report["fidelity_mm"] = summary(distance)
    return report


def format_quality(report: dict) -> str:
    """Readable lines for the console."""
    lines = [f"triangles {report.get('triangles')}, closed {report.get('closed')} "
             f"({report.get('boundary_or_non_manifold_edges')} open or non-manifold edges), "
             f"vertex manifold {report.get('vertex_manifold')}"]
    if "edge_mm" in report:
        lines.append(f"edge length [mm]: median {report['edge_mm']['median']}, p90 {report['edge_mm']['p90']}")
        lines.append(f"facet normal noise [deg]: median {report['facet_noise_deg']['median']}, "
                     f"p90 {report['facet_noise_deg']['p90']}, p99 {report['facet_noise_deg']['p99']}")
        lines.append(f"angle between neighbouring triangles [deg]: median {report['dihedral_deg']['median']}, "
                     f"p90 {report['dihedral_deg']['p90']}, p99 {report['dihedral_deg']['p99']}")
        rf = report["rf"]
        lines.append(f"bump height rms {report['height_rms_mm']:.3f} mm; at {rf['frequency_ghz']:g} GHz "
                     f"lambda/8 = {rf['rayleigh_limit_mm']:.2f} mm ({'smooth' if rf['smooth_by_rayleigh'] else 'ROUGH'}"
                     f" by Rayleigh), lambda/32 = {rf['fraunhofer_limit_mm']:.2f} mm")
    if "volume_l" in report:
        lines.append(f"volume {report['volume_l']:.2f} l, area {report['area_m2']:.3f} m2")
    if "fidelity_mm" in report:
        lines.append(f"distance from the fused cloud to the mesh [mm]: median {report['fidelity_mm']['median']}, "
                     f"p90 {report['fidelity_mm']['p90']}, p99 {report['fidelity_mm']['p99']}")
    return "\n".join(lines)
