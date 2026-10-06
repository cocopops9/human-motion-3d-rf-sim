"""Smoothing of an existing closed mesh ('bodyscan smooth').

The conversion from point cloud to mesh ('bodyscan mesh') does not smooth;
everything that smooths the surface is here, with its parameters exposed
directly:

    scale_mm            size of what is removed: the filter averages the facet
                        normals within 2 x scale_mm, so bumps shorter than about
                        that are flattened and longer ones are kept
    rounds              how many times the filter is applied: the strength
    finish_rounds       rounds at finish_scale_mm after the main ones, which
    finish_scale_mm     remove the small dimples a wide filter leaves
    normal_sigma        how much a crease is protected (small) or smoothed (large)
    normal_iterations,  inner passes of one round (normal filtering, vertex
    vertex_iterations   update); 0 vertex passes = automatic
    relax_iterations    tangential relaxation before the first round
    keep_volume         restore the input volume after every round
    max_deviation_mm    optional bound on the distance from the input surface

One round (meshing.smoothing.bilateral_smooth) filters the facet normals
(bilateral filter, Zheng et al. 2011) and moves the vertices to agree with
them (Sun et al. 2007). Vertices only move: a closed, manifold mesh stays
closed and manifold.

The filter shrinks curved parts slightly (the vertex update cuts the corners
of the normal field); with keep_volume the surface is offset along its
normals by the uniform amount that restores the input volume.

With max_deviation_mm > 0, every vertex is pulled back after every round so
that it ends within that distance of the input surface: the excess is
averaged over the neighbourhood before it is subtracted (a vertex-by-vertex
clamp leaves kinks at the edge of the band), then a hard clamp guarantees
the bound. The limit also stops the smoothing wherever the bumps are taller
than the limit: a limit of 3 mm or less keeps the scan-row stripes of tt11,
which need 2 to 7 mm to go. It is off by default; the deviation from the
input is always measured and reported, so the shape change is known.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from bodyscan.config import param
from bodyscan.geometry.neighbors import RadiusSearch
from bodyscan.log import info
from bodyscan.meshing.quality import summary
from bodyscan.meshing.smoothing import bilateral_smooth, median_edge_mm, vertex_normals


@dataclass
class SmoothingConfig:
    """Smoothing of an existing mesh (all values are used as given)."""
    scale_mm: float = param(12.0, "spatial scale (Gaussian sigma) of the filter: every facet averages the facets "
                                  "within twice this distance", unit="mm",
                            effect="sets the size of what is removed. tt11, 4 rounds: 6 mm removes facet noise and "
                                   "small dimples but keeps the scan-row stripes; 12 mm removes the stripes (about "
                                   "3 cm apart); 16 to 24 mm also flattens clothing folds and the face, and moves "
                                   "the surface 2 to 3 times farther")
    rounds: int = param(4, "number of times the filter is applied",
                        effect="the strength. tt11 at 12 mm, no finishing rounds: 1, 2, 4, 8 rounds move the surface by a median of "
                               "0.26, 0.45, 0.74, 1.13 mm (99th percentile 1.8, 2.8, 4.3, 6.4 mm); the stripes "
                               "fade progressively and are gone at 8")
    finish_rounds: int = param(2, "rounds at finish_scale_mm after the main rounds (only when finish_scale_mm is "
                                  "smaller than scale_mm; 0 = none)",
                               effect="a wide filter leaves small dimples and, above about 16 mm, fine ripples; "
                                      "2 rounds at 6 mm remove them (tt11, 24 mm: facet noise p90 5.7 -> 1.6 deg) "
                                      "and move the surface by less than 0.1 mm more")
    finish_scale_mm: float = param(6.0, "spatial scale of the finishing rounds", unit="mm")
    normal_sigma: float = param(1.5, "bilateral range sigma: how different two facet normals may be and still be "
                                     "averaged (difference of unit normals; 0.35 is about 20 deg, 1.5 is nearly "
                                     "isotropic)",
                                effect="small values keep creases (fingers, chin, folds) but also turn scan defects "
                                       "into sharp creases and leave the surface in flat patches; large values "
                                       "smooth everything alike")
    normal_iterations: int = param(4, "normal filtering passes per round",
                                   effect="more passes spread the average farther within one round")
    vertex_iterations: int = param(0, "vertex update passes per round; 0 = automatic: 15 for a median edge of "
                                      "3.6 mm, times (3.6 mm / edge)^2, times (scale / 12 mm)^2 above 12 mm",
                                   effect="too few passes: the vertices do not follow the filtered normals and the "
                                          "surface keeps small dimples")
    relax_iterations: int = param(5, "tangential relaxation passes before the first round (better shaped "
                                     "triangles, the surface does not move)",
                                  effect="0 keeps the input triangles; marching cubes leaves slivers whose normals "
                                         "are noise")
    keep_volume: bool = param(True, "offset the surface along its normals to restore the input volume after every "
                                    "round (closed meshes only)",
                              effect="off: the filter shrinks thin parts (arms, fingers) slightly at every round")
    max_deviation_mm: float = param(0.0, "keep every vertex within this distance of the input surface (0 = no "
                                         "limit)", unit="mm",
                                    effect="a bound on the shape change; it also stops the smoothing where the "
                                           "bumps are taller than the limit (tt11 stripes need 2 to 7 mm)")
    smooth_limit: bool = param(True, "with a limit: pull the vertices back with a smooth correction (averaged "
                                     "over about 1.5 edges) before the hard clamp",
                               effect="off: a vertex-by-vertex clamp, which leaves kinks at the edge of the band")


class DeviationLimit:
    """Distance from the input surface, and the pull back onto a band of it."""

    def __init__(self, mesh: o3d.geometry.TriangleMesh, max_deviation_m: float):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        self.max_deviation = float(max_deviation_m)

    def closest(self, vertices: np.ndarray):
        answer = self.scene.compute_closest_points(o3d.core.Tensor(vertices.astype(np.float32)))
        points = answer["points"].numpy().astype(np.float64)
        return points, np.linalg.norm(vertices - points, axis=1)

    def distance(self, vertices: np.ndarray) -> np.ndarray:
        return self.closest(vertices)[1]

    def apply(self, vertices: np.ndarray):
        """Vertices pulled back to at most max_deviation from the input surface,
        and the fraction of vertices that had to be pulled."""
        points, distance = self.closest(vertices)
        far = distance > self.max_deviation
        result = vertices.copy()
        result[far] = points[far] + (vertices[far] - points[far]) * (self.max_deviation / distance[far])[:, None]
        return result, float(far.mean())


class SurfaceAverage:
    """Gaussian average of a per-vertex field over the vertices within 2 sigma
    (the neighbourhoods are computed once, for the given vertex positions)."""

    def __init__(self, vertices: np.ndarray, sigma: float):
        search = RadiusSearch(vertices, 2.0 * sigma)
        rows, index, weight = [], [], []
        for part, found, squared in search.chunks(vertices):
            valid = found >= 0
            rows.append(np.nonzero(valid)[0] + part.start)
            index.append(found[valid])
            weight.append(np.exp(-squared[valid] / (2.0 * sigma ** 2)))
        self.rows, self.index, self.weight = np.concatenate(rows), np.concatenate(index), np.concatenate(weight)
        self.total = np.bincount(self.rows, weights=self.weight, minlength=len(vertices))
        self.count = len(vertices)

    def __call__(self, field: np.ndarray) -> np.ndarray:
        columns = [np.bincount(self.rows, weights=self.weight * field[self.index, k], minlength=self.count)
                   for k in range(field.shape[1])]
        return np.stack(columns, axis=1) / np.maximum(self.total, 1e-30)[:, None]


def smooth_pull_back(vertices: np.ndarray, limit: DeviationLimit, sigma: float, passes: int = 8) -> np.ndarray:
    """Bring the vertices back within the limit with a correction that is
    itself smooth: the excess beyond the limit is averaged over the
    neighbourhood before it is applied, so the pull back adds no kinks.
    Repeated until the excess is negligible."""
    average = SurfaceAverage(vertices, sigma)
    for _ in range(passes):
        points, distance = limit.closest(vertices)
        excess = np.maximum(distance - limit.max_deviation, 0.0)
        if excess.max() <= 0.02 * limit.max_deviation:
            break
        correction = (vertices - points) * (excess / np.maximum(distance, 1e-12))[:, None]
        vertices = vertices - average(correction)
    return vertices


def closed_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    v0, v1, v2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def surface_area(vertices: np.ndarray, triangles: np.ndarray) -> float:
    v0, v1, v2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())


def restore_volume(vertices: np.ndarray, triangles: np.ndarray, target: float) -> np.ndarray:
    """Uniform offset along the vertex normals that brings the closed volume back to 'target'
    (to first order: dV = area x offset)."""
    offset = (target - closed_volume(vertices, triangles)) / max(surface_area(vertices, triangles), 1e-12)
    return vertices + offset * vertex_normals(vertices, triangles)


class MeshSmoother:
    """Rounds of bilateral smoothing, each followed by the optional volume
    restoration and deviation limit, all relative to the input mesh."""

    def __init__(self, mesh: o3d.geometry.TriangleMesh, config: SmoothingConfig):
        self.input = mesh
        self.config = config
        self.triangles = np.asarray(mesh.triangles).astype(np.int64)
        # The distance to the input is always measured; the limit is applied only when set.
        self.reference = DeviationLimit(mesh, max(config.max_deviation_mm, 0.0) / 1000.0)
        self.closed = len(mesh.get_non_manifold_edges(allow_boundary_edges=False)) == 0
        self.target_volume = closed_volume(np.asarray(mesh.vertices), self.triangles) if self.closed else None
        if config.keep_volume and not self.closed:
            info("the mesh is not closed: keep_volume is ignored")

    def round(self, mesh, scale_mm: float, first: bool):
        """One round; returns the mesh and the fraction of vertices held at the limit."""
        c = self.config
        smoothed = bilateral_smooth(mesh, scale_mm / 1000.0, c.normal_sigma, c.normal_iterations, c.vertex_iterations,
                                    relax_iterations=c.relax_iterations if first else 0, verbose=first)
        vertices = np.asarray(smoothed.vertices)
        if c.keep_volume and self.target_volume is not None:
            vertices = restore_volume(vertices, self.triangles, self.target_volume)
        held = 0.0
        if c.max_deviation_mm > 0:
            if c.smooth_limit:
                vertices = smooth_pull_back(vertices, self.reference,
                                            1.5 * median_edge_mm(vertices, self.triangles) / 1000.0)
            vertices, held = self.reference.apply(vertices)        # the guarantee
        result = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), self.input.triangles)
        result.compute_vertex_normals()
        return result, held

    def schedule(self) -> list[float]:
        """The scale of every round, main rounds first."""
        c = self.config
        scales = [c.scale_mm] * max(c.rounds, 0)
        if c.finish_rounds > 0 and 0 < c.finish_scale_mm < c.scale_mm:
            scales += [c.finish_scale_mm] * c.finish_rounds
        return scales

    def statistics(self, mesh, number: int, held: float) -> dict:
        vertices = np.asarray(mesh.vertices)
        distance_mm = self.reference.distance(vertices) * 1000.0
        result = {"round": number, "deviation_mm": summary(distance_mm),
                  "deviation_max_mm": round(float(distance_mm.max()), 3)}
        if self.config.max_deviation_mm > 0:
            result["held_at_limit_fraction"] = round(held, 4)
        if self.target_volume is not None:
            result["volume_change_percent"] = round(
                100.0 * (closed_volume(vertices, self.triangles) / self.target_volume - 1.0), 3)
        return result


def smooth_mesh(mesh: o3d.geometry.TriangleMesh, config: SmoothingConfig, progress=None):
    """The smoothed mesh and its statistics (deviation from the input, volume change).
    progress(number, scale_mm, statistics) is called after every round."""
    if config.scale_mm <= 0:
        raise SystemExit("scale_mm must be positive")
    smoother = MeshSmoother(mesh, config)
    current, held = mesh, 0.0
    scales = smoother.schedule()
    for number, scale_mm in enumerate(scales, start=1):
        current, held = smoother.round(current, scale_mm, first=(number == 1))
        if progress is not None:
            progress(number, scale_mm, smoother.statistics(current, number, held))
    if not scales:
        current = o3d.geometry.TriangleMesh(mesh)
    return current, smoother.statistics(current, len(scales), held)
