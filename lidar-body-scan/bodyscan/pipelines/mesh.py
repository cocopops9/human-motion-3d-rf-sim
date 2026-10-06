"""Mesh pipeline: point cloud in, closed mesh out (conversion only, no smoothing).

    LoadCloud        .ply/.pcd/.xyz (normals kept) or .npz (organized frame or points), optional crop
    Reconstruct      Poisson (default), ball pivoting, alpha shape, or range-image grid
    CloseSurface     clean, orient, cap holes, watertight remesh through a signed distance field, clip,
                     optional decimation to a target edge length
    ReportQuality    topology, triangle size, facet normal noise, bump height vs lambda, fidelity
    WriteMesh        <out>.ply (or .obj) and <out>_quality.json

Smoothing is a separate command, 'bodyscan smooth', applied to the output.
Two steps here still limit the detail by construction, and are not
smoothing filters that can be switched off: Poisson reconstruction fits a
surface on an octree (depth 9: cells of about 4 mm for a person; depth 10:
about 2 mm), and the watertight remesh samples the surface on a grid
('watertight', 4 mm). Both set the finest detail kept; neither removes
bumps larger than its cell.

The defaults are for a person from 'bodyscan fuse' (z = 0 at the platform
top, so clip_below 0 gives flat soles). configs/mesh_generic.toml keeps an
open surface for any other object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from bodyscan.config import param, section
from bodyscan.jsonio import write_json
from bodyscan.log import info
from bodyscan.meshing import (AlphaMesher, BallPivotingMesher, GridMesher, PoissonMesher, cap_holes,
                              clean_mesh, crop_input, denoise_by_plane_projection, estimate_normals, format_quality,
                              decimate_to_edge, load_cloud_input, mesh_quality, orient_by_cloud_normals, prepare_cloud,
                              watertight_remesh)
from bodyscan.pipelines.base import Context, Pipeline, Step, StepList


@dataclass
class MeshInputConfig:
    """Input region."""
    crop_min: tuple[float, float, float] | None = param(None, "keep only points above this corner (X Y Z) [m]")
    crop_max: tuple[float, float, float] | None = param(None, "keep only points below this corner (X Y Z) [m]")
    sensor_origin: tuple[float, float, float] = param((0.0, 0.0, 0.0), "sensor position in the cloud's frame "
                                                                       "(grid method, orient 'sensor') [m]")


@dataclass
class ReconstructionConfig:
    """Surface reconstruction."""
    method: str = param("poisson", "reconstruction method", choices=("auto", "grid", "poisson", "bpa", "alpha"),
                        effect="poisson closes the surface; grid needs an organized frame; bpa and alpha "
                               "interpolate the points and leave holes")
    depth: int = param(9, "Poisson octree depth", effect="10 doubles the resolution (about 2 mm cells for a person), "
                                                         "and keeps more noise")
    trim_distance: float = param(0.0, "Poisson: remove vertices farther than this many mean point spacings from "
                                      "the data (0 = keep the closed surface)")
    density_quantile: float = param(0.0, "Poisson: also trim vertices below this density quantile (0 = off)")
    voxel: float = param(0.0, "voxel downsampling before meshing (0 = off)", unit="m")
    outlier_neighbors: int = param(20, "statistical outlier removal: neighbours (0 = off)")
    outlier_std: float = param(2.0, "statistical outlier removal: threshold in standard deviations")
    remove_plane: bool = param(False, "remove the dominant plane (floor) first")
    plane_threshold: float = param(0.01, "plane removal distance", unit="m")
    normal_radius: float = param(0.0, "normal estimation radius (0 = 4 x mean spacing)", unit="m")
    orient: str = param("auto", "normal orientation", choices=("auto", "input", "consistent", "sensor"),
                        effect="input: normals stored in the file (fusion output); consistent: clouds without "
                               "normals; sensor: a single view")
    denoise: float = param(0.0, "project each point onto its local plane (neighbours within this) before "
                                "meshing (0 = off)", unit="m", effect="removes scan-row layering, flattens "
                                                                       "details smaller than the radius")
    bpa_radii: tuple[float, ...] = param((1.5, 3.0, 6.0), "ball pivoting radii in mean point spacings")
    alpha: float = param(0.0, "alpha shape parameter (0 = 3 x mean spacing)", unit="m")
    max_jump: float = param(0.05, "grid: largest range spread inside a triangle", unit="m")
    max_relative_jump: float = param(0.03, "grid: range spread allowed as a fraction of the range")
    wrap: bool = param(True, "grid: connect the last column to the first (360 deg sensors)")


@dataclass
class ClosingConfig:
    """Cleaning and closing the surface."""
    watertight: float = param(0.004, "rebuild a closed manifold mesh on a signed distance grid of this voxel "
                                     "(0 = off)", unit="m", effect="smaller keeps finer details (fingers), "
                                                                    "slower, more triangles before decimation")
    clip_below: float | None = param(0.0, "cut the closed mesh flat at this height (flat soles at the platform "
                                          "top); not set: no cut", unit="m")
    fill_holes: float = param(0.0, "cap holes up to this size before closing (0 = off)", unit="m")
    largest_component: bool = param(False, "keep only the largest piece")
    min_component: int = param(0, "drop pieces with fewer triangles than this")
    target_edge_mm: float = param(0.0, "mean triangle edge after a quadric decimation (0 = no decimation: the grid "
                                       "of the watertight remesh, about 400 000 triangles for a person at 4 mm)",
                                  unit="mm",
                                  effect="6 mm gives about 140 000 triangles but rougher facets (decimation leaves "
                                         "kinks; run 'smooth' afterwards); a coarser watertight grid (0.006) gives "
                                         "a similar count with regular triangles")


@dataclass
class QualityConfig:
    """Quality report."""
    frequency_ghz: float = param(60.0, "carrier frequency the mesh is made for", unit="GHz",
                                 effect="sets the roughness targets of the report (lambda / 8 and lambda / 32)")
    radius: float = param(0.01, "neighbourhood of the facet-noise and bump-height measures", unit="m")


@dataclass
class MeshConfig:
    input: MeshInputConfig = section(MeshInputConfig)
    reconstruction: ReconstructionConfig = section(ReconstructionConfig)
    closing: ClosingConfig = section(ClosingConfig)
    quality: QualityConfig = section(QualityConfig)


class LoadCloud(Step):
    name = "load"

    def run(self, ctx: Context) -> None:
        c = ctx.config.input
        cloud_input = load_cloud_input(ctx.cloud_path)
        if c.crop_min is not None and c.crop_max is not None:
            cloud_input = crop_input(cloud_input, c.crop_min, c.crop_max)
        if len(cloud_input.points) < 10:
            raise SystemExit("fewer than 10 points left after cropping")
        info(f"input points: {len(cloud_input.points)}  organized: {cloud_input.is_organized}")
        ctx.cloud_input = cloud_input
        ctx.report["input"] = str(Path(ctx.cloud_path).resolve())


class Reconstruct(Step):
    name = "reconstruct"

    def mesher(self, r: ReconstructionConfig):
        return {"poisson": PoissonMesher(r.depth, r.trim_distance, r.density_quantile),
                "bpa": BallPivotingMesher(r.bpa_radii), "alpha": AlphaMesher(r.alpha)}[r.method]

    def run(self, ctx: Context) -> None:
        r = ctx.config.reconstruction
        cloud_input = ctx.cloud_input
        method = r.method if r.method != "auto" else ("grid" if cloud_input.is_organized else "poisson")
        ctx.cloud = None
        if method == "grid":
            if not cloud_input.is_organized:
                raise SystemExit("the grid method needs an organized frame (.npz with 'xyz')")
            mesher = GridMesher(r.max_jump, r.max_relative_jump, r.wrap, ctx.config.input.sensor_origin)
            ctx.mesh = mesher.build_grid(cloud_input.grid, cloud_input.valid)
        else:
            cloud = prepare_cloud(cloud_input.points, cloud_input.normals, r.voxel, r.outlier_neighbors, r.outlier_std,
                                  r.remove_plane, r.plane_threshold)
            cloud = estimate_normals(cloud, r.orient, r.normal_radius, ctx.config.input.sensor_origin)
            if r.denoise > 0:
                cloud = denoise_by_plane_projection(cloud, r.denoise)
            ctx.mesh = self.mesher(r).build(cloud)
            ctx.cloud = cloud
        ctx.method = method
        info(f"method: {method}: {len(ctx.mesh.triangles)} triangles")


class CloseSurface(Step):
    name = "close"

    def run(self, ctx: Context) -> None:
        c = ctx.config.closing
        mesh = clean_mesh(ctx.mesh, c.largest_component, c.min_component, 0, drop_fragments=(ctx.method == "poisson"))
        if ctx.cloud is not None:
            mesh = orient_by_cloud_normals(mesh, ctx.cloud)
        if c.fill_holes > 0:
            mesh = cap_holes(mesh, c.fill_holes)
        if c.watertight > 0:
            mesh = watertight_remesh(mesh, c.watertight, c.clip_below)
            info(f"watertight remesh on a {1000 * c.watertight:g} mm grid: {len(mesh.triangles)} triangles")
        if c.target_edge_mm > 0:
            before = len(mesh.triangles)
            mesh = decimate_to_edge(mesh, c.target_edge_mm / 1000.0)
            info(f"decimation to a mean edge of {c.target_edge_mm:g} mm: {before} -> {len(mesh.triangles)} triangles")
        mesh.compute_vertex_normals()
        ctx.mesh = mesh


class ReportQuality(Step):
    name = "quality"

    def run(self, ctx: Context) -> None:
        q = ctx.config.quality
        report = mesh_quality(ctx.mesh, q.frequency_ghz, q.radius, ctx.cloud_input.points)
        info("final mesh:\n  " + format_quality(report).replace("\n", "\n  "))
        ctx.report["quality"] = report


class WriteMesh(Step):
    name = "write"

    def run(self, ctx: Context) -> None:
        out = Path(ctx.out)
        # A name with dots (person_2.5mm) keeps them: only a known mesh suffix is taken as the format.
        path = out if out.suffix.lower() in (".ply", ".obj", ".stl", ".off") else out.with_name(out.name + ".ply")
        path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(path), ctx.mesh)
        report_path = path.with_name(path.stem + "_quality.json")
        write_json(report_path, ctx.report)
        info(f"wrote {path} and {report_path}")


class MeshPipeline(Pipeline):
    config_class = MeshConfig
    name = "mesh"

    def steps(self) -> StepList:
        return StepList([LoadCloud(), Reconstruct(), CloseSurface(), ReportQuality(), WriteMesh()])


def quality_only(mesh_path, frequency_ghz=60.0, radius=0.01, cloud_path=None) -> dict:
    """Quality report of an existing mesh (for 'bodyscan quality')."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    points = None
    if cloud_path:
        points = np.asarray(o3d.io.read_point_cloud(str(cloud_path)).points)
    return mesh_quality(mesh, frequency_ghz, radius, points)
