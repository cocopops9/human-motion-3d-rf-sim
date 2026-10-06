"""Meshing commands."""

from __future__ import annotations

from pathlib import Path

from bodyscan.commands.base import Command, PipelineCommand
from bodyscan.jsonio import write_json
from bodyscan.log import info
from bodyscan.meshing import format_quality
from bodyscan.pipelines.mesh import MeshPipeline, quality_only
from bodyscan.pipelines.smooth import SmoothMeshPipeline


class Mesh(PipelineCommand):
    """Point cloud -> closed mesh (conversion only; smoothing is 'bodyscan smooth').

    Example (fused person, default settings):
        python -m bodyscan mesh person_tt16.ply --out person_tt16_mesh.ply
        python -m bodyscan smooth person_tt16_mesh.ply --out person_tt16_smooth.ply

    The report <out>_quality.json gives the facet normal noise, the bump height
    against lambda/8, the triangle sizes and the distance to the fused cloud."""
    name = "mesh"
    help = "point cloud -> closed mesh with a quality report (no smoothing)"
    pipeline_class = MeshPipeline
    input_help = "point cloud (.ply/.pcd/.xyz/.npz), e.g. the output of 'fuse'"
    default_out = "mesh.ply"

    def inputs(self, args):
        return {"cloud_path": Path(args.input), "out": Path(args.out)}


class Quality(Command):
    """Quality report of an existing mesh: topology, triangle sizes, facet
    normal noise, angles between neighbouring triangles, bump height against
    the wavelength, and (with --cloud) the distance from a point cloud.

        python -m bodyscan quality person_mesh.ply --cloud person_tt16.ply --frequency-ghz 60"""
    name = "quality"
    help = "quality numbers of a mesh for ray tracing"

    def add_arguments(self, parser):
        parser.add_argument("mesh")
        parser.add_argument("--cloud", default=None, help="point cloud the mesh was made from (fidelity)")
        parser.add_argument("--frequency-ghz", type=float, default=60.0)
        parser.add_argument("--radius", type=float, default=0.01, help="neighbourhood of the measures [m]")
        parser.add_argument("--json", default=None, help="also write the report here")

    def run(self, args):
        report = quality_only(args.mesh, args.frequency_ghz, args.radius, args.cloud)
        info(format_quality(report))
        if args.json:
            write_json(args.json, report)
        return 0


class Smooth(PipelineCommand):
    """Smooth an existing mesh, with every parameter of the smoothing set directly.

    One round filters the facet normals over --scale-mm and moves the vertices
    to agree with them; --rounds sets the strength. The volume is restored
    after every round (--no-keep-volume to turn it off), and the distance of
    every vertex from the input surface is measured and reported.
    --max-deviation-mm bounds it (off by default). Measured on tt11:

        --scale-mm 12 --finish-rounds 0, --rounds 1 / 2 / 4 / 8: distance from the input
        median 0.26 / 0.45 / 0.74 / 1.13 mm, 99th percentile 1.8 / 2.8 /
        4.3 / 6.4 mm; the scan-row stripes fade and are gone at 8 rounds.

    Defaults (12 mm, 4 rounds, then 2 rounds at 6 mm):
        python -m bodyscan smooth person_tt11_mesh.ply --out person_tt11_smooth.ply

    Stronger, or only the facet noise, or with a bound on the shape change:
        python -m bodyscan smooth person_tt11_mesh.ply --out tt11_s.ply --rounds 8
        python -m bodyscan smooth person_tt11_mesh.ply --out tt11_s.ply --scale-mm 6 --rounds 2
        python -m bodyscan smooth person_tt11_mesh.ply --out tt11_s.ply --max-deviation-mm 3

    Compare values (one mesh per value, and a table):
        python -m bodyscan smooth person_tt11_mesh.ply --out tt11_s.ply --sweep rounds=1,2,4,8"""
    name = "smooth"
    help = "smooth an existing mesh (direct parameters, optional bound on the shape change)"
    pipeline_class = SmoothMeshPipeline
    input_help = "mesh (.ply/.obj/.stl), e.g. the output of 'mesh'"
    default_out = "smoothed.ply"

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument("--cloud", default=None, help="point cloud the mesh was made from (adds the distance "
                                                          "from it to the report)")

    def inputs(self, args):
        return {"mesh_path": Path(args.input), "out": Path(args.out), "cloud_path": args.cloud}
