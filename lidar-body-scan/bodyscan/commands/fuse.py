"""Fusion commands."""

from __future__ import annotations

from bodyscan.commands.base import PipelineCommand
from bodyscan.pipelines.turntable import TurntablePipeline


class FuseTurntable(PipelineCommand):
    """Fuse a turntable capture (person standing still on the rotating platform) into one cloud.

    Examples:
        python -m bodyscan fuse C:\\lidar\\tt16 --out person_tt16
        python -m bodyscan fuse C:\\lidar\\tt16 --out person_tt16 --center 1.139 0.214

    The platform centre is found from the ring of the platform around the
    person; --center X Y (from 'bodyscan detect --rotation --human') imposes it.

    Outputs: <out>.ply (input of 'mesh'), <out>_confidence.ply, <out>_views.ply,
    <out>.json (report with the full configuration), <out>_angle.png."""
    name = "fuse"
    help = "turntable capture -> fused point cloud of the person"
    pipeline_class = TurntablePipeline
    input_help = "capture directory (lut.npz, background/, frames/)"
    default_out = "person_tt"
