"""In-place fusion command."""

from __future__ import annotations

from bodyscan.commands.base import PipelineCommand
from bodyscan.pipelines.inplace import InPlacePipeline


class FuseInPlace(PipelineCommand):
    """Fuse an in-place capture (person turning on the spot by steps, holding still at every stop).

    Examples:
        python -m bodyscan fuse-inplace C:\\lidar\\person_run --out person
        python -m bodyscan fuse-inplace C:\\lidar\\person_run --center 1.40 0.20
        python -m bodyscan fuse-inplace C:\\lidar\\person_run --crop-min 0.1 0.9 -1.3 --crop-max 1.5 2.3 0.9

    Without --center or a crop box the person is found in the frames.
    Outputs: <out>.ply (input of 'mesh'), <out>_confidence.ply, <out>_views.ply,
    <out>.json, <out>_top.png, <out>_motion.png."""
    name = "fuse-inplace"
    help = "in-place capture (person turning by steps) -> fused point cloud"
    pipeline_class = InPlacePipeline
    input_help = "capture directory (lut.npz, background/, frames/)"
    default_out = "person"
