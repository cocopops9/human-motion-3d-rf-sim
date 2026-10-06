"""Detection command."""

from __future__ import annotations

from pathlib import Path

from bodyscan.commands.base import PipelineCommand
from bodyscan.pipelines.detect import DetectPipeline


class Detect(PipelineCommand):
    """Find rotating objects and/or people in a recording, and their centres.

        python -m bodyscan detect C:\\lidar\\tt16 --rotation            rotating objects (axis)
        python -m bodyscan detect C:\\lidar\\tt16 --human               people (shape cascade)
        python -m bodyscan detect C:\\lidar\\tt16 --rotation --human    rotating people

    Input: a capture directory (lut.npz, frames/, optional background/) or a
    folder of point clouds (one .ply/.pcd/.npz per frame, sensor frame).
    Centres are in the floor frame of 'bodyscan fuse' (use them as --center X Y).
    Without either flag every object is reported with both tests."""
    name = "detect"
    help = "find rotating objects and people in a recording"
    pipeline_class = DetectPipeline
    input_help = "capture directory or folder of point clouds"
    default_out = "detections"

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument("--rotation", action="store_true", help="keep objects turning about a vertical axis")
        parser.add_argument("--human", action="store_true", help="keep objects shaped like a standing person")

    def inputs(self, args):
        return {"input_dir": Path(args.input), "out": Path(args.out), "flags": (args.rotation, args.human)}
