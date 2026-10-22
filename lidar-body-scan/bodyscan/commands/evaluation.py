"""Commands: evaluate-take (fit quality of a tracked take, with body heatmaps), bench-rig
(coverage of the body by candidate sensor layouts) and view (the interactive viewer)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bodyscan.commands.base import Command, ConfiguredCommand
from bodyscan.config import section
from bodyscan.evaluation.coverage import BENCH_MOTIONS, CoverageConfig
from bodyscan.evaluation.take import TakeEvaluationConfig
from bodyscan.log import info


@dataclass
class EvaluateTakeConfig:
    evaluation: TakeEvaluationConfig = section(TakeEvaluationConfig)


@dataclass
class BenchRigConfig:
    coverage: CoverageConfig = section(CoverageConfig)


class EvaluateTake(ConfiguredCommand):
    """Fit quality of a tracked take against its measurements, with body heatmaps.

        python -m bodyscan evaluate-take C:\\\\lidar\\\\cluster_results_v160\\\\results\\\\s02_walk01\\\\motion.npz
            --segments C:\\\\lidar\\\\open\\\\s02_walk01_seg --avatar C:\\\\lidar\\\\s02_avatar.npz
            --model SMPLX_MALE.npz --out C:\\\\lidar\\\\evaluation_v160\\\\s02_walk01

    Every pixel around the person is compared, along its ray, with the body posed at
    the pixel's time: fitted, violation (the body stands where the sensor saw
    through), unexplained (the sensor measured the person where the body has no
    surface), silhouette or unobserved. Writes metrics.json, metrics.npz,
    heatmap_median.ply, heatmap_p95.ply, heatmap_violation.ply, heatmap_coverage.ply
    and report.md. Regions no sensor observed have no error value (grey). Needs
    PyTorch."""
    name = "evaluate-take"
    help = "fit quality of a tracked take, per frame, per body part and per vertex (heatmaps)"
    config_class = EvaluateTakeConfig

    def add_arguments(self, parser):
        parser.add_argument("motion", help="motion .npz (bodyscan track)")
        parser.add_argument("--segments", required=True, help="segment-motion folder of the take")
        parser.add_argument("--avatar", required=True, help="avatar .npz used for the tracking")
        parser.add_argument("--model", default=None, help="model file, if it moved since the avatar was fitted")
        parser.add_argument("--out", required=True, help="output folder")
        self.add_config_arguments(parser)

    def run(self, args):
        config = self.prepare(args)
        if config is None:
            return 0
        from bodyscan.evaluation.take import evaluate_take, report_text, write_take_results
        result = evaluate_take(args.motion, args.segments, args.avatar, args.model, config.evaluation)
        write_take_results(result, args.out, config.evaluation.min_coverage)
        info(report_text({k: v for k, v in result.items() if k != "arrays"}).split("\n\n| part")[0])
        return 0


class BenchRig(ConfiguredCommand):
    """Coverage of the body by candidate layouts of the Ouster, the Helios and the Xtion.

        python -m bodyscan bench-rig --out bench_rig
        python -m bodyscan bench-rig --layouts ouster triangle-120 opposite-90 --motions walk-across jump-side
            --avatar C:\\\\lidar\\\\s02_avatar.npz --model SMPLX_MALE.npz --out bench_rig_s02

    For every layout and motion: the share of every body part each sensor sees
    (field of view, facing, self-occlusion), for the union of the sensors, and
    the sample spacing on the visible surface. Writes bench_rig.json and
    bench_rig.md. Layouts are listed in bodyscan.sensors.layouts. Needs PyTorch."""
    name = "bench-rig"
    help = "coverage of the body by candidate sensor layouts (placement study)"
    config_class = BenchRigConfig

    def add_arguments(self, parser):
        from bodyscan.sensors.layouts import layout_names
        parser.add_argument("--layouts", nargs="+", default=layout_names(), help="layouts to compare (default: all)")
        parser.add_argument("--motions", nargs="+", default=list(BENCH_MOTIONS), choices=list(BENCH_MOTIONS),
                            help="bench motions (default: all)")
        parser.add_argument("--zone-distance", type=float, default=2.3,
                            help="distance of the capture zone centre from the Ouster [m]")
        parser.add_argument("--avatar", default=None, help="avatar .npz of the subject (default: the test body)")
        parser.add_argument("--model", default=None, help="model file of the avatar, or a body model file")
        parser.add_argument("--out", required=True, help="output folder")
        self.add_config_arguments(parser)

    def run(self, args):
        config = self.prepare(args)
        if config is None:
            return 0
        from bodyscan.evaluation.coverage import bench_rig
        bench_rig(args.layouts, args.motions, Path(args.out), args.model, args.avatar, config.coverage,
                  args.zone_distance)
        return 0


class View(Command):
    """Interactive viewer of tracked takes in the web browser (replaces the review videos).

        python -m bodyscan view --results C:\\\\lidar\\\\cluster_results_v160\\\\results --segments C:\\\\lidar\\\\open
            --avatar C:\\\\lidar\\\\s02_avatar.npz --model SMPLX_MALE.npz --evaluations C:\\\\lidar\\\\evaluation_v160

    Then open http://localhost:8080. Choose the motion type and the take, Load,
    then scrub or play; show the person's points (by sensor or by distance to the
    body), the body (plain, guessed limbs, or a heatmap: this frame's error, the
    take's median or p95 error, violation rate, coverage), the skeleton and the
    sensor; the panel gives the numbers of the frame and of the take. Needs
    PyTorch and viser (python -m pip install viser)."""
    name = "view"
    help = "interactive viewer of tracked takes (points, body, heatmaps, numbers) in the browser"

    def add_arguments(self, parser):
        parser.add_argument("--results", required=True, help="folder with one subfolder per take holding motion.npz")
        parser.add_argument("--segments", default=None, help="folder with the <take>_seg segment-motion folders")
        parser.add_argument("--evaluations", default=None, help="folder with the evaluate-take output of every take")
        parser.add_argument("--avatar", required=True, help="avatar .npz used for the tracking")
        parser.add_argument("--model", default=None, help="model file, if it moved since the avatar was fitted")
        parser.add_argument("--take", default=None, help="take to load at start")
        parser.add_argument("--port", type=int, default=8080)
        parser.add_argument("--host", default="0.0.0.0")

    def run(self, args):
        try:
            import viser  # noqa: F401
        except ImportError:
            raise SystemExit("the viewer needs viser: python -m pip install viser")
        from bodyscan.viewer.app import run_viewer
        run_viewer(args.results, args.segments, args.evaluations, args.avatar, args.model, args.port, args.host,
                   args.take)
        return 0
