"""Detection pipeline: where are the rotating objects and the people in a recording?

    LoadFrames        a capture directory (lut.npz + frames) or a folder of point clouds;
                      max_frames frames spread over the recording
    FindScene         floor frame (from the empty scene, or from the static part of the
                      frames) and the foreground model (background or objects mode)
    FindObjects       objects in every frame, followed across frames, judged by the
                      selectors: --rotation, --human, --near X Y (detection.finder)
    WriteDetections   <out>.json, a table on the console, <out>_top.png

The same ObjectFinder finds the person of 'bodyscan fuse' and 'fuse-inplace',
with the same flags. All coordinates are in the floor frame of the fusion (z
up, z = 0 on the floor, origin below the sensor): the axis of a rotating
person is exactly what 'bodyscan fuse --center X Y' expects. Several flags
keep the objects that pass every test (rotating AND people).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bodyscan.config import param, section
from bodyscan.detection.finder import (ObjectFinder, SelectConfig, object_table, sampling_of, scene_segmenter,
                                       selector_chain, spread_indices)
from bodyscan.detection.human import HumanCascade, HumanConfig
from bodyscan.detection.rotation import RotationAnalyzer, RotationConfig
from bodyscan.detection.segmentation import SegmentationConfig
from bodyscan.detection.selectors import HumanSelector, RotationSelector, SelectorChain
from bodyscan.detection.tracking import TrackingConfig
from bodyscan.io import NpzRecording, median_range, open_recording
from bodyscan.jsonio import write_json
from bodyscan.log import info
from bodyscan.pipelines.base import Context, Pipeline, Step, StepList
from bodyscan.scene import FloorConfig, RansacFloor


@dataclass
class DetectFrameConfig:
    """Which frames are analysed."""
    max_frames: int = param(60, "frames used, spread evenly over the recording",
                            effect="more: slower, steadier decisions; the rotation test needs frames a few "
                                   "degrees of turn apart, so spreading them over the run is better than consecutive")
    min_range: float = param(0.3, "returns closer than this are ignored", unit="m")
    max_range: float = param(10.0, "returns farther than this are ignored", unit="m")
    period: float = param(0.1, "frame period of point-cloud folders (no timestamps)", unit="s")
    body_radius: float = param(0.10, "a person's centre lies this far behind the visible surface (seen from the "
                                     "sensor), when it is not given by a rotation axis", unit="m")
    beam_spacing: float = param(0.70, "vertical angle between beams, for point-cloud folders (range images "
                                      "take it from the sensor model); OS0-128: 0.70, OS0-64: 1.41", unit="deg")
    column_spacing: float = param(0.176, "horizontal angle between columns, for point-cloud folders "
                                         "(2048 columns: 0.176, 1024: 0.352)", unit="deg")


@dataclass
class DetectConfig:
    select: SelectConfig = section(SelectConfig)
    frames: DetectFrameConfig = section(DetectFrameConfig)
    floor: FloorConfig = section(FloorConfig)
    segmentation: SegmentationConfig = section(SegmentationConfig)
    tracking: TrackingConfig = section(TrackingConfig)
    rotation: RotationConfig = section(RotationConfig)
    human: HumanConfig = section(HumanConfig)


class LoadFrames(Step):
    """The recording and the frames used (read later, one at a time)."""
    name = "load"

    def run(self, ctx: Context) -> None:
        c = ctx.config.frames
        source = open_recording(ctx.input_dir, c.min_range, c.max_range, c.period)
        ctx.source = source
        ctx.indices = spread_indices(len(source), c.max_frames)
        info(f"{len(ctx.indices)} of {len(source)} frames"
             + (f"; {source.background_count()} empty-scene frames" if source.background_count() else
                "; no empty-scene frames"))
        ctx.report["input"] = str(Path(ctx.input_dir).resolve())
        ctx.report["frames_used"] = ctx.indices


class FindScene(Step):
    """Floor and foreground model."""
    name = "scene"

    def run(self, ctx: Context) -> None:
        c = ctx.config
        source = ctx.source
        background = None
        static = None
        if isinstance(source, NpzRecording) and source.background_count():
            background = source.background_range(0.5)
            static = source.sensor.xyz(background)
            static = static[np.isfinite(static[..., 0])]
        elif not source.organized and source.background_count():
            points = np.concatenate([source.load_background(k).points for k in range(source.background_count())])
            static = points[::max(1, len(points) // 400000)]
        else:
            some = [source.load(int(k)) for k in ctx.indices[::max(1, len(ctx.indices) // 15)]]
            if source.organized:
                # The static scene is what most frames agree on (people move, turn or leave).
                median = median_range([f.range_m for f in some], 0.5)
                static = source.sensor.xyz(median)
                static = static[np.isfinite(static[..., 0])]
            else:
                static = np.concatenate([source.points(f)[::4] for f in some[:10]])
        floor = RansacFloor(c.floor).estimate(static)
        ctx.floor = floor
        ctx.segmenter, mode = scene_segmenter(source, floor, c.segmentation, background, static)
        info(f"floor: sensor {floor.sensor_height:.3f} m above it, tilt {floor.tilt_deg:.1f} deg; segmentation: {mode}")
        ctx.report.update({"sensor_height_m": floor.sensor_height, "sensor_tilt_deg": floor.tilt_deg,
                           "world_from_sensor": floor.world_from_sensor, "segmentation": mode})


class FindObjects(Step):
    """Objects followed across the frames and judged by the selectors asked
    for. Without any, the person and rotation tests are run on every object
    for the report, and every object is listed as selected. A variant passes
    its own factories: cascade_factory(config, sampling) -> HumanCascade,
    analyzer_factory(config) -> RotationAnalyzer."""
    name = "objects"

    def __init__(self, cascade_factory=HumanCascade, analyzer_factory=RotationAnalyzer):
        self.cascade_factory, self.analyzer_factory = cascade_factory, analyzer_factory

    def run(self, ctx: Context) -> None:
        c = ctx.config
        sampling = sampling_of(ctx.source, c.segmentation.voxel, c.frames.column_spacing, c.frames.beam_spacing)
        chain = selector_chain(c.select, c.rotation, c.human, sampling)
        for k, selector in enumerate(chain.selectors):           # the factories of a variant
            if isinstance(selector, HumanSelector):
                chain.selectors[k] = HumanSelector(self.cascade_factory(c.human, sampling))
            elif isinstance(selector, RotationSelector):
                chain.selectors[k] = RotationSelector(self.analyzer_factory(c.rotation))
        informative = None
        if not (c.select.rotation or c.select.human):
            informative = SelectorChain([HumanSelector(self.cascade_factory(c.human, sampling)),
                                         RotationSelector(self.analyzer_factory(c.rotation))])
        finder = ObjectFinder(ctx.segmenter, c.tracking, chain, informative, c.frames.body_radius)
        middle = int(ctx.indices[len(ctx.indices) // 2])

        def keep(frame):
            if frame.index == middle:
                ctx.middle_frame = frame

        ctx.objects = finder.find(ctx.source, ctx.indices, c.frames.period, keep)
        ctx.times = finder.times
        info(f"{ctx.times[-1]:.1f} s of recording")
        ObjectFinder.group_parts(ctx.objects, c.rotation)
        ctx.results = [o.as_dict(ctx.floor) for o in ctx.objects]
        ctx.report["objects"] = ctx.results
        ctx.report["select"] = {"rotation": c.select.rotation, "human": c.select.human, "near": c.select.near}


class WriteDetections(Step):
    name = "write"

    def run(self, ctx: Context) -> None:
        select = ctx.config.select
        selected = [r for r in ctx.results if r["selected"]]
        what = " and ".join(w for w, flag in (("rotating", select.rotation), ("human", select.human),
                                              ("near", select.near is not None)) if flag) or "all"
        object_table(ctx.objects)
        info(f"\n{len(selected)} {what} object(s) found")
        for r in selected:
            info(f"  object {r['id']}: centre ({r['center'][0]:.3f}, {r['center'][1]:.3f}) m in the floor frame "
                 f"[{r['center_from']}] -> bodyscan fuse ... --center {r['center'][0]:.3f} {r['center'][1]:.3f}")
        out = Path(ctx.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        ctx.report["selected"] = [r["id"] for r in selected]
        write_json(f"{out}.json", ctx.report)
        plotted = self.plot(ctx, f"{out}_top.png")
        info(f"wrote {out}.json" + (f", {out}_top.png" if plotted else ""))

    @staticmethod
    def plot(ctx, path) -> bool:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return False
        figure, axis = plt.subplots(figsize=(8, 8))
        world = ctx.floor.to_world(ctx.source.points(ctx.middle_frame))
        keep = (world[:, 2] > 0.05) & (world[:, 2] < 2.4)
        sample = world[keep][::5]
        axis.scatter(sample[:, 0], sample[:, 1], s=0.3, c="0.75")
        colors = iter(plt.cm.tab10.colors * 10)
        for found, result in zip(ctx.objects, ctx.results):
            color = next(colors)
            last = found.track.clusters[len(found.track.clusters) // 2].points
            axis.scatter(last[:, 0], last[:, 1], s=1, color=color)
            marker = "*" if result["selected"] else "x"
            axis.plot(result["center"][0], result["center"][1], marker,
                      color="k" if result["selected"] else color, ms=12)
            axis.annotate(f"{result['id']}", result["center"], textcoords="offset points", xytext=(6, 6), fontsize=9)
        axis.plot(*ctx.floor.sensor_position[:2], "r^", ms=10)
        axis.set_aspect("equal")
        axis.grid(alpha=0.3)
        axis.set_title("objects (colour) and centres; * = selected; red triangle = sensor")
        figure.tight_layout()
        figure.savefig(path, dpi=100)
        plt.close(figure)
        return True


class DetectPipeline(Pipeline):
    config_class = DetectConfig
    name = "detect"

    def steps(self) -> StepList:
        return StepList([LoadFrames(), FindScene(), FindObjects(), WriteDetections()])
