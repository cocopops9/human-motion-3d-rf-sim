"""Detection pipeline: where are the rotating objects and the people in a recording?

    LoadFrames        a capture directory (lut.npz + frames) or a folder of point clouds;
                      max_frames frames spread over the recording
    FindScene         floor frame (from the empty scene, or from the static part of the
                      frames) and the segmenter (background or objects mode)
    SegmentAndTrack   objects in every frame, followed across frames
    Classify          rotation test (--rotation) and person cascade (--human) per object
    WriteDetections   <out>.json, a table on the console, <out>_top.png

All coordinates are in the floor frame of the fusion (z up, z = 0 on the
floor, origin below the sensor): the axis of a rotating person is exactly
what 'bodyscan fuse --center X Y' expects. Both flags together keep the
objects that are rotating AND people.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bodyscan.config import param, section
from bodyscan.detection.human import HumanCascade, HumanConfig, ViewSampling
from bodyscan.detection.rotation import RotationAnalyzer, RotationConfig
from bodyscan.detection.segmentation import (BackgroundSegmenter, ObjectSegmenter, PointBackgroundSegmenter,
                                             SegmentationConfig)
from bodyscan.detection.tracking import Tracker, TrackingConfig
from bodyscan.io import NpzRecording, median_range, open_recording
from bodyscan.jsonio import write_json
from bodyscan.log import Progress, info
from bodyscan.motion import consistent_frame_times
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
    frames: DetectFrameConfig = section(DetectFrameConfig)
    floor: FloorConfig = section(FloorConfig)
    segmentation: SegmentationConfig = section(SegmentationConfig)
    tracking: TrackingConfig = section(TrackingConfig)
    rotation: RotationConfig = section(RotationConfig)
    human: HumanConfig = section(HumanConfig)


class LoadFrames(Step):
    name = "load"

    def run(self, ctx: Context) -> None:
        c = ctx.config.frames
        source = open_recording(ctx.input_dir, c.min_range, c.max_range, c.period)
        count = len(source)
        chosen = np.unique(np.linspace(0, count - 1, min(c.max_frames, count)).round().astype(int))
        frames = [source.load(int(k)) for k in chosen]
        sweep = np.array([np.median(f.timestamps[f.timestamps > 0]) if f.timestamps is not None and
                          np.any(f.timestamps > 0) else np.nan for f in frames])
        host = np.array([f.host_time for f in frames])
        if np.all(np.isfinite(host)) or np.any(np.isfinite(sweep)):
            times, _ = consistent_frame_times(np.full(len(frames), np.nan), sweep,
                                              np.where(np.isfinite(host), host, chosen * c.period))
        else:
            times = chosen * c.period
        ctx.source, ctx.frames, ctx.times = source, frames, times - times[0]
        info(f"{len(frames)} of {count} frames, {ctx.times[-1]:.1f} s of recording"
             + (f"; {source.background_count()} empty-scene frames" if source.background_count() else
                "; no empty-scene frames"))
        ctx.report["input"] = str(Path(ctx.input_dir).resolve())
        ctx.report["frames_used"] = chosen


class FindScene(Step):
    name = "scene"

    def run(self, ctx: Context) -> None:
        c = ctx.config
        source = ctx.source
        background = None
        background_points = None
        if isinstance(source, NpzRecording) and source.background_count():
            background = source.background_range(0.5)
            static = source.sensor.xyz(background)
            static = static[np.isfinite(static[..., 0])]
        elif not source.organized and source.background_count():
            background_points = np.concatenate([source.load_background(k).points
                                                for k in range(source.background_count())])
            static = background_points[::max(1, len(background_points) // 400000)]
        elif source.organized:
            # The static scene is what most frames agree on (people move, turn or leave).
            median = median_range([f.range_m for f in ctx.frames[::max(1, len(ctx.frames) // 15)]], 0.5)
            static = source.sensor.xyz(median)
            static = static[np.isfinite(static[..., 0])]
        else:
            static = np.concatenate([source.points(f)[::4] for f in ctx.frames[::max(1, len(ctx.frames) // 10)]])
        floor = RansacFloor(c.floor).estimate(static)
        ctx.floor = floor
        if background is not None:
            ctx.segmenter = BackgroundSegmenter(source, floor, c.segmentation, background)
            mode = "background (empty-scene frames)"
        elif background_points is not None:
            ctx.segmenter = PointBackgroundSegmenter(source, floor, c.segmentation, background_points)
            mode = "background (empty-scene point clouds)"
        else:
            ctx.segmenter = ObjectSegmenter(source, floor, c.segmentation, static)
            mode = f"objects ({len(ctx.segmenter.walls)} walls removed)"
        info(f"floor: sensor {floor.sensor_height:.3f} m above it, tilt {floor.tilt_deg:.1f} deg; segmentation: {mode}")
        ctx.report.update({"sensor_height_m": floor.sensor_height, "sensor_tilt_deg": floor.tilt_deg,
                           "world_from_sensor": floor.world_from_sensor, "segmentation": mode})


class SegmentAndTrack(Step):
    name = "track"

    def run(self, ctx: Context) -> None:
        tracker = Tracker(ctx.config.tracking)
        progress = Progress("frames", len(ctx.frames))
        for n, (frame, time) in enumerate(zip(ctx.frames, ctx.times)):
            tracker.update(ctx.segmenter.segment(frame, float(time)))
            progress.maybe(n + 1, 10)
        ctx.tracks = tracker.result()
        info(f"{len(ctx.tracks)} objects seen in at least {ctx.config.tracking.min_frames} frames")


class Classify(Step):
    """Rotation test and person cascade of every tracked object. A variant
    passes its own factories: cascade_factory(config, sampling) -> HumanCascade,
    analyzer_factory(config) -> RotationAnalyzer."""
    name = "classify"

    def __init__(self, cascade_factory=HumanCascade, analyzer_factory=RotationAnalyzer):
        self.cascade_factory, self.analyzer_factory = cascade_factory, analyzer_factory

    def run(self, ctx: Context) -> None:
        c = ctx.config
        want_rotation, want_human = ctx.flags
        analyzer = self.analyzer_factory(c.rotation)
        if ctx.source.sensor is not None:
            horizontal, vertical = ctx.source.sensor.angular_steps()
        else:
            horizontal, vertical = np.radians(c.frames.column_spacing), np.radians(c.frames.beam_spacing)
        cascade = self.cascade_factory(c.human, ViewSampling(horizontal, vertical, c.segmentation.voxel))
        floor = ctx.floor
        sensor_xy = floor.sensor_position[:2]
        results = []
        for track in ctx.tracks:
            points = np.concatenate([cl.points for cl in track.clusters])
            centroid = np.median(points[:, :2], axis=0)
            entry = {"id": track.id, "frames": len(track.clusters), "centroid": centroid,
                     "top_m": float(np.percentile(points[:, 2], 99.5)),
                     "moved_m": float(np.linalg.norm(np.ptp(track.centroids[:, :2], axis=0)))}
            human = cascade.track(track) if (want_human or not want_rotation) else None
            # With both flags the cheaper person test goes first: only people are
            # tested for rotation.
            test_rotation = (want_rotation or not want_human) and not (want_human and want_rotation
                                                                       and not human.human)
            rotation = analyzer.analyze(track) if test_rotation else None
            entry["human"] = None if human is None else human.as_dict()
            entry["rotation"] = None if rotation is None else rotation.as_dict()
            if rotation is not None and rotation.rotating:
                center, source = np.asarray(rotation.axis), "rotation axis"
            elif human is not None and human.human:
                away = centroid - sensor_xy
                center = centroid + c.frames.body_radius * away / max(np.linalg.norm(away), 1e-6)
                source = "visible surface moved back by the body radius"
            else:
                center, source = centroid, "centroid of the visible surface"
            entry["center"] = center
            entry["center_from"] = source
            entry["center_sensor_frame"] = np.linalg.inv(floor.world_from_sensor)[:3, :3] @ (
                np.array([center[0], center[1], 0.0]) - floor.world_from_sensor[:3, 3])
            selected = True
            if want_rotation:
                selected &= bool(rotation is not None and rotation.rotating)
            if want_human:
                selected &= bool(human is not None and human.human)
            entry["selected"] = selected
            entry["points"] = int(len(points))
            results.append(entry)
        self.group_parts(results, c.rotation)
        ctx.results = results
        ctx.report["objects"] = results
        ctx.report["flags"] = {"rotation": want_rotation, "human": want_human}


    @staticmethod
    def group_parts(results, config) -> None:
        """Rotating objects turning about the same axis at the same speed are parts
        of one body (the segmentation can split an arm from the torso): the part
        with the most points stands for the body, the others are marked as its
        parts and never selected on their own."""
        rotating = sorted((r for r in results if r["rotation"] and r["rotation"]["rotating"]),
                          key=lambda r: -r["points"])
        for r in results:
            r["part_of"], r["parts"] = None, []
        for k, main in enumerate(rotating):
            if main["part_of"] is not None:
                continue
            axis, speed = np.asarray(main["rotation"]["axis"]), main["rotation"]["speed_deg_s"]
            for other in rotating[k + 1:]:
                if other["part_of"] is not None:
                    continue
                close = np.linalg.norm(np.asarray(other["rotation"]["axis"]) - axis) <= config.group_distance
                same_speed = abs(other["rotation"]["speed_deg_s"] - speed) <= config.group_speed * abs(speed)
                if close and same_speed:
                    other["part_of"], other["selected"] = main["id"], False
                    main["parts"].append(other["id"])


class WriteDetections(Step):
    name = "write"

    def run(self, ctx: Context) -> None:
        selected = [r for r in ctx.results if r["selected"]]
        want_rotation, want_human = ctx.flags
        what = " and ".join(w for w, flag in (("rotating", want_rotation), ("human", want_human)) if flag) or "all"
        info(f"\n{'id':>3} {'frames':>6} {'top':>6} {'center x':>9} {'center y':>9}  "
             f"{'human':>6} {'rotating':>9} {'speed':>9}  note")
        for r in ctx.results:
            human = r["human"]["human"] if r["human"] else None
            rotation = r["rotation"]
            info(f"{r['id']:>3} {r['frames']:>6} {r['top_m']:>6.2f} {r['center'][0]:>9.3f} {r['center'][1]:>9.3f}  "
                 f"{'yes' if human else ('no' if human is not None else '-'):>6} "
                 f"{'yes' if rotation and rotation['rotating'] else ('no' if rotation else '-'):>9} "
                 f"{(rotation['speed_deg_s'] if rotation else 0.0):>7.2f}/s  "
                 + ("SELECTED " if r["selected"] else "")
                 + (f"part of {r['part_of']} " if r["part_of"] is not None else "")
                 + (f"parts {r['parts']} " if r["parts"] else "")
                 + (rotation["reason"] if rotation and not rotation["rotating"] else ""))
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
        frame = ctx.frames[len(ctx.frames) // 2]
        world = ctx.floor.to_world(ctx.source.points(frame))
        keep = (world[:, 2] > 0.05) & (world[:, 2] < 2.4)
        sample = world[keep][::5]
        axis.scatter(sample[:, 0], sample[:, 1], s=0.3, c="0.75")
        colors = iter(plt.cm.tab10.colors * 10)
        for track, result in zip(ctx.tracks, ctx.results):
            color = next(colors)
            last = track.clusters[len(track.clusters) // 2].points
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
        return StepList([LoadFrames(), FindScene(), SegmentAndTrack(), Classify(), WriteDetections()])
