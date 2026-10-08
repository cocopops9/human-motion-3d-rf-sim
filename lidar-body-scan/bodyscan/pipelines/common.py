"""Steps and helpers shared by the fusion pipelines (turntable and in place)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bodyscan.config import param
from bodyscan.detection.finder import (FoundObject, ObjectFinder, SelectConfig, object_table, sampling_of,
                                       scene_segmenter, selector_chain, spread_indices)
from bodyscan.detection.human import HumanConfig
from bodyscan.detection.rotation import RotationConfig
from bodyscan.detection.segmentation import SegmentationConfig
from bodyscan.detection.tracking import TrackingConfig
from bodyscan.io import NpzRecording
from bodyscan.log import info
from bodyscan.pipelines.base import Context, Step
from bodyscan.scene import RangeBackground, RansacFloor


@dataclass
class FrameConfig:
    """Reading the frames."""
    min_range: float = param(0.3, "returns closer than this are ignored", unit="m")
    max_range: float = param(10.0, "returns farther than this are ignored", unit="m")


class LoadRecording(Step):
    """A capture directory with its empty-scene frames."""
    name = "load"

    def run(self, ctx: Context) -> None:
        c = ctx.config
        ctx.source = NpzRecording(ctx.run_dir, c.frames.min_range, c.frames.max_range, require_background=True)
        ctx.report["run"] = str(Path(ctx.run_dir).resolve())


class SceneFromBackground(Step):
    """Floor frame and background model from the empty-scene frames
    (needs the sections 'floor' and 'isolation' in the configuration)."""
    name = "scene"

    def __init__(self, floor_estimator=None):
        self.floor_estimator = floor_estimator

    def run(self, ctx: Context) -> None:
        c = ctx.config
        source = ctx.source
        background = source.background_range(0.5)
        points = source.sensor.xyz(background)
        points = points[np.isfinite(points[..., 0])]
        floor = (self.floor_estimator or RansacFloor(c.floor)).estimate(points)
        ctx.floor = floor
        ctx.background_range = background
        ctx.background_world = floor.to_world(points)
        ctx.background = RangeBackground(background, c.isolation.bg_threshold, c.isolation.bg_relative)
        info(f"background: {source.background_count()} frames; sensor {floor.sensor_height:.3f} m above the "
             f"floor, tilt of its axis from the up direction {floor.tilt_deg:.1f} deg")
        ctx.report.update({"sensor_height_m": floor.sensor_height, "sensor_tilt_deg": floor.tilt_deg,
                           "world_from_sensor": floor.world_from_sensor})


@dataclass
class SubjectConfig:
    """Finding the object to scan in the frames, with the tests of [select].

    Its region of interest is measured on it: a vertical cylinder
    about its centre (the rotation axis, or its body centre) reaching its
    farthest points plus a margin; nothing is assumed about where it stands."""
    search_frames: int = param(60, "frames spread over the recording used to find it",
                               effect="the rotation test wants consecutive ones less than 45 deg of turn apart")
    margin: float = param(0.15, "the region of interest reaches this beyond the farthest point of the object "
                                "found (its arms and hands included)", unit="m",
                          effect="smaller may cut a hand held out in frames not used to find it; larger keeps "
                                 "more of whatever stands next to it (removed later as a separate cluster)")
    body_depth: float = param(0.10, "a person's centre lies this far behind the visible surface, when no "
                                     "rotation axis gives it", unit="m")
    min_total_turn: float = param(0.0, "with --rotation: the object must turn at least this much over the "
                                       "recording (360: a whole lap)", unit="deg")


def find_subject(source, floor, background_range, select: SelectConfig, subject: SubjectConfig,
                 near=None, period: float = 0.1) -> tuple[FoundObject | None, list[FoundObject]]:
    """The object of interest of a recording with empty-scene frames: the
    foreground of subject.search_frames frames, followed across them and
    judged by the [select] tests (detection.finder, the same as 'bodyscan
    detect'); 'near' (X Y) adds the test 'near this point'. Returns the best
    selected object (or None) and every object found."""
    segmentation = SegmentationConfig()
    segmenter, _ = scene_segmenter(source, floor, segmentation, background_range)
    rotation = RotationConfig(min_total_turn=subject.min_total_turn)
    chain = selector_chain(select, rotation, HumanConfig(), sampling_of(source, segmentation.voxel), near)
    finder = ObjectFinder(segmenter, TrackingConfig(), chain, body_radius=subject.body_depth)
    objects = finder.find(source, spread_indices(len(source), subject.search_frames), period)
    ObjectFinder.group_parts(objects, rotation)
    object_table(objects)
    return ObjectFinder.best(objects), objects


class FindSubject(Step):
    """The object to scan and its region of interest (needs the sections
    'select', 'subject' and 'isolation'). Writes ctx.subject (FoundObject),
    ctx.center (X Y, floor frame) and ctx.region_radius."""
    name = "subject"

    def near(self, ctx: Context):
        """A point the object must be near ([select] near), or None."""
        return None

    def run(self, ctx: Context) -> None:
        c = ctx.config
        sel = c.select
        tests = [name for name, on in (("rotating", sel.rotation), ("person", sel.human)) if on]
        near = self.near(ctx)
        info("finding the object to scan: " + (" and ".join(tests) if tests else "the largest object seen in "
                                                 "the most frames (no test asked)")
             + ("" if near is None and sel.near is None else ", near the point given"))
        subject, objects = find_subject(ctx.source, ctx.floor, ctx.background_range, sel, c.subject, near)
        ctx.report["objects"] = [o.as_dict(ctx.floor) for o in objects]
        if subject is None:
            raise SystemExit(
                "no object passed the tests asked for (see the table above). Change the tests "
                "(--rotation/--no-rotation, --human/--no-human, --near X Y), or impose the centre with --center "
                "X Y and the region with --radius R")
        ctx.subject = subject
        ctx.center = subject.center.copy()
        info(f"object {subject.id}: centre ({subject.center[0]:.3f}, {subject.center[1]:.3f}) m "
             f"[{subject.center_from}], reach {subject.reach():.2f} m, height {subject.bottom:.2f} to "
             f"{subject.top:.2f} m")
        ctx.report["subject"] = subject.as_dict(ctx.floor)

    @staticmethod
    def region_radius(ctx: Context, center) -> float:
        """Radius of the region of interest about 'center': [isolation] radius
        when given, else the reach of the object found plus the margin."""
        radius = ctx.config.isolation.radius
        if radius is None:
            radius = ctx.subject.reach(center) + ctx.config.subject.margin
        return float(radius)
