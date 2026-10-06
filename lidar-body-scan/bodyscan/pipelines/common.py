"""Steps and helpers shared by the fusion pipelines (turntable and in place)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bodyscan.config import param
from bodyscan.detection.human import HumanCascade, HumanConfig, ViewSampling
from bodyscan.detection.segmentation import BackgroundSegmenter, SegmentationConfig
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


def locate_person(source, floor, background_range, frames: int = 12, body_radius: float = 0.10,
                  human: HumanConfig | None = None, segmentation: SegmentationConfig | None = None):
    """Horizontal position (floor frame) of the standing person of a recording
    with empty-scene frames, or None: the foreground of 'frames' frames spread
    over the run is cut into objects, the objects that pass the person cascade
    (bodyscan detect --human) give their centre (the visible surface moved
    back by body_radius), and the median of these centres is returned with
    the number of frames in which a person was found."""
    segmenter = BackgroundSegmenter(source, floor, segmentation or SegmentationConfig(), background_range)
    horizontal, vertical = source.sensor.angular_steps()
    cascade = HumanCascade(human or HumanConfig(), ViewSampling(horizontal, vertical, segmenter.config.voxel))
    sensor_xy = floor.sensor_position[:2]
    centers = []
    for index in np.unique(np.linspace(0, len(source) - 1, min(frames, len(source))).round().astype(int)):
        frame = source.load(int(index))
        best = None
        for cluster in segmenter.segment(frame, 0.0):
            result = cascade.frame(cluster.points, cluster.sensor)
            if result.passed and (best is None or len(cluster.points) > best[1]):
                center = np.median(cluster.points[:, :2], axis=0)
                away = center - sensor_xy
                best = (center + body_radius * away / max(np.linalg.norm(away), 1e-6), len(cluster.points))
        if best is not None:
            centers.append(best[0])
    if not centers:
        return None, 0
    return np.median(np.array(centers), axis=0), len(centers)
