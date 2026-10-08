"""Finding the object of interest in a recording, with no region of the room assumed.

One mechanism serves every pipeline:

    1. scene       the empty scene (background frames, empty-scene clouds, or
                   what most frames agree on) and the floor;
    2. foreground  in every frame, what is not the empty scene, between two
                   heights above the floor (Segmenter);
    3. objects     the foreground cut into clusters and followed across
                   frames (Tracker): one Track per object;
    4. selection   a chain of selectors (selectors.py): near a given point,
                   shaped like a person, turning about a vertical axis;
    5. result      per object a FoundObject: its centre (the rotation axis
                   when it turns, else its body centre), its horizontal reach
                   from that centre and its height range, measured on its own
                   points. A fusion pipeline builds its region of interest
                   from these measurements, not from a fixed place in the room.

'bodyscan detect' lists every object; 'bodyscan fuse' and 'fuse-inplace'
take the best selected one (the most points over the frames).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bodyscan.config import param
from bodyscan.detection.human import HumanCascade, HumanConfig, ViewSampling
from bodyscan.detection.rotation import RotationAnalyzer, RotationConfig
from bodyscan.detection.segmentation import (BackgroundSegmenter, ObjectSegmenter, PointBackgroundSegmenter,
                                             SegmentationConfig)
from bodyscan.detection.selectors import (HumanSelector, NearSelector, RotationSelector, SelectorChain, Verdict)
from bodyscan.detection.tracking import Track, Tracker, TrackingConfig
from bodyscan.log import Progress, info
from bodyscan.motion import consistent_frame_times


@dataclass
class SelectConfig:
    """Which object is the one of interest (the same tests in every command).

    'detect' keeps the objects that pass them, 'fuse' and 'fuse-inplace' scan
    the best one. No test: every object (detect), or the object with the most
    points over the frames (fusion)."""
    rotation: bool = param(False, "keep objects turning about a vertical axis (rotation test)",
                           effect="the axis found is the centre of the object, to a few mm")
    human: bool = param(False, "keep objects shaped like a standing person (Haar-like cascade)",
                        effect="rejects furniture, stands, boxes, walls; also people sitting or lying")
    near: tuple[float, float] | None = param(None, "keep objects whose centre is within near_radius of this "
                                                   "point (floor frame X Y)", unit="m",
                                             effect="chooses one object among several; not a crop region")
    near_radius: float = param(0.5, "distance for 'near'", unit="m")


@dataclass
class FoundObject:
    """One tracked object with the verdicts of the selectors and its measurements."""
    track: Track
    selected: bool
    verdicts: dict[str, Verdict]
    center: np.ndarray
    center_from: str
    bottom: float
    top: float
    part_of: int | None = None
    parts: list[int] = field(default_factory=list)

    @property
    def id(self) -> int:
        return self.track.id

    @property
    def points(self) -> np.ndarray:
        """All points of the object over its frames (floor frame, thinned)."""
        return np.concatenate([cluster.points for cluster in self.track.clusters])

    @property
    def rotating(self) -> bool:
        verdict = self.verdicts.get("rotation")
        return bool(verdict is not None and verdict.passed)

    @property
    def human(self) -> bool:
        verdict = self.verdicts.get("human")
        return bool(verdict is not None and verdict.passed)

    def reach(self, center=None, quantile: float = 99.5) -> float:
        """Horizontal distance from 'center' (default: its own) within which
        'quantile' per cent of its points lie, over every frame: for a turning
        object the radius of the volume it sweeps, arms included."""
        center = self.center if center is None else np.asarray(center, dtype=np.float64)
        return float(np.percentile(np.linalg.norm(self.points[:, :2] - center, axis=1), quantile))

    def as_dict(self, floor=None) -> dict:
        points = self.points
        entry = {"id": self.id, "frames": len(self.track.clusters), "points": int(len(points)),
                 "centroid": np.median(points[:, :2], axis=0), "top_m": self.top, "bottom_m": self.bottom,
                 "reach_m": round(self.reach(), 3),
                 "moved_m": float(np.linalg.norm(np.ptp(self.track.centroids[:, :2], axis=0))),
                 "center": self.center, "center_from": self.center_from, "selected": self.selected,
                 "part_of": self.part_of, "parts": self.parts}
        for name in ("human", "rotation", "near"):
            verdict = self.verdicts.get(name)
            entry[name] = None if verdict is None else verdict.details | {"passed": verdict.passed}
        if floor is not None:
            entry["center_sensor_frame"] = np.linalg.inv(floor.world_from_sensor)[:3, :3] @ (
                np.array([self.center[0], self.center[1], 0.0]) - floor.world_from_sensor[:3, 3])
        return entry


# ----------------------------------------------------------------------------
# Scene and frames
# ----------------------------------------------------------------------------

def spread_indices(count: int, wanted: int) -> np.ndarray:
    """'wanted' frame indices spread evenly over a recording of 'count' frames."""
    return np.unique(np.linspace(0, count - 1, min(wanted, count)).round().astype(int))


def frame_times(frames, indices, period: float) -> np.ndarray:
    """Time of each frame on one clock, from 0: sensor timestamps when there
    are, else the host time, else index * period."""
    sweep = np.array([np.median(f.timestamps[f.timestamps > 0]) if f.timestamps is not None and
                      np.any(f.timestamps > 0) else np.nan for f in frames])
    host = np.array([f.host_time for f in frames])
    if np.all(np.isfinite(host)) or np.any(np.isfinite(sweep)):
        times, _ = consistent_frame_times(np.full(len(frames), np.nan), sweep,
                                          np.where(np.isfinite(host), host, np.asarray(indices) * period))
    else:
        times = np.asarray(indices, dtype=np.float64) * period
    return times - times[0]


def scene_segmenter(source, floor, config: SegmentationConfig, background_range=None, static_points=None):
    """The foreground model of a recording: against the empty-scene range
    images, the empty-scene point clouds, or (no empty scene) everything above
    the floor minus the walls of the static scene. Returns (segmenter, mode)."""
    if background_range is not None:
        return BackgroundSegmenter(source, floor, config, background_range), "background (empty-scene frames)"
    if not source.organized and source.background_count():
        background = np.concatenate([source.load_background(k).points for k in range(source.background_count())])
        return (PointBackgroundSegmenter(source, floor, config, background),
                "background (empty-scene point clouds)")
    segmenter = ObjectSegmenter(source, floor, config, static_points)
    return segmenter, f"objects ({len(segmenter.walls)} walls removed)"


def selector_chain(select: SelectConfig, rotation: RotationConfig, human: HumanConfig,
                   sampling: ViewSampling, near=None) -> SelectorChain:
    """The selectors asked for by 'select' ('near' overrides select.near)."""
    selectors = []
    point = near if near is not None else select.near
    if point is not None:
        selectors.append(NearSelector(point, select.near_radius))
    if select.human:
        selectors.append(HumanSelector(HumanCascade(human, sampling)))
    if select.rotation:
        selectors.append(RotationSelector(RotationAnalyzer(rotation)))
    return SelectorChain(selectors)


def sampling_of(source, voxel: float, column_spacing_deg: float = 0.176, beam_spacing_deg: float = 0.70):
    """Angular sample spacing of the sensor (from its model, or given for point-cloud folders)."""
    if source.sensor is not None:
        horizontal, vertical = source.sensor.angular_steps()
    else:
        horizontal, vertical = np.radians(column_spacing_deg), np.radians(beam_spacing_deg)
    return ViewSampling(horizontal, vertical, voxel)


# ----------------------------------------------------------------------------
# Finder
# ----------------------------------------------------------------------------

class ObjectFinder:
    """Foreground objects of a recording, followed across frames and judged by
    a chain of selectors.

    chain        the tests an object must pass to be selected
    informative  tests run on every object for the report only (detect without
                 flags runs the person and rotation tests on everything)
    body_radius  a person's centre lies this far behind the visible surface,
                 when no rotation axis gives it
    """

    def __init__(self, segmenter, tracking: TrackingConfig, chain: SelectorChain,
                 informative: SelectorChain | None = None, body_radius: float = 0.10):
        self.segmenter, self.tracking, self.chain = segmenter, tracking, chain
        self.informative = informative
        self.body_radius = body_radius

    def tracks(self, source, indices, period: float = 0.1, keep=None) -> list[Track]:
        """Objects in the frames 'indices' of 'source', followed across them.
        Frames are read one at a time; keep(frame) can hold on to some of them."""
        tracker = Tracker(self.tracking)
        frames, clusters_of = [], []
        progress = Progress("frames", len(indices))
        for n, index in enumerate(indices):
            frame = source.load(int(index))
            clusters = self.segmenter.segment(frame, 0.0)
            tracker.update(clusters)
            clusters_of.append(clusters)
            frames.append(_TimeOnly(frame.timestamps, frame.host_time))
            if keep is not None:
                keep(frame)
            progress.maybe(n + 1, 10)
        times = frame_times(frames, indices, period)
        for clusters, time in zip(clusters_of, times):
            for cluster in clusters:
                cluster.time = float(time)
        self.times = times
        return tracker.result()

    def classify(self, tracks: list[Track]) -> list[FoundObject]:
        sensor_xy = self.segmenter.floor.sensor_position[:2]
        found = []
        for track in tracks:
            selected, verdicts = self.chain.evaluate(track)
            if self.informative is not None:
                for name, verdict in self.informative.evaluate(track, every=True)[1].items():
                    verdicts.setdefault(name, verdict)
            points = np.concatenate([cluster.points for cluster in track.clusters])
            centroid = np.median(points[:, :2], axis=0)
            rotation, human = verdicts.get("rotation"), verdicts.get("human")
            if rotation is not None and rotation.axis is not None:
                center, source = rotation.axis, "rotation axis"
            elif human is not None and human.passed:
                away = centroid - sensor_xy
                center = centroid + self.body_radius * away / max(np.linalg.norm(away), 1e-6)
                source = "visible surface moved back by the body radius"
            else:
                center, source = centroid, "median of its points"
            found.append(FoundObject(track, selected, verdicts, np.asarray(center, dtype=np.float64), source,
                                     float(np.percentile(points[:, 2], 0.5)),
                                     float(np.percentile(points[:, 2], 99.5))))
        return found

    def find(self, source, indices, period: float = 0.1, keep=None) -> list[FoundObject]:
        tracks = self.tracks(source, indices, period, keep)
        info(f"{len(tracks)} objects seen in at least {self.tracking.min_frames} frames"
             + (f"; tests: {', '.join(self.chain.names)}" if self.chain.selectors else "; no test asked"))
        return self.classify(tracks)

    @staticmethod
    def group_parts(objects: list[FoundObject], config: RotationConfig) -> None:
        """Rotating objects turning about the same axis at the same speed are parts
        of one body (the segmentation can split an arm from the torso): the part
        with the most points stands for the body, the others are marked as its
        parts and never selected on their own."""
        rotating = sorted((o for o in objects if o.rotating), key=lambda o: -len(o.points))
        for k, main in enumerate(rotating):
            if main.part_of is not None:
                continue
            speed = main.verdicts["rotation"].details["speed_deg_s"]
            for other in rotating[k + 1:]:
                if other.part_of is not None:
                    continue
                close = np.linalg.norm(other.center - main.center) <= config.group_distance
                other_speed = other.verdicts["rotation"].details["speed_deg_s"]
                if close and abs(other_speed - speed) <= config.group_speed * abs(speed):
                    other.part_of, other.selected = main.id, False
                    main.parts.append(other.id)

    @staticmethod
    def best(objects: list[FoundObject]) -> FoundObject | None:
        """The selected object with the most points over its frames (seen
        longest and largest), parts of another object excluded."""
        candidates = [o for o in objects if o.selected and o.part_of is None]
        return max(candidates, key=lambda o: len(o.points)) if candidates else None


@dataclass
class _TimeOnly:
    """The time fields of a frame (the frame itself is not kept)."""
    timestamps: np.ndarray | None
    host_time: float


def object_table(objects: list[FoundObject]) -> None:
    """One line per object on the console: tests, centre, reason of rejection."""
    info(f"\n{'id':>3} {'frames':>6} {'top':>6} {'center x':>9} {'center y':>9} {'reach':>6}  "
         f"{'human':>6} {'rotating':>9} {'speed':>9}  note")
    for o in objects:
        human, rotation = o.verdicts.get("human"), o.verdicts.get("rotation")
        speed = rotation.details.get("speed_deg_s", 0.0) if rotation is not None else 0.0
        failed = [f"{name}: {v.reason}" for name, v in o.verdicts.items() if not v.passed and v.reason]
        info(f"{o.id:>3} {len(o.track.clusters):>6} {o.top:>6.2f} {o.center[0]:>9.3f} {o.center[1]:>9.3f} "
             f"{o.reach():>6.2f}  "
             f"{'-' if human is None else ('yes' if human.passed else 'no'):>6} "
             f"{'-' if rotation is None else ('yes' if rotation.passed else 'no'):>9} "
             f"{speed:>7.2f}/s  "
             + ("SELECTED " if o.selected else "")
             + (f"part of {o.part_of} " if o.part_of is not None else "")
             + (f"parts {o.parts} " if o.parts else "")
             + ("; ".join(failed) if not o.selected else ""))
