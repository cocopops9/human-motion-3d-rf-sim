"""Cutting a moving person out of every frame of a recording (numpy and Open3D only).

For every frame:

    1. foreground   pixels closer than the empty scene (background frames of
                    the recording), minus mixed edge pixels, within the height
                    and distance limits, in the floor frame
    2. person       with a prediction of where the person is (previous frames,
                    constant velocity): every foreground cluster inside a
                    vertical cylinder of radius 'gate' around it (an arm cut
                    off by a gap stays with the body); without one (first frame,
                    or the person was lost): the largest cluster that stands
                    at least 0.8 m tall
    3. silhouette   a crop of the range image around the person with, per
                    pixel, 1 = a return of the person, 2 = no return where the
                    empty scene always returns, next to the person: light that
                    the person blocked without sending it back (dark fabric,
                    grazing angles), 0 = anything else; with the measured and
                    the empty-scene ranges and the time of every pixel of the
                    crop. The tracker uses the crop to keep the body out of
                    the pixels where the sensor saw through to the background.

Output folder (read by bodyscan.dynamic.segment.SegmentedRecording):

    scene.npz            floor frame, empty-scene range image, lookup table, pixel shift
    frames/person_XXXXX.npz   one PersonFrame per frame (frames without a person are skipped)
    segments.json        settings, per-frame counts and positions, warnings
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from bodyscan.config import param
from bodyscan.dynamic.image import RangeImageGeometry
from bodyscan.geometry import clusters_3d
from bodyscan.io import NpzRecording, natural_key
from bodyscan.log import Progress, info, warning
from bodyscan.scene import FloorConfig, RangeBackground, RansacFloor, mixed_pixel_mask


@dataclass
class MotionSegmentationConfig:
    """Cutting the moving person out of every frame (floor frame: z up, z = 0 on the floor)."""
    bg_threshold: float = param(0.08, "a pixel is foreground if closer than the empty scene by this", unit="m",
                                effect="smaller keeps the soles closer to the floor but lets floor noise in")
    bg_relative: float = param(0.015, "... or by this fraction of its range, whichever is larger")
    edge_jump: float = param(0.05, "mixed-pixel filter: range jump on both sides of a pixel (0 = off)", unit="m")
    min_height: float = param(0.01, "drop points below this height above the floor", unit="m")
    max_height: float = param(2.4, "drop points above this height", unit="m")
    max_distance: float = param(9.0, "drop points farther than this from the sensor (horizontal)", unit="m")
    cluster_distance: float = param(0.12, "points closer than this belong to the same cluster", unit="m")
    min_points: int = param(40, "smallest cluster kept")
    gate: float = param(0.9, "the person is everything within this horizontal radius of where they are "
                             "predicted to be", unit="m",
                        effect="must cover outstretched arms; larger may take in furniture next to the person")
    min_person_height: float = param(0.8, "a new person (first frame, or after being lost) stands at least this "
                                          "tall", unit="m")
    lost_after: int = param(5, "frames without the person before searching the whole scene again")
    dark_pixels: bool = param(True, "count pixels without return where the empty scene always returns, next to "
                                    "the person, as part of the silhouette (dark clothes)")
    dark_reliability: float = param(0.95, "fraction of the empty-scene frames in which such a pixel must return")
    dark_reach: int = param(3, "dark pixels count within this many pixels of a person return")
    crop_rows: int = param(6, "rows kept above and below the person in the silhouette crop")
    crop_columns: int = param(16, "columns kept on both sides of the person in the silhouette crop")
    min_columns: float = param(0.9, "drop a frame that received less than this fraction of its columns")


@dataclass
class PersonFrame:
    """The person in one frame (floor frame). Columns are image columns;
    crop_origin[1] may be negative or beyond the width (the crop is unwrapped
    around the person, indices taken modulo the width)."""
    index: int
    time: float
    points: np.ndarray
    point_times: np.ndarray
    pixels: np.ndarray
    ranges: np.ndarray
    crop_origin: tuple
    crop_mask: np.ndarray
    crop_range: np.ndarray
    crop_background: np.ndarray
    center: np.ndarray
    quality: dict = field(default_factory=dict)
    crop_times: np.ndarray | None = None          # sensor time of every crop pixel [s]

    def save(self, path) -> None:
        np.savez_compressed(path, index=np.int64(self.index), time=np.float64(self.time),
                            points=self.points.astype(np.float32), point_times=self.point_times,
                            pixels=self.pixels.astype(np.int32), ranges=self.ranges.astype(np.float32),
                            crop_origin=np.array(self.crop_origin, dtype=np.int64), crop_mask=self.crop_mask,
                            crop_range=self.crop_range.astype(np.float32),
                            crop_background=self.crop_background.astype(np.float32),
                            center=np.asarray(self.center, dtype=np.float64), quality=json.dumps(self.quality),
                            **({} if self.crop_times is None else {"crop_times": self.crop_times}))

    @classmethod
    def load(cls, path) -> "PersonFrame":
        d = np.load(path, allow_pickle=False)
        return cls(int(d["index"]), float(d["time"]), d["points"].astype(np.float64),
                   d["point_times"].astype(np.float64), d["pixels"].astype(np.int64), d["ranges"].astype(np.float64),
                   tuple(int(v) for v in d["crop_origin"]), d["crop_mask"], d["crop_range"].astype(np.float64),
                   d["crop_background"].astype(np.float64), d["center"], json.loads(str(d["quality"])),
                   d["crop_times"].astype(np.float64) if "crop_times" in d.files else None)


class MotionSegmenter:
    """Person in every frame of a recording with empty-scene frames."""

    def __init__(self, source: NpzRecording, config: MotionSegmentationConfig | None = None,
                 floor_config: FloorConfig | None = None):
        self.source = source
        self.config = config or MotionSegmentationConfig()
        if source.background_count() == 0:
            raise SystemExit("the recording has no empty-scene frames (background/): record the empty room first")
        ranges = [source.load_background(k).range_m for k in range(source.background_count())]
        stack = np.stack(ranges)
        self.background_reliability = np.mean(np.isfinite(stack), axis=0)
        self.background_range = source.background_range(0.5)
        points = source.sensor.xyz(self.background_range)
        points = points[np.isfinite(points[..., 0])]
        self.floor = RansacFloor(floor_config or FloorConfig()).estimate(points)
        lut = np.load(Path(source.directory) / "lut.npz")
        shift = lut["pixel_shift"] if "pixel_shift" in lut.files else _shift_from_metadata(source.metadata,
                                                                                            lut["direction"].shape[0])
        self.geometry = RangeImageGeometry(lut["direction"], lut["offset"], self.floor.world_from_sensor, shift)
        c = self.config
        self.background = RangeBackground(self.background_range, c.bg_threshold, c.bg_relative)
        self.sensor_xy = self.floor.sensor_position[:2]

    def scene_arrays(self) -> dict:
        lut = np.load(Path(self.source.directory) / "lut.npz")
        return {"world_from_sensor": self.floor.world_from_sensor, "sensor_height": self.floor.sensor_height,
                "tilt_deg": self.floor.tilt_deg, "background_range": self.background_range.astype(np.float32),
                "background_reliability": self.background_reliability.astype(np.float32),
                "direction": lut["direction"], "offset": lut["offset"], "pixel_shift": self.geometry.pixel_shift}

    # ------------------------------------------------------------------------------------------
    def _foreground(self, range_m):
        c = self.config
        mask = self.background.foreground(range_m)
        if c.edge_jump > 0:
            mask &= ~mixed_pixel_mask(range_m, c.edge_jump)
        world = self.floor.to_world(self.source.sensor.xyz(range_m)[mask])
        horizontal = np.linalg.norm(world[:, :2] - self.sensor_xy, axis=1)
        keep = (world[:, 2] > c.min_height) & (world[:, 2] < c.max_height) & (horizontal < c.max_distance)
        rows, cols = np.nonzero(mask)
        return world[keep], rows[keep], cols[keep]

    def _choose(self, world, prediction):
        """Indices of the person's points, or None."""
        c = self.config
        if len(world) < c.min_points:
            return None
        if prediction is not None:
            near = np.linalg.norm(world[:, :2] - prediction, axis=1) < c.gate
            if near.sum() < c.min_points:
                return None
            candidates = np.flatnonzero(near)
            labels = clusters_3d(world[candidates], c.cluster_distance, 3)
            if labels.size and labels.max() >= 0:
                counts = np.bincount(labels[labels >= 0])
                big = np.flatnonzero(counts >= max(c.min_points // 4, 5))
                keep = np.isin(labels, big)
                if keep.sum() >= c.min_points:
                    return candidates[keep]
            return candidates
        labels = clusters_3d(world, c.cluster_distance, 3)
        best, best_count = None, 0
        for label in range(labels.max() + 1 if labels.size else 0):
            members = np.flatnonzero(labels == label)
            if len(members) < c.min_points:
                continue
            z = world[members, 2]
            if np.percentile(z, 98) - np.percentile(z, 2) >= c.min_person_height and len(members) > best_count:
                best, best_count = members, len(members)
        if best is None:
            return None
        # take every cluster within the gate of that one (arms, a separated foot)
        center = np.median(world[best, :2], axis=0)
        return self._choose(world, center)

    def frame(self, index: int, prediction=None) -> PersonFrame | None:
        c = self.config
        frame = self.source.load(index)
        if frame.columns_ok < c.min_columns:
            return None
        range_m = frame.range_m
        world, rows, cols = self._foreground(range_m)
        chosen = self._choose(world, prediction)
        if chosen is None:
            return None
        points, rows, cols = world[chosen], rows[chosen], cols[chosen]
        geometry = self.geometry
        if frame.timestamps is not None and np.any(frame.timestamps > 0):
            stamps = frame.timestamps.copy()
            lost = stamps <= 0
            if lost.any():                                     # lost columns: interpolate their times
                good = np.flatnonzero(~lost)
                stamps[lost] = np.interp(np.flatnonzero(lost), good, stamps[good])
            image_times = geometry.pixel_times(stamps)
        else:
            image_times = np.full((geometry.height, geometry.width), frame.host_time)
        times = image_times[rows, cols]
        ranges = range_m[rows, cols]
        # silhouette crop, unwrapped around the person
        width = geometry.width
        reference = float(np.median(cols))
        unwrapped = geometry.column_distance_unwrapped(cols.astype(np.float64), reference).round().astype(int)
        r0 = max(int(rows.min()) - c.crop_rows, 0)
        r1 = min(int(rows.max()) + c.crop_rows + 1, geometry.height)
        c0 = int(unwrapped.min()) - c.crop_columns
        c1 = int(unwrapped.max()) + c.crop_columns + 1
        crop_rows = np.arange(r0, r1)
        crop_cols = np.mod(np.arange(c0, c1), width)
        crop_range = range_m[np.ix_(crop_rows, crop_cols)]
        crop_background = self.background_range[np.ix_(crop_rows, crop_cols)]
        mask = np.zeros(crop_range.shape, dtype=np.uint8)
        mask[rows - r0, unwrapped - c0] = 1
        dark = 0
        if c.dark_pixels:
            reliable = self.background_reliability[np.ix_(crop_rows, crop_cols)] >= c.dark_reliability
            person_range = float(np.median(ranges))
            with np.errstate(invalid="ignore"):
                behind = crop_background > person_range - 0.3
            candidates = np.isnan(crop_range) & reliable & behind
            near = _dilate(mask == 1, c.dark_reach)
            dark_mask = candidates & near
            mask[dark_mask] = 2
            dark = int(dark_mask.sum())
        valid = times > 0
        time = float(np.mean(times[valid])) if valid.any() else frame.host_time
        z = points[:, 2]
        quality = {"points": int(len(points)), "dark_pixels": dark, "columns_ok": float(frame.columns_ok),
                   "height_m": float(np.percentile(z, 99) - np.percentile(z, 1)),
                   "distance_m": float(np.linalg.norm(np.median(points[:, :2], axis=0) - self.sensor_xy))}
        return PersonFrame(index, time, points, times, np.stack([rows, cols], axis=1), ranges, (r0, c0), mask,
                           crop_range, crop_background, np.median(points[:, :2], axis=0), quality,
                           image_times[np.ix_(crop_rows, crop_cols)])

    def run(self, out, start: int = 0, stop: int | None = None) -> dict:
        """Segment every frame and write the output folder; returns the summary."""
        out = Path(out)
        (out / "frames").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "scene.npz", **self.scene_arrays())
        stop = len(self.source) if stop is None else min(stop, len(self.source))
        prediction, velocity, last_time, missing = None, np.zeros(2), None, 0
        period = 1.0 / self.frame_rate()
        summary = {"frames": [], "missing": []}
        progress = Progress("segmented frames", stop - start)
        for k in range(start, stop):
            guess = None
            if prediction is not None and missing < self.config.lost_after:
                guess = prediction + velocity * period * (missing + 1)
            person = self.frame(k, guess)
            if person is None and guess is not None:
                person = self.frame(k, None)                    # search the whole scene once
            if person is None:
                missing += 1
                summary["missing"].append(k)
            else:
                if prediction is not None and last_time is not None and person.time > last_time + 1e-6:
                    measured = (person.center - prediction) / (person.time - last_time)
                    velocity = 0.5 * velocity + 0.5 * measured            # m/s, smoothed
                prediction, last_time, missing = person.center, person.time, 0
                person.save(out / "frames" / f"person_{k:05d}.npz")
                summary["frames"].append({"index": k, "time": person.time, "center": person.center.tolist(),
                                          **person.quality})
            progress.maybe(k - start + 1, 100)
        summary.update({"run": str(Path(self.source.directory).resolve()),
                        "sensor_height_m": self.floor.sensor_height, "sensor_tilt_deg": self.floor.tilt_deg,
                        "config": asdict(self.config), "image": [self.geometry.height, self.geometry.width]})
        if summary["missing"]:
            warning(f"no person in {len(summary['missing'])} of {stop - start} frames")
        (out / "segments.json").write_text(json.dumps(summary, indent=1))
        info(f"person found in {len(summary['frames'])} of {stop - start} frames -> {out}")
        return summary

    def frame_rate(self) -> float:
        """Frames per second, from the column timestamps of the first frame
        (a frame spans one turn), else from the image width (1024: 20 Hz)."""
        stamps = self.source.load(0).timestamps
        if stamps is not None and np.count_nonzero(stamps > 0) > 10:
            valid = np.flatnonzero(stamps > 0)
            column_period = (stamps[valid[-1]] - stamps[valid[0]]) / max(valid[-1] - valid[0], 1)
            if column_period > 0:
                return float(1.0 / (column_period * len(stamps)))
        return 20.0 if self.geometry.width <= 1024 else 10.0


def _dilate(mask: np.ndarray, steps: int) -> np.ndarray:
    grown = mask.copy()
    for _ in range(steps):
        step = grown.copy()
        step[1:] |= grown[:-1]
        step[:-1] |= grown[1:]
        step[:, 1:] |= grown[:, :-1]
        step[:, :-1] |= grown[:, 1:]
        grown = step
    return grown


def _shift_from_metadata(metadata: dict, height: int) -> np.ndarray:
    """pixel_shift_by_row from the Ouster metadata json (several layouts across firmware versions)."""
    for container in (metadata.get("data_format", {}), metadata.get("lidar_data_format", {}), metadata):
        if isinstance(container, dict) and "pixel_shift_by_row" in container:
            return np.asarray(container["pixel_shift_by_row"], dtype=np.int64)
    return np.zeros(height, dtype=np.int64)


class SegmentedRecording:
    """The output folder of MotionSegmenter.run."""

    def __init__(self, folder):
        self.folder = Path(folder)
        if not (self.folder / "scene.npz").exists():
            raise SystemExit(f"{self.folder} is not a segmentation folder (no scene.npz): run segment-motion first")
        scene = np.load(self.folder / "scene.npz")
        self.world_from_sensor = scene["world_from_sensor"]
        self.sensor_height = float(scene["sensor_height"])
        self.background_range = scene["background_range"].astype(np.float64)
        self.geometry = RangeImageGeometry(scene["direction"], scene["offset"], self.world_from_sensor,
                                           scene["pixel_shift"])
        self.paths = sorted((self.folder / "frames").glob("person_*.npz"), key=natural_key)
        summary = self.folder / "segments.json"
        self.summary = json.loads(summary.read_text()) if summary.exists() else {}
        if not self.paths:
            raise SystemExit(f"no person frames in {self.folder / 'frames'}")

    @property
    def sensor_position(self) -> np.ndarray:
        return self.world_from_sensor[:3, 3].copy()

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, k: int) -> PersonFrame:
        return PersonFrame.load(self.paths[k])

    def times(self) -> np.ndarray:
        return np.array([f["time"] for f in self.summary.get("frames", [])]) if self.summary.get("frames") else \
            np.array([self.load(k).time for k in range(len(self))])
