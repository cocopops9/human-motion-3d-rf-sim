"""The person in the images of additional sensors, frame by frame of the Ouster segmentation (numpy only).

The Ouster segmentation (segment-motion) has found the person in every frame: the frame's time and
where the person stands (segments.json). An additional sensor whose pose in the same floor frame is
known (calibration) and whose images carry times on the same clock (corrected by the sensor's time
offset) is cut frame by frame:

    1. image       the sensor image whose person was measured closest in time to the Ouster's
                   (within max_time_difference; a 10 Hz LiDAR misses every other 20 Hz frame)
    2. foreground  pixels closer than the sensor's empty scene by max(bg_threshold, bg_relative x
                   range), or with a return where the empty scene had none, minus mixed edge pixels
                   (range jump on both sides by max(edge_jump, edge_relative x range))
    3. person      the foreground within a vertical cylinder of radius 'gate' around where the Ouster
                   saw the person, between min_height and max_height above the floor
    4. silhouette  a crop of the image around the person, as for the Ouster: 1 = a return of the
                   person, 2 = no return where the empty scene returns from behind the person (dark
                   fabric, grazing angles, the projector shadow of a depth camera), 0 = anything else;
                   with the measured and empty-scene ranges and the time of every pixel. The crop
                   covers the whole gate cylinder as the sensor sees it (with dark_reach < 0 every
                   pixel without a return in it is dark): a suit that returns little light to one
                   sensor (the Helios saw about 20 points of the person per turn on 3 Nov) must not
                   look like empty space where the body is

Output (read by bodyscan.dynamic.views.ExtraSensors): <out>/<name>/scene.npz, <out>/<name>/frames/
person_XXXXX.npz (XXXXX: the Ouster frame index), <out>/<name>/segments.json, and the sensor's entry
in <out>/views.json. The person files have the arrays of bodyscan.dynamic.segment.PersonFrame.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class ExtraSegmentationConfig:
    """Cutting the person out of an additional sensor's images (floor frame: z up, z = 0 on the floor)."""
    bg_threshold: float = 0.08          # m
    bg_relative: float = 0.015
    edge_jump: float = 0.05             # m
    edge_relative: float = 0.0          # depth cameras: their noise grows with the range
    min_height: float = 0.01            # m
    max_height: float = 2.4             # m
    max_distance: float = 9.0           # m, horizontal from the sensor
    gate: float = 0.9                   # m, around where the Ouster saw the person
    min_points: int = 15
    max_time_difference: float = 0.035  # s, between the sensor's person and the Ouster's
    crop_rows: int = 6
    crop_columns: int = 12
    dark_reach: int = 3                 # pixels from the person's returns; negative: the whole crop
    crop_gate: bool = False             # the crop covers the gate cylinder (else the person's pixels only)
    voxel: float = 0.0                  # m: keep one point per voxel of this size (0: every pixel); a dense
                                        # depth camera otherwise outnumbers the LiDARs with correlated errors
    # range bias of the sensor on the person, added to the range of every foreground pixel (the empty scene
    # keeps its ranges): b0 + b1 (r - r_ref), r clipped to [r_min, r_max]. A LiDAR can read a dark suit
    # several centimetres short although it measures the room right (the Helios 16P of 3 Nov: +8 cm at 4 m,
    # fitted on the person against the avatar fitted to the Ouster).
    foreground_range_b0: float = 0.0    # m
    foreground_range_b1: float = 0.0    # m per m
    foreground_range_ref: float = 4.0   # m
    foreground_range_min: float = 0.0   # m
    foreground_range_max: float = 100.0  # m


def dilate(mask: np.ndarray, steps: int) -> np.ndarray:
    grown = mask.copy()
    for _ in range(steps):
        step = grown.copy()
        step[1:] |= grown[:-1]
        step[:-1] |= grown[1:]
        step[:, 1:] |= grown[:, :-1]
        step[:, :-1] |= grown[:, 1:]
        grown = step
    return grown


def mixed_pixels(range_m: np.ndarray, jump: np.ndarray, wraps: bool) -> np.ndarray:
    """Pixels that break the range profile on both sides, horizontally or vertically (a beam or pixel on
    an edge returns a range between the foreground and the background); 'jump' may vary per pixel."""
    r = range_m
    if wraps:
        left, right = np.roll(r, 1, axis=1), np.roll(r, -1, axis=1)
    else:
        left, right = np.full_like(r, np.nan), np.full_like(r, np.nan)
        left[:, 1:], right[:, :-1] = r[:, :-1], r[:, 1:]
    up, down = np.full_like(r, np.nan), np.full_like(r, np.nan)
    up[1:], down[:-1] = r[:-1], r[1:]
    with np.errstate(invalid="ignore"):
        rows = (np.abs(r - left) > jump) & (np.abs(r - right) > jump) & (np.abs(left + right - 2 * r) > jump)
        cols = (np.abs(r - up) > jump) & (np.abs(r - down) > jump) & (np.abs(up + down - 2 * r) > jump)
    return rows | cols


class SensorImages:
    """A sequence of images of one sensor, already on the session clock.

    kind: "spinning" or "pinhole"; geometry: RangeImageGeometry or PinholeGeometry with the sensor's
    pose in the floor frame; background: (H, W) empty-scene range [m], NaN where it never returned;
    frame_times: (J,) a reference time per image [s]; load(j) -> (range (H, W) [m], NaN: no return,
    times (H, W) [s]). Subclasses read the device files; this base class holds arrays in memory."""

    def __init__(self, kind: str, geometry, background: np.ndarray, ranges=None, times=None,
                 frame_times=None, period: float | None = None):
        self.kind = kind
        self.geometry = geometry
        self.background = np.asarray(background, dtype=np.float64)
        self._ranges = ranges
        self._times = times
        self.frame_times = np.asarray(frame_times if frame_times is not None else [], dtype=np.float64)
        steps = np.diff(self.frame_times)
        self.period = float(period if period is not None else (np.median(steps) if len(steps) else 0.1))

    def __len__(self) -> int:
        return len(self.frame_times)

    def load(self, j: int):
        return np.asarray(self._ranges[j], dtype=np.float64), np.asarray(self._times[j], dtype=np.float64)

    def rays(self) -> tuple[np.ndarray, np.ndarray]:
        """(H, W, 3) unit direction and (H, W, 3) origin offset of every pixel in the sensor frame."""
        g = self.geometry
        if self.kind == "pinhole":
            directions = g.ray_directions()
            return directions, np.zeros_like(directions)
        return g.direction, g.offset


def _write_person(path, index, time, points, point_times, pixels, ranges, crop_origin, crop_mask, crop_range,
                  crop_background, center, quality, crop_times) -> None:
    """The arrays of bodyscan.dynamic.segment.PersonFrame.save, without importing it (Open3D)."""
    np.savez_compressed(path, index=np.int64(index), time=np.float64(time), points=points.astype(np.float32),
                        point_times=point_times.astype(np.float64), pixels=pixels.astype(np.int32),
                        ranges=ranges.astype(np.float32), crop_origin=np.array(crop_origin, dtype=np.int64),
                        crop_mask=crop_mask, crop_range=crop_range.astype(np.float32),
                        crop_background=crop_background.astype(np.float32),
                        center=np.asarray(center, dtype=np.float64), quality=json.dumps(quality),
                        crop_times=crop_times.astype(np.float64))


class ExtraSegmenter:
    def __init__(self, images: SensorImages, config: ExtraSegmentationConfig | None = None):
        self.images = images
        self.config = config or ExtraSegmentationConfig()
        g = images.geometry
        self.wraps = images.kind == "spinning"
        self.directions, self.offsets = images.rays()
        self.rotation = g.world_from_sensor[:3, :3]
        self.translation = g.world_from_sensor[:3, 3]
        self.cache = {}

    # -- which image ------------------------------------------------------------------------------
    def person_time(self, j: int, center_xy: np.ndarray, height: float = 1.0) -> float:
        """Time [s] at which image j measured the place of the person (the pixel of the point at
        'height' above center_xy)."""
        g = self.images.geometry
        rows, cols = g.pixels_of(np.array([[center_xy[0], center_xy[1], height]]))
        _, times = self._image(j)
        return float(times[int(rows[0]), int(cols[0])])

    def _image(self, j: int):
        if j not in self.cache:
            if len(self.cache) > 8:
                self.cache.pop(next(iter(self.cache)))
            self.cache[j] = self.images.load(j)
        return self.cache[j]

    def choose(self, time: float, center_xy: np.ndarray) -> tuple[int, float] | None:
        """The image whose person was measured closest to 'time', and the time difference [s]."""
        times = self.images.frame_times
        if len(times) == 0:
            return None
        near = np.flatnonzero(np.abs(times - time) <= 1.5 * self.images.period + self.config.max_time_difference)
        best, best_dt = None, np.inf
        for j in near:
            dt = self.person_time(int(j), center_xy) - time
            if abs(dt) < abs(best_dt):
                best, best_dt = int(j), dt
        if best is None or abs(best_dt) > self.config.max_time_difference:
            return None
        return best, best_dt

    # -- the person in one image --------------------------------------------------------------------
    def person(self, index: int, j: int, center_xy: np.ndarray) -> dict | None:
        """The person of Ouster frame 'index' in image j (arrays of a PersonFrame), or None."""
        c = self.config
        g = self.images.geometry
        range_m, times = self._image(j)
        background = self.images.background
        with np.errstate(invalid="ignore"):
            closer = range_m < background - np.maximum(c.bg_threshold, c.bg_relative * range_m)
            mask = np.isfinite(range_m) & (closer | ~np.isfinite(background))
            jump = np.maximum(c.edge_jump, c.edge_relative * np.nan_to_num(range_m, nan=0.0))
            if c.edge_jump > 0:
                mask &= ~mixed_pixels(range_m, jump, self.wraps)
        if c.foreground_range_b0 != 0.0 or c.foreground_range_b1 != 0.0:
            range_m = range_m.copy()
            measured = range_m[mask]
            clipped = np.clip(measured, c.foreground_range_min, c.foreground_range_max)
            range_m[mask] = measured + c.foreground_range_b0 + c.foreground_range_b1 * (clipped - c.foreground_range_ref)
        rows, cols = np.nonzero(mask)
        if len(rows) < c.min_points:
            return None
        local = self.directions[rows, cols] * range_m[rows, cols][:, None] + self.offsets[rows, cols]
        world = local @ self.rotation.T + self.translation
        horizontal = np.linalg.norm(world[:, :2] - self.translation[:2], axis=1)
        near = np.linalg.norm(world[:, :2] - np.asarray(center_xy)[None], axis=1) < c.gate
        keep = near & (world[:, 2] > c.min_height) & (world[:, 2] < c.max_height) & (horizontal < c.max_distance)
        if keep.sum() < c.min_points:
            return None
        points, rows, cols = world[keep], rows[keep], cols[keep]
        ranges = range_m[rows, cols]
        point_times = times[rows, cols]
        # silhouette crop (unwrapped around the person for a spinning sensor)
        if self.wraps:
            reference = float(np.median(cols))
            unwrapped = (reference + np.mod(cols - reference + g.width / 2, g.width) - g.width / 2).round().astype(int)
        else:
            reference = 0.0
            unwrapped = cols.astype(int)
        row_low, row_high = int(rows.min()), int(rows.max())
        col_low, col_high = int(unwrapped.min()), int(unwrapped.max())
        if c.crop_gate:
            # the gate cylinder as the sensor sees it: where the body may be
            angle = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
            ring = np.stack([center_xy[0] + c.gate * np.cos(angle), center_xy[1] + c.gate * np.sin(angle)], axis=1)
            heights = np.linspace(max(c.min_height, 0.0), c.max_height, 7)
            cylinder = np.array([[x, y, z] for x, y in ring for z in heights])
            g_rows, g_cols, _ = g.project(cylinder)
            seen = g_rows > -100
            if seen.any():
                g_cols = g_cols[seen]
                if self.wraps:
                    g_cols = reference + np.mod(g_cols - reference + g.width / 2, g.width) - g.width / 2
                row_low = min(row_low, int(np.floor(g_rows[seen].min())))
                row_high = max(row_high, int(np.ceil(g_rows[seen].max())))
                col_low = min(col_low, int(np.floor(g_cols.min())))
                col_high = max(col_high, int(np.ceil(g_cols.max())))
        r0 = max(row_low - c.crop_rows, 0)
        r1 = min(row_high + c.crop_rows + 1, g.height)
        c0 = col_low - c.crop_columns
        c1 = col_high + c.crop_columns + 1
        if not self.wraps:
            c0, c1 = max(c0, 0), min(c1, g.width)
        crop_rows = np.arange(r0, r1)
        crop_cols = np.mod(np.arange(c0, c1), g.width) if self.wraps else np.arange(c0, c1)
        crop_range = range_m[np.ix_(crop_rows, crop_cols)]
        crop_background = background[np.ix_(crop_rows, crop_cols)]
        crop_times = times[np.ix_(crop_rows, crop_cols)]
        crop_mask = np.zeros(crop_range.shape, dtype=np.uint8)
        crop_mask[rows - r0, unwrapped - c0] = 1
        # light the person blocked without returning it: no return next to the person where the empty
        # scene returns from behind the person
        person_range = float(np.median(ranges))
        with np.errstate(invalid="ignore"):
            behind = crop_background > person_range - 0.3
        dark = np.isnan(crop_range) & behind
        if c.dark_reach >= 0:
            dark &= dilate(crop_mask == 1, c.dark_reach)
        crop_mask[dark] = 2
        pixels_found = len(points)
        if c.voxel > 0:
            # one point per voxel (the first in pixel order: a measured point on the surface, so the
            # point-to-plane distance is unbiased); the silhouette above keeps every pixel
            key = np.floor(points / c.voxel).astype(np.int64)
            _, first = np.unique(key, axis=0, return_index=True)
            pick = np.sort(first)
            points, rows, cols, ranges, point_times = (points[pick], rows[pick], cols[pick], ranges[pick],
                                                       point_times[pick])
        time = float(np.median(point_times))
        z = points[:, 2]
        quality = {"points": int(len(points)), "person_pixels": int(pixels_found), "dark_pixels": int(dark.sum()),
                   "image": int(j),
                   "height_m": float(np.percentile(z, 99) - np.percentile(z, 1)),
                   "distance_m": float(np.linalg.norm(np.median(points[:, :2], axis=0) - self.translation[:2]))}
        return {"index": index, "time": time, "points": points, "point_times": point_times,
                "pixels": np.stack([rows, cols], axis=1), "ranges": ranges, "crop_origin": (r0, c0),
                "crop_mask": crop_mask, "crop_range": crop_range, "crop_background": crop_background,
                "center": np.median(points[:, :2], axis=0), "quality": quality, "crop_times": crop_times}

    # -- whole take ---------------------------------------------------------------------------------
    def run(self, track: list[dict], out, name: str, sigma: float, extra_entry: dict | None = None,
            progress=None) -> dict:
        """Cut the person for every Ouster frame of 'track' (segments.json frames: index, time,
        center) and write the sensor's folder and its entry in views.json."""
        out = Path(out)
        folder = out / name
        (folder / "frames").mkdir(parents=True, exist_ok=True)
        g = self.images.geometry
        scene = {"world_from_sensor": g.world_from_sensor, "background_range": self.images.background.astype(np.float32)}
        if self.images.kind == "pinhole":
            scene["intrinsics"] = np.array([g.width, g.height, g.fx, g.fy, g.cx, g.cy], dtype=np.float64)
        else:
            scene.update({"direction": g.direction, "offset": g.offset, "pixel_shift": g.pixel_shift})
        np.savez_compressed(folder / "scene.npz", **scene)
        summary = {"frames": [], "missing": [], "config": asdict(self.config)}
        for number, entry in enumerate(track):
            index, time = int(entry["index"]), float(entry["time"])
            center = np.asarray(entry["center"], dtype=np.float64)[:2]
            chosen = self.choose(time, center)
            person = None if chosen is None else self.person(index, chosen[0], center)
            if person is None:
                summary["missing"].append(index)
            else:
                person["quality"]["time_difference_ms"] = round(1000 * chosen[1], 2)
                _write_person(folder / "frames" / f"person_{index:05d}.npz", **person)
                summary["frames"].append({"index": index, "time": person["time"],
                                          "center": person["center"].tolist(), **person["quality"]})
            if progress is not None:
                progress(number + 1, len(track))
        (folder / "segments.json").write_text(json.dumps(summary, indent=1))
        views_path = out / "views.json"
        views = json.loads(views_path.read_text()) if views_path.exists() else {"sensors": []}
        views["sensors"] = [s for s in views["sensors"] if s["name"] != name]
        views["sensors"].append({"name": name, "kind": self.images.kind, "sigma": float(sigma),
                                 "foreground_margin": float(self.config.bg_threshold + 0.02),
                                 **(extra_entry or {})})
        views_path.write_text(json.dumps(views, indent=1))
        return summary
