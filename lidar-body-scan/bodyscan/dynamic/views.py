"""Additional sensors for the tracker: their image geometry, empty scene and per-frame person crops.

A take tracked with several sensors is the Ouster segmentation folder (<take>_seg, unchanged: the
floor frame, the timing and the frame numbering come from it) plus a folder of additional sensors
written by the session tools of the three-sensor rig:

    views.json                        {"sensors": [{"name", "kind", "sigma", ...}, ...], ...}
    <name>/scene.npz                  world_from_sensor (floor frame of the Ouster segmentation),
                                      the image geometry and the empty-scene range image
    <name>/frames/person_XXXXX.npz    PersonFrame of that sensor for Ouster frame index XXXXX

The PersonFrame of an additional sensor has the same arrays as the Ouster's: the person's points in
the floor frame with their times, their pixels, and a crop of that sensor's image around the person
(mask, measured and empty-scene ranges, pixel times). The tracker uses its points in the data term
(whitened by the sensor's noise) and its crop in the free-space and in-front terms, exactly as it
uses the Ouster's.

Image kinds:

    spinning   a spinning LiDAR (Helios 16): rows are beams (top to bottom), columns azimuth steps;
               RangeImageGeometry from direction and offset tables, as for the Ouster
    pinhole    a depth camera (Xtion): rows and columns of a pinhole image (PinholeGeometry); the
               ranges are distances from the optical centre along the ray, not depths
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bodyscan.dynamic.image import RangeImageGeometry


class PinholeGeometry:
    """A pinhole depth image with the interface of RangeImageGeometry.

    Optical frame: x right, y down, z forward (along the optical axis). Pixel (row v, column u):
    u = fx x / z + cx, v = fy y / z + cy. Points behind the camera or outside the image project to
    row -1000 (outside any image), so the free-space and visibility tests skip them. Columns never
    wrap; column_distance_unwrapped is the identity."""

    wraps = False

    def __init__(self, width: int, height: int, fx: float, fy: float, cx: float, cy: float,
                 world_from_sensor: np.ndarray):
        self.width, self.height = int(width), int(height)
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.world_from_sensor = np.asarray(world_from_sensor, dtype=np.float64)
        self.rotation = self.world_from_sensor[:3, :3]
        self.translation = self.world_from_sensor[:3, 3]
        self.vertical_step = 1.0 / self.fy            # radians per pixel at the image centre
        self.column_step = 1.0 / self.fx
        self.pixel_shift = np.zeros(self.height, dtype=np.int64)

    def to_sensor(self, world: np.ndarray) -> np.ndarray:
        return (np.asarray(world, dtype=np.float64) - self.translation) @ self.rotation

    def project(self, world: np.ndarray):
        """(N, 3) world points -> fractional rows, columns and the distance from the optical centre."""
        q = self.to_sensor(world)
        z = q[:, 2]
        safe = np.where(z > 1e-6, z, 1.0)
        column = self.fx * q[:, 0] / safe + self.cx
        row = self.fy * q[:, 1] / safe + self.cy
        outside = (z <= 1e-6) | (column < -0.5) | (column > self.width - 0.5)
        row = np.where(outside, -1000.0, row)
        column = np.where(outside, -1000.0, column)
        return row, column, np.maximum(np.linalg.norm(q, axis=1), 1e-9)

    def pixels_of(self, world: np.ndarray):
        row, column, _ = self.project(world)
        return (np.clip(np.round(row).astype(int), 0, self.height - 1),
                np.clip(np.round(column).astype(int), 0, self.width - 1))

    def column_distance_unwrapped(self, columns: np.ndarray, reference: float) -> np.ndarray:
        return np.asarray(columns, dtype=np.float64)

    def pixel_times(self, frame_time: float) -> np.ndarray:
        return np.full((self.height, self.width), float(frame_time))

    def ray_directions(self) -> np.ndarray:
        """(H, W, 3) unit ray of every pixel in the optical frame."""
        v, u = np.mgrid[0:self.height, 0:self.width].astype(np.float64)
        rays = np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], axis=-1)
        return rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    def torch_tables(self, device, dtype):
        import torch
        return {"rotation": torch.as_tensor(self.rotation, dtype=dtype, device=device),
                "translation": torch.as_tensor(self.translation, dtype=dtype, device=device)}

    def project_torch(self, world, tables, reference_column: float, pixels=None):
        """Differentiable projection: rows, columns and distances of (N, 3) world points."""
        q = (world - tables["translation"]) @ tables["rotation"]
        z = q[:, 2].clamp_min(1e-6)
        rows = self.fy * q[:, 1] / z + self.cy
        columns = self.fx * q[:, 0] / z + self.cx
        import torch
        return rows, columns, torch.linalg.norm(q, dim=1).clamp_min(1e-9)

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}


@dataclass
class SensorView:
    """One sensor as the tracker sees it: its image geometry (pose in the floor frame included), its
    empty scene and how much its points and its free space count.

    sigma: range noise relative to the Ouster's (its points' residuals are divided by it);
    sigma_quadratic: growth of that noise with the range r of a point from the sensor [1/m^2]: the noise
    of a point is max(sigma, sigma_quadratic r^2) (a structured-light depth camera; 0: constant);
    person_range: (nearest, farthest) distance of the person from the sensor [m] at which its frames
    are used; a frame whose points lie nearer or farther (median range) is dropped, neither fitted nor
    held out (None: every frame is used);
    weight: weight of its free-space and in-front terms relative to the Ouster's;
    foreground_margin: how much closer than the empty scene a surface must be to be segmented [m];
    row_tolerance: a vertex is tested for free space only within this many rows of a beam (sparse
    beams: between two beams the sensor measured nothing); 0.5 tests every vertex."""
    name: str
    geometry: object
    background_range: np.ndarray
    sigma: float = 1.0
    weight: float = 1.0
    foreground_margin: float = 0.1
    row_tolerance: float = 0.5
    fitted: bool = True
    row_supersample: int | None = None
    sigma_quadratic: float = 0.0
    person_range: tuple | None = None

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.geometry.translation, dtype=np.float64).copy()

    def point_sigma(self, points: np.ndarray) -> np.ndarray:
        """(N,) noise of world points (N, 3) of this sensor relative to the Ouster's."""
        if self.sigma_quadratic <= 0:
            return np.full(len(points), self.sigma)
        distance = np.linalg.norm(np.asarray(points, dtype=np.float64) - self.position, axis=1)
        return np.maximum(self.sigma, self.sigma_quadratic * distance * distance)

    def in_range(self, ranges: np.ndarray) -> bool:
        """Whether a frame with these point ranges [m] lies inside person_range (True without points)."""
        if self.person_range is None or len(ranges) == 0:
            return True
        return bool(self.person_range[0] <= float(np.median(ranges)) <= self.person_range[1])

    @property
    def wraps(self) -> bool:
        return bool(getattr(self.geometry, "wraps", True))


def parse_sensor_options(text: str, count: int, what: str) -> dict:
    """{sensor name: tuple of 'count' numbers} from 'name:a:b,name:a:b' (an option of the tracker)."""
    out = {}
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        fields = item.split(":")
        if len(fields) != count + 1:
            raise SystemExit(f"{what}: '{item}' must be name" + ":number" * count)
        try:
            out[fields[0].strip()] = tuple(float(v) for v in fields[1:])
        except ValueError:
            raise SystemExit(f"{what}: '{item}' must be name" + ":number" * count) from None
    return out


def geometry_from_scene(kind: str, scene) -> object:
    """The image geometry stored in a sensor's scene.npz."""
    world_from_sensor = scene["world_from_sensor"]
    if kind == "pinhole":
        w, h, fx, fy, cx, cy = (float(v) for v in scene["intrinsics"])
        return PinholeGeometry(int(w), int(h), fx, fy, cx, cy, world_from_sensor)
    if kind == "spinning":
        return RangeImageGeometry(scene["direction"], scene["offset"], world_from_sensor,
                                  scene["pixel_shift"] if "pixel_shift" in scene.files else None)
    raise ValueError(f"unknown sensor kind {kind!r}")


class ExtraSensors:
    """The additional sensors of a take (folder written by the session tools), see the module text."""

    def __init__(self, folder, exclude=(), weight: float = 1.0, noise: str = "", ranges: str = ""):
        """noise: 'name:sigma:sigma_quadratic,...' and ranges: 'name:nearest:farthest,...' replace the
        values of views.json for the named sensors (SensorView.sigma, sigma_quadratic, person_range)."""
        from bodyscan.dynamic.segment import PersonFrame
        self._person = PersonFrame
        self.folder = Path(folder)
        meta_path = self.folder / "views.json"
        if not meta_path.exists():
            raise SystemExit(f"{self.folder} has no views.json: not a folder of additional sensors")
        self.meta = json.loads(meta_path.read_text())
        excluded = {name.strip() for name in exclude if name.strip()}
        noise_of = parse_sensor_options(noise, 2, "extra_sensor_noise")
        range_of = parse_sensor_options(ranges, 2, "extra_sensor_range")
        names = {entry["name"] for entry in self.meta["sensors"]}
        for option, given in (("extra_sensor_noise", noise_of), ("extra_sensor_range", range_of)):
            if set(given) - names:
                raise SystemExit(f"{option} names no additional sensor of {self.folder}: {sorted(set(given) - names)}")
        self.views: list[SensorView] = []
        self.paths: list[dict] = []
        self.dropped = {}                      # sensor name: frames left out because the person was out of range
        self._out_of_range = {}                # sensor name: the Ouster frame indices of those frames
        for entry in self.meta["sensors"]:
            name = entry["name"]
            scene = np.load(self.folder / name / "scene.npz")
            geometry = geometry_from_scene(entry["kind"], scene)
            view = SensorView(name=name, geometry=geometry,
                              background_range=scene["background_range"].astype(np.float64),
                              sigma=float(entry.get("sigma", 1.0)), weight=float(entry.get("weight", 1.0)) * weight,
                              foreground_margin=float(entry.get("foreground_margin", 0.1)),
                              row_tolerance=float(entry.get("row_tolerance", 0.5)),
                              fitted=name not in excluded,
                              row_supersample=(int(entry["row_supersample"]) if entry.get("row_supersample")
                                               else None),
                              sigma_quadratic=float(entry.get("sigma_quadratic", 0.0)),
                              person_range=(tuple(float(v) for v in entry["person_range"])
                                            if entry.get("person_range") else None))
            if name in noise_of:
                view.sigma, view.sigma_quadratic = noise_of[name]
            if name in range_of:
                view.person_range = range_of[name]
            self.dropped[name] = 0
            self.views.append(view)
            frames = {}
            for path in sorted((self.folder / name / "frames").glob("person_*.npz")):
                frames[int(path.stem.split("_")[-1])] = path
            self.paths.append(frames)
        unknown = excluded - {v.name for v in self.views}
        if unknown:
            raise SystemExit(f"--exclude-sensors names no additional sensor of {self.folder}: {sorted(unknown)}")

    def frames_for(self, index: int) -> list:
        """[(position in self.views, PersonFrame)] of the sensors that saw the person in Ouster frame 'index'."""
        out = []
        for number, frames in enumerate(self.paths):
            path = frames.get(int(index))
            if path is None:
                continue
            person = self._person.load(path)
            view = self.views[number]
            if view.person_range is not None:
                ranges = np.linalg.norm(person.points - view.position, axis=1) if len(person.points) else np.zeros(0)
                if not view.in_range(ranges):
                    self._out_of_range.setdefault(view.name, set()).add(int(index))
                    self.dropped[view.name] = len(self._out_of_range[view.name])
                    continue
            out.append((number, person))
        return out

    def summary(self) -> dict:
        return {"folder": str(self.folder),
                "sensors": [{"name": v.name, "sigma": v.sigma, "sigma_quadratic": v.sigma_quadratic,
                             "person_range": list(v.person_range) if v.person_range else None,
                             "frames_out_of_range": self.dropped.get(v.name, 0),
                             "weight": v.weight, "fitted": v.fitted,
                             "frames": len(p)} for v, p in zip(self.views, self.paths)],
                "calibration": self.meta.get("calibration"), "time_offsets": self.meta.get("time_offsets")}
