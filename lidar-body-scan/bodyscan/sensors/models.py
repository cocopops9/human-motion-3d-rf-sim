"""Sensor models: the ray geometry, timing and range noise of every sensor type a rig can hold.

Sensor frame (all sensor types): x forward, y to the left, z up, origin at the
optical centre. A spinning LiDAR measures column 0 along +x and turns counter-
clockwise seen from above; a depth camera looks along +x.

Two types cover the sensors of the lab:

    SpinningLidar   one row per beam (fixed elevation, top beam first), one
                    column per azimuth step; column c of a frame starting at
                    t0 is measured at t0 + c / (columns x frame_rate)
                    (Ouster OS0-128, RoboSense Helios 16)
    DepthCamera     a pinhole depth image; every pixel of a frame is measured
                    at the same time (global shutter assumed; the camera's
                    internal latency is a clock offset of the rig)
                    (ASUS Xtion, structured light)

Noise is the one-sigma range error along the ray, sigma(r) = constant +
linear r + quadratic r^2 [m]. For a depth camera the error is on the depth
(distance along the optical axis); range_sigma converts it to the ray.

The factory functions give the nominal models of the lab's sensors. The real
Ouster geometry of a recording is its lookup table (bodyscan.io); the nominal
model is for simulation and planning.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Union

import numpy as np


@dataclass(frozen=True)
class RangeNoise:
    """One-sigma range error sigma(r) = constant + linear r + quadratic r^2 [m]."""
    constant: float = 0.0
    linear: float = 0.0
    quadratic: float = 0.0

    def sigma(self, distance) -> np.ndarray:
        r = np.asarray(distance, dtype=np.float64)
        return self.constant + self.linear * r + self.quadratic * r * r


@dataclass(frozen=True)
class SpinningLidar:
    """A spinning multi-beam LiDAR: rows are beams (top first), columns are azimuth steps."""
    model: str
    elevations_deg: tuple
    columns: int
    frame_rate: float
    min_range: float
    max_range: float
    noise: RangeNoise = field(default_factory=RangeNoise)
    kind: str = "spinning_lidar"

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.elevations_deg), self.columns

    @property
    def mode(self) -> str:
        return f"{self.columns}x{int(round(self.frame_rate))}"

    def directions(self) -> np.ndarray:
        """(rows, columns, 3) unit ray directions in the sensor frame."""
        elevation = np.radians(np.asarray(self.elevations_deg, dtype=np.float64))
        azimuth = np.arange(self.columns) * (2.0 * np.pi / self.columns)
        el, az = np.meshgrid(elevation, azimuth, indexing="ij")
        return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)

    def pixel_times(self, frame_start: float) -> np.ndarray:
        """(rows, columns) measurement time [s] of every pixel of the frame starting at frame_start."""
        column_time = np.arange(self.columns) / (self.columns * self.frame_rate)
        return np.broadcast_to(frame_start + column_time, self.shape).copy()

    def range_sigma(self, distance, directions=None) -> np.ndarray:
        return self.noise.sigma(distance)

    def in_view(self, points_sensor: np.ndarray) -> np.ndarray:
        """(N,) True for sensor-frame points inside the vertical field of view and the range limits."""
        q = np.asarray(points_sensor, dtype=np.float64)
        distance = np.linalg.norm(q, axis=1)
        elevation = np.degrees(np.arcsin(np.clip(q[:, 2] / np.maximum(distance, 1e-12), -1.0, 1.0)))
        top, bottom = max(self.elevations_deg), min(self.elevations_deg)
        half_step = 0.5 * abs(self.elevations_deg[0] - self.elevations_deg[1]) if len(self.elevations_deg) > 1 else 0.0
        return ((elevation <= top + half_step) & (elevation >= bottom - half_step)
                & (distance >= self.min_range) & (distance <= self.max_range))

    def angular_steps_deg(self) -> tuple[float, float]:
        """(vertical, horizontal) mean angle between neighbouring samples [deg]."""
        elevations = np.asarray(self.elevations_deg, dtype=np.float64)
        vertical = float(np.mean(np.abs(np.diff(elevations)))) if len(elevations) > 1 else 0.0
        return vertical, 360.0 / self.columns

    def to_dict(self) -> dict:
        d = asdict(self)
        d["elevations_deg"] = [float(e) for e in self.elevations_deg]
        return d


@dataclass(frozen=True)
class DepthCamera:
    """A pinhole depth camera. roll_deg turns the image about the optical axis
    (0: landscape; 90: portrait, the image's long side vertical)."""
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    frame_rate: float
    min_range: float
    max_range: float
    noise: RangeNoise = field(default_factory=RangeNoise)
    roll_deg: float = 0.0
    kind: str = "depth_camera"

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def mode(self) -> str:
        return f"{self.width}x{self.height}@{int(round(self.frame_rate))}"

    def _camera_rays(self) -> np.ndarray:
        """(H, W, 3) rays in the optical frame (x right, y down, z forward), z component 1."""
        u = (np.arange(self.width) + 0.5 - self.cx) / self.fx
        v = (np.arange(self.height) + 0.5 - self.cy) / self.fy
        uu, vv = np.meshgrid(u, v)
        return np.stack([uu, vv, np.ones_like(uu)], axis=-1)

    def sensor_from_optical(self) -> np.ndarray:
        """3x3 rotation from the optical frame (x right, y down, z forward) to the sensor frame
        (x forward, y left, z up), including the roll of the camera about its optical axis."""
        base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        roll = math.radians(self.roll_deg)
        c, s = math.cos(roll), math.sin(roll)
        turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])     # about the optical z axis
        return base @ turn

    def directions(self) -> np.ndarray:
        """(H, W, 3) unit ray directions in the sensor frame."""
        rays = self._camera_rays() @ self.sensor_from_optical().T
        return rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    def pixel_times(self, frame_start: float) -> np.ndarray:
        return np.full(self.shape, float(frame_start))

    def depth_of(self, distance, directions) -> np.ndarray:
        """Depth (along the optical axis, the sensor's +x) of points at 'distance' along unit rays."""
        return np.asarray(distance, dtype=np.float64) * np.asarray(directions, dtype=np.float64)[..., 0]

    def range_sigma(self, distance, directions) -> np.ndarray:
        """One-sigma error along the ray: the depth error divided by the cosine to the optical axis."""
        cosine = np.clip(np.asarray(directions, dtype=np.float64)[..., 0], 1e-6, 1.0)
        return self.noise.sigma(self.depth_of(distance, directions)) / cosine

    def in_view(self, points_sensor: np.ndarray) -> np.ndarray:
        """(N,) True for sensor-frame points that project inside the image and the depth limits."""
        q = np.asarray(points_sensor, dtype=np.float64) @ self.sensor_from_optical()   # into the optical frame
        depth = q[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * q[:, 0] / depth + self.cx
            v = self.fy * q[:, 1] / depth + self.cy
        return ((depth >= self.min_range) & (depth <= self.max_range) & (u >= 0) & (u < self.width)
                & (v >= 0) & (v < self.height))

    def field_of_view_deg(self) -> tuple[float, float]:
        """(horizontal, vertical) field of view of the image as mounted [deg]."""
        across = 2.0 * math.degrees(math.atan(self.width / (2.0 * self.fx)))
        up = 2.0 * math.degrees(math.atan(self.height / (2.0 * self.fy)))
        portrait = abs(round(self.roll_deg / 90.0)) % 2 == 1
        return (up, across) if portrait else (across, up)

    def to_dict(self) -> dict:
        return asdict(self)


SensorModel = Union[SpinningLidar, DepthCamera]


def sensor_from_dict(data: dict) -> SensorModel:
    """The sensor model written by to_dict (the 'kind' field selects the type)."""
    values = dict(data)
    kind = values.pop("kind", "spinning_lidar")
    noise = RangeNoise(**values.pop("noise", {}))
    if kind == "spinning_lidar":
        values["elevations_deg"] = tuple(float(e) for e in values["elevations_deg"])
        return SpinningLidar(noise=noise, **values)
    if kind == "depth_camera":
        return DepthCamera(noise=noise, **values)
    raise ValueError(f"unknown sensor kind {kind!r}")


# ----------------------------------------------------------------------------
# The lab's sensors (nominal models)
# ----------------------------------------------------------------------------

def ouster_os0_128(columns: int = 1024, frame_rate: float = 20.0) -> SpinningLidar:
    """Ouster OS0-128: 128 beams over about 90 deg (+-45), 865 nm. Nominal model with evenly spaced
    beams; a real recording carries its own lookup table. Noise: the 8 mm of the synthetic bench."""
    elevations = tuple(float(e) for e in np.linspace(45.0, -45.0, 128))
    return SpinningLidar("Ouster OS0-128", elevations, columns, frame_rate, 0.3, 50.0, RangeNoise(0.008))


def helios_16(columns: int = 1800, frame_rate: float = 10.0) -> SpinningLidar:
    """RoboSense Helios 16: 16 beams from +15 to -15 deg (2 deg apart); 0.2 deg at 10 Hz (1800 columns),
    0.4 deg at 20 Hz (900), 0.1 deg at 5 Hz (3600); range 0.2 to 150 m; precision 1 cm typical."""
    elevations = tuple(float(e) for e in np.linspace(15.0, -15.0, 16))
    return SpinningLidar("RoboSense Helios 16", elevations, columns, frame_rate, 0.2, 150.0, RangeNoise(0.01))


def xtion(orientation: str = "landscape", frame_rate: float = 30.0) -> DepthCamera:
    """ASUS Xtion (PrimeSense structured light): 640 x 480 depth, 58 x 45 deg, 0.8 to 3.5 m.
    Depth noise grows with depth squared, about 4 cm at 5 m for this sensor class (moderate
    confidence): sigma = 0.0016 z^2. orientation 'portrait' turns the camera 90 deg about its
    optical axis (full height of a standing person from about 1.6 m)."""
    if orientation not in ("landscape", "portrait"):
        raise ValueError("orientation must be 'landscape' or 'portrait'")
    width, height = 640, 480
    fx = (width / 2.0) / math.tan(math.radians(29.0))
    fy = (height / 2.0) / math.tan(math.radians(22.5))
    return DepthCamera("ASUS Xtion", width, height, fx, fy, width / 2.0, height / 2.0, frame_rate, 0.8, 3.5,
                       RangeNoise(0.0, 0.0, 0.0016), 90.0 if orientation == "portrait" else 0.0)


def preset(name: str, **options) -> SensorModel:
    """A nominal sensor model by name: 'ouster', 'helios', 'xtion', 'xtion-portrait'."""
    if name == "ouster":
        return ouster_os0_128(**options)
    if name == "helios":
        return helios_16(**options)
    if name == "xtion":
        return xtion("landscape", **options)
    if name == "xtion-portrait":
        return xtion("portrait", **options)
    raise ValueError(f"unknown sensor preset {name!r}: ouster, helios, xtion, xtion-portrait")
