"""Recordings on disk: a sequence of frames, optional background frames, a sensor model.

The capture scripts write one directory per run:

    lut.npz          per-pixel unit direction and offset: xyz = range_m * direction + offset
    metadata.json    sensor metadata (Ouster)
    capture.json     capture log (commanded turn, phases, lost packets, ...)
    background/      bg_XXXXX.npz frames of the empty scene
    frames/          frame_XXXXX.npz: range (H, W) uint16 [mm] (0 = no return), time [s] (host
                     clock), timestamps (W,) uint64 [ns] (sensor clock per column), phase,
                     columns_ok, frame_id, reflectivity

FrameSource is the interface the processing uses; NpzRecording reads the
directories above, PointCloudFolder reads one point cloud file per frame (any
sensor, no range image). A new sensor or file format is supported by writing
another FrameSource (and, for range images, a SensorModel).
"""

from __future__ import annotations

import json
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def natural_key(path) -> list:
    """Sort key that puts frame_2 before frame_10."""
    parts = re.split(r"(\d+)", Path(path).name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


@dataclass
class Frame:
    """One sweep of the sensor.

    range_m      (H, W) range in metres, NaN where there is no return (range-image sources)
    points       (N, 3) points in the sensor frame (point-cloud sources; None for range images,
                 use SensorModel.xyz)
    timestamps   (W,) sensor time of every column [s], 0 for a lost column; None if unknown
    host_time    time of the frame on the recording PC [s]
    phase        capture phase label (0 unknown, 1 still before, 2 turning, 3 still after)
    columns_ok   fraction of the columns received (lost UDP packets lower it)
    """
    index: int
    range_m: np.ndarray | None = None
    points: np.ndarray | None = None
    timestamps: np.ndarray | None = None
    host_time: float = float("nan")
    phase: int = 0
    columns_ok: float = 1.0
    path: Path | None = None


class SensorModel(ABC):
    """Geometry of a range-image sensor: from range image to points in the sensor frame."""

    @abstractmethod
    def xyz(self, range_m: np.ndarray) -> np.ndarray:
        """(H, W, 3) points; NaN where the range is NaN."""

    @property
    @abstractmethod
    def directions(self) -> np.ndarray:
        """(H, W, 3) unit beam directions in the sensor frame."""

    def angular_steps(self) -> tuple[float, float]:
        """(horizontal, vertical) angle between neighbouring pixels [rad]: the
        column step over the full turn, and the median elevation step between rows."""
        directions = self.directions
        elevation = np.arcsin(np.clip(directions[..., 2], -1.0, 1.0))
        rows = np.median(elevation, axis=1)
        vertical = float(np.median(np.abs(np.diff(rows)))) if len(rows) > 1 else np.radians(0.7)
        return 2.0 * np.pi / directions.shape[1], vertical


class LutSensorModel(SensorModel):
    """Affine per-pixel model xyz = range * direction + offset (exact for Ouster sensors;
    the lut.npz written at capture time, so processing needs no SDK)."""

    def __init__(self, direction: np.ndarray, offset: np.ndarray):
        self.direction = np.asarray(direction, dtype=np.float64)
        self.offset = np.asarray(offset, dtype=np.float64)

    @classmethod
    def load(cls, path) -> "LutSensorModel":
        data = np.load(path)
        return cls(data["direction"], data["offset"])

    def xyz(self, range_m: np.ndarray) -> np.ndarray:
        return range_m[..., None] * self.direction + self.offset

    @property
    def directions(self) -> np.ndarray:
        return self.direction


class FrameSource(ABC):
    """A recording: frames, optional background frames, optional sensor model."""

    directory: Path
    sensor: SensorModel | None = None
    capture: dict = {}

    @property
    def organized(self) -> bool:
        """True when frames are range images (a SensorModel turns them into points)."""
        return self.sensor is not None

    @abstractmethod
    def __len__(self) -> int:
        """Number of frames."""

    @abstractmethod
    def load(self, index: int) -> Frame:
        """Frame number 'index' (0 based)."""

    def background_count(self) -> int:
        return 0

    def load_background(self, index: int) -> Frame:
        raise IndexError("this recording has no background frames")

    def points(self, frame: Frame) -> np.ndarray:
        """(N, 3) valid points of a frame in the sensor frame."""
        if frame.points is not None:
            return frame.points
        xyz = self.sensor.xyz(frame.range_m)
        return xyz[np.isfinite(xyz[..., 0])]


class NpzRecording(FrameSource):
    """A run directory written by the capture commands (see the module docstring)."""

    def __init__(self, directory, min_range: float = 0.3, max_range: float = 10.0, require_background=False):
        self.directory = Path(directory)
        if not (self.directory / "lut.npz").exists():
            raise SystemExit(f"{self.directory} has no lut.npz: not a capture directory")
        self.sensor = LutSensorModel.load(self.directory / "lut.npz")
        self.min_range, self.max_range = min_range, max_range
        self.background_paths = sorted((self.directory / "background").glob("*.npz"), key=natural_key)
        self.frame_paths = sorted((self.directory / "frames").glob("*.npz"), key=natural_key)
        capture = self.directory / "capture.json"
        self.capture = json.loads(capture.read_text()) if capture.exists() else {}
        metadata = self.directory / "metadata.json"
        self.metadata = json.loads(metadata.read_text()) if metadata.exists() else {}
        if not self.frame_paths:
            raise SystemExit(f"no frames in {self.directory / 'frames'}")
        if require_background and not self.background_paths:
            raise SystemExit("no background frames: record the empty scene first (the capture commands do)")

    def __len__(self) -> int:
        return len(self.frame_paths)

    def background_count(self) -> int:
        return len(self.background_paths)

    def _read(self, path, index) -> Frame:
        data = np.load(path)
        range_m = data["range"].astype(np.float64) / 1000.0
        range_m[(range_m < self.min_range) | (range_m > self.max_range)] = np.nan
        return Frame(
            index=index, range_m=range_m,
            timestamps=(np.asarray(data["timestamps"], dtype=np.float64) * 1e-9
                        if "timestamps" in data.files else None),
            host_time=float(data["time"]) if "time" in data.files else float("nan"),
            phase=int(data["phase"]) if "phase" in data.files else 0,
            columns_ok=float(data["columns_ok"]) if "columns_ok" in data.files else 1.0,
            path=Path(path))

    def load(self, index: int) -> Frame:
        return self._read(self.frame_paths[index], index)

    def load_background(self, index: int) -> Frame:
        return self._read(self.background_paths[index], index)

    def background_range(self, min_fraction: float = 0.5) -> np.ndarray | None:
        """Per-pixel median range of the background frames (NaN where valid in
        fewer than 'min_fraction' of them); None without background frames."""
        if not self.background_paths:
            return None
        return median_range([self.load_background(k).range_m for k in range(self.background_count())],
                            min_fraction)


class PointCloudFolder(FrameSource):
    """One point cloud file per frame (.ply, .pcd, .xyz, or .npz with 'points'),
    points in the sensor frame. Optional subfolder 'background' with the empty
    scene. Frame times are index x 'period' seconds."""

    SUFFIXES = (".ply", ".pcd", ".xyz", ".npz")

    def __init__(self, directory, period: float = 0.1):
        self.directory = Path(directory)
        self.sensor = None
        self.period = period
        frames_dir = self.directory / "frames" if (self.directory / "frames").is_dir() else self.directory
        self.frame_paths = sorted((p for p in frames_dir.iterdir() if p.suffix.lower() in self.SUFFIXES),
                                  key=natural_key)
        background_dir = self.directory / "background"
        self.background_paths = sorted((p for p in background_dir.iterdir() if p.suffix.lower() in self.SUFFIXES),
                                       key=natural_key) if background_dir.is_dir() else []
        if not self.frame_paths:
            raise SystemExit(f"no point cloud files in {frames_dir}")
        self.capture = {}

    def __len__(self) -> int:
        return len(self.frame_paths)

    def background_count(self) -> int:
        return len(self.background_paths)

    @staticmethod
    def _points(path) -> np.ndarray:
        if Path(path).suffix.lower() == ".npz":
            data = np.load(path)
            key = "points" if "points" in data.files else data.files[0]
            points = np.asarray(data[key], dtype=np.float64).reshape(-1, 3)
        else:
            import open3d as o3d
            points = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float64)
        return points[np.all(np.isfinite(points), axis=1)]

    def load(self, index: int) -> Frame:
        return Frame(index=index, points=self._points(self.frame_paths[index]), host_time=index * self.period,
                     path=self.frame_paths[index])

    def load_background(self, index: int) -> Frame:
        return Frame(index=index, points=self._points(self.background_paths[index]), host_time=0.0,
                     path=self.background_paths[index])


def open_recording(directory, min_range: float = 0.3, max_range: float = 10.0, period: float = 0.1) -> FrameSource:
    """NpzRecording if the directory has a lut.npz, else PointCloudFolder."""
    if (Path(directory) / "lut.npz").exists():
        return NpzRecording(directory, min_range, max_range)
    return PointCloudFolder(directory, period)


def median_range(ranges, min_fraction: float) -> np.ndarray:
    """Per-pixel median over frames; NaN where valid in fewer than min_fraction of them."""
    stack = np.stack(ranges)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(stack, axis=0)
    enough = np.mean(np.isfinite(stack), axis=0) >= min_fraction
    return np.where(enough, median, np.nan)
