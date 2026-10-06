"""Floor plane and the floor frame (z up, z = 0 on the floor).

FloorEstimator is the interface; RansacFloor finds the floor in a static scene
for any sensor orientation, given a rough 'up' direction in the sensor frame
(upright sensor: (0, 0, 1); sensor on its side: the sensor axis that points
up). CropBoxFloor fits the floor only around a crop box, for scenes where the
dominant plane below the sensor is not the floor near the person (a pipeline
passes it to SceneFromBackground)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from bodyscan.config import param
from bodyscan.geometry import Plane, ransac_planes, refit_plane, rotation_to_z, transform_points


@dataclass
class FloorConfig:
    """Floor detection in the empty scene (defines the floor frame: z up, z = 0 on the floor)."""
    up: tuple[float, float, float] = param(
        (0.0, 0.0, 1.0), "rough up direction in the sensor frame",
        effect="(0, 0, 1) for an upright sensor; for a sensor mounted on its side give the sensor axis that "
               "points up, e.g. (0, 1, 0)")
    max_tilt_deg: float = param(60.0, "largest angle between the floor normal and the up direction", unit="deg",
                                effect="larger: accepts a more tilted sensor, but also walls when close to 90")
    min_sensor_height: float = param(0.2, "lowest accepted height of the sensor above the floor", unit="m")
    max_sensor_height: float = param(4.0, "highest accepted height of the sensor above the floor", unit="m",
                                     effect="excludes the ceiling and far planes")
    ransac_threshold: float = param(0.02, "RANSAC inlier distance of the plane search", unit="m")
    refit_band: float = param(0.015, "final least-squares refit on the points within this of the plane", unit="m")


@dataclass
class FloorFrame:
    """world <- sensor transform with z along the floor normal and z = 0 on the floor."""
    world_from_sensor: np.ndarray
    sensor_height: float
    tilt_deg: float

    def to_world(self, points: np.ndarray) -> np.ndarray:
        return transform_points(self.world_from_sensor, points)

    @property
    def sensor_position(self) -> np.ndarray:
        return self.world_from_sensor[:3, 3].copy()

    @classmethod
    def from_plane(cls, plane: Plane, up: np.ndarray) -> "FloorFrame":
        normal, offset = plane.normal, plane.offset
        if offset < 0:                                   # sensor (origin) on the positive side
            normal, offset = -normal, -offset
        transform = np.eye(4)
        transform[:3, :3] = rotation_to_z(normal)
        transform[:3, 3] = [0.0, 0.0, offset]
        tilt = float(np.degrees(np.arccos(np.clip(normal @ up, -1.0, 1.0))))
        return cls(transform, float(offset), tilt)


class FloorEstimator(ABC):
    @abstractmethod
    def estimate(self, points: np.ndarray) -> FloorFrame:
        """Floor frame from (N, 3) sensor-frame points of a static scene."""


class RansacFloor(FloorEstimator):
    """The plane with the most points among the planes BELOW the sensor (within
    the accepted heights) whose normal is within max_tilt_deg of 'up': the
    sensor may be tilted, or rolled about its viewing axis. Walls (normal
    about 90 deg from up) and the ceiling (above the sensor) are excluded.
    Planes are found one after the other by RANSAC on a subsample of the
    empty scene, then the chosen one is refitted to all the points within
    refit_band."""

    def __init__(self, config: FloorConfig | None = None):
        self.config = config or FloorConfig()

    def estimate(self, points: np.ndarray) -> FloorFrame:
        c = self.config
        up = np.asarray(c.up, dtype=np.float64)
        up /= np.linalg.norm(up)
        candidates = []
        for plane in ransac_planes(points, count=6, threshold=c.ransac_threshold, min_inliers=500):
            normal, offset = plane.normal, plane.offset
            if offset < 0:
                normal, offset = -normal, -offset
            tilt = float(np.degrees(np.arccos(np.clip(normal @ up, -1.0, 1.0))))
            if tilt <= c.max_tilt_deg and c.min_sensor_height <= offset <= c.max_sensor_height:
                candidates.append(Plane(normal, offset, plane.inliers))
        if not candidates:
            raise SystemExit("no floor found in the empty scene: no large plane below the sensor within "
                             f"{c.max_tilt_deg:g} deg of the up direction {tuple(up.round(3))} "
                             "(sensor on its side? set [floor] up)")
        best = max(candidates, key=lambda p: p.inliers)
        refined = refit_plane(points, best, c.refit_band)
        if refined.offset < 0:
            refined = Plane(-refined.normal, -refined.offset, refined.inliers)
        return FloorFrame.from_plane(refined, up)


class CropBoxFloor(FloorEstimator):
    """Floor fitted to the empty scene inside the horizontal footprint of a
    crop box (sensor frame), below the 60th height percentile; the dominant
    plane there must be within max_tilt_deg of the sensor z axis. Origin below
    the sensor."""

    def __init__(self, box_min, box_max, max_tilt_deg: float = 25.0):
        self.box_min = np.asarray(box_min, dtype=np.float64)
        self.box_max = np.asarray(box_max, dtype=np.float64)
        self.max_tilt_deg = max_tilt_deg

    def estimate(self, points: np.ndarray) -> FloorFrame:
        import open3d as o3d
        with np.errstate(invalid="ignore"):
            footprint = np.all((points[:, :2] >= self.box_min[:2] - 0.5) & (points[:, :2] <= self.box_max[:2] + 0.5),
                               axis=1)
        points = points[footprint]
        points = points[points[:, 2] < np.nanpercentile(points[:, 2], 60)] if len(points) else points
        if len(points) < 200:
            raise SystemExit("too few background points near the crop box to fit the floor")
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        plane, inliers = cloud.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=2000)
        normal = np.array(plane[:3])
        offset = plane[3]
        scale = np.linalg.norm(normal)
        normal, offset = normal / scale, offset / scale
        if normal[2] < 0:
            normal, offset = -normal, -offset
        tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1, 1))))
        if tilt > self.max_tilt_deg:
            raise SystemExit(f"the dominant plane near the box is tilted {tilt:.0f} deg: not a floor. "
                             "Check the crop box.")
        transform = np.eye(4)
        transform[:3, :3] = rotation_to_z(normal)
        transform[:3, 3] = [0.0, 0.0, offset]
        return FloorFrame(transform, float(offset), tilt)
