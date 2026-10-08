"""Foreground isolation: from a range image to the cloud of the object of interest.

ForegroundIsolator chains the tests every pipeline needs (closer than the
empty scene, inside a region of interest, not a mixed edge pixel, largest
cluster, no statistical outliers, normals towards the sensor). The region of
interest is a Region object: the fusion pipelines measure it on the object
found by detection.finder (a cylinder about its centre reaching its farthest
points), or take a crop box given by the user."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import open3d as o3d

from bodyscan.config import param
from bodyscan.geometry import largest_cluster
from bodyscan.scene.background import RangeBackground, mixed_pixel_mask
from bodyscan.scene.floor import FloorFrame


@dataclass
class IsolationConfig:
    """Cutting the object out of every frame (floor frame), inside its region of interest."""
    radius: float | None = param(None, "region of interest: radius about the centre of the object; not set: "
                                       "measured on the object found ([subject] margin beyond its farthest point)",
                                 unit="m", effect="set it only to impose a region; it must include the hands "
                                                  "(0.55 cut hands held out sideways in tt16)")
    min_height: float = param(0.05, "drop points below this height above the floor", unit="m",
                              effect="lower keeps more of the shoes (platform top at 0.03 m) but may keep "
                                     "platform noise and reflections")
    max_height: float = param(2.3, "drop points above this height", unit="m")
    bg_threshold: float = param(0.05, "a pixel is foreground if closer than the empty scene by this", unit="m",
                                effect="smaller keeps feet and soles closer to the platform, more noise")
    bg_relative: float = param(0.01, "... or by this fraction of its range, whichever is larger")
    edge_jump: float = param(0.05, "mixed-pixel filter: range jump on both sides of a pixel (0 = off)", unit="m",
                             effect="removes the ghost points between an edge and the background")
    cluster_eps: float = param(0.06, "only the largest cluster is kept; DBSCAN distance (0 = off)", unit="m",
                               effect="smaller may split off hands or feet; larger merges nearby objects")
    normal_radius: float = param(0.05, "neighbourhood of the per-frame normal estimate", unit="m")
    min_columns: float = param(0.90, "drop a frame that received less than this fraction of its columns "
                                     "(lost UDP packets)")
    max_person_loss: float = param(0.10, "drop a frame if more than this fraction of the columns across the "
                                         "person was lost (UDP packets)")


class Region(ABC):
    """Region of interest."""

    @abstractmethod
    def contains(self, world: np.ndarray, sensor: np.ndarray) -> np.ndarray:
        """Boolean mask, from points in the floor frame ('world') and in the sensor frame (same shape)."""


class CylinderRegion(Region):
    """Vertical cylinder around a centre (floor frame), between two heights."""

    def __init__(self, center, radius, z_min, z_max):
        self.center = np.asarray(center, dtype=np.float64)
        self.radius, self.z_min, self.z_max = radius, z_min, z_max

    def contains(self, world, sensor):
        with np.errstate(invalid="ignore"):
            radial = np.linalg.norm(world[..., :2] - self.center, axis=-1)
            return (radial < self.radius) & (world[..., 2] > self.z_min) & (world[..., 2] < self.z_max)


class SensorBoxRegion(Region):
    """Box in the sensor frame, plus a height range above the floor."""

    def __init__(self, box_min, box_max, z_min, z_max):
        self.box_min = np.asarray(box_min, dtype=np.float64)
        self.box_max = np.asarray(box_max, dtype=np.float64)
        self.z_min, self.z_max = z_min, z_max

    def contains(self, world, sensor):
        with np.errstate(invalid="ignore"):
            in_box = np.all((sensor >= self.box_min) & (sensor <= self.box_max), axis=-1)
            return in_box & (world[..., 2] > self.z_min) & (world[..., 2] < self.z_max)


@dataclass
class IsolatedCloud:
    """Foreground cloud of one frame (floor frame) with normals towards the sensor;
    'time' is the mean sensor time of its points and 'point_times' the sensor time
    of every point (its column), when the frame has timestamps."""
    cloud: o3d.geometry.PointCloud
    time: float | None
    point_times: np.ndarray | None


class ForegroundIsolator:
    """Range image to the foreground cloud of one object."""

    def __init__(self, sensor, background: RangeBackground, floor: FloorFrame, region: Region,
                 edge_jump: float = 0.05, cluster_eps: float = 0.06, normal_radius: float = 0.05):
        self.sensor, self.background, self.floor, self.region = sensor, background, floor, region
        self.edge_jump, self.cluster_eps, self.normal_radius = edge_jump, cluster_eps, normal_radius

    def world(self, range_m: np.ndarray):
        xyz = self.sensor.xyz(range_m)
        transform = self.floor.world_from_sensor
        return np.einsum("ij,hwj->hwi", transform[:3, :3], xyz) + transform[:3, 3], xyz

    def mask(self, range_m: np.ndarray):
        """Foreground pixels and the (H, W, 3) floor-frame points."""
        world, xyz = self.world(range_m)
        mask = self.background.foreground(range_m) & self.region.contains(world, xyz)
        if self.edge_jump > 0:
            mask &= ~mixed_pixel_mask(range_m, self.edge_jump)
        return mask, world

    def cloud(self, range_m: np.ndarray, timestamps: np.ndarray | None = None) -> IsolatedCloud:
        """The person cloud.

        The LiDAR sweeps the 360 degrees in one frame period; if the person
        straddles the column where the sweep starts, the two sides of the body
        are measured almost one period apart. The mean time is continuous in
        that case (a median would jump by a whole period), and the per-point
        times let the views undo the turn within the frame."""
        mask, world = self.mask(range_m)
        points = world[mask]
        point_times = None
        if timestamps is not None:
            point_times = np.broadcast_to(timestamps[None, :], mask.shape)[mask]
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        keep = np.arange(len(points))
        if self.cluster_eps > 0 and len(points) >= 20:
            keep = largest_cluster(points, self.cluster_eps, 10)
            if len(keep) < len(points):
                cloud = cloud.select_by_index(keep)
        if len(cloud.points) > 30:
            cloud, inliers = cloud.remove_statistical_outlier(20, 2.0)
            keep = keep[np.asarray(inliers, dtype=np.int64)]
        if len(cloud.points) >= 3:
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=30))
            cloud.orient_normals_towards_camera_location(self.floor.sensor_position)
        when = None
        if point_times is not None:
            point_times = point_times[keep]
            valid = point_times > 0
            if valid.any():
                when = float(np.mean(point_times[valid]))
                point_times = np.where(valid, point_times, when)
            else:
                point_times = None
        return IsolatedCloud(cloud, when, point_times)


def frame_complete(mask: np.ndarray, columns_ok: float, timestamps, min_columns: float, max_person_loss: float,
                   margin: int = 64) -> bool:
    """False if the frame lost too many columns (UDP packets of 16 columns)
    overall, or too many of the columns across the person. A lost packet
    elsewhere in the 360 deg sweep does not matter for the person. 'margin'
    covers the per-row pixel shift between measurement columns and image
    columns (up to 63 on this sensor)."""
    if columns_ok < min_columns:
        return False
    if timestamps is None or columns_ok >= 0.999 or not mask.any():
        return True
    lost = timestamps == 0
    person = np.flatnonzero(mask.any(axis=0))
    if person.size == 0:
        return True
    width = mask.shape[1]
    # columns spanned by the person (handles a person across the start of the sweep)
    span = np.zeros(width, dtype=bool)
    gaps = np.diff(np.concatenate([person, [person[0] + width]]))
    start = person[(np.argmax(gaps) + 1) % person.size]
    length = width - int(gaps.max()) + 1
    span[(start + np.arange(length)) % width] = True
    widened = span.copy()
    for shift in range(-margin, margin + 1, 8):
        widened |= np.roll(span, shift)
    return lost[widened].mean() <= max_person_loss
