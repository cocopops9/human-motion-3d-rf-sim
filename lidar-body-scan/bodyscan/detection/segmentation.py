"""Objects in every frame: points above the floor, grouped into clusters.

Segmenters with the same interface (segment(frame) -> list of Cluster):

    BackgroundSegmenter       range images with empty-scene frames: what is closer
                              than the empty scene (people and objects brought in)
    PointBackgroundSegmenter  point clouds with empty-scene clouds: what is not
                              near a point of the empty scene
    ObjectSegmenter           no empty scene: every point above the floor and below
                              the ceiling, minus the large vertical planes (walls);
                              static furniture is kept and left to the classifiers
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from bodyscan.config import param
from bodyscan.geometry import ransac_planes
from bodyscan.scene import RangeBackground, mixed_pixel_mask


@dataclass
class SegmentationConfig:
    """Cutting every frame into objects (floor frame)."""
    min_height: float = param(0.10, "ignore points below this height (floor, platform)", unit="m")
    max_height: float = param(2.4, "ignore points above this height (ceiling, lamps)", unit="m")
    max_distance: float = param(8.0, "ignore points farther than this from the sensor (horizontal)", unit="m")
    voxel: float = param(0.02, "points are thinned to one per voxel of this size before clustering", unit="m")
    cluster_distance: float = param(0.10, "points closer than this belong to the same object", unit="m",
                                    effect="larger merges a person with a nearby object; smaller splits arms off")
    min_points: int = param(60, "smallest object (thinned points)")
    wall_min_points: int = param(3000, "a vertical plane with at least this many points of the static scene is a "
                                       "wall and is removed (objects mode)")
    wall_distance: float = param(0.06, "points within this of a wall are removed", unit="m")
    bg_threshold: float = param(0.10, "background mode: closer than the empty scene by this", unit="m")


@dataclass
class Cluster:
    """One object in one frame (floor frame), thinned points."""
    frame: int
    time: float
    points: np.ndarray
    sensor: np.ndarray                      # sensor position (floor frame), for normals and views
    centroid: np.ndarray = field(init=False)

    def __post_init__(self):
        self.centroid = np.median(self.points, axis=0)


def cluster_points(points, voxel, distance, min_points):
    """Thin to one point per voxel, DBSCAN; list of point arrays."""
    if len(points) < min_points:
        return []
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(voxel)
    thinned = np.asarray(cloud.points)
    if len(thinned) < min_points:
        return []
    labels = np.asarray(cloud.cluster_dbscan(eps=distance, min_points=3, print_progress=False))
    result = []
    for label in range(labels.max() + 1 if labels.size else 0):
        members = thinned[labels == label]
        if len(members) >= min_points:
            result.append(members)
    return result


class Segmenter:
    def __init__(self, source, floor, config: SegmentationConfig):
        self.source, self.floor, self.config = source, floor, config

    def world_points(self, frame):
        """(N, 3) valid points of the frame in the floor frame, within the height and distance limits,
        and the boolean mask of the range image pixels used (None for point-cloud sources)."""
        raise NotImplementedError

    def _limits(self, world):
        c = self.config
        horizontal = np.hypot(world[:, 0] - self.floor.sensor_position[0], world[:, 1] - self.floor.sensor_position[1])
        return (world[:, 2] > c.min_height) & (world[:, 2] < c.max_height) & (horizontal < c.max_distance)

    def segment(self, frame, time) -> list[Cluster]:
        c = self.config
        points = self.world_points(frame)
        return [Cluster(frame.index, time, members, self.floor.sensor_position)
                for members in cluster_points(points, c.voxel, c.cluster_distance, c.min_points)]


class BackgroundSegmenter(Segmenter):
    """Foreground against the empty scene (range images with background frames)."""

    def __init__(self, source, floor, config, background_range):
        super().__init__(source, floor, config)
        self.background = RangeBackground(background_range, config.bg_threshold, 0.01)

    def world_points(self, frame):
        mask = self.background.foreground(frame.range_m) & ~mixed_pixel_mask(frame.range_m, 0.05)
        sensor_points = self.source.sensor.xyz(frame.range_m)[mask]
        world = self.floor.to_world(sensor_points)
        return world[self._limits(world)]


class PointBackgroundSegmenter(Segmenter):
    """Foreground against empty-scene point clouds (point-cloud folders with a
    background/ subfolder): a point is background when a point of the empty
    scene lies in its voxel or a neighbouring one (voxels of bg_threshold)."""

    def __init__(self, source, floor, config, background_points):
        super().__init__(source, floor, config)
        self.size = config.bg_threshold
        world = floor.to_world(background_points)
        self.keys = np.unique(self._keys(np.floor(world / self.size).astype(np.int64)))

    @staticmethod
    def _keys(cells: np.ndarray) -> np.ndarray:
        cells = cells + 2 ** 20                                      # non-negative, 21 bits per axis
        return (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]

    def world_points(self, frame):
        world = self.floor.to_world(self.source.points(frame))
        world = world[self._limits(world)]
        cells = np.floor(world / self.size).astype(np.int64)
        background = np.zeros(len(world), dtype=bool)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    keys = self._keys(cells + [dx, dy, dz])
                    found = np.searchsorted(self.keys, keys)
                    found = np.clip(found, 0, len(self.keys) - 1)
                    background |= self.keys[found] == keys
        return world[~background]


class ObjectSegmenter(Segmenter):
    """Everything above the floor except the walls found in the static scene."""

    def __init__(self, source, floor, config, static_points):
        super().__init__(source, floor, config)
        world = floor.to_world(static_points)
        self.walls = [plane for plane in ransac_planes(world, count=8, threshold=0.03,
                                                       min_inliers=config.wall_min_points, sample=60000)
                      if abs(plane.normal[2]) < 0.2]

    def world_points(self, frame):
        world = self.floor.to_world(self.source.points(frame))
        keep = self._limits(world)
        for wall in self.walls:
            keep &= np.abs(wall.distance(world)) > self.config.wall_distance
        return world[keep]
