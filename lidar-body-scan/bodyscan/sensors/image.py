"""One frame of one sensor as rays, ranges and per-pixel state, and its simulation by ray casting.

RangeImage is the common currency of every sensor: a spinning LiDAR frame and
a depth-camera frame both become (H, W) arrays of

    directions   unit ray directions in the sensor frame
    origins      ray origins in the sensor frame (beam offsets; zero for a pinhole)
    range        measured distance along the ray [m] (NaN where there is no value)
    state        VALID (a return), NO_RETURN (the sensor measured nothing along the
                 ray: open space, or a surface that sent no light back) or UNKNOWN
                 (no information: a lost packet, a pixel the camera flagged invalid)
    times        measurement time of every pixel [s]
    sigma        one-sigma range error of every pixel [m]

plus world_from_sensor (4 x 4). The free-space test and the error heatmap
treat UNKNOWN pixels as missing, never as seen through.

render() simulates a frame: rays cast against triangle meshes (Open3D), ranges
beyond the sensor's limits become NO_RETURN, and Gaussian noise of the sensor's
model is added (a depth camera's noise is on the depth, then converted to the
ray).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from bodyscan.sensors.models import DepthCamera, SensorModel

VALID, NO_RETURN, UNKNOWN = 0, 1, 2


@dataclass
class RangeImage:
    sensor: str
    directions: np.ndarray
    origins: np.ndarray
    range: np.ndarray
    state: np.ndarray
    times: np.ndarray
    sigma: np.ndarray
    world_from_sensor: np.ndarray

    def __post_init__(self):
        shape = self.range.shape
        for name in ("state", "times", "sigma"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"RangeImage.{name} has shape {getattr(self, name).shape}, expected {shape}")
        if self.directions.shape != shape + (3,) or self.origins.shape != shape + (3,):
            raise ValueError("RangeImage directions and origins must be (H, W, 3)")

    @property
    def shape(self) -> tuple:
        return self.range.shape

    @property
    def rotation(self) -> np.ndarray:
        return np.asarray(self.world_from_sensor, dtype=np.float64)[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        return np.asarray(self.world_from_sensor, dtype=np.float64)[:3, 3]

    def world_rays(self, mask=None) -> tuple[np.ndarray, np.ndarray]:
        """World ray origins and unit directions, (N, 3) each, of the pixels in 'mask' (all if None)."""
        origins = self.origins if mask is None else self.origins[mask]
        directions = self.directions if mask is None else self.directions[mask]
        origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        return origins @ self.rotation.T + self.translation, directions @ self.rotation.T

    def world_points(self, mask=None) -> np.ndarray:
        """(N, 3) world points of the VALID pixels in 'mask'."""
        valid = self.state == VALID
        chosen = valid if mask is None else (valid & mask)
        origins, directions = self.world_rays(chosen)
        return origins + directions * self.range[chosen].reshape(-1, 1)


def mesh_scene(meshes) -> o3d.t.geometry.RaycastingScene:
    """A ray-casting scene of meshes given as (vertices, faces) pairs or Open3D legacy meshes."""
    scene = o3d.t.geometry.RaycastingScene()
    for mesh in meshes:
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        else:
            vertices, faces = mesh
            tensor = o3d.t.geometry.TriangleMesh()
            tensor.vertex.positions = o3d.core.Tensor(np.asarray(vertices, dtype=np.float32))
            tensor.triangle.indices = o3d.core.Tensor(np.asarray(faces, dtype=np.int32))
            scene.add_triangles(tensor)
    return scene


def cast(scene: o3d.t.geometry.RaycastingScene, origins: np.ndarray, directions: np.ndarray) -> dict:
    """First hits of world rays (N, 3): distance (inf: no hit), triangle index (-1: none), barycentric
    (u, v) of the hit in its triangle (the weights of its corners are 1 - u - v, u, v), and the
    geometry index of the mesh hit (order of addition to the scene)."""
    rays = np.concatenate([np.asarray(origins, dtype=np.float32), np.asarray(directions, dtype=np.float32)], axis=1)
    answer = scene.cast_rays(o3d.core.Tensor(rays))
    distance = answer["t_hit"].numpy().astype(np.float64)
    triangle = answer["primitive_ids"].numpy().astype(np.int64)
    uv = answer["primitive_uvs"].numpy().astype(np.float64)
    geometry = answer["geometry_ids"].numpy().astype(np.int64)
    missed = ~np.isfinite(distance)
    triangle[missed] = -1
    geometry[missed] = -1
    return {"distance": distance, "triangle": triangle, "uv": uv, "geometry": geometry}


def render(sensor: SensorModel, world_from_sensor: np.ndarray, scene, frame_start: float = 0.0,
           rng: np.random.Generator | None = None, name: str = "", mask=None) -> RangeImage:
    """Simulated RangeImage of a frame: rays of 'sensor' placed at world_from_sensor cast against
    'scene' (a RaycastingScene or a list of meshes). With an rng, Gaussian range noise of the sensor
    model is added. 'mask' (H, W) restricts the cast to some pixels (the others are NO_RETURN):
    a frame that only needs the rays near a person casts only those."""
    if not isinstance(scene, o3d.t.geometry.RaycastingScene):
        scene = mesh_scene(scene)
    directions = sensor.directions()
    shape = directions.shape[:2]
    origins = np.zeros(shape + (3,))
    rotation = np.asarray(world_from_sensor, dtype=np.float64)[:3, :3]
    translation = np.asarray(world_from_sensor, dtype=np.float64)[:3, 3]
    chosen = np.ones(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    distance = np.full(shape, np.inf)
    if chosen.any():
        world_directions = directions[chosen] @ rotation.T
        world_origins = np.broadcast_to(translation, world_directions.shape)
        distance[chosen] = cast(scene, world_origins, world_directions)["distance"]
    if isinstance(sensor, DepthCamera):
        depth = sensor.depth_of(distance, directions)
        inside = np.isfinite(depth) & (depth >= sensor.min_range) & (depth <= sensor.max_range)
    else:
        inside = np.isfinite(distance) & (distance >= sensor.min_range) & (distance <= sensor.max_range)
    sigma = np.where(inside, sensor.range_sigma(np.where(inside, distance, 1.0), directions), 0.0)
    measured = np.where(inside, distance, np.nan)
    if rng is not None:
        measured = np.where(inside, measured + rng.normal(0.0, 1.0, shape) * sigma, np.nan)
    state = np.where(inside, VALID, NO_RETURN).astype(np.uint8)
    return RangeImage(name or sensor.model, directions, origins, measured, state, sensor.pixel_times(frame_start),
                      sigma, np.asarray(world_from_sensor, dtype=np.float64))
