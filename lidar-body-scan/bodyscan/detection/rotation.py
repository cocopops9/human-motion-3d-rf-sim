"""Is a tracked object turning about a vertical axis, and where is the axis?

For pairs of observations of one track, a registration with a turn about the
vertical and a shift (4 DOF) gives p' = R p + t. A turn by theta about a
vertical axis through c is t = (I - R) c, so c = (I - R)^-1 t (well defined
once theta is a few degrees). A rotating object gives the same angular speed
(theta / time between the frames) and the same axis for every pair; a static
object gives theta near 0; a walking person gives axes all over the place; a
rotationally symmetric object (a bin, the platform) gives random turns.
The axis is finally refined by the joint fit of the turntable pipeline
(motion.joint_axis_fit), which measures it to a few mm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from bodyscan.config import param
from bodyscan.geometry import Target, evaluate, yaw_of, yaw_transform
from bodyscan.motion import Samples, joint_axis_fit
from bodyscan.registration import icp_4dof


@dataclass
class RotationConfig:
    """Rotation test of every tracked object."""
    min_speed: float = param(0.5, "slower objects are not rotating", unit="deg/s")
    min_turn: float = param(5.0, "pairs that turned less than this do not give an axis", unit="deg")
    max_axis_spread: float = param(0.10, "the axes of the pairs must agree within this (robust spread)", unit="m",
                                   effect="larger accepts noisier rotations, and also some walking people")
    max_speed_spread: float = param(0.35, "the speeds of the pairs must agree within this fraction (robust spread)")
    min_sign_agreement: float = param(0.8, "fraction of pairs turning the same way")
    min_pairs: int = param(4, "pairs needed for a decision")
    max_pairs: int = param(40, "at most this many pairs per object")
    min_fitness: float = param(0.5, "pairs whose registration overlap is below this are ignored")
    registration_voxel: float = param(0.02, "voxel of the clouds registered", unit="m")
    group_distance: float = param(0.15, "rotating objects whose axes are closer than this, turning at the same "
                                        "speed, are parts of one body (an arm split from the torso)", unit="m",
                                  effect="larger also joins two objects on platforms close to each other")
    group_speed: float = param(0.35, "speeds of parts of one body agree within this fraction (same direction)")


@dataclass
class RotationResult:
    rotating: bool
    axis: np.ndarray | None
    speed_deg_s: float
    pairs: int
    speed_spread: float
    axis_spread_m: float
    sign_agreement: float
    reason: str

    def as_dict(self) -> dict:
        axis = None if self.axis is None else [round(float(v), 4) for v in self.axis]
        return {"rotating": self.rotating, "axis": axis,
                "speed_deg_s": round(self.speed_deg_s, 3), "pairs": self.pairs,
                "speed_spread": round(self.speed_spread, 3), "axis_spread_m": round(self.axis_spread_m, 4),
                "sign_agreement": round(self.sign_agreement, 3), "reason": self.reason}


def _cloud(points, sensor, voxel):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(voxel)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    cloud.orient_normals_towards_camera_location(sensor)
    return cloud


class RotationAnalyzer:
    def __init__(self, config: RotationConfig):
        self.config = config

    def register(self, source, target, expected_deg):
        """4-DOF registration from turns about the source centroid; best by overlap then rmse."""
        points = np.asarray(source.points)
        center = points.mean(axis=0)
        best = None
        for start in expected_deg:
            transform = icp_4dof(points, target, yaw_transform(np.radians(start), center),
                                 (0.08, 0.04, 0.02), (20, 15, 15))
            fitness, rmse = evaluate(points, target, transform, 0.03)
            if best is None or fitness > best[1] + 0.02 or (fitness > best[1] - 0.02 and rmse < best[2]):
                best = (transform, fitness, rmse)
        return best

    def unchanged(self, source, target) -> bool:
        """True when 'source' lies on 'target' without moving: a registration from
        the identity keeps most points within 3 cm, turns less than a degree and
        shifts less than 2 cm."""
        points = np.asarray(source.points)
        transform = icp_4dof(points, target, np.eye(4), (0.05, 0.03), (10, 10))
        fitness, _ = evaluate(points, target, transform, 0.03)
        return (fitness >= 0.8 and abs(np.degrees(yaw_of(transform))) < 1.0
                and np.linalg.norm(transform[:2, 3]) < 0.02)

    def analyze(self, track) -> RotationResult:
        c = self.config
        clusters = track.clusters
        if len(clusters) < 3:
            return RotationResult(False, None, 0.0, 0, 0.0, 0.0, 0.0, "seen in fewer than 3 frames")
        clouds = [_cloud(cl.points, cl.sensor, c.registration_voxel) for cl in clusters]
        targets = [Target(cloud) for cloud in clouds]
        times = track.times
        # Cheap test first (most objects of a room are furniture): an object found
        # unchanged between the first, middle and last frames is static.
        middle = len(clusters) // 2
        if all(self.unchanged(clouds[i], targets[j]) for i, j in ((0, middle), (middle, -1), (0, -1))):
            return RotationResult(False, None, 0.0, 0, 0.0, 0.0, 0.0, "static")
        # Neighbouring observations first, with a wide search; their median speed
        # predicts the turn of the longer pairs.
        records = []
        for i in range(len(clusters) - 1):
            transform, fitness, _ = self.register(clouds[i], targets[i + 1], np.arange(-45.0, 45.1, 15.0))
            if fitness >= c.min_fitness:
                records.append((i, i + 1, transform, fitness))
        speeds = [np.degrees(yaw_of(r[2])) / max(times[r[1]] - times[r[0]], 1e-6) for r in records]
        speed = float(np.median(speeds)) if speeds else 0.0
        lags = [2, 3, 5, 8]
        candidates = [(i, i + lag) for lag in lags for i in range(len(clusters) - lag)]
        if len(candidates) > c.max_pairs:
            candidates = [candidates[k] for k in np.linspace(0, len(candidates) - 1, c.max_pairs).round().astype(int)]
        for i, j in candidates:
            expected = speed * (times[j] - times[i])
            if abs(expected) > 150.0:
                continue
            transform, fitness, _ = self.register(clouds[i], targets[j], expected + np.array([-15.0, 0.0, 15.0]))
            if fitness >= c.min_fitness:
                records.append((i, j, transform, fitness))
        if len(records) < c.min_pairs:
            return RotationResult(False, None, speed, len(records), 0.0, 0.0, 0.0,
                                  f"only {len(records)} pairs registered (object changes shape or is too small)")

        turns = np.array([np.degrees(yaw_of(r[2])) for r in records])
        gaps = np.array([times[r[1]] - times[r[0]] for r in records])
        rates = turns / np.maximum(gaps, 1e-6)
        speed = float(np.median(rates))
        sign_agreement = float(np.mean(np.sign(rates) == np.sign(speed))) if speed != 0 else 0.0
        speed_spread = float(1.4826 * np.median(np.abs(rates - speed)) / max(abs(speed), 1e-6))
        axes = []
        for (i, j, transform, _), turn in zip(records, turns):
            if abs(turn) < c.min_turn:
                continue
            rotation = transform[:2, :2]
            try:
                axes.append(np.linalg.solve(np.eye(2) - rotation, transform[:2, 3]))
            except np.linalg.LinAlgError:
                continue
        axes = np.array(axes)
        axis, axis_spread = None, np.inf
        if len(axes) >= 2:
            axis = np.median(axes, axis=0)
            axis_spread = float(1.4826 * np.median(np.linalg.norm(axes - axis, axis=1)))
        reasons = []
        if abs(speed) < c.min_speed:
            reasons.append(f"speed {abs(speed):.2f} deg/s below {c.min_speed:g}")
        if sign_agreement < c.min_sign_agreement:
            reasons.append(f"turn sign agreement {sign_agreement:.2f}")
        if speed_spread > c.max_speed_spread:
            reasons.append(f"speed spread {speed_spread:.2f}")
        if axis is None or axis_spread > c.max_axis_spread:
            reasons.append("no consistent axis" if axis is None else f"axis spread {axis_spread:.3f} m")
        rotating = not reasons
        if rotating:
            axis = self.refine_axis(clouds, records, turns, axis)
        return RotationResult(rotating, axis, speed, len(records), speed_spread,
                              float(axis_spread) if np.isfinite(axis_spread) else -1.0, sign_agreement,
                              "rotating" if rotating else "; ".join(reasons))

    def refine_axis(self, clouds, records, turns, axis):
        """Joint fit of the axis and the turn of every pair 10 to 90 deg apart."""
        chosen = [(r[0], r[1], np.radians(t)) for r, t in zip(records, turns) if 10.0 <= abs(t) <= 90.0]
        if len(chosen) < 3:
            return axis
        samples = Samples(clouds, self.config.registration_voxel, 0.03)
        refined, _, fitness = joint_axis_fit(samples, [(i, j) for i, j, _ in chosen], [t for _, _, t in chosen],
                                             np.asarray(axis, dtype=np.float64), iterations=(6, 5, 5),
                                             distances=(0.05, 0.03, 0.02))
        if np.linalg.norm(refined - axis) > 0.15 or np.median(fitness) < 0.3:
            return axis
        return refined
