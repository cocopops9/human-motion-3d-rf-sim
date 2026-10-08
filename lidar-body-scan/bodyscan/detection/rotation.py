"""Is a tracked object turning about a vertical axis, and where is the axis?

For pairs of observations of one track, a registration with a turn about the
vertical and a shift (4 DOF) gives p' = R p + t. A turn by theta about a
vertical axis through c is t = (I - R) c, so c = (I - R)^-1 t (well defined
once theta is a few degrees). A rotating object gives the same angular speed
(theta / time between the frames) and the same axis for every pair; a static
object gives theta near 0; a walking person gives axes all over the place; a
rotationally symmetric object (a bin, the platform) gives random turns.

A recording may hold the object still for a while (a turntable starts after
the person stepped on it, and stops after the last lap): pairs slower than
min_speed are still pairs (as are pairs that turned less than still_turn), kept out of the speed, sign and axis statistics
and counted apart. The turn covered by the track (the neighbouring turns
summed from the first turning one to the last) can be required to reach min_total_turn
(360 deg: the object made a whole lap, every side was seen).
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
    still_turn: float = param(1.0, "pairs that turned less than this (or slower than min_speed) are still pairs: "
                                   "the object was not turning between them", unit="deg",
                              effect="above the registration noise (a few tenths of a degree)")
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
    min_total_turn: float = param(0.0, "the object must turn at least this much over the recording (0: any turn; "
                                       "360: a whole lap)", unit="deg",
                                  effect="a turntable scan needs every side: 'bodyscan fuse' asks for most of a lap")


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
    still_pairs: int = 0
    turned_deg: float = 0.0

    def as_dict(self) -> dict:
        axis = None if self.axis is None else [round(float(v), 4) for v in self.axis]
        return {"rotating": self.rotating, "axis": axis,
                "speed_deg_s": round(self.speed_deg_s, 3), "pairs": self.pairs, "still_pairs": self.still_pairs,
                "turned_deg": round(self.turned_deg, 1),
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
        # Neighbouring observations first, with a wide search. A longer pair is
        # then searched around the sum of the neighbouring turns it spans (or the
        # median turning speed times its duration when one of them is missing),
        # which also holds across a start or a stop of the rotation.
        records = []
        steps = {}
        for i in range(len(clusters) - 1):
            transform, fitness, _ = self.register(clouds[i], targets[i + 1], np.arange(-45.0, 45.1, 15.0))
            if fitness >= c.min_fitness:
                records.append((i, i + 1, transform, fitness))
                steps[i] = np.degrees(yaw_of(transform))
        step_rates = np.array([steps[i] / max(times[i + 1] - times[i], 1e-6) for i in steps])
        step_turns = np.array([steps[i] for i in steps])
        turning = step_rates[(np.abs(step_rates) >= c.min_speed) & (np.abs(step_turns) >= c.still_turn)]
        speed = float(np.median(turning)) if turning.size else 0.0
        lags = [2, 3, 5, 8]
        candidates = [(i, i + lag) for lag in lags for i in range(len(clusters) - lag)]
        if len(candidates) > c.max_pairs:
            candidates = [candidates[k] for k in np.linspace(0, len(candidates) - 1, c.max_pairs).round().astype(int)]
        for i, j in candidates:
            if all(k in steps for k in range(i, j)):
                expected = sum(steps[k] for k in range(i, j))
            else:
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
        moving = (np.abs(rates) >= c.min_speed) & (np.abs(turns) >= c.still_turn)
        still = int(np.sum(~moving))
        if moving.sum() < c.min_pairs:
            return RotationResult(False, None, float(np.median(rates)), len(records), 0.0, 0.0, 0.0,
                                  f"only {int(moving.sum())} turning pairs ({still} still): static or slower "
                                  f"than {c.min_speed:g} deg/s", still_pairs=still)
        records = [r for r, m in zip(records, moving) if m]
        turns, rates = turns[moving], rates[moving]
        speed = float(np.median(rates))
        sign_agreement = float(np.mean(np.sign(rates) == np.sign(speed)))
        speed_spread = float(1.4826 * np.median(np.abs(rates - speed)) / max(abs(speed), 1e-6))
        turned = self.turned(steps, times, speed, records)
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
        if sign_agreement < c.min_sign_agreement:
            reasons.append(f"turn sign agreement {sign_agreement:.2f}")
        if speed_spread > c.max_speed_spread:
            reasons.append(f"speed spread {speed_spread:.2f}")
        if axis is None or axis_spread > c.max_axis_spread:
            reasons.append("no consistent axis" if axis is None else f"axis spread {axis_spread:.3f} m")
        if turned < c.min_total_turn:
            reasons.append(f"turned {turned:.0f} deg, less than {c.min_total_turn:g}")
        rotating = not reasons
        if rotating:
            axis = self.refine_axis(clouds, records, turns, axis)
        return RotationResult(rotating, axis, speed, len(records) + still, speed_spread,
                              float(axis_spread) if np.isfinite(axis_spread) else -1.0, sign_agreement,
                              "rotating" if rotating else "; ".join(reasons), still_pairs=still, turned_deg=turned)

    def turned(self, steps, times, speed, records) -> float:
        """Turn covered by the track [deg]: the sum of the neighbouring turns from
        the first to the last turning neighbour pair (a missing one counts as
        speed times its duration); without turning neighbour pairs, speed times
        the time spanned by the turning pairs."""
        c = self.config
        moving = [i for i, turn in steps.items()
                  if abs(turn) >= c.still_turn and abs(turn) / max(times[i + 1] - times[i], 1e-6) >= c.min_speed]
        if not moving:
            return abs(speed) * float(times[max(r[1] for r in records)] - times[min(r[0] for r in records)])
        return float(sum(abs(steps[i]) if i in steps else abs(speed) * (times[i + 1] - times[i])
                         for i in range(min(moving), max(moving) + 1)))

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
