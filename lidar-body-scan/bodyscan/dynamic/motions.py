"""Motion of one body over time: storage, smooth resampling, procedural walks and jumps, AMASS.

A PoseSequence holds the pose of one person at increasing times:

    root_rotation  (T, 3)      world rotation of the upright body, axis-angle (identity: facing +x, z up)
    root_position  (T, 3)      world position of the pelvis joint [m]
    body           (T, 21, 3)  SMPL-X body joint rotations (axis-angle, parent frames)
    left_hand      (T, 15, 3)  finger joints, or None for the relaxed mean hand of the model
    right_hand     (T, 15, 3)

Resampling is C1 (positions and velocities continuous; see
bodyscan.body.rotations.catmull_rom_weights), so a mesh sequence exported at
any rate has smooth vertex velocities, which is what a Doppler simulation needs.

Procedural motions (for the synthetic test bench; the real motion comes
from the recordings): the pelvis follows a planned path (walking: along the
route with the bob and sway of gait; jumping: squat, push-off, ballistic
flight, landing), the feet follow planned footprints, rolling from heel to
ball, and the legs come from inverse kinematics on the body model, so feet
on the floor neither slide nor go through it; the trunk and arms follow
gait and jump profiles.

AMASS (https://amass.is.tue.mpg.de, registration needed) gives real motion
capture as SMPL-X parameters; load_amass converts it to a PoseSequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bodyscan.body import skeleton
from bodyscan.body.rotations import (axis_angle_to_matrix_np, interpolate_positions, interpolate_rotations,
                                     matrix_to_axis_angle_np, rotation_z_np)

GRAVITY = 9.81


@dataclass
class PoseSequence:
    times: np.ndarray
    root_rotation: np.ndarray
    root_position: np.ndarray
    body: np.ndarray
    left_hand: np.ndarray | None = None
    right_hand: np.ndarray | None = None

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64)
        self.root_rotation = np.asarray(self.root_rotation, dtype=np.float64).reshape(-1, 3)
        self.root_position = np.asarray(self.root_position, dtype=np.float64).reshape(-1, 3)
        self.body = np.asarray(self.body, dtype=np.float64).reshape(-1, skeleton.NUM_BODY_JOINTS, 3)
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("pose sequence times must increase")

    def __len__(self) -> int:
        return len(self.times)

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0])

    def sample(self, times) -> "PoseSequence":
        """The sequence at other times (C1 interpolation; held at the ends)."""
        times = np.asarray(times, dtype=np.float64)
        hands = [None if h is None else interpolate_rotations(self.times, h, times)
                 for h in (self.left_hand, self.right_hand)]
        return PoseSequence(times, interpolate_rotations(self.times, self.root_rotation, times),
                            interpolate_positions(self.times, self.root_position, times),
                            interpolate_rotations(self.times, self.body, times), hands[0], hands[1])

    def subset(self, mask) -> "PoseSequence":
        def pick(values):
            return None if values is None else values[mask]

        return PoseSequence(self.times[mask], self.root_rotation[mask], self.root_position[mask], self.body[mask],
                            pick(self.left_hand), pick(self.right_hand))

    def shifted(self, seconds: float) -> "PoseSequence":
        return PoseSequence(self.times + seconds, self.root_rotation, self.root_position, self.body,
                            self.left_hand, self.right_hand)

    def root_matrices(self) -> np.ndarray:
        return axis_angle_to_matrix_np(self.root_rotation)

    def turned(self, angle: float) -> "PoseSequence":
        """The same motion turned by 'angle' radians about the vertical axis through the origin."""
        turn = rotation_z_np(angle)
        return PoseSequence(self.times, matrix_to_axis_angle_np(turn @ self.root_matrices()),
                            self.root_position @ turn.T, self.body, self.left_hand, self.right_hand)

    def rotations(self, model):
        """(T, 55, 3, 3) joint rotations for bodyscan.body.model (torch)."""
        return model.rotations(body=self.body, left_hand=self.left_hand, right_hand=self.right_hand,
                               count=len(self))

    def posed(self, shaped, batch: int = 256, subset=None, dtype=np.float32):
        """World vertices (T, V, 3) and joints (T, 55, 3) as numpy arrays of 'dtype'
        (the computation runs in the dtype of the model)."""
        import torch
        model = shaped.model
        vertices, joints = [], []
        for start in range(0, len(self), batch):
            part = self.subset(slice(start, start + batch))
            with torch.no_grad():
                out = shaped.pose(part.rotations(model), model.tensor(part.root_matrices()),
                                  model.tensor(part.root_position), subset=subset)
            vertices.append(out.vertices.cpu().numpy().astype(dtype))
            joints.append(out.joints.cpu().numpy().astype(dtype))
        return np.concatenate(vertices), np.concatenate(joints)

    def save(self, path, **extra) -> None:
        arrays = {"times": self.times, "root_rotation": self.root_rotation, "root_position": self.root_position,
                  "body": self.body}
        if self.left_hand is not None:
            arrays["left_hand"] = self.left_hand
        if self.right_hand is not None:
            arrays["right_hand"] = self.right_hand
        arrays.update(extra)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path) -> "PoseSequence":
        data = np.load(path, allow_pickle=False)
        return cls(data["times"], data["root_rotation"], data["root_position"], data["body"],
                   data["left_hand"] if "left_hand" in data.files else None,
                   data["right_hand"] if "right_hand" in data.files else None)


def concatenate(parts: list[PoseSequence]) -> PoseSequence:
    """Sequences one after the other (each must start after the previous one ends)."""
    hands = []
    for side in ("left_hand", "right_hand"):
        values = [getattr(p, side) for p in parts]
        hands.append(None if any(v is None for v in values) else np.concatenate(values))
    return PoseSequence(np.concatenate([p.times for p in parts]), np.concatenate([p.root_rotation for p in parts]),
                        np.concatenate([p.root_position for p in parts]), np.concatenate([p.body for p in parts]),
                        hands[0], hands[1])


# ----------------------------------------------------------------------------
# Building blocks of the procedural poses (anatomical angles to SMPL-X rotations)
# ----------------------------------------------------------------------------

def _rx(deg):
    return axis_angle_to_matrix_np(np.stack([np.radians(deg), np.zeros_like(deg), np.zeros_like(deg)], -1))


def _ry(deg):
    return axis_angle_to_matrix_np(np.stack([np.zeros_like(deg), np.radians(deg), np.zeros_like(deg)], -1))


def _rz(deg):
    return axis_angle_to_matrix_np(np.stack([np.zeros_like(deg), np.zeros_like(deg), np.radians(deg)], -1))


@dataclass
class Angles:
    """Anatomical angles of a pose over time [deg] (arrays of the same length).
    Legs: hip flexion, knee flexion, ankle dorsiflexion, toe extension, hip
    abduction. Trunk: forward flexion spread over the three spine joints, twist
    (positive turns the chest to the left). Arms: shoulder elevation from the
    side of the body (0 hanging down, 90 horizontal), forward swing, palm
    twist (90: palms forward), elbow flexion."""
    n: int

    def __post_init__(self):
        zero = lambda: np.zeros(self.n)                                    # noqa: E731
        for side in ("left", "right"):
            for name in ("hip", "knee", "ankle", "toe", "abduction", "elevation", "swing", "twist", "elbow"):
                setattr(self, f"{side}_{name}", zero())
        self.trunk_flexion = zero()
        self.trunk_twist = zero()
        self.neck_flexion = zero()
        self.head_twist = zero()


def angles_to_body(a: Angles) -> np.ndarray:
    """(T, 21, 3) SMPL-X body pose from anatomical angles (see skeleton for the signs)."""
    body = np.zeros((a.n, skeleton.NUM_JOINTS, 3))
    J = skeleton.JOINT
    for side, sign in (("left", 1.0), ("right", -1.0)):
        g = lambda name: getattr(a, f"{side}_{name}")                      # noqa: E731
        hip = _rz(sign * g("abduction")) @ _rx(-g("hip"))
        body[:, J[f"{side}_hip"]] = matrix_to_axis_angle_np(hip)
        body[:, J[f"{side}_knee"], 0] = np.radians(g("knee"))
        body[:, J[f"{side}_ankle"], 0] = np.radians(-g("ankle"))
        body[:, J[f"{side}_foot"], 0] = np.radians(-g("toe"))
        # arm: twist about its own axis (palms), lowered to the side (elevation 90 = horizontal),
        # then swung forward about the lateral axis
        lower = _rz(-sign * (90.0 - g("elevation")))
        shoulder = _rx(-g("swing")) @ lower @ _rx(-g("twist"))
        body[:, J[f"{side}_shoulder"]] = matrix_to_axis_angle_np(shoulder)
        body[:, J[f"{side}_elbow"], 1] = np.radians(-sign * g("elbow"))
    for name in ("spine1", "spine2", "spine3"):
        body[:, J[name], 0] = np.radians(a.trunk_flexion / 3.0)
        body[:, J[name], 1] = np.radians(a.trunk_twist / 3.0)
    body[:, J["neck"], 0] = np.radians(a.neck_flexion)
    body[:, J["head"], 1] = np.radians(a.head_twist)
    return body[:, skeleton.BODY]


def a_pose(n: int = 1, elevation: float = 40.0) -> Angles:
    """Standing in the A-pose of the capture protocol: arms 'elevation' deg
    from the body, palms forward, knees almost straight."""
    a = Angles(n)
    for side in ("left", "right"):
        getattr(a, f"{side}_elevation")[:] = elevation
        getattr(a, f"{side}_twist")[:] = 90.0
        getattr(a, f"{side}_elbow")[:] = 8.0
        getattr(a, f"{side}_knee")[:] = 2.0
    return a


def _bump(phase, center, width):
    """Periodic Gaussian bump over the gait cycle (phase in cycles)."""
    d = (phase - center + 0.5) % 1.0 - 0.5
    return np.exp(-0.5 * (d / width) ** 2)


def leg_angles(phase):
    """Sagittal angles [deg] of one leg over a gait cycle (phase 0: heel strike):
    hip flexion, knee flexion, ankle dorsiflexion, toe extension (shapes of
    the normal adult gait curves, e.g. Winter, Biomechanics and Motor Control
    of Human Movement)."""
    hip = 10.0 + 21.0 * np.cos(2 * np.pi * (phase - 0.02))
    knee = 4.0 + 14.0 * _bump(phase, 0.14, 0.06) + 58.0 * _bump(phase, 0.72, 0.10)
    ankle = 9.0 * _bump(phase, 0.42, 0.12) - 17.0 * _bump(phase, 0.63, 0.06) - 5.0 * _bump(phase, 0.06, 0.03)
    toe = 28.0 * _bump(phase, 0.56, 0.05)
    return hip, knee, ankle, toe


def _ramp(t, start, length):
    """0 before start, 1 after start + length, smooth in between."""
    u = np.clip((t - start) / max(length, 1e-9), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def _blend(a: Angles, b: Angles, weight) -> Angles:
    out = Angles(a.n)
    for name, value in vars(a).items():
        if name == "n":
            continue
        setattr(out, name, (1 - weight) * value + weight * getattr(b, name))
    return out


# ----------------------------------------------------------------------------
# Root placement by ground contact
# ----------------------------------------------------------------------------

def foot_vertices(shaped, side: str | None = None) -> np.ndarray:
    """Indices of the vertices of the feet (skinning parts 'left_foot', 'right_foot'), or of one side."""
    labels = skeleton.part_labels(shaped.weights.detach().cpu().numpy())
    sides = ("left", "right") if side is None else (side,)
    parts = [skeleton.PART_NAMES.index(f"{name}_foot") for name in sides]
    return np.flatnonzero(np.isin(labels, parts))


def contact_weights(times: np.ndarray, contacts: np.ndarray, blend: float = 0.08) -> np.ndarray:
    """(T, 2) weight of each foot in placing the body: 0 in the air, rising
    smoothly to 1 within 'blend' seconds after touch-down and falling to 0
    within 'blend' seconds before lift-off. A foot that is down for the whole
    sequence (standing) has weight 1."""
    n = len(times)
    weights = np.zeros((n, 2))
    for side in range(2):
        down = contacts[:, side]
        since = np.full(n, np.inf)
        until = np.full(n, np.inf)
        last = -np.inf
        for k in range(n):
            if down[k] and (k == 0 or not down[k - 1]):
                last = times[k] if k > 0 else -np.inf
            since[k] = times[k] - last if down[k] else 0.0
        nxt = np.inf
        for k in range(n - 1, -1, -1):
            if down[k] and (k == n - 1 or not down[k + 1]):
                nxt = times[k] if k < n - 1 else np.inf
            until[k] = nxt - times[k] if down[k] else 0.0
        u = np.clip(np.minimum(since, until) / blend, 0.0, 1.0)
        weights[:, side] = np.where(down, u * u * (3 - 2 * u), 0.0)
    return weights


def place_on_ground(sequence: PoseSequence, shaped, flights=(), contacts=None, start_xy=(0.0, 0.0),
                    lift_tolerance: float = 0.003, batch: int = 512) -> PoseSequence:
    """Root positions from ground contact (the input root positions are ignored).

    contacts: (T, 2) booleans, left and right foot on the floor (None: both,
    as when standing). Each foot on the floor has an anchor: its lowest
    vertex, fixed on the floor where it touched down (it moves to the new
    lowest vertex when it rises more than lift_tolerance above it, as the foot
    rolls from heel to toe). The pelvis is placed so that the anchors of the
    feet on the floor stay where they are and their lowest vertices touch the
    floor, each foot weighted by contact_weights: with one foot down the
    placement is exact; while both are down, a mismatch of the generic joint
    angles is shared between the feet instead of making the pelvis jump.
    flights: (takeoff, landing, extra horizontal velocity (2,)) intervals of a
    ballistic flight; the vertical take-off velocity is the one that brings the
    body back to its take-off height at 'landing'. Sample finely (about 240 Hz)."""
    sides = [foot_vertices(shaped, "left"), foot_vertices(shaped, "right")]
    relative = sequence.subset(slice(None))
    relative.root_position = np.zeros_like(relative.root_position)
    rel_all, _ = relative.posed(shaped, batch, subset=np.concatenate(sides))
    rel_all = rel_all.astype(np.float64)
    rel = [rel_all[:, :len(sides[0])], rel_all[:, len(sides[0]):]]      # (T, F, 3) relative to the pelvis
    t = sequence.times
    n = len(t)
    contacts = np.ones((n, 2), dtype=bool) if contacts is None else np.asarray(contacts, dtype=bool).copy()
    airborne_mask = np.zeros(n, dtype=bool)
    for takeoff, landing, _ in flights:
        airborne_mask |= (t >= takeoff) & (t < landing)
    contacts[airborne_mask] = False
    weights = contact_weights(t, contacts)
    positions = np.zeros((n, 3))
    position = np.array([start_xy[0], start_xy[1], 0.0])
    anchors = [None, None]                                             # (vertex, anchor xy) per foot
    airborne = None
    for k in range(n):
        if airborne is None:
            for takeoff, landing, extra in flights:
                if 1 < k and t[k - 1] < takeoff <= t[k]:
                    t0 = t[k - 1]
                    vertical = 0.5 * GRAVITY * (landing - t0)
                    airborne = (t0, positions[k - 1].copy(), np.array([extra[0], extra[1], vertical]), landing)
                    break
        if airborne is not None:
            t0, p0, v0, landing = airborne
            dt = t[k] - t0
            position = p0 + v0 * dt + np.array([0.0, 0.0, -0.5 * GRAVITY * dt * dt])
            positions[k] = position
            anchors = [None, None]
            if t[k] >= landing:
                airborne = None
            continue
        down = [side for side in (0, 1) if contacts[k, side]] or [0, 1]
        lowest = {side: int(np.argmin(rel[side][k, :, 2])) for side in (0, 1)}
        for side in (0, 1):
            if side not in down:
                anchors[side] = None
        # height: each foot's lowest vertex on the floor, weighted
        w = np.array([max(weights[k, side], 1e-6) if side in down else 0.0 for side in (0, 1)])
        heights = np.array([-rel[side][k, lowest[side], 2] for side in (0, 1)])
        position[2] = float((w * heights).sum() / w.sum())
        # horizontal: feet with a valid anchor place the pelvis
        estimates, estimate_weights, renew = {}, {}, []
        for side in down:
            anchor = anchors[side]
            if anchor is None or rel[side][k, anchor[0], 2] - rel[side][k, lowest[side], 2] > lift_tolerance:
                renew.append(side)
                if anchor is None:
                    continue
            estimates[side] = anchor[1] - rel[side][k, anchor[0], :2]
            estimate_weights[side] = w[side]
        if estimates:
            total = sum(estimate_weights.values())
            position[:2] = sum(estimates[side] * estimate_weights[side] for side in estimates) / total
        for side in renew:
            # a foot keeps its own estimate of the pelvis when its anchor moves (heel to toe), so that
            # the weighted placement does not jump; a foot that just touched down starts from the pelvis
            base = estimates.get(side, position[:2])
            anchors[side] = (lowest[side], base + rel[side][k, lowest[side], :2])
        positions[k] = position
    result = sequence.subset(slice(None))
    result.root_position = positions
    return result


# ----------------------------------------------------------------------------
# Procedural motions
# ----------------------------------------------------------------------------

def _root_rotation(yaw_deg, roll_deg=0.0) -> np.ndarray:
    """World root rotations: roll about the facing direction, then yaw about z [deg]."""
    yaw = np.asarray(yaw_deg, dtype=np.float64)
    roll = np.broadcast_to(np.asarray(roll_deg, dtype=np.float64), yaw.shape)
    return matrix_to_axis_angle_np(_rz(yaw) @ _rx(roll))


@dataclass
class GaitPlan:
    """Everything of a walk that does not need the body model's joints:
    times, activity (0 standing .. 1 walking), gait phase, path and heading,
    the ankle and ball targets of each foot, its toe extension and its pivot
    (0: on the heel, 1: rolling on the ball)."""
    times: np.ndarray
    activity: np.ndarray
    phase: np.ndarray
    heading: np.ndarray
    path: np.ndarray
    ankle: dict
    ball: dict
    toe_extension: dict
    pivot: dict


def _standing_reference(shaped):
    """Pelvis height when standing in the A-pose, and per foot: the flat-foot
    geometry (ankle and ball joints, back of the heel on the floor; foot frame
    x forward, z up, origin on the floor below the ankle) and the offset (x, y)
    of the ankle from the pelvis (body frame: x forward, y left)."""
    standing = PoseSequence(np.zeros(1), np.zeros((1, 3)), np.zeros((1, 3)), angles_to_body(a_pose(1)))
    _, joints = standing.posed(shaped)
    soles, _ = standing.posed(shaped, subset=foot_vertices(shaped))
    height = -float(soles[0, :, 2].min())
    joints = joints[0].astype(np.float64)
    joints[:, 2] += height
    feet = {}
    for side in ("left", "right"):
        ankle = joints[skeleton.JOINT[f"{side}_ankle"]]
        ball = joints[skeleton.JOINT[f"{side}_foot"]]
        sole, _ = standing.posed(shaped, subset=foot_vertices(shaped, side))
        feet[side] = {"ankle": np.array([0.0, 0.0, ankle[2]]),
                      "ball": np.array([ball[0] - ankle[0], 0.0, ball[2]]),
                      "heel": np.array([float(sole[0, :, 0].min()) - ankle[0], 0.0, 0.0]),
                      "offset": ankle[:2].copy()}
    return height, feet


def _foot_pose(geometry, pitch_deg, pivot_blend):
    """Ankle and ball joints of a foot pitched by pitch_deg (positive: toes up),
    in the foot frame: turned about the back of the heel on the floor
    (pivot_blend 0, heel strike) or about the ball joint (pivot_blend 1: the
    heel rises while the toes stay flat on the floor)."""
    a = np.radians(-pitch_deg)
    rotation = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    result = []
    for name in ("ankle", "ball"):
        about_heel = geometry["heel"] + rotation @ (geometry[name] - geometry["heel"])
        about_ball = geometry["ball"] + rotation @ (geometry[name] - geometry["ball"])
        result.append((1 - pivot_blend) * about_heel + pivot_blend * about_ball)
    return result


def plan_walk(feet, times, stepping, progress, speed, cadence, heading_deg, turn_rate_deg, start_xy, step_width,
              clearance, stance: float = 0.62) -> GaitPlan:
    """Footprints and foot trajectories of a walk (see walk). 'stepping'
    (0..1) sets the rate of the gait phase and the lift of the swinging foot,
    'progress' (0..1) the speed, so the step length and the heel-toe roll: the
    walk starts and ends with steps of nearly zero length, as people do."""
    n = len(times)
    dt = np.diff(times, prepend=times[0] - (times[1] - times[0]))
    cycle_hz = cadence / 120.0
    phase = np.cumsum(stepping * dt) * cycle_hz
    activity = progress
    heading = np.radians(heading_deg + np.cumsum(progress * dt) * turn_rate_deg)
    forward = np.stack([np.cos(heading), np.sin(heading)], axis=1)
    lateral_dir = np.stack([-np.sin(heading), np.cos(heading)], axis=1)
    path = np.asarray(start_xy, dtype=np.float64) + np.cumsum(speed * (progress * dt)[:, None] * forward, axis=0)
    ankles, balls, toes, pivots = {}, {}, {}, {}
    for side, offset, lateral in (("left", 0.0, 1.0), ("right", 0.5, -1.0)):
        leg_phase = phase + offset
        # footprints: (origin xy, heading); a new one at every heel strike, half a step ahead of the pelvis
        origins = [path[0] + lateral * step_width * lateral_dir[0]]
        yaws = [heading[0]]
        starts = [-np.inf]
        for k in np.flatnonzero(np.diff(np.floor(leg_phase)) > 0) + 1:
            ahead = 0.5 * speed * activity[k] / (2.0 * cycle_hz)
            origins.append(path[k] + ahead * forward[k] + lateral * step_width * lateral_dir[k])
            yaws.append(heading[k])
            starts.append(times[k])
        starts = np.array(starts)
        ankle = np.zeros((n, 3))
        ball = np.zeros((n, 3))
        toe = np.zeros(n)
        pivot = np.zeros(n)
        g = feet[side]
        for k in range(n):
            u = leg_phase[k] % 1.0
            current = int(np.searchsorted(starts, times[k], side="right") - 1)
            amplitude = activity[k]
            origin, yaw = origins[current], yaws[current]
            has_next = current + 1 < len(origins)
            if u < stance or not has_next:
                if u < 0.08:
                    pitch, blend = 12.0 * (1.0 - _ease(u / 0.08, "smooth")), 0.0
                elif u < 0.40:
                    pitch, blend = 0.0, 0.0
                elif u < stance:
                    pitch, blend = -35.0 * _ease((u - 0.40) / (stance - 0.40), "in"), 1.0
                else:                                   # last footprint: the foot settles flat
                    pitch, blend = 0.0, 0.0
                pitch *= amplitude
                lift = 0.0
            else:
                fraction = (u - stance) / (1.0 - stance)
                e = _ease(fraction, "smooth")
                origin = (1 - e) * origins[current] + e * origins[current + 1]
                yaw = yaws[current] + e * np.angle(np.exp(1j * (yaws[current + 1] - yaws[current])))
                pitch = amplitude * ((1 - e) * -35.0 + e * 12.0)
                lift = stepping[k] * clearance * np.sin(np.pi * fraction) ** 1.2
                blend = 1.0 - e
            ankle_local, ball_local = _foot_pose(g, pitch, blend)
            c, s_ = np.cos(yaw), np.sin(yaw)
            to_world = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
            base = np.array([origin[0], origin[1], lift])
            ankle[k] = base + to_world @ ankle_local
            ball[k] = base + to_world @ ball_local
            toe[k] = blend * max(0.0, -pitch)
            pivot[k] = blend
        ankles[side], balls[side], toes[side], pivots[side] = ankle, ball, toe, pivot
    return GaitPlan(times, activity, phase, heading, path, ankles, balls, toes, pivots)


def _gait_upper_body(plan: GaitPlan) -> np.ndarray:
    """(T, 21, 3) body pose with the trunk, arms and toes of walking (legs at rest)."""
    n = len(plan.times)
    cycle = 2 * np.pi * plan.phase
    upper = Angles(n)
    for side, offset in (("left", 0.0), ("right", 0.5)):
        arm_phase = (plan.phase + offset + 0.5) % 1.0                    # arm swings with the opposite leg
        getattr(upper, f"{side}_elevation")[:] = 10.0
        getattr(upper, f"{side}_swing")[:] = 3.0 + 16.0 * np.cos(2 * np.pi * arm_phase)
        getattr(upper, f"{side}_elbow")[:] = 16.0 + 7.0 * (1.0 + np.cos(2 * np.pi * arm_phase))
    upper.trunk_flexion[:] = 4.0
    upper.trunk_twist[:] = 6.0 * np.cos(cycle)                           # chest turns against the pelvis
    upper.head_twist[:] = -2.0 * np.cos(cycle)
    pose = _blend(a_pose(n), upper, plan.activity)
    for side in ("left", "right"):
        getattr(pose, f"{side}_toe")[:] = plan.toe_extension[side]
    return angles_to_body(pose)


def solve_legs(shaped, body, root_rotation, pelvis, ankle_targets, ball_targets, initial, iterations=300):
    """Leg inverse kinematics: hip (3), knee and ankle rotations of both legs
    so that the ankle and ball joints reach their targets, with knees bending
    forward only, little hip and knee twist, and smooth joint paths. 'body'
    gives every other joint; 'initial' the starting leg angles (T, 21, 3).
    Returns the body pose (T, 21, 3) and the reached world joints (T, 55, 3)."""
    import torch
    from bodyscan.body.rotations import axis_angle_to_matrix
    model = shaped.model
    n = len(pelvis)
    legs = [skeleton.JOINT[f"{side}_{name}"] - 1 for side in ("left", "right") for name in ("hip", "knee", "ankle")]
    flat = torch.tensor(np.ascontiguousarray(initial[:, legs]).reshape(-1), dtype=model.dtype,
                        device=model.device).requires_grad_(True)                     # L-BFGS needs one flat tensor
    fixed = torch.tensor(body, dtype=model.dtype, device=model.device)
    root_t = model.tensor(axis_angle_to_matrix_np(root_rotation))
    pelvis_t = model.tensor(pelvis)
    sides = ("left", "right")
    ankle_ids = [skeleton.JOINT[f"{side}_ankle"] for side in sides]
    ball_ids = [skeleton.JOINT[f"{side}_foot"] for side in sides]
    ankle_t = torch.stack([model.tensor(ankle_targets[side]) for side in sides], dim=1)
    ball_t = torch.stack([model.tensor(ball_targets[side]) for side in sides], dim=1)
    hands = model.rotations(count=1)[0, 25:]

    def joints_of(free_values):
        pose = fixed.clone()
        pose[:, legs] = free_values
        full = torch.cat([torch.zeros(n, 1, 3, dtype=model.dtype, device=model.device), pose,
                          torch.zeros(n, 3, 3, dtype=model.dtype, device=model.device)], dim=1)
        rotations = torch.cat([axis_angle_to_matrix(full), hands.expand(n, -1, -1, -1)], dim=1)
        return pose, shaped.joints_world(rotations, root_t, pelvis_t)

    optimizer = torch.optim.LBFGS([flat], lr=1.0, max_iter=iterations, history_size=20,
                                  line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        optimizer.zero_grad()
        free = flat.view(n, len(legs), 3)
        _, joints = joints_of(free)
        loss = ((joints[:, ankle_ids] - ankle_t) ** 2).sum() + 0.5 * ((joints[:, ball_ids] - ball_t) ** 2).sum()
        knees = free[:, [1, 4], 0]
        loss = loss + 10.0 * (torch.relu(-knees) ** 2).sum()                  # knees bend forward only
        loss = loss + 1e-4 * (free[:, [0, 3], 1] ** 2).sum()                  # little hip twist
        loss = loss + 1e-4 * (free[:, [1, 4], 1:] ** 2).sum()                 # knee: one axis
        if n > 2:
            loss = loss + 1e-5 * ((free[2:] - 2 * free[1:-1] + free[:-2]) ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        pose, joints = joints_of(flat.view(n, len(legs), 3))
    return pose.cpu().numpy(), joints.cpu().numpy()


def _cover(values: np.ndarray, width: int) -> np.ndarray:
    """A smooth curve that is nowhere below 'values': running maximum over
    2 width + 1 samples, then a raised-cosine weighted mean over the same span
    (every value in the span of a sample is at least that sample's value)."""
    window = 2 * width + 1
    peaks = np.lib.stride_tricks.sliding_window_view(np.pad(values, width, mode="edge"), window).max(axis=1)
    weights = 0.5 * (1.0 + np.cos(np.pi * np.arange(-width, width + 1) / (width + 1)))
    weights /= weights.sum()
    return np.lib.stride_tricks.sliding_window_view(np.pad(peaks, width, mode="edge"), window) @ weights


def _flatten_toes(body, joints, feet, weight, frames=None) -> None:
    """Toe extension from the solved feet (in place on body (T, 21, 3)): a foot
    rolling on its ball (weight 1) keeps its toes flat on the floor, whatever
    pitch the inverse kinematics reached. Only 'frames' (default all) change."""
    rows = slice(None) if frames is None else frames
    for side in ("left", "right"):
        g = feet[side]
        d = joints[:, skeleton.JOINT[f"{side}_foot"]] - joints[:, skeleton.JOINT[f"{side}_ankle"]]
        flat = np.degrees(np.arctan2(g["ball"][2] - g["ankle"][2], g["ball"][0]))
        pitch = np.degrees(np.arctan2(d[:, 2], np.linalg.norm(d[:, :2], axis=1))) - flat
        extension = weight[side] * np.maximum(-pitch, 0.0)
        body[rows, skeleton.JOINT[f"{side}_foot"] - 1, 0] = np.radians(-extension[rows])


def _lift_out_of_floor(sequence: PoseSequence, shaped, tolerance: float = 0.002, half_window: float = 0.08) -> float:
    """Raise the pelvis where a foot went more than 'tolerance' below the floor
    (in place; skinning blends the toes a few millimetres into the floor when
    the heel is up, as soft tissue would compress). The lift is spread over
    +-half_window seconds so that it adds no velocity spike. Returns the
    deepest point [m] before the lift."""
    soles, _ = sequence.posed(shaped, subset=foot_vertices(shaped))
    depth = np.maximum(-soles[:, :, 2].min(axis=1).astype(np.float64) - tolerance, 0.0)
    if depth.max() > 0:
        step = float(np.median(np.diff(sequence.times))) if len(sequence) > 1 else 1.0
        sequence.root_position[:, 2] += _cover(depth, max(int(round(half_window / step)), 1))
    return float(depth.max()) + tolerance if depth.max() > 0 else 0.0


def walk(shaped, duration: float = 8.0, speed: float = 1.15, cadence: float = 105.0, heading_deg: float = 0.0,
         turn_rate_deg: float = 0.0, start_xy=(0.0, 0.0), stand_before: float = 2.0, stand_after: float = 1.0,
         rate: float = 240.0, clearance: float = 0.06, step_width: float = 0.09,
         solve_rate: float = 60.0) -> PoseSequence:
    """Walking, preceded and followed by standing in the A-pose.

    The feet follow planned footprints (a step of speed / cadence, placed
    half a step ahead of the pelvis at heel strike, step_width either side of
    the path): in stance a foot stays on its footprint, rolling from the heel
    (toes up 12 deg at heel strike) to the ball (heel up 35 deg at toe-off);
    in swing it moves to its next footprint with 'clearance' metres of lift.
    Stepping starts and stops within 0.3 s, the speed (so the step length)
    rises and falls over 1.5 s inside that, so the first and last steps are
    short. The pelvis follows the path with a vertical bob and a lateral sway,
    lowered where the legs cannot reach their feet. The leg joints come from
    inverse kinematics on the body model (at solve_rate, then resampled
    smoothly); the trunk and arms from gait profiles (arm swing against the
    legs, chest counter-rotation).

    speed [m/s], cadence [steps/min], heading_deg (0: +x), turn_rate_deg
    [deg/s] (a circle of radius R needs speed / R rad/s)."""
    total = stand_before + duration + stand_after
    times = np.arange(0.0, total + 1e-9, 1.0 / solve_rate)
    end = stand_before + duration
    stepping = _ramp(times, stand_before, 0.3) * (1.0 - _ramp(times, end - 0.3, 0.3))
    ramp = min(1.5, max(duration - 0.9, 0.2) / 2.0)
    progress = _ramp(times, stand_before + 0.15, ramp) * (1.0 - _ramp(times, end - 0.3 - ramp, ramp))
    height, feet = _standing_reference(shaped)
    plan = plan_walk(feet, times, stepping, progress, speed, cadence, heading_deg, turn_rate_deg, start_xy,
                     step_width, clearance)
    body = _gait_upper_body(plan)
    cycle = 2 * np.pi * plan.phase
    root = _root_rotation(np.degrees(plan.heading) - 4.0 * progress * np.cos(cycle), 3.0 * progress * np.sin(cycle))
    lateral = np.stack([-np.sin(plan.heading), np.cos(plan.heading)], axis=1)
    pelvis = np.zeros((len(times), 3))
    pelvis[:, :2] = plan.path + (0.02 * stepping * np.sin(cycle))[:, None] * lateral
    pelvis[:, 2] = height - progress * (0.015 + 0.012 * np.cos(2 * cycle))
    initial = Angles(len(times))
    for side, offset in (("left", 0.0), ("right", 0.5)):
        hip, knee, ankle, _ = leg_angles((plan.phase + offset) % 1.0)
        getattr(initial, f"{side}_hip")[:] = hip * progress
        getattr(initial, f"{side}_knee")[:] = np.maximum(knee * stepping, 8.0)
        getattr(initial, f"{side}_ankle")[:] = ankle * progress
    start = angles_to_body(initial)
    ankle_ids = {side: skeleton.JOINT[f"{side}_ankle"] for side in ("left", "right")}
    lowered = np.zeros(len(times))
    for _ in range(6):
        body, joints = solve_legs(shaped, body, root, pelvis, plan.ankle, plan.ball, start)
        start = body
        # where a leg falls short of its foot, lower the pelvis by the shortfall (at most 8 cm in all)
        shortfall = np.zeros(len(times))
        for side in ("left", "right"):
            miss = plan.ankle[side] - joints[:, ankle_ids[side]]
            shortfall = np.maximum(shortfall, np.maximum(-miss[:, 2], 0.0))
        if shortfall.max() < 0.002:
            break
        step = np.minimum(1.2 * _cover(shortfall, 5), np.maximum(0.08 - lowered, 0.0))
        pelvis[:, 2] -= step
        lowered += step
    _flatten_toes(body, joints, feet, plan.pivot)
    sequence = PoseSequence(times, root, pelvis, body)
    _lift_out_of_floor(sequence, shaped)
    return sequence.sample(np.arange(0.0, total, 1.0 / rate))


def _ease(u, kind):
    u = np.clip(u, 0.0, 1.0)
    if kind == "in":                       # accelerating: fastest at the end (push-off)
        return u ** 1.7
    if kind == "out":                      # decelerating: fastest at the start (landing)
        return 1.0 - (1.0 - u) ** 2
    return u * u * (3 - 2 * u)             # smooth start and end


JUMP_SQUAT, JUMP_PUSH, JUMP_ABSORB, JUMP_RECOVER = 0.5, 0.3, 0.3, 0.6      # phase durations [s]
JUMP_HEEL_RISE = 35.0               # foot pitch on the toes at take-off and at landing [deg]

JUMP_POSES = {   # hip flexion, knee, ankle dorsiflexion, trunk flexion, arm swing, elbow, elevation, twist [deg]
    "stand": (0, 2, 0, 0, 0, 8, 40, 90),             # the A-pose of the protocol
    "squat": (70, 88, 26, 24, -40, 20, 10, 30),      # bottom of the countermovement, arms back
    "takeoff": (-4, 0, -36, 2, 135, 12, 10, 0),      # legs extended, arms up in front
    "landed": (8, 10, -20, 4, 40, 20, 15, 10),       # touchdown, arms coming down
    "absorb": (55, 75, 24, 18, 25, 25, 15, 30),      # bottom of the landing
}
_JUMP_NAMES = ("hip", "knee", "ankle", "trunk", "swing", "elbow", "elevation", "twist")


def _jump_tracks(t, starts, flight):
    """Angle tracks of jumps starting at 'starts': the poses of JUMP_POSES at
    the start, the bottom of the squat, take-off, touchdown, the bottom of the
    landing and the end of the recovery, joined by smooth steps (position and
    velocity continuous: an arm swing starts and ends at rest). The trunk and
    arms are used as they are, the legs only start the inverse kinematics.
    Also returns where a jump is under way."""
    p = {name: np.array(value, dtype=float) for name, value in JUMP_POSES.items()}
    tracks = np.tile(p["stand"], (len(t), 1))
    active = np.zeros(len(t), dtype=bool)
    for start in starts:
        takeoff = start + JUMP_SQUAT + JUMP_PUSH
        landing = takeoff + flight
        keys = [(start, p["stand"]), (start + JUMP_SQUAT, p["squat"]), (takeoff, p["takeoff"]),
                (landing, p["landed"]), (landing + JUMP_ABSORB, p["absorb"]),
                (landing + JUMP_ABSORB + JUMP_RECOVER, p["stand"])]
        for (a, v0), (b, v1) in zip(keys[:-1], keys[1:]):
            inside = (t >= a) & (t < b)
            tracks[inside] = v0 + (v1 - v0) * _ease((t[inside] - a) / (b - a), "smooth")[:, None]
            active |= inside
    return tracks, active


def _hermite(p0, p1, v0, v1, duration, u):
    """Cubic from p0 (velocity v0) to p1 (velocity v1) over 'duration' seconds, at fractions u (0..1)."""
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    u2, u3 = u * u, u * u * u
    return ((2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * duration * v0 + (-2 * u3 + 3 * u2) * p1
            + (u3 - u2) * duration * v1)


@dataclass
class JumpPlan:
    """Pelvis path (T, 3), per foot the ankle and ball targets (T, 3) and the toe
    extension (T,) [deg], the flight fraction of every time (-1 on the floor),
    the start times of the jumps and the flight time."""
    pelvis: np.ndarray
    ankle: dict
    ball: dict
    toe: dict
    airborne: np.ndarray
    starts: list
    flight: float


def plan_jumps(times, feet, standing_height, count, height, forward, heading_deg, start_xy, stand_before, pause,
               targets: bool = True) -> JumpPlan:
    """The planned path of a series of jumps (see jump); 'feet' from _standing_reference."""
    v = np.sqrt(2.0 * GRAVITY * height)                               # take-off speed
    flight = 2.0 * v / GRAVITY
    heading = np.radians(heading_deg)
    to_world = np.array([[np.cos(heading), -np.sin(heading), 0.0], [np.sin(heading), np.cos(heading), 0.0],
                         [0.0, 0.0, 1.0]])
    direction = to_world[:2, 0]
    start = np.asarray(start_xy, dtype=np.float64)
    # on the toes at take-off and at landing the pelvis rises with the ankles (2 cm less: knees not locked)
    rise = min(_foot_pose(feet[side], -JUMP_HEEL_RISE, 1.0)[0][2] - feet[side]["ankle"][2] for side in feet) - 0.02
    # deep enough that the push-off speeds up and the landing slows down without reversing
    depth = max(0.22, JUMP_PUSH * v / 3.0 - rise + 0.02)
    depth_land = max(0.15, JUMP_ABSORB * v / 3.0 - rise + 0.02)
    # forward jumps: horizontal speed in the air, and the pelvis ahead of the feet at take-off
    # (behind them at the landing) by 'lean'
    speed = max(forward - 0.02, 0.0) / (flight + 2.0 * JUMP_PUSH / 3.0) if forward > 0 else 0.0
    lean = 0.5 * (forward - speed * flight)
    one = JUMP_SQUAT + JUMP_PUSH + flight + JUMP_ABSORB + JUMP_RECOVER
    starts = [stand_before + k * (one + pause) for k in range(count)]
    times = np.asarray(times, dtype=np.float64)
    n = len(times)
    z = np.zeros(n)                     # pelvis height above standing
    x = np.zeros(n)                     # pelvis along the heading from the start
    pitch = np.zeros(n)                 # foot pitch [deg] (negative: heel up)
    footprint = np.zeros(n)             # where the feet stand, along the heading
    landing = np.zeros(n)               # in the air: where they will land
    airborne = np.full(n, -1.0)
    takeoff_times = []
    for k, s0 in enumerate(starts):
        base = k * forward
        t_push = s0 + JUMP_SQUAT
        t_off = t_push + JUMP_PUSH
        t_land = t_off + flight
        t_absorb = t_land + JUMP_ABSORB
        t_end = t_absorb + JUMP_RECOVER
        takeoff_times.append((t_off, t_land, base))
        later = times >= s0
        x[later], footprint[later] = base, base
        phase = (times >= s0) & (times < t_push)                      # squat
        z[phase] = -depth * _ease((times[phase] - s0) / JUMP_SQUAT, "smooth")
        phase = (times >= t_push) & (times < t_off)                   # push-off, heel rising at the end
        u = (times[phase] - t_push) / JUMP_PUSH
        z[phase] = _hermite(-depth, rise, 0.0, v, JUMP_PUSH, u)
        x[phase] = base + _hermite(0.0, lean, 0.0, speed, JUMP_PUSH, u)
        pitch[phase] = -JUMP_HEEL_RISE * _ease((times[phase] - t_off + 0.6 * JUMP_PUSH) / (0.6 * JUMP_PUSH), "in")
        phase = (times >= t_off) & (times < t_land)                   # flight
        tau = times[phase] - t_off
        z[phase] = rise + v * tau - 0.5 * GRAVITY * tau ** 2
        x[phase] = base + lean + speed * tau
        pitch[phase] = -JUMP_HEEL_RISE
        airborne[phase] = tau / flight
        landing[phase] = base + forward
        later = times >= t_land
        x[later], footprint[later] = base + forward, base + forward
        phase = (times >= t_land) & (times < t_absorb)                # landing on the toes, heel coming down
        u = (times[phase] - t_land) / JUMP_ABSORB
        z[phase] = _hermite(rise, -depth_land, -v, 0.0, JUMP_ABSORB, u)
        x[phase] = base + forward - lean + _hermite(0.0, lean, speed, 0.0, JUMP_ABSORB, u)
        pitch[phase] = -JUMP_HEEL_RISE * (1.0 - _ease((times[phase] - t_land) / 0.15, "out"))
        phase = (times >= t_absorb) & (times < t_end)                 # back to standing
        z[phase] = -depth_land * (1.0 - _ease((times[phase] - t_absorb) / JUMP_RECOVER, "smooth"))
    pelvis = np.zeros((n, 3))
    pelvis[:, :2] = start + x[:, None] * direction
    pelvis[:, 2] = standing_height + z
    ankles, balls, toes = {}, {}, {}
    for side in ("left", "right"):
        g = feet[side]
        offset = to_world[:2, :2] @ g["offset"]
        toe = np.where(airborne < 0, np.maximum(-pitch, 0.0),
                       JUMP_HEEL_RISE * (1.0 - np.sin(np.pi * np.clip(airborne, 0.0, 1.0))))
        toes[side] = toe
        if not targets:
            continue

        def on_floor(along, pitch_deg):
            origin = np.array([*(start + along * direction + offset), 0.0])
            ankle_local, ball_local = _foot_pose(g, pitch_deg, 1.0)
            return origin + to_world @ ankle_local, origin + to_world @ ball_local

        ankle = np.zeros((n, 3))
        ball = np.zeros((n, 3))
        for k in range(n):
            if airborne[k] < 0:
                ankle[k], ball[k] = on_floor(footprint[k], pitch[k])
                continue
            # in the air: from where the foot left the floor to where it lands, relative to the pelvis
            t_off, t_land, base = next(item for item in takeoff_times if item[0] <= times[k] < item[1])
            leave = np.array([*(start + (base + lean) * direction), standing_height + rise])
            arrive = np.array([*(start + (base + forward - lean) * direction), standing_height + rise])
            a0, b0 = on_floor(base, -JUMP_HEEL_RISE)
            a1, b1 = on_floor(base + forward, -JUMP_HEEL_RISE)
            e = _ease(airborne[k], "smooth")
            tuck = np.array([0.0, 0.0, 0.3 * height * np.sin(np.pi * airborne[k])])
            ankle[k] = pelvis[k] + (1 - e) * (a0 - leave) + e * (a1 - arrive) + tuck
            ball[k] = pelvis[k] + (1 - e) * (b0 - leave) + e * (b1 - arrive) + tuck
        ankles[side], balls[side] = ankle, ball
    return JumpPlan(pelvis, ankles, balls, toes, airborne, starts, flight)


def jump(shaped, count: int = 3, height: float = 0.25, forward: float = 0.0, heading_deg: float = 0.0,
         start_xy=(0.0, 0.0), stand_before: float = 2.0, pause: float = 1.2, stand_after: float = 1.0,
         rate: float = 240.0, solve_rate: float = 120.0) -> PoseSequence:
    """Countermovement jumps between A-pose stands, in place or 'forward' metres each.

    The pelvis follows a planned path: a squat, a push-off of 0.3 s that ends
    on the toes at the take-off speed v = sqrt(2 g height), a ballistic flight
    of 2 v / g, a landing on the toes that stops the fall within 0.3 s, and the
    recovery to standing; position and velocity are continuous throughout.
    'height' is the rise of the pelvis above its take-off height (0.25 m: an
    ordinary jump; 0.4 m: a strong one). On the floor the feet stay on their
    footprints (the heels rise in the push-off and come down after the
    landing) and the legs come from inverse kinematics; in the air the feet
    move from where they left the floor to where they land, tucked up by 0.3
    height. A forward jump leaves with the pelvis ahead of the feet and lands
    with it behind them."""
    standing_height, feet = _standing_reference(shaped)
    flight = 2.0 * np.sqrt(2.0 * GRAVITY * height) / GRAVITY
    one = JUMP_SQUAT + JUMP_PUSH + flight + JUMP_ABSORB + JUMP_RECOVER
    total = stand_before + count * one + max(count - 1, 0) * pause + stand_after
    times = np.arange(0.0, total + 1e-9, 1.0 / solve_rate)
    arguments = (feet, standing_height, count, height, forward, heading_deg, start_xy, stand_before, pause)
    plan = plan_jumps(times, *arguments)
    tracks, _ = _jump_tracks(times, plan.starts, plan.flight)       # starts and ends in the A-pose
    pose = a_pose(len(times))
    for side in ("left", "right"):
        for k, name in enumerate(_JUMP_NAMES):
            if name != "trunk":
                getattr(pose, f"{side}_{name}")[:] = tracks[:, k]
        getattr(pose, f"{side}_toe")[:] = plan.toe[side]
    pose.trunk_flexion[:] = tracks[:, _JUMP_NAMES.index("trunk")]
    body = angles_to_body(pose)
    root = _root_rotation(np.full(len(times), heading_deg))
    body, joints = solve_legs(shaped, body, root, plan.pelvis, plan.ankle, plan.ball, body)
    _flatten_toes(body, joints, feet, {"left": np.ones(len(times)), "right": np.ones(len(times))},
                  frames=plan.airborne < 0)
    output = np.arange(0.0, total, 1.0 / rate)
    sequence = PoseSequence(times, root, plan.pelvis, body).sample(output)
    sequence.root_position = plan_jumps(output, *arguments, targets=False).pelvis     # the exact path
    _lift_out_of_floor(sequence, shaped)
    return sequence


def standing(shaped, duration: float = 3.0, xy=(0.0, 0.0), heading_deg: float = 0.0, rate: float = 60.0,
             sway: float = 0.01, seed: int = 0) -> PoseSequence:
    """Standing in the A-pose with a little postural sway (the start of every take)."""
    t = np.arange(0.0, duration, 1.0 / rate)
    rng = np.random.default_rng(seed)
    pose = a_pose(len(t))
    drift = np.cumsum(rng.normal(0.0, 1.0, (len(t), 2)), axis=0)
    drift = drift - drift.mean(axis=0)
    drift *= sway / max(np.abs(drift).max(), 1e-9)
    sequence = PoseSequence(t, _root_rotation(np.full(len(t), heading_deg)), np.zeros((len(t), 3)),
                            angles_to_body(pose))
    placed = place_on_ground(sequence, shaped, start_xy=xy)
    placed.root_position[:, :2] += drift
    return placed


# ----------------------------------------------------------------------------
# AMASS
# ----------------------------------------------------------------------------

def load_amass(path, shaped=None, ground: bool = True) -> PoseSequence:
    """An AMASS sequence (SMPL-X 'poses' (T, 165) or SMPL+H (T, 156), 'trans',
    'mocap_frame_rate') as a PoseSequence. AMASS is z up and its root rotation
    includes the change from the model's y-up frame, so the world root
    rotation is R(poses[:, :3]) . WORLD_FROM_MODEL^T; the pelvis is at the
    rest pelvis of the body plus 'trans'. With 'shaped' (the body that will
    play the motion, e.g. an avatar) its rest pelvis is used, and with
    ground=True the sequence is lifted or lowered so that the soles touch
    the floor (the body may be taller or shorter than the AMASS subject)."""
    data = np.load(path, allow_pickle=True)
    poses = np.asarray(data["poses"], dtype=np.float64)
    trans = np.asarray(data["trans"], dtype=np.float64)
    rate = float(data["mocap_frame_rate"] if "mocap_frame_rate" in data.files else data["mocap_framerate"])
    count = len(poses)
    times = np.arange(count) / rate
    root = axis_angle_to_matrix_np(poses[:, :3]) @ skeleton.WORLD_FROM_MODEL.T
    body = poses[:, 3:66].reshape(count, 21, 3)
    if poses.shape[1] >= 165:
        left, right = poses[:, 75:120], poses[:, 120:165]
    elif poses.shape[1] >= 156:
        left, right = poses[:, 66:111], poses[:, 111:156]
    else:
        left = right = None
    pelvis = np.zeros(3) if shaped is None else shaped.rest_joints[0].detach().cpu().numpy().astype(np.float64)
    sequence = PoseSequence(times, matrix_to_axis_angle_np(root), trans + pelvis, body,
                            None if left is None else left.reshape(count, 15, 3),
                            None if right is None else right.reshape(count, 15, 3))
    if ground and shaped is not None:
        soles, _ = sequence.posed(shaped, subset=foot_vertices(shaped))
        lowest = soles[:, :, 2].min(axis=1)
        sequence.root_position[:, 2] -= np.percentile(lowest, 5)
    return sequence


def load_motion(spec: str, shaped, **options) -> PoseSequence:
    """A motion by name ('walk', 'walk-circle', 'jump', 'jump-forward', 'stand')
    or an AMASS / PoseSequence .npz file."""
    path = Path(spec)
    if path.suffix == ".npz" and path.exists():
        data = np.load(path, allow_pickle=True)
        if "poses" in data.files:
            return load_amass(path, shaped)
        return PoseSequence.load(path)
    if spec == "walk":
        return walk(shaped, **options)
    if spec == "walk-circle":
        radius = options.pop("radius", 2.5)
        options.setdefault("turn_rate_deg", np.degrees(options.get("speed", 1.15) / radius))
        return walk(shaped, **options)
    if spec == "jump":
        return jump(shaped, **options)
    if spec == "jump-forward":
        options.setdefault("forward", 0.5)
        return jump(shaped, **options)
    if spec == "stand":
        return standing(shaped, **options)
    raise SystemExit(f"unknown motion {spec!r}: walk, walk-circle, jump, jump-forward, stand, or an .npz file")
