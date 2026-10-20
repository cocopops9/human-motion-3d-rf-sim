"""What a human body can do beyond the joint boxes: elbows and knees bend in one plane (hinges), and no
joint turns faster than a person's joint does (speeds).

Hinges. The limits of bodyscan.body.skeleton are boxes on the axis-angle components
of each joint. For the elbows and knees a box is too loose: a rotation about
the axis across the hinge (z for an elbow, z for a knee) bends the limb out of
its plane of flexion, which no elbow or knee can do beyond a few degrees (the
carrying angle of the elbow, the play of the knee). The trackers of versions
1.3 to 1.5 used that freedom: an arm hanging with the palm forward was
represented by the upper arm turned outwards by 75 to 90 deg and the elbow
bent sideways by 20 deg (the edge of the box), instead of the upper arm turned
less and the elbow bent in its plane. On the s02 takes the elbows of both
arms were bent out of their plane by more than 5 deg in 30 to 100 % of the
frames and the knees by more than 8 deg in up to half the frames of the
side-on jumps; on synthetic takes, whose true elbows and knees never leave
their plane, in 20 to 70 % of the frames.

Here the bone of a hinge joint is described by its direction d = R b in the
frame of its parent (b: the bone in the rest pose):

    flexion    the angle of d from b in the plane of flexion (b, f): the plane
               the bone moves in when the joint bends (f: the direction it
               moves towards first)
    off        the angle of d out of that plane, towards n

and both are kept within anatomical ranges: a penalty for the optimisation
(torch, hinge_penalty) and a projection for poses built outside it
(numpy, project_hinges: the bone is moved back into its range, its twist kept).

Speeds. A tracked or guessed joint can jump between two frames (a fit that found
another pose for one frame, a guess switching between partners); limit_speeds
spreads every change faster than SPEED_LIMITS over the frames around it.

SMPL-X model space: y up, z forward, x towards the left of the body; in the
T-pose of the model the arms lie along +x and -x with the palms down, the
legs along -y (skeleton). An elbow bends the forearm forward (+z), a knee
the shank backwards (-z). The elbow's off-plane direction n is +y for both
arms: towards the side away from the body when the arm hangs (valgus, the
carrying angle, positive).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bodyscan.body import skeleton


@dataclass(frozen=True)
class Hinge:
    joint: str
    bone: tuple            # rest direction of the bone in the parent frame
    flexes: tuple          # direction the bone moves towards when the joint bends
    off: tuple             # direction out of the plane of flexion (positive off angle)
    flexion: tuple         # (min, max) [deg]
    off_range: tuple       # (min, max) [deg]


HINGES = (
    # elbows: hyperextension at most 5 deg; out of plane: varus 5 deg, valgus (carrying angle) 12 deg
    Hinge("left_elbow", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-5.0, 150.0), (-5.0, 12.0)),
    Hinge("right_elbow", (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-5.0, 150.0), (-5.0, 12.0)),
    # knees: hyperextension 8 deg; out of plane (varus and valgus) 6 deg
    Hinge("left_knee", (0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (-8.0, 155.0), (-6.0, 6.0)),
    Hinge("right_knee", (0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (-8.0, 155.0), (-6.0, 6.0)),
)

# indices into the 21 body joints
HINGE_INDEX = np.array([skeleton.JOINT[h.joint] - 1 for h in HINGES])


def _arrays(dtype=np.float64):
    bone = np.array([h.bone for h in HINGES], dtype=dtype)
    flexes = np.array([h.flexes for h in HINGES], dtype=dtype)
    off = np.array([h.off for h in HINGES], dtype=dtype)
    flexion = np.radians(np.array([h.flexion for h in HINGES], dtype=dtype))
    off_range = np.radians(np.array([h.off_range for h in HINGES], dtype=dtype))
    return bone, flexes, off, flexion, off_range


# ----------------------------------------------------------------------------
# numpy
# ----------------------------------------------------------------------------

def hinge_angles(body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(flexion, off) [rad] of the hinge joints, each (..., 4) in HINGES order, of body poses
    (..., 21, 3) axis-angle."""
    from bodyscan.body.rotations import axis_angle_to_matrix_np
    bone, flexes, off, _, _ = _arrays()
    R = axis_angle_to_matrix_np(np.asarray(body, dtype=np.float64)[..., HINGE_INDEX, :])
    d = np.einsum("...ij,...j->...i", R, np.broadcast_to(bone, R.shape[:-1]))
    flexion = np.arctan2(np.sum(d * flexes, axis=-1), np.sum(d * bone, axis=-1))
    out = np.arcsin(np.clip(np.sum(d * off, axis=-1), -1.0, 1.0))
    return flexion, out


def hinge_excess(body: np.ndarray) -> np.ndarray:
    """(...,) the largest excess [deg] of any hinge joint beyond its flexion or off-plane range."""
    _, _, _, flexion_range, off_range = _arrays()
    flexion, off = hinge_angles(body)
    excess = np.maximum.reduce([flexion_range[:, 0] - flexion, flexion - flexion_range[:, 1],
                                off_range[:, 0] - off, off - off_range[:, 1], np.zeros_like(off)])
    return np.degrees(excess.max(axis=-1))


def _twist_about(R: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Twist angle [rad] of rotations R (..., 3, 3) about the unit 'axis' (swing-twist)."""
    from bodyscan.body.rotations import matrix_to_quaternion_np
    q = matrix_to_quaternion_np(R)
    return 2.0 * np.arctan2(np.sum(q[..., 1:] * axis, axis=-1), q[..., 0])


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotations (..., 3, 3) taking unit vectors a to unit vectors b (..., 3)."""
    from bodyscan.body.rotations import axis_angle_to_matrix_np
    cross = np.cross(a, b)
    sine = np.linalg.norm(cross, axis=-1, keepdims=True)
    cosine = np.sum(a * b, axis=-1, keepdims=True)
    angle = np.arctan2(sine, cosine)
    axis = np.where(sine > 1e-9, cross / np.maximum(sine, 1e-12), 0.0)
    return axis_angle_to_matrix_np(axis * angle)


def project_hinges(body: np.ndarray) -> np.ndarray:
    """Body poses (..., 21, 3) with every hinge joint moved into its ranges: the bone's direction
    is clamped in flexion and out of plane, its twist about its own axis kept (R = swing . twist).
    Joints within their ranges are returned unchanged."""
    from bodyscan.body.rotations import axis_angle_to_matrix_np, matrix_to_axis_angle_np
    body = np.array(body, dtype=np.float64, copy=True)
    bone, flexes, off, flexion_range, off_range = _arrays()
    flexion, out = hinge_angles(body)
    clamped_flexion = np.clip(flexion, flexion_range[:, 0], flexion_range[:, 1])
    clamped_out = np.clip(out, off_range[:, 0], off_range[:, 1])
    change = (np.abs(clamped_flexion - flexion) > 1e-9) | (np.abs(clamped_out - out) > 1e-9)
    if not np.any(change):
        return body
    R = axis_angle_to_matrix_np(body[..., HINGE_INDEX, :])
    shape = R.shape[:-2]
    b = np.broadcast_to(bone, shape + (3,))
    twist = _twist_about(R, bone)
    target = (np.cos(clamped_out)[..., None] * (np.cos(clamped_flexion)[..., None] * b
                                                 + np.sin(clamped_flexion)[..., None] * flexes)
              + np.sin(clamped_out)[..., None] * off)
    twist_rotation = axis_angle_to_matrix_np(b * twist[..., None])
    rebuilt = _rotation_between(b, target) @ twist_rotation
    new = matrix_to_axis_angle_np(rebuilt)
    body[..., HINGE_INDEX, :] = np.where(change[..., None], new, body[..., HINGE_INDEX, :])
    return body


# ----------------------------------------------------------------------------
# torch
# ----------------------------------------------------------------------------

def hinge_penalty(body_pose):
    """Squared excess [rad^2] of the hinge joints beyond their flexion and off-plane ranges, summed
    (body_pose (..., 21, 3) torch axis-angle). Differentiable everywhere the joint is not bent by
    exactly 180 deg."""
    import torch
    from bodyscan.body.rotations import axis_angle_to_matrix
    bone, flexes, off, flexion_range, off_range = [torch.as_tensor(a, dtype=body_pose.dtype, device=body_pose.device)
                                                   for a in _arrays()]
    index = torch.as_tensor(HINGE_INDEX, device=body_pose.device)
    R = axis_angle_to_matrix(body_pose.index_select(-2, index))                     # (..., 4, 3, 3)
    d = (R * bone[:, None, :]).sum(-1)                                              # R @ bone
    flexion = torch.atan2((d * flexes).sum(-1), (d * bone).sum(-1))
    out = torch.asin((d * off).sum(-1).clamp(-0.999999, 0.999999))
    return (torch.relu(flexion_range[:, 0] - flexion) ** 2 + torch.relu(flexion - flexion_range[:, 1]) ** 2
            + torch.relu(off_range[:, 0] - out) ** 2 + torch.relu(out - off_range[:, 1]) ** 2).sum()


# ----------------------------------------------------------------------------
# joint speeds
# ----------------------------------------------------------------------------

# The fastest rotation [deg/s] of each body joint relative to its parent that the tracker lets a body
# make from one frame to the next: about 1.5 to 2 times what these joints reach in walking, running and
# jumping (the true shoulders of the synthetic jumping jacks turn at up to 850 deg/s, hips and knees of
# the synthetic walks and jogs at up to 410 deg/s), well below throwing or kicking (elbows and knees
# 1500 to 2500 deg/s). The tracked s02 takes of version 1.5 exceeded them in 0 to 3.6 % of the frames,
# single frames where a joint jumped by 30 to 110 deg and back (the far collar of the side-on jumps, the
# arms after a data gap).
SPEED_LIMITS = {
    "left_hip": 1000.0, "right_hip": 1000.0, "spine1": 400.0, "left_knee": 1500.0, "right_knee": 1500.0,
    "spine2": 400.0, "left_ankle": 1200.0, "right_ankle": 1200.0, "spine3": 400.0, "left_foot": 1500.0,
    "right_foot": 1500.0, "neck": 500.0, "left_collar": 600.0, "right_collar": 600.0, "head": 500.0,
    "left_shoulder": 1200.0, "right_shoulder": 1200.0, "left_elbow": 1500.0, "right_elbow": 1500.0,
    "left_wrist": 1500.0, "right_wrist": 1500.0,
}


def _step_angles(q: np.ndarray) -> np.ndarray:
    """(N - 1,) rotation angle [rad] between consecutive unit quaternions (N, 4)."""
    dots = np.abs(np.sum(q[1:] * q[:-1], axis=-1))
    return 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))


def limit_speeds(times, body, scale: float = 1.0, longest: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Body poses (F, 21, 3) in which no joint turns faster than its SPEED_LIMITS (times 'scale')
    between two frames, and (F, 21) the joint-frames changed. A faster change of a joint is spread
    over the frames around it: the worst step first, the span widened one frame at a time towards the
    faster neighbouring step (on a tie, so that it stays centred on the step) until the joint can turn
    from the frame before the span to the frame after it within the limit (or the span reaches
    'longest' seconds), and the rotations inside the span interpolated spherically in time between
    those two frames. A single frame where the joint jumped
    and came back is so replaced by the mean of its neighbours; a turn made in one frame is spread over
    as many as a joint needs. Joints within their limits are returned unchanged."""
    from bodyscan.body.rotations import axis_angle_to_quaternion_np, quaternion_to_axis_angle_np, slerp_np
    times = np.asarray(times, dtype=np.float64)
    body = np.array(body, dtype=np.float64, copy=True)
    frames = len(times)
    changed = np.zeros(body.shape[:2], dtype=bool)
    if frames < 3 or scale <= 0:
        return body, changed
    steps = np.maximum(np.diff(times), 1e-6)
    quaternions = axis_angle_to_quaternion_np(body)                                 # (F, 21, 4)
    for name, limit in SPEED_LIMITS.items():
        j = skeleton.JOINT[name] - 1
        cap = np.radians(limit * scale) * (1.0 + 1e-6)
        q = quaternions[:, j].copy()
        speed = _step_angles(q) / steps
        unresolved = np.zeros(frames - 1, dtype=bool)
        while True:
            over = np.flatnonzero((speed > cap) & ~unresolved)
            if len(over) == 0:
                break
            first = over[np.argmax(speed[over])]
            a, b = first, first + 1
            while True:
                angle = _step_angles(q[[a, b]])[0]
                span = times[b] - times[a]
                if angle <= cap * span or span >= longest or (a == 0 and b == frames - 1):
                    break
                before = speed[a - 1] if a > 0 else -np.inf
                after = speed[b] if b < frames - 1 else -np.inf
                if before > after or (before == after and first - a <= b - first - 1):
                    a -= 1                          # towards the faster side; on a tie, centred on the step
                else:
                    b += 1
            if b - a >= 2:
                inside = slice(a + 1, b)
                fraction = (times[inside] - times[a]) / (times[b] - times[a])
                q[inside] = slerp_np(np.broadcast_to(q[a], (b - a - 1, 4)), np.broadcast_to(q[b], (b - a - 1, 4)),
                                     fraction)
                changed[inside, j] = True
                speed[a:b] = _step_angles(q[a:b + 1]) / steps[a:b]
            unresolved[a:b] |= speed[a:b] > cap
        quaternions[:, j] = q
    body[changed] = quaternion_to_axis_angle_np(quaternions[changed])
    return body, changed
