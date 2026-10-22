"""Kinematic and plausibility measures of a tracked motion (numpy).

These need no measurement: they tell whether the motion could be a person's
and how much of it is jitter, the part of the velocity that becomes false
micro-Doppler in the radio simulation.

    joint speed and acceleration   per main joint (pelvis to wrists)
    high-frequency share           share of the velocity power above cutoff_hz,
                                   after resampling to a uniform rate (a jitter
                                   measure: a walking body has little power
                                   above about 6 Hz except at the feet)
    turns                          frames whose root heading turns faster than
                                   max_turn_rate (720 deg/s by default, the
                                   threshold of the 1.5.0 and 1.6.0 reviews)
    hinges                         share of frames with an elbow or knee out of
                                   its plane of flexion, beyond the ranges of
                                   bodyscan.body.anatomy (plus a margin)
    joint speeds                   share of frames with a joint turning faster
                                   than bodyscan.body.anatomy.SPEED_LIMITS
"""

from __future__ import annotations

import numpy as np

from bodyscan.body import skeleton


def _uniform(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, float]:
    """values (T, ...) resampled at the median frame spacing; returns (values, rate)."""
    dt = float(np.median(np.diff(times)))
    grid = np.arange(times[0], times[-1] + 0.5 * dt, dt)
    flat = values.reshape(len(times), -1)
    resampled = np.stack([np.interp(grid, times, flat[:, k]) for k in range(flat.shape[1])], axis=1)
    return resampled.reshape((len(grid),) + values.shape[1:]), 1.0 / dt


def high_frequency_share(times: np.ndarray, positions: np.ndarray, cutoff_hz: float = 6.0) -> np.ndarray:
    """(J,) share of the velocity power above cutoff_hz for positions (T, J, 3)."""
    if len(times) < 8:
        return np.full(positions.shape[1], np.nan)
    uniform, rate = _uniform(times, positions)
    velocity = np.diff(uniform, axis=0) * rate
    velocity = velocity - velocity.mean(axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(velocity, axis=0)) ** 2                    # (F, J, 3)
    frequency = np.fft.rfftfreq(len(velocity), 1.0 / rate)
    power = spectrum.sum(axis=2)                                               # (F, J)
    total = power.sum(axis=0)
    high = power[frequency > cutoff_hz].sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, high / total, np.nan)


def kinematic_metrics(times, joints, root_rotation=None, body=None, cutoff_hz: float = 6.0,
                      max_turn_rate: float = 720.0, hinge_margin_deg: float = 0.5) -> dict:
    """Measures of a motion: times (T,), joints (T, 55 or 22, 3) [m], root_rotation (T, 3) and body
    (T, 21, 3) axis-angle (optional; without them no turn, hinge or joint-speed measures)."""
    times = np.asarray(times, dtype=np.float64)
    joints = np.asarray(joints, dtype=np.float64)[:, :len(skeleton.MAIN_JOINTS)]
    names = [skeleton.JOINT_NAMES[j] for j in skeleton.MAIN_JOINTS]
    result: dict = {"frames": int(len(times))}
    if len(times) < 3:
        return result
    velocity = np.gradient(joints, times, axis=0)
    acceleration = np.gradient(velocity, times, axis=0)
    speed = np.linalg.norm(velocity, axis=2)
    accel = np.linalg.norm(acceleration, axis=2)
    share = high_frequency_share(times, joints, cutoff_hz)
    result["joints"] = {name: {"speed_p95_m_s": float(np.percentile(speed[:, k], 95)),
                               "acceleration_rms_m_s2": float(np.sqrt(np.mean(accel[:, k] ** 2))),
                               "high_frequency_share": float(share[k])} for k, name in enumerate(names)}
    result["high_frequency_share_mean"] = float(np.nanmean(share)) if np.isfinite(share).any() else float("nan")
    result["cutoff_hz"] = float(cutoff_hz)
    if root_rotation is not None:
        from bodyscan.body.rotations import axis_angle_to_matrix_np, yaw_of_matrix_np
        heading = np.unwrap(yaw_of_matrix_np(axis_angle_to_matrix_np(np.asarray(root_rotation, dtype=np.float64))))
        rate = np.degrees(np.abs(np.diff(heading)) / np.maximum(np.diff(times), 1e-9))
        result["turns"] = {"frames_over_limit": int(np.sum(rate > max_turn_rate)), "limit_deg_s": max_turn_rate,
                           "max_deg_s": float(rate.max()), "p99_deg_s": float(np.percentile(rate, 99))}
    if body is not None:
        from bodyscan.body import anatomy
        from bodyscan.body.rotations import axis_angle_to_quaternion_np
        body = np.asarray(body, dtype=np.float64)
        _, off = anatomy.hinge_angles(body)
        low = np.radians([h.off_range[0] - hinge_margin_deg for h in anatomy.HINGES])
        high = np.radians([h.off_range[1] + hinge_margin_deg for h in anatomy.HINGES])
        outside = (off < low) | (off > high)
        result["hinges_out_of_plane"] = {h.joint: float(outside[:, k].mean()) for k, h in enumerate(anatomy.HINGES)}
        steps = np.maximum(np.diff(times), 1e-9)
        quaternions = axis_angle_to_quaternion_np(body)
        over = {}
        for name, limit in anatomy.SPEED_LIMITS.items():
            q = quaternions[:, skeleton.JOINT[name] - 1]
            dots = np.abs(np.sum(q[1:] * q[:-1], axis=-1))
            angular = np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0))) / steps
            over[name] = float(np.mean(angular > limit * (1.0 + 1e-6)))
        result["joint_speed_over_limit"] = over
        result["joint_speed_over_limit_any"] = float(np.max(list(over.values())))
    return result


def flag_shares(motion_data) -> dict:
    """Share of frames with each tracker flag (motion.npz 'flags' and 'flag_names'), and the arm depth
    inside the trunk over 2 cm when recorded. motion_data: the npz or a dict of its arrays."""
    shares = {}
    if "flags" in motion_data and "flag_names" in motion_data:
        flags = np.asarray(motion_data["flags"], dtype=np.int64)
        for bit, name in enumerate(str(n) for n in motion_data["flag_names"]):
            shares[name] = float(np.mean((flags >> bit) & 1)) if len(flags) else 0.0
    if "arm_in_trunk" in motion_data:
        depth = np.asarray(motion_data["arm_in_trunk"], dtype=np.float64)
        if depth.size:
            shares["arm_more_than_2cm_in_trunk"] = float(np.mean(depth > 0.02))
    return shares
