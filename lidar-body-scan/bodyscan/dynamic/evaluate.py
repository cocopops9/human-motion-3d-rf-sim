"""Accuracy of a tracked motion against the truth of a synthetic recording.

Metrics (all in the floor frame of the processing, at the tracked frame times):

    MPJPE            mean distance of the 22 main joints (pelvis to wrists)
    root             pelvis position error, facing (yaw) error
    vertex error     mean distance of corresponding vertices (same body topology)
    velocity         per body part, error of the mean velocity of its vertices
                     (what drives the Doppler shift); also as Doppler at a
                     reference frequency, 2 dv / lambda
    acceleration     error of the joint accelerations (jitter and over-smoothing)
    seen / hidden    joint error split by whether the joint's body part was
                     observed by the sensor (observed fraction from the tracker)
    foot skating     horizontal speed of the estimated feet while the true
                     feet stand on the floor
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bodyscan.body import skeleton

C = 299_792_458.0


def _accelerations(points: np.ndarray, times: np.ndarray) -> np.ndarray:
    """(T, ..., 3) -> (T - 2, ...) acceleration magnitudes with non-uniform times."""
    dt = np.maximum(np.diff(times), 1e-6)
    velocity = np.diff(points, axis=0) / dt.reshape((-1,) + (1,) * (points.ndim - 1))
    span = np.maximum(times[2:] - times[:-2], 1e-6).reshape((-1,) + (1,) * (points.ndim - 1))
    return np.linalg.norm(2.0 * np.diff(velocity, axis=0) / span, axis=-1)


def evaluate(motion_path, run_dir, model, estimate_shaped, truth_shaped, frequency_ghz: float = 60.0) -> dict:
    """Compare a motion.npz with truth.npz of the synthetic run directory."""
    from bodyscan.dynamic.motions import PoseSequence
    from bodyscan.dynamic.simulate import load_truth, to_sensor_floor_frame
    data = np.load(motion_path, allow_pickle=False)
    estimate = PoseSequence.load(motion_path)
    times = estimate.times
    truth, meta = load_truth(run_dir)
    inside = (times >= truth.times[0]) & (times <= truth.times[-1])
    estimate = estimate.subset(inside)
    times = estimate.times
    sampled = truth.sample(times)
    pose = meta["sensor_pose"]
    true_vertices, true_joints = sampled.posed(truth_shaped)
    true_vertices = to_sensor_floor_frame(true_vertices.reshape(-1, 3), pose).reshape(true_vertices.shape)
    true_joints = to_sensor_floor_frame(true_joints.reshape(-1, 3), pose).reshape(true_joints.shape)
    est_vertices, est_joints = estimate.posed(estimate_shaped)
    main = list(skeleton.MAIN_JOINTS)
    joint_error = np.linalg.norm(est_joints[:, main] - true_joints[:, main], axis=2)            # (T, 22)
    per_frame = joint_error.mean(axis=1)
    result = {
        "frames": int(len(times)),
        "mpjpe_mm": {"mean": float(1000 * per_frame.mean()), "median": float(1000 * np.median(per_frame)),
                     "p90": float(1000 * np.percentile(per_frame, 90)), "max": float(1000 * per_frame.max())},
        "per_joint_median_mm": {skeleton.JOINT_NAMES[j]: float(1000 * np.median(joint_error[:, k]))
                                for k, j in enumerate(main)},
        "root_mm": float(1000 * np.median(np.linalg.norm(est_joints[:, 0] - true_joints[:, 0], axis=1))),
    }
    true_yaw = np.arctan2(*np.flip(_facing(sampled, pose), axis=1).T)
    est_yaw = np.arctan2(*np.flip(_facing(estimate, None), axis=1).T)
    result["yaw_deg"] = float(np.degrees(np.median(np.abs(np.angle(np.exp(1j * (est_yaw - true_yaw)))))))
    same_topology = est_vertices.shape == true_vertices.shape
    if same_topology:
        result["vertex_error_mm"] = float(1000 * np.linalg.norm(est_vertices - true_vertices, axis=2).mean())
    # velocities per body part (mean of the part's vertices), the input of a Doppler simulation
    if len(times) > 2:
        labels_est = skeleton.part_labels(estimate_shaped.weights.detach().cpu().numpy())
        labels_true = skeleton.part_labels(truth_shaped.weights.detach().cpu().numpy())
        dt = np.diff(times)[:, None, None]
        part_error = {}
        wavelength = C / (frequency_ghz * 1e9)
        for k, name in enumerate(skeleton.PART_NAMES):
            mine, theirs = labels_est == k, labels_true == k
            if not mine.any() or not theirs.any():
                continue
            v_est = np.diff(est_vertices[:, mine].mean(axis=1), axis=0) / dt[:, 0]
            v_true = np.diff(true_vertices[:, theirs].mean(axis=1), axis=0) / dt[:, 0]
            rms = float(np.sqrt(np.mean(np.sum((v_est - v_true) ** 2, axis=1))))
            part_error[name] = {"rms_m_s": rms, "doppler_hz": 2 * rms / wavelength,
                                "true_speed_m_s": float(np.sqrt(np.mean(np.sum(v_true ** 2, axis=1))))}
        result["velocity_error"] = part_error
        result["velocity_error_rms_m_s"] = float(np.sqrt(np.mean([v["rms_m_s"] ** 2 for v in part_error.values()])))
        result["reference_frequency_ghz"] = frequency_ghz
        a_true = _accelerations(true_joints[:, main], times)
        a_est = _accelerations(est_joints[:, main], times)
        result["acceleration_error_m_s2"] = float(np.mean(np.abs(a_est - a_true)))
        result["acceleration_true_m_s2"] = float(np.mean(a_true))
        result["acceleration_estimate_m_s2"] = float(np.mean(a_est))
    # error of joints seen versus hidden
    if "observed" in data.files:
        observed = data["observed"][inside]
        part_of = np.array([skeleton.PART_NAMES.index(skeleton.PART_OF_JOINT[j]) for j in main])
        seen = observed[:, part_of] >= 0.3
        hidden = observed[:, part_of] < 0.05
        result["seen_joint_mm"] = float(1000 * joint_error[seen].mean()) if seen.any() else None
        result["hidden_joint_mm"] = float(1000 * joint_error[hidden].mean()) if hidden.any() else None
        result["hidden_fraction"] = float(hidden.mean())
    # foot skating while the true foot stands on the floor
    skating = []
    for name in ("left", "right"):
        ankle = skeleton.JOINT[f"{name}_ankle"]
        true_speed = np.linalg.norm(np.diff(true_joints[:, ankle, :2], axis=0), axis=1) / np.diff(times)
        true_low = true_joints[1:, ankle, 2] < 0.15
        standing = (true_speed < 0.1) & true_low
        est_speed = np.linalg.norm(np.diff(est_joints[:, ankle, :2], axis=0), axis=1) / np.diff(times)
        if standing.any():
            skating.append(est_speed[standing])
    if skating:
        speeds = np.concatenate(skating)
        result["foot_skating_m_s"] = {"median": float(np.median(speeds)), "p90": float(np.percentile(speeds, 90))}
    return result


def _facing(sequence, sensor_pose) -> np.ndarray:
    """Facing direction (unit xy) of every frame: the body's forward axis (+x of the upright frame)."""
    matrices = sequence.root_matrices()
    forward = matrices[:, :2, 0]
    if sensor_pose is not None:
        yaw = np.radians(sensor_pose["yaw_deg"])
        rotation = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        forward = forward @ rotation.T
    return forward / np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), 1e-9)


def report_text(result: dict) -> str:
    m = result["mpjpe_mm"]
    lines = [f"{result['frames']} frames: joint error (MPJPE, 22 joints) mean {m['mean']:.1f} mm, median "
             f"{m['median']:.1f} mm, p90 {m['p90']:.1f} mm, worst frame {m['max']:.1f} mm",
             f"  pelvis {result['root_mm']:.1f} mm, facing {result['yaw_deg']:.1f} deg"]
    if "vertex_error_mm" in result:
        lines.append(f"  vertices {result['vertex_error_mm']:.1f} mm (mean)")
    if "velocity_error_rms_m_s" in result:
        f = result["reference_frequency_ghz"]
        worst = sorted(result["velocity_error"].items(), key=lambda item: -item[1]["rms_m_s"])[:3]
        lines.append(f"  body-part velocity error {result['velocity_error_rms_m_s']:.3f} m/s rms (Doppler at {f:g} GHz: "
                     f"{2 * result['velocity_error_rms_m_s'] / (C / (f * 1e9)):.0f} Hz); largest: "
                     + ", ".join(f"{name} {v['rms_m_s']:.2f} m/s" for name, v in worst))
        lines.append(f"  joint acceleration: error {result['acceleration_error_m_s2']:.1f} m/s2 (true mean "
                     f"{result['acceleration_true_m_s2']:.1f}, estimate {result['acceleration_estimate_m_s2']:.1f})")
    if result.get("seen_joint_mm") is not None:
        hidden = result.get("hidden_joint_mm")
        lines.append(f"  joints of observed parts {result['seen_joint_mm']:.1f} mm; of hidden parts "
                     + (f"{hidden:.1f} mm" if hidden is not None else "none hidden")
                     + f" ({100 * result['hidden_fraction']:.0f}% of joint-frames hidden)")
    if "foot_skating_m_s" in result:
        s = result["foot_skating_m_s"]
        lines.append(f"  feet standing: estimated ankle speed median {s['median']:.3f} m/s, p90 {s['p90']:.3f} m/s")
    return "\n".join(lines)


def save(result: dict, path) -> None:
    Path(path).write_text(json.dumps(result, indent=2))
