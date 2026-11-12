"""Viewer packs: everything the motion viewer shows of one tracked take, in two or three files.

    viewer_frames.npz        per frame: the pose, the joints and the metrics of the frame
    viewer_points.npz        per point of every sensor, in frame order (frame_offsets): the world point, its
                             sensor, its distance to the body, the body part and the face it was matched to
    viewer_uncertainty.npz   optional: the uncertainty of every joint's position per frame

The viewer (a small web server and a page, outside this package) reads only these files and the body
model, so a take tracked on a cluster is packed there and looked at on a laptop.

Every sensor of the take is packed, whether the run fitted it or not: the distance of a sensor's points
to the body is then either the fit residual (fitted sensor) or an independent check of the pose (a sensor
left out of the fit, or a frame of a sensor dropped because the person stood out of its range).
viewer_frames.npz records which is which (sensor_fitted, sensor_used).

Distances are in metres and are not divided by the sensors' noise. The body is moved to each sensor's
time before its points are compared with it, as in the tracker.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bodyscan.body import skeleton
from bodyscan.log import info

# Joint position uncertainty: noise of an Ouster point on the person [m] (the median distance of its points to
# the fitted body in the s02 takes), and the weak prior that keeps the uncertainty of an unmeasured joint finite.
OUSTER_NOISE = 0.007
PRIOR_ANGLE = 1.0           # rad per joint angle
PRIOR_POSITION = 1.0        # m per root coordinate


@dataclass
class PackOptions:
    uncertainty_step: int = 0      # joint uncertainty every this many frames (0: not computed)
    batch: int = 32                # frames posed at once


# ------------------------------------------------------------------------------------------------ the tracker of a motion
def load_motion(path) -> dict:
    """A motion .npz as the arrays the tracker's own quality code reads (root_matrix, slot added)."""
    from scipy.spatial.transform import Rotation
    with np.load(path, allow_pickle=True) as data:
        motion = {key: data[key] for key in data.files}
    motion["root_matrix"] = Rotation.from_rotvec(motion["root_rotation"]).as_matrix()
    motion["slot"] = motion["frame_index"]
    return motion


def tracker_of(motion: dict, segments, extra_sensors, avatar_path, model_path=None, hand_pca=None):
    """The tracker that reads a motion: its pose model and hands, every sensor of the take fitted (so that
    every sensor's points are matched to the body) and no range limits."""
    import torch
    from bodyscan.dynamic.avatar import Avatar
    from bodyscan.dynamic.segment import SegmentedRecording
    from bodyscan.dynamic.track import TrackPipelineConfig, Tracker
    run = run_config(motion)
    config = TrackPipelineConfig()
    c = config.tracking
    c.device, c.precision = "cpu", "float64"
    c.pose_model = str(motion["pose_model"]) if "pose_model" in motion else "body21"
    if c.pose_model == "full55":
        c.hand_pca = str(hand_pca) if hand_pca else str(run.get("hand_pca", ""))
        c.hand_components = int(motion["hand_components"])
        c.flat_hand_mean = bool(motion["flat_hand_mean"])
        c.face_fixed = bool(motion["face_fixed"]) if "face_fixed" in motion else True
    c.extra_sensors = str(extra_sensors) if extra_sensors else ""
    c.extra_sensor_noise = str(run.get("extra_sensor_noise", ""))
    avatar = Avatar.load(avatar_path)
    model = avatar.model(model_path, "cpu", torch.float64)
    return Tracker(avatar, model, SegmentedRecording(segments), config)


def run_config(motion: dict) -> dict:
    """The tracking section of the configuration a motion was made with ({} for motions without one)."""
    if "config" not in motion:
        return {}
    return json.loads(str(motion["config"])).get("tracking", {})


# ------------------------------------------------------------------------------------------------ one frame
def pose_frames(tracker, motion: dict, span: np.ndarray):
    """Vertices (F, V, 3), vertex normals (F, V, 3) and joints (F, 55, 3) of the frames 'span' (numpy)."""
    import torch
    from bodyscan.body.model import vertex_normals
    with torch.no_grad():
        extra = None
        if tracker.layout.full and "extra" in motion and motion["extra"].shape[1] > 0:
            extra = torch.as_tensor(motion["extra"][span], dtype=tracker.dtype)
        vertices, joints = tracker.pose(torch.zeros(len(span), 3, dtype=tracker.dtype),
                                        torch.as_tensor(motion["root_matrix"][span], dtype=tracker.dtype),
                                        torch.as_tensor(motion["root_position"][span], dtype=tracker.dtype),
                                        torch.as_tensor(motion["body"][span], dtype=tracker.dtype), extra=extra)
        normals = vertex_normals(vertices, tracker.faces)
    return vertices.numpy().astype(np.float64), normals.numpy().astype(np.float64), joints.numpy().astype(np.float64)


def match_frame(tracker, motion: dict, k: int, vertices: np.ndarray, normals: np.ndarray, i: int, span: np.ndarray):
    """The frame k (position i of the posed frames 'span') and its match: every sensor's points against the
    body that sensor sees, the body moved to the sensor's time by the vertex velocities of the frames around."""
    frame = tracker.load_frame(int(motion["slot"][k]))
    if len(tracker.views) > 1 and len(span) >= 2:
        before, after = max(i - 1, 0), min(i + 1, len(span) - 1)
        gap = motion["times"][span[after]] - motion["times"][span[before]]
        if gap > 0:
            frame.set_velocity((vertices[after] - vertices[before]) / gap)
    return frame, tracker.match(vertices[i], normals[i], frame)


def point_table(tracker, frame, match, vertices: np.ndarray) -> dict:
    """Per point of a frame: sensor number, distance [m], signed distance (positive outside the body), body
    part and face of its match (-1 beyond the tracker's reach), and per sensor the vertices in its sight."""
    count = len(frame.points_np)
    sensor = np.zeros(count, dtype=np.int64)
    part = np.full(count, -1, dtype=np.int64)
    face = np.full(count, -1, dtype=np.int64)
    signed = np.full(count, np.nan)
    valid = match.point_index.numpy()
    faces = match.face_ids.numpy()
    corners = tracker.faces_np[faces]
    weights = match.bary.numpy().astype(np.float64)
    surface = np.einsum("mk,mkd->md", weights, vertices[corners])
    if match.point_shift is not None:
        surface = surface + match.point_shift.numpy().astype(np.float64)
    normal = np.cross(vertices[corners[:, 1]] - vertices[corners[:, 0]], vertices[corners[:, 2]] - vertices[corners[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    side = np.sign(((frame.points_np[valid] - surface) * normal).sum(axis=1))
    signed[valid] = side * match.distance[valid]
    part[valid] = tracker.part_labels[corners[:, 0]]
    face[valid] = faces
    in_sight = np.zeros((len(tracker.views), len(vertices)), dtype=bool)
    for view_match in match.views:
        sensor[view_match.frame.slice] = view_match.number
        in_sight[view_match.number] = view_match.visible
    return {"sensor": sensor, "distance": match.distance, "signed": signed, "part": part, "face": face,
            "in_sight": in_sight}


# ------------------------------------------------------------------------------------------------ uncertainty
def joint_uncertainty(tracker, motion: dict, k: int, frame, match, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standard deviation of every joint's position [m] (55,) and of the 69 pose parameters (root rotation,
    root position, 21 body joints) that follows from the noise of the frame's points: the Gauss-Newton
    covariance of the point-to-plane residuals (each point with its sensor's noise, robust weights at the
    fitted pose) with a weak prior. A lower bound: wrong matches and model errors are not in it."""
    import torch
    valid = match.point_index.numpy()
    if len(valid) < 10:
        return np.full(55, np.nan), np.full(69, np.nan)
    corners_np = tracker.faces_np[match.face_ids.numpy()]
    corners = torch.as_tensor(corners_np)
    weights = match.bary.to(torch.float64)
    points = torch.as_tensor(frame.points_np[valid], dtype=torch.float64)
    shift = None if match.point_shift is None else match.point_shift.to(torch.float64)
    normal = torch.as_tensor(normals[corners_np].mean(axis=1))
    normal = normal / normal.norm(dim=1, keepdim=True).clamp_min(1e-9)
    sigma = torch.ones(len(valid), dtype=torch.float64)
    if frame.point_sigma_np is not None:
        sigma = torch.as_tensor(frame.point_sigma_np[valid], dtype=torch.float64)
    extra = None
    if tracker.layout.full and "extra" in motion and motion["extra"].shape[1] > 0:
        extra = torch.as_tensor(motion["extra"][k:k + 1], dtype=torch.float64)
    reference = torch.as_tensor(motion["root_matrix"][k:k + 1], dtype=torch.float64)
    theta = torch.cat([torch.zeros(3, dtype=torch.float64),
                       torch.as_tensor(motion["root_position"][k], dtype=torch.float64),
                       torch.as_tensor(motion["body"][k], dtype=torch.float64).reshape(-1)])

    def residuals_and_joints(parameters):
        vertices, joints = tracker.pose(parameters[:3][None], reference, parameters[3:6][None],
                                        parameters[6:].reshape(1, 21, 3), extra=extra)
        surface = (weights[:, :, None] * vertices[0][corners]).sum(dim=1)
        if shift is not None:
            surface = surface + shift
        return ((points - surface) * normal).sum(dim=1) / (OUSTER_NOISE * sigma), joints[0]

    with torch.no_grad():
        residual, _ = residuals_and_joints(theta)
    robust = 1.0 / (1.0 + (residual / 3.0) ** 2) ** 2                    # Geman-McClure weights, scale 3 sigma
    jacobian, joint_jacobian = torch.autograd.functional.jacobian(residuals_and_joints, theta, vectorize=True,
                                                                  strategy="forward-mode")
    jacobian = jacobian.reshape(len(valid), 69)
    joint_jacobian = joint_jacobian.reshape(165, 69)
    prior = torch.full((69,), 1.0 / PRIOR_ANGLE ** 2, dtype=torch.float64)
    prior[3:6] = 1.0 / PRIOR_POSITION ** 2
    covariance = torch.linalg.inv(jacobian.T @ (robust[:, None] * jacobian) + torch.diag(prior))
    joint_variance = torch.diag(joint_jacobian @ covariance @ joint_jacobian.T).reshape(55, 3).sum(dim=1)
    return (torch.sqrt(joint_variance.clamp_min(0)).numpy(), torch.sqrt(torch.diag(covariance).clamp_min(0)).numpy())


# ------------------------------------------------------------------------------------------------ per frame metrics
def kinematics(motion: dict) -> dict:
    """Per frame numbers from the pose alone: joint rotation between frames, acceleration, jerk, sliding of
    the feet in contact, trunk twist, toe angles, joint limit excess per joint."""
    from scipy.spatial.transform import Rotation
    J = skeleton.JOINT
    times = motion["times"]
    frames = len(times)
    body = motion["body"]
    joints = motion["joints"].astype(np.float64)
    step = np.maximum(np.diff(times), 1e-4)
    joint_speed = np.zeros((frames, 21))                               # deg per frame
    for joint in range(21):
        rotation = Rotation.from_rotvec(body[:, joint])
        joint_speed[1:, joint] = np.degrees((rotation[1:] * rotation[:-1].inv()).magnitude())
    root = Rotation.from_rotvec(motion["root_rotation"])
    root_speed = np.zeros(frames)
    root_speed[1:] = np.degrees((root[1:] * root[:-1].inv()).magnitude())
    velocity = np.diff(joints[:, :22], axis=0) / step[:, None, None]
    accel, jerk = np.zeros(frames), np.zeros(frames)
    if frames > 2:
        acceleration = 2 * np.diff(velocity, axis=0) / np.maximum(times[2:] - times[:-2], 1e-4)[:, None, None]
        accel[1:-1] = np.linalg.norm(acceleration, axis=2).max(axis=1)
        if frames > 3:
            jerk[2:-1] = np.linalg.norm(np.diff(acceleration, axis=0) / step[1:-1, None, None], axis=2).max(axis=1)
    sliding = np.zeros(frames)
    contacts = motion["contacts"]
    for side, name in enumerate(("left", "right")):
        speed = np.linalg.norm(np.diff(joints[:, J[f"{name}_ankle"], :2], axis=0), axis=1) / step
        sliding[1:] = np.maximum(sliding[1:], np.where(contacts[1:, side] & contacts[:-1, side], speed, 0.0))

    def heading(vector):
        return np.degrees(np.arctan2(vector[:, 1], vector[:, 0]))
    twist = heading(joints[:, J["left_shoulder"]] - joints[:, J["right_shoulder"]]) \
        - heading(joints[:, J["left_hip"]] - joints[:, J["right_hip"]])
    lower, upper = skeleton.limit_arrays()
    excess = np.degrees(np.maximum(lower[skeleton.BODY] - body, 0) + np.maximum(body - upper[skeleton.BODY], 0)).max(axis=2)
    return {"joint_speed": joint_speed, "root_speed": root_speed, "accel": accel, "jerk": jerk, "sliding": sliding,
            "trunk_twist": (twist + 180.0) % 360.0 - 180.0,
            "toe_x": np.degrees(body[:, [J["left_foot"] - 1, J["right_foot"] - 1], 0]), "limit_excess_joint": excess}


# ------------------------------------------------------------------------------------------------ the pack
def write_viewer_pack(motion_path, segments, avatar_path, out, extra_sensors=None, model_path=None, hand_pca=None,
                      options: PackOptions | None = None) -> dict:
    """Pack a tracked take for the viewer; returns a short summary (frames, points, seconds)."""
    from bodyscan.dynamic.views import parse_sensor_options
    options = options or PackOptions()
    started = time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    motion = load_motion(motion_path)
    tracker = tracker_of(motion, segments, extra_sensors, avatar_path, model_path, hand_pca)
    run = run_config(motion)
    names = [view.name for view in tracker.views]
    sensors, parts = len(names), len(skeleton.PART_NAMES)
    frames = len(motion["times"])
    vertex_count = int(tracker.shaped.vertices.shape[0])
    positions = np.array([np.asarray(tracker.sensor, dtype=np.float64)]
                         + [np.asarray(view.position, dtype=np.float64) for view in tracker.views[1:]])
    excluded = {name for name in str(run.get("exclude_sensors", "")).split(",") if name}
    fitted = np.array([name == "ouster" or (bool(run.get("extra_sensors")) and name not in excluded) for name in names])
    limits = parse_sensor_options(str(run.get("extra_sensor_range", "")), 2, "extra_sensor_range")

    columns = {key: [] for key in ("frame", "sensor", "distance", "signed", "part", "face", "xyz", "dt")}
    in_sight = np.zeros((frames, sensors, vertex_count), dtype=bool)
    sensor_points = np.zeros((frames, sensors), dtype=np.int64)
    sensor_used = np.zeros((frames, sensors), dtype=bool)
    per_sensor = {key: np.full((frames, sensors), np.nan) for key in ("median", "p90", "signed", "unexplained")}
    part_points = np.zeros((frames, sensors, parts), dtype=np.int32)
    part_residual = np.full((frames, sensors, parts), np.nan, dtype=np.float32)
    part_signed = np.full((frames, sensors, parts), np.nan, dtype=np.float32)
    free_share = np.zeros(frames)
    joint_std = np.full((frames, 55), np.nan)
    parameter_std = np.full((frames, 69), np.nan)
    reach = tracker.config.tracking.max_correspondence

    for first in range(0, frames, options.batch):
        span = np.arange(first, min(first + options.batch, frames))
        vertices, normals, _ = pose_frames(tracker, motion, span)
        for i, k in enumerate(span):
            frame, match = match_frame(tracker, motion, int(k), vertices, normals, i, span)
            table = point_table(tracker, frame, match, vertices[i])
            count = len(table["distance"])
            columns["frame"].append(np.full(count, k, dtype=np.int32))
            columns["sensor"].append(table["sensor"].astype(np.int8))
            columns["distance"].append(table["distance"].astype(np.float32))
            columns["signed"].append(table["signed"].astype(np.float32))
            columns["part"].append(table["part"].astype(np.int8))
            columns["face"].append(table["face"].astype(np.int32))
            columns["xyz"].append(frame.points_np.astype(np.float32))
            columns["dt"].append((frame.point_times - motion["times"][k]).astype(np.float32))
            in_sight[k] = table["in_sight"]
            free_share[k] = tracker.free_share(match)
            for s, name in enumerate(names):
                mine = table["sensor"] == s
                if not mine.any():
                    continue
                distance, signed = table["distance"][mine], table["signed"][mine]
                sensor_points[k, s] = int(mine.sum())
                per_sensor["median"][k, s] = np.median(distance)
                per_sensor["p90"][k, s] = np.percentile(distance, 90)
                per_sensor["unexplained"][k, s] = np.mean((distance > 0.05) & (distance < reach))
                if np.isfinite(signed).any():
                    per_sensor["signed"][k, s] = np.nanmedian(signed)
                if fitted[s]:
                    ranges = np.linalg.norm(frame.points_np[mine] - positions[s], axis=1)
                    sensor_used[k, s] = name not in limits or limits[name][0] <= np.median(ranges) <= limits[name][1]
                part_of = table["part"][mine]
                for p in np.unique(part_of[part_of >= 0]):
                    chosen = part_of == p
                    part_points[k, s, p] = int(chosen.sum())
                    part_residual[k, s, p] = np.median(distance[chosen])
                    if np.isfinite(signed[chosen]).any():
                        part_signed[k, s, p] = np.nanmedian(signed[chosen])
            if options.uncertainty_step > 0 and k % options.uncertainty_step == 0:
                joint_std[k], parameter_std[k] = joint_uncertainty(tracker, motion, int(k), frame, match, normals[i])
        info(f"  packed frames: {span[-1] + 1}/{frames}  elapsed {time.time() - started:5.0f} s")

    points = {key: np.concatenate(value) for key, value in columns.items()}
    offsets = np.searchsorted(points["frame"], np.arange(frames + 1))      # the points are in frame order
    del points["frame"]                                                    # given by frame_offsets
    for key in ("distance", "signed", "dt"):                               # millimetres and milliseconds suffice
        points[key] = points[key].astype(np.float16)
    np.savez_compressed(out / "viewer_points.npz", frame_offsets=offsets.astype(np.int64),
                        seen_by=np.packbits(in_sight, axis=2), sensor_names=np.array(names),
                        part_names=np.array(skeleton.PART_NAMES), **points)

    heldout_sensor = np.full((frames, sensors), np.nan)
    if "sensor_heldout_residual_median" in motion:
        for i, name in enumerate(str(n) for n in motion["sensor_names"]):
            if name in names:
                heldout_sensor[:, names.index(name)] = motion["sensor_heldout_residual_median"][:, i]
    summary = {}
    motion_json = Path(motion_path).with_suffix(".json")
    if motion_json.exists():
        summary = {key: value for key, value in json.loads(motion_json.read_text()).items()
                   if not isinstance(value, (dict, list))}
    pelvis = motion["joints"][:, 0].astype(np.float64)
    np.savez_compressed(
        out / "viewer_frames.npz", times=motion["times"], frame_index=motion["frame_index"], joints=motion["joints"],
        root_rotation=motion["root_rotation"], root_position=motion["root_position"], body=motion["body"],
        extra=motion["extra"] if "extra" in motion else np.zeros((frames, 0)),
        hands=(np.concatenate([motion["left_hand"], motion["right_hand"]], axis=1) if "left_hand" in motion
               else np.zeros((frames, 0, 3))),
        pose_model=str(motion["pose_model"]) if "pose_model" in motion else "body21",
        flat_hand_mean=bool(motion["flat_hand_mean"]) if "flat_hand_mean" in motion else False,
        hand_components=int(motion["hand_components"]) if "hand_components" in motion else 0,
        sensor_names=np.array(names), sensor_positions=positions, sensor_fitted=fitted, sensor_used=sensor_used,
        sensor_distance=np.linalg.norm(pelvis[:, None] - positions[None], axis=2),
        holdout_fraction=float(run.get("holdout_fraction", 0.0)),
        residual_median=motion["residual_median"], residual_p90=motion["residual_p90"],
        sensor_residual_median=per_sensor["median"], sensor_residual_p90=per_sensor["p90"],
        sensor_signed_median=per_sensor["signed"], sensor_unexplained=per_sensor["unexplained"].astype(np.float32),
        sensor_points=sensor_points, sensor_heldout_residual_median=heldout_sensor,
        heldout_residual_median=motion["heldout_residual_median"], unexplained=motion["unexplained"],
        clutter=motion["clutter"], free_share=free_share, floor=motion["floor"],
        limit_excess_deg=motion["limit_excess_deg"], flags=motion["flags"], flag_names=motion["flag_names"],
        observed=motion["observed"], part_names=np.array(skeleton.PART_NAMES), contacts=motion["contacts"],
        part_points=part_points, part_residual=part_residual, part_signed=part_signed, guessed=motion["guessed"],
        guessed_limbs=motion["guessed_limbs"], hand_points=motion["hand_points"], summary=json.dumps(summary),
        options=json.dumps({key: run.get(key) for key in ("extra_sensor_noise", "extra_sensor_range", "exclude_sensors",
                                                         "reference_count", "toe_weight", "init_full", "pose_model")}),
        **kinematics(motion))
    if options.uncertainty_step > 0:
        computed = np.flatnonzero(np.isfinite(joint_std).any(axis=1))
        if len(computed):
            nearest = computed[np.abs(np.arange(frames)[:, None] - computed[None]).argmin(axis=1)]
            np.savez_compressed(out / "viewer_uncertainty.npz", joint_std=joint_std[nearest],
                                params_std=parameter_std[nearest], computed_frames=computed, sigma0=OUSTER_NOISE,
                                prior_angle=PRIOR_ANGLE, prior_position=PRIOR_POSITION)
    return {"frames": frames, "points": int(offsets[-1]), "sensors": names,
            "seconds": round(time.time() - started, 1), "out": str(out)}
