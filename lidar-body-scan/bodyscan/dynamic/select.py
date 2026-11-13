"""Selection among several tracked motions of the same take.

The fit of a frame has several local minima where a limb moves fast or is seen by few points: two runs of
the same configuration that differ only in rounding put an arm more than 10 cm apart in about a third of
the frames of a jump (s02_jump01, frames 225 to 261, 3 Nov 2026). The measurements usually tell which of
the two is right, so several runs are tracked (other rounding, other sensor combinations) and combined:

1. every motion is scored frame by frame against the measurements with the tracker's own score (the mean
   distance of the points to the visible body, capped, per point of the Ouster, plus the share of the body
   lying where a sensor saw through);
2. a path through the motions is chosen by dynamic programming: the least sum of the scores plus a cost for
   changing motion, which grows with how far apart the two motions are at the frame of the change (a change
   where they nearly coincide is free, a change that would make a limb jump is paid for);
3. each arm is then chosen on its own in the same way, the rest of the body kept from step 2, because one
   run often has the left arm right and another the right arm.

The result is a motion file with the chosen poses, meant as the warm start of a last tracking pass
(`bodyscan track --init-motion`), which fits every frame from it and refines the windows.

On that jump window the selection among ten runs leaves 485 Ouster points 5 to 25 cm from the body, against
832 on average for the single runs, and two selections from disjoint halves of the runs disagree on an arm
in 6 of 37 frames, against 11 to 13 for two single runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bodyscan.body import skeleton
from bodyscan.log import info

# The joints (indices into the 21 body joints) chosen on their own, and the joints (of the 55) whose
# positions measure how far apart two motions are in that part of the body.
ARM_JOINTS = {side: [skeleton.JOINT[f"{side}_{name}"] - 1 for name in ("collar", "shoulder", "elbow", "wrist")]
              for side in ("left", "right")}
ARM_POSITIONS = {side: [skeleton.JOINT[f"{side}_{name}"] for name in ("elbow", "wrist")] for side in ("left", "right")}
BODY_POSITIONS = list(range(22))


@dataclass
class SelectOptions:
    switch_weight: float = 0.05     # cost of a change of motion per metre the two motions are apart there
    switch_free: float = 0.03       # two motions closer than this [m] at a frame count as the same there
    limbs: bool = True              # choose each arm on its own after the whole body
    holdout_fraction: float = 0.0   # points left out of the scores, as in the fits (track --holdout-fraction)
    holdout_seed: int = 0
    rounds: int = 1                 # passes over the two arms
    same_within: float = 0.03       # an arm closer than this [m] to the chosen one is not scored again


# ------------------------------------------------------------------------------------------------ motions
def load_motions(paths) -> tuple[list[dict], np.ndarray]:
    """The motions cut to the frames they all have (frame_index), in the same order; and those indices."""
    from scipy.spatial.transform import Rotation
    raw = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            raw.append({key: data[key] for key in ("root_rotation", "root_position", "body", "frame_index", "times",
                                                   "joints", "model_sha1") if key in data.files})
    common = sorted(set.intersection(*[set(int(v) for v in m["frame_index"]) for m in raw]))
    if not common:
        raise SystemExit("the motions have no frame in common")
    motions = []
    for m in raw:
        row = {int(v): k for k, v in enumerate(m["frame_index"])}
        rows = [row[index] for index in common]
        motions.append({"root_matrix": Rotation.from_rotvec(m["root_rotation"][rows]).as_matrix(),
                        "root_rotation": np.asarray(m["root_rotation"][rows], dtype=np.float64),
                        "root_position": np.asarray(m["root_position"][rows], dtype=np.float64),
                        "body": np.asarray(m["body"][rows], dtype=np.float64).reshape(len(rows), 21, 3),
                        "joints": np.asarray(m["joints"][rows], dtype=np.float64)[:, :22],
                        "times": np.asarray(m["times"][rows], dtype=np.float64),
                        "model_sha1": str(m["model_sha1"]) if "model_sha1" in m else ""})
    return motions, np.array(common, dtype=np.int64)


def scoring_tracker(segments, avatar_path, extra_sensors=None, model_path=None, sensor_range: str = "",
                    sensor_noise: str = "", holdout_fraction: float = 0.0, holdout_seed: int = 0):
    """The tracker that scores poses: body pose, every sensor of the take, the reference count. With a
    held-out fraction the same points are left out of the scores as out of the fits (same seed), so that
    they stay an independent check of the chosen motion."""
    import torch
    from bodyscan.dynamic.avatar import Avatar
    from bodyscan.dynamic.segment import SegmentedRecording
    from bodyscan.dynamic.track import TrackPipelineConfig, Tracker
    config = TrackPipelineConfig()
    c = config.tracking
    c.device, c.precision, c.pose_model = "cpu", "float64", "body21"
    c.extra_sensors = str(extra_sensors) if extra_sensors else ""
    c.extra_sensor_range, c.extra_sensor_noise = sensor_range, sensor_noise
    c.holdout_fraction, c.holdout_seed = holdout_fraction, holdout_seed
    avatar = Avatar.load(avatar_path)
    return Tracker(avatar, avatar.model(model_path, "cpu", torch.float64), SegmentedRecording(segments), config)


# ------------------------------------------------------------------------------------------------ the choice
def least_cost_path(scores: np.ndarray, positions: np.ndarray, weight: float, free: float) -> np.ndarray:
    """(F,) the motion chosen at every frame. scores (K, F): lower is better; positions (K, F, J, 3): the
    joints that say how far apart two motions are. A change from motion i to motion j at frame t costs
    weight * max(largest distance between their joints at t - free, 0)."""
    count, frames = scores.shape
    total = scores[:, 0].copy()
    back = np.zeros((count, frames), dtype=np.int64)
    for t in range(1, frames):
        apart = np.linalg.norm(positions[:, None, t] - positions[None, :, t], axis=3).max(axis=2)
        step = total[:, None] + weight * np.maximum(apart - free, 0.0)
        back[:, t] = step.argmin(axis=0)
        total = step.min(axis=0) + scores[:, t]
    path = np.zeros(frames, dtype=np.int64)
    path[-1] = total.argmin()
    for t in range(frames - 1, 0, -1):
        path[t - 1] = back[path[t], t]
    return path


class Composite:
    """The pose of every frame put together from the motions: root, trunk, legs and head from path['rest'],
    each arm from its own path."""

    def __init__(self, motions: list[dict], rest: np.ndarray):
        self.motions = motions
        self.paths = {"rest": rest.copy(), "left": rest.copy(), "right": rest.copy()}

    def pose(self, t: int, override: tuple | None = None):
        """(root matrix, root position, body) of frame t; override (part, motion): that part from that motion."""
        chosen = {part: int(path[t]) for part, path in self.paths.items()}
        if override is not None:
            chosen[override[0]] = int(override[1])
        base = self.motions[chosen["rest"]]
        body = base["body"][t].copy()
        for side in ("left", "right"):
            body[ARM_JOINTS[side]] = self.motions[chosen[side]]["body"][t][ARM_JOINTS[side]]
        return base["root_matrix"][t], base["root_position"][t], body


def frame_score(tracker, frame, poses) -> float:
    """The score of the middle pose of three (frame before, this frame, frame after; each (root matrix,
    position, body, time)): the points of the other sensors are compared with the body moved to their time."""
    before, here, after = poses
    if len(tracker.views) > 1 and after[3] > before[3]:
        frame.set_velocity((tracker.vertices_of(*after[:3]) - tracker.vertices_of(*before[:3])) / (after[3] - before[3]))
    return float(tracker.explain(here[0], here[1], here[2], frame)["score"])


def select_motion(motion_paths, segments, avatar_path, out, extra_sensors=None, model_path=None,
                  sensor_range: str = "", sensor_noise: str = "", options: SelectOptions | None = None) -> dict:
    """Choose among the motions (see the module text) and write the chosen poses as a motion file for the
    warm start of 'bodyscan track'; returns a summary."""
    from scipy.spatial.transform import Rotation
    options = options or SelectOptions()
    started = time.time()
    motions, indices = load_motions(motion_paths)
    names = [Path(p).parent.name if Path(p).stem == "motion" else Path(p).stem for p in motion_paths]
    count, frames = len(motions), len(indices)
    tracker = scoring_tracker(segments, avatar_path, extra_sensors, model_path, sensor_range, sensor_noise,
                              options.holdout_fraction, options.holdout_seed)
    times = motions[0]["times"]

    def neighbours(t):
        return max(t - 1, 0), min(t + 1, frames - 1)

    # 1. every motion, whole body
    scores = np.zeros((count, frames))
    for t in range(frames):
        frame = tracker.load_frame(int(indices[t]))
        a, b = neighbours(t)
        for h, m in enumerate(motions):
            poses = [(m["root_matrix"][k], m["root_position"][k], m["body"][k], times[k]) for k in (a, t, b)]
            scores[h, t] = frame_score(tracker, frame, poses)
        if (t + 1) % 50 == 0 or t + 1 == frames:
            info(f"  scored frames: {t + 1}/{frames}  elapsed {time.time() - started:5.0f} s")
    joints = np.array([m["joints"] for m in motions])                               # (K, F, 22, 3)
    rest = least_cost_path(scores, joints[:, :, BODY_POSITIONS], options.switch_weight, options.switch_free)
    composite = Composite(motions, rest)
    chosen_score = scores[rest, np.arange(frames)].copy()
    whole_body_score = float(chosen_score.mean())

    # 2. each arm on its own
    rescored = 0
    if options.limbs and count > 1:
        for _ in range(options.rounds):
            for side in ("left", "right"):
                arm = joints[:, :, ARM_POSITIONS[side]]                              # (K, F, 2, 3)
                current = composite.paths[side]
                arm_scores = np.repeat(chosen_score[None], count, axis=0)
                for t in range(frames):
                    apart = np.linalg.norm(arm[:, t] - arm[current[t], t][None], axis=2).max(axis=1)
                    candidates = [h for h in range(count) if h != current[t] and apart[h] > options.same_within]
                    if not candidates:
                        continue
                    frame = tracker.load_frame(int(indices[t]))
                    a, b = neighbours(t)
                    for h in candidates:
                        poses = [(*composite.pose(k, (side, h)), times[k]) for k in (a, t, b)]
                        arm_scores[h, t] = frame_score(tracker, frame, poses)
                        rescored += 1
                path = least_cost_path(arm_scores, arm, options.switch_weight, options.switch_free)
                # an arm that counts as the same as the chosen one stays the chosen one
                same = np.linalg.norm(arm[path, np.arange(frames)] - arm[current, np.arange(frames)], axis=2).max(axis=1) \
                    <= options.same_within
                path[same] = current[same]
                composite.paths[side] = path
                chosen_score = arm_scores[path, np.arange(frames)]
                info(f"  {side} arm chosen: {int((path != composite.paths['rest']).sum())} frames from another "
                     f"motion than the body, elapsed {time.time() - started:5.0f} s")

    # 3. the chosen poses as a motion file
    rotation, position, body = [], [], []
    for t in range(frames):
        r, p, b = composite.pose(t)
        rotation.append(Rotation.from_matrix(r).as_rotvec())
        position.append(p)
        body.append(b)
    import torch
    with torch.no_grad():
        _, posed = tracker.pose(torch.zeros(frames, 3, dtype=tracker.dtype),
                                torch.as_tensor(np.array([composite.pose(t)[0] for t in range(frames)]), dtype=tracker.dtype),
                                torch.as_tensor(np.array(position), dtype=tracker.dtype),
                                torch.as_tensor(np.array(body), dtype=tracker.dtype))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, times=times, frame_index=indices, root_rotation=np.array(rotation),
                        root_position=np.array(position), body=np.array(body), joints=posed.numpy().astype(np.float32),
                        model_sha1=motions[0]["model_sha1"], pose_model="body21", selected_from=np.array(names),
                        path_rest=composite.paths["rest"], path_left_arm=composite.paths["left"],
                        path_right_arm=composite.paths["right"], scores=scores.astype(np.float32),
                        chosen_score=chosen_score.astype(np.float32))
    used = {part: {names[h]: int((path == h).sum()) for h in sorted(set(path.tolist()))}
            for part, path in composite.paths.items()}
    summary = {"motions": names, "frames": frames,
               "score_mm": {name: round(1000 * float(scores[h].mean()), 3) for h, name in enumerate(names)},
               "score_whole_body_choice_mm": round(1000 * whole_body_score, 3),
               "score_chosen_mm": round(1000 * float(chosen_score.mean()), 3),
               "changes_of_motion": int((np.diff(composite.paths["rest"]) != 0).sum()),
               "frames_from_each_motion": used, "arm_poses_scored": rescored,
               "seconds": round(time.time() - started, 1), "out": str(out)}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    return summary
