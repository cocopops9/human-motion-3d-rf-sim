"""Stage 1 with several hypotheses (beam tracking): instead of one pose per frame, the tracker carries the
'width' best tracks so far. Every frame, each track is continued twice: from the constant-velocity prediction
(the ordinary tracker) and from its last pose held still (a limb that stops or turns: the prediction's
velocity overshoots there). Each continuation costs what explain() scores it plus a price for leaving its own
prediction (switch_weight per metre of the largest joint distance beyond switch_free, as select-motion
charges for changing motion). The cheapest 'width' continuations that differ from each other (by more than
'distinct' at some joint) carry on; the motion is the cheapest track at the end, read back to the first frame.

The ordinary tracker decides each frame with that frame's points alone; a wrong decision (an arm left behind
in a fast swing) is then carried on by the continuity terms until a search happens to undo it. Here a
continuation that explained its frame slightly worse survives as long as it stays among the best, and wins
when the frames after it are explained better from it: the decision waits for the evidence of the next
frames. The cost is about 2 * width fits per frame instead of one."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bodyscan.body import skeleton
from bodyscan.log import Progress, info


@dataclass
class BeamOptions:
    width: int = 3                   # tracks carried from frame to frame
    switch_weight: float = 0.05      # price per metre of leaving the prediction (score units: metres)
    switch_free: float = 0.03        # ... beyond this distance (largest over the 22 main joints)
    distinct: float = 0.03           # continuations closer than this at every joint count as one
    still: bool = True               # also continue every track from its last pose held still


@dataclass
class Node:
    """One frame of one track; 'parent' is the frame before on the same track (None at a start)."""
    rotation: np.ndarray
    position: np.ndarray
    body: np.ndarray
    extra: np.ndarray
    energy: float
    time: float
    slot: int
    index: int
    explained: dict
    joints: np.ndarray
    cost: float
    restart: bool
    searched: bool
    parent: "Node | None"
    model: str = ""                  # the continuation that made it: moving, still (or empty at a start)


def joints_of_pose(tracker, rotation, position, body, extra=None) -> np.ndarray:
    """(22, 3) main joints of one pose."""
    import torch

    def tensor(values):
        return torch.as_tensor(np.asarray(values, dtype=np.float64), dtype=tracker.dtype, device=tracker.device)

    with torch.no_grad():
        _, joints = tracker.pose(torch.zeros(1, 3, dtype=tracker.dtype, device=tracker.device),
                                 tensor(rotation)[None], tensor(position)[None], tensor(body).reshape(1, 21, 3),
                                 extra=(tensor(extra).reshape(1, -1) if tracker.layout.full and extra is not None
                                        and len(extra) else None))
    return joints[0, list(skeleton.MAIN_JOINTS)].cpu().numpy().astype(np.float64)


def _history(node: Node, count: int) -> list:
    """The last 'count' nodes of a track up to 'node', oldest first (fewer after a start)."""
    chain = []
    while node is not None and len(chain) < count:
        chain.append(node)
        if node.restart:
            break
        node = node.parent
    return chain[::-1]


def _recent(node: Node, seconds: float, time: float) -> list:
    """The nodes of a track within 'seconds' before 'time', oldest first, back to its last start."""
    chain = []
    while node is not None and time - node.time <= seconds:
        chain.append(node)
        if node.restart:
            break
        node = node.parent
    return chain[::-1]


def track_beam(tracker, start: int = 0, stop: int | None = None, options: BeamOptions | None = None) -> dict:
    """Stage 1 with a beam of tracks over the segmented frames start..stop; the same arrays as
    Tracker.track (in the order of time)."""
    from bodyscan.body.rotations import matrix_to_axis_angle_np
    options = options or BeamOptions()
    c = tracker.config.tracking
    count = len(tracker.segmented)
    stop = count if stop is None else min(stop, count)
    beams: list[Node] = []
    progress = Progress("tracked frames (beam)", stop - start)
    searches = refused = lost = 0
    chosen_still = 0

    def make_node(result, explained, frame, person, k, parent, prediction_joints, restart, searched):
        joints = joints_of_pose(tracker, result[0], result[1], result[2], tracker.extra_of(result))
        step = 0.0
        if prediction_joints is not None:
            apart = float(np.linalg.norm(joints - prediction_joints, axis=1).max())
            step = options.switch_weight * max(apart - options.switch_free, 0.0)
        cost = (parent.cost if parent is not None else 0.0) + explained["score"] + step
        return Node(result[0], result[1], result[2], result[5], float(result[3]), float(frame.time), int(k),
                    int(person.index), {"distance": float(np.median(explained["distance"])),
                                        "observed_parts": explained["observed_parts"],
                                        "score": explained["score"]},
                    joints, cost, restart, searched, parent)

    for visited, k in enumerate(range(start, stop)):
        person = tracker.segmented.load(k)
        frame = tracker.frame_data(person)
        if not beams:
            result = tracker.initialise(frame)
            explained = tracker.explain_fit(result, frame)
            beams = [make_node(result, explained, frame, person, k, None, None, True, False)]
            progress.maybe(visited + 1, 25)
            continue
        candidates = []
        for parent in beams:
            history = _history(parent, 2)
            gap = c.gap_frames > 0 and frame.time - parent.time > c.gap_frames * tracker.frame_period
            if gap:
                history = history[-1:]
            if tracker.extra is not None:
                # the additional sensors' points moved to their times with this track's velocity (none
                # known after a start or a gap)
                if len(history) == 2 and history[1].time > history[0].time:
                    a, b = history
                    frame.set_velocity((tracker.vertices_of(b.rotation, b.position, b.body, b.extra)
                                        - tracker.vertices_of(a.rotation, a.position, a.body, a.extra))
                                       / (b.time - a.time))
                else:
                    frame.set_velocity(np.zeros(3))
            moving = tracker.predict([n.rotation for n in history], [n.position for n in history],
                                     [n.body for n in history], [n.time for n in history], frame.time,
                                     [n.extra for n in history])
            still = (parent.rotation, parent.position, parent.body, parent.extra)
            models = [("moving", moving)] + ([("still", still)] if options.still and len(history) == 2 else [])
            recent = _recent(parent, 1.0, frame.time)
            for name, prediction in models:
                result = tracker.fit_frame(frame, prediction[0], prediction[1], prediction[2], prediction)
                explained = tracker.explain_fit(result, frame)
                restarted = searched = False
                if np.median(explained["distance"]) > c.lost_residual:
                    yaw = tracker.yaw(parent.rotation)
                    retry = tracker.initialise(frame, parent.body, yaw + np.radians([0, -45, 45, -90, 90, 180]),
                                               parent.position, parent.extra)
                    retried = tracker.explain_fit(retry, frame)
                    if np.median(retried["distance"]) < np.median(explained["distance"]):
                        result, explained, restarted = retry, retried, True
                        lost += 1
                misfit = 0 < c.recover_fraction < explained["unexplained"] and explained["unexplained_count"] >= 10
                in_free_space = 0 < c.recover_free_space < explained["free"]
                plain = (result, explained)
                if misfit or in_free_space:
                    near = tracker.limbs_to_search(result[0], result[1], result[2], frame, explained)
                    limbs = (near - {"trunk"}) or None
                    trunk = in_free_space or "trunk" in near or explained["unexplained"] > 3 * c.recover_fraction
                    result = tracker.recover_limbs(frame, result, None if restarted else prediction,
                                                   None if restarted else (parent.rotation, parent.position,
                                                                           parent.body, parent.extra),
                                                   limbs=limbs, trunk=trunk)
                    explained = tracker.explain_fit(result, frame)
                    searched = True
                    searches += 1
                if not tracker.plausible_turn(result[0], frame.time, [n.rotation for n in recent],
                                              [n.time for n in recent]):
                    result, explained = plain
                    restarted = False
                    refused += 1
                predicted_joints = None if restarted else joints_of_pose(tracker, *prediction[:3], prediction[3])
                node = make_node(result, explained, frame, person, k, parent, predicted_joints, restarted, searched)
                node.model = name
                candidates.append(node)
        candidates.sort(key=lambda n: n.cost)
        kept: list[Node] = []
        for node in candidates:
            if all(np.linalg.norm(node.joints - other.joints, axis=1).max() > options.distinct for other in kept):
                kept.append(node)
            if len(kept) >= options.width:
                break
        beams = kept
        progress.maybe(visited + 1, 25)
    best = min(beams, key=lambda n: n.cost)
    chain = []
    node = best
    while node is not None:
        chain.append(node)
        node = node.parent
    chain.reverse()
    chosen_still = sum(1 for n in chain if getattr(n, "model", "") == "still")
    info(f"  beam of {options.width} tracks: the motion kept continues from the pose held still in {chosen_still} "
         f"of {len(chain)} frames; limbs searched in {searches} continuations, turns refused in {refused}, "
         f"restarts after a lost track in {lost}")
    rotations = [n.rotation for n in chain]
    return {"times": np.array([n.time for n in chain]), "frame_index": np.array([n.index for n in chain]),
            "slot": np.array([n.slot for n in chain], dtype=np.int64),
            "restart": np.array([n.restart for n in chain], dtype=bool),
            "searches": int(sum(n.searched for n in chain)), "turns_refused": refused, "facing_changes": 0,
            "facing_refused": 0, "turn_frames_refitted": 0,
            "root_rotation": np.array([matrix_to_axis_angle_np(r) for r in rotations]),
            "root_matrix": np.array(rotations), "root_position": np.array([n.position for n in chain]),
            "body": np.array([n.body for n in chain]), "energy": np.array([n.energy for n in chain]),
            "residual": np.array([n.explained["distance"] for n in chain]),
            "extra": np.array([n.extra for n in chain]).reshape(len(chain), tracker.layout.size),
            "warm_frames": 0,
            "observed": np.array([n.explained["observed_parts"] for n in chain]).reshape(len(chain),
                                                                                         len(skeleton.PART_NAMES))}
