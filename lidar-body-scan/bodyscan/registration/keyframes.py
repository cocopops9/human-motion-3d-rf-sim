"""Registration of the keyframes of a person turning in place (no turntable).

The person turns on the spot by steps and holds still in between. Every
keyframe (one per stop) gets its own pose: the body drifts a few cm at every
step and the turning axis wanders, so nothing assumes a fixed axis. The
registrations allow only what a standing person can do between stops: a turn
about the vertical and a shift (4 DOF), optionally a small lean (max_tilt),
with a robust kernel so that limbs that moved count little.

    1. consecutive keyframes k -> k+1, full yaw search (+-max_step), two kinds
       of start for every yaw (turn about the estimated body axis, or about
       the centroid)
    2. the majority sense of turning is enforced
    3. skip pairs k -> k+2 check the chain; the weaker step of a disagreement
       is searched again from the start implied by the skip pair
    4. every other overlapping pair, searched near the chained turn
    5. turn of every keyframe from all pairs at once (weighted least squares
       with a Huber reweighting): a side view constrains the turn poorly and
       must not decide it alone
    6. model poses from the turns; every overlapping pair refined from them
       with a bounded correction; pose graph; one more refinement
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from bodyscan.config import param
from bodyscan.geometry import (Target, evaluate, project_to_4dof, tilt_degrees, transform_points, wrapped_degrees,
                               yaw_of, yaw_transform)
from bodyscan.log import Progress, info, warning
from bodyscan.registration.icp import Pair, icp_4dof, icp_robust, information_matrix
from bodyscan.registration.turns import solve_turns

registration = o3d.pipelines.registration


@dataclass
class KeyframeConfig:
    """Registration of the keyframes (one per stop) of a person turning in place."""
    reg_voxel: float = param(0.012, "voxel of the clouds registered", unit="m",
                             effect="smaller: slower, slightly more precise")
    body_radius: float = param(0.10, "distance from the visible surface to the body's vertical axis", unit="m",
                               effect="used for the starting poses only")
    fine_distance: float = param(0.025, "overlap distance of the registration quality", unit="m")
    max_step: float = param(90.0, "largest turn between two stops that is searched", unit="deg",
                            effect="smaller is faster; must exceed the largest real step")
    hypothesis_step: float = param(10.0, "spacing of the yaw search", unit="deg")
    sense_tolerance: float = param(0.10, "overlap a match in the majority turning sense may lose and still "
                                         "replace a match turning the other way")
    loop_max_angle: float = param(60.0, "keyframes farther apart than this in turn are not registered", unit="deg")
    min_fitness: float = param(0.3, "pairs with less overlap are not used")
    max_correction: float = param(10.0, "bound of the turn correction of a pair from the model", unit="deg")
    max_shift: float = param(0.10, "bound of the shift correction of a pair from the model", unit="m")
    max_tilt: float = param(5.0, "largest lean between two stops (0 = turn and shift only, 4 DOF)", unit="deg",
                            effect="a lean of 1 deg moves the head by 3 cm: 0 cannot align head and feet together")


def window(expected_deg: float, half_width: float, step: float) -> np.ndarray:
    """Yaw hypotheses [deg] around an expected turn."""
    return expected_deg + np.arange(-half_width, half_width + 0.1, step)


class Keyframes:
    """Registration copies of the keyframe clouds (floor frame) and the pair registration."""

    def __init__(self, clouds, config: KeyframeConfig):
        self.config = config
        self.clouds = [c.voxel_down_sample(config.reg_voxel) for c in clouds]
        self.points = [np.asarray(c.points) for c in self.clouds]
        self.targets = [Target(c) for c in self.clouds]
        self.fine = config.fine_distance

    def __len__(self):
        return len(self.clouds)

    def register(self, i: int, j: int, starts, uncertain: bool) -> Pair:
        """Best alignment of keyframe i onto keyframe j among the starts.

        Every start is refined with 4 DOF (turn + shift), which is robust far
        from the answer; the three best are then refined with tilt allowed
        (if max_tilt > 0), which is accurate close to the answer."""
        results = []
        for start in starts:
            transform = icp_4dof(self.points[i], self.targets[j], start)
            fitness, rmse = evaluate(self.points[i], self.targets[j], transform, self.fine)
            results.append((fitness, -rmse, transform))
        results.sort(key=lambda item: (item[0], item[1]), reverse=True)
        candidates = [transform for _, _, transform in results[:3]]
        if self.config.max_tilt > 0:
            candidates = [icp_robust(self.points[i], self.targets[j], t, (0.04, 0.025, 0.015), (25, 20, 20),
                                     self.config.max_tilt) for t in candidates] + candidates
        best = None
        for transform in candidates:
            fitness, rmse = evaluate(self.points[i], self.targets[j], transform, self.fine)
            if (best is None or fitness > best.fitness + 0.02
                    or (fitness > best.fitness - 0.02 and rmse < best.rmse)):
                best = Pair(i, j, transform, fitness, rmse, uncertain)
        best.information = information_matrix(self.clouds[i], self.clouds[j], self.fine, best.transform)
        return best

    def pair_at(self, i: int, j: int, transform: np.ndarray, uncertain: bool) -> Pair:
        """Pair with a given transform (no registration)."""
        fitness, rmse = evaluate(self.points[i], self.targets[j], transform, self.fine)
        return Pair(i, j, transform, fitness, rmse, uncertain,
                    information_matrix(self.clouds[i], self.clouds[j], self.fine, transform))

    def body_axis(self, i: int) -> np.ndarray:
        """Horizontal position of the body's vertical axis in keyframe i: the
        centroid of the torso-height points (0.4 to 1.4 m, arms included) moved
        away from the sensor by body_radius (the visible surface lies on the
        sensor side of the axis; the sensor is above the origin of the floor frame)."""
        points = self.points[i]
        band = points[(points[:, 2] > 0.4) & (points[:, 2] < 1.4)]
        center = (band if len(band) > 50 else points)[:, :2].mean(axis=0)
        distance = np.linalg.norm(center)
        return center + self.config.body_radius * center / max(distance, 1e-6)

    def yaw_hypotheses(self, i: int, j: int, yaws_deg) -> list:
        """Starts for the alignment of keyframe i onto j, for every yaw:
        (a) turn about the body axis of i and move that axis onto the axis of
        j (a person who stepped away from the mark); (b) turn about the
        centroid of i, no shift (a person who turned on the spot)."""
        axis_i, axis_j = self.body_axis(i), self.body_axis(j)
        starts = []
        for yaw in yaws_deg:
            starts.append(yaw_transform(np.radians(yaw), np.append(axis_i, 0.0), np.append(axis_j - axis_i, 0.0)))
            starts.append(yaw_transform(np.radians(yaw), self.points[i].mean(axis=0)))
        return starts


@dataclass
class KeyframeResult:
    poses: list             # pose of every keyframe in the frame of keyframe 0 (floor frame)
    edges: list             # pairs kept by the pose graph
    keys: Keyframes
    stats: dict


class KeyframeRegistration:
    """Poses of the keyframes of a person turning in place (see the module docstring)."""

    def __init__(self, config: KeyframeConfig):
        self.config = config

    def small_correction(self, start, result, center) -> bool:
        c = self.config
        correction = np.linalg.inv(start) @ result
        turn = abs(np.degrees(yaw_of(correction)))
        shift = np.linalg.norm(transform_points(correction, center[None])[0] - center)
        return turn <= c.max_correction and shift <= c.max_shift and tilt_degrees(correction) <= max(c.max_tilt, 0.5)

    def run(self, clouds) -> KeyframeResult:
        c = self.config
        keys = Keyframes(clouds, c)
        count = len(keys)
        hypotheses = np.arange(-c.max_step, c.max_step + 0.1, c.hypothesis_step)

        info(f"consecutive pairs (full turn search, {len(hypotheses)} turns x 2 starts each):")
        progress = Progress("consecutive", count - 1)
        sequential = []
        for k in range(count - 1):
            sequential.append(keys.register(k, k + 1, keys.yaw_hypotheses(k, k + 1, hypotheses), False))
            progress.maybe(k + 1, 5)

        # The person turns one way: steps whose best match turns the other way
        # by more than a few degrees are searched again on the majority side.
        turns = np.array([wrapped_degrees(yaw_of(p.transform)) for p in sequential])
        sense = np.sign(np.sum(np.sign(turns[np.abs(turns) > 5.0]))) or 1.0
        for k, turn in enumerate(turns):
            if np.sign(turn) != sense and abs(turn) > 5.0:
                side = hypotheses[np.sign(hypotheses) == sense]
                candidate = keys.register(k, k + 1, keys.yaw_hypotheses(k, k + 1, side), False)
                if candidate.fitness >= sequential[k].fitness - c.sense_tolerance:
                    info(f"  step {k}->{k + 1}: turn {turn:+.0f} deg against the majority sense, replaced by "
                         f"{wrapped_degrees(yaw_of(candidate.transform)):+.0f} deg")
                    sequential[k] = candidate

        # Skip pairs: k is also registered onto k+2. If the chain k -> k+1 -> k+2
        # disagrees with the direct match, the weaker step is searched again
        # from the start implied by the direct match: a side view that fits a
        # wrong turn is caught by its neighbours.
        skips = []
        info("skip pairs (search within +-40 deg of the chained turn):")
        progress = Progress("skip", max(count - 2, 1))
        for k in range(count - 2):
            expected = wrapped_degrees(yaw_of(sequential[k].transform) + yaw_of(sequential[k + 1].transform))
            direct = keys.register(k, k + 2, keys.yaw_hypotheses(k, k + 2, window(expected, 40.0, c.hypothesis_step)),
                                   True)
            progress.maybe(k + 1, 5)
            skips.append(direct)
            chain = sequential[k + 1].transform @ sequential[k].transform
            disagreement = abs(wrapped_degrees(yaw_of(np.linalg.inv(chain) @ direct.transform)))
            if disagreement <= c.max_correction or direct.fitness < c.min_fitness:
                continue
            weak = k if sequential[k].fitness <= sequential[k + 1].fitness else k + 1
            implied = (np.linalg.inv(sequential[k + 1].transform) @ direct.transform if weak == k
                       else direct.transform @ np.linalg.inv(sequential[k].transform))
            candidate = keys.register(weak, weak + 1, [implied], False)
            if candidate.fitness >= sequential[weak].fitness - c.sense_tolerance:
                info(f"  step {weak}->{weak + 1}: turn {wrapped_degrees(yaw_of(sequential[weak].transform)):+.0f} deg "
                     f"disagrees with the skip match {k}->{k + 2}; replaced by "
                     f"{wrapped_degrees(yaw_of(candidate.transform)):+.0f} deg")
                sequential[weak] = candidate

        overlaps = np.array([p.fitness for p in sequential])
        for pair in sequential:
            if pair.fitness < 0.5 * np.median(overlaps):
                warning(f"weak alignment of keyframes {pair.source}->{pair.target} (overlap {pair.fitness:.2f}, "
                        f"typical {np.median(overlaps):.2f}): turn too large at that step, or the person moved "
                        "during the stop")

        # Turn of every keyframe from all the measured relative turns at once,
        # each weighted by how strongly its geometry constrains the turn.
        chain = np.zeros(count)
        for k, pair in enumerate(sequential):
            chain[k + 1] = chain[k] + yaw_of(pair.transform)
        measurements = list(sequential) + [d for d in skips if d.fitness >= c.min_fitness]
        candidates = [(i, j) for i in range(count) for j in range(i + 3, count)
                      if abs(wrapped_degrees(chain[j] - chain[i])) <= c.loop_max_angle]
        info(f"other overlapping pairs (search within +-20 deg): {len(candidates)}")
        progress = Progress("pairs", max(len(candidates), 1))
        for n, (i, j) in enumerate(candidates):
            expected = wrapped_degrees(chain[j] - chain[i])
            pair = keys.register(i, j, keys.yaw_hypotheses(i, j, window(expected, 20.0, c.hypothesis_step)), True)
            if pair.fitness >= c.min_fitness:
                measurements.append(pair)
            progress.maybe(n + 1, 20)
        yaws = solve_turns(count, chain, measurements, np.radians(c.max_correction / 2.0))
        change = np.degrees(np.diff(yaws) - np.diff(chain))
        for k in np.flatnonzero(np.abs(change) > 3.0):
            info(f"  step {k}->{k + 1}: turn {np.degrees(chain[k + 1] - chain[k]):+.1f} deg corrected to "
                 f"{np.degrees(yaws[k + 1] - yaws[k]):+.1f} deg by the other matches")

        # Model poses: every keyframe turned about its own body axis, the axis
        # moved onto the axis of keyframe 0.
        axes = [np.append(keys.body_axis(k), 0.0) for k in range(count)]
        poses = [yaw_transform(-(yaws[k] - yaws[0]), axes[k], axes[0] - axes[k]) for k in range(count)]

        # Every overlapping pair from the model, with a bounded correction.
        edges, kept_model = [], 0
        # Consecutive keyframes always (a step may exceed loop_max_angle), others when they overlap.
        todo = [(i, j) for i in range(count) for j in range(i + 1, count)
                if j == i + 1 or abs(wrapped_degrees(yaw_of(np.linalg.inv(poses[j]) @ poses[i]))) <= c.loop_max_angle]
        info("refining every overlapping pair from the solved turns:")
        progress = Progress("edges", max(len(todo), 1))
        for n, (i, j) in enumerate(todo):
            progress.maybe(n + 1, 25)
            start = np.linalg.inv(poses[j]) @ poses[i]
            pair = keys.register(i, j, [start], j != i + 1)
            if not self.small_correction(start, pair.transform, keys.points[i].mean(axis=0)):
                pair = keys.pair_at(i, j, start, j != i + 1)
                kept_model += 1
            if j == i + 1 or pair.fitness >= c.min_fitness:
                edges.append(pair)
        poses, edges = self.optimize(poses, edges, keys)

        refined = []
        info("final refinement of the kept pairs:")
        progress = Progress("refine", max(len(edges), 1))
        for n, edge in enumerate(edges):
            progress.maybe(n + 1, 25)
            start = np.linalg.inv(poses[edge.target]) @ poses[edge.source]
            pair = keys.register(edge.source, edge.target, [start], edge.uncertain)
            refined.append(pair if self.small_correction(start, pair.transform, keys.points[edge.source].mean(axis=0))
                           else edge)
        poses, edges = self.optimize(poses, refined, keys)
        loops = sum(1 for e in edges if e.uncertain)
        return KeyframeResult(poses, edges, keys, {"sequential_edges": len(edges) - loops, "loop_closures": loops,
                                                   "model_kept": kept_model})

    def optimize(self, poses, edges, keys):
        graph = registration.PoseGraph()
        for pose in poses:
            graph.nodes.append(registration.PoseGraphNode(pose))
        for edge in edges:
            graph.edges.append(registration.PoseGraphEdge(edge.source, edge.target, edge.transform,
                                                          edge.information, uncertain=edge.uncertain))
        option = registration.GlobalOptimizationOption(max_correspondence_distance=keys.fine,
                                                       edge_prune_threshold=0.25, preference_loop_closure=1.0,
                                                       reference_node=0)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            registration.global_optimization(graph, registration.GlobalOptimizationLevenbergMarquardt(),
                                             registration.GlobalOptimizationConvergenceCriteria(), option)
        kept = {(e.source_node_id, e.target_node_id) for e in graph.edges}
        poses = [np.asarray(node.pose) if self.config.max_tilt > 0 else project_to_4dof(np.asarray(node.pose))
                 for node in graph.nodes]
        return poses, [e for e in edges if (e.source, e.target) in kept]
