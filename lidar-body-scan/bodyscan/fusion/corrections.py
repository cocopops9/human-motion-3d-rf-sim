"""Per-view corrections: small, bounded motions that a rigid platform turn cannot describe.

Every correction is a ViewCorrection: apply(clouds, groups, reference_views)
returns the corrected clouds and a report. A pipeline holds a list of them,
so a new correction (for example per leg) is one more subclass in the list.

    SwayCorrection     rigid, bounded (the whole body sways on the platform)
    ViewAngleSearch    turn of every view about the axis, searched (off by default)
    SlabCorrection     turn and shift per horizontal slab (lean, head turn, hips)
    LimbCorrection     each free-hanging arm on its own (arms sink and swing)

Every correction is made against the views of the other groups (leave one
group out), with a fixed point budget per reference.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import open3d as o3d

from bodyscan.config import param
from bodyscan.geometry import (Target, components_2d, reference_cloud, tilt_degrees, transform_points, yaw_of,
                               yaw_transform)
from bodyscan.log import Progress, info
from bodyscan.registration import icp_robust


@dataclass
class ViewConfig:
    """Views and the whole-view corrections."""
    view_step: float = param(3.0, "one view every this many degrees of turn", unit="deg",
                             effect="smaller: more views (more points per surface patch), slower")
    max_views: int = param(400, "the step grows for long rotations so that at most this many views are used")
    reference_groups: int = param(9, "views split into this many groups at random; each view is corrected "
                                     "against the views of the other groups")
    axis_iterations: int = param(1, "refinements of the axis position from the view corrections (0 = off)")
    max_tilt: float = param(2.0, "sway correction bound: tilt", unit="deg")
    max_turn_correction: float = param(3.0, "sway correction bound: turn", unit="deg")
    max_shift: float = param(0.03, "sway correction bound: shift", unit="m")
    max_dropped_views: float = param(0.25, "views whose correction is out of bounds are dropped if they are at "
                                           "most this fraction of all views (otherwise kept, with a warning)")
    view_range: tuple[float, float] | None = param(None, "fuse only the views between these angles of the "
                                                         "measured turn [deg], e.g. 0 360 for the first lap")
    angle_iterations: int = param(0, "passes of a per-view search of the platform angle against the other "
                                     "views (off: the motion model is already better, about 1 deg rms)")
    angle_search: float = param(15.0, "largest deviation from the motion model searched per view", unit="deg")
    angle_truncation: float = param(0.03, "distance cap of the alignment cost of the angle search", unit="m")


@dataclass
class SlabConfig:
    """Non-rigid correction by horizontal slabs (lean, head, hips)."""
    slab_iterations: int = param(1, "passes (0 = off)")
    slab_height: float = param(0.30, "height of a slab", unit="m")
    slab_step: float = param(0.10, "spacing of the slab centres", unit="m")
    slab_max_turn: float = param(5.0, "bound of a slab turn", unit="deg")
    slab_max_shift: float = param(0.03, "bound of a slab shift", unit="m")


@dataclass
class LimbConfig:
    """Per-arm correction of every view (A-pose: arms held away from the body)."""
    limb_iterations: int = param(2, "passes of the per-arm correction (0 = off)")
    limb_min_height: float = param(0.45, "arms are searched above this height", unit="m")
    limb_min_offset: float = param(0.20, "a part of a horizontal slice farther than this from the torso axis is "
                                         "an arm", unit="m")
    limb_capture: float = param(0.04, "view points within this of an arm of the union take its label", unit="m")
    limb_max_turn: float = param(12.0, "bound of the arm correction", unit="deg")
    limb_max_shift: float = param(0.06, "bound of the arm correction", unit="m")
    limb_blend: float = param(0.10, "the arm correction fades out over this length below the shoulder", unit="m")


class ViewCorrection(ABC):
    name = "correction"

    @abstractmethod
    def apply(self, clouds: list, groups=None, reference_views=None):
        """Returns (corrected clouds, report)."""


def _leave_out_target(targets, reference_all, k, groups, budget, voxel=None):
    """Registration target of view k: every other view (groups None) or the
    views of the other groups, built once per group."""
    key = k if groups is None else int(groups[k])
    if key not in targets:
        reference = o3d.geometry.PointCloud()
        for other, cloud in enumerate(reference_all):
            if (other != k) if groups is None else (groups[other] != key):
                reference += cloud
        if groups is None:
            targets.clear()
        targets[key] = Target(reference_cloud(reference, max_points=budget) if voxel is None
                              else reference.voxel_down_sample(voxel))
    return targets[key]


class SwayCorrection(ViewCorrection):
    """Small bounded rigid correction of every view against the others (the
    person sways a little on the platform). Report: RMS correction [mm] and,
    per view, the horizontal shift of its centroid (NaN where out of bounds)."""
    name = "sway"

    def __init__(self, config: ViewConfig, label="view corrections"):
        self.config, self.label = config, label

    def apply(self, clouds, groups=None, reference_views=None):
        c = self.config
        clouds = [copy.deepcopy(cloud) for cloud in clouds]
        moved = []
        shifts = np.full((len(clouds), 2), np.nan)
        progress = Progress(self.label, len(clouds))
        reference_all = [cloud.voxel_down_sample(0.01) for cloud in clouds]
        budget = int(np.mean([len(cloud.points) for cloud in reference_all]) * (reference_views or len(clouds)))
        targets = {}
        for k in range(len(clouds)):
            target = _leave_out_target(targets, reference_all, k, groups, budget)
            points = np.asarray(reference_all[k].points)
            result = icp_robust(points, target, np.eye(4), (0.04, 0.025, 0.015), (20, 15, 15), c.max_tilt)
            center = points.mean(axis=0)
            shift = np.linalg.norm(transform_points(result, center[None])[0] - center)
            if abs(np.degrees(yaw_of(result))) <= c.max_turn_correction and shift <= c.max_shift \
                    and tilt_degrees(result) <= c.max_tilt:
                clouds[k].transform(result)
                moved.append(shift)
                shifts[k] = transform_points(result, center[None])[0, :2] - center[:2]
            if (k + 1) % 25 == 0 or k == len(clouds) - 1:
                progress.step(k + 1)
        rms = 1000 * float(np.sqrt(np.mean(np.square(moved)))) if moved else 0.0
        return clouds, {"rms_mm": rms, "shifts": shifts}


def axis_error_from_shifts(shifts, angles):
    """Axis error e (xy) from the per-view corrections.

    A view turned back by -theta about an axis that is off by e is displaced by
    (R(theta)^T - I) e; the consensus of all views over a full turn is
    displaced by -e, so the correction applied to view k is -R(theta_k)^T e.
    Least squares over the views gives e (sway averages out). Only valid
    when the view angles go round the body (one lap or more); None otherwise."""
    angles = np.asarray(angles, dtype=np.float64)
    if abs(np.mean(np.exp(1j * angles))) > 0.3:
        return None                    # the views do not go round the body: the consensus is not centred
    rows, rhs = [], []
    for shift, angle in zip(shifts, angles):
        if not np.all(np.isfinite(shift)):
            continue
        c, s = np.cos(angle), np.sin(angle)
        rotation_t = np.array([[c, s], [-s, c]])
        rows.append(-rotation_t)
        rhs.append(shift)
    if len(rows) < 8:
        return None
    return np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]


class ViewAngleSearch(ViewCorrection):
    """Turn of every view about the platform axis, measured against the views
    of the other groups (1 degree of freedom, searched over +-angle_search,
    then refined), angle_iterations passes. Report: correction per view [rad]
    (the refined platform angle is the model angle minus it)."""
    name = "angle search"

    def __init__(self, config: ViewConfig, pivot):
        self.config = config
        self.pivot = np.asarray(pivot, dtype=np.float64)

    def apply(self, clouds, groups=None, reference_views=None):
        c = self.config
        clouds = [copy.deepcopy(cloud) for cloud in clouds]
        pivot = self.pivot
        total = np.zeros(len(clouds))
        coarse = np.radians(np.arange(-c.angle_search, c.angle_search + 1e-9, 1.0))
        fine_steps = np.radians(np.arange(-1.0, 1.0001, 0.25))
        rng = np.random.default_rng(0)
        for iteration in range(c.angle_iterations):
            reference_all = [cloud.voxel_down_sample(0.01) for cloud in clouds]
            budget = int(np.mean([len(cloud.points) for cloud in reference_all]) * (reference_views or len(clouds)))
            targets = {}
            deltas = np.zeros(len(clouds))
            reliable = np.ones(len(clouds), dtype=bool)
            progress = Progress(f"angle pass {iteration + 1}", len(clouds))
            for k in range(len(clouds)):
                target = _leave_out_target(targets, reference_all, k, groups, budget)
                points = np.asarray(reference_all[k].points)
                if len(points) < 200:
                    reliable[k] = False
                    continue
                if len(points) > 3000:
                    points = points[rng.choice(len(points), 3000, replace=False)]
                relative = points[:, :2] - pivot[:2]

                def cost(delta):
                    cos, sin = np.cos(delta), np.sin(delta)
                    moved = points.copy()
                    moved[:, 0] = pivot[0] + cos * relative[:, 0] - sin * relative[:, 1]
                    moved[:, 1] = pivot[1] + sin * relative[:, 0] + cos * relative[:, 1]
                    _, gap = target.nearest(moved)
                    return float(np.mean(np.minimum(gap, c.angle_truncation) ** 2))

                costs = np.array([cost(d) for d in coarse])
                best = int(np.argmin(costs))
                if best in (0, len(coarse) - 1) or costs[best] > 0.9 * np.median(costs):
                    reliable[k] = False          # at the edge of the search, or no clear minimum
                    continue
                trial = coarse[best] + fine_steps
                fine = np.array([cost(d) for d in trial])
                m = int(np.clip(np.argmin(fine), 1, len(fine) - 2))
                a, b, cc = fine[m - 1], fine[m], fine[m + 1]
                curvature = a - 2 * b + cc
                step = 0.5 * (a - cc) / curvature if curvature > 0 else 0.0
                deltas[k] = trial[m] + np.clip(step, -1.0, 1.0) * (fine_steps[1] - fine_steps[0])
                if (k + 1) % 25 == 0 or k == len(clouds) - 1:
                    progress.step(k + 1)
            # a view far from its neighbours in time is a failed search: use their median
            for k in np.flatnonzero(reliable):
                neighbours = [n for n in range(max(0, k - 3), min(len(clouds), k + 4)) if n != k and reliable[n]]
                if len(neighbours) >= 3 and abs(deltas[k] - np.median(deltas[neighbours])) > np.radians(3.0):
                    deltas[k] = np.median(deltas[neighbours])
            for k in np.flatnonzero(~reliable):
                neighbours = [n for n in range(max(0, k - 3), min(len(clouds), k + 4)) if reliable[n]]
                deltas[k] = np.median(deltas[neighbours]) if neighbours else 0.0
            # One view's search has about 1 deg of noise (a body is nearly a
            # cylinder seen from one side); the platform angle changes smoothly
            # from view to view: a running median over 5 views keeps the trend.
            half = 2
            padded = np.pad(deltas, half, mode="edge")
            deltas = np.array([np.median(padded[k:k + 2 * half + 1]) for k in range(len(deltas))])
            for k, delta in enumerate(deltas):
                if delta != 0.0:
                    clouds[k].transform(yaw_transform(delta, pivot))
            total += deltas
            info(f"  angle pass {iteration + 1}: correction rms {np.degrees(np.sqrt(np.mean(deltas ** 2))):.2f} deg, "
                 f"max {np.degrees(np.abs(deltas).max()):.2f} deg; {int((~reliable).sum())} views without a clear "
                 f"minimum (their neighbours' value used)")
        return clouds, {"deltas": total}


class SlabCorrection(ViewCorrection):
    """Non-rigid correction, slice by slice in height.

    A person is not rigid: the lean changes (a lean pivots about the ankles,
    so the shift grows with height), the head turns, the hips shift. Each
    view is cut into horizontal slabs (slab_height, every slab_step) and
    every slab gets its own small turn about the vertical and horizontal
    shift, aligning it to the other views together. The corrections are
    bounded, smoothed along the height and interpolated linearly, so the body
    stays continuous. All views are corrected against the same state (no
    order bias). clean_reference: build each reference with a fixed point
    budget (needed when the views of several laps overlap). Report: RMS
    correction per pass [mm]."""
    name = "slabs"

    def __init__(self, config: SlabConfig, clean_reference: bool = True):
        self.config, self.clean_reference = config, clean_reference

    def apply(self, clouds, groups=None, reference_views=None):
        c = self.config
        clouds = [copy.deepcopy(cloud) for cloud in clouds]
        history = []
        budget = None
        if self.clean_reference and reference_views:
            budget = int(np.mean([len(cloud.points) for cloud in clouds]) * reference_views)
        for iteration in range(c.slab_iterations):
            corrections = []
            progress = Progress(f"slab pass {iteration + 1}", len(clouds))
            targets = {}
            for k in range(len(clouds)):
                if (k + 1) % 10 == 0 or k == len(clouds) - 1:
                    progress.step(k + 1)
                if self.clean_reference:
                    target = _leave_out_target(targets, clouds, k, groups, budget or 400000)
                else:
                    target = _leave_out_target(targets, clouds, k, groups, None, voxel=0.008)
                points = np.asarray(clouds[k].points)
                top = points[:, 2].max()
                centers = np.arange(c.slab_step, top + 1e-9, c.slab_step)
                table = np.full((len(centers), 5), np.nan)     # turn, pivot x, pivot y, shift x, shift y
                for n, center in enumerate(centers):
                    inside = np.abs(points[:, 2] - center) <= c.slab_height / 2
                    if inside.sum() < 60:
                        continue
                    slab = points[inside]
                    pivot = slab.mean(axis=0)
                    result = icp_robust(slab, target, np.eye(4), (0.04, 0.025, 0.015), (20, 15, 15), 0.0)
                    turn = yaw_of(result)
                    moved = transform_points(result, pivot[None])[0]
                    shift = moved[:2] - pivot[:2]
                    if abs(np.degrees(turn)) <= c.slab_max_turn and np.linalg.norm(shift) <= c.slab_max_shift:
                        table[n] = [turn, pivot[0], pivot[1], shift[0], shift[1]]
                valid = np.isfinite(table[:, 0])
                if valid.sum() < 2:
                    corrections.append(None)
                    continue
                for column in range(5):                    # fill gaps, then smooth over 3 slabs
                    table[:, column] = np.interp(centers, centers[valid], table[valid, column])
                kernel = np.array([0.25, 0.5, 0.25])
                padded = np.vstack([table[:1], table, table[-1:]])
                table = kernel[0] * padded[:-2] + kernel[1] * padded[1:-1] + kernel[2] * padded[2:]
                corrections.append((centers, table))

            displacement = []
            for cloud, correction in zip(clouds, corrections):
                if correction is None:
                    continue
                centers, table = correction
                points, normals = np.asarray(cloud.points), np.asarray(cloud.normals)
                values = [np.interp(points[:, 2], centers, table[:, column]) for column in range(5)]
                turn, pivot = values[0], np.column_stack([values[1], values[2]])
                shift = np.column_stack([values[3], values[4]])
                cos, sin = np.cos(turn), np.sin(turn)
                relative = points[:, :2] - pivot
                new_xy = np.column_stack([cos * relative[:, 0] - sin * relative[:, 1],
                                          sin * relative[:, 0] + cos * relative[:, 1]]) + pivot + shift
                displacement.append(np.linalg.norm(new_xy - points[:, :2], axis=1))
                new_normals = np.column_stack([cos * normals[:, 0] - sin * normals[:, 1],
                                               sin * normals[:, 0] + cos * normals[:, 1], normals[:, 2]])
                cloud.points = o3d.utility.Vector3dVector(np.column_stack([new_xy, points[:, 2]]))
                cloud.normals = o3d.utility.Vector3dVector(new_normals)
            history.append(1000 * float(np.sqrt(np.mean(np.concatenate(displacement) ** 2)))
                           if displacement else 0.0)
        return clouds, {"rms_mm_per_pass": history}


def segment_arms(points, min_height, min_offset):
    """Arms of the consensus body when they hang free of the torso (A-pose).

    In every 2 cm horizontal slab between min_height and the top of the
    torso, the points split into connected components (2.5 cm); a component
    whose centroid is farther than min_offset from the torso axis is part of
    an arm, and belongs to the arm on its side of the body (sign along the
    lateral axis of the torso: the main axis of the chest slab). Returns a
    label per point (0 body, 1 and 2 the arms) and, per arm, the height of
    the highest slab where it is separate from the torso (it joins the
    shoulder above it); None when the arms touch the body."""
    labels = np.zeros(len(points), dtype=np.int64)
    chest = points[(points[:, 2] > 1.0) & (points[:, 2] < 1.4)]
    if len(chest) < 200:
        return labels, None
    axis_xy = np.median(chest[:, :2], axis=0)
    centered = chest[:, :2] - chest[:, :2].mean(axis=0)
    lateral = np.linalg.svd(centered, full_matrices=False)[2][0]
    tops = {1: None, 2: None}
    for low in np.arange(min_height, 1.6, 0.02):
        inside = np.flatnonzero((points[:, 2] >= low) & (points[:, 2] < low + 0.02))
        if len(inside) < 50:
            continue
        component = components_2d(points[inside, :2], 0.025)
        ids, counts = np.unique(component, return_counts=True)
        if len(ids) < 2:
            continue
        for i, count in zip(ids, counts):
            if count < 15:
                continue
            members = inside[component == i]
            centroid = points[members, :2].mean(axis=0)
            if np.linalg.norm(centroid - axis_xy) < min_offset:
                continue
            side = 1 if (centroid - axis_xy) @ lateral > 0 else 2
            labels[members] = side
            tops[side] = low + 0.02 if tops[side] is None else max(tops[side], low + 0.02)
    if tops[1] is None and tops[2] is None:
        return labels, None
    return labels, tops


def blend_transform(points, transform, weights):
    """x + w (T x - x): full correction where w = 1, none where w = 0."""
    moved = transform_points(transform, points)
    return points + weights[:, None] * (moved - points)


class LimbCorrection(ViewCorrection):
    """Rigid correction of each free-hanging arm of every view against the
    arms of the other views (held away from the body for minutes, the arms
    sink, swing and turn; the whole-body and slab corrections follow the
    torso, so the views of a drifting arm do not overlap, and the fused arm
    comes out thinner or blurred).

    The arms are segmented on the union of the views; every view point near
    an arm takes its label; the arm points of the view are registered (6
    degrees of freedom, bounded) to the arm of the views of the other groups,
    and the correction fades out over limb_blend below the shoulder, so the
    arm stays attached. Report: shoulder heights and hand displacement per view."""
    name = "limbs"

    def __init__(self, config: LimbConfig):
        self.config = config

    def apply(self, clouds, groups=None, reference_views=None):
        c = self.config
        clouds = [copy.deepcopy(cloud) for cloud in clouds]
        report = []
        tops = None
        for iteration in range(c.limb_iterations):
            union = o3d.geometry.PointCloud()
            owner = []
            for k, cloud in enumerate(clouds):
                down = cloud.voxel_down_sample(0.008)
                union += down
                owner.append(np.full(len(down.points), k))
            owner = np.concatenate(owner)
            union_points = np.asarray(union.points)
            labels, tops = segment_arms(union_points, c.limb_min_height, c.limb_min_offset)
            if tops is None:
                info("  arms not separate from the body (arms down along the torso?): no arm correction")
                return clouds, None
            index = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(union_points))
            index.knn_index()
            budget = None
            if reference_views:
                budget = int(np.mean([len(cloud.points) for cloud in clouds]) * reference_views)
            targets = {}
            sides = [side for side in (1, 2) if tops[side] is not None and (labels == side).sum() >= 100]
            arm_tip = {side: union_points[labels == side][np.argmin(union_points[labels == side][:, 2])]
                       for side in sides}
            chest_y = float(np.median(union_points[(union_points[:, 2] > 1.0) & (union_points[:, 2] < 1.4), 1]))
            names = {side: ("arm at +y" if arm_tip[side][1] > chest_y else "arm at -y") for side in sides}
            hand_moves = {1: np.full(len(clouds), np.nan), 2: np.full(len(clouds), np.nan)}
            progress = Progress(f"arm pass {iteration + 1}", len(clouds))
            for k, cloud in enumerate(clouds):
                points, normals = np.asarray(cloud.points), np.asarray(cloud.normals)
                found, squared = index.knn_search(o3d.core.Tensor(points), 1)
                nearest = labels[found.numpy()[:, 0]]
                near = np.sqrt(squared.numpy()[:, 0]) < c.limb_capture
                key = k if groups is None else int(groups[k])
                for side in sides:
                    own = near & (nearest == side)
                    own &= points[:, 2] < tops[side]
                    if own.sum() < 100:
                        continue
                    target_key = (key, side)
                    if target_key not in targets:
                        others = (labels == side) & ((owner != k) if groups is None else (groups[owner] != key))
                        reference = union.select_by_index(np.flatnonzero(others))
                        if budget:
                            reference = reference_cloud(reference, max_points=max(budget // 4, 20000))
                        if len(reference.points) < 100:
                            continue
                        if groups is None:
                            targets = {}
                        targets[target_key] = Target(reference)
                    source = points[own]
                    result = icp_robust(source, targets[target_key], np.eye(4), (0.05, 0.03, 0.015),
                                        (20, 15, 15), c.limb_max_turn)
                    angle = float(np.degrees(np.arccos(np.clip((np.trace(result[:3, :3]) - 1) / 2, -1.0, 1.0))))
                    centroid = source.mean(axis=0)
                    shift = float(np.linalg.norm(transform_points(result, centroid[None])[0] - centroid))
                    if angle > c.limb_max_turn or shift > c.limb_max_shift:
                        continue
                    weights = np.clip((tops[side] - points[own, 2]) / c.limb_blend, 0.0, 1.0)
                    points[own] = blend_transform(source, result, weights)
                    rotated = normals[own] @ result[:3, :3].T
                    mixed = normals[own] + weights[:, None] * (rotated - normals[own])
                    normals[own] = mixed / np.maximum(np.linalg.norm(mixed, axis=1, keepdims=True), 1e-12)
                    hand_moves[side][k] = float(np.linalg.norm(transform_points(result, arm_tip[side][None])[0]
                                                               - arm_tip[side]))
                cloud.points = o3d.utility.Vector3dVector(points)
                cloud.normals = o3d.utility.Vector3dVector(normals)
                if (k + 1) % 25 == 0 or k == len(clouds) - 1:
                    progress.step(k + 1)
            report.append({side: hand_moves[side] for side in sides})
            summary = []
            for side in sides:
                moves = hand_moves[side]
                ok = np.isfinite(moves)
                if ok.any():
                    summary.append(f"{names[side]}: corrected in {ok.sum()} of {len(moves)} views, hand moved by "
                                   f"median {1000 * np.median(moves[ok]):.0f} mm, "
                                   f"p90 {1000 * np.percentile(moves[ok], 90):.0f} mm")
            info(f"  arm pass {iteration + 1}: " + "; ".join(summary))
        return clouds, {"tops": tops, "hand_moves": report}
