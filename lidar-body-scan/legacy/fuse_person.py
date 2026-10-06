"""
Fuse a capture_person.py run into one point cloud of the person.

    python fuse_person.py person_run --crop-min 0.1 0.9 -1.3 --crop-max 1.5 2.3 0.9 --out person

Setting
-------
The LiDAR is fixed; the person turns on the spot by small steps and holds
still in between. Compared with a turntable (fuse_views.py) there are four
extra difficulties, handled as follows.

    Imperfect rotation   The body drifts a few cm at every step and the turning
                         axis wanders. Every view gets its own pose; nothing
                         assumes a fixed axis.
    Steps                Frames recorded while stepping show legs and arms in
                         transit. A per-pixel motion score finds the still
                         periods; only those are used, one keyframe per stop.
    Micro-movements      Breathing and sway (mm to 1 cm). A keyframe is the
                         per-pixel median of up to --keyframe-frames still
                         frames, which also reduces the range noise.
    Non-rigid changes    Arms and legs are not exactly in the same position at
                         every stop. Registration is restricted to what a
                         standing person can do between stops: a turn about
                         the vertical plus a translation (4 degrees of
                         freedom, no tilt), with a robust (Tukey) kernel so
                         that limbs that moved count little. After fusion,
                         points not confirmed by at least --min-views views
                         (ghost limbs seen in one stop only) are removed.

Pipeline
    1. background: per-pixel median range of the empty scene recorded at the
       start of the run; floor plane fitted inside the crop box. The output
       frame has z up from the floor (z = 0 on the floor).
    2. person mask per frame: closer than the background, inside the crop box,
       above the floor, no mixed pixels.
    3. motion score per frame, still segments, one median keyframe per segment.
    4. keyframe k onto k+1: 4-DOF ICP from several yaw hypotheses; majority
       sense of rotation enforced. Loop closures between keyframes that
       overlap (including last turn onto first), pose graph, one refinement.
    5. fusion, multi-view support filter, voxel averaging, normals oriented
       outwards (towards the sensor in each view).

Output (frame: floor z = 0, origin on the floor below the centre of the body,
axes of the first keyframe)
    <out>.ply          fused cloud with normals
    <out>_views.ply    aligned keyframes before averaging, one colour per keyframe
    <out>.json         keyframes, yaw per keyframe, edges, residuals, transforms
    <out>_motion.png   motion score over time with the chosen still segments

Then, for a closed manifold mesh (floor at z = 0, flat soles):
    python pointcloud_to_mesh.py person.ply --method poisson --depth 9 --trim-distance 0 ^
        --watertight 0.004 --clip-below 0 --target-triangles 100000 --out person_mesh.ply

Limits
    * A rigid fusion of a non-rigid body: expect residual blur of about 1 cm on
      the torso and more on hands and feet. The accurate model is a body model
      (SMPL) fitted with one shape and one pose per keyframe.
    * With the sensor at 1.2 m the top of the head, the soles and the underside
      of the arms and chin are never seen; the mesher closes them by invention.
    * Needs the empty-scene background (capture_person.py records it first).
"""

import argparse
import copy
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import open3d as o3d

registration = o3d.pipelines.registration

VERSION = "2026-10-01a (shared helpers for fuse_turntable.py 2026-10-01b)"


# ----------------------------------------------------------------------------
# Run loading
# ----------------------------------------------------------------------------

def natural_key(path):
    parts = re.split(r"(\d+)", Path(path).name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


class Run:
    """A capture_person.py output directory."""

    def __init__(self, directory, min_range, max_range):
        self.directory = Path(directory)
        lut = np.load(self.directory / "lut.npz")
        self.direction = lut["direction"].astype(np.float64)
        self.offset = lut["offset"].astype(np.float64)
        self.min_range, self.max_range = min_range, max_range
        self.background_paths = sorted((self.directory / "background").glob("*.npz"), key=natural_key)
        self.frame_paths = sorted((self.directory / "frames").glob("*.npz"), key=natural_key)
        if not self.frame_paths:
            raise SystemExit(f"no frames in {self.directory / 'frames'}")

    def load(self, path):
        """Return (range [m] with NaN for no return, time [s], columns_ok)."""
        data = np.load(path)
        range_m = data["range"].astype(np.float64) / 1000.0
        valid = (range_m > self.min_range) & (range_m < self.max_range)
        range_m[~valid] = np.nan
        time = float(data["time"]) if "time" in data.files else float("nan")
        columns_ok = float(data["columns_ok"]) if "columns_ok" in data.files else 1.0
        return range_m, time, columns_ok

    def xyz(self, range_m):
        """(H, W, 3) sensor-frame points; NaN where there is no return."""
        return range_m[..., None] * self.direction + self.offset


def median_range(ranges, min_fraction):
    """Per-pixel median over frames; NaN where valid in fewer than min_fraction of them."""
    stack = np.stack(ranges)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(stack, axis=0)
    enough = np.mean(np.isfinite(stack), axis=0) >= min_fraction
    return np.where(enough, median, np.nan)


# ----------------------------------------------------------------------------
# Floor and output frame
# ----------------------------------------------------------------------------

def inside_box(points, box_min, box_max):
    with np.errstate(invalid="ignore"):
        return np.all((points >= box_min) & (points <= box_max), axis=-1)


def fit_floor(run, background_range, args):
    """Floor plane from the empty scene inside the crop box footprint.

    Returns the 4x4 transform world <- sensor: z along the floor normal,
    z = 0 on the floor, origin below the sensor.
    """
    xyz = run.xyz(background_range)
    box_min = np.array(args.crop_min, dtype=np.float64)
    box_max = np.array(args.crop_max, dtype=np.float64)
    footprint = inside_box(xyz[..., :2], box_min[:2] - 0.5, box_max[:2] + 0.5) & np.isfinite(xyz[..., 0])
    points = xyz[footprint]
    points = points[points[:, 2] < np.nanpercentile(points[:, 2], 60)] if len(points) else points
    if len(points) < 200:
        raise SystemExit("too few background points near the crop box to fit the floor")

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    plane, inliers = cloud.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=2000)
    normal = np.array(plane[:3])
    offset = plane[3]
    scale = np.linalg.norm(normal)
    normal, offset = normal / scale, offset / scale
    if normal[2] < 0:
        normal, offset = -normal, -offset
    tilt = np.degrees(np.arccos(np.clip(normal[2], -1, 1)))
    if tilt > 25:
        raise SystemExit(f"the dominant plane near the box is tilted {tilt:.0f} deg: not a floor. "
                         "Check the crop box.")
    height = offset                       # signed distance of the sensor origin from the plane

    # Rotation taking the normal to +z, with the smallest turn.
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, z)
    s = np.linalg.norm(axis)
    rotation = np.eye(3) if s < 1e-12 else o3d.geometry.get_rotation_matrix_from_axis_angle(
        axis / s * np.arctan2(s, normal @ z))
    world_from_sensor = np.eye(4)
    world_from_sensor[:3, :3] = rotation
    world_from_sensor[:3, 3] = [0.0, 0.0, height]
    return world_from_sensor, {"sensor_height_m": float(height), "sensor_tilt_deg": float(tilt),
                               "floor_inliers": len(inliers)}


def transform_points(transform, points):
    return points @ transform[:3, :3].T + transform[:3, 3]


# ----------------------------------------------------------------------------
# Person mask and motion
# ----------------------------------------------------------------------------

def mixed_pixel_mask(range_m, jump):
    """Pixels that break the range profile on both sides (see fuse_views.py)."""
    r = range_m
    left, right = np.roll(r, 1, axis=1), np.roll(r, -1, axis=1)
    up, down = np.full_like(r, np.nan), np.full_like(r, np.nan)
    up[1:], down[:-1] = r[:-1], r[1:]
    with np.errstate(invalid="ignore"):
        rows = (np.abs(r - left) > jump) & (np.abs(r - right) > jump) & (np.abs(left + right - 2 * r) > jump)
        cols = (np.abs(r - up) > jump) & (np.abs(r - down) > jump) & (np.abs(up + down - 2 * r) > jump)
    return rows | cols


class Isolator:
    """Everything needed to cut the person out of a range image."""

    def __init__(self, run, background_range, world_from_sensor, args):
        self.run = run
        self.background = background_range
        self.world_from_sensor = world_from_sensor
        self.args = args
        self.box_min = np.array(args.crop_min, dtype=np.float64)
        self.box_max = np.array(args.crop_max, dtype=np.float64)

    def mask(self, range_m):
        """Person pixels: closer than the empty scene, in the box, above the floor."""
        args = self.args
        margin = np.maximum(args.bg_threshold, args.bg_relative * np.nan_to_num(self.background, nan=0.0))
        with np.errstate(invalid="ignore"):
            closer = (range_m < self.background - margin) | (np.isfinite(range_m) & np.isnan(self.background))
        xyz = self.run.xyz(range_m)
        in_box = inside_box(xyz, self.box_min, self.box_max)
        height = np.einsum("ijk,k->ij", xyz, self.world_from_sensor[2, :3]) + self.world_from_sensor[2, 3]
        with np.errstate(invalid="ignore"):
            above_floor = (height > args.floor_margin) & (height < args.max_height)
        mask = closer & in_box & above_floor
        if args.edge_jump > 0:
            mask &= ~mixed_pixel_mask(range_m, args.edge_jump)
        return mask

    def cloud(self, range_m):
        """World-frame person cloud with normals facing the sensor."""
        args = self.args
        points = transform_points(self.world_from_sensor, self.run.xyz(range_m)[self.mask(range_m)])
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if args.cluster_eps > 0 and len(points) >= 20:
            labels = np.asarray(cloud.cluster_dbscan(eps=args.cluster_eps, min_points=10))
            if labels.size and labels.max() >= 0:
                largest = np.bincount(labels[labels >= 0]).argmax()
                cloud = cloud.select_by_index(np.flatnonzero(labels == largest))
        if len(cloud.points) > 30:
            cloud, _ = cloud.remove_statistical_outlier(20, 2.0)
        if len(cloud.points) >= 3:
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=args.normal_radius, max_nn=30))
            cloud.orient_normals_towards_camera_location(self.world_from_sensor[:3, 3])
        return cloud


def motion_scores(run, isolator, args):
    """Fraction of person pixels that changed between consecutive frames.

    A pixel counts as changed if it belongs to the person in only one of the
    two frames, or if its range moved by more than --motion-threshold.
    Returns scores (score[i] compares frame i with frame i-1), times, the
    number of person pixels per frame, and the columns_ok values.
    """
    scores, times, sizes, columns = [], [], [], []
    previous = None
    for index, path in enumerate(run.frame_paths):
        range_m, time, columns_ok = run.load(path)
        mask = isolator.mask(range_m)
        if previous is None:
            score = np.nan
        else:
            previous_range, previous_mask = previous
            union = mask | previous_mask
            with np.errstate(invalid="ignore"):
                moved = np.abs(range_m - previous_range) > args.motion_threshold
            changed = union & ((mask != previous_mask) | moved)
            score = changed.sum() / max(union.sum(), 1)
        scores.append(score)
        times.append(time)
        sizes.append(int(mask.sum()))
        columns.append(columns_ok)
        previous = (range_m, mask)
        if (index + 1) % 100 == 0:
            print(f"  motion: {index + 1}/{len(run.frame_paths)} frames")
    return np.array(scores), np.array(times), np.array(sizes), np.array(columns)


def still_segments(scores, sizes, columns, args):
    """Runs of frames that differ little from both neighbours.

    The threshold adapts to the run: noise and silhouette flicker set the
    floor of the score, so a frame is still when its score is below
    --still-factor times the 20th percentile (and never above --still-max).
    """
    finite = scores[np.isfinite(scores)]
    base = float(np.percentile(finite, 20)) if finite.size else 0.0
    threshold = min(args.still_max, max(args.still_factor * base, 0.02))

    low = np.nan_to_num(scores, nan=np.inf) < threshold
    still = np.zeros(len(scores), dtype=bool)
    # Still = unchanged with respect to the previous and to the next frame.
    still[:-1] = low[:-1] & low[1:]
    still &= columns >= 0.99
    still &= sizes >= args.min_pixels

    segments, start = [], None
    for index, flag in enumerate(np.append(still, False)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= args.min_still:
                segments.append((start, index))
            start = None
    return segments, threshold


# ----------------------------------------------------------------------------
# 4-DOF registration (turn about the vertical + translation)
# ----------------------------------------------------------------------------

def yaw_transform(angle, pivot, shift=(0.0, 0.0, 0.0)):
    """Rotate by 'angle' about the vertical line through 'pivot', then shift."""
    c, s = np.cos(angle), np.sin(angle)
    transform = np.eye(4)
    transform[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    pivot = np.asarray(pivot, dtype=np.float64)
    transform[:3, 3] = pivot - transform[:3, :3] @ pivot + np.asarray(shift)
    return transform


def yaw_of(transform):
    return float(np.arctan2(transform[1, 0], transform[0, 0]))


def project_to_4dof(transform):
    """Drop any residual tilt: keep the yaw and the translation."""
    result = yaw_transform(yaw_of(transform), (0.0, 0.0, 0.0))
    result[:3, 3] = transform[:3, 3]
    return result


class Target:
    """Registration target: points, normals and a nearest-neighbour index."""

    def __init__(self, cloud):
        self.points = np.asarray(cloud.points)
        self.normals = np.asarray(cloud.normals)
        self.index = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(self.points))
        self.index.knn_index()

    def nearest(self, query):
        indices, squared = self.index.knn_search(o3d.core.Tensor(query), 1)
        return indices.numpy()[:, 0], np.sqrt(squared.numpy()[:, 0])


def tilt_degrees(transform):
    """Angle between the transformed vertical and the vertical."""
    return float(np.degrees(np.arccos(np.clip(transform[2, 2], -1.0, 1.0))))


def icp_robust(source, target, initial, distances=(0.06, 0.035, 0.02), iterations=(30, 25, 20),
               max_tilt_deg=0.0):
    """Point-to-plane ICP with a Tukey kernel, 4 or 6 degrees of freedom.

    max_tilt_deg = 0: turn about the vertical and a 3D shift (4 DOF).
    max_tilt_deg > 0: the body may also tilt, up to that angle in total. A
    standing person sways like an inverted pendulum about the ankles and
    rarely stands with the same lean at two stops; at head height 1 degree
    of lean is 3 cm, far more than the sensor noise, so a registration
    without tilt cannot align head and feet at the same time.

    The robust kernel down-weights points whose residual is large, i.e.
    limbs that moved between the stops.
    """
    transform = initial.copy()
    max_turn = np.radians(3.0)
    free_tilt = max_tilt_deg > 0
    for distance, count in zip(distances, iterations):
        for _ in range(count):
            moved = transform_points(transform, source)
            index, gap = target.nearest(moved)
            use = gap < distance
            if use.sum() < max(30, 0.1 * len(source)):
                break                     # lost the surface: a wrong start, stop here
            p, q, n = moved[use], target.points[index[use]], target.normals[index[use]]
            residual = np.einsum("ij,ij->i", n, p - q)
            weight = np.where(np.abs(residual) < distance, (1.0 - (residual / distance) ** 2) ** 2, 0.0)
            center = p.mean(axis=0)
            lever = p - center
            # d/dw of n . (w x lever) = (lever x n) . w ; columns: w_x, w_y, w_z, t_x, t_y, t_z
            jacobian = np.column_stack([np.cross(lever, n), n])
            columns = [0, 1, 2, 3, 4, 5] if free_tilt else [2, 3, 4, 5]
            reduced = jacobian[:, columns]
            normal_matrix = reduced.T @ (reduced * weight[:, None]) + 1e-9 * np.eye(len(columns))
            solution = np.linalg.solve(normal_matrix, -reduced.T @ (weight * residual))
            step = np.zeros(6)
            step[columns] = solution
            # Trust region: a poorly constrained yaw (a side view is almost a
            # cylinder) must not jump across the body in one iteration.
            step[:3] = np.clip(step[:3], -max_turn, max_turn)
            length = np.linalg.norm(step[3:])
            if length > 0.5 * distance:
                step[3:] *= 0.5 * distance / length
            update = np.eye(4)
            update[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(step[:3])
            update[:3, 3] = center - update[:3, :3] @ center + step[3:]
            candidate = update @ transform
            if free_tilt and tilt_degrees(candidate) > max_tilt_deg:
                step[:2] = 0.0            # tilt bound reached: continue without tilting further
                update[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(step[:3])
                update[:3, 3] = center - update[:3, :3] @ center + step[3:]
                candidate = update @ transform
            transform = candidate
            if np.linalg.norm(step[:3]) < 1e-5 and np.linalg.norm(step[3:]) < 1e-5:
                break
    return transform


def icp_4dof(source, target, initial, distances=(0.06, 0.035, 0.02), iterations=(30, 25, 20)):
    return icp_robust(source, target, initial, distances, iterations, 0.0)


def evaluate(source, target, transform, distance):
    """Overlap (fraction of source points within 'distance') and point-to-plane RMSE [m]."""
    moved = transform_points(transform, source)
    index, gap = target.nearest(moved)
    use = gap < distance
    if not use.any():
        return 0.0, float("inf")
    residual = np.einsum("ij,ij->i", target.normals[index[use]], moved[use] - target.points[index[use]])
    return float(use.mean()), float(np.sqrt(np.mean(residual ** 2)))


class Pair:
    def __init__(self, source, target, transform, fitness, rmse, uncertain):
        self.source, self.target = source, target
        self.transform = transform
        self.fitness, self.rmse = fitness, rmse
        self.uncertain = uncertain
        self.information = None


class Keyframes:
    """Registration copies of the keyframe clouds."""

    def __init__(self, clouds, voxel, fine_distance, body_radius=0.10, max_tilt=0.0):
        self.body_radius = body_radius
        self.max_tilt = max_tilt
        self.clouds = [c.voxel_down_sample(voxel) for c in clouds]
        self.points = [np.asarray(c.points) for c in self.clouds]
        self.targets = [Target(c) for c in self.clouds]
        self.fine = fine_distance

    def register(self, i, j, starts, uncertain):
        """Best alignment of keyframe i onto keyframe j among the starts.

        Every start is refined with 4 DOF (turn + shift), which is robust far
        from the answer; the three best are then refined with tilt allowed
        (if --max-tilt > 0), which is accurate close to the answer.
        """
        results = []
        for start in starts:
            transform = icp_4dof(self.points[i], self.targets[j], start)
            fitness, rmse = evaluate(self.points[i], self.targets[j], transform, self.fine)
            results.append((fitness, -rmse, transform))
        results.sort(key=lambda item: (item[0], item[1]), reverse=True)
        candidates = [transform for _, _, transform in results[:3]]
        if self.max_tilt > 0:
            candidates = [icp_robust(self.points[i], self.targets[j], t, (0.04, 0.025, 0.015),
                                     (25, 20, 20), self.max_tilt) for t in candidates] + candidates

        best = None
        for transform in candidates:
            fitness, rmse = evaluate(self.points[i], self.targets[j], transform, self.fine)
            better = (best is None or fitness > best.fitness + 0.02
                      or (fitness > best.fitness - 0.02 and rmse < best.rmse))
            if better:
                best = Pair(i, j, transform, fitness, rmse, uncertain)
        best.information = registration.get_information_matrix_from_point_clouds(
            self.clouds[i], self.clouds[j], self.fine, best.transform)
        return best

    def body_axis(self, i):
        """Horizontal position of the body's vertical axis in keyframe i.

        The visible surface of a standing person lies on the sensor side of the
        axis, so the centroid of the visible points is pulled towards the
        sensor by about the body radius. The axis is estimated by moving the
        centroid of the torso-height points (0.4 to 1.4 m, arms included) away
        from the sensor by --body-radius. Error: a few cm, independent of how
        far the person walked between stops, which is what matters here.
        """
        points = self.points[i]
        band = points[(points[:, 2] > 0.4) & (points[:, 2] < 1.4)]
        center = (band if len(band) > 50 else points)[:, :2].mean(axis=0)
        distance = np.linalg.norm(center)
        return center + self.body_radius * center / max(distance, 1e-6)

    def yaw_hypotheses(self, i, j, yaws_deg):
        """Starts for the alignment of keyframe i onto j, for each yaw:
        (a) turn about the estimated body axis of i and move that axis onto the
        axis of j: handles a person who stepped away from the mark;
        (b) turn about the centroid of i, no shift: a person who turned on the spot."""
        axis_i, axis_j = self.body_axis(i), self.body_axis(j)
        starts = []
        for yaw in yaws_deg:
            starts.append(yaw_transform(np.radians(yaw), np.append(axis_i, 0.0),
                                        np.append(axis_j - axis_i, 0.0)))
            starts.append(yaw_transform(np.radians(yaw), self.points[i].mean(axis=0)))
        return starts


def correction_is_small(start, result, center, args):
    correction = np.linalg.inv(start) @ result
    turn = abs(np.degrees(yaw_of(correction)))
    shift = np.linalg.norm(transform_points(correction, center[None])[0] - center)
    return (turn <= args.max_correction and shift <= args.max_shift
            and tilt_degrees(correction) <= max(args.max_tilt, 0.5))


def wrapped_degrees(angle):
    return (np.degrees(angle) + 180.0) % 360.0 - 180.0


class Progress:
    """One line per step with the elapsed time, so that a long stage is visibly alive."""

    def __init__(self, label, total):
        self.label, self.total, self.start = label, total, time.time()

    def step(self, done):
        elapsed = time.time() - self.start
        remaining = elapsed / done * (self.total - done) if done else 0.0
        print(f"  {self.label}: {done}/{self.total}  elapsed {elapsed:5.0f} s  remaining about {remaining:5.0f} s",
              flush=True)


def window(expected_deg, half_width, step):
    """Yaw hypotheses [deg] around an expected turn."""
    return expected_deg + np.arange(-half_width, half_width + 0.1, step)


def register_keyframes(keys, args):
    count = len(keys.points)
    hypotheses = np.arange(-args.max_step, args.max_step + 0.1, args.hypothesis_step)

    # Sequential: k onto k+1, free yaw search.
    print(f"consecutive pairs (full turn search, {len(hypotheses)} turns x 2 starts each):")
    progress = Progress("consecutive", count - 1)
    sequential = []
    for k in range(count - 1):
        sequential.append(keys.register(k, k + 1, keys.yaw_hypotheses(k, k + 1, hypotheses), False))
        if (k + 1) % 5 == 0 or k == count - 2:
            progress.step(k + 1)

    # The person turns one way. Steps whose best match turns the other way by
    # more than a few degrees are re-searched on the majority side only.
    # (Pair k maps keyframe k onto k+1, so its yaw is the person's turn.)
    turns = np.array([wrapped_degrees(yaw_of(p.transform)) for p in sequential])
    sense = np.sign(np.sum(np.sign(turns[np.abs(turns) > 5.0]))) or 1.0
    for k, turn in enumerate(turns):
        if np.sign(turn) != sense and abs(turn) > 5.0:
            side = hypotheses[np.sign(hypotheses) == sense]
            candidate = keys.register(k, k + 1, keys.yaw_hypotheses(k, k + 1, side), False)
            if candidate.fitness >= sequential[k].fitness - args.sense_tolerance:
                print(f"  step {k}->{k + 1}: turn {turn:+.0f} deg against the majority sense, "
                      f"replaced by {wrapped_degrees(yaw_of(candidate.transform)):+.0f} deg")
                sequential[k] = candidate

    # Consistency check with skip edges: keyframe k is also registered
    # directly onto k+2 (free search). If the chain k -> k+1 -> k+2 disagrees
    # with the direct match, the weaker of the two steps (lower overlap) is
    # redone from the start implied by the direct match and the other step.
    # A single side view that fits a wrong turn is thereby caught by its
    # neighbours. The skip edges also enter the pose graph.
    skips = []
    print("skip pairs (search within +-40 deg of the chained turn):")
    progress = Progress("skip", count - 2)
    for k in range(count - 2):
        expected = wrapped_degrees(yaw_of(sequential[k].transform) + yaw_of(sequential[k + 1].transform))
        direct = keys.register(k, k + 2, keys.yaw_hypotheses(k, k + 2, window(expected, 40.0, args.hypothesis_step)),
                               True)
        if (k + 1) % 5 == 0 or k == count - 3:
            progress.step(k + 1)
        skips.append(direct)
        chain = sequential[k + 1].transform @ sequential[k].transform
        disagreement = abs(wrapped_degrees(yaw_of(np.linalg.inv(chain) @ direct.transform)))
        if disagreement <= args.max_correction or direct.fitness < args.min_fitness:
            continue
        weak = k if sequential[k].fitness <= sequential[k + 1].fitness else k + 1
        implied = (np.linalg.inv(sequential[k + 1].transform) @ direct.transform if weak == k
                   else direct.transform @ np.linalg.inv(sequential[k].transform))
        candidate = keys.register(weak, weak + 1, [implied], False)
        if candidate.fitness >= sequential[weak].fitness - args.sense_tolerance:
            print(f"  step {weak}->{weak + 1}: turn {wrapped_degrees(yaw_of(sequential[weak].transform)):+.0f} deg "
                  f"disagrees with the skip match {k}->{k + 2}; replaced by "
                  f"{wrapped_degrees(yaw_of(candidate.transform)):+.0f} deg")
            sequential[weak] = candidate

    overlaps = np.array([p.fitness for p in sequential])
    for pair in sequential:
        if pair.fitness < 0.5 * np.median(overlaps):
            print(f"  warning: weak alignment keyframes {pair.source}->{pair.target} "
                  f"(overlap {pair.fitness:.2f}, typical {np.median(overlaps):.2f}): "
                  "turn too large at that step, or the person moved during the stop")

    # Global turn angles. Chaining the steps accumulates every error, and a
    # side view (a profile) constrains the turn poorly: it can fit a turn
    # that is 15 degrees off with a good overlap. So the turn of every
    # keyframe is solved from all measured relative turns at once
    # (consecutive, skip, and pairs that close the full turn), each weighted
    # by how strongly its geometry constrains the turn (the rotational
    # information about the vertical). The loop error then goes to the
    # weakly constrained steps instead of being spread evenly. A Huber
    # reweighting reduces the influence of gross outliers.
    chain = np.zeros(count)
    for k, pair in enumerate(sequential):
        chain[k + 1] = chain[k] + yaw_of(pair.transform)

    measurements = list(sequential) + [d for d in skips if d.fitness >= args.min_fitness]
    # The chained turn drifts by a few degrees over a few steps; the search is
    # centred on it. Pairs across the full turn (chain difference near 360)
    # are added by the same rule, as the wrapped angle is small there.
    candidates = [(i, j) for i in range(count) for j in range(i + 3, count)
                  if abs(wrapped_degrees(chain[j] - chain[i])) <= args.loop_max_angle]
    print(f"other overlapping pairs (search within +-20 deg): {len(candidates)}")
    progress = Progress("pairs", len(candidates))
    for n, (i, j) in enumerate(candidates):
        expected = wrapped_degrees(chain[j] - chain[i])
        pair = keys.register(i, j, keys.yaw_hypotheses(i, j, window(expected, 20.0, args.hypothesis_step)), True)
        if pair.fitness >= args.min_fitness:
            measurements.append(pair)
        if (n + 1) % 20 == 0 or n == len(candidates) - 1:
            progress.step(n + 1)
    yaws = solve_turns(count, chain, measurements, args)
    change = np.degrees(np.diff(yaws) - np.diff(chain))
    for k in np.flatnonzero(np.abs(change) > 3.0):
        print(f"  step {k}->{k + 1}: turn {np.degrees(chain[k + 1] - chain[k]):+.1f} deg corrected to "
              f"{np.degrees(yaws[k + 1] - yaws[k]):+.1f} deg by the other matches")

    # Model poses from the solved turns: each keyframe turned about its own
    # body axis, the axis moved onto the axis of keyframe 0.
    axes = [np.append(keys.body_axis(k), 0.0) for k in range(count)]
    # pose_k maps keyframe k into keyframe 0: turn by -(yaw_k - yaw_0).
    poses = [yaw_transform(-(yaws[k] - yaws[0]), axes[k], axes[0] - axes[k]) for k in range(count)]

    # Edges between all overlapping keyframes, each starting from the model
    # and allowed only a bounded correction (shift, tilt, a few degrees of
    # turn); a larger correction keeps the model.
    edges, rejected = [], 0
    todo = [(i, j) for i in range(count) for j in range(i + 1, count)
            if abs(wrapped_degrees(yaw_of(np.linalg.inv(poses[j]) @ poses[i]))) <= args.loop_max_angle]
    print("refining every overlapping pair from the solved turns:")
    progress = Progress("edges", len(todo))
    for n, (i, j) in enumerate(todo):
        if (n + 1) % 25 == 0 or n == len(todo) - 1:
            progress.step(n + 1)
        start = np.linalg.inv(poses[j]) @ poses[i]
        pair = keys.register(i, j, [start], j != i + 1)
        center = keys.points[i].mean(axis=0)
        if not correction_is_small(start, pair.transform, center, args):
            fitness, rmse = evaluate(keys.points[i], keys.targets[j], start, keys.fine)
            pair = Pair(i, j, start, fitness, rmse, j != i + 1)
            pair.information = registration.get_information_matrix_from_point_clouds(
                keys.clouds[i], keys.clouds[j], keys.fine, start)
            rejected += 1
        if j == i + 1 or pair.fitness >= args.min_fitness:
            edges.append(pair)

    poses, edges = optimize(poses, edges, keys, args)

    # One refinement from the optimized poses.
    refined = []
    print("final refinement of the kept pairs:")
    progress = Progress("refine", len(edges))
    for n, edge in enumerate(edges):
        if (n + 1) % 25 == 0 or n == len(edges) - 1:
            progress.step(n + 1)
        start = np.linalg.inv(poses[edge.target]) @ poses[edge.source]
        pair = keys.register(edge.source, edge.target, [start], edge.uncertain)
        center = keys.points[edge.source].mean(axis=0)
        refined.append(pair if correction_is_small(start, pair.transform, center, args) else edge)
    poses, edges = optimize(poses, refined, keys, args)
    loops = sum(1 for e in edges if e.uncertain)
    return poses, edges, {"sequential_edges": len(edges) - loops, "loop_closures": loops,
                          "model_kept": rejected}


def solve_turns(count, chain, measurements, args, iterations=5):
    """Least-squares turn angles from relative turn measurements (radians).

    Each measurement i -> j observes yaw_j - yaw_i. Its angle is unwrapped
    near the chained value, so pairs across the full turn close the loop.
    Weight: rotational information about the vertical, times a Huber factor.
    """
    rows = []
    for pair in measurements:
        i, j = pair.source, pair.target
        expected = chain[j] - chain[i]
        observed = expected + np.radians(wrapped_degrees(yaw_of(pair.transform) - expected))
        information = max(float(pair.information[2, 2]), 1e-9)
        rows.append((i, j, observed, information))
    base = np.array([r[3] for r in rows])
    base = base / np.median(base)
    weights = base.copy()
    yaws = chain.copy()
    huber = np.radians(args.max_correction / 2.0)
    for _ in range(iterations):
        a = np.zeros((len(rows) + 1, count))
        b = np.zeros(len(rows) + 1)
        for r, (i, j, observed, _) in enumerate(rows):
            w = np.sqrt(weights[r])
            a[r, j], a[r, i], b[r] = w, -w, w * observed
        a[-1, 0], b[-1] = 1e3, 0.0                      # yaw of keyframe 0 fixed at 0
        yaws = np.linalg.lstsq(a, b, rcond=None)[0]
        residual = np.array([abs((yaws[j] - yaws[i]) - observed) for i, j, observed, _ in rows])
        weights = base * np.minimum(1.0, huber / np.maximum(residual, 1e-9))
    return yaws


def optimize(poses, edges, keys, args):
    graph = registration.PoseGraph()
    for pose in poses:
        graph.nodes.append(registration.PoseGraphNode(pose))
    for edge in edges:
        graph.edges.append(registration.PoseGraphEdge(edge.source, edge.target, edge.transform,
                                                      edge.information, uncertain=edge.uncertain))
    option = registration.GlobalOptimizationOption(max_correspondence_distance=keys.fine,
                                                   edge_prune_threshold=0.25,
                                                   preference_loop_closure=1.0, reference_node=0)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        registration.global_optimization(graph, registration.GlobalOptimizationLevenbergMarquardt(),
                                         registration.GlobalOptimizationConvergenceCriteria(), option)
    kept = {(e.source_node_id, e.target_node_id) for e in graph.edges}
    poses = [np.asarray(node.pose) if args.max_tilt > 0 else project_to_4dof(np.asarray(node.pose))
             for node in graph.nodes]
    return poses, [e for e in edges if (e.source, e.target) in kept]


# ----------------------------------------------------------------------------
# Fusion
# ----------------------------------------------------------------------------

def view_colors(count):
    hues = np.linspace(0.0, 1.0, count, endpoint=False)
    k = (np.array([5.0, 3.0, 1.0])[None, :] + hues[:, None] * 6.0) % 6.0
    return 1.0 - np.clip(np.minimum(k, 4.0 - k), 0.0, 1.0)


def support_filter(points, labels, radius, min_views):
    """Keep points that have neighbours from at least 'min_views' different keyframes
    (the point's own keyframe included) within 'radius'."""
    if min_views <= 1:
        return np.ones(len(points), dtype=bool)
    index = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(points))
    index.hybrid_index(radius)
    keep = np.zeros(len(points), dtype=bool)
    chunk = 250000                                     # bounded memory for runs with many views
    for begin in range(0, len(points), chunk):
        query = points[begin:begin + chunk]
        neighbours, _, _ = index.hybrid_search(o3d.core.Tensor(query), radius, 48)
        neighbours = neighbours.numpy()
        neighbour_labels = np.where(neighbours >= 0, labels[np.clip(neighbours, 0, None)], -1)
        ordered = np.sort(neighbour_labels, axis=1)
        distinct = (ordered[:, :1] >= 0).astype(int)[:, 0] + np.sum(
            (ordered[:, 1:] != ordered[:, :-1]) & (ordered[:, 1:] >= 0), axis=1)
        keep[begin:begin + chunk] = distinct >= min_views
    return keep


def reference_cloud(cloud, voxel=0.008, max_points=400000, seed=0):
    """Registration reference from the union of many noisy views, with a fixed
    point budget.

    A voxel subsampling keeps every voxel that any point reaches, so the
    reference is a shell as thick as the range noise, and the more points go
    in, the further out into the noise tails it reaches; on a convex body the
    outer tail has more voxels, so views aligned to a reference built from
    several laps were pulled outward (2 to 3 mm in simulation). A random
    subset of at most 'max_points' (about one lap of views) keeps the
    distribution and the shell the same whatever the number of laps."""
    if len(cloud.points) > max_points:
        keep = np.random.default_rng(seed).choice(len(cloud.points), max_points, replace=False)
        cloud = cloud.select_by_index(np.sort(keep))
    return cloud.voxel_down_sample(voxel)


def slab_refine(clouds, args, groups=None, clean_reference=False, reference_views=None):
    """Non-rigid correction of the aligned keyframes, slice by slice in height.

    A person is not rigid between stops: the lean changes (a lean pivots
    about the ankles, so the shift grows with height), the head turns, the
    hips shift. Each keyframe is cut into horizontal slabs (--slab-height,
    every --slab-step) and every slab gets its own small turn about the
    vertical and horizontal shift, aligning it to all the other keyframes
    together (their union is the reference). The corrections are bounded,
    smoothed along the height and interpolated linearly, so the body stays
    continuous. All keyframes are corrected against the same state
    (no order bias); --slab-iterations passes.

    groups: optional group label per cloud. Without it the reference of a
    cloud is all the others (leave one out, cost quadratic in the number of
    clouds); with it, all the clouds of the other groups (one reference per
    group, for runs with hundreds of views). clean_reference: build the
    reference with reference_cloud (a fixed point budget of
    'reference_views' average views; needed when the views of several laps
    overlap, see there).
    Returns the corrected clouds and the RMS correction per pass [mm].
    """
    clouds = [copy.deepcopy(c) for c in clouds]
    history = []
    budget = None
    if clean_reference and reference_views:
        budget = int(np.mean([len(c.points) for c in clouds]) * reference_views)
    for iteration in range(args.slab_iterations):
        corrections = []
        progress = Progress(f"slab pass {iteration + 1}", len(clouds))
        targets = {}
        for k in range(len(clouds)):
            if (k + 1) % 10 == 0 or k == len(clouds) - 1:
                progress.step(k + 1)
            key = k if groups is None else groups[k]
            if key not in targets:
                reference = o3d.geometry.PointCloud()
                for other, cloud in enumerate(clouds):
                    if (other != k) if groups is None else (groups[other] != key):
                        reference += cloud
                if groups is None:
                    targets.clear()                      # leave one out: one reference per cloud
                targets[key] = Target(reference_cloud(reference, max_points=budget or 400000) if clean_reference
                                      else reference.voxel_down_sample(0.008))
            target = targets[key]
            points = np.asarray(clouds[k].points)
            top = points[:, 2].max()
            centers = np.arange(args.slab_step, top + 1e-9, args.slab_step)
            table = np.full((len(centers), 5), np.nan)     # turn, pivot x, pivot y, shift x, shift y
            for n, center in enumerate(centers):
                inside = np.abs(points[:, 2] - center) <= args.slab_height / 2
                if inside.sum() < 60:
                    continue
                slab = points[inside]
                pivot = slab.mean(axis=0)
                result = icp_robust(slab, target, np.eye(4), (0.04, 0.025, 0.015), (20, 15, 15), 0.0)
                turn = yaw_of(result)
                moved = transform_points(result, pivot[None])[0]
                shift = moved[:2] - pivot[:2]
                if abs(np.degrees(turn)) <= args.slab_max_turn and np.linalg.norm(shift) <= args.slab_max_shift:
                    table[n] = [turn, pivot[0], pivot[1], shift[0], shift[1]]
            valid = np.isfinite(table[:, 0])
            if valid.sum() < 2:
                corrections.append(None)
                continue
            for c in range(5):                           # fill gaps, then smooth over 3 slabs
                table[:, c] = np.interp(centers, centers[valid], table[valid, c])
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
            values = [np.interp(points[:, 2], centers, table[:, c]) for c in range(5)]
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
        history.append(1000 * float(np.sqrt(np.mean(np.concatenate(displacement) ** 2))) if displacement else 0.0)
    return clouds, history


def fuse(clouds, poses, args):
    views = o3d.geometry.PointCloud()
    labels = []
    for k, (cloud, pose, color) in enumerate(zip(clouds, poses, view_colors(len(clouds)))):
        aligned = copy.deepcopy(cloud).transform(pose)
        aligned.paint_uniform_color(color)
        views += aligned
        labels.append(np.full(len(aligned.points), k))
    labels = np.concatenate(labels)

    keep = support_filter(np.asarray(views.points), labels, args.support_radius, args.min_views)
    supported = views.select_by_index(np.flatnonzero(keep))
    fused = supported.voxel_down_sample(args.voxel) if args.voxel > 0 else copy.deepcopy(supported)
    fused.normalize_normals()
    if len(fused.points) > 30:
        fused, _ = fused.remove_statistical_outlier(20, 2.0)
    fused.colors = o3d.utility.Vector3dVector()
    return fused, views, float(1.0 - keep.mean())


def top_view(path, views):
    """Horizontal slice at torso height of the aligned keyframes, coloured by keyframe order.
    A correct fusion is one closed ring; separate rings mean misplaced keyframes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    points, colors = np.asarray(views.points), np.asarray(views.colors)
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axis, (low, high) in zip(axes, [(1.0, 1.15), (0.3, 0.45)]):
        keep = (points[:, 2] > low) & (points[:, 2] < high)
        axis.scatter(points[keep, 0], points[keep, 1], s=1, c=colors[keep])
        axis.set_aspect("equal")
        axis.set_title(f"top view, height {low:.2f} to {high:.2f} m (colour = keyframe order)")
    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)


def motion_plot(path, times, scores, threshold, segments, used):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    figure, axis = plt.subplots(figsize=(11, 3.2))
    for n, (start, stop) in enumerate(segments):
        color = "tab:green" if n in used else "tab:gray"
        axis.axvspan(times[start], times[stop - 1], color=color, alpha=0.25, lw=0)
    axis.plot(times, scores, color="black", lw=0.8)
    axis.axhline(threshold, color="tab:red", lw=0.8, ls="--")
    axis.set_xlabel("time since capture start [s]")
    axis.set_ylabel("changed person pixels")
    axis.set_title("motion score; green = still segments used as keyframes, red = threshold")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return True


def to_builtin(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fuse a person turning in place into one point cloud.")
    parser.add_argument("run", help="capture_person.py output directory")
    parser.add_argument("--out", default="person", help="output base name")
    parser.add_argument("--crop-min", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                        help="capture region, sensor frame [m]; include the arms and the drift of the feet")
    parser.add_argument("--crop-max", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--min-range", type=float, default=0.3)
    parser.add_argument("--max-range", type=float, default=10.0)

    iso = parser.add_argument_group("person isolation")
    iso.add_argument("--bg-threshold", type=float, default=0.05, help="closer than the empty scene by [m]")
    iso.add_argument("--bg-relative", type=float, default=0.01, help="... or by this fraction of the range")
    iso.add_argument("--floor-margin", type=float, default=0.025, help="drop points below this height [m]")
    iso.add_argument("--max-height", type=float, default=2.3, help="[m] above the floor")
    iso.add_argument("--edge-jump", type=float, default=0.05, help="mixed pixel filter [m], 0 = off")
    iso.add_argument("--cluster-eps", type=float, default=0.06, help="largest cluster [m], 0 = off")
    iso.add_argument("--normal-radius", type=float, default=0.05)

    still = parser.add_argument_group("still periods and keyframes")
    still.add_argument("--motion-threshold", type=float, default=0.03,
                       help="range change that counts as motion for a pixel [m]")
    still.add_argument("--still-factor", type=float, default=2.5,
                       help="still if the score is below this times its 20th percentile")
    still.add_argument("--still-max", type=float, default=0.25, help="upper bound of the still threshold")
    still.add_argument("--min-still", type=int, default=5, help="shortest still segment [frames]")
    still.add_argument("--keyframe-frames", type=int, default=20,
                       help="frames (centre of each still segment) in the per-pixel median; 20 frames = 2 s, "
                            "about half a breath, so every keyframe shows the same mid-breath chest")
    still.add_argument("--min-pixels", type=int, default=800, help="frames with fewer person pixels are ignored")
    still.add_argument("--keyframe-stride", type=int, default=1,
                       help="use every n-th still segment; the processing time grows with the square "
                            "of the number of keyframes, and beyond about 24 per turn-and-a-half "
                            "the extra ones add little (2 halves the time to a quarter)")
    still.add_argument("--skip-seconds", type=float, default=0.0, help="ignore the start of the capture [s]")

    reg = parser.add_argument_group("registration")
    reg.add_argument("--reg-voxel", type=float, default=0.012, help="[m]")
    reg.add_argument("--body-radius", type=float, default=0.10,
                     help="assumed distance from the visible surface to the body axis [m]")
    reg.add_argument("--fine-distance", type=float, default=0.025, help="overlap distance [m]")
    reg.add_argument("--max-step", type=float, default=90.0, help="largest turn between stops searched [deg]")
    reg.add_argument("--hypothesis-step", type=float, default=10.0, help="yaw search spacing [deg]")
    reg.add_argument("--sense-tolerance", type=float, default=0.10,
                     help="overlap a majority-sense match may lose and still replace a reversed one")
    reg.add_argument("--loop-max-angle", type=float, default=60.0, help="[deg]")
    reg.add_argument("--min-fitness", type=float, default=0.3)
    reg.add_argument("--max-correction", type=float, default=10.0, help="loop closure turn bound [deg]")
    reg.add_argument("--max-shift", type=float, default=0.10, help="loop closure shift bound [m]")
    reg.add_argument("--max-tilt", type=float, default=5.0,
                     help="largest lean between two stops [deg]; 0 = turn and shift only (4 DOF)")

    slab = parser.add_argument_group("non-rigid correction (lean, head, hips)")
    slab.add_argument("--slab-iterations", type=int, default=2, help="0 = rigid fusion only")
    slab.add_argument("--slab-height", type=float, default=0.30, help="[m]")
    slab.add_argument("--slab-step", type=float, default=0.10, help="[m]")
    slab.add_argument("--slab-max-turn", type=float, default=10.0, help="[deg]")
    slab.add_argument("--slab-max-shift", type=float, default=0.05, help="[m]")

    out = parser.add_argument_group("fusion")
    out.add_argument("--min-views", type=int, default=2,
                     help="keep points confirmed by this many keyframes (1 = off)")
    out.add_argument("--support-radius", type=float, default=0.02, help="[m]")
    out.add_argument("--voxel", type=float, default=0.005, help="[m], 0 = off")
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(f"fuse_person.py version {VERSION}")
    run = Run(args.run, args.min_range, args.max_range)
    if not run.background_paths:
        raise SystemExit("the run has no background frames; record them (capture_person.py does it first)")

    # 1. Background and floor.
    background = median_range([run.load(p)[0] for p in run.background_paths], 0.5)
    world_from_sensor, floor = fit_floor(run, background, args)
    print(f"background: {len(run.background_paths)} frames; floor: sensor {floor['sensor_height_m']:.3f} m "
          f"above it, tilt {floor['sensor_tilt_deg']:.1f} deg")
    isolator = Isolator(run, background, world_from_sensor, args)

    # 2-3. Motion, still segments, keyframes.
    scores, times, sizes, columns = motion_scores(run, isolator, args)
    if args.skip_seconds > 0:
        sizes = np.where(times < args.skip_seconds, 0, sizes)
    segments, threshold = still_segments(scores, sizes, columns, args)
    print(f"frames: {len(run.frame_paths)}, person pixels median {int(np.median(sizes))}, "
          f"still threshold {threshold:.3f}, still segments {len(segments)}")
    if len(segments) < 3:
        raise SystemExit("fewer than 3 still segments. Check the crop box and the motion plot, or "
                         "raise --still-max / --still-factor, or hold still longer at each stop.")

    clouds, keyframes, used = [], [], []
    for n, (start, stop) in enumerate(segments):
        if n % args.keyframe_stride:
            continue
        middle = (start + stop) // 2
        half = args.keyframe_frames // 2
        chosen = range(max(start, middle - half), min(stop, middle - half + args.keyframe_frames))
        range_m = median_range([run.load(run.frame_paths[i])[0] for i in chosen], 0.6)
        cloud = isolator.cloud(range_m)
        if len(cloud.points) < args.min_pixels // 2:
            print(f"  segment {n} ({times[start]:.1f}-{times[stop - 1]:.1f} s) dropped: "
                  f"{len(cloud.points)} points")
            continue
        clouds.append(cloud)
        used.append(n)
        keyframes.append({"segment": n, "frames": [int(chosen[0]), int(chosen[-1])],
                          "time_s": [float(times[start]), float(times[stop - 1])],
                          "points": len(cloud.points)})
    print(f"keyframes: {len(clouds)} ({', '.join(str(k['points']) for k in keyframes)} points)")
    if len(clouds) < 3:
        raise SystemExit("fewer than 3 usable keyframes")

    # 4. Registration.
    keys = Keyframes(clouds, args.reg_voxel, args.fine_distance, args.body_radius, args.max_tilt)
    poses, edges, stats = register_keyframes(keys, args)
    yaws = np.unwrap([yaw_of(p) for p in poses])
    turn = np.degrees(yaws - yaws[0])
    if turn[-1] < 0:
        turn = -turn
    print(f"registration: {stats['sequential_edges']} consecutive edges, {stats['loop_closures']} other edges, "
          f"{stats['model_kept']} edges kept at the solved-turn model")
    print(f"turn per keyframe [deg]: {' '.join(f'{a:.0f}' for a in turn)}")
    coverage = float(turn.max() - turn.min())
    if coverage < 330:
        print(f"  warning: the keyframes cover {coverage:.0f} deg only; part of the body was never seen")

    print("step table (keyframe pair: turn, walked distance of the body axis, overlap):")
    for edge in edges:
        if edge.target != edge.source + 1:
            continue
        walked = np.linalg.norm(keys.body_axis(edge.target) - keys.body_axis(edge.source))
        print(f"  {edge.source:2d}->{edge.target:2d}  turn {turn[edge.target] - turn[edge.source]:+6.1f} deg  "
              f"walked {100 * walked:5.1f} cm  overlap {edge.fitness:.2f}")

    # 5. Fusion.
    aligned = [copy.deepcopy(cloud).transform(pose) for cloud, pose in zip(clouds, poses)]
    if args.slab_iterations > 0:
        aligned, history = slab_refine(aligned, args)
        print(f"non-rigid slab correction: RMS {', '.join(f'{h:.1f}' for h in history)} mm per pass")
    fused, views, removed = fuse(aligned, [np.eye(4)] * len(aligned), args)
    pieces = np.asarray(fused.voxel_down_sample(0.02).cluster_dbscan(eps=0.05, min_points=5))
    if pieces.size and pieces.max() >= 0:
        sizes = np.bincount(pieces[pieces >= 0])
        # An arm held away from the body can be a separate piece of a few
        # percent; a misplaced group of keyframes is a large second body.
        big = int(np.sum(sizes > 0.25 * sizes.max()))
        if big > 1:
            print(f"  WARNING: the fused cloud has {big} separate large pieces (a person is one). "
                  f"Some keyframes are placed wrongly: open {args.out}_views.ply and {args.out}_top.png; "
                  f"a chain break usually sits at a step with low overlap in the table above.")
    points = np.asarray(fused.points)
    center = np.median(points[:, :2], axis=0)
    output_from_world = np.eye(4)
    output_from_world[:2, 3] = -center
    fused.transform(output_from_world)
    views.transform(output_from_world)
    height = float(np.asarray(fused.points)[:, 2].max())

    top_view(f"{args.out}_top.png", views)
    o3d.io.write_point_cloud(f"{args.out}.ply", fused)
    o3d.io.write_point_cloud(f"{args.out}_views.ply", views)
    plotted = motion_plot(f"{args.out}_motion.png", times, scores, threshold, segments, set(used))

    quality = []
    for edge in edges:
        relative = np.linalg.inv(poses[edge.target]) @ poses[edge.source]
        fitness, rmse = evaluate(keys.points[edge.source], keys.targets[edge.target], relative, keys.fine)
        quality.append({"source": edge.source, "target": edge.target, "loop": edge.uncertain,
                        "fitness": round(fitness, 3), "rmse_mm": round(1000 * rmse, 2)})
    sequential = [q for q in quality if q["target"] == q["source"] + 1]
    rmse = np.array([q["rmse_mm"] for q in quality])

    report = {
        "version": VERSION, "run": str(Path(args.run).resolve()), "floor": floor,
        "still_threshold": threshold, "segments": [[int(a), int(b)] for a, b in segments],
        "keyframes": keyframes, "turn_deg": turn, "coverage_deg": coverage,
        "poses_output_frame": [output_from_world @ p for p in poses],
        "world_from_sensor": world_from_sensor, "output_from_world": output_from_world,
        "edges": quality, "registration": stats,
        "support_filter_removed_fraction": removed,
        "fused_points": len(fused.points), "highest_point_m": height,
        "parameters": vars(args),
    }
    Path(f"{args.out}.json").write_text(json.dumps(report, indent=2, default=to_builtin))

    print(f"alignment: sequential overlap {min(q['fitness'] for q in sequential):.2f} to "
          f"{max(q['fitness'] for q in sequential):.2f}, point-to-plane RMSE median {np.median(rmse):.1f} mm, "
          f"max {rmse.max():.1f} mm")
    print(f"support filter removed {100 * removed:.1f}% of the points (seen in fewer than {args.min_views} keyframes)")
    print(f"fused: {len(fused.points)} points, highest point {height:.3f} m above the floor")
    print(f"wrote {args.out}.ply, {args.out}_views.ply, {args.out}.json, {args.out}_top.png"
          + (f", {args.out}_motion.png" if plotted else ""))


if __name__ == "__main__":
    main()
