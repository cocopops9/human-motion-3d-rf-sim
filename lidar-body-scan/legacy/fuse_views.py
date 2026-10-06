"""
Fuse several views of a rotating object into a single point cloud.

Setting
-------
The LiDAR is fixed and the object turns between frames (turntable). In the
sensor frame the background is therefore static and only the object moves.
The script

    1. isolates the object in every frame: subtraction of an empty-scene
       capture, optional crop box, removal of mixed "edge" pixels, largest
       connected cluster, outlier removal;
    2. registers every view to the first one: several initial guesses
       (turntable model, FPFH feature matching, identity), point-to-plane ICP
       between neighbouring views and between any two views that overlap
       (loop closures), then a pose graph optimization that spreads the
       residual drift over the whole loop;
    3. fits the turntable rotation axis to the recovered poses. The residuals
       tell how far the recovered motion is from a pure rotation, and the
       fitted axis is a calibration that later runs can reuse (--axis-json),
       which is needed for symmetric or featureless objects;
    4. merges the aligned views, averages them on a voxel grid and writes the
       fused cloud with normals. Each normal is oriented towards the sensor in
       its own view, which is outward for the object, so the orientation is
       correct without any guessing.

Output
------
    <out>.ply         fused cloud (points + normals) in the frame of the first view
    <out>_views.ply   aligned views before averaging, one colour per view
    <out>.json        poses, registration edges, fitted axis, angle per view

Examples
--------
    Frames from ouster_extract.py, empty scene recorded separately:
        python fuse_views.py turntable_frames --background empty_frames --out fused

    Known step between captures and an axis calibrated in a previous run:
        python fuse_views.py turntable_frames --background empty_frames ^
            --step-deg 20 --axis-json calibration.json --out fused

    Then build the mesh:
        python pointcloud_to_mesh.py fused.ply --method poisson

Capture protocol
----------------
    1. Sensor fixed. Record the empty scene with the turntable in place
       (20 to 50 frames): ouster_extract.py --out empty_frames --frames 30
    2. Place the object, stop the table at each step (10 to 30 degrees) and
       record a few frames per stop; use one frame per stop (or --every).
    3. Do not move the sensor between 1 and 2: the background model is per pixel.

Limits
------
    * Without --background, static structures (turntable, floor) dominate
      the registration and the fusion fails; a crop box alone is not enough.
    * Rotationally symmetric objects (cylinder, sphere) cannot be registered
      from geometry alone: give the turntable axis and the angles.
    * Frames captured while the table turns are distorted during the 0.1 s
      sweep (no deskewing here). Capture with the table stopped, or turn slowly.
"""

import argparse
import copy
import glob
import json
import re
import warnings
from pathlib import Path

import numpy as np
import open3d as o3d

registration = o3d.pipelines.registration

VERSION = "2026-09-29b (numeric file order)"
CLOUD_SUFFIXES = (".npz", ".ply", ".pcd", ".xyz")


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------

def natural_key(path):
    """Sort key that orders numbers by value: view2 before view10."""
    parts = re.split(r"(\d+)", Path(path).name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def expand_inputs(items):
    """Accept files, directories and glob patterns (PowerShell does not expand globs).

    Files are ordered by the numbers in their names (view2 before view10), since
    consecutive files must be neighbouring rotations. A directory written by
    ouster_extract.py may hold both frame_XXXXX.npz and
    frame_XXXXX.ply; the .npz files are preferred because they keep the
    organized (H, W) structure.
    """
    paths = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            files = sorted((p for p in path.iterdir() if p.suffix in CLOUD_SUFFIXES), key=natural_key)
            npz_files = [p for p in files if p.suffix == ".npz"]
            paths.extend(npz_files if npz_files else files)
        elif any(char in item for char in "*?["):
            paths.extend(sorted((Path(p) for p in glob.glob(item)), key=natural_key))
        else:
            paths.append(path)
    return paths


class Frame:
    """One view: valid points, plus the organized (H, W) grid when available."""

    def __init__(self, name, points, grid=None, valid=None, timestamp=None):
        self.name = name
        self.points = points          # (N, 3) valid points, meters
        self.grid = grid              # (H, W, 3) or None
        self.valid = valid            # (H, W) bool or None
        self.timestamp = timestamp    # seconds, or None

    @property
    def is_organized(self):
        return self.grid is not None

    def ranges(self):
        return np.linalg.norm(self.grid, axis=2)


def load_frame(path, min_range, max_range):
    path = Path(path)
    timestamp = None

    if path.suffix == ".npz":
        data = np.load(path)
        if "timestamps" in data.files:
            stamps = np.asarray(data["timestamps"], dtype=np.float64)
            stamps = stamps[stamps > 0]
            if stamps.size:
                timestamp = float(np.median(stamps)) * 1e-9

        if "xyz" in data.files and data["xyz"].ndim == 3:
            grid = data["xyz"].astype(np.float64)
            ranges = np.linalg.norm(grid, axis=2)
            valid = (ranges > min_range) & (ranges < max_range)
            return Frame(path.name, grid[valid], grid, valid, timestamp)
        if "points" not in data.files:
            raise ValueError(f"{path} has neither an 'xyz' grid nor a 'points' array")
        points = data["points"].astype(np.float64)
    else:
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points)

    ranges = np.linalg.norm(points, axis=1)
    keep = (ranges > min_range) & (ranges < max_range)
    return Frame(path.name, points[keep], timestamp=timestamp)


# ----------------------------------------------------------------------------
# Object isolation
# ----------------------------------------------------------------------------

class Background:
    """Empty-scene model: per-pixel median range, and the points themselves."""

    def __init__(self, frames):
        organized = [frame for frame in frames if frame.is_organized]
        self.ranges = None
        if organized:
            stack = np.stack([np.where(f.valid, f.ranges(), np.nan) for f in organized])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN pixels
                self.ranges = np.nanmedian(stack, axis=0)

        points = np.concatenate([frame.points for frame in frames])
        self.cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def foreground_mask(frame, background, args):
    """Pixels whose range is clearly shorter than in the empty scene.

    Pixels where the empty scene had no return are also kept (nothing was
    there before, so a return now must come from the object).
    """
    ranges = frame.ranges()
    if background.ranges.shape != ranges.shape:
        raise ValueError(f"{frame.name}: background and frame have different resolutions")

    margin = np.maximum(args.bg_threshold, args.bg_relative * background.ranges)
    with np.errstate(invalid="ignore"):
        closer = ranges < background.ranges - margin
    return frame.valid & (closer | np.isnan(background.ranges))


def mixed_pixel_mask(frame, jump):
    """Pixels that break the range profile along a row or along a column.

    At a silhouette the laser footprint falls partly on the object and partly
    on the background, and the returned range lies somewhere in between
    ("mixed" or "flying" pixels). Such a pixel differs from the neighbours on
    both sides, while a genuine boundary pixel differs only from the
    background side, so the object is not eroded. A surface seen at a grazing
    angle also changes range quickly from pixel to pixel, but steadily, so a
    pixel is flagged only when the second difference is large as well.
    Mixed pixels that fall almost exactly half way between the two surfaces
    look like a slope and survive; the outlier filter usually removes them.
    Structures one pixel thin are removed.
    """
    r = np.where(frame.valid, frame.ranges(), np.nan)
    left = np.roll(r, 1, axis=1)          # columns cover 360 degrees
    right = np.roll(r, -1, axis=1)
    up = np.full_like(r, np.nan)
    down = np.full_like(r, np.nan)
    up[1:] = r[:-1]
    down[:-1] = r[1:]

    with np.errstate(invalid="ignore"):
        along_row = ((np.abs(r - left) > jump) & (np.abs(r - right) > jump)
                     & (np.abs(left + right - 2.0 * r) > jump))
        along_column = ((np.abs(r - up) > jump) & (np.abs(r - down) > jump)
                        & (np.abs(up + down - 2.0 * r) > jump))
    return along_row | along_column


def inside_box(points, box_min, box_max):
    return np.all((points >= box_min) & (points <= box_max), axis=1)


def largest_cluster(cloud, eps, min_points):
    labels = np.asarray(cloud.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.size == 0 or labels.max() < 0:
        return o3d.geometry.PointCloud()
    largest = np.bincount(labels[labels >= 0]).argmax()
    return cloud.select_by_index(np.flatnonzero(labels == largest))


def isolate_object(frame, background, args):
    """Return the object points of one frame, with normals facing the sensor."""
    pixel_background = background is not None and background.ranges is not None and frame.is_organized

    if frame.is_organized:
        mask = frame.valid.copy()
        if pixel_background:
            mask &= foreground_mask(frame, background, args)
        if args.edge_jump > 0:
            mask &= ~mixed_pixel_mask(frame, args.edge_jump)
        points = frame.grid[mask]
    else:
        points = frame.points

    if args.crop_min is not None and args.crop_max is not None:
        points = points[inside_box(points, np.asarray(args.crop_min), np.asarray(args.crop_max))]

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    if background is not None and not pixel_background and len(cloud.points) > 0:
        distances = np.asarray(cloud.compute_point_cloud_distance(background.cloud))
        cloud = cloud.select_by_index(np.flatnonzero(distances > args.bg_threshold))

    if args.cluster_eps > 0 and len(cloud.points) >= args.cluster_min_points:
        cloud = largest_cluster(cloud, args.cluster_eps, args.cluster_min_points)

    if args.outlier_neighbors > 0 and len(cloud.points) > args.outlier_neighbors:
        cloud, _ = cloud.remove_statistical_outlier(args.outlier_neighbors, args.outlier_std)

    if len(cloud.points) >= 3:
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=args.normal_radius, max_nn=30))
        # The sensor sits at the origin of every frame, so this is exact.
        cloud.orient_normals_towards_camera_location(np.zeros(3))
    return cloud


# ----------------------------------------------------------------------------
# Rigid transforms and the turntable model
# ----------------------------------------------------------------------------

def rotation_about_axis(point, direction, angle):
    """4x4 rotation by 'angle' (radians) about the line through 'point' along 'direction'."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    k = np.array([[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]])
    rotation = np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(point) - rotation @ np.asarray(point)
    return transform


def rotation_angle(transform):
    """Magnitude of the rotation of a 4x4 transform, radians."""
    cosine = (np.trace(transform[:3, :3]) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def axis_and_angle(rotation):
    """Unit axis and signed angle of a rotation matrix, stable up to 180 degrees."""
    values, vectors = np.linalg.eig(rotation)
    axis = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    axis = axis / np.linalg.norm(axis)
    return axis, signed_angle_about(rotation, axis)


def signed_angle_about(rotation, direction):
    """Angle of 'rotation' measured about a given unit direction."""
    sine_axis = 0.5 * np.array([rotation[2, 1] - rotation[1, 2],
                                rotation[0, 2] - rotation[2, 0],
                                rotation[1, 0] - rotation[0, 1]])
    cosine = (np.trace(rotation) - 1.0) / 2.0
    return float(np.arctan2(sine_axis @ direction, cosine))


class Turntable:
    """Known rotation axis and angles: the object in view k is the object of
    view 0 rotated by angles[k] about the axis."""

    def __init__(self, point, direction, angles_deg):
        self.point = np.asarray(point, dtype=np.float64)
        self.direction = np.asarray(direction, dtype=np.float64)
        self.direction /= np.linalg.norm(self.direction)
        self.angles = np.radians(np.asarray(angles_deg, dtype=np.float64))
        self.sign = 1.0

    def pose(self, k):
        """Transform mapping view k into view 0."""
        return rotation_about_axis(self.point, self.direction, -self.sign * self.angles[k])

    def relative(self, i, j):
        """Transform mapping view i into view j."""
        return np.linalg.inv(self.pose(j)) @ self.pose(i)


def fit_rotation_axis(poses, reference_point, min_angle_deg=5.0):
    """Fit one fixed rotation axis to poses that map every view into view 0.

    For a pure rotation about a line through p with direction d, each pose is
    x -> R x + (I - R) p. The direction is the weighted mean of the individual
    rotation axes (weight 1 - cos(angle), since small rotations define their
    axis poorly); the point solves (I - R_k) p = t_k in the least squares
    sense and is then moved along d next to 'reference_point'.
    """
    rotations = [pose[:3, :3] for pose in poses]
    translations = [pose[:3, 3] for pose in poses]

    axes, angles = zip(*(axis_and_angle(r) for r in rotations))
    axes = np.array(axes)
    angles = np.array(angles)
    usable = np.abs(angles) > np.radians(min_angle_deg)
    if usable.sum() < 2:
        return None

    weights = 1.0 - np.cos(angles)
    reference_axis = axes[np.argmax(np.where(usable, weights, -1.0))]
    signs = np.where(axes @ reference_axis < 0, -1.0, 1.0)
    axes = axes * signs[:, None]

    direction = np.sum(axes[usable] * weights[usable, None], axis=0)
    direction /= np.linalg.norm(direction)

    # Poses map view k to view 0, i.e. they undo the table rotation: the view
    # angle is minus the pose angle. Choose the sign of the direction so that
    # the angles grow from view to view.
    view_angles = -np.array([signed_angle_about(r, direction) for r in rotations])
    view_angles = np.unwrap(view_angles)
    if np.median(np.diff(view_angles)) < 0:
        direction = -direction
        view_angles = -view_angles

    a = np.vstack([np.eye(3) - rotations[k] for k in np.flatnonzero(usable)])
    b = np.concatenate([translations[k] for k in np.flatnonzero(usable)])
    point = np.linalg.lstsq(a, b, rcond=None)[0]
    point = point + direction * ((reference_point - point) @ direction)

    residual_mm = [1000.0 * np.linalg.norm(t - (np.eye(3) - r) @ point)
                   for r, t in zip(rotations, translations)]
    tilt_deg = [np.degrees(np.arccos(np.clip(abs(axis @ direction), -1.0, 1.0)))
                for axis, ok in zip(axes, usable) if ok]

    return {
        "point": point.tolist(),
        "direction": direction.tolist(),
        "view_angles_deg": np.degrees(view_angles).tolist(),
        "translation_residual_mm": residual_mm,
        "axis_deviation_deg": tilt_deg,
    }


def load_axis(args):
    if args.axis_json:
        data = json.loads(Path(args.axis_json).read_text())
        axis = data.get("axis", data)
        return axis["point"], axis["direction"]
    if args.axis_point and args.axis_dir:
        return args.axis_point, args.axis_dir
    return None


def commanded_angles(paths, frames, args):
    """Turntable angle of every loaded file in degrees, or None if unknown."""
    if args.angles_file:
        values = np.loadtxt(args.angles_file, delimiter=None, ndmin=1).astype(np.float64)
        if len(values) != len(paths):
            raise ValueError(f"{args.angles_file}: {len(values)} angles for {len(paths)} input files")
        return values - values[0]
    if args.step_deg is not None:
        return args.step_deg * np.arange(len(paths), dtype=np.float64)
    if args.deg_per_second is not None:
        stamps = [frame.timestamp for frame in frames]
        if any(stamp is None for stamp in stamps):
            raise ValueError("--deg-per-second needs frame timestamps (.npz from ouster_extract.py)")
        return args.deg_per_second * (np.array(stamps) - stamps[0])
    return None


# ----------------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------------

class RegistrationClouds:
    """Downsampled copies of the object clouds, used only for registration."""

    def __init__(self, clouds, args, with_features):
        self.voxel = args.reg_voxel
        self.fine_distance = 1.5 * args.reg_voxel
        self.icp_distances = [5.0 * args.reg_voxel, 2.5 * args.reg_voxel, 1.5 * args.reg_voxel]
        self.clouds = []
        self.features = []
        for cloud in clouds:
            down = cloud.voxel_down_sample(self.voxel)
            down.normalize_normals()
            self.clouds.append(down)
            if with_features:
                self.features.append(registration.compute_fpfh_feature(
                    down, o3d.geometry.KDTreeSearchParamHybrid(radius=args.feature_radius, max_nn=100)))


class PairResult:
    def __init__(self, source, target, transformation, fitness, rmse, information, uncertain):
        self.source = source
        self.target = target
        self.transformation = transformation
        self.fitness = fitness
        self.rmse = rmse
        self.information = information
        self.uncertain = uncertain


def refine_icp(source, target, initial, distances, iterations=(60, 40, 30)):
    """Coarse-to-fine point-to-plane ICP with a robust kernel.

    Precise, but it can slide along the plane when one flat face dominates
    the view, because in-plane motion does not change the residual.
    """
    transformation = initial
    for distance, count in zip(distances, iterations):
        estimation = registration.TransformationEstimationPointToPlane(registration.TukeyLoss(k=distance))
        result = registration.registration_icp(
            source, target, distance, transformation, estimation,
            registration.ICPConvergenceCriteria(max_iteration=count))
        transformation = result.transformation
    return transformation


def refine_generalized_icp(source, target, initial, distances, iterations=(60, 40, 30)):
    """Coarse-to-fine generalized ICP (plane-to-plane): more robust on sparse,
    planar-dominated LiDAR views, slightly less precise when both work."""
    transformation = initial
    for distance, count in zip(distances, iterations):
        result = registration.registration_generalized_icp(
            source, target, distance, transformation,
            registration.TransformationEstimationForGeneralizedICP(),
            registration.ICPConvergenceCriteria(max_iteration=count))
        transformation = result.transformation
    return transformation


def point_to_plane_rmse(source, target, transformation, max_distance):
    """RMS distance of source points to the tangent planes of their nearest target points.

    On scanline data the plain nearest-neighbour distance is dominated by the
    gap between laser rows, so it says little about how well the surfaces
    agree; the distance along the target normal does.
    """
    points = np.asarray(copy.deepcopy(source).transform(transformation).points)
    target_points = np.asarray(target.points)
    target_normals = np.asarray(target.normals)
    tree = o3d.geometry.KDTreeFlann(target)

    residuals = []
    for point in points:
        found, index, squared = tree.search_knn_vector_3d(point, 1)
        if found and squared[0] <= max_distance ** 2:
            residuals.append((point - target_points[index[0]]) @ target_normals[index[0]])
    if not residuals:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(residuals))))


def feature_guess(regs, i, j):
    """Global registration from FPFH correspondences (RANSAC)."""
    distance = 3.0 * regs.voxel
    result = registration.registration_ransac_based_on_feature_matching(
        regs.clouds[i], regs.clouds[j], regs.features[i], regs.features[j], True, distance,
        registration.TransformationEstimationPointToPoint(False), 3,
        [registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         registration.CorrespondenceCheckerBasedOnDistance(distance)],
        registration.RANSACConvergenceCriteria(100000, 0.999))
    return result.transformation


def register_pair(regs, i, j, guesses, uncertain):
    """Refine every initial guess with ICP and keep the best alignment of view i onto view j."""
    source, target = regs.clouds[i], regs.clouds[j]
    candidates = []
    for guess in guesses:
        candidates.append(guess)
        candidates.append(refine_icp(source, target, guess, regs.icp_distances))
        candidates.append(refine_generalized_icp(source, target, guess, regs.icp_distances))

    # Highest overlap wins; among (nearly) equal overlaps, the smaller
    # point-to-plane residual.
    best = None
    for transformation in candidates:
        fitness = registration.evaluate_registration(
            source, target, regs.fine_distance, transformation).fitness
        if best is not None and fitness < best.fitness - 0.02:
            continue
        rmse = point_to_plane_rmse(source, target, transformation, regs.fine_distance)
        if best is None or fitness > best.fitness + 0.02 or rmse < best.rmse:
            best = PairResult(i, j, transformation, fitness, rmse, None, uncertain)

    best.information = registration.get_information_matrix_from_point_clouds(
        source, target, regs.fine_distance, best.transformation)
    return best


def initial_guesses(regs, i, j, args, turntable, current_poses=None):
    guesses = [np.eye(4)]
    if current_poses is not None:
        guesses.append(np.linalg.inv(current_poses[j]) @ current_poses[i])
    if turntable is not None:
        guesses.append(turntable.relative(i, j))
    if args.init in ("auto", "features") and regs.features and (turntable is None or args.init == "features"):
        guesses.append(feature_guess(regs, i, j))
    return guesses


def choose_rotation_sign(regs, turntable):
    """The sense of rotation is often unknown: keep the one that aligns views 0 and 1 better."""
    scores = {}
    for sign in (1.0, -1.0):
        turntable.sign = sign
        pair = register_pair(regs, 0, 1, [turntable.relative(0, 1)], False)
        # The refined result must stay close to the model, otherwise ICP
        # simply found the right answer from the wrong start.
        deviation = rotation_angle(np.linalg.inv(turntable.relative(0, 1)) @ pair.transformation)
        scores[sign] = (pair.fitness - deviation, -pair.rmse)
    turntable.sign = max(scores, key=scores.get)


def fit_axis_to_steps(transforms, max_deviation_deg, max_residual):
    """Robustly fit one rotation axis to the motions between consecutive views.

    On a turntable every step is a rotation about the same line. A step that
    disagrees (different axis, or translation not explained by the rotation)
    is a registration failure, typically a symmetric object matched in a
    flipped position. Returns (point, direction, median step angle, inlier
    mask) or None when too few steps are usable.
    """
    rotations = [t[:3, :3] for t in transforms]
    translations = [t[:3, 3] for t in transforms]
    axes = np.array([axis_and_angle(r)[0] for r in rotations])
    magnitudes = np.array([rotation_angle(t) for t in transforms])
    usable = np.flatnonzero(magnitudes > np.radians(3.0))
    if len(usable) < 2:
        return None

    # Consensus: the step axis agreeing with the largest number of others.
    cos_limit = np.cos(np.radians(max_deviation_deg))
    agreement = np.abs(axes[usable] @ axes[usable].T) > cos_limit
    seed = usable[np.argmax(agreement.sum(axis=1))]
    signs = np.where(axes @ axes[seed] < 0, -1.0, 1.0)
    aligned = axes * signs[:, None]
    candidates = [k for k in usable if aligned[k] @ aligned[seed] > cos_limit]

    direction = np.mean(aligned[candidates], axis=0)
    direction /= np.linalg.norm(direction)
    a = np.vstack([np.eye(3) - rotations[k] for k in candidates])
    b = np.concatenate([translations[k] for k in candidates])
    point = np.linalg.lstsq(a, b, rcond=None)[0]

    inliers = np.zeros(len(transforms), dtype=bool)
    for k in candidates:
        residual = np.linalg.norm(translations[k] - (np.eye(3) - rotations[k]) @ point)
        inliers[k] = residual < max_residual

    # The table turns one way: a step with the opposite sense of rotation is
    # a mirror-like false match even if its axis agrees.
    angles = np.array([signed_angle_about(r, direction) for r in rotations])
    majority = np.sign(np.sum(np.sign(angles[inliers])))
    inliers &= np.sign(angles) == majority
    if inliers.sum() < 2:
        return None
    return point, direction, float(np.median(angles[inliers])), inliers


def correction_is_small(start, result, center, args):
    """True if 'result' differs from 'start' by a small rigid correction.

    The displacement is measured at the object's centre: measured at the
    sensor origin, 1-2 m away, a tiny rotation would look like a large shift.
    """
    correction = np.linalg.inv(start) @ result
    shift = np.linalg.norm(correction[:3, :3] @ center + correction[:3, 3] - center)
    return (rotation_angle(correction) <= np.radians(args.max_correction)
            and shift <= args.max_shift)


def register_from_model(regs, i, j, guess, args, uncertain):
    """Register view i onto view j starting from a model of the table rotation.

    ICP may correct the model only slightly. A larger correction means that
    geometry alone cannot fix the rotation (object rotationally symmetric or
    featureless in these views) or that ICP found a false match. Returns
    (pair, accepted); for a rejected refinement, 'pair' holds the model itself.
    """
    pair = register_pair(regs, i, j, [guess], uncertain)
    center = np.asarray(regs.clouds[i].get_center())
    if correction_is_small(guess, pair.transformation, center, args):
        return pair, True

    pair.transformation = guess
    pair.fitness = registration.evaluate_registration(
        regs.clouds[i], regs.clouds[j], regs.fine_distance, guess).fitness
    pair.rmse = point_to_plane_rmse(regs.clouds[i], regs.clouds[j], guess, regs.fine_distance)
    pair.information = registration.get_information_matrix_from_point_clouds(
        regs.clouds[i], regs.clouds[j], regs.fine_distance, guess)
    return pair, False


def self_calibrate_turntable(regs, args):
    """Register consecutive views freely, then fit the common rotation axis.

    Returns a Turntable whose angles are the measured step angles (the median
    step for steps that disagree with the axis), or None.
    """
    count = len(regs.clouds)
    steps = [register_pair(regs, i, i + 1, initial_guesses(regs, i, i + 1, args, None), False)
             for i in range(count - 1)]
    model = fit_axis_to_steps([s.transformation for s in steps],
                              args.max_axis_deviation, args.max_axis_residual)
    if model is None:
        return None

    point, direction, median_step, inliers = model
    angles = [0.0]
    for step, inlier in zip(steps, inliers):
        angle = signed_angle_about(step.transformation[:3, :3], direction) if inlier else median_step
        angles.append(angles[-1] + angle)
    print(f"  self-calibrated axis from {int(inliers.sum())}/{len(inliers)} steps, "
          f"median step {np.degrees(median_step):.1f} deg")
    return Turntable(point, direction, np.degrees(angles))


def build_edges(regs, args, turntable):
    """Sequential edges (view k to k+1) and loop closures between overlapping views.

    Without a known turntable the axis is first self-calibrated from the
    views. Every edge then starts from the turntable model, which keeps the
    chain free of the tilt drift that free pairwise registration accumulates
    and lets loop closures start close to the answer.
    """
    count = len(regs.clouds)

    if turntable is None and args.init != "identity" and not args.free_motion:
        turntable = self_calibrate_turntable(regs, args)
        if turntable is None:
            print("  warning: could not self-calibrate the rotation axis (object symmetric or steps "
                  "too small). Give --axis-json and the angles.")

    sequential = []
    model_steps = []
    for k in range(count - 1):
        if turntable is None:
            sequential.append(register_pair(regs, k, k + 1,
                                            initial_guesses(regs, k, k + 1, args, None), False))
            continue
        pair, accepted = register_from_model(regs, k, k + 1, turntable.relative(k, k + 1), args, False)
        sequential.append(pair)
        if not accepted:
            model_steps.append(k)
    if model_steps:
        print(f"  {len(model_steps)}/{count - 1} steps kept the turntable model unrefined "
              f"(geometry cannot resolve the rotation there; symmetric or featureless object?)")

    # A step with much less overlap than the others usually means the files
    # are not in rotation order, or a large turn between two captures.
    overlaps = np.array([pair.fitness for pair in sequential])
    typical = float(np.median(overlaps))
    for pair in sequential:
        if pair.fitness < min(args.min_fitness, 0.5 * typical):
            print(f"  warning: weak alignment between views {pair.source} and {pair.target} "
                  f"(overlap {pair.fitness:.2f}, typical {typical:.2f}). Check that the input "
                  f"files are in rotation order and that the turn between them was small.")

    poses = [np.eye(4)]
    for pair in sequential:
        poses.append(poses[-1] @ np.linalg.inv(pair.transformation))

    # Loop closures between views that overlap. A loop closure only corrects
    # drift, so a refinement far from its start is a false match and rejected.
    edges = list(sequential)
    rejected = 0
    max_angle = np.radians(args.loop_max_angle)
    for i in range(count):
        for j in range(i + 2, count):
            start = turntable.relative(i, j) if turntable is not None else np.linalg.inv(poses[j]) @ poses[i]
            if rotation_angle(start) > max_angle:
                continue
            pair, accepted = register_from_model(regs, i, j, start, args, True)
            if accepted and pair.fitness >= args.min_fitness:
                edges.append(pair)
            else:
                rejected += 1
    return poses, edges, rejected


def optimize_pose_graph(poses, edges, regs, args):
    graph = registration.PoseGraph()
    for pose in poses:
        graph.nodes.append(registration.PoseGraphNode(pose))
    for edge in edges:
        graph.edges.append(registration.PoseGraphEdge(
            edge.source, edge.target, edge.transformation, edge.information, uncertain=edge.uncertain))

    option = registration.GlobalOptimizationOption(
        max_correspondence_distance=regs.fine_distance,
        edge_prune_threshold=0.25,
        preference_loop_closure=args.loop_preference,
        reference_node=0)
    registration.global_optimization(
        graph, registration.GlobalOptimizationLevenbergMarquardt(),
        registration.GlobalOptimizationConvergenceCriteria(), option)

    kept = {(edge.source_node_id, edge.target_node_id) for edge in graph.edges}
    return [np.asarray(node.pose) for node in graph.nodes], kept


def register_views(regs, args, turntable):
    poses, edges, rejected = build_edges(regs, args, turntable)
    loops_tried = sum(1 for e in edges if e.uncertain)
    poses, kept = optimize_pose_graph(poses, edges, regs, args)
    edges = [e for e in edges if (e.source, e.target) in kept]

    # Refinement: redo every surviving edge starting from the optimized poses.
    # As everywhere, a refinement that moves far from the start is a slide
    # along a symmetry, not an improvement, and the previous edge is kept.
    for _ in range(args.refine_passes):
        refined = []
        for edge in edges:
            start = np.linalg.inv(poses[edge.target]) @ poses[edge.source]
            pair = register_pair(regs, edge.source, edge.target, [start], edge.uncertain)
            center = np.asarray(regs.clouds[edge.source].get_center())
            refined.append(pair if correction_is_small(start, pair.transformation, center, args) else edge)
        poses, kept = optimize_pose_graph(poses, refined, regs, args)
        edges = [e for e in refined if (e.source, e.target) in kept]

    loops = sum(1 for e in edges if e.uncertain)
    stats = {"sequential_edges": len(edges) - loops, "loop_closures": loops,
             "loop_closures_rejected": rejected,
             "edges_pruned": loops_tried - loops}
    return poses, edges, stats


def alignment_quality(regs, poses, edges):
    """Overlap and point-to-plane RMSE of every kept edge under the final poses.

    The RMSE includes the sensor noise of both views (about sqrt(2) times the
    single-view noise when the alignment is perfect), so it is an upper bound
    on the misalignment and a proxy for the thickness of the fused shell.
    """
    results = []
    for edge in edges:
        source, target = regs.clouds[edge.source], regs.clouds[edge.target]
        relative = np.linalg.inv(poses[edge.target]) @ poses[edge.source]
        fitness = registration.evaluate_registration(source, target, regs.fine_distance, relative).fitness
        rmse = point_to_plane_rmse(source, target, relative, regs.fine_distance)
        results.append((edge.source, edge.target, fitness, 1000.0 * rmse))
    return results


# ----------------------------------------------------------------------------
# Fusion and output
# ----------------------------------------------------------------------------

def view_colors(count):
    hues = np.linspace(0.0, 1.0, count, endpoint=False)
    # HSV to RGB with full saturation and value.
    k = (np.array([5.0, 3.0, 1.0])[None, :] + hues[:, None] * 6.0) % 6.0
    return 1.0 - np.clip(np.minimum(k, 4.0 - k), 0.0, 1.0)


def fuse(clouds, poses, args):
    views = o3d.geometry.PointCloud()
    colors = view_colors(len(clouds))
    for cloud, pose, color in zip(clouds, poses, colors):
        aligned = copy.deepcopy(cloud).transform(pose)       # also rotates the normals
        aligned.paint_uniform_color(color)
        views += aligned

    fused = views.voxel_down_sample(args.voxel) if args.voxel > 0 else copy.deepcopy(views)
    fused.normalize_normals()
    if args.outlier_neighbors > 0 and len(fused.points) > args.outlier_neighbors:
        fused, _ = fused.remove_statistical_outlier(args.outlier_neighbors, args.outlier_std)
    fused.colors = o3d.utility.Vector3dVector()               # drop the debug colours
    return fused, views


def keep_near_axis(cloud, axis_fit, radius):
    """Keep the points whose distance from the fitted rotation axis is at most 'radius'."""
    points = np.asarray(cloud.points)
    point = np.asarray(axis_fit["point"])
    direction = np.asarray(axis_fit["direction"])
    offset = points - point
    distance = np.linalg.norm(offset - np.outer(offset @ direction, direction), axis=1)
    return cloud.select_by_index(np.flatnonzero(distance <= radius))


def to_builtin(value):
    """JSON fallback for numpy scalars and arrays."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def describe(values, unit, digits=1):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return "n/a"
    return (f"median {np.median(values):.{digits}f} {unit}, "
            f"max {np.max(values):.{digits}f} {unit}")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fuse views of a rotating object into one point cloud.")
    parser.add_argument("inputs", nargs="+", help="frames: files, directories or glob patterns (.npz/.ply/.pcd/.xyz)")
    parser.add_argument("--out", default="fused", help="output base name (writes .ply, _views.ply, .json)")
    parser.add_argument("--every", type=int, default=1, help="use every n-th frame")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all")
    parser.add_argument("--min-range", type=float, default=0.3, help="[m]")
    parser.add_argument("--max-range", type=float, default=10.0, help="[m]")

    iso = parser.add_argument_group("object isolation")
    iso.add_argument("--background", nargs="+", default=None,
                     help="frames of the empty scene (same sensor pose, turntable included)")
    iso.add_argument("--bg-threshold", type=float, default=0.03,
                     help="a pixel is foreground if closer than the background by this much [m]")
    iso.add_argument("--bg-relative", type=float, default=0.01,
                     help="additional margin as a fraction of the background range")
    iso.add_argument("--crop-min", type=float, nargs=3, metavar=("X", "Y", "Z"))
    iso.add_argument("--crop-max", type=float, nargs=3, metavar=("X", "Y", "Z"))
    iso.add_argument("--edge-jump", type=float, default=0.05,
                     help="remove mixed pixels differing from both neighbours by this much [m], 0 = off")
    iso.add_argument("--cluster-eps", type=float, default=0.05,
                     help="DBSCAN radius for keeping the largest cluster [m], 0 = off")
    iso.add_argument("--cluster-min-points", type=int, default=10)
    iso.add_argument("--outlier-neighbors", type=int, default=20, help="0 = off")
    iso.add_argument("--outlier-std", type=float, default=2.0)
    iso.add_argument("--normal-radius", type=float, default=0.05, help="[m]")
    iso.add_argument("--min-points", type=int, default=200, help="drop views with fewer object points")

    reg = parser.add_argument_group("registration")
    reg.add_argument("--init", default="auto", choices=["auto", "turntable", "features", "identity"],
                     help="auto: turntable model if axis and angles are known, otherwise features")
    reg.add_argument("--reg-voxel", type=float, default=0.01, help="voxel size used for registration [m]")
    reg.add_argument("--feature-radius", type=float, default=0.08, help="FPFH radius [m]")
    reg.add_argument("--min-fitness", type=float, default=0.3,
                     help="minimum overlap fraction to accept a loop closure")
    reg.add_argument("--loop-max-angle", type=float, default=60.0,
                     help="try loop closures between views rotated less than this [deg]")
    reg.add_argument("--loop-preference", type=float, default=1.0)
    reg.add_argument("--refine-passes", type=int, default=1)
    reg.add_argument("--free-motion", action="store_true",
                     help="the object is moved by hand, not on a turntable: do not assume a fixed "
                          "rotation axis (steps are only checked by overlap, loops by consistency)")
    reg.add_argument("--max-axis-deviation", type=float, default=5.0,
                     help="a step whose rotation axis deviates more than this from the common axis "
                          "is re-registered [deg]")
    reg.add_argument("--max-axis-residual", type=float, default=0.01,
                     help="... or whose translation is not explained by the rotation within this [m]")
    reg.add_argument("--max-correction", type=float, default=10.0,
                     help="reject refinements that rotate a pose by more than this [deg]")
    reg.add_argument("--max-shift", type=float, default=0.05,
                     help="... or move the object centre by more than this [m]")

    table = parser.add_argument_group("turntable (optional)")
    table.add_argument("--step-deg", type=float, default=None, help="constant rotation between input files")
    table.add_argument("--angles-file", default=None, help="text file, one angle [deg] per input file")
    table.add_argument("--deg-per-second", type=float, default=None,
                       help="continuous rotation speed; angles from the frame timestamps")
    table.add_argument("--axis-json", default=None, help="axis from a previous run's output .json")
    table.add_argument("--axis-point", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    table.add_argument("--axis-dir", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    table.add_argument("--keep-radius", type=float, default=0.0,
                       help="after fusion, keep only points within this distance of the fitted "
                            "rotation axis [m]; removes static clutter smeared into arcs. 0 = off")
    table.add_argument("--snap-to-axis", action="store_true",
                       help="replace the poses by pure rotations about the fitted axis")

    out = parser.add_argument_group("fusion")
    out.add_argument("--voxel", type=float, default=0.005, help="averaging voxel of the fused cloud [m], 0 = off")
    return parser.parse_args()


def main():
    args = parse_arguments()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    o3d.utility.random.seed(0)

    print(f"fuse_views.py version {VERSION}")
    paths = expand_inputs(args.inputs)
    names = [p.name for p in paths]
    shown = names if len(names) <= 8 else names[:4] + ["..."] + names[-3:]
    print(f"input order ({len(names)} files): {', '.join(shown)}")
    frames = [load_frame(p, args.min_range, args.max_range) for p in paths]
    angles = commanded_angles(paths, frames, args)

    selected = list(range(0, len(frames), max(1, args.every)))
    if args.max_frames > 0:
        selected = selected[:args.max_frames]
    if len(selected) < 2:
        raise SystemExit("need at least two views")

    background = None
    if args.background:
        background_frames = [load_frame(p, args.min_range, args.max_range)
                             for p in expand_inputs(args.background)]
        background = Background(background_frames)
        print(f"background: {len(background_frames)} frames")
    else:
        print("warning: no --background. Anything static left after cropping (turntable, floor) "
              "does not rotate and pulls the registration towards 'no motion'. Record the empty "
              "scene (turntable included) and pass it with --background.")

    # 1. Object isolation
    clouds, used, dropped = [], [], []
    for k in selected:
        cloud = isolate_object(frames[k], background, args)
        if len(cloud.points) < args.min_points:
            dropped.append(frames[k].name)
            continue
        clouds.append(cloud)
        used.append(k)
    if len(clouds) < 2:
        raise SystemExit(f"fewer than two views with at least {args.min_points} object points")

    sizes = [len(c.points) for c in clouds]
    print(f"views: {len(selected)} selected, {len(used)} used"
          + (f", dropped (too few points): {', '.join(dropped)}" if dropped else ""))
    print(f"object points per view: min {min(sizes)}, median {int(np.median(sizes))}, max {max(sizes)}")

    # 2. Registration
    axis = load_axis(args)
    turntable = None
    if args.init in ("auto", "turntable") and axis is not None and angles is not None:
        turntable = Turntable(axis[0], axis[1], angles[used])
    elif args.init == "turntable":
        raise SystemExit("--init turntable needs an axis (--axis-json or --axis-point/--axis-dir) "
                         "and angles (--step-deg, --angles-file or --deg-per-second)")

    use_features = args.init == "features" or (args.init == "auto" and turntable is None)
    regs = RegistrationClouds(clouds, args, with_features=use_features)
    if turntable is not None:
        choose_rotation_sign(regs, turntable)
        print(f"initialization: turntable model (sense of rotation {'+' if turntable.sign > 0 else '-'})")
    else:
        print(f"initialization: {'FPFH features + identity' if use_features else 'identity'}")

    poses, edges, stats = register_views(regs, args, turntable)
    quality = alignment_quality(regs, poses, edges)
    worst = max(quality, key=lambda q: q[3])
    print(f"registration: {stats['sequential_edges']} sequential edges, {stats['loop_closures']} loop closures "
          f"({stats['loop_closures_rejected']} rejected, {stats['edges_pruned']} pruned by the optimizer)")
    print(f"alignment: inlier rmse {describe([q[3] for q in quality], 'mm')} "
          f"(worst between views {worst[0]} and {worst[1]}), "
          f"overlap {describe([q[2] for q in quality], '', 2)}")

    # 3. Turntable axis fit
    reference = np.mean(np.asarray(clouds[0].points), axis=0)
    axis_fit = fit_rotation_axis(poses, reference)
    if axis_fit is None:
        print("turntable axis: not identifiable (views rotated by less than 5 degrees)")
    else:
        view_angles = np.array(axis_fit["view_angles_deg"])
        print(f"turntable axis: point {np.round(axis_fit['point'], 4).tolist()} m, "
              f"direction {np.round(axis_fit['direction'], 4).tolist()}")
        print(f"  individual axes deviate by {describe(axis_fit['axis_deviation_deg'], 'deg', 2)}")
        print(f"  translation not explained by the rotation: {describe(axis_fit['translation_residual_mm'], 'mm')}")
        print(f"  angle per view [deg]: {' '.join(f'{a:.1f}' for a in view_angles)}")
        if angles is not None:
            command = angles[used]
            if np.dot(command, view_angles) < 0:
                command = -command
            print(f"  difference from the commanded angles: {describe(np.abs(view_angles - command), 'deg', 2)}")
        if args.snap_to_axis:
            poses = [rotation_about_axis(axis_fit["point"], axis_fit["direction"], -np.radians(a))
                     for a in view_angles]
            print("  poses replaced by pure rotations about the fitted axis")

    # 4. Fusion
    fused, views = fuse(clouds, poses, args)

    # Static clutter that survived isolation (no background, or a loose crop
    # box) is rotated with the views and smears into arcs around the object.
    # The object itself lies within a known distance of the rotation axis.
    if args.keep_radius > 0:
        if axis_fit is None:
            print("warning: --keep-radius ignored, the rotation axis could not be fitted")
        else:
            fused = keep_near_axis(fused, axis_fit, args.keep_radius)
            views = keep_near_axis(views, axis_fit, args.keep_radius)
            print(f"kept points within {args.keep_radius:g} m of the rotation axis: {len(fused.points)}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fused_path = out.with_suffix(".ply")
    views_path = out.with_name(out.stem + "_views.ply")
    o3d.io.write_point_cloud(str(fused_path), fused)
    o3d.io.write_point_cloud(str(views_path), views)

    report = {
        "inputs": [frames[k].name for k in used],
        "dropped": dropped,
        "poses_to_first_view": [p.tolist() for p in poses],
        "edges": [{"source": s, "target": t, "fitness": f, "inlier_rmse_mm": r} for s, t, f, r in quality],
        "registration": stats,
        "axis": axis_fit,
        "parameters": vars(args),
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=to_builtin))
    print(f"fused cloud: {len(fused.points)} points -> {fused_path}")
    print(f"aligned views (one colour per view) -> {views_path}")
    print(f"poses and axis -> {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
