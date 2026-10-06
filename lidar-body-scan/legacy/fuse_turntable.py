"""
Fuse a capture_turntable.py run (person standing still on the rotating platform)
into one point cloud. Any rotation: a partial turn (e.g. 50 deg), one lap, or
several laps (e.g. 1800 deg); the rotation is measured from the data.

    python fuse_turntable.py C:\\lidar\\tt1 --out person_tt

Setting
-------
The LiDAR is fixed, the platform turns at an almost constant speed and the
person stands still on it. Every frame is then the same body rotated by the
platform angle theta(t) about the platform axis. The script

    1. background and floor: per-pixel median of the empty-platform frames;
       floor plane (the largest plane below the sensor: the sensor may be
       tilted or rolled about its viewing axis); output frame with z up; the
       platform ring found in the empty scene gives the platform centre
       (--center to impose one).
    2. person in every frame: closer than the empty scene, inside a cylinder
       around the platform centre (--radius), above the platform
       top (--min-height). The time of a frame is the sensor timestamp of the
       columns that see the person (jitter-free, unlike the PC clock).
    3. platform angle versus time, from the data:
       a. frames every --sample-seconds (at most --max-samples) are registered
          to their neighbours with ONE unknown, the turn about --center, plus
          pairs two samples apart; solved together: approximate angles, the
          still parts and the turn.
       b. the axis position and one turn per long pair (20 to 60 degrees
          apart) are fitted jointly: a wrong axis leaves a shift that grows
          with the turn, so the axis comes out to a few mm.
       c. a motion model is fitted to the long pairs: trapezoidal stepper
          profile (start, constant speed, ramp, total). For rotations longer
          than --move-deg also a sequence of moves of --move-deg with stops
          in between (girogirotondo_timer.m sends one lap per command); the
          data choose. Then, lap by lap, pairs one or more laps apart
          (revisits: the body faces the same way again) are predicted by the
          model, registered and added. Registration of partial views reads
          every turn short by 1 to 3 % (the parts that rotate out of view
          pull towards "no motion"); the whole laps of the revisits are
          exact, so they calibrate that scale, and the speed and the total
          stay accurate over many laps.
       d. consecutive frames at the full frame rate around every start and
          stop of the platform: the timing of the moves (ramps, stops),
          which separates speed from stops when there are several moves.
       e. a free-form correction of the profile (piecewise linear, a knot
          every --correction-spacing s), fitted to all the pairs, the short
          ones between neighbouring samples included (local speed): under
          load the motor can lose steps, the platform then falls behind the
          profile by degrees, and the stepper model alone cannot follow it.
       Each class of pairs is weighted by its own noise. In simulation the
       angles come out at 0.1 to 0.5 deg rms when the platform follows the
       profile, and at about 1 deg rms (instead of 3 to 4) when it loses
       steps or stops between moves in a way the profile misses.
    4. views every --view-step degrees of turn (with several laps, several
       views per direction; at most --max-views): each point rotated back by
       the platform angle at the time of its column (the sensor sweeps the
       columns during the frame) about the fitted axis; optionally the turn
       of every view searched against the other views (--angle-iterations);
       an axis polish from the pattern of the per-view
       corrections (only if the views go round the body); small bounded
       corrections per view (sway; a view out of the bounds is dropped) and
       the height-slab correction of fuse_person.py; then each free-hanging
       arm of every view is registered on its own (--limb-iterations): held
       away from the body for minutes, the arms sink and swing, and a
       whole-body correction cannot follow them. Every correction is made
       against the views of the other groups (--reference-groups), with a
       fixed point budget per reference.
    5. fusion: multi-view support filter (--min-views per lap), voxel
       averaging, then every output point moved onto the median surface of
       all the view points within 1 cm (robust local fit: this is where more
       laps reduce the noise), normals outward, and a confidence per point.

What more laps give: more views of every surface patch, so a more precise
fitted surface (the standard error falls about as 1/sqrt(points)), a stricter
outlier filter, and more revisit pairs for the angles. What they do not give: points where the sensor never
looks (the top of the head from a sensor below it, under the chin, soles),
and vertical resolution, which stays the beam spacing (about 2 cm at 1.6 m)
because the person does not move vertically.

Output frame: origin on the platform axis at the platform top (z = 0 where the
feet stand), axes of the first frame.
    <out>.ply              fused cloud with normals (input of pointcloud_to_mesh.py)
    <out>_confidence.ply   the same points with scalar fields views, points,
                           spread_mm, coloured by the spread (green 0, red 5 mm)
    <out>_views.ply        all views, one colour each
    <out>.json, <out>_angle.png

Then (no --denoise: the cloud is already fitted to the surface, and a second
plane projection over 3 cm pulls convex parts about 1 mm inward):
    python pointcloud_to_mesh.py person_tt.ply --method poisson --depth 9 --trim-distance 0 ^
        --watertight 0.004 --clip-below 0 --smooth 10 --target-triangles 150000 ^
        --out person_tt_mesh.ply

Calibration: the platform ring (radius 0.582 m, top about 0.03 m) is found in
the empty-scene frames, near (1.619, 0.280) m of the floor frame (accumulated.ply
of 2026-09-30) or anywhere within 0.8 m of it, so the sensor may be moved or
rolled between sessions; --center X Y imposes a centre.

Sensor mounting: rolled by 10 to 20 deg about its viewing axis, the beam rings
cross the body obliquely and the turn fills the 2 cm gaps between them; upright,
the body is sampled at the same heights in every view near the sensor height
(the stripes on the chest and back). Keep the whole body inside the field of
view (about +-43 deg vertically at a 15 deg roll).

Fuse part of a recording: --view-range 0 360 (first lap), 720 1080 (third lap);
the angles are still measured on the whole recording. Comparing the laps
shows whether the person moved between them.
"""

import argparse
import copy
import json
import re
import warnings
from pathlib import Path

import numpy as np
import open3d as o3d

from fuse_person import (Pair, Progress, Target, evaluate, icp_robust, median_range,
                         mixed_pixel_mask, slab_refine, solve_turns, support_filter, reference_cloud,
                         tilt_degrees, to_builtin,
                         transform_points, view_colors, wrapped_degrees, yaw_of, yaw_transform)

registration = o3d.pipelines.registration

VERSION = "2026-10-02d (frames with lost packets kept unless the loss hits the person; one clock for frame times)"


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

def natural_key(path):
    parts = re.split(r"(\d+)", Path(path).name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


class Run:
    def __init__(self, directory, args):
        self.directory = Path(directory)
        lut = np.load(self.directory / "lut.npz")
        self.direction = lut["direction"].astype(np.float64)
        self.offset = lut["offset"].astype(np.float64)
        self.args = args
        self.background_paths = sorted((self.directory / "background").glob("*.npz"), key=natural_key)
        self.frame_paths = sorted((self.directory / "frames").glob("*.npz"), key=natural_key)
        capture = self.directory / "capture.json"
        self.capture = json.loads(capture.read_text()) if capture.exists() else {}
        if not self.frame_paths:
            raise SystemExit(f"no frames in {self.directory / 'frames'}")
        if not self.background_paths:
            raise SystemExit("no background frames: record the empty platform first (capture_turntable.py does)")

    def load(self, path):
        data = np.load(path)
        range_m = data["range"].astype(np.float64) / 1000.0
        range_m[(range_m < self.args.min_range) | (range_m > self.args.max_range)] = np.nan
        info = {
            "time": float(data["time"]) if "time" in data.files else np.nan,
            "phase": int(data["phase"]) if "phase" in data.files else 0,
            "columns_ok": float(data["columns_ok"]) if "columns_ok" in data.files else 1.0,
            "timestamps": (np.asarray(data["timestamps"], dtype=np.float64) * 1e-9
                           if "timestamps" in data.files else None),
        }
        return range_m, info

    def xyz(self, range_m):
        return range_m[..., None] * self.direction + self.offset


# ----------------------------------------------------------------------------
# Floor, platform, person
# ----------------------------------------------------------------------------

def rotation_to_z(normal):
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, z)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.eye(3)
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis / s * np.arctan2(s, normal @ z))


def fit_floor(points, max_tilt_deg=60.0):
    """world <- sensor transform: z along the floor normal, z = 0 on the floor.

    The floor is the plane with the most points among the planes BELOW the
    sensor (0.2 to 4 m) whose normal is within max_tilt_deg of the sensor z
    axis: the sensor may be tilted, or rolled about its viewing axis (a roll
    of 10 to 20 deg makes the beam rings cross the body obliquely, so that the
    turn fills the 2 cm gaps between them). Walls (normal about 90 deg from
    the floor's) and the ceiling (above the sensor) are excluded. Planes are
    found one after the other by RANSAC on a subsample of the empty scene,
    then the chosen one is refitted to all the points within 1.5 cm."""
    rng = np.random.default_rng(0)
    o3d.utility.random.seed(0)
    sample = points[rng.choice(len(points), min(len(points), 40000), replace=False)]
    remaining = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample))
    candidates = []
    for _ in range(6):
        if len(remaining.points) < 1000:
            break
        plane, inliers = remaining.segment_plane(0.02, 3, 1000)
        normal, offset = np.array(plane[:3]), plane[3]
        scale = np.linalg.norm(normal)
        normal, offset = normal / scale, offset / scale
        if offset < 0:                                   # sensor (origin) on the positive side
            normal, offset = -normal, -offset
        tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
        if tilt <= max_tilt_deg and 0.2 <= offset <= 4.0 and len(inliers) >= 500:
            candidates.append((len(inliers), normal, offset))
        remaining = remaining.select_by_index(inliers, invert=True)
    if not candidates:
        raise SystemExit("no floor found in the background: no large plane below the sensor within "
                         f"{max_tilt_deg:g} deg of its z axis")
    _, normal, offset = max(candidates, key=lambda c: c[0])
    near = points[np.abs(points @ normal + offset) < 0.015]
    if len(near) > 1000:                                 # least-squares refit on all the floor points
        center = near.mean(axis=0)
        normal = np.linalg.svd(near - center, full_matrices=False)[2][2]
        if normal @ (-center) < 0:
            normal = -normal
        offset = -float(normal @ center)
    transform = np.eye(4)
    transform[:3, :3] = rotation_to_z(normal)
    transform[:3, 3] = [0.0, 0.0, offset]
    return transform, float(offset), float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))


def fit_circle(xy, iterations=6, tolerance=0.02):
    keep = np.ones(len(xy), dtype=bool)
    center, radius = None, None
    for _ in range(iterations):
        a = np.column_stack([2 * xy[keep], np.ones(keep.sum())])
        b = (xy[keep] ** 2).sum(axis=1)
        solution = np.linalg.lstsq(a, b, rcond=None)[0]
        center = solution[:2]
        radius = float(np.sqrt(solution[2] + center @ center))
        residual = np.abs(np.linalg.norm(xy - center, axis=1) - radius)
        keep = residual < max(tolerance, 2.5 * np.median(residual[keep]))
    return center, radius, float(np.sqrt(np.mean(residual[keep] ** 2))), int(keep.sum())


PLATFORM_CENTER = np.array([1.619, 0.280])     # floor frame, sensor at its 2026-09-30 place
PLATFORM_RADIUS = 0.582


def platform_ring(world, center, args):
    """Circle fitted to the low ring of the platform in the empty scene."""
    distance = np.linalg.norm(world[:, :2] - center, axis=1)
    ring = world[(world[:, 2] > 0.005) & (world[:, 2] < 0.10) & (distance > 0.35) & (distance < 0.80)]
    if len(ring) < 100:
        return None
    return fit_circle(ring[:, :2])


def plausible_ring(ring):
    return ring is not None and abs(ring[1] - PLATFORM_RADIUS) < 0.03 and ring[2] < 0.025 and ring[3] >= 300


def find_platform(world, start, args, search=0.8):
    """The platform ring near 'start' (floor frame); if it is not there (the
    sensor was moved or rolled), a Hough vote for circles of the platform
    radius among the low points within +-search m. Returns the ring fit or
    None."""
    ring = platform_ring(world, start, args)
    if plausible_ring(ring):
        return platform_ring(world, ring[0], args) or ring           # re-centre the annulus once
    low = world[(world[:, 2] > 0.005) & (world[:, 2] < 0.10)][:, :2]
    low = low[np.all(np.abs(low - start) < search + PLATFORM_RADIUS + 0.1, axis=1)]
    if len(low) < 300:
        return None
    if len(low) > 20000:
        low = low[np.random.default_rng(0).choice(len(low), 20000, replace=False)]
    angles = np.linspace(0.0, 2 * np.pi, 90, endpoint=False)
    circle = PLATFORM_RADIUS * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    votes = (low[:, None, :] + circle[None]).reshape(-1, 2)
    edges = [np.arange(start[c] - search, start[c] + search + 0.02, 0.02) for c in range(2)]
    histogram, ex, ey = np.histogram2d(votes[:, 0], votes[:, 1], bins=edges)
    for _ in range(3):                                              # best peaks first
        i, j = np.unravel_index(np.argmax(histogram), histogram.shape)
        if histogram[i, j] <= 0:
            break
        guess = np.array([0.5 * (ex[i] + ex[i + 1]), 0.5 * (ey[j] + ey[j + 1])])
        ring = platform_ring(world, guess, args)
        if plausible_ring(ring):
            return platform_ring(world, ring[0], args) or ring
        histogram[max(i - 3, 0):i + 4, max(j - 3, 0):j + 4] = 0
    return None


class Isolator:
    def __init__(self, run, background, world_from_sensor, center, args):
        self.run, self.background = run, background
        self.world_from_sensor = world_from_sensor
        self.center, self.args = center, args

    def mask(self, range_m):
        args = self.args
        margin = np.maximum(args.bg_threshold, args.bg_relative * np.nan_to_num(self.background, nan=0.0))
        with np.errstate(invalid="ignore"):
            closer = (range_m < self.background - margin) | (np.isfinite(range_m) & np.isnan(self.background))
        xyz = self.run.xyz(range_m)
        world = np.einsum("ij,hwj->hwi", self.world_from_sensor[:3, :3], xyz) + self.world_from_sensor[:3, 3]
        with np.errstate(invalid="ignore"):
            radial = np.linalg.norm(world[..., :2] - self.center, axis=-1)
            inside = (radial < args.radius) & (world[..., 2] > args.min_height) & (world[..., 2] < args.max_height)
        mask = closer & inside
        if args.edge_jump > 0:
            mask &= ~mixed_pixel_mask(range_m, args.edge_jump)
        return mask, world

    def cloud(self, range_m, timestamps=None):
        """World-frame person cloud with outward normals, the sensor time of the
        person (mean over its points) and the sensor time of every point (its
        column), or None without timestamps.

        The LiDAR sweeps the 360 degrees in one frame period; if the person
        straddles the column where the sweep starts, the two sides of the body
        are measured almost one period apart. The mean time is continuous in
        that case (a median would jump by a whole period), and the per-point
        times let build_views undo the turn within the frame."""
        mask, world = self.mask(range_m)
        points = world[mask]
        point_times = None
        if timestamps is not None:
            point_times = np.broadcast_to(timestamps[None, :], mask.shape)[mask]
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        keep = np.arange(len(points))
        if len(points) >= 20:
            labels = np.asarray(cloud.cluster_dbscan(eps=self.args.cluster_eps, min_points=10))
            if labels.size and labels.max() >= 0:
                keep = np.flatnonzero(labels == np.bincount(labels[labels >= 0]).argmax())
                cloud = cloud.select_by_index(keep)
        if len(cloud.points) > 30:
            cloud, inliers = cloud.remove_statistical_outlier(20, 2.0)
            keep = keep[np.asarray(inliers, dtype=np.int64)]
        if len(cloud.points) >= 3:
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
            cloud.orient_normals_towards_camera_location(self.world_from_sensor[:3, 3])
        when = None
        if point_times is not None:
            point_times = point_times[keep]
            valid = point_times > 0
            if valid.any():
                when = float(np.mean(point_times[valid]))
                point_times = np.where(valid, point_times, when)
            else:
                point_times = None
        return cloud, when, point_times


# ----------------------------------------------------------------------------
# Platform angle versus time
# ----------------------------------------------------------------------------

class Samples:
    """Downsampled clouds of the sampled frames, for registration."""

    def __init__(self, clouds, voxel, fine):
        self.clouds = [c.voxel_down_sample(voxel) for c in clouds]
        for c in self.clouds:
            if not c.has_normals():
                c.estimate_normals()
        self.points = [np.asarray(c.points) for c in self.clouds]
        self.targets = [Target(c) for c in self.clouds]
        self.fine = fine

    def register_turn(self, i, j, expected_degrees, pivot, uncertain):
        """Best one-unknown (turn about 'pivot') alignment of sample i onto j,
        over the starting turns 'expected_degrees'."""
        best = None
        for start in np.radians(expected_degrees):
            angle, fitness, rmse = icp_turn_about(self.points[i], self.targets[j], start, pivot)
            if best is None or fitness > best[1] + 0.02 or (fitness > best[1] - 0.02 and rmse < best[2]):
                best = (angle, fitness, rmse)
        transform = yaw_transform(best[0], pivot)
        pair = Pair(i, j, transform, best[1], best[2], uncertain)
        pair.information = registration.get_information_matrix_from_point_clouds(
            self.clouds[i], self.clouds[j], self.fine, transform)
        return pair


def estimate_angles(samples, times, phases, center, args):
    """Platform angle of every sample [rad] (angle of sample 0 = 0).

    Every pair is registered with one unknown, the turn about the vertical
    through --center (the body does not translate on the platform). A free
    turn-plus-shift registration is ambiguous on partial views of a body (a
    small turn looks like a sideways shift); fixing the axis removes that.
    The axis position itself is refined later from the views.
    """
    count = len(samples.points)
    pivot = np.append(center, 0.0)
    print(f"angle estimation from {count} sampled frames (turn about the platform centre):")
    progress = Progress("consecutive", count - 1)
    sequential, expected = [], 0.0
    for k in range(count - 1):
        # Wide search at the start and where the platform starts or stops;
        # elsewhere the speed is nearly constant and the last step predicts the next.
        boundary = k < 3 or phases[k] != phases[k + 1] or (k > 0 and phases[k - 1] != phases[k])
        width = 30.0 if boundary else 12.0
        pair = samples.register_turn(k, k + 1, expected + np.arange(-width, width + 0.1, 3.0), pivot, False)
        sequential.append(pair)
        turn = wrapped_degrees(yaw_of(pair.transform))
        expected = turn if (phases[k] == 2 or phases[k + 1] == 2) else 0.0
        if (k + 1) % 20 == 0 or k == count - 2:
            progress.step(k + 1)

    chain = np.zeros(count)
    for k, pair in enumerate(sequential):
        chain[k + 1] = chain[k] + yaw_of(pair.transform)

    # Pairs two samples apart. The pairs between laps (the body faces the
    # same way again) come later, once the motion model predicts them well:
    # over several turns the chained angles drift by tens of degrees.
    measurements = list(sequential)
    lag_pairs = [(k, k + 2) for k in range(count - 2)]
    progress = Progress("longer pairs", len(lag_pairs))
    for n, (i, j) in enumerate(lag_pairs):
        expected = np.degrees(chain[j] - chain[i])
        measurements.append(samples.register_turn(i, j, expected + np.arange(-6.0, 6.1, 3.0), pivot, True))
        if (n + 1) % 40 == 0 or n == len(lag_pairs) - 1:
            progress.step(n + 1)
    print(f"  {len(measurements)} relative turns")

    class Settings:
        max_correction = args.huber_deg * 2.0
    angles = solve_turns(count, chain, measurements, Settings())
    return angles, chain, measurements


def icp_turn_about(source, target, angle, pivot, distances=(0.05, 0.03, 0.015), iterations=(25, 20, 15)):
    """Point-to-plane ICP with ONE unknown: the turn about the vertical line through 'pivot'.

    On the platform the body does not translate, so fixing the axis removes the
    ambiguity between a small turn and a sideways shift of a partial view."""
    for distance, count in zip(distances, iterations):
        for _ in range(count):
            moved = transform_points(yaw_transform(angle, pivot), source)
            index, gap = target.nearest(moved)
            use = gap < distance
            if use.sum() < 30:
                return angle, 0.0, float("inf")
            p, q, n = moved[use], target.points[index[use]], target.normals[index[use]]
            residual = np.einsum("ij,ij->i", n, p - q)
            weight = np.where(np.abs(residual) < distance, (1.0 - (residual / distance) ** 2) ** 2, 0.0)
            lever = p - pivot
            jacobian = n[:, 1] * lever[:, 0] - n[:, 0] * lever[:, 1]
            step = -np.sum(weight * jacobian * residual) / (np.sum(weight * jacobian ** 2) + 1e-12)
            angle += float(np.clip(step, -0.05, 0.05))
            if abs(step) < 1e-6:
                break
    fitness, rmse = evaluate(source, target, yaw_transform(angle, pivot), distance)
    return angle, fitness, rmse


def profile_angle(t, start, speed, ramp, total):
    """Trapezoidal speed profile of a stepper move: linear ramp up in 'ramp'
    seconds, constant 'speed', symmetric ramp down, 'total' angle (radians)."""
    ramp = max(ramp, 1e-3)
    ramp_angle = 0.5 * speed * ramp
    if 2 * ramp_angle > total:                                  # triangular profile (short move)
        ramp_angle = total / 2
        ramp = 2 * ramp_angle / speed
    cruise = (total - 2 * ramp_angle) / speed
    u = np.asarray(t, dtype=np.float64) - start
    angle = np.where(u < ramp, 0.5 * speed / ramp * u ** 2, 0.0)
    angle = np.where((u >= ramp) & (u < ramp + cruise), ramp_angle + speed * (u - ramp), angle)
    down = u - ramp - cruise
    angle = np.where((u >= ramp + cruise) & (u < 2 * ramp + cruise),
                     total - ramp_angle + speed * down - 0.5 * speed / ramp * down ** 2, angle)
    angle = np.where(u >= 2 * ramp + cruise, total, angle)
    return np.where(u <= 0, 0.0, angle)


class Motion:
    """Platform angle versus time [rad]: one stepper move, or several moves of
    'move' radians each in sequence (the last one takes the rest of 'total').

    Every move has the trapezoidal profile of profile_angle with the same
    cruise speed and ramp time; 'starts' holds the start time of every move.
    girogirotondo_timer.m sends one command per full turn at most (larger
    step counts may not fit the controller's integer), so a turn of several
    laps is a sequence of moves with short stops in between; a single
    continuous move is the special case with one start.
    """

    def __init__(self, speed, ramp, starts, total, move=None):
        self.speed = float(speed)
        self.ramp = float(ramp)
        self.starts = np.atleast_1d(np.asarray(starts, dtype=np.float64)).copy()
        self.correction = None             # (knots [s], values [rad]): free-form part, see fit_correction
        self.total = float(total)
        self.move = move

    def moves(self):
        count = len(self.starts)
        if count == 1:
            return [self.total]
        last = max(self.total - self.move * (count - 1), 1e-3)
        return [self.move] * (count - 1) + [last]

    def duration(self, angle):
        ramp_angle = 0.5 * self.speed * max(self.ramp, 1e-3)
        if 2 * ramp_angle > angle:
            return 2 * angle / self.speed
        return self.ramp + angle / self.speed

    def ordered(self):
        """A move starts only after the previous one has finished."""
        moves = self.moves()
        for m in range(1, len(self.starts)):
            self.starts[m] = max(self.starts[m], self.starts[m - 1] + self.duration(moves[m - 1]))
        return self

    def angle(self, t):
        base = sum(profile_angle(t, start, self.speed, self.ramp, angle)
                   for start, angle in zip(self.starts, self.moves()))
        if self.correction is None:
            return base
        knots, values = self.correction
        return base + np.interp(t, knots, values)

    def end(self):
        return float(self.starts[-1] + self.duration(self.moves()[-1]))

    def vector(self):
        return np.concatenate([[self.speed, self.ramp, self.total], self.starts])

    def from_vector(self, values):
        return Motion(values[0], values[1], values[3:], values[2], self.move).ordered()

    def describe(self):
        text = (f"speed {np.degrees(self.speed):.3f} deg/s, ramp {self.ramp:.2f} s, "
                f"total {np.degrees(self.total):.2f} deg")
        if len(self.starts) == 1:
            return f"one move from {self.starts[0]:.2f} s, " + text
        stops = [self.starts[m + 1] - (self.starts[m] + self.duration(a))
                 for m, a in enumerate(self.moves()[:-1])]
        return (f"{len(self.starts)} moves of up to {np.degrees(self.move):g} deg from {self.starts[0]:.2f} s, "
                + text + f", stops between moves {', '.join(f'{s:.2f}' for s in stops)} s")


def fit_motion(times, measurements, motion, fit_total, scale=0.0, fit_scale=False, iterations=40,
               dense_scale=0.0):
    """Gauss-Newton with Huber weights of a Motion to relative turn
    measurements (i, j, measured turn [rad], weight, laps[, kind]).

    Registration of partial views underestimates a turn by a fraction that
    grows with it (the parts of the body that rotate out of view pull towards
    'no motion'; 1 to 3 % in simulation, depending on the body and the
    sway). So the measured turns are modelled as (1 + scale) times the true
    ones, except for the whole laps of the revisit pairs, which are exact:
        measured = 2 pi laps + (1 + scale) (theta_j - theta_i - 2 pi laps).
    The scale is fitted when revisit pairs exist (fit_scale), else fixed.
    Measurements of kind 1 (consecutive frames around the starts and stops
    of the moves) have their own scale, always fitted: they carry the timing
    of the moves (when the platform starts, reaches speed, stops), which no
    scale bias changes.
    Returns the model, the scale, the residual rms [deg], the residuals and
    the scale of the kind-1 measurements."""
    i = np.array([m[0] for m in measurements])
    j = np.array([m[1] for m in measurements])
    measured = np.array([m[2] for m in measurements])
    base = np.array([m[3] for m in measurements], dtype=np.float64)
    laps = 2 * np.pi * np.array([m[4] for m in measurements], dtype=np.float64)
    dense = np.array([len(m) > 5 and m[5] == 1 for m in measurements])
    starts = len(motion.starts)
    free = [0, 1] + ([2] if fit_total else []) + list(range(3, 3 + starts)) + ([3 + starts] if fit_scale else []) \
        + ([4 + starts] if dense.any() else [])
    deltas = np.array([1e-4, 1e-3, 1e-4] + [1e-3] * starts + [1e-4, 1e-4])
    limits = np.array([0.0, 0.5, 0.2] + [1.0] * starts + [0.01, 0.1])

    def residual(values):
        model = motion.from_vector(values[:-2]).angle(times)
        factor = 1.0 + np.where(dense, values[-1], values[-2])
        return laps + factor * (model[j] - model[i] - laps) - measured

    values = np.append(motion.from_vector(motion.vector()).vector(), [scale, dense_scale])
    # Three classes with very different noise (long pairs: sway and partial
    # views, degrees; revisits: about 1 deg; consecutive frames: about 0.2
    # deg): each is weighted by the inverse of its own robust variance.
    classes = np.where(dense, 2, np.where(laps > 0, 1, 0))

    def class_weights(r):
        weights = np.empty_like(r)
        for c in (0, 1, 2):
            members = classes == c
            if not members.any():
                continue
            spread = max(1.4826 * np.median(np.abs(r[members])), np.radians(0.05 if c == 2 else 0.2))
            weights[members] = base[members] * np.minimum(1.0, 2.0 * spread / np.maximum(np.abs(r[members]), 1e-12)) \
                / spread ** 2
        return weights / np.median(weights)

    weights = class_weights(residual(values))
    for _ in range(iterations):
        r = residual(values)
        jacobian = np.zeros((len(r), len(free)))
        for c, k in enumerate(free):
            shifted = values.copy()
            shifted[k] += deltas[k]
            jacobian[:, c] = (residual(shifted) - r) / deltas[k]
        sw = np.sqrt(weights)
        step = np.linalg.lstsq(jacobian * sw[:, None], -r * sw, rcond=None)[0]
        bound = limits[free].copy()
        bound[0] = 0.2 * values[0]                               # speed: at most 20 % per step
        step = np.clip(step, -bound, bound)
        values[free] += step
        values[0] = max(values[0], 1e-3)
        values[1] = float(np.clip(values[1], 0.05, 2.5))         # at 0 the derivative would vanish; a longer
        #                                                        ramp only mimics a stop between moves
        values[2] = max(values[2], 1e-3)
        values[-2] = float(np.clip(values[-2], -0.1, 0.1))     # long pairs: a few per cent short
        values[-1] = float(np.clip(values[-1], -0.9, 0.2))     # consecutive frames: far shorter at low speed
        values = np.append(motion.from_vector(values[:-2]).vector(), values[-2:])
        r = residual(values)
        weights = class_weights(r)
        if np.all(np.abs(step) < 1e-7):
            break
    r = residual(values)
    return (motion.from_vector(values[:-2]), float(values[-2]), float(np.degrees(np.sqrt(np.mean(r ** 2)))), r,
            float(values[-1]))


def fit_correction(times, measurements, model, factors, classes, spacing, smoothing, iterations=6):
    """Free-form correction c(t) of the motion model, piecewise linear with a
    knot every 'spacing' seconds, fitted to relative turns (i, j, measured
    [rad], weight, laps, ...) with a per-measurement scale factor (the
    registration of every class of pairs reads the turns short by its own
    fraction), class weights (inverse robust variance per class) and a
    penalty on the second differences of c ('smoothing').

    The stepper profile assumes the platform follows the motor: the same
    speed in every lap, the same ramps, stops of the commanded length. Under
    load the motor can lose steps (the platform then falls behind and the
    lap takes longer), and the choice between one move and several can be
    wrong; the angles are then off by degrees in places, which shifts the
    arms and hands by centimetres and distorts the fused arms (one thinner,
    one thicker in simulation). The correction follows any such deviation
    that the pairs see, and stays near zero where the model fits; the short
    pairs between neighbouring samples give it the local speed, the long
    pairs and the revisits the scale.
    Returns (knots, values [rad]) and the residuals after the correction."""
    i = np.array([m[0] for m in measurements])
    j = np.array([m[1] for m in measurements])
    measured = np.array([m[2] for m in measurements])
    base = np.array([m[3] for m in measurements], dtype=np.float64)
    laps = 2 * np.pi * np.array([m[4] for m in measurements], dtype=np.float64)
    knots = np.arange(times.min(), times.max() + spacing, spacing)
    count = len(knots)
    if count < 4:
        return (None, None), None

    def basis(t):
        cell = np.clip(np.searchsorted(knots, t, side="right") - 1, 0, count - 2)
        fraction = np.clip((t - knots[cell]) / (knots[cell + 1] - knots[cell]), 0.0, 1.0)
        matrix = np.zeros((len(t), count))
        matrix[np.arange(len(t)), cell] = 1.0 - fraction
        matrix[np.arange(len(t)), cell + 1] = fraction
        return matrix

    angles = model.angle(times)
    start = laps + factors * (angles[j] - angles[i] - laps) - measured
    design = factors[:, None] * (basis(times[j]) - basis(times[i]))
    second = np.diff(np.eye(count), n=2, axis=0)
    regular = smoothing * second.T @ second + 1e-6 * np.eye(count)
    values = np.zeros(count)
    residual = start.copy()
    for _ in range(iterations):
        weights = np.empty_like(residual)
        for c in np.unique(classes):
            members = classes == c
            floor = np.radians(0.05 if c == 2 else 0.2)
            spread = max(1.4826 * np.median(np.abs(residual[members])), floor)
            weights[members] = base[members] * np.minimum(
                1.0, 2.0 * spread / np.maximum(np.abs(residual[members]), 1e-12)) / spread ** 2
        weights /= np.median(weights)
        normal = design.T @ (design * weights[:, None]) + regular
        values = np.linalg.solve(normal, -design.T @ (weights * start))
        residual = start + design @ values
    values -= values[0]
    return (knots, values), residual


class FreeMotion:
    """Platform angle versus time [rad] without a motor model: a baseline (the
    chained angles, interpolated) plus a piecewise linear correction fitted
    to all the pair measurements (see fit_free). Same interface as Motion
    for the rest of the program (angle, correction, describe)."""

    def __init__(self, base_times, base_angles, correction=None):
        self.base_times = np.asarray(base_times, dtype=np.float64)
        self.base_angles = np.asarray(base_angles, dtype=np.float64)
        self.correction = correction

    def angle(self, t):
        base = np.interp(t, self.base_times, self.base_angles)
        if self.correction is None:
            return base
        knots, values = self.correction
        return base + np.interp(t, knots, values)

    def describe(self):
        return "angles solved from all the pairs, no motor model"


class MeanMotion:
    """Mean of two angle solutions that agree (motor model with its correction,
    and the model-free solution): their errors are partly independent, so
    the mean is more accurate than either (simulation: 1.2 to 1.6 deg rms
    against 1.2 to 2.0 for the better and worse of the two)."""

    def __init__(self, first, second):
        self.parts = (first, second)
        self.correction = None

    def angle(self, t):
        return 0.5 * (self.parts[0].angle(t) + self.parts[1].angle(t))

    def describe(self):
        return "mean of the motor model with its correction and the model-free solution"


def measurement_classes(measurements):
    """Class of every measurement: 0 long pair, 1 revisit (whole laps apart),
    2 consecutive frames around a start or stop, 3 neighbouring samples."""
    kinds = np.array([m[5] if len(m) > 5 else 0 for m in measurements])
    laps = np.array([m[4] for m in measurements])
    return np.where(kinds == 1, 2, np.where(kinds == 3, 3, np.where(laps > 0, 1, 0)))


def fit_free(times, measurements, baseline, spacing, smoothing, fit_long_scale, iterations=15):
    """Angles from the pairs alone: baseline(t) plus c(t), piecewise linear
    with a knot every 'spacing' seconds, fitted together with the scale of
    every class of registration (Gauss-Newton, Huber weights per class,
    penalty 'smoothing' on the second differences of c).

    Each class of pairs reads the turns short by its own fraction (long pairs
    1 to 3 %, neighbouring samples and consecutive frames much more at low
    speed): measured = 2 pi laps + (1 + s_class) (true - 2 pi laps). The
    revisit pairs share the scale of the long pairs and fix it, through their
    whole laps; without revisits that scale stays 0 (the angles may then be
    short by 1 to 3 %). Nothing is assumed about the motor: the solution
    follows lost steps, stalls, stops and a recording that ends while the
    platform still turns.
    Returns the FreeMotion-ready correction (knots, values), the scales
    {group: s} (group 0: long pairs and revisits, 2: consecutive frames,
    3: neighbouring samples), and the residuals [rad]."""
    i = np.array([m[0] for m in measurements])
    j = np.array([m[1] for m in measurements])
    measured = np.array([m[2] for m in measurements])
    base = np.array([m[3] for m in measurements], dtype=np.float64)
    laps = 2 * np.pi * np.array([m[4] for m in measurements], dtype=np.float64)
    classes = measurement_classes(measurements)
    groups = np.where(classes == 1, 0, classes)
    knots = np.arange(times.min(), times.max() + spacing, spacing)
    count = len(knots)
    cell = np.clip(np.searchsorted(knots, times, side="right") - 1, 0, count - 2)
    fraction = np.clip((times - knots[cell]) / spacing, 0.0, 1.0)
    basis = np.zeros((len(times), count))
    basis[np.arange(len(times)), cell] = 1.0 - fraction
    basis[np.arange(len(times)), cell + 1] = fraction
    design = basis[j] - basis[i]
    base_turn = baseline(times[j]) - baseline(times[i])

    free_groups = [g for g in (0, 2, 3) if np.any(groups == g) and (g != 0 or fit_long_scale)]
    scale = {0: 0.0, 2: 0.0, 3: 0.0}
    for g in (2, 3):
        members = (groups == g) & (np.abs(base_turn) > np.radians(0.5))
        if members.sum() >= 5:
            scale[g] = float(np.clip(np.median(measured[members] / base_turn[members]) - 1.0, -0.9, 0.5))
    values = np.zeros(count)
    second = np.diff(np.eye(count), n=2, axis=0)
    regular = smoothing * second.T @ second + 1e-6 * np.eye(count)

    def residual_of(values, scale):
        turn = base_turn + design @ values
        factor = 1.0 + np.array([scale[g] for g in groups])
        return laps + factor * (turn - laps) - measured, turn, factor

    residual, turn, factor = residual_of(values, scale)
    for _ in range(iterations):
        weights = np.empty_like(residual)
        for c in np.unique(classes):
            members = classes == c
            floor = np.radians(0.05 if c >= 2 else 0.2)
            spread = max(1.4826 * np.median(np.abs(residual[members])), floor)
            weights[members] = base[members] * np.minimum(
                1.0, 2.0 * spread / np.maximum(np.abs(residual[members]), 1e-12)) / spread ** 2
        weights /= np.median(weights)
        jacobian = np.hstack([factor[:, None] * design] +
                             [np.where(groups == g, turn - laps, 0.0)[:, None] for g in free_groups])
        penalty = np.zeros((jacobian.shape[1], jacobian.shape[1]))
        penalty[:count, :count] = regular
        unknowns = np.concatenate([values, [scale[g] for g in free_groups]])
        normal = jacobian.T @ (jacobian * weights[:, None]) + penalty
        gradient = jacobian.T @ (weights * residual) + penalty @ np.concatenate([unknowns[:count],
                                                                                 np.zeros(len(free_groups))])
        step = np.linalg.solve(normal, -gradient)
        values = values + step[:count]
        for k, g in enumerate(free_groups):
            low, high = (-0.1, 0.1) if g == 0 else (-0.9, 0.5)
            scale[g] = float(np.clip(scale[g] + step[count + k], low, high))
        residual, turn, factor = residual_of(values, scale)
        if np.abs(step).max() < 1e-8:
            break
    values = values - np.interp(times.min(), knots, values)
    return (knots, values), scale, residual


def robust_cost(residual, base, scale):
    """Mean Huber loss with a fixed scale, to compare two motion models on the same data."""
    a = np.abs(residual) / scale
    loss = np.where(a < 2.0, 0.5 * a ** 2, 2.0 * a - 2.0)
    return float(np.sum(base * loss) / np.sum(base))


def joint_axis_fit(samples, pairs, turns, center, iterations=(8, 6, 6), distances=(0.04, 0.025, 0.015)):
    """Axis position (x, y) and one turn per pair, fitted together.

    Every pair (i, j) says: sample j = sample i turned by its own angle about
    the same vertical axis. With pairs 20 to 60 degrees apart the axis is well
    determined (a wrong axis leaves a shift (I - R) e that grows with the
    turn), while the per-pair angles absorb sway. Gauss-Newton with a Tukey
    kernel; the shared axis is solved through the Schur complement.
    Returns axis, turns [rad], overlap per pair.
    """
    center = center.copy()
    turns = np.array(turns, dtype=np.float64)
    progress = Progress("axis fit", sum(iterations))
    done = 0
    for distance, count in zip(distances, iterations):
        for _ in range(count):
            pivot = np.append(center, 0.0)
            c_matrix = np.zeros((2, 2))
            d_vector = np.zeros(2)
            blocks = []
            for n, (i, j) in enumerate(pairs):
                moved = transform_points(yaw_transform(turns[n], pivot), samples.points[i])
                index, gap = samples.targets[j].nearest(moved)
                use = gap < distance
                if use.sum() < 30:
                    blocks.append(None)
                    continue
                p = moved[use]
                q = samples.targets[j].points[index[use]]
                normal = samples.targets[j].normals[index[use]]
                residual = np.einsum("ij,ij->i", normal, p - q)
                weight = np.where(np.abs(residual) < distance, (1.0 - (residual / distance) ** 2) ** 2, 0.0)
                lever = p - pivot
                j_turn = normal[:, 1] * lever[:, 0] - normal[:, 0] * lever[:, 1]
                c, s = np.cos(turns[n]), np.sin(turns[n])
                i_minus_r = np.array([[1 - c, s], [-s, 1 - c]])
                j_axis = normal[:, :2] @ i_minus_r
                a = np.sum(weight * j_turn ** 2) + 1e-12
                b = np.sum(weight * j_turn * residual)
                coupling = (weight * j_turn) @ j_axis
                c_matrix += j_axis.T @ (j_axis * weight[:, None])
                d_vector += j_axis.T @ (weight * residual)
                blocks.append((a, b, coupling))
            reduced_c, reduced_d = c_matrix.copy(), d_vector.copy()
            for block in blocks:
                if block is None:
                    continue
                a, b, coupling = block
                reduced_c -= np.outer(coupling, coupling) / a
                reduced_d -= coupling * b / a
            step_axis = -np.linalg.solve(reduced_c + 1e-9 * np.eye(2), reduced_d)
            length = np.linalg.norm(step_axis)
            if length > 0.01:
                step_axis *= 0.01 / length                    # at most 1 cm per iteration
            for n, block in enumerate(blocks):
                if block is None:
                    continue
                a, b, coupling = block
                turns[n] += float(np.clip(-(b + coupling @ step_axis) / a, -0.03, 0.03))
            center = center + step_axis
            done += 1
            if done % 5 == 0:
                progress.step(done)
    pivot = np.append(center, 0.0)
    fitness = np.array([evaluate(samples.points[i], samples.targets[j], yaw_transform(t, pivot), samples.fine)[0]
                        for (i, j), t in zip(pairs, turns)])
    return center, turns, fitness


def revisit_pairs(samples, predicted, lap, axis, sense, args):
    """Pairs one or more full laps apart (the body faces the same way again).

    predicted: model angle of every sample [rad], increasing. For every
    sample i the sample j nearest to exactly 'lap' turns later is taken (if
    within --revisit-window), up to --revisit-pairs pairs spread over the
    run. They are registered with a small search around the predicted
    remainder; the measured turn is lap * 360 deg plus the registered rest,
    so these pairs fix the speed and the total over many laps.
    Returns (i, j, measured turn [rad] in the positive sense, fitness, lap, Pair).
    """
    pivot = np.append(axis, 0.0)
    target = 2 * np.pi * lap
    window = np.radians(args.revisit_window)
    candidates = []
    for i in range(len(predicted)):
        gap = np.abs(predicted - predicted[i] - target)
        gap[:i + 1] = np.inf
        j = int(np.argmin(gap))
        if gap[j] <= window:
            candidates.append((i, j))
    if not candidates:
        return []
    chosen = [candidates[int(k)] for k in
              np.unique(np.round(np.linspace(0, len(candidates) - 1, min(args.revisit_pairs, len(candidates)))))]
    result = []
    for i, j in chosen:
        rest = sense * (predicted[j] - predicted[i] - target)             # raw (sensor) sense
        pair = samples.register_turn(i, j, np.degrees(rest) + np.arange(-12.0, 12.1, 4.0), pivot, True)
        if pair.fitness < 0.5:
            continue
        measured = target + sense * np.radians(wrapped_degrees(yaw_of(pair.transform)))
        result.append((i, j, measured, pair.fitness, lap, pair))
    return result


def boundary_pairs(model, frame_times, usable, load_cloud, axis, sense, args):
    """Consecutive frames (full frame rate) around every start and stop of the
    platform, registered with one unknown (the turn about the axis).
    Returns (frame a, frame b, measured turn [rad] in the positive sense, fitness)."""
    margin = args.boundary_margin
    ramp = max(model.ramp, 0.5)
    windows = []
    for start, angle in zip(model.starts, model.moves()):
        end = start + model.duration(angle)
        windows += [(start - margin, start + ramp + margin), (end - ramp - margin, end + margin)]
    windows.sort()
    merged = [list(windows[0])]
    for low, high in windows[1:]:
        if low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    frames = [k for k in np.flatnonzero(usable) if any(low <= frame_times[k] <= high for low, high in merged)]
    if len(frames) < 4:
        return []
    clouds = {k: load_cloud(k) for k in frames}
    pivot = np.append(axis, 0.0)
    progress = Progress("frames around the starts and stops", len(frames))
    result = []
    for n, (a, b) in enumerate(zip(frames[:-1], frames[1:])):
        if b - a > 2:
            continue
        source = np.asarray(clouds[a].points)
        target = Target(clouds[b])
        expected = sense * (model.angle(np.array([frame_times[b]]))[0] - model.angle(np.array([frame_times[a]]))[0])
        best = None
        for start in expected + np.radians([-2.0, 0.0, 2.0]):
            angle, fitness, rmse = icp_turn_about(source, target, start, pivot)
            if best is None or fitness > best[1] + 0.02 or (fitness > best[1] - 0.02 and rmse < best[2]):
                best = (angle, fitness, rmse)
        if best[1] >= 0.5:
            result.append((a, b, sense * best[0], best[1]))
        if (n + 1) % 50 == 0 or n == len(frames) - 2:
            progress.step(n + 1)
    return result


def plausible_sequence(model, max_stop=3.0, max_ramp=2.4):
    """A fitted sequence of moves that a stepper could have made: ramps below
    the fit's bound, stops between moves of at most a few seconds, and a total
    that needs every move but no more (the last move longer than 10 deg, and
    not longer than one command plus 10 deg: in the 2026-10-02 recording a
    fit of 3 moves put 622 deg into the last one)."""
    moves = model.moves()
    stops = [model.starts[m + 1] - (model.starts[m] + model.duration(a)) for m, a in enumerate(moves[:-1])]
    if model.ramp > max_ramp or any(stop < -0.05 or stop > max_stop for stop in stops):
        return False
    return np.radians(10.0) < moves[-1] <= model.move + np.radians(10.0)


def estimate_motion(samples, times, phases, angles, axis, commanded_deg, args,
                    frame_times=None, usable=None, load_cloud=None, local_pairs=None):
    """Motion model of the platform fitted to long-baseline turns about the fitted axis.

    Consecutive frames turn only a few degrees, and the LiDAR samples the body
    along the same beams in both, which pulls ICP towards 'no motion' (a bias
    of about 0.2 deg per pair in simulation). Pairs 20 to 60 degrees apart
    carry the same bias on a much larger turn, and the model has few
    parameters, so the fitted angles are far more accurate than chained or
    solved per-pair turns. Pairs one or more laps apart (revisits) are added
    lap by lap, each time predicted by the model fitted so far; their whole
    laps are exact, so they calibrate the scale of the registered turns (see
    fit_motion) and fix the speed and the total over the whole run. Any
    rotation works: a partial turn has no revisits (long pairs alone, scale
    not calibrated), several laps have many.

    Two hypotheses are fitted when the turn is longer than --move-deg: one
    continuous move, and a sequence of moves of --move-deg with stops in
    between (what girogirotondo_timer.m does); the data decide.
    """
    sense = np.sign(angles[-1] - angles[0]) or 1.0
    signed = angles * sense
    span = float(np.degrees(signed.max() - signed.min()))
    pair_min = min(args.pair_min_deg, 0.4 * span)
    pair_max = max(pair_min + 1.0, min(args.pair_max_deg, span))
    candidates = []
    for i in range(len(signed)):
        for j in range(i + 1, len(signed)):
            difference = np.degrees(signed[j] - signed[i])
            if pair_min <= difference <= pair_max:
                candidates.append((i, j))
    if len(candidates) < 10:
        return None
    candidates = candidates[::max(1, len(candidates) // args.max_pairs)]
    print(f"motion model: {len(candidates)} long pairs ({pair_min:.0f} to {pair_max:.0f} deg); "
          f"joint fit of the axis and the pair turns")
    turns = np.array([sense * (signed[j] - signed[i]) for i, j in candidates])
    axis, turns, fitness = joint_axis_fit(samples, candidates, turns, np.asarray(axis, dtype=np.float64))
    long_pairs = [(i, j, sense * turn, fit, 0) for (i, j), turn, fit in zip(candidates, turns, fitness)
                  if fit >= 0.4]
    if len(long_pairs) < 10:
        return None

    # Hypotheses, each first fitted to the chained angles (approximate, a few
    # per cent short, but with the stops between moves visible), then to the
    # pairs: one continuous move, and sequences of --move-deg moves (the
    # number of moves from the chained total, one fewer or one more).
    moving = np.flatnonzero(phases == 2)
    t_on, t_off = (times[moving[0]], times[moving[-1]]) if len(moving) >= 2 else (times[0], times[-1])
    steps = np.diff(signed) / np.maximum(np.diff(times), 1e-6)
    inside = (times[1:] > t_on) & (times[:-1] < t_off)
    cruise = 1.03 * np.median(steps[inside]) if inside.sum() >= 3 else np.radians(span) / max(t_off - t_on, 0.5)
    cruise = max(cruise, np.radians(1.0))
    chain_pairs = [(0, k, signed[k] - signed[0], 1.0, 0) for k in range(1, len(signed))]
    # One continuous move, fitted to the long pairs from three starting points
    # (the fit to the chained angles, and constant speed over the turn with
    # the chained total raised by 3 and 8 %, since the chained angles are
    # short): from a poor start the fit can end with the ramp at its bound
    # and the start and total far off.
    long_base = np.array([m[3] for m in long_pairs])
    initial = [fit_motion(times, chain_pairs, Motion(cruise, 0.5, [t_on - 0.25], np.radians(span) * 1.02, None),
                          True)[0]]
    for factor in (1.03, 1.08):
        total = np.radians(span) * factor
        initial.append(Motion(total / max(t_off - t_on - 0.5, 1.0), 0.5, [t_on - 0.25], total, None))
    one, one_cost = None, np.inf
    for start_model in initial:
        fitted = fit_motion(times, long_pairs, start_model, True)
        cost = robust_cost(fitted[3], long_base, np.radians(1.0))
        if cost < one_cost:
            one, one_cost = fitted[0], cost
    hypotheses = [one]
    all_times = times
    move = np.radians(args.move_deg) if args.move_deg > 0 else None
    if move is not None:
        # Sequences of moves, started from the continuous fit (its start,
        # speed and ramp) with several stop lengths: a fit from a poor start
        # ends in a wrong minimum (long ramps instead of stops).
        nominal = int(np.ceil(one.total / move - 0.02))
        counts = {nominal - 1, nominal, nominal + 1}
        if commanded_deg is not None:
            counts.add(int(np.ceil(commanded_deg / np.degrees(move) - 1e-6)))   # the commands actually sent
        for count in sorted(counts):
            if count < 2:
                continue
            total = float(np.clip(one.total, (count - 1) * move + np.radians(5.0), count * move))
            best, best_cost = None, np.inf
            for stop in (0.25, 0.75, 1.5, 3.0):
                starts = [one.starts[0]]
                for _ in range(1, count):
                    starts.append(starts[-1] + one.duration(move) + stop)
                fitted = fit_motion(times, long_pairs, Motion(one.speed, one.ramp, starts, total, move).ordered(),
                                    True)
                spread = max(1.4826 * np.median(np.abs(fitted[3])), np.radians(0.2))
                cost = robust_cost(fitted[3], long_base, spread)
                if cost < best_cost:
                    best, best_cost = fitted[0], cost
            hypotheses.append(best)

    state = {"dense_scale": 0.0}

    def fit_hypotheses(measurements, hypotheses, scale, fit_scale):
        """Fit every hypothesis; the best by robust cost (a sequence must beat the
        continuous move by 5 %, so that a continuous rotation is not split)."""
        fitted = [fit_motion(all_times, measurements, h, True, scale, fit_scale) for h in hypotheses]
        base = np.array([m[3] for m in measurements])
        spread = max(1.4826 * np.median(np.abs(fitted[0][3])), np.radians(0.2))
        costs = [robust_cost(f[3], base, spread) * (1.0 if n == 0 else 1.0 / 0.95) for n, f in enumerate(fitted)]
        for n, f in enumerate(fitted[1:], start=1):
            if not plausible_sequence(f[0]):
                costs[n] = np.inf               # long ramps or long stops: a wrong minimum, not a stepper
        best = int(np.argmin(costs))
        model, scale, rms, residual, dense_scale = fitted[best]
        label = "one continuous move" if best == 0 else f"{len(model.starts)} moves with stops"
        state["dense_scale"] = dense_scale
        return [f[0] for f in fitted], model, scale, rms, residual, label

    hypotheses, model, scale, rms, residual, label = fit_hypotheses(long_pairs, hypotheses, 0.0, False)
    print(f"  long pairs only: {label}; {model.describe()}")

    # Model-free solution (see fit_free): the chained angles as baseline,
    # corrected by every pair. It predicts the revisit pairs (a wrong motor
    # model would predict them wrongly, and wrong revisits spoil everything
    # after them) and is the fallback when the motor model does not fit.
    local = []
    for pair in local_pairs or []:
        a, b = pair.source, pair.target
        expected = signed[b] - signed[a]
        observed = expected + np.radians(wrapped_degrees(np.degrees(sense * yaw_of(pair.transform)) -
                                                         np.degrees(expected)))
        local.append((a, b, observed, max(pair.fitness, 1e-3), 0, 3))
    baseline = lambda t: np.interp(t, times, signed)

    def solve_free(measurements, fit_scale, solve_times):
        correction, scales, free_residual = fit_free(solve_times, measurements + local, baseline,
                                                     max(args.correction_spacing, 0.5),
                                                     args.correction_smoothing, fit_scale)
        return FreeMotion(times, signed, correction), scales, free_residual[:len(measurements)]

    free, free_scales, _ = solve_free(long_pairs, False, times)

    revisits, lap = [], 1
    while True:
        predicted = free.angle(times)
        if (predicted.max() - predicted.min()) + np.radians(args.revisit_window) < 2 * np.pi * lap:
            break
        new = revisit_pairs(samples, predicted, lap, axis, sense, args)
        # A revisit far from the prediction locked onto a wrong pose (the body
        # seen from the back looks like the front): dropped.
        new = [r for r in new
               if abs(r[2] - (predicted[r[1]] - predicted[r[0]])) <= np.radians(args.revisit_window)]
        revisits += new
        measurements = long_pairs + [r[:5] for r in revisits]
        fit_scale = len(revisits) >= 8
        free, free_scales, _ = solve_free(measurements, fit_scale, times)
        hypotheses, model, scale, rms, residual, label = fit_hypotheses(measurements, hypotheses, scale, fit_scale)
        print(f"  lap {lap}: {len(new)} revisit pairs; total so far {np.degrees(predicted.max() - predicted.min()):.1f} "
              f"deg; registration scale {100 * free_scales[0]:+.2f} %; motor model: {label}, "
              f"speed {np.degrees(model.speed):.3f} deg/s")
        lap += 1
    measurements = long_pairs + [r[:5] for r in revisits]
    fit_scale = len(revisits) >= 8

    # Timing of the starts and stops from consecutive frames at the full frame
    # rate. With several moves, the revisit pairs fix the scale and the
    # period of the moves, but the speed trades off against the ramps and
    # the stops unless these are measured; the same pairs also place the
    # start and the end of a single move.
    dense = []
    if load_cloud is not None and args.boundary_margin > 0:
        dense = boundary_pairs(model, frame_times, usable, load_cloud, axis, sense, args)
        if dense:
            index = {}
            extra_times = []
            for a, b, _, _ in dense:
                for k in (a, b):
                    if k not in index:
                        index[k] = len(times) + len(extra_times)
                        extra_times.append(frame_times[k])
            all_times = np.concatenate([times, extra_times])
            measurements = measurements + [(index[a], index[b], turn, fit, 0, 1) for a, b, turn, fit in dense]
            hypotheses, model, scale, rms, residual, label = fit_hypotheses(measurements, hypotheses, scale,
                                                                            fit_scale)
            print(f"  starts and stops: {len(dense)} consecutive-frame pairs; motor model: {label}; "
                  f"speed {np.degrees(model.speed):.3f} deg/s, ramp {model.ramp:.2f} s")

    # A commanded turn (capture.json or --turn-deg) is exact for a stepper
    # through a gear unless it stalls: use it when the data agree.
    fitted_total = float(np.degrees(model.total))
    total_fixed = False
    if commanded_deg is not None and abs(fitted_total - commanded_deg) <= args.loop_tolerance:
        model.total = np.radians(commanded_deg)
        model, scale, rms, residual, state["dense_scale"] = fit_motion(all_times, measurements, model.ordered(),
                                                                     False, scale, fit_scale)
        total_fixed = True
    # Free-form correction on top of the stepper profile (lost steps, wrong
    # number of moves, speed changes): see fit_correction.
    correction_rms = correction_max = None
    if args.correction_spacing > 0 and len(measurements) >= 50:
        classes = measurement_classes(measurements)
        factors = 1.0 + np.where(classes == 2, state["dense_scale"], scale)
        combined, combined_classes, combined_factors = list(measurements), classes, factors
        if local:
            predicted = np.array([model.angle(all_times[[b]])[0] - model.angle(all_times[[a]])[0]
                                  for a, b, _, _, _, _ in local])
            observed = np.array([m[2] for m in local])
            moving = np.abs(predicted) > np.radians(0.5)
            if moving.sum() >= 10:
                local_scale = float(np.clip(np.median(observed[moving] / predicted[moving]) - 1.0, -0.9, 0.2))
                combined = combined + local
                combined_classes = np.concatenate([classes, np.full(len(local), 3)])
                combined_factors = np.concatenate([factors, np.full(len(local), 1.0 + local_scale)])
        correction, corrected = fit_correction(all_times, combined, model, combined_factors, combined_classes,
                                               args.correction_spacing, args.correction_smoothing)
        if correction[0] is not None:
            model.correction = correction
            residual = corrected[:len(measurements)]
            rms = float(np.degrees(np.sqrt(np.mean(residual ** 2))))
            values = np.degrees(correction[1])
            inside = (correction[0] >= model.starts[0]) & (correction[0] <= model.end())
            deviation = values[inside] - np.median(values[inside]) if inside.any() else values
            correction_rms = float(np.sqrt(np.mean(deviation ** 2)))
            correction_max = float(np.abs(deviation).max())
            print(f"  motor model with free-form correction (knots every {args.correction_spacing:g} s): "
                  f"correction rms {correction_rms:.2f} deg, max {correction_max:.2f} deg; "
                  f"pair residual rms {rms:.2f} deg")

    # The model-free solution with every pair, and the choice between the two.
    free, free_scales, free_residual = solve_free(measurements, fit_scale, all_times)
    free_rms = float(np.degrees(np.sqrt(np.mean(free_residual ** 2))))
    moving_samples = (times >= t_on) & (times <= t_off)
    if moving_samples.sum() < 3:
        moving_samples = np.ones(len(times), dtype=bool)
    gap = np.degrees(model.angle(times) - free.angle(times))
    gap -= np.median(gap[~moving_samples]) if np.any(~moving_samples) else gap[0]
    gap_rms = float(np.sqrt(np.mean(gap[moving_samples] ** 2)))
    gap_max = float(np.abs(gap[moving_samples]).max())
    print(f"  model-free solution: registration scale {100 * free_scales[0]:+.2f} %, "
          f"pair residual rms {free_rms:.2f} deg; motor model minus model-free: rms {gap_rms:.2f} deg, "
          f"max {gap_max:.2f} deg")
    reasons = []
    if args.angle_source == "free":
        reasons.append("--angle-source free")
    if len(model.starts) > 1 and not plausible_sequence(model):
        reasons.append("the fitted sequence of moves is not one a stepper makes")
    if abs(scale) >= 0.099:
        reasons.append("the registration scale of the motor model is at its bound")
    if gap_rms > args.model_agreement or gap_max > 3 * args.model_agreement:
        reasons.append(f"it differs from the model-free solution by more than {args.model_agreement:g} deg rms "
                       f"or {3 * args.model_agreement:g} deg max")
    if args.angle_source == "profile":
        reasons = []
    if reasons:
        print("  angles from the model-free solution: motor model rejected (" + "; ".join(reasons) + ")")
        chosen, label = free, "model-free solution"
        residual, rms, scale = free_residual, free_rms, free_scales[0]
    elif args.angle_source == "profile":
        print("  angles from the motor model with its correction (--angle-source profile)")
        chosen, label = model, "motor model with correction"
    else:
        print("  angles: mean of the motor model with its correction and the model-free solution "
              "(the two agree)")
        chosen, label = MeanMotion(model, free), "mean of motor model and model-free solution"
        scale = free_scales[0]
    if model.end() > all_times.max() - 0.5 and t_off >= times[-1] - 1.0:
        print("  WARNING: the recording ends while the platform still turns: the end of the turn is missing "
              "(record longer); the angles up to the end of the recording are still measured")

    revisit_residual = np.degrees(residual[len(long_pairs):len(long_pairs) + len(revisits)]) \
        if revisits else np.array([])
    solved = np.array([signed[j] - signed[i] - turn for i, j, turn, _, _ in long_pairs])
    middle = np.array([0.5 * (all_times[m[0]] + all_times[m[1]]) for m in measurements])
    ordered = np.degrees(residual)[np.argsort(middle)]
    half = 7
    padded = np.pad(ordered, half, mode="edge")
    smoothed = np.array([np.median(padded[k:k + 2 * half + 1]) for k in range(len(ordered))])
    saved = measurements + local
    return {"model": chosen, "motor_model": model, "free_model": free, "label": label, "sense": sense,
            "axis": axis, "pair_rms_deg": rms,
            "solved_rms_deg": float(np.degrees(np.sqrt(np.mean(solved ** 2)))),
            "systematic_deviation_deg": float(np.abs(smoothed).max()),
            "pair_mid_times_s": np.sort(middle), "pair_residual_smoothed_deg": smoothed,
            "pair_list": [[float(all_times[m[0]]), float(all_times[m[1]]), float(np.degrees(m[2])), float(m[3]),
                           int(m[4]), int(m[5]) if len(m) > 5 else 0] for m in saved],
            "boundary_pairs": len(dense),
            "correction_rms_deg": correction_rms, "correction_max_deg": correction_max,
            "dense_scale": state["dense_scale"],
            "registration_scale": scale, "scale_calibrated": bool(fit_scale),
            "free_scales": {str(k): v for k, v in free_scales.items()},
            "motor_minus_free_rms_deg": gap_rms, "motor_minus_free_max_deg": gap_max,
            "motor_model_rejected": reasons,
            "pair_residual_deg": np.degrees(residual),
            "revisit_residual_rms_deg": float(np.sqrt(np.mean(revisit_residual ** 2))) if revisits else None,
            "revisit_pairs": [r[5] for r in revisits], "laps_with_revisits": lap - 1,
            "fitted_total_deg": fitted_total, "total_fixed_to_command": total_fixed,
            "pairs": len(long_pairs), "revisits": len(revisits),
            "params": {"speed_deg_s": float(np.degrees(model.speed)), "ramp_s": model.ramp,
                       "total_deg": float(np.degrees(model.total)), "starts_s": model.starts.tolist(),
                       "move_deg": None if model.move is None else float(np.degrees(model.move))}}


def constant_speed_fit(times, angles, phases):
    """theta = omega (t - t0) on the middle 80 % of the rotation; returns omega, t0, residual."""
    rotating = np.flatnonzero(phases == 2)
    if len(rotating) < 6:
        return None
    total = angles[rotating[-1]] - angles[rotating[0]]
    middle = rotating[(np.abs(angles[rotating] - angles[rotating[0]]) > 0.1 * abs(total))
                      & (np.abs(angles[rotating] - angles[rotating[0]]) < 0.9 * abs(total))]
    if len(middle) < 4:
        return None
    slope, intercept = np.polyfit(times[middle], angles[middle], 1)
    residual = angles[middle] - (slope * times[middle] + intercept)
    return {"deg_per_s": float(np.degrees(slope)), "t0": float(-intercept / slope),
            "residual_rms_deg": float(np.degrees(np.sqrt(np.mean(residual ** 2)))),
            "residual_max_deg": float(np.degrees(np.abs(residual).max()))}


def angle_plot(path, sample_times, sample_angles, times, used_angles, source, measured=None, profile_only=None,
               alternatives=None):
    """Top: the angles used (red), the chained angles of neighbouring samples
    (black dots; a few per cent short) and the stepper profile alone (gray,
    dashed). Bottom: everything minus the angles used: chained angles,
    stepper profile, and the other solutions ('alternatives': {label:
    angles}, e.g. the motor model with its correction, or the model-free
    solution). Curves that stay near zero agree with the angles used; a curve
    that drifts away by tens of degrees is a solution that was rejected."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    used_at_samples = np.interp(sample_times, times, used_angles)
    axes[0].plot(times, np.degrees(used_angles), lw=1.0, color="tab:red", label=f"used ({source})")
    axes[0].plot(sample_times, np.degrees(sample_angles), ".", ms=3, color="black", label="chained (neighbouring samples)")
    axes[1].plot(sample_times, np.degrees(sample_angles - used_at_samples), ".", ms=3, color="black")
    if profile_only is not None:
        axes[0].plot(times, np.degrees(profile_only), "--", lw=0.8, color="tab:gray", label="stepper profile alone")
        axes[1].plot(times, np.degrees(profile_only - used_angles), "--", lw=0.8, color="tab:gray")
    colors = iter(["tab:orange", "tab:green", "tab:purple"])
    for label, angles in (alternatives or {}).items():
        axes[1].plot(times, np.degrees(angles - used_angles), lw=1.2, color=next(colors), label=label)
    if alternatives:
        axes[1].legend(fontsize=8)
    if measured is not None:
        view_times, view_angles = measured
        axes[0].plot(view_times, np.degrees(view_angles), ".", ms=3, color="tab:blue", label="measured per view")
        axes[1].plot(view_times, np.degrees(view_angles - np.interp(view_times, times, used_angles)), ".", ms=3,
                     color="tab:blue")
    axes[0].set_ylabel("platform angle [deg]")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="tab:red", lw=0.8)
    axes[1].set_ylabel("minus the angles used [deg]")
    axes[1].set_xlabel("sensor time [s]")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return True


# ----------------------------------------------------------------------------
# Views and fusion
# ----------------------------------------------------------------------------

def reference_groups(count, groups):
    """Group label per view for the leave-group-out references: exact leave
    one out for few views, otherwise views spread over 'groups' groups at
    random (a fixed seed), so that no group misses one direction or one lap."""
    if groups <= 0 or count <= groups:
        return None
    return np.random.default_rng(0).permutation(count) % groups


def groupwise_rigid(clouds, args, label="view corrections", groups=None, reference_views=None):
    """Small bounded rigid correction of every view against the others (the
    person sways a little on the platform). The reference of a view is all
    the other views, or, with 'groups', all the views of the other groups,
    with a fixed point budget (reference_cloud: a voxel subsampling of the
    views of several laps reaches further into the noise and pulls the views
    outward).
    Returns corrected clouds, the RMS correction [mm] and, per view, the
    horizontal shift of its centroid (NaN where the correction was out of
    bounds)."""
    clouds = [copy.deepcopy(c) for c in clouds]
    moved = []
    shifts = np.full((len(clouds), 2), np.nan)
    progress = Progress(label, len(clouds))
    reference_all = [c.voxel_down_sample(0.01) for c in clouds]
    budget = int(np.mean([len(c.points) for c in reference_all]) * (reference_views or len(clouds)))
    targets = {}
    for k in range(len(clouds)):
        key = k if groups is None else int(groups[k])
        if key not in targets:
            reference = o3d.geometry.PointCloud()
            for other, cloud in enumerate(reference_all):
                if (other != k) if groups is None else (groups[other] != key):
                    reference += cloud
            if groups is None:
                targets.clear()
            targets[key] = Target(reference_cloud(reference, max_points=budget))
        target = targets[key]
        points = np.asarray(reference_all[k].points)
        result = icp_robust(points, target, np.eye(4), (0.04, 0.025, 0.015), (20, 15, 15), args.max_tilt)
        center = points.mean(axis=0)
        shift = np.linalg.norm(transform_points(result, center[None])[0] - center)
        if abs(np.degrees(yaw_of(result))) <= args.max_turn_correction and shift <= args.max_shift \
                and tilt_degrees(result) <= args.max_tilt:
            clouds[k].transform(result)
            moved.append(shift)
            shifts[k] = transform_points(result, center[None])[0, :2] - center[:2]
        if (k + 1) % 25 == 0 or k == len(clouds) - 1:
            progress.step(k + 1)
    return clouds, (1000 * float(np.sqrt(np.mean(np.square(moved)))) if moved else 0.0), shifts


def refine_view_angles(clouds, args, pivot, groups=None, reference_views=None):
    """Turn of every view about the platform axis, measured against the views
    of the other groups (1 degree of freedom, searched over +---angle-search
    deg, then refined), --angle-iterations passes.

    The motion model gives the angle of every frame from a few parameters;
    if the platform does not follow it (lost steps under load, a wrong
    choice between one move and several, a speed that changes), the views
    are off by up to several degrees, which displaces the arms and hands
    (0.3 to 0.5 m from the axis) by 1 to 4 cm, and the fused arms come out
    thinner (a 4 deg rms error removed about 25 % of the arm cross-section in
    simulation). The bounded rigid correction of each view (3 deg) cannot
    recover that; this search can, and it measures the platform angle
    directly from the data, view by view.
    Returns the corrected clouds and the correction per view [rad]: the
    refined platform angle is the model angle minus it."""
    clouds = [copy.deepcopy(c) for c in clouds]
    pivot = np.asarray(pivot, dtype=np.float64)
    total = np.zeros(len(clouds))
    coarse = np.radians(np.arange(-args.angle_search, args.angle_search + 1e-9, 1.0))
    fine_steps = np.radians(np.arange(-1.0, 1.0001, 0.25))
    rng = np.random.default_rng(0)
    for iteration in range(args.angle_iterations):
        reference_all = [c.voxel_down_sample(0.01) for c in clouds]
        budget = int(np.mean([len(c.points) for c in reference_all]) * (reference_views or len(clouds)))
        targets = {}
        deltas = np.zeros(len(clouds))
        reliable = np.ones(len(clouds), dtype=bool)
        progress = Progress(f"angle pass {iteration + 1}", len(clouds))
        for k in range(len(clouds)):
            key = k if groups is None else int(groups[k])
            if key not in targets:
                reference = o3d.geometry.PointCloud()
                for other, cloud in enumerate(reference_all):
                    if (other != k) if groups is None else (groups[other] != key):
                        reference += cloud
                if groups is None:
                    targets.clear()
                targets[key] = Target(reference_cloud(reference, max_points=budget))
            target = targets[key]
            points = np.asarray(reference_all[k].points)
            if len(points) < 200:
                reliable[k] = False
                continue
            if len(points) > 3000:
                points = points[rng.choice(len(points), 3000, replace=False)]
            relative = points[:, :2] - pivot[:2]

            def cost(delta):
                c, s_ = np.cos(delta), np.sin(delta)
                moved = points.copy()
                moved[:, 0] = pivot[0] + c * relative[:, 0] - s_ * relative[:, 1]
                moved[:, 1] = pivot[1] + s_ * relative[:, 0] + c * relative[:, 1]
                _, gap = target.nearest(moved)
                return float(np.mean(np.minimum(gap, args.angle_truncation) ** 2))

            costs = np.array([cost(d) for d in coarse])
            best = int(np.argmin(costs))
            if best in (0, len(coarse) - 1) or costs[best] > 0.9 * np.median(costs):
                reliable[k] = False          # at the edge of the search, or no clear minimum
                continue
            trial = coarse[best] + fine_steps
            fine = np.array([cost(d) for d in trial])
            m = int(np.clip(np.argmin(fine), 1, len(fine) - 2))
            a, b, c = fine[m - 1], fine[m], fine[m + 1]
            curvature = a - 2 * b + c
            step = 0.5 * (a - c) / curvature if curvature > 0 else 0.0
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
        print(f"  angle pass {iteration + 1}: correction rms {np.degrees(np.sqrt(np.mean(deltas ** 2))):.2f} deg, "
              f"max {np.degrees(np.abs(deltas).max()):.2f} deg; {int((~reliable).sum())} views without a clear "
              f"minimum (their neighbours' value used)")
    return clouds, total


def lap_times(times, angles):
    """Time of every whole lap of a monotonic angle curve [rad]: list of (lap, duration [s])."""
    signed = np.abs(angles - angles[0])
    laps = []
    previous = None
    for lap in range(1, int(signed.max() // (2 * np.pi)) + 1):
        crossing = np.interp(2 * np.pi * lap, signed, times)
        start = np.interp(2 * np.pi * (lap - 1), signed, times) if previous is None else previous
        laps.append((lap, float(crossing - start)))
        previous = crossing
    return laps


def components_2d(xy, radius):
    """Connected components of 2D points (neighbours closer than 'radius'):
    DBSCAN with one point per cluster is single linkage."""
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.column_stack([xy, np.zeros(len(xy))])))
    return np.asarray(cloud.cluster_dbscan(radius, 1, print_progress=False))


def segment_arms(points, args):
    """Arms of the consensus body when they hang free of the torso (A-pose).

    In every 2 cm horizontal slab between --limb-min-height and the top of
    the torso, the points split into connected components (2.5 cm); a
    component whose centroid is farther than --limb-min-offset from the torso
    axis is part of an arm, and belongs to the arm on its side of the body
    (sign along the lateral axis of the torso: the main axis of the chest
    slab). Returns a label per point (0 body, 1 and 2 the arms) and, per arm,
    the height of the highest slab where it is separate from the torso (it
    joins the shoulder above it); None when the arms touch the body."""
    labels = np.zeros(len(points), dtype=np.int64)
    chest = points[(points[:, 2] > 1.0) & (points[:, 2] < 1.4)]
    if len(chest) < 200:
        return labels, None
    axis_xy = np.median(chest[:, :2], axis=0)
    centered = chest[:, :2] - chest[:, :2].mean(axis=0)
    lateral = np.linalg.svd(centered, full_matrices=False)[2][0]
    tops = {1: None, 2: None}
    for low in np.arange(args.limb_min_height, 1.6, 0.02):
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
            if np.linalg.norm(centroid - axis_xy) < args.limb_min_offset:
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


def limb_refine(clouds, args, groups=None, reference_views=None):
    """Rigid correction of each free-hanging arm of every view against the
    arms of the other views (the arms drift on their own: held away from
    the body for minutes, they sink, swing and turn; the whole-body and
    height-slab corrections follow the torso, so the views of a drifting arm
    do not overlap, and the fused arm comes out thinner or blurred).

    The arms are segmented on the union of the views (segment_arms); every
    view point near an arm takes its label; the arm points of the view are
    registered (6 degrees of freedom, bounded by --limb-max-turn and
    --limb-max-shift) to the arm of the views of the other groups, and the
    correction fades out over --limb-blend below the shoulder, so the arm
    stays attached. --limb-iterations passes. Returns the corrected clouds
    and a report per arm (hand displacement per view)."""
    clouds = [copy.deepcopy(c) for c in clouds]
    report = []
    for iteration in range(args.limb_iterations):
        union = o3d.geometry.PointCloud()
        owner = []
        for k, cloud in enumerate(clouds):
            down = cloud.voxel_down_sample(0.008)
            union += down
            owner.append(np.full(len(down.points), k))
        owner = np.concatenate(owner)
        union_points = np.asarray(union.points)
        labels, tops = segment_arms(union_points, args)
        if tops is None:
            print("  arms not separate from the body (arms down along the torso?): no arm correction")
            return clouds, None
        index = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(union_points))
        index.knn_index()
        budget = None
        if reference_views:
            budget = int(np.mean([len(c.points) for c in clouds]) * reference_views)
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
            near = np.sqrt(squared.numpy()[:, 0]) < args.limb_capture
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
                                    (20, 15, 15), args.limb_max_turn)
                angle = float(np.degrees(np.arccos(np.clip((np.trace(result[:3, :3]) - 1) / 2, -1.0, 1.0))))
                centroid = source.mean(axis=0)
                shift = float(np.linalg.norm(transform_points(result, centroid[None])[0] - centroid))
                if angle > args.limb_max_turn or shift > args.limb_max_shift:
                    continue
                weights = np.clip((tops[side] - points[own, 2]) / args.limb_blend, 0.0, 1.0)
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
                               f"median {1000 * np.median(moves[ok]):.0f} mm, p90 {1000 * np.percentile(moves[ok], 90):.0f} mm")
        print(f"  arm pass {iteration + 1}: " + "; ".join(summary))
    return clouds, {"tops": tops, "hand_moves": report}


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




def frame_complete(mask, info, args, margin=64):
    """False if the frame lost too many columns (UDP packets of 16 columns)
    overall, or too many of the columns across the person. A lost packet
    elsewhere in the 360 deg sweep does not matter for the person. 'margin'
    covers the per-row pixel shift between measurement columns and image
    columns (up to 63 on this sensor)."""
    if info["columns_ok"] < args.min_columns:
        return False
    stamps = info["timestamps"]
    if stamps is None or info["columns_ok"] >= 0.999 or not mask.any():
        return True
    lost = stamps == 0
    person = np.flatnonzero(mask.any(axis=0))
    if person.size == 0:
        return True
    width = mask.shape[1]
    # columns spanned by the person (handles a person across the start of the sweep)
    span = np.zeros(width, dtype=bool)
    gaps = np.diff(np.concatenate([person, [person[0] + width]]))
    start = person[(np.argmax(gaps) + 1) % person.size]
    length = width - int(gaps.max()) + 1
    span[(start + np.arange(length)) % width] = True
    widened = span.copy()
    for shift in range(-margin, margin + 1, 8):
        widened |= np.roll(span, shift)
    return lost[widened].mean() <= args.max_person_loss


def consistent_frame_times(person_times, sweep_times, host_times, tolerance=0.5):
    """One time per frame, all on the SENSOR clock, strictly increasing.

    Preferred: the mean timestamp of the person's pixels. Without it (no
    person on the platform, e.g. after stepping off, or no valid column on
    the person) the median timestamp of the whole sweep, which is on the same
    clock. Without any sensor timestamp, the host time mapped onto the sensor
    clock by a robust straight-line fit (sensor = a * host + b). Mixing the
    two clocks would put those frames thousands of seconds away from the
    others, and every interpolation in time (np.interp needs increasing
    times) would then return wrong angles for whole parts of the recording.
    A frame whose time is more than 'tolerance' seconds off the line is also
    put back on it. Returns (times, number of frames repaired)."""
    host = np.asarray(host_times, dtype=np.float64)
    times = np.where(np.isfinite(person_times), person_times, sweep_times).astype(np.float64)
    known = np.isfinite(times) & np.isfinite(host)
    if known.sum() < 10:
        return host - host[0], int(len(host))
    slope, intercept = 1.0, float(np.median(times[known] - host[known]))
    for _ in range(3):                                   # robust line through the frames with both clocks
        residual = times[known] - (slope * host[known] + intercept)
        good = np.abs(residual - np.median(residual)) < tolerance
        if good.sum() < 10:
            break
        slope, intercept = np.polyfit(host[known][good], times[known][good], 1)
    predicted = slope * host + intercept
    bad = ~np.isfinite(times) | (np.abs(times - predicted) > tolerance)
    times[bad] = predicted[bad]
    for k in range(1, len(times)):                       # strictly increasing for np.interp
        if times[k] <= times[k - 1]:
            times[k] = times[k - 1] + 1e-4
            bad[k] = True
    return times, int(bad.sum())


def build_views(run, isolator, view_frames, frame_angles, axis, times, time_origin):
    """Person cloud of every view frame, turned back by the platform angle
    about the axis: per point, at the time of its column (sensor timestamps),
    so that the turn during the frame is undone too; per frame without them."""
    pivot = np.append(axis, 0.0)
    clouds = []
    progress = Progress("views", len(view_frames))
    for n, k in enumerate(view_frames):
        range_m, info = run.load(run.frame_paths[k])
        cloud, _, point_times = isolator.cloud(range_m, info["timestamps"])
        if point_times is None or len(cloud.points) == 0:
            cloud.transform(yaw_transform(-frame_angles[k], pivot))
        else:
            angle = -np.interp(point_times - time_origin, times, frame_angles)
            c, s = np.cos(angle), np.sin(angle)
            points = np.asarray(cloud.points) - pivot
            normals = np.asarray(cloud.normals)
            turned = np.column_stack([c * points[:, 0] - s * points[:, 1], s * points[:, 0] + c * points[:, 1],
                                      points[:, 2]]) + pivot
            turned_normals = np.column_stack([c * normals[:, 0] - s * normals[:, 1],
                                              s * normals[:, 0] + c * normals[:, 1], normals[:, 2]])
            cloud.points = o3d.utility.Vector3dVector(turned)
            cloud.normals = o3d.utility.Vector3dVector(turned_normals)
        clouds.append(cloud)
        if (n + 1) % 50 == 0 or n == len(view_frames) - 1:
            progress.step(n + 1)
    return clouds


def neighbourhood(index, source, labels, query, normals, radius):
    """Statistics of the source points within 'radius' of every query point:
    count, distinct views, median offset along the normal, spread along the
    normal (1.4826 x MAD)."""
    count = np.zeros(len(query), dtype=np.int64)
    distinct = np.zeros(len(query), dtype=np.int64)
    offset = np.zeros(len(query))
    spread = np.zeros(len(query))
    chunk = 40000
    for begin in range(0, len(query), chunk):
        part = query[begin:begin + chunk]
        neighbours, _, _ = index.hybrid_search(o3d.core.Tensor(part), radius, 256)
        neighbours = neighbours.numpy()
        valid = neighbours >= 0
        safe = np.clip(neighbours, 0, None)
        along = np.einsum("nkj,nj->nk", source[safe] - part[:, None, :], normals[begin:begin + chunk])
        along = np.where(valid, along, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # empty neighbourhoods give NaN
            median = np.nanmedian(along, axis=1)
            spread[begin:begin + chunk] = 1.4826 * np.nanmedian(np.abs(along - median[:, None]), axis=1)
        offset[begin:begin + chunk] = median
        count[begin:begin + chunk] = valid.sum(axis=1)
        view_ids = np.where(valid, labels[safe], -1)
        ordered = np.sort(view_ids, axis=1)
        distinct[begin:begin + chunk] = (ordered[:, :1] >= 0).astype(int)[:, 0] + np.sum(
            (ordered[:, 1:] != ordered[:, :-1]) & (ordered[:, 1:] >= 0), axis=1)
    return count, distinct, np.nan_to_num(offset), np.nan_to_num(spread)


def fuse_with_confidence(clouds, args, min_views, reference_views=None):
    """Support filter, voxel averaging, robust surface fit, and a confidence per output point.

    Surface fit: the voxel averages of the union of the views hold one or two
    points each and keep the range noise. Every output point is moved along
    its normal onto the median of the supported view points around it, first
    within 2 x --confidence-radius, then within --confidence-radius, and
    points that end up on the same spot are merged. The starting points and
    the coarse pass use a random subset of one lap of views (more data would
    put more starting points further out in the noise tails, beyond the reach
    of the fit); the final pass uses every point of every lap, so more laps
    make the surface more precise (--no-surface-fit: keep the voxel averages).

    Confidence, from the final neighbourhood (--confidence-radius): the number
    of distinct views, of points, and their spread along the normal
    (1.4826 x median absolute deviation, in mm; it also contains the surface
    curvature within the radius, about 1 mm on an arm for 1 cm). Standard
    error of the fitted surface about 1.25 x spread / sqrt(points). The
    support filter (--min-views per lap) rejects what only a few views saw."""
    views = o3d.geometry.PointCloud()
    labels = []
    for k, (cloud, color) in enumerate(zip(clouds, view_colors(len(clouds)))):
        colored = copy.deepcopy(cloud)
        colored.paint_uniform_color(color)
        views += colored
        labels.append(np.full(len(colored.points), k))
    labels = np.concatenate(labels)
    points = np.asarray(views.points)
    keep = support_filter(points, labels, args.support_radius, min_views)
    supported = views.select_by_index(np.flatnonzero(keep))
    labels = labels[keep]
    source = np.asarray(supported.points)
    index = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(source))
    index.hybrid_index(2 * args.confidence_radius)
    budget = len(source)
    if reference_views:
        budget = min(budget, int(len(source) / max(len(clouds), 1) * reference_views))
    subset = np.sort(np.random.default_rng(1).choice(len(source), budget, replace=False)) \
        if budget < len(source) else np.arange(len(source))
    seeds = supported.select_by_index(subset)
    coarse_index = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(source[subset]))
    coarse_index.hybrid_index(2 * args.confidence_radius)

    fused = seeds.voxel_down_sample(args.voxel) if args.voxel > 0 else copy.deepcopy(seeds)
    fused.normalize_normals()
    if len(fused.points) > 30:
        fused, _ = fused.remove_statistical_outlier(20, 2.0)
    fused.colors = o3d.utility.Vector3dVector()
    if args.surface_fit:
        positions, normals = np.asarray(fused.points).copy(), np.asarray(fused.normals)
        passes = ((coarse_index, source[subset], labels[subset], 2 * args.confidence_radius),
                  (index, source, labels, args.confidence_radius))
        for search, points, point_labels, radius in passes:
            count, _, offset, _ = neighbourhood(search, points, point_labels, positions, normals, radius)
            positions += np.where(count >= 5, offset, 0.0)[:, None] * normals
        fused.points = o3d.utility.Vector3dVector(positions)
        fused = fused.voxel_down_sample(args.voxel)                  # merge points that met on the surface
        fused.normalize_normals()
    query, query_normals = np.asarray(fused.points), np.asarray(fused.normals)
    count, distinct, offset, spread = neighbourhood(index, source, labels, query, query_normals,
                                                    args.confidence_radius)
    confidence = {"views": distinct, "points": count, "spread_mm": 1000 * spread, "offset_mm": 1000 * offset}

    if len(views.points) > 2_000_000:              # keep the _views.ply file a manageable size
        views = views.voxel_down_sample(0.004)
    return fused, views, float(1.0 - keep.mean()), confidence


def write_confidence(path, fused, confidence):
    """PLY with scalar fields 'views', 'points', 'spread_mm' (CloudCompare shows them),
    coloured by the spread: green 0 mm to red 5 mm and more."""
    cloud = o3d.t.geometry.PointCloud()
    cloud.point.positions = o3d.core.Tensor(np.asarray(fused.points, dtype=np.float32))
    cloud.point.normals = o3d.core.Tensor(np.asarray(fused.normals, dtype=np.float32))
    level = np.clip(confidence["spread_mm"] / 5.0, 0.0, 1.0)
    colors = np.column_stack([level, 1.0 - level, np.zeros_like(level)]).astype(np.float32)
    cloud.point.colors = o3d.core.Tensor(colors)
    for name in ("views", "points", "spread_mm"):
        cloud.point[name] = o3d.core.Tensor(confidence[name].astype(np.float32)[:, None])
    return o3d.t.io.write_point_cloud(str(path), cloud)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fuse a turntable capture of a person into one cloud.")
    parser.add_argument("run", help="capture_turntable.py output directory")
    parser.add_argument("--out", default="person_tt")
    parser.add_argument("--min-range", type=float, default=0.3)
    parser.add_argument("--max-range", type=float, default=10.0)

    geo = parser.add_argument_group("platform and person region (floor frame)")
    geo.add_argument("--center", type=float, nargs=2, default=None, metavar=("X", "Y"),
                     help="platform centre in the floor frame [m]; default: the platform ring found in the "
                          "empty-scene frames (searched around 1.619 0.280, the 2026-09-30 calibration)")
    geo.add_argument("--radius", type=float, default=0.55, help="person region radius around the centre [m]")
    geo.add_argument("--platform-top", type=float, default=0.03, help="height of the platform top [m]")
    geo.add_argument("--min-height", type=float, default=0.05, help="drop points below [m] (platform, feet soles)")
    geo.add_argument("--max-height", type=float, default=2.3)
    geo.add_argument("--min-columns", type=float, default=0.90,
                     help="drop a frame that lost more than this fraction of its columns (lost UDP packets)")
    geo.add_argument("--max-person-loss", type=float, default=0.10,
                     help="drop a frame if more than this fraction of the columns across the person was lost")
    geo.add_argument("--bg-threshold", type=float, default=0.05)
    geo.add_argument("--bg-relative", type=float, default=0.01)
    geo.add_argument("--edge-jump", type=float, default=0.05)
    geo.add_argument("--cluster-eps", type=float, default=0.06)

    ang = parser.add_argument_group("platform motion (any rotation: partial, one lap, several laps)")
    ang.add_argument("--turn-deg", type=float, default=None,
                     help="commanded rotation [deg], if known (default: capture.json, else found in the data); "
                          "used only when the data agree within --loop-tolerance")
    ang.add_argument("--move-deg", type=float, default=360.0,
                     help="largest single motor command [deg]: girogirotondo_timer.m splits longer rotations "
                          "into moves of this size with short stops; 0 = always one continuous move. "
                          "Both hypotheses are fitted and the data decide.")
    ang.add_argument("--sample-seconds", type=float, default=0.5, help="spacing of the frames used [s]")
    ang.add_argument("--max-samples", type=int, default=320,
                     help="the spacing grows for long runs so that at most this many frames are registered")
    ang.add_argument("--reg-voxel", type=float, default=0.01)
    ang.add_argument("--fine-distance", type=float, default=0.02)
    ang.add_argument("--angle-source", default="auto", choices=["auto", "profile", "free", "pairs"],
                     help="profile: motor model with free-form correction; free: model-free solution of all "
                          "the pairs; pairs: old per-pair solve; auto: the mean of the two when they agree (see "
                          "--model-agreement), else the model-free solution")
    ang.add_argument("--model-agreement", type=float, default=3.0,
                     help="auto: the motor model is used only if it differs from the model-free solution by "
                          "at most this rms [deg] (and 3 times this at most)")
    ang.add_argument("--profile-tolerance", type=float, default=15.0,
                     help="warn if the smoothed pair residual exceeds this [deg]")
    ang.add_argument("--loop-tolerance", type=float, default=3.0,
                     help="the commanded turn is used if the data agree within this [deg]")
    ang.add_argument("--pair-min-deg", type=float, default=20.0)
    ang.add_argument("--pair-max-deg", type=float, default=60.0)
    ang.add_argument("--max-pairs", type=int, default=400)
    ang.add_argument("--revisit-pairs", type=int, default=60, help="pairs per lap one or more laps apart")
    ang.add_argument("--revisit-window", type=float, default=30.0,
                     help="pairs whose predicted turn is within this of a whole number of laps [deg]")
    ang.add_argument("--boundary-margin", type=float, default=0.6,
                     help="consecutive frames within this of every start and stop of the platform are "
                          "registered too (timing of the moves) [s]; 0 = off")
    ang.add_argument("--ignore-phases", action="store_true",
                     help="find the still parts and the turn in the data even if the capture labelled them")
    ang.add_argument("--huber-deg", type=float, default=2.0, help="outlier scale of the angle solve [deg]")
    ang.add_argument("--correction-spacing", type=float, default=2.0,
                     help="free-form correction of the stepper profile: one knot every this many seconds "
                          "(0 = off; the platform may lose steps under load)")
    ang.add_argument("--correction-smoothing", type=float, default=3.0,
                     help="penalty on the curvature of the free-form correction")

    view = parser.add_argument_group("views and corrections")
    view.add_argument("--view-step", type=float, default=3.0, help="one view every this many degrees of turn")
    view.add_argument("--max-views", type=int, default=400,
                      help="the step grows for long rotations so that at most this many views are used")
    view.add_argument("--reference-groups", type=int, default=9,
                      help="the views are split into this many groups at random; each view is corrected "
                           "against the views of the other groups (one reference per group)")
    view.add_argument("--axis-iterations", type=int, default=1,
                      help="refinements of the axis position from the view corrections (0 = off)")
    view.add_argument("--max-tilt", type=float, default=2.0, help="sway correction bound [deg]")
    view.add_argument("--max-turn-correction", type=float, default=3.0, help="[deg]")
    view.add_argument("--max-shift", type=float, default=0.03, help="[m]")
    view.add_argument("--max-dropped-views", type=float, default=0.25,
                      help="views whose correction is out of bounds are dropped if they are at most this "
                           "fraction of all views (otherwise kept, with a warning)")
    view.add_argument("--view-range", type=float, nargs=2, default=None, metavar=("START", "END"),
                      help="fuse only the views between these angles of the measured turn [deg], e.g. "
                           "0 360 for the first lap, 720 1080 for the third; the angles are still measured "
                           "on the whole recording")
    view.add_argument("--angle-iterations", type=int, default=0,
                      help="passes of a per-view search of the platform angle against the other views (default "
                           "off: with the free-form correction of the motion model the angles are already "
                           "better than this search, about 1 deg rms in simulation)")
    view.add_argument("--angle-search", type=float, default=15.0,
                      help="largest deviation from the motion model searched per view [deg]")
    view.add_argument("--angle-truncation", type=float, default=0.03,
                      help="distance cap of the alignment cost of the angle search [m]")
    view.add_argument("--limb-iterations", type=int, default=2,
                      help="passes of the per-arm correction (0 = off): each free-hanging arm of every view is "
                           "registered on its own to the arms of the other views")
    view.add_argument("--limb-min-height", type=float, default=0.45,
                      help="arms are searched above this height [m]")
    view.add_argument("--limb-min-offset", type=float, default=0.20,
                      help="a part of a horizontal slice farther than this from the torso axis is an arm [m]")
    view.add_argument("--limb-capture", type=float, default=0.04,
                      help="view points within this of an arm of the union take its label [m]")
    view.add_argument("--limb-max-turn", type=float, default=12.0, help="bound of the arm correction [deg]")
    view.add_argument("--limb-max-shift", type=float, default=0.06, help="bound of the arm correction [m]")
    view.add_argument("--limb-blend", type=float, default=0.10,
                      help="the arm correction fades out over this length below the shoulder [m]")
    view.add_argument("--slab-iterations", type=int, default=1)
    view.add_argument("--slab-height", type=float, default=0.30)
    view.add_argument("--slab-step", type=float, default=0.10)
    view.add_argument("--slab-max-turn", type=float, default=5.0)
    view.add_argument("--slab-max-shift", type=float, default=0.03)

    out = parser.add_argument_group("fusion")
    out.add_argument("--min-views", type=int, default=3,
                     help="support filter: a point needs neighbours from this many views PER LAP")
    out.add_argument("--support-radius", type=float, default=0.02)
    out.add_argument("--voxel", type=float, default=0.005)
    out.add_argument("--confidence-radius", type=float, default=0.01,
                     help="neighbourhood of the surface fit and of the per-point confidence [m]")
    out.add_argument("--no-surface-fit", dest="surface_fit", action="store_false",
                     help="keep the voxel averages (noisier) instead of the local robust surface")
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(f"fuse_turntable.py version {VERSION}")
    run = Run(args.run, args)
    commanded = args.turn_deg if args.turn_deg is not None else run.capture.get("turn_deg")
    commanded = None if commanded is None else float(commanded)

    # 1. Background and floor.
    background = median_range([run.load(p)[0] for p in run.background_paths], 0.5)
    points = run.xyz(background)
    points = points[np.isfinite(points[..., 0])]
    world_from_sensor, height, tilt = fit_floor(points)
    world = transform_points(world_from_sensor, points)
    print(f"background: {len(run.background_paths)} frames; sensor {height:.3f} m above the floor, "
          f"tilt of its z axis from the vertical {tilt:.1f} deg")
    start = np.array(args.center if args.center is not None else PLATFORM_CENTER, dtype=np.float64)
    ring = find_platform(world, start, args) if args.center is None else platform_ring(world, start, args)
    center = start.copy()
    if ring is not None:
        ring_center, ring_radius, ring_rms, ring_n = ring
        print(f"platform ring in this background: centre ({ring_center[0]:.3f}, {ring_center[1]:.3f}) m, "
              f"radius {ring_radius:.3f} m, rms {1000 * ring_rms:.0f} mm, {ring_n} points")
        if args.center is None and plausible_ring(ring):
            center = np.asarray(ring_center, dtype=np.float64)
            print("  centre of the person region and start of the axis fit: the ring centre "
                  "(--center X Y to impose one)")
        elif np.linalg.norm(ring_center - center) > 0.10:
            print(f"  WARNING: the ring is more than 10 cm from --center ({center[0]:.3f}, {center[1]:.3f}): "
                  f"did the platform or the sensor move?")
    elif args.center is None:
        print(f"  WARNING: platform ring not found; using the calibrated centre ({center[0]:.3f}, "
              f"{center[1]:.3f}) m. If the sensor was moved, pass --center X Y")
    isolator = Isolator(run, background, world_from_sensor, center, args)

    # 2. Person in every frame: sensor time of the person, phase, pixel count.
    print(f"frames: {len(run.frame_paths)}; reading ...")
    person_times, sweep_times, host_times, phases, counts = [], [], [], [], []
    progress = Progress("frames", len(run.frame_paths))
    for n, path in enumerate(run.frame_paths):
        range_m, info = run.load(path)
        mask, _ = isolator.mask(range_m)
        person_time, sweep_time = np.nan, np.nan
        if info["timestamps"] is not None:
            valid = info["timestamps"][info["timestamps"] > 0]
            if valid.size:
                sweep_time = float(np.median(valid))
            if mask.any():
                # mean over the person's pixels: continuous even across the start of the sweep
                stamps = np.broadcast_to(info["timestamps"][None, :], mask.shape)[mask]
                stamps = stamps[stamps > 0]
                if stamps.size:
                    person_time = float(np.mean(stamps))
        person_times.append(person_time)
        sweep_times.append(sweep_time)
        host_times.append(info["time"])
        phases.append(info["phase"] if frame_complete(mask, info, args) else -1)
        counts.append(int(mask.sum()))
        if (n + 1) % 200 == 0 or n == len(run.frame_paths) - 1:
            progress.step(n + 1)
    phases, counts = np.array(phases), np.array(counts)
    times, repaired = consistent_frame_times(np.array(person_times), np.array(sweep_times), np.array(host_times))
    if repaired:
        print(f"  frame times: {repaired} frame(s) without a usable sensor timestamp of the person "
              f"(empty platform, lost packets) put on the sensor clock")
    time_origin = float(times[0])
    times = times - time_origin
    unlabelled = args.ignore_phases or not np.any(phases == 2)
    if unlabelled:
        print("  platform driven from another PC (no phase labels): the still parts and the turn "
              "are found in the data")
        phases[phases >= 0] = 2
    usable = (phases > 0) & (counts > 0.3 * np.median(counts[counts > 0]))
    print(f"  person pixels per frame: median {int(np.median(counts))}; usable frames {int(usable.sum())}")

    # 3. Angle versus time on sampled frames.
    frame_period = np.median(np.diff(times))
    stride = max(1, int(round(args.sample_seconds / frame_period)),
                 int(np.ceil(usable.sum() / max(args.max_samples, 10))))
    sampled = [k for k in range(0, len(times), stride) if usable[k]]
    print(f"registration samples: {len(sampled)}, one every {stride * frame_period:.2f} s")
    sample_clouds = [isolator.cloud(run.load(run.frame_paths[k])[0])[0] for k in sampled]
    samples = Samples(sample_clouds, args.reg_voxel, args.fine_distance)
    angles, chain, measurements = estimate_angles(samples, times[sampled], phases[sampled], center, args)
    axis = center.copy()
    sense = np.sign(angles[-1] - angles[0]) or 1.0
    sample_phases = phases[sampled].copy()
    if unlabelled:
        # Still before the turn: within 3 deg of the start; still after: within
        # 3 deg of the end. The chained angles underestimate the turn by a few
        # per cent; the motion model below measures it.
        signed = np.degrees(angles * sense)
        moving = np.flatnonzero((signed > 3.0) & (signed < signed[-1] - 3.0))
        if len(moving) >= 4:
            sample_phases[:moving[0]] = 1
            sample_phases[moving[-1] + 1:] = 3
            sample_phases[moving[0]:moving[-1] + 1] = 2
            print(f"  turn from about {times[sampled][moving[0]]:.1f} s to {times[sampled][moving[-1]]:.1f} s "
                  f"(chained estimate {signed[-1]:.0f} deg)")
            if not np.any(sample_phases == 3):
                print("  WARNING: no still frames after the turn: stop the recording later (ENTER) or use a longer --duration")
        else:
            print("  WARNING: no turn found in the recording")
    fit = constant_speed_fit(times[sampled], angles * sense, sample_phases)
    if fit is not None:
        print(f"constant-speed model on the chained angles: {fit['deg_per_s']:.2f} deg/s "
              f"({360 / fit['deg_per_s']:.1f} s per lap)")

    # Motion model fitted to long-baseline and revisit pairs (the default source of the angles).
    profile = None
    if args.angle_source in ("auto", "profile", "free"):
        profile = estimate_motion(samples, times[sampled], sample_phases, angles, axis, commanded, args,
                                  frame_times=times, usable=usable, local_pairs=measurements,
                                  load_cloud=lambda k: isolator.cloud(run.load(run.frame_paths[k])[0])[0]
                                  .voxel_down_sample(args.reg_voxel))
    if profile is not None:
        axis = np.asarray(profile["axis"])
        model = profile["model"]
        print(f"  axis from the joint fit: ({axis[0]:.4f}, {axis[1]:.4f}) m, "
              f"{1000 * np.linalg.norm(axis - center):.1f} mm from --center")
        print(f"  motion: {model.describe()}")
        print(f"  long-pair residual rms {profile['pair_rms_deg']:.2f} deg ({profile['pairs']} long pairs, "
              f"{profile['revisits']} revisit pairs over {profile['laps_with_revisits']} laps"
              + (f", revisit residual rms {profile['revisit_residual_rms_deg']:.2f} deg"
                 if profile["revisit_residual_rms_deg"] is not None else "")
              + f"); systematic deviation {profile['systematic_deviation_deg']:.2f} deg")
        print(f"  registration scale {100 * profile['registration_scale']:+.2f} % "
              + ("(calibrated by the revisit pairs)" if profile["scale_calibrated"]
                 else "(not calibrated: no revisit pairs; the angles may be short by 1 to 3 %)"))
        if commanded is not None:
            print(f"  total turn of the motor model {profile['fitted_total_deg']:.2f} deg, commanded {commanded:g} deg"
                  + (" (commanded value used)" if profile["total_fixed_to_command"] else ""))
    use_profile = profile is not None and args.angle_source != "pairs"
    if use_profile and profile["systematic_deviation_deg"] > args.profile_tolerance:
        print(f"  WARNING: the pairs deviate from the angles by up to "
              f"{profile['systematic_deviation_deg']:.1f} deg after smoothing (some pairs are wrong, or the "
              f"person moved). Look at {args.out}_angle.png")
    profile_only = None
    alternatives = {}
    if use_profile:
        frame_angles = profile["sense"] * profile["model"].angle(times)
        frame_angles -= frame_angles[sampled[0]]
        print(f"  angles: {profile['label']}")
        motor = profile["motor_model"]
        saved, motor.correction = motor.correction, None
        profile_only = profile["sense"] * motor.angle(times)
        profile_only -= profile_only[sampled[0]]
        motor.correction = saved
        for label, other in (("motor model with correction", motor), ("model-free solution", profile["free_model"])):
            if other is not profile["model"]:
                angles_other = profile["sense"] * other.angle(times)
                alternatives[label] = angles_other - angles_other[sampled[0]]
    else:
        # Per-pair solve, with the revisit pairs if the model found them
        # (they keep several laps consistent); unwrapped near the model.
        reference = angles
        extra = []
        if profile is not None:
            reference = profile["sense"] * profile["model"].angle(times[sampled])
            reference = reference - reference[0]
            extra = profile["revisit_pairs"]

        class Settings:
            max_correction = args.huber_deg * 2.0
        solved = solve_turns(len(sampled), reference, measurements + extra, Settings())
        frame_angles = np.interp(times, times[sampled], solved)
        print("  angles from the per-pair solve")
    total = float(np.degrees(abs(frame_angles[usable].max() - frame_angles[usable].min())))
    laps = total / 360.0
    print(f"platform: total turn {total:.1f} deg = {laps:.2f} laps")
    measured_laps = lap_times(times[usable], frame_angles[usable] * sense)
    if measured_laps:
        print("  time of every lap: " + ", ".join(f"lap {lap} {duration:.1f} s ({360.0 / duration:.2f} deg/s)"
                                                   for lap, duration in measured_laps))
    if total < 300.0:
        print(f"  note: less than one lap; the cloud covers about {total + 120:.0f} deg of the body "
              f"(the sensor sees roughly 120 deg at a time)")

    # 4. Views every --view-step degrees of turn (several laps: several views per direction).
    candidates = np.flatnonzero(usable)
    if args.view_range is not None:
        low, high = sorted(args.view_range)
        turned = np.degrees(frame_angles[candidates]) * sense
        candidates = candidates[(turned >= low) & (turned <= high)]
        if len(candidates) == 0:
            raise SystemExit(f"--view-range {low:g} {high:g}: no frame in this part of the turn "
                             f"(total turn {total:.1f} deg)")
        covered = float(np.degrees(np.ptp(frame_angles[candidates])))
        print(f"views restricted to {low:g} to {high:g} deg of turn (--view-range): {covered:.1f} deg covered")
        if covered < 300.0:
            print("  note: less than one lap in this range; parts of the body are seen from one side only")
        total, laps = covered, covered / 360.0
    step = max(args.view_step, total / max(args.max_views, 1))
    view_frames, last = [], None
    for k in candidates:
        a = np.degrees(frame_angles[k]) * sense
        if last is None or abs(a - last) >= step:
            view_frames.append(int(k))
            last = a
    print(f"views: {len(view_frames)} frames, one every {step:.2f} deg of turn"
          + (f" (about {len(view_frames) / laps:.0f} per lap)" if laps >= 1.5 else ""))
    groups = reference_groups(len(view_frames), args.reference_groups)
    # Reference budget: one lap of views (minus one group), whatever the number of laps.
    reference_views = min(len(view_frames), 360.0 / step) * (1.0 - 1.0 / max(args.reference_groups, 1))

    # Axis refinement: correct the views against each other and read the axis
    # error from the pattern of the corrections (a wrong axis displaces the
    # views along a circle as they turn; sway does not follow the turn).
    raw = build_views(run, isolator, view_frames, frame_angles, axis, times, time_origin)
    model_angles = frame_angles.copy()
    view_turns = np.zeros(len(view_frames))
    view_correction_record = []
    if args.angle_iterations > 0:
        raw, view_turns = refine_view_angles(raw, args, np.append(axis, 0.0), groups, reference_views)
        frame_angles = frame_angles.copy()
        frame_angles[view_frames] -= view_turns
        view_correction_record = [[int(k), float(np.degrees(d))] for k, d in zip(view_frames, view_turns)]
        deviation = np.degrees(view_turns)
        print(f"platform angle searched view by view: deviation from the motion model rms "
              f"{np.sqrt(np.mean(deviation ** 2)):.2f} deg, max {np.abs(deviation).max():.2f} deg")
    for iteration in range(args.axis_iterations):
        _, _, shifts = groupwise_rigid(raw, args, f"axis check {iteration + 1}", groups, reference_views)
        error = axis_error_from_shifts(shifts, frame_angles[view_frames])
        if error is None:
            print("  axis polish skipped (the views do not go round the body)")
            break
        old_pivot, axis = np.append(axis, 0.0), axis + error
        print(f"  axis corrected by ({1000 * error[0]:+.1f}, {1000 * error[1]:+.1f}) mm -> "
              f"({axis[0]:.4f}, {axis[1]:.4f}) m")
        for cloud, k in zip(raw, view_frames):                   # re-turn about the corrected axis
            cloud.transform(yaw_transform(frame_angles[k], old_pivot))
            cloud.transform(yaw_transform(-frame_angles[k], np.append(axis, 0.0)))
        if np.linalg.norm(error) < 0.0005:
            break

    clouds, rigid_rms, shifts = groupwise_rigid(raw, args, "view corrections", groups, reference_views)
    print(f"sway correction per view: RMS {rigid_rms:.1f} mm")
    # A view whose correction is out of bounds (--max-turn-correction,
    # --max-shift, --max-tilt) does not fit the others: a wrong angle (the
    # platform lost steps), or the person moved more than the bounds. Fused
    # as it is, it would blur the surface, so it is dropped, unless that
    # would remove too many views (then the angles themselves are suspect).
    outliers = ~np.isfinite(shifts[:, 0])
    dropped_views = []
    if outliers.any():
        fraction = outliers.mean()
        if fraction <= args.max_dropped_views:
            dropped_views = [int(view_frames[k]) for k in np.flatnonzero(outliers)]
            keep_views = np.flatnonzero(~outliers)
            clouds = [clouds[k] for k in keep_views]
            view_frames = [view_frames[k] for k in keep_views]
            groups = None if groups is None else groups[keep_views]
            print(f"  {outliers.sum()} of {len(outliers)} views dropped: their correction was out of bounds "
                  f"(turn > {args.max_turn_correction:g} deg, shift > {100 * args.max_shift:g} cm or "
                  f"tilt > {args.max_tilt:g} deg)")
        else:
            print(f"  WARNING: {100 * fraction:.0f}% of the views could not be corrected within the bounds; "
                  f"they are kept. The platform angles are probably off (speed changes, lost steps): "
                  f"look at {args.out}_angle.png and try --angle-source pairs")
    if args.slab_iterations > 0:
        clouds, history = slab_refine(clouds, args, groups, clean_reference=True, reference_views=reference_views)
        print(f"height-slab correction: RMS {', '.join(f'{h:.1f}' for h in history)} mm")
    limb_report = None
    if args.limb_iterations > 0:
        clouds, limb_report = limb_refine(clouds, args, groups, reference_views)

    # 5. Fusion, output frame on the axis at the platform top.
    min_views = args.min_views * max(1, int(np.floor(laps + 0.1)))
    fused, views, removed, confidence = fuse_with_confidence(clouds, args, min_views, reference_views)
    output_from_world = np.eye(4)
    output_from_world[:3, 3] = [-axis[0], -axis[1], -args.platform_top]
    fused.transform(output_from_world)
    views.transform(output_from_world)
    top = float(np.asarray(fused.points)[:, 2].max())
    o3d.io.write_point_cloud(f"{args.out}.ply", fused)
    o3d.io.write_point_cloud(f"{args.out}_views.ply", views)
    wrote_confidence = write_confidence(f"{args.out}_confidence.ply", fused, confidence)
    plotted = angle_plot(f"{args.out}_angle.png", times[sampled], angles * sense, times, model_angles * sense,
                         profile["label"] if use_profile else "per-pair solve",
                         (times[view_frames], frame_angles[view_frames] * sense) if args.angle_iterations > 0
                         else None, None if profile_only is None else profile_only * sense,
                         {label: angles_other * sense for label, angles_other in alternatives.items()})

    spread, per_point = confidence["spread_mm"], confidence["points"]
    standard_error = 1.25 * spread / np.sqrt(np.maximum(per_point, 1))   # of a median; spread includes curvature
    report = {
        "version": VERSION, "run": str(Path(args.run).resolve()),
        "sensor_height_m": height, "sensor_tilt_deg": tilt, "world_from_sensor": world_from_sensor,
        "platform_ring": None if ring is None else {"center": ring[0], "radius": ring[1], "rms_m": ring[2]},
        "axis_start": center, "axis_center": axis, "total_turn_deg": total, "laps": laps,
        "commanded_turn_deg": commanded,
        "constant_speed_fit": fit, "sampled_frames": sampled, "sample_times_s": times[sampled],
        "sample_angles_deg": np.degrees(angles * sense), "view_frames": view_frames, "view_step_deg": step,
        "profile": None if profile is None else {k: v for k, v in profile.items()
                                                 if k not in ("model", "motor_model", "free_model", "revisit_pairs")},
        "angles_from_profile": bool(use_profile),
        "frame_times_s": times, "time_origin_s": time_origin, "frame_angles_deg": np.degrees(frame_angles),
        "sway_correction_rms_mm": rigid_rms, "support_filter_min_views": min_views,
        "dropped_view_frames": dropped_views, "view_range_deg": args.view_range,
        "lap_times_s": [duration for _, duration in measured_laps],
        "view_angle_correction_deg": view_correction_record,
        "model_frame_angles_deg": np.degrees(model_angles),
        "arm_correction": None if limb_report is None else {
            "shoulder_heights_m": limb_report["tops"],
            "hand_move_median_mm": [{str(side): (float(1000 * np.nanmedian(moves)) if np.isfinite(moves).any()
                                                 else None) for side, moves in p.items()}
                                    for p in limb_report["hand_moves"]]},
        "support_filter_removed_fraction": removed,
        "confidence": {"views_median": float(np.median(confidence["views"])),
                       "points_median": float(np.median(per_point)),
                       "spread_median_mm": float(np.median(spread)),
                       "spread_p90_mm": float(np.percentile(spread, 90)),
                       "standard_error_median_mm": float(np.median(standard_error))},
        "output_from_world": output_from_world, "highest_point_m": top, "parameters": vars(args),
    }
    Path(f"{args.out}.json").write_text(json.dumps(report, indent=2, default=to_builtin))
    print(f"support filter (at least {min_views} views) removed {100 * removed:.1f}% of the points")
    print(f"fused: {len(fused.points)} points, highest point {top:.3f} m above the platform top")
    print(f"within {1000 * args.confidence_radius:.0f} mm of an output point: median "
          f"{np.median(confidence['views']):.0f} views, {np.median(per_point):.0f} points; "
          f"spread along the normal median {np.median(spread):.1f} mm, "
          f"p90 {np.percentile(spread, 90):.1f} mm; standard error of the fitted surface median "
          f"{np.median(standard_error):.2f} mm")
    print(f"wrote {args.out}.ply, {args.out}_views.ply"
          + (f", {args.out}_confidence.ply" if wrote_confidence else "")
          + f", {args.out}.json" + (f", {args.out}_angle.png" if plotted else ""))


if __name__ == "__main__":
    main()
