"""Relative turn measurements between frames of a turntable recording.

Samples holds downsampled clouds of the sampled frames; every pair is
registered with ONE unknown, the turn about the platform axis (icp_turn_about).
The pair families:

    chained_angles   neighbouring samples (k, k+1) and (k, k+2): approximate
                     angles, the still parts and the turn
    joint_axis_fit   pairs 20 to 60 deg apart with the axis position fitted jointly
    revisit_pairs    pairs whole laps apart (exact whole laps calibrate the
                     registration scale)
    boundary_pairs   consecutive frames at the full frame rate around every
                     start and stop of the platform (timing of the moves)
"""

from __future__ import annotations

import numpy as np

from bodyscan.geometry import Target, evaluate, transform_points, wrapped_degrees, yaw_of, yaw_transform
from bodyscan.log import Progress, info
from bodyscan.registration import Pair, icp_turn_about, information_matrix, solve_turns


class Samples:
    """Downsampled clouds of the sampled frames, for registration."""

    def __init__(self, clouds, voxel: float, fine: float):
        self.clouds = [c.voxel_down_sample(voxel) for c in clouds]
        for c in self.clouds:
            if not c.has_normals():
                c.estimate_normals()
        self.points = [np.asarray(c.points) for c in self.clouds]
        self.targets = [Target(c) for c in self.clouds]
        self.fine = fine

    def __len__(self) -> int:
        return len(self.points)

    def register_turn(self, i: int, j: int, expected_degrees, pivot, uncertain: bool) -> Pair:
        """Best one-unknown (turn about 'pivot') alignment of sample i onto j,
        over the starting turns 'expected_degrees'."""
        best = None
        for start in np.radians(expected_degrees):
            angle, fitness, rmse = icp_turn_about(self.points[i], self.targets[j], start, pivot)
            if best is None or fitness > best[1] + 0.02 or (fitness > best[1] - 0.02 and rmse < best[2]):
                best = (angle, fitness, rmse)
        transform = yaw_transform(best[0], pivot)
        pair = Pair(i, j, transform, best[1], best[2], uncertain)
        pair.information = information_matrix(self.clouds[i], self.clouds[j], self.fine, transform)
        return pair


def chained_angles(samples: Samples, times, phases, center, huber_deg: float):
    """Platform angle of every sample [rad] (angle of sample 0 = 0), from
    neighbouring pairs and pairs two samples apart, solved together.

    Every pair is registered with one unknown, the turn about the vertical
    through 'center' (the body does not translate on the platform). A free
    turn-plus-shift registration is ambiguous on partial views of a body (a
    small turn looks like a sideways shift); fixing the axis removes that.
    The axis position itself is refined later.
    Returns (solved angles, chained angles, the Pair measurements)."""
    count = len(samples)
    pivot = np.append(center, 0.0)
    info(f"angle estimation from {count} sampled frames (turn about the platform centre):")
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
    # same way again) come later, once a motion model predicts them well:
    # over several turns the chained angles drift by tens of degrees.
    measurements = list(sequential)
    lag_pairs = [(k, k + 2) for k in range(count - 2)]
    progress = Progress("longer pairs", len(lag_pairs))
    for n, (i, j) in enumerate(lag_pairs):
        expected = np.degrees(chain[j] - chain[i])
        measurements.append(samples.register_turn(i, j, expected + np.arange(-6.0, 6.1, 3.0), pivot, True))
        if (n + 1) % 40 == 0 or n == len(lag_pairs) - 1:
            progress.step(n + 1)
    info(f"  {len(measurements)} relative turns")
    angles = solve_turns(count, chain, measurements, np.radians(huber_deg))
    return angles, chain, measurements


def joint_axis_fit(samples: Samples, pairs, turns, center, iterations=(8, 6, 6), distances=(0.04, 0.025, 0.015)):
    """Axis position (x, y) and one turn per pair, fitted together.

    Every pair (i, j) says: sample j = sample i turned by its own angle about
    the same vertical axis. With pairs 20 to 60 degrees apart the axis is well
    determined (a wrong axis leaves a shift (I - R) e that grows with the
    turn), while the per-pair angles absorb sway. Gauss-Newton with a Tukey
    kernel; the shared axis is solved through the Schur complement.
    Returns axis, turns [rad], overlap per pair."""
    center = np.array(center, dtype=np.float64)
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


def revisit_pairs(samples: Samples, predicted, lap: int, axis, sense: float, window_deg: float, per_lap: int):
    """Pairs one or more full laps apart (the body faces the same way again).

    predicted: model angle of every sample [rad], increasing. For every
    sample i the sample j nearest to exactly 'lap' turns later is taken (if
    within window_deg), up to per_lap pairs spread over the run. They are
    registered with a small search around the predicted remainder; the
    measured turn is lap * 360 deg plus the registered rest, so these pairs
    fix the speed and the total over many laps.
    Returns (i, j, measured turn [rad] in the positive sense, fitness, lap, Pair)."""
    pivot = np.append(axis, 0.0)
    target = 2 * np.pi * lap
    window = np.radians(window_deg)
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
              np.unique(np.round(np.linspace(0, len(candidates) - 1, min(per_lap, len(candidates)))))]
    result = []
    for i, j in chosen:
        rest = sense * (predicted[j] - predicted[i] - target)             # raw (sensor) sense
        pair = samples.register_turn(i, j, np.degrees(rest) + np.arange(-12.0, 12.1, 4.0), pivot, True)
        if pair.fitness < 0.5:
            continue
        measured = target + sense * np.radians(wrapped_degrees(yaw_of(pair.transform)))
        result.append((i, j, measured, pair.fitness, lap, pair))
    return result


def boundary_pairs(model, frame_times, usable, load_cloud, axis, sense: float, margin: float):
    """Consecutive frames (full frame rate) within 'margin' seconds of every
    start and stop of a StepperMotion, registered with one unknown (the turn
    about the axis). Returns (frame a, frame b, measured turn [rad] in the
    positive sense, fitness)."""
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
