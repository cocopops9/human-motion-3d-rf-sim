"""Fitting platform motion models to relative turn measurements.

A measurement is a tuple (i, j, measured turn [rad], weight, laps[, kind]):
between the times of samples i and j the platform turned by 'measured',
with 'laps' whole laps known exactly (revisit pairs). Kinds: 0 long pair,
1 consecutive frames around a start or stop, 3 neighbouring samples."""

from __future__ import annotations

import numpy as np

from bodyscan.motion.models import StepperMotion


def robust_cost(residual, base, scale) -> float:
    """Mean Huber loss with a fixed scale, to compare two motion models on the same data."""
    a = np.abs(residual) / scale
    loss = np.where(a < 2.0, 0.5 * a ** 2, 2.0 * a - 2.0)
    return float(np.sum(base * loss) / np.sum(base))


def measurement_classes(measurements) -> np.ndarray:
    """Class of every measurement: 0 long pair, 1 revisit (whole laps apart),
    2 consecutive frames around a start or stop, 3 neighbouring samples."""
    kinds = np.array([m[5] if len(m) > 5 else 0 for m in measurements])
    laps = np.array([m[4] for m in measurements])
    return np.where(kinds == 1, 2, np.where(kinds == 3, 3, np.where(laps > 0, 1, 0)))


def fit_motion(times, measurements, motion: StepperMotion, fit_total, scale=0.0, fit_scale=False, iterations=40,
               dense_scale=0.0):
    """Gauss-Newton with Huber weights of a StepperMotion to relative turns.

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
    knot every 'spacing' seconds, fitted to relative turns with a
    per-measurement scale factor (the registration of every class of pairs
    reads the turns short by its own fraction), class weights (inverse robust
    variance per class) and a penalty on the second differences of c
    ('smoothing').

    The stepper profile assumes the platform follows the motor: the same
    speed in every lap, the same ramps, stops of the commanded length. Under
    load the motor can lose steps (the platform then falls behind and the
    lap takes longer), and the choice between one move and several can be
    wrong; the angles are then off by degrees in places, which shifts the
    arms and hands by centimetres and distorts the fused arms. The correction
    follows any such deviation that the pairs see, and stays near zero where
    the model fits; the short pairs between neighbouring samples give it the
    local speed, the long pairs and the revisits the scale.
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
    Returns the correction (knots, values), the scales {group: s} (group 0:
    long pairs and revisits, 2: consecutive frames, 3: neighbouring samples),
    and the residuals [rad]."""
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


def plausible_sequence(model: StepperMotion, max_stop: float = 3.0, max_ramp: float = 2.4) -> bool:
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
