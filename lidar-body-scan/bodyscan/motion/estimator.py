"""Platform angle versus time from the data: the estimator.

PlatformMotionEstimator.estimate() runs the stages below and returns a dict
(the 'profile' section of the report) whose 'model' entry is the MotionModel
used for every frame. Each stage is a method, so a subclass can change one
(for example choose() for another selection policy, or hypotheses() for a
platform with another motor) and keep the rest.

    long_pairs     pairs 20 to 60 deg apart, axis fitted jointly
    hypotheses     one continuous move, or sequences of moves with stops
    revisits       lap by lap, pairs whole laps apart, predicted by the
                   model-free solution (calibrate the registration scale)
    boundaries     consecutive frames around every start and stop
    correction     free-form correction of the stepper profile
    choose         motor model, model-free solution, or their mean
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bodyscan.config import param
from bodyscan.geometry import wrapped_degrees, yaw_of
from bodyscan.log import info
from bodyscan.motion.fitting import (fit_correction, fit_free, fit_motion, measurement_classes, plausible_sequence,
                                     robust_cost)
from bodyscan.motion.models import FreeMotion, MeanMotion, StepperMotion
from bodyscan.motion.pairs import Samples, boundary_pairs, joint_axis_fit, revisit_pairs


@dataclass
class MotionConfig:
    """Platform angle versus time (any rotation: partial, one lap, several laps)."""
    turn_deg: float | None = param(None, "commanded rotation, if known (default: capture.json, else found in the "
                                         "data); used only when the data agree within --loop-tolerance", unit="deg")
    move_deg: float = param(360.0, "largest single motor command; longer rotations are split into moves of this "
                                   "size with short stops (0 = always one continuous move)", unit="deg",
                            effect="both hypotheses are fitted and the data decide")
    sample_seconds: float = param(0.5, "spacing of the frames registered for the angles", unit="s",
                                  effect="smaller: more pairs, slower; the spacing grows for long runs anyway")
    max_samples: int = param(320, "at most this many frames are registered for the angles")
    reg_voxel: float = param(0.01, "voxel of the clouds registered for the angles", unit="m")
    fine_distance: float = param(0.02, "overlap distance of the registration quality", unit="m")
    angle_source: str = param("auto", "where the frame angles come from", choices=("auto", "profile", "free", "pairs"),
                              effect="auto: mean of the motor model and the model-free solution when they agree, "
                                     "else the model-free one; profile: motor model; free: model-free; pairs: "
                                     "per-pair solve (old)")
    model_agreement: float = param(3.0, "auto: the motor model is used only within this rms of the model-free "
                                        "solution (and 3 times this at most)", unit="deg")
    profile_tolerance: float = param(15.0, "warn if the smoothed pair residual exceeds this", unit="deg")
    loop_tolerance: float = param(3.0, "the commanded turn is used if the data agree within this", unit="deg")
    pair_min_deg: float = param(20.0, "shortest turn of a long pair", unit="deg")
    pair_max_deg: float = param(60.0, "longest turn of a long pair", unit="deg",
                                effect="larger pairs constrain the speed better but overlap less")
    max_pairs: int = param(400, "at most this many long pairs")
    revisit_pairs: int = param(60, "pairs per lap one or more laps apart")
    revisit_window: float = param(30.0, "pairs whose predicted turn is within this of a whole number of laps",
                                  unit="deg")
    boundary_margin: float = param(0.6, "consecutive frames within this of every start and stop are registered "
                                        "too (timing of the moves); 0 = off", unit="s")
    ignore_phases: bool = param(False, "find the still parts and the turn in the data even if the capture "
                                       "labelled them")
    huber_deg: float = param(2.0, "outlier scale of the angle solve", unit="deg")
    correction_spacing: float = param(2.0, "free-form correction of the stepper profile: one knot every this many "
                                           "seconds (0 = off)", unit="s",
                                      effect="smaller follows faster speed changes, but noisier")
    correction_smoothing: float = param(3.0, "penalty on the curvature of the free-form correction",
                                        effect="larger: smoother correction")


class PlatformMotionEstimator:
    """Motion model of the platform fitted to long-baseline turns about the fitted axis.

    Consecutive frames turn only a few degrees, and the LiDAR samples the body
    along the same beams in both, which pulls ICP towards 'no motion'. Pairs
    20 to 60 degrees apart carry the same bias on a much larger turn, and the
    model has few parameters, so the fitted angles are far more accurate than
    chained or solved per-pair turns. Pairs one or more laps apart (revisits)
    are added lap by lap, each time predicted by the model fitted so far;
    their whole laps are exact, so they calibrate the scale of the registered
    turns and fix the speed and the total over the whole run.

    Two hypotheses are fitted when the turn is longer than move_deg: one
    continuous move, and a sequence of moves of move_deg with stops in between
    (what girogirotondo_timer.m does); the data decide."""

    def __init__(self, config: MotionConfig):
        self.config = config

    # ------------------------------------------------------------------
    def long_pairs(self, samples: Samples, signed, sense, axis):
        c = self.config
        span = float(np.degrees(signed.max() - signed.min()))
        pair_min = min(c.pair_min_deg, 0.4 * span)
        pair_max = max(pair_min + 1.0, min(c.pair_max_deg, span))
        candidates = []
        for i in range(len(signed)):
            for j in range(i + 1, len(signed)):
                difference = np.degrees(signed[j] - signed[i])
                if pair_min <= difference <= pair_max:
                    candidates.append((i, j))
        if len(candidates) < 10:
            return None, axis, span
        candidates = candidates[::max(1, len(candidates) // c.max_pairs)]
        info(f"motion model: {len(candidates)} long pairs ({pair_min:.0f} to {pair_max:.0f} deg); "
             f"joint fit of the axis and the pair turns")
        turns = np.array([sense * (signed[j] - signed[i]) for i, j in candidates])
        axis, turns, fitness = joint_axis_fit(samples, candidates, turns, np.asarray(axis, dtype=np.float64))
        long_pairs = [(i, j, sense * turn, fit, 0) for (i, j), turn, fit in zip(candidates, turns, fitness)
                      if fit >= 0.4]
        return (long_pairs if len(long_pairs) >= 10 else None), axis, span

    def hypotheses(self, times, phases, signed, span, long_pairs, commanded_deg):
        """Stepper hypotheses, each first fitted to the chained angles
        (approximate, a few per cent short, but with the stops between moves
        visible), then to the pairs: one continuous move, and sequences of
        move_deg moves (the number of moves from the chained total, one fewer,
        one more, and the commanded count)."""
        c = self.config
        moving = np.flatnonzero(phases == 2)
        t_on, t_off = (times[moving[0]], times[moving[-1]]) if len(moving) >= 2 else (times[0], times[-1])
        steps = np.diff(signed) / np.maximum(np.diff(times), 1e-6)
        inside = (times[1:] > t_on) & (times[:-1] < t_off)
        cruise = 1.03 * np.median(steps[inside]) if inside.sum() >= 3 else np.radians(span) / max(t_off - t_on, 0.5)
        cruise = max(cruise, np.radians(1.0))
        chain_pairs = [(0, k, signed[k] - signed[0], 1.0, 0) for k in range(1, len(signed))]
        # One continuous move, fitted to the long pairs from three starting
        # points (the fit to the chained angles, and constant speed over the
        # turn with the chained total raised by 3 and 8 %, since the chained
        # angles are short): from a poor start the fit can end with the ramp at
        # its bound and the start and total far off.
        long_base = np.array([m[3] for m in long_pairs])
        initial = [fit_motion(times, chain_pairs,
                              StepperMotion(cruise, 0.5, [t_on - 0.25], np.radians(span) * 1.02, None), True)[0]]
        for factor in (1.03, 1.08):
            total = np.radians(span) * factor
            initial.append(StepperMotion(total / max(t_off - t_on - 0.5, 1.0), 0.5, [t_on - 0.25], total, None))
        one, one_cost = None, np.inf
        for start_model in initial:
            fitted = fit_motion(times, long_pairs, start_model, True)
            cost = robust_cost(fitted[3], long_base, np.radians(1.0))
            if cost < one_cost:
                one, one_cost = fitted[0], cost
        result = [one]
        move = np.radians(c.move_deg) if c.move_deg > 0 else None
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
                    fitted = fit_motion(times, long_pairs,
                                        StepperMotion(one.speed, one.ramp, starts, total, move).ordered(), True)
                    spread = max(1.4826 * np.median(np.abs(fitted[3])), np.radians(0.2))
                    cost = robust_cost(fitted[3], long_base, spread)
                    if cost < best_cost:
                        best, best_cost = fitted[0], cost
                result.append(best)
        return result, t_on, t_off

    @staticmethod
    def fit_hypotheses(all_times, measurements, hypotheses, scale, fit_scale):
        """Fit every hypothesis; the best by robust cost (a sequence must beat the
        continuous move by 5 %, so that a continuous rotation is not split).
        Returns (fitted hypotheses, model, scale, rms, residual, label, dense scale)."""
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
        return [f[0] for f in fitted], model, scale, rms, residual, label, dense_scale

    def choose(self, model, free, gap_rms, gap_max, scale):
        """Reasons to reject the motor model (empty: it may be used)."""
        c = self.config
        reasons = []
        if c.angle_source == "free":
            reasons.append("--angle-source free")
        if len(model.starts) > 1 and not plausible_sequence(model):
            reasons.append("the fitted sequence of moves is not one a stepper makes")
        if abs(scale) >= 0.099:
            reasons.append("the registration scale of the motor model is at its bound")
        if gap_rms > c.model_agreement or gap_max > 3 * c.model_agreement:
            reasons.append(f"it differs from the model-free solution by more than {c.model_agreement:g} deg rms "
                           f"or {3 * c.model_agreement:g} deg max")
        if c.angle_source == "profile":
            reasons = []
        return reasons

    # ------------------------------------------------------------------
    def estimate(self, samples: Samples, times, phases, angles, axis, commanded_deg,
                 frame_times=None, usable=None, load_cloud=None, local_pairs=None):
        """Returns the profile dict, or None if there are too few long pairs."""
        c = self.config
        sense = np.sign(angles[-1] - angles[0]) or 1.0
        signed = angles * sense
        long_pairs, axis, span = self.long_pairs(samples, signed, sense, axis)
        if long_pairs is None:
            return None
        hypotheses, t_on, t_off = self.hypotheses(times, phases, signed, span, long_pairs, commanded_deg)
        all_times = times
        hypotheses, model, scale, rms, residual, label, dense_scale = self.fit_hypotheses(
            all_times, long_pairs, hypotheses, 0.0, False)
        info(f"  long pairs only: {label}; {model.describe()}")

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

        def baseline(t):
            return np.interp(t, times, signed)

        def solve_free(measurements, fit_scale, solve_times):
            correction, scales, free_residual = fit_free(solve_times, measurements + local, baseline,
                                                         max(c.correction_spacing, 0.5),
                                                         c.correction_smoothing, fit_scale)
            return FreeMotion(times, signed, correction), scales, free_residual[:len(measurements)]

        free, free_scales, _ = solve_free(long_pairs, False, times)

        revisits, lap = [], 1
        while True:
            predicted = free.angle(times)
            if (predicted.max() - predicted.min()) + np.radians(c.revisit_window) < 2 * np.pi * lap:
                break
            new = revisit_pairs(samples, predicted, lap, axis, sense, c.revisit_window, c.revisit_pairs)
            # A revisit far from the prediction locked onto a wrong pose (the body
            # seen from the back looks like the front): dropped.
            new = [r for r in new
                   if abs(r[2] - (predicted[r[1]] - predicted[r[0]])) <= np.radians(c.revisit_window)]
            revisits += new
            measurements = long_pairs + [r[:5] for r in revisits]
            fit_scale = len(revisits) >= 8
            free, free_scales, _ = solve_free(measurements, fit_scale, times)
            hypotheses, model, scale, rms, residual, label, dense_scale = self.fit_hypotheses(
                all_times, measurements, hypotheses, scale, fit_scale)
            info(f"  lap {lap}: {len(new)} revisit pairs; total so far "
                 f"{np.degrees(predicted.max() - predicted.min()):.1f} deg; registration scale "
                 f"{100 * free_scales[0]:+.2f} %; motor model: {label}, speed {np.degrees(model.speed):.3f} deg/s")
            lap += 1
        measurements = long_pairs + [r[:5] for r in revisits]
        fit_scale = len(revisits) >= 8

        # Timing of the starts and stops from consecutive frames at the full
        # frame rate. With several moves, the revisit pairs fix the scale and
        # the period of the moves, but the speed trades off against the ramps
        # and the stops unless these are measured; the same pairs also place
        # the start and the end of a single move.
        dense = []
        if load_cloud is not None and c.boundary_margin > 0:
            dense = boundary_pairs(model, frame_times, usable, load_cloud, axis, sense, c.boundary_margin)
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
                hypotheses, model, scale, rms, residual, label, dense_scale = self.fit_hypotheses(
                    all_times, measurements, hypotheses, scale, fit_scale)
                info(f"  starts and stops: {len(dense)} consecutive-frame pairs; motor model: {label}; "
                     f"speed {np.degrees(model.speed):.3f} deg/s, ramp {model.ramp:.2f} s")

        # A commanded turn (capture.json or --turn-deg) is exact for a stepper
        # through a gear unless it stalls: use it when the data agree.
        fitted_total = float(np.degrees(model.total))
        total_fixed = False
        if commanded_deg is not None and abs(fitted_total - commanded_deg) <= c.loop_tolerance:
            model.total = np.radians(commanded_deg)
            model, scale, rms, residual, dense_scale = fit_motion(all_times, measurements, model.ordered(),
                                                                  False, scale, fit_scale)
            total_fixed = True
        # Free-form correction on top of the stepper profile (lost steps, wrong
        # number of moves, speed changes): see fit_correction.
        correction_rms = correction_max = None
        if c.correction_spacing > 0 and len(measurements) >= 50:
            classes = measurement_classes(measurements)
            factors = 1.0 + np.where(classes == 2, dense_scale, scale)
            combined, combined_classes, combined_factors = list(measurements), classes, factors
            if local:
                predicted = np.array([model.angle(all_times[[b]])[0] - model.angle(all_times[[a]])[0]
                                      for a, b, _, _, _, _ in local])
                observed = np.array([m[2] for m in local])
                moving_local = np.abs(predicted) > np.radians(0.5)
                if moving_local.sum() >= 10:
                    local_scale = float(np.clip(np.median(observed[moving_local] / predicted[moving_local]) - 1.0,
                                                -0.9, 0.2))
                    combined = combined + local
                    combined_classes = np.concatenate([classes, np.full(len(local), 3)])
                    combined_factors = np.concatenate([factors, np.full(len(local), 1.0 + local_scale)])
            correction, corrected = fit_correction(all_times, combined, model, combined_factors, combined_classes,
                                                   c.correction_spacing, c.correction_smoothing)
            if correction[0] is not None:
                model.correction = correction
                residual = corrected[:len(measurements)]
                rms = float(np.degrees(np.sqrt(np.mean(residual ** 2))))
                values = np.degrees(correction[1])
                inside = (correction[0] >= model.starts[0]) & (correction[0] <= model.end())
                deviation = values[inside] - np.median(values[inside]) if inside.any() else values
                correction_rms = float(np.sqrt(np.mean(deviation ** 2)))
                correction_max = float(np.abs(deviation).max())
                info(f"  motor model with free-form correction (knots every {c.correction_spacing:g} s): "
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
        info(f"  model-free solution: registration scale {100 * free_scales[0]:+.2f} %, "
             f"pair residual rms {free_rms:.2f} deg; motor model minus model-free: rms {gap_rms:.2f} deg, "
             f"max {gap_max:.2f} deg")
        reasons = self.choose(model, free, gap_rms, gap_max, scale)
        if reasons:
            info("  angles from the model-free solution: motor model rejected (" + "; ".join(reasons) + ")")
            chosen, label = free, "model-free solution"
            residual, rms, scale = free_residual, free_rms, free_scales[0]
        elif c.angle_source == "profile":
            info("  angles from the motor model with its correction (--angle-source profile)")
            chosen, label = model, "motor model with correction"
        else:
            info("  angles: mean of the motor model with its correction and the model-free solution "
                 "(the two agree)")
            chosen, label = MeanMotion(model, free), "mean of motor model and model-free solution"
            scale = free_scales[0]
        if model.end() > all_times.max() - 0.5 and t_off >= times[-1] - 1.0:
            info("  WARNING: the recording ends while the platform still turns: the end of the turn is missing "
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
                "pair_list": [[float(all_times[m[0]]), float(all_times[m[1]]), float(np.degrees(m[2])),
                               float(m[3]), int(m[4]), int(m[5]) if len(m) > 5 else 0] for m in saved],
                "boundary_pairs": len(dense),
                "correction_rms_deg": correction_rms, "correction_max_deg": correction_max,
                "dense_scale": dense_scale,
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
