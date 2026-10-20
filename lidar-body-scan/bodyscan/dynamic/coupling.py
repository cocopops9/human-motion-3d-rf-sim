"""Left-right coupling: a guess for a limb the sensor does not see, from its partner.

A person seen side-on shows the LiDAR one arm and mostly one leg; the far
arm is behind the trunk for most of the time. The points cannot tell where
it is, and without a guess the tracker keeps it still or lets it drift
(into the trunk, out to the side). People move their two arms, and their two
legs, in one of two ways during most activities:

    alternating   walking, running, running on the spot: each limb repeats
                  its partner's motion, mirrored, half a cycle later (the left
                  arm swings forward while the right one swings back); the
                  pelvis rises and falls twice per cycle, once per step
    together      jumps, squats, both arms raised: each limb mirrors its
                  partner at the same time; the pelvis rises and falls once
                  per cycle

Which of the two is going on, and the length of the cycle, is estimated
from what the sensor does see, in a window of a few seconds around every
frame: the observed limbs and the height of the pelvis. A limb repeats its
cycle (the autocorrelation of its angle peaks at the cycle T); in an
alternating gait the pelvis then repeats at T / 2 (its autocorrelation at
T / 2 is positive), in a motion together at T (negative at T / 2); when both
arms or both legs are observed, the correlation of the pair says it
directly; a deep squat or a jump (the pelvis drops while the hips flex)
without a repeating cycle counts as together. Where none of this is clear
(standing, gestures, turning) no limb is coupled.

The guess for a hidden limb A with partner B is then

    target_A(t) = mirror(rotations_B(t - shift)),   shift 0 or T / 2

(mirror: the same joint rotations seen in a mirror at the body's midplane,
x, -y, -z of the axis-angle in SMPL-X; each rotation relative to its parent
joint, so the guessed arm hangs from the chest as its partner does:
joint_guesses). After the refinement (stage 2) the tracker replaces the
hidden limb by its guess turned by the rotation from it observed whenever
the limb shows (a hand appearing in front of the trunk), carried between
observations (carried_offsets): the hidden limb moves like its partner, and
meets what was observed just before and just after (bodyscan.dynamic.track,
Tracker.complete_hidden_limbs). Rotations are combined as rotations
(quaternion products and spherical interpolation), never by adding their
axis-angle components.

The plan changes smoothly in time: the cycle is held steady, and the
confidence fades in and out over FADE seconds around every change of mode,
so that no guess switches on, off or over between two frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bodyscan.body import skeleton
from bodyscan.config import param


@dataclass
class CouplingConfig:
    """Hidden limbs guessed from their partner on the other side (after stage 2)."""
    enabled: bool = param(True, "guess a limb the sensor does not see from its partner on the other side, "
                                "mirrored, at the same time (jumps) or half a cycle later (walking, running)",
                          effect="false: a hidden limb keeps still or drifts (version 1.4)")
    offset_memory: float = param(0.4, "how long an offset from the partner's motion, seen while the limb showed, "
                                      "carries on after it disappears", unit="s",
                                 effect="longer: the hidden limb keeps its observed pose longer before it follows "
                                        "its partner")
    window: float = param(2.0, "half-width of the time window in which the kind of motion (alternating or "
                               "together) and its cycle are estimated", unit="s",
                          effect="longer: steadier estimate, slower to notice a change of activity")
    couple_legs: bool = param(True, "couple the legs as well (the far leg of a person seen side-on)")
    start_from_guess: bool = param(True, "the hidden limbs also start stage 2 from their guess (it replaces them "
                                         "after stage 2 in any case): a far arm left where stage 1 put it "
                                         "disturbs the refinement of the rest of the body",
                                   effect="false: the hidden limbs start stage 2 where stage 1 left them (synthetic "
                                          "side-on jumps: hand of the arm the sensor sees 0.24 m/s of velocity "
                                          "error with the guessed start, 0.34 without)")
    hide_guessed_limbs: bool = param(True, "a guessed arm standing where the sensor saw something else (empty "
                                           "space beyond it, a surface behind it) is moved out of its sight by the "
                                           "least turn of its shoulder and elbow (Tracker.hide_guessed_limbs)",
                                     effect="false: the guessed arm stays where its partner's mirror puts it "
                                            "(synthetic jog across the line of sight: visible body in free space "
                                            "4.2 % instead of 3.0 %, far wrist 120 instead of 104 mm)")


# Limbs, their partners and the joints coupled (indices into the 21 body joints), with the weight
# of each joint (the shoulder and hip carry the limb, the wrist and toes matter little).
LIMB_JOINTS = {
    "left_arm": ("left_collar", "left_shoulder", "left_elbow", "left_wrist"),
    "right_arm": ("right_collar", "right_shoulder", "right_elbow", "right_wrist"),
    "left_leg": ("left_hip", "left_knee", "left_ankle", "left_foot"),
    "right_leg": ("right_hip", "right_knee", "right_ankle", "right_foot"),
}
PARTNER = {"left_arm": "right_arm", "right_arm": "left_arm", "left_leg": "right_leg", "right_leg": "left_leg"}
JOINT_WEIGHTS = {"collar": 0.5, "shoulder": 1.0, "elbow": 0.8, "wrist": 0.2,
                 "hip": 1.0, "knee": 0.8, "ankle": 0.4, "foot": 0.2}
LIMB_PARTS = {
    "left_arm": ("left_upper_arm", "left_forearm", "left_hand"),
    "right_arm": ("right_upper_arm", "right_forearm", "right_hand"),
    "left_leg": ("left_thigh", "left_shank", "left_foot"),
    "right_leg": ("right_thigh", "right_shank", "right_foot"),
}
LIMBS = tuple(LIMB_JOINTS)

# A limb counts as fully observed when this fraction of its vertices is seen with points
# (the side towards the sensor is about half of it).
OBSERVED_FULL = 0.3

TOGETHER, ALTERNATING, UNKNOWN = 1, -1, 0


def body_indices(limb: str) -> list[int]:
    """Indices into the 21 body joints of the coupled joints of a limb."""
    return [skeleton.JOINT[name] - 1 for name in LIMB_JOINTS[limb]]


def joint_weights(limb: str) -> np.ndarray:
    return np.array([JOINT_WEIGHTS[name.split("_", 1)[1]] for name in LIMB_JOINTS[limb]])


def mirror(rotations: np.ndarray) -> np.ndarray:
    """Axis-angle rotations (..., 3) seen in a mirror at the body's midplane (x -> -x in SMPL-X
    model space): (x, -y, -z). In SMPL-X every joint frame is parallel to the model axes in the
    rest pose, so this maps a pose of the left limb onto the same pose of the right limb."""
    out = np.array(rotations, dtype=np.float64, copy=True)
    out[..., 1:] *= -1.0
    return out


def limb_observation(observed: np.ndarray) -> dict:
    """{limb: (F,) 0..1} from the per-part observed fractions (F, parts) of the tracker: the
    fraction of each part's vertices seen with points, averaged over the limb's parts weighted by
    nothing (a hand counts as much as an upper arm: each tells where the limb is), scaled so that
    OBSERVED_FULL counts as fully observed."""
    out = {}
    observed = np.nan_to_num(np.asarray(observed, dtype=np.float64), nan=0.0)
    for limb, parts in LIMB_PARTS.items():
        columns = [skeleton.PART_NAMES.index(part) for part in parts]
        out[limb] = np.clip(observed[:, columns].mean(axis=1) / OBSERVED_FULL, 0.0, 1.0)
    return out


# ----------------------------------------------------------------------------
# Signals of the motion
# ----------------------------------------------------------------------------

def _rotation_matrices(rotvec: np.ndarray) -> np.ndarray:
    from bodyscan.body.rotations import axis_angle_to_matrix_np
    return axis_angle_to_matrix_np(rotvec)


def signals(joints: np.ndarray, root_matrix: np.ndarray, body: np.ndarray) -> dict:
    """Angles that show the cycle of a motion, per frame [rad]:
        legs  (F, 2) left, right: forward angle of the thigh (knee - hip) in the pelvis frame
              (0: straight down, positive: forward)
        arms  (F, 2): forward angle of the line shoulder-wrist in the chest frame (0: hanging)
        pelvis_z (F,): height of the pelvis [m]
    joints (F, >=22, 3) world positions; root_matrix (F, 3, 3) the world root rotation of the
    tracker (world = root . WORLD_FROM_MODEL . model: in the model y is up, z forward); body
    (F, 21, 3) joint rotations (for the chest frame)."""
    J = skeleton.JOINT
    pelvis = root_matrix @ skeleton.WORLD_FROM_MODEL
    forward_p, up_p = pelvis[:, :, 2], pelvis[:, :, 1]
    chest = pelvis
    for name in ("spine1", "spine2", "spine3"):
        chest = chest @ _rotation_matrices(body[:, J[name] - 1])
    forward_c, up_c = chest[:, :, 2], chest[:, :, 1]

    def angle(vector, forward, up):
        return np.arctan2(np.einsum("fi,fi->f", vector, forward), -np.einsum("fi,fi->f", vector, up))

    legs = np.stack([angle(joints[:, J[f"{s}_knee"]] - joints[:, J[f"{s}_hip"]], forward_p, up_p)
                     for s in ("left", "right")], axis=1)
    arms = np.stack([angle(joints[:, J[f"{s}_wrist"]] - joints[:, J[f"{s}_shoulder"]], forward_c, up_c)
                     for s in ("left", "right")], axis=1)
    return {"legs": legs, "arms": arms, "pelvis_z": joints[:, J["pelvis"], 2].astype(np.float64)}


# ----------------------------------------------------------------------------
# Kind of motion and cycle
# ----------------------------------------------------------------------------

@dataclass
class CouplingPlan:
    """Per frame: mode (TOGETHER, ALTERNATING, UNKNOWN), confidence 0..1, cycle [s] (NaN where
    unknown) and the time shift of the partner [s] (0 together, cycle / 2 alternating)."""
    mode: np.ndarray
    confidence: np.ndarray
    cycle: np.ndarray
    shift: np.ndarray

    def summary(self) -> dict:
        n = max(len(self.mode), 1)
        alternating = self.mode == ALTERNATING
        return {"together": float(np.mean(self.mode == TOGETHER)), "alternating": float(np.mean(alternating)),
                "unknown": float(np.mean(self.mode == UNKNOWN)),
                "cycle_median_s": float(np.nanmedian(self.cycle[alternating])) if alternating.any() else None,
                "frames": n}


def _weighted_correlation(x, y, w):
    w = np.asarray(w, dtype=np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    mx, my = (w * x).sum() / total, (w * y).sum() / total
    dx, dy = x - mx, y - my
    sxx, syy = (w * dx * dx).sum(), (w * dy * dy).sum()
    if sxx <= 1e-12 or syy <= 1e-12:
        return 0.0
    return float((w * dx * dy).sum() / np.sqrt(sxx * syy))


def _detrend(x, w):
    """x minus its weighted linear trend over the window."""
    n = len(x)
    s = np.linspace(-1.0, 1.0, n)
    total = max(w.sum(), 1e-9)
    ms, mx = (w * s).sum() / total, (w * x).sum() / total
    var = (w * (s - ms) ** 2).sum()
    slope = (w * (s - ms) * (x - mx)).sum() / var if var > 1e-12 else 0.0
    return x - mx - slope * (s - ms)


def _autocorrelation(x, w, lags):
    """Normalised autocorrelation of x (detrended, weights w: 0 where missing) at integer lags."""
    x = _detrend(x, w)
    out = np.zeros(len(lags))
    for k, lag in enumerate(lags):
        a, b = x[:len(x) - lag], x[lag:]
        wa = w[:len(x) - lag] * w[lag:]
        if wa.sum() < 8:
            out[k] = np.nan
            continue
        num = (wa * a * b).sum()
        den = np.sqrt((wa * a * a).sum() * (wa * b * b).sum())
        out[k] = num / den if den > 1e-12 else np.nan
    return out


def _cycle(x, w, step, shortest=0.3, longest=1.8, minimum=0.3):
    """Length [s] of the repeating cycle of x (first autocorrelation peak above 'minimum' after a dip
    below half of it), its autocorrelation value; (nan, 0) when x does not repeat."""
    lags = np.arange(max(int(round(shortest / step)), 2), int(round(longest / step)) + 2)
    r = _autocorrelation(x, w, lags)
    dipped = False
    for k in range(1, len(lags) - 1):
        if not np.isfinite(r[k]):
            continue
        if r[k] < 0.5 * minimum:
            dipped = True
        if dipped and r[k] >= minimum and r[k] >= r[k - 1] and r[k] >= r[k + 1]:
            # parabolic refinement of the peak
            a, b, c = r[k - 1], r[k], r[k + 1]
            denominator = a - 2 * b + c
            offset = 0.5 * (a - c) / denominator if abs(denominator) > 1e-9 else 0.0
            return (lags[k] + np.clip(offset, -0.5, 0.5)) * step, float(b)
    return np.nan, 0.0


def _cycle_from_peaks(x, w, step, prominence=np.radians(6.0), shortest=0.35, longest=1.8):
    """Length [s] of the cycle from the times between successive maxima (and between successive
    minima) of x, for bursts too short for the autocorrelation (a run of three strides); its
    regularity 0..1 (1: equal intervals); (nan, 0) with fewer than two intervals."""
    x = _detrend(x, w)
    smooth = np.convolve(x, np.ones(3) / 3.0, mode="same")
    intervals = []
    for sign in (1.0, -1.0):
        y = sign * smooth
        peaks = [i for i in range(1, len(y) - 1)
                 if w[i] > 0.3 and y[i] >= y[i - 1] and y[i] > y[i + 1]
                 and y[i] - min(y[max(i - int(0.4 / step), 0):i + int(0.4 / step) + 1]) > prominence]
        times = np.array(peaks, dtype=np.float64) * step
        gaps = np.diff(times)
        intervals.extend(gaps[(gaps >= shortest) & (gaps <= longest)])
    if len(intervals) < 2:
        return np.nan, 0.0
    intervals = np.array(intervals)
    length = float(np.median(intervals))
    spread = float(np.median(np.abs(intervals - length))) / length
    return length, float(np.clip(1.0 - spread / 0.15, 0.0, 1.0))


def _autocorrelation_at(x, w, lag_s, step):
    """Autocorrelation at a lag in seconds (linear between integer lags)."""
    lag = lag_s / step
    low = int(np.floor(lag))
    if low < 1:
        return np.nan
    values = _autocorrelation(x, w, np.array([low, low + 1]))
    if not np.all(np.isfinite(values)):
        return np.nan
    u = lag - low
    return float((1 - u) * values[0] + u * values[1])


def estimate(times: np.ndarray, sig: dict, observation: dict, config: CouplingConfig | None = None,
             step: float | None = None) -> CouplingPlan:
    """Mode and cycle of the motion around every frame (see the module docstring).
    times (F,) [s]; sig from signals(); observation from limb_observation()."""
    config = config or CouplingConfig()
    times = np.asarray(times, dtype=np.float64)
    count = len(times)
    if count < 4:
        empty = np.zeros(count)
        return CouplingPlan(np.zeros(count, dtype=np.int64), empty, np.full(count, np.nan), empty.copy())
    if step is None:
        steps = np.diff(times)
        step = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 0.05
    grid = np.arange(times[0], times[-1] + 0.5 * step, step)
    # grid samples far from any frame (frames dropped) carry no weight
    nearest = np.abs(times[np.clip(np.searchsorted(times, grid), 0, count - 1)] - grid)
    nearest = np.minimum(nearest, np.abs(times[np.clip(np.searchsorted(times, grid) - 1, 0, count - 1)] - grid))
    present = (nearest <= 1.5 * step).astype(np.float64)

    def on_grid(values):
        return np.interp(grid, times, values)

    z = on_grid(sig["pelvis_z"])
    legs = [on_grid(np.unwrap(sig["legs"][:, k])) for k in range(2)]
    arms = [on_grid(np.unwrap(sig["arms"][:, k])) for k in range(2)]
    leg_obs = [on_grid(observation[f"{s}_leg"]) * present for s in ("left", "right")]
    arm_obs = [on_grid(observation[f"{s}_arm"]) * present for s in ("left", "right")]
    half = max(int(round(config.window / step)), 8)
    n = len(grid)
    score = np.zeros(n)
    weight_total = np.zeros(n)
    cycle = np.full(n, np.nan)
    every = 2                                      # estimated every other sample, interpolated between
    centres = np.arange(0, n, every)
    for g in centres:
        # a full window wherever the take is long enough: near its ends the window is moved inwards
        # (a window cut short by the end of the take often misses the cycle, and the guess would
        # switch off and on there)
        lo = min(max(g - half, 0), max(n - (2 * half + 1), 0))
        hi = min(lo + 2 * half + 1, n)
        if hi - lo < 12:
            continue
        sl = slice(lo, hi)
        w_present = present[sl]
        zs = z[sl]
        z_spread = float(np.sqrt(np.average((_detrend(zs, w_present)) ** 2, weights=np.maximum(w_present, 1e-9))))
        votes, weights = [], []
        best_cycle, best_strength = np.nan, 0.0
        # one observed limb and the pelvis: legs first (the thigh repeats its cycle most clearly)
        for group, observed_group in ((legs, leg_obs), (arms, arm_obs)):
            means = [float(o[sl].mean()) for o in observed_group]
            side = int(np.argmax(means))
            if means[side] < 0.5:
                continue
            x, w = group[side][sl], observed_group[side][sl]
            spread = float(np.sqrt(np.average(_detrend(x, w) ** 2, weights=np.maximum(w, 1e-9))))
            if spread < np.radians(5.0):
                continue
            length, strength = _cycle(x, w, step)
            if not np.isfinite(length):
                # a short burst (three strides of a run): the times between its peaks
                length, regularity = _cycle_from_peaks(x, w, step)
                strength = 0.5 * regularity
            if np.isfinite(length):
                if strength > best_strength:
                    best_cycle, best_strength = length, strength
                if z_spread > 0.006:
                    half_lag = _autocorrelation_at(zs, w_present, 0.5 * length, step)
                    if np.isfinite(half_lag):
                        votes.append(-np.clip(half_lag / 0.3, -1.0, 1.0))
                        weights.append(means[side] * np.clip((strength - 0.2) / 0.3, 0.0, 1.0)
                                       * np.clip(z_spread / 0.015, 0.0, 1.0))
            # a squat or a jump that does not repeat: the pelvis drops while the hips flex
            if group is legs and float(zs.max() - zs.min()) > 0.08:
                r = _weighted_correlation(x, -zs, w)
                if r > 0.6:
                    votes.append(1.0)
                    weights.append(0.6 * means[side] * np.clip((r - 0.6) / 0.2, 0.0, 1.0))
        # both limbs of a pair observed: their correlation says it directly
        for group, observed_group in ((arms, arm_obs), (legs, leg_obs)):
            w = observed_group[0][sl] * observed_group[1][sl]
            if w.mean() < 0.4:
                continue
            a, b = group[0][sl], group[1][sl]
            spreads = [float(np.sqrt(np.average(_detrend(v, w) ** 2, weights=np.maximum(w, 1e-9)))) for v in (a, b)]
            if min(spreads) < np.radians(5.0):
                continue
            r = _weighted_correlation(_detrend(a, w), _detrend(b, w), w)
            votes.append(np.clip(r / 0.5, -1.0, 1.0))
            weights.append(float(w.mean()) * np.clip((abs(r) - 0.2) / 0.3, 0.0, 1.0))
        if weights and sum(weights) > 1e-6:
            score[g] = float(np.dot(votes, weights) / sum(weights))
            weight_total[g] = float(sum(weights))
        cycle[g] = best_cycle
    # fill the samples between the centres, smooth over half a second
    fill = np.arange(n)
    score = np.interp(fill, centres, score[centres])
    weight_total = np.interp(fill, centres, weight_total[centres])
    valid = centres[np.isfinite(cycle[centres])]
    if len(valid):
        # the cycle of a motion changes slowly: carried from the nearest centre where one was found
        # (within half a window), the jumps of single estimates taken out by a running median over a
        # second and a short smoothing (the shift of the partner follows the cycle, and a guess whose
        # shift jumps plays its partner's motion faster or backwards for a moment)
        cycle = np.interp(fill, valid, cycle[valid])
        cycle = np.where(_distance_to(fill, valid) * step <= 0.5 * config.window, cycle, np.nan)
        cycle = _steady(cycle, max(int(round(0.5 / step)), 1), 0.2 / step)
    else:
        cycle = np.full(n, np.nan)
    width = max(int(round(0.25 / step)), 1)
    score = _running_median(score, width)
    confidence_grid = np.clip((np.abs(score) - 0.2) / 0.5, 0.0, 1.0) * np.clip(weight_total / 0.5, 0.0, 1.0)
    mode_grid = np.where(confidence_grid > 0, np.sign(score), 0).astype(np.int64)
    # alternating needs a cycle
    no_cycle = (mode_grid == ALTERNATING) & ~np.isfinite(cycle)
    mode_grid[no_cycle], confidence_grid[no_cycle] = UNKNOWN, 0.0
    # the confidence changes smoothly and fades out towards every change of mode (to unknown too):
    # the guess of a limb then never switches on or off, or from one mode to the other, between two
    # frames (it is blended with the tracked limb by the confidence)
    from scipy.ndimage import gaussian_filter1d
    confidence_grid = gaussian_filter1d(confidence_grid, max(0.1 / step, 0.5), mode="nearest")
    changes = np.flatnonzero(mode_grid[1:] != mode_grid[:-1]) + 0.5
    if len(changes):
        to_change = (_distance_to(fill, changes) - 0.5) * step
        confidence_grid = confidence_grid * np.clip(to_change / FADE, 0.0, 1.0)
    confidence_grid[mode_grid == UNKNOWN] = 0.0
    # back to the frames
    index = np.clip(np.round((times - grid[0]) / step).astype(int), 0, n - 1)
    mode = mode_grid[index]
    confidence = confidence_grid[index]
    frame_cycle = cycle[index]
    shift = np.where(mode == ALTERNATING, 0.5 * np.nan_to_num(frame_cycle), 0.0)
    confidence = np.where(mode == UNKNOWN, 0.0, confidence)
    return CouplingPlan(mode, confidence, frame_cycle, shift)


# Time over which the confidence of the plan fades in after, and out before, a change of mode [s].
FADE = 0.25


def _running_median(values, half):
    out = np.empty_like(values)
    for i in range(len(values)):
        out[i] = np.median(values[max(i - half, 0):i + half + 1])
    return out


def _distance_to(positions: np.ndarray, marks: np.ndarray) -> np.ndarray:
    """Distance of every position to the nearest of the sorted 'marks' (same units)."""
    positions = np.asarray(positions, dtype=np.float64)
    marks = np.asarray(marks, dtype=np.float64)
    after = np.clip(np.searchsorted(marks, positions), 0, len(marks) - 1)      # first mark at or after
    before = np.clip(after - 1, 0, len(marks) - 1)
    return np.minimum(np.abs(positions - marks[before]), np.abs(positions - marks[after]))


def _steady(values: np.ndarray, half: int, sigma: float) -> np.ndarray:
    """A running median over +-half samples and a Gaussian smoothing (sigma samples) of the finite
    stretches of values (NaN stays NaN)."""
    from scipy.ndimage import gaussian_filter1d
    out = np.array(values, dtype=np.float64, copy=True)
    finite = np.isfinite(out)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], finite.astype(int), [0]])))
    for first, last in zip(edges[::2], edges[1::2]):
        stretch = _running_median(out[first:last], half)
        out[first:last] = gaussian_filter1d(stretch, sigma, mode="nearest") if last - first > 1 else stretch
    return out


# ----------------------------------------------------------------------------
# Guesses
# ----------------------------------------------------------------------------

@dataclass
class JointGuess:
    """The guess for one limb: the rotations (F, J, 3) axis-angle of its joints, each relative to its
    parent joint (the joint rotations of SMPL-X), and the source weight (F,): the confidence of the plan
    times how well the partner was observed at the times its motion is taken from."""
    rotations: np.ndarray
    weight: np.ndarray


def joint_guesses(times: np.ndarray, body: np.ndarray, plan: CouplingPlan, observation: dict,
                  limbs=LIMBS) -> dict:
    """{limb: JointGuess}. The partner's joint rotations at t - shift (and at t + shift for an alternating
    motion, the two blended by how well the partner was observed then) seen in the mirror of the body's
    midplane (mirror()): the guessed arm hangs from the chest as its partner does, the guessed leg from
    the pelvis. Mirrored relative to the trunk, not in the world: a person sees two arms as moving alike
    when they do so relative to the shoulders (version 1.5 mirrored the world orientations at the
    vertical plane of the pelvis' facing, which doubled every twist of the chest: on the s02 side-on jumps,
    with the chest tracked twisted by 23 to 28 deg (p95), the far wrist was 6 to 14 cm (median) away from
    the mirror image of the near one relative to the chest).
    The partner's motion is interpolated in time on quaternions (rotations, not their components). The
    source weight is smoothed over a few frames, so that a partner hidden for a frame or two, whose pose
    there the refinement carries smoothly, does not switch the guess off and on, and fades out towards
    the ends of the take where half a cycle before or after leaves it."""
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d
    from bodyscan.body.rotations import axis_angle_to_quaternion_np, interpolate_rotations, slerp_np, \
        quaternion_to_axis_angle_np
    times = np.asarray(times, dtype=np.float64)
    shift = np.asarray(plan.shift, dtype=np.float64)
    alternating = shift > 0
    out = {}
    for limb in limbs:
        partner = PARTNER[limb]
        source = mirror(np.asarray(body, dtype=np.float64)[:, body_indices(partner)])           # (F, J, 3)
        seen = gaussian_filter1d(maximum_filter1d(observation[partner], 5, mode="nearest"), 2.0, mode="nearest")
        before, after = times - shift, times + shift
        seen_before = np.interp(before, times, seen) * np.where(alternating,
                                                                np.clip((before - times[0]) / FADE, 0.0, 1.0), 1.0)
        seen_after = np.interp(after, times, seen) * np.clip((times[-1] - after) / FADE, 0.0, 1.0) * alternating
        # the share of the partner half a cycle later: by how well it was observed then and before
        blend = (seen_after + 0.05 * alternating) / (seen_before + seen_after + 0.05 * (1.0 + alternating))
        support = np.clip(np.maximum(seen_before, seen_after), 0.0, 1.0)
        rotation = axis_angle_to_quaternion_np(interpolate_rotations(times, source, before))
        if np.any(blend > 0):
            later = axis_angle_to_quaternion_np(interpolate_rotations(times, source, after))
            rotation = slerp_np(rotation, later, np.broadcast_to(blend[:, None], rotation.shape[:-1]))
        out[limb] = JointGuess(quaternion_to_axis_angle_np(rotation), plan.confidence * support)
    return out


IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def carried_offsets(times: np.ndarray, offsets: np.ndarray, anchored: np.ndarray, memory: float,
                    longest: float = 3.0) -> np.ndarray:
    """Rotation offsets (F, J, 4) unit quaternions of a limb from its guess, known where the limb was
    observed ('anchored' > 0.5), carried into the frames between: across a gap between two observations at
    most 'longest' seconds apart they turn from the one to the other (spherical interpolation), so the
    limb meets its observations on both sides and in between moves like its partner; before the first and
    after the last observation, and across longer gaps, they fade towards no offset with exp(-dt / memory)
    of the nearest observations (shared between the one before and the one after by their fades): far
    from any observation, the guess alone."""
    from bodyscan.body.rotations import slerp_np
    times = np.asarray(times, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    count = len(times)
    identity = np.broadcast_to(IDENTITY, offsets.shape[1:]).copy()
    out = np.broadcast_to(identity, offsets.shape).copy()
    marks = np.flatnonzero(np.asarray(anchored) > 0.5)
    if len(marks) == 0 or memory <= 0:
        return out
    for i in range(count):
        k = np.searchsorted(marks, i)
        if k < len(marks) and marks[k] == i:
            out[i] = offsets[i]
            continue
        a = marks[k - 1] if k > 0 else None
        b = marks[k] if k < len(marks) else None
        joints = offsets.shape[1:-1]
        if a is not None and b is not None and times[b] - times[a] <= longest:
            u = (times[i] - times[a]) / max(times[b] - times[a], 1e-9)
            out[i] = slerp_np(offsets[a], offsets[b], np.full(joints, u))
            continue
        fade_a = np.exp(-(times[i] - times[a]) / memory) if a is not None else 0.0
        fade_b = np.exp(-(times[b] - times[i]) / memory) if b is not None else 0.0
        if fade_a + fade_b <= 1e-9:
            continue
        if a is not None and b is not None:
            mixed = slerp_np(offsets[a], offsets[b], np.full(joints, fade_b / (fade_a + fade_b)))
        else:
            mixed = offsets[a] if a is not None else offsets[b]
        out[i] = slerp_np(identity, mixed, np.full(joints, max(fade_a, fade_b)))
    return out
