"""Platform angle versus time: the models.

MotionModel is the interface (angle(t) in radians, describe()); a pipeline
only ever calls these two. Implementations:

    StepperMotion  trapezoidal stepper profile of one move or of a sequence of
                   moves with stops (what the MATLAB timer sends), plus an
                   optional free-form correction c(t)
    FreeMotion     no motor model: a baseline (the chained angles) plus c(t)
    MeanMotion     mean of two models that agree (their errors are partly independent)

A new model (for example a platform with an encoder, or a different motor
profile) is a new subclass of MotionModel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class MotionModel(ABC):
    correction: tuple[np.ndarray, np.ndarray] | None = None   # (knots [s], values [rad])

    @abstractmethod
    def angle(self, t) -> np.ndarray:
        """Platform angle [rad] at the times t [s]."""

    @abstractmethod
    def describe(self) -> str:
        """One line for the console and the report."""

    def _corrected(self, t, base):
        if self.correction is None:
            return base
        knots, values = self.correction
        return base + np.interp(t, knots, values)


def profile_angle(t, start: float, speed: float, ramp: float, total: float) -> np.ndarray:
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


class StepperMotion(MotionModel):
    """One stepper move, or several moves of 'move' radians each in sequence
    (the last one takes the rest of 'total').

    Every move has the trapezoidal profile of profile_angle with the same
    cruise speed and ramp time; 'starts' holds the start time of every move.
    girogirotondo_timer.m sends one command per full turn at most (larger
    step counts may not fit the controller's integer), so a turn of several
    laps is a sequence of moves with short stops in between; a single
    continuous move is the special case with one start."""

    def __init__(self, speed, ramp, starts, total, move=None):
        self.speed = float(speed)
        self.ramp = float(ramp)
        self.starts = np.atleast_1d(np.asarray(starts, dtype=np.float64)).copy()
        self.correction = None
        self.total = float(total)
        self.move = move

    def moves(self) -> list[float]:
        count = len(self.starts)
        if count == 1:
            return [self.total]
        last = max(self.total - self.move * (count - 1), 1e-3)
        return [self.move] * (count - 1) + [last]

    def duration(self, angle: float) -> float:
        ramp_angle = 0.5 * self.speed * max(self.ramp, 1e-3)
        if 2 * ramp_angle > angle:
            return 2 * angle / self.speed
        return self.ramp + angle / self.speed

    def ordered(self) -> "StepperMotion":
        """A move starts only after the previous one has finished."""
        moves = self.moves()
        for m in range(1, len(self.starts)):
            self.starts[m] = max(self.starts[m], self.starts[m - 1] + self.duration(moves[m - 1]))
        return self

    def profile(self, t) -> np.ndarray:
        """The stepper profile alone, without the correction."""
        return sum(profile_angle(t, start, self.speed, self.ramp, angle)
                   for start, angle in zip(self.starts, self.moves()))

    def angle(self, t) -> np.ndarray:
        return self._corrected(t, self.profile(t))

    def end(self) -> float:
        return float(self.starts[-1] + self.duration(self.moves()[-1]))

    def vector(self) -> np.ndarray:
        return np.concatenate([[self.speed, self.ramp, self.total], self.starts])

    def from_vector(self, values) -> "StepperMotion":
        return StepperMotion(values[0], values[1], values[3:], values[2], self.move).ordered()

    def describe(self) -> str:
        text = (f"speed {np.degrees(self.speed):.3f} deg/s, ramp {self.ramp:.2f} s, "
                f"total {np.degrees(self.total):.2f} deg")
        if len(self.starts) == 1:
            return f"one move from {self.starts[0]:.2f} s, " + text
        stops = [self.starts[m + 1] - (self.starts[m] + self.duration(a))
                 for m, a in enumerate(self.moves()[:-1])]
        return (f"{len(self.starts)} moves of up to {np.degrees(self.move):g} deg from {self.starts[0]:.2f} s, "
                + text + f", stops between moves {', '.join(f'{s:.2f}' for s in stops)} s")


class FreeMotion(MotionModel):
    """Platform angle without a motor model: a baseline (the chained angles,
    interpolated) plus a piecewise linear correction fitted to all the pair
    measurements (see fitting.fit_free)."""

    def __init__(self, base_times, base_angles, correction=None):
        self.base_times = np.asarray(base_times, dtype=np.float64)
        self.base_angles = np.asarray(base_angles, dtype=np.float64)
        self.correction = correction

    def angle(self, t) -> np.ndarray:
        return self._corrected(t, np.interp(t, self.base_times, self.base_angles))

    def describe(self) -> str:
        return "angles solved from all the pairs, no motor model"


class MeanMotion(MotionModel):
    """Mean of two angle solutions that agree (motor model with its correction,
    and the model-free solution): their errors are partly independent, so
    the mean is more accurate than either (simulation: 1.2 to 1.6 deg rms
    against 1.2 to 2.0 for the better and worse of the two)."""

    def __init__(self, first: MotionModel, second: MotionModel):
        self.parts = (first, second)
        self.correction = None

    def angle(self, t) -> np.ndarray:
        return 0.5 * (self.parts[0].angle(t) + self.parts[1].angle(t))

    def describe(self) -> str:
        return "mean of the motor model with its correction and the model-free solution"


# Name kept for readers of the earlier single-file scripts.
Motion = StepperMotion
