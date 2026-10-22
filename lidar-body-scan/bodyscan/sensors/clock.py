"""Sensor clocks mapped to the session clock.

Every recorder stamps each packet or frame with the host time on arrival
(time.perf_counter_ns(), one clock for all processes of a PC). The sensor
also stamps it with its own clock. Arrival is the true time plus a delay that
is never negative (network, driver, scheduling), so the line that maps sensor
time to session time is the lower envelope of the (sensor time, arrival time)
points:

    session = offset + rate x sensor

fit_clock() takes the lower convex hull of the points and the hull edge that
spans the mean sensor time. That line is the solution of the linear programme
"minimise the mean delay subject to no negative delay", so it is not pulled
up by the occasional long delay (a stall of the receiving PC) and needs no
threshold. Arrivals earlier than the line by more than reject_sigma robust
standard deviations of a first least-squares fit are impossible for a correct
timestamp and are dropped first.

The fitted offset includes the smallest delay of the link (a constant that
the arrival times cannot separate from the clock offset). Constant offsets
between sensors are estimated later from the motion itself (the start jump,
and a time-offset parameter of the tracker).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ClockModel:
    """session time = offset + rate x sensor time [s]."""
    offset: float = 0.0
    rate: float = 1.0

    def to_session(self, sensor_time) -> np.ndarray:
        return self.offset + self.rate * np.asarray(sensor_time, dtype=np.float64)

    def to_sensor(self, session_time) -> np.ndarray:
        return (np.asarray(session_time, dtype=np.float64) - self.offset) / self.rate

    def shifted(self, seconds: float) -> "ClockModel":
        """The same clock with session times later by 'seconds'."""
        return ClockModel(self.offset + seconds, self.rate)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ClockModel":
        return cls(float(data.get("offset", 0.0)), float(data.get("rate", 1.0)))


def _lower_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of the lower convex hull of points sorted by x (monotone chain)."""
    hull: list[int] = []
    for i in range(len(x)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            cross = (x[b] - x[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (x[i] - x[a])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.array(hull, dtype=np.int64)


def fit_clock(sensor_times, arrival_times, reject_sigma: float = 6.0) -> tuple[ClockModel, dict]:
    """ClockModel from paired sensor and arrival times [s], and a report: points used and rejected,
    delays above the line (median, p95, max) and the clock rate error in parts per million."""
    s = np.asarray(sensor_times, dtype=np.float64).ravel()
    h = np.asarray(arrival_times, dtype=np.float64).ravel()
    if s.shape != h.shape or len(s) == 0:
        raise ValueError("sensor_times and arrival_times must be non-empty and of the same length")
    keep = np.isfinite(s) & np.isfinite(h)
    s, h = s[keep], h[keep]
    order = np.argsort(s, kind="stable")
    s, h = s[order], h[order]
    s0, h0 = float(s[0]), float(h[0])
    x, y = s - s0, h - h0
    rejected = 0
    if len(x) >= 3 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (intercept + slope * x)
        median = np.median(residual)
        spread = 1.4826 * np.median(np.abs(residual - median))
        if spread > 0:
            plausible = residual >= median - reject_sigma * spread
            rejected = int((~plausible).sum())
            x, y = x[plausible], y[plausible]
    if len(x) == 1 or np.ptp(x) == 0:
        rate = 1.0
        intercept = float(np.min(y - x))
    else:
        hull = _lower_hull(x, y)
        mean_x = float(np.mean(x))
        edge = int(np.clip(np.searchsorted(x[hull], mean_x) - 1, 0, len(hull) - 2))
        a, b = hull[edge], hull[edge + 1]
        rate = float((y[b] - y[a]) / (x[b] - x[a]))
        intercept = float(y[a] - rate * x[a])
    delays = y - (intercept + rate * x)
    model = ClockModel(h0 + intercept - rate * s0, rate)
    report = {"points": int(len(x)), "rejected": rejected,
              "delay_median_ms": float(1000 * np.median(delays)),
              "delay_p95_ms": float(1000 * np.percentile(delays, 95)),
              "delay_max_ms": float(1000 * np.max(delays)),
              "rate_error_ppm": float(1e6 * (rate - 1.0))}
    return model, report
