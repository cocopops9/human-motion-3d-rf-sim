"""Frame times on one clock."""

from __future__ import annotations

import numpy as np


def consistent_frame_times(person_times, sweep_times, host_times, tolerance: float = 0.5):
    """One time per frame, all on the SENSOR clock, strictly increasing.

    Preferred: the mean timestamp of the person's pixels. Without it (no
    person on the platform, e.g. after stepping off, or no valid column on
    the person) the median timestamp of the whole sweep, which is on the same
    clock. Without any sensor timestamp, the host time mapped onto the sensor
    clock by a robust straight-line fit (sensor = a * host + b). Mixing the
    two clocks would put those frames thousands of seconds away from the
    others, and every interpolation in time (np.interp needs increasing
    times) would then return wrong angles for whole parts of the recording
    (the tt13 fusion of 2026-10-02 failed that way). A frame whose time is
    more than 'tolerance' seconds off the line is also put back on it.
    Returns (times, number of frames repaired)."""
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
