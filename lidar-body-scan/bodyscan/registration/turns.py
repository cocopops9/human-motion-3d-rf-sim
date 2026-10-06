"""Global turn angles from relative turn measurements."""

from __future__ import annotations

import numpy as np

from bodyscan.geometry import wrapped_degrees, yaw_of


def solve_turns(count: int, chain: np.ndarray, measurements, huber: float, iterations: int = 5) -> np.ndarray:
    """Least-squares turn angles [rad] of 'count' clouds from relative turns.

    Each measurement (a Pair i -> j) observes yaw_j - yaw_i; its angle is
    unwrapped near the chained value, so pairs across the full turn close the
    loop. Weight: rotational information about the vertical (information
    matrix entry [2, 2]) times a Huber factor with scale 'huber' [rad]."""
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
    for _ in range(iterations):
        a = np.zeros((len(rows) + 1, count))
        b = np.zeros(len(rows) + 1)
        for r, (i, j, observed, _) in enumerate(rows):
            w = np.sqrt(weights[r])
            a[r, j], a[r, i], b[r] = w, -w, w * observed
        a[-1, 0], b[-1] = 1e3, 0.0                      # yaw of cloud 0 fixed at 0
        yaws = np.linalg.lstsq(a, b, rcond=None)[0]
        residual = np.array([abs((yaws[j] - yaws[i]) - observed) for i, j, observed, _ in rows])
        weights = base * np.minimum(1.0, huber / np.maximum(residual, 1e-9))
    return yaws
