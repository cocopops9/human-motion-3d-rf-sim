"""Static background of a range-image sensor and the foreground test."""

from __future__ import annotations

import numpy as np

from bodyscan.io.recording import median_range


def mixed_pixel_mask(range_m: np.ndarray, jump: float) -> np.ndarray:
    """Pixels that break the range profile on both sides, horizontally or vertically:
    the beam hit an edge and returned a range between the foreground and the background."""
    r = range_m
    left, right = np.roll(r, 1, axis=1), np.roll(r, -1, axis=1)
    up, down = np.full_like(r, np.nan), np.full_like(r, np.nan)
    up[1:], down[:-1] = r[:-1], r[1:]
    with np.errstate(invalid="ignore"):
        rows = (np.abs(r - left) > jump) & (np.abs(r - right) > jump) & (np.abs(left + right - 2 * r) > jump)
        cols = (np.abs(r - up) > jump) & (np.abs(r - down) > jump) & (np.abs(up + down - 2 * r) > jump)
    return rows | cols


class RangeBackground:
    """Per-pixel range of the empty scene. A pixel is foreground when it is
    closer than the background by more than max(threshold, relative x range),
    or when it has a return where the empty scene had none."""

    def __init__(self, range_m: np.ndarray, threshold: float = 0.05, relative: float = 0.01):
        self.range_m = range_m
        self.threshold = threshold
        self.relative = relative

    @classmethod
    def from_frames(cls, ranges, threshold=0.05, relative=0.01, min_fraction=0.5) -> "RangeBackground":
        """Median of empty-scene frames (or, without them, of a sequence in
        which everything of interest moves: what stays is the background)."""
        return cls(median_range(list(ranges), min_fraction), threshold, relative)

    def foreground(self, range_m: np.ndarray) -> np.ndarray:
        margin = np.maximum(self.threshold, self.relative * np.nan_to_num(self.range_m, nan=0.0))
        with np.errstate(invalid="ignore"):
            return (range_m < self.range_m - margin) | (np.isfinite(range_m) & np.isnan(self.range_m))
