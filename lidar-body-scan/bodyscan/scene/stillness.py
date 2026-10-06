"""Still periods of a person turning in place by steps, and one keyframe per stop.

Frames recorded while the person steps show legs and arms in transit; only
the still periods are used. A per-pixel motion score finds them, and the
keyframe of a stop is the per-pixel median of up to keyframe_frames still
frames at its centre (which also reduces the range noise and freezes the
breathing at mid-breath)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bodyscan.config import param
from bodyscan.io.recording import median_range
from bodyscan.log import Progress


@dataclass
class StillConfig:
    """Still periods and keyframes (in-place capture)."""
    motion_threshold: float = param(0.03, "range change that counts as motion for a pixel", unit="m")
    still_factor: float = param(2.5, "a frame is still when its motion score is below this times the 20th "
                                     "percentile of the scores", effect="larger accepts frames with more motion")
    still_max: float = param(0.25, "upper bound of the still threshold (fraction of changed person pixels)")
    min_still: int = param(5, "shortest still period", unit="frames")
    keyframe_frames: int = param(20, "frames (centre of each still period) in the per-pixel median; 20 frames = "
                                     "2 s, about half a breath", unit="frames")
    min_pixels: int = param(800, "frames with fewer person pixels are ignored")
    keyframe_stride: int = param(1, "use every n-th still period", effect="2 halves the keyframes and divides "
                                                                         "the registration time by about 4")
    skip_seconds: float = param(0.0, "ignore the start of the capture", unit="s")


@dataclass
class MotionTrace:
    scores: np.ndarray      # score[i]: fraction of person pixels that changed between frames i-1 and i
    times: np.ndarray       # host time of every frame [s]
    sizes: np.ndarray       # person pixels per frame
    columns: np.ndarray     # fraction of columns received per frame


def motion_trace(source, isolator, motion_threshold: float) -> MotionTrace:
    """A pixel counts as changed if it belongs to the person in only one of two
    consecutive frames, or if its range moved by more than motion_threshold."""
    scores, times, sizes, columns = [], [], [], []
    previous = None
    progress = Progress("motion", len(source))
    for index in range(len(source)):
        frame = source.load(index)
        mask, _ = isolator.mask(frame.range_m)
        if previous is None:
            score = np.nan
        else:
            previous_range, previous_mask = previous
            union = mask | previous_mask
            with np.errstate(invalid="ignore"):
                moved = np.abs(frame.range_m - previous_range) > motion_threshold
            changed = union & ((mask != previous_mask) | moved)
            score = changed.sum() / max(union.sum(), 1)
        scores.append(score)
        times.append(frame.host_time)
        sizes.append(int(mask.sum()))
        columns.append(frame.columns_ok)
        previous = (frame.range_m, mask)
        progress.maybe(index + 1, 200)
    times = np.array(times, dtype=np.float64)
    if not np.all(np.isfinite(times)):
        times = np.arange(len(times)) * 0.1
    return MotionTrace(np.array(scores), times - times[0], np.array(sizes), np.array(columns))


def still_segments(trace: MotionTrace, config: StillConfig) -> tuple[list, float]:
    """Runs of frames that differ little from both neighbours, and the threshold used.

    The threshold adapts to the run: noise and silhouette flicker set the
    floor of the score, so a frame is still when its score is below
    still_factor times the 20th percentile (never above still_max)."""
    scores = trace.scores
    finite = scores[np.isfinite(scores)]
    base = float(np.percentile(finite, 20)) if finite.size else 0.0
    threshold = min(config.still_max, max(config.still_factor * base, 0.02))
    low = np.nan_to_num(scores, nan=np.inf) < threshold
    still = np.zeros(len(scores), dtype=bool)
    still[:-1] = low[:-1] & low[1:]                    # unchanged with respect to the previous and the next frame
    still &= trace.columns >= 0.99
    sizes = np.where(trace.times < config.skip_seconds, 0, trace.sizes)
    still &= sizes >= config.min_pixels
    segments, start = [], None
    for index, flag in enumerate(np.append(still, False)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= config.min_still:
                segments.append((start, index))
            start = None
    return segments, threshold


def keyframe_range(source, segment, frames: int) -> tuple[np.ndarray, range]:
    """Per-pixel median range of up to 'frames' frames at the centre of a still segment."""
    start, stop = segment
    middle = (start + stop) // 2
    chosen = range(max(start, middle - frames // 2), min(stop, middle - frames // 2 + frames))
    return median_range([source.load(i).range_m for i in chosen], 0.6), chosen
