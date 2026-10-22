"""Fit quality of a tracked body against the measurements, ray by ray, and its map on the body.

Without a reference system the measurements are the only evidence, so the
quality of a fit is judged ray by ray. For every pixel near the person the
posed body (at the pixel's own time) is ray cast along the pixel's ray, and
the pixel is put in one class:

    FITTED       the pixel measured the person and the body surface is within
                 the tolerance of the measured range
    VIOLATION    the body stands where the sensor saw through: the ray went on
                 past the body surface (to the person further back, or to the
                 room) by more than the tolerance
    UNEXPLAINED  the sensor measured the person where the body has no surface
                 within the tolerance: a missing or misplaced part, or a body
                 surface behind the measured one
    SILHOUETTE   a dark pixel (no return, but the light was blocked next to the
                 person) on which the body stands: consistent, without a range
    UNOBSERVED   the body is there, but the pixel tells nothing about it (no
                 return without the dark label, an unknown pixel, an object in
                 front of the body, a room surface touching the body)
    EMPTY        neither the person nor the body: not counted

tolerance = max(k sigma, minimum) with sigma the pixel's range noise.

The map on the body: every pixel whose ray hits the body gives its triangle
and barycentric weights; FITTED, VIOLATION and UNEXPLAINED pixels with a hit
add their absolute range error (capped) to the three corners of the triangle,
weighted by the barycentric weights. Per frame, a vertex gets the weighted
mean error of the pixels that hit around it, or nothing if no pixel did: a
vertex no sensor observed in a frame has no error value in that frame. That
is what makes the heatmap honest: an unobserved region is grey, never green.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# pixel kinds (what the sensor reported)
OTHER, PERSON, DARK, NONE, UNKNOWN_KIND = 0, 1, 2, 3, 4
# pixel classes (what the fit makes of it)
FITTED, VIOLATION, UNEXPLAINED, SILHOUETTE, UNOBSERVED, EMPTY = 0, 1, 2, 3, 4, 5
CLASS_NAMES = ("fitted", "violation", "unexplained", "silhouette", "unobserved", "empty")


@dataclass
class RayBatch:
    """World rays of some pixels of one sensor and what the sensor reported along them."""
    origins: np.ndarray            # (N, 3) [m]
    directions: np.ndarray         # (N, 3) unit
    range: np.ndarray              # (N,) measured distance along the ray [m], NaN without a value
    kind: np.ndarray               # (N,) OTHER, PERSON, DARK, NONE or UNKNOWN_KIND
    sigma: np.ndarray              # (N,) one-sigma range error [m]
    times: np.ndarray              # (N,) measurement time [s]
    sensor: str = ""

    def __len__(self) -> int:
        return len(self.range)

    def subset(self, mask) -> "RayBatch":
        return RayBatch(self.origins[mask], self.directions[mask], self.range[mask], self.kind[mask],
                        self.sigma[mask], self.times[mask], self.sensor)


@dataclass
class ClassifyConfig:
    sigma_factor: float = 3.0          # tolerance = max(sigma_factor x sigma, min_tolerance)
    min_tolerance: float = 0.02        # [m]
    max_error: float = 0.2             # errors above this are capped in the per-vertex map [m]


def classify(batch: RayBatch, model_distance: np.ndarray, config: ClassifyConfig | None = None) -> dict:
    """Pixel classes (N,) and signed range errors e = measured - body (N,; NaN where either is missing)."""
    c = config or ClassifyConfig()
    hit = np.isfinite(model_distance)
    measured = np.isfinite(batch.range)
    error = np.where(hit & measured, batch.range - np.where(hit, model_distance, 0.0), np.nan)
    tolerance = np.maximum(c.sigma_factor * np.asarray(batch.sigma, dtype=np.float64), c.min_tolerance)
    classes = np.full(len(batch), EMPTY, dtype=np.uint8)
    kind = np.asarray(batch.kind)
    person = (kind == PERSON) & measured
    with np.errstate(invalid="ignore"):
        close = np.abs(error) <= tolerance
        beyond = error > tolerance
        before = error < -tolerance
    classes[person & hit & close] = FITTED
    classes[person & hit & beyond] = VIOLATION
    classes[person & hit & before] = UNEXPLAINED
    classes[person & ~hit] = UNEXPLAINED
    other = (kind == OTHER) & measured
    classes[other & hit & beyond] = VIOLATION
    classes[other & hit & ~beyond] = UNOBSERVED              # something in front of, or touching, the body
    dark = kind == DARK
    classes[dark & hit] = SILHOUETTE
    classes[dark & ~hit] = UNEXPLAINED
    rest = ((kind == NONE) | (kind == UNKNOWN_KIND) | ((kind == PERSON) & ~measured)
            | ((kind == OTHER) & ~measured))
    classes[rest & hit] = UNOBSERVED
    return {"classes": classes, "error": error, "hit": hit, "tolerance": tolerance}


def class_counts(classes: np.ndarray) -> dict:
    counts = np.bincount(np.asarray(classes, dtype=np.int64), minlength=len(CLASS_NAMES))
    return {name: int(counts[k]) for k, name in enumerate(CLASS_NAMES)}


class VertexMap:
    """Per-frame, per-vertex range error of the body, accumulated over the pixels of all sensors."""

    def __init__(self, vertex_count: int, frame_count: int):
        self.vertex_count, self.frame_count = int(vertex_count), int(frame_count)
        self.error = np.full((self.frame_count, self.vertex_count), np.nan, dtype=np.float32)
        self._weight = np.zeros((self.frame_count, self.vertex_count), dtype=np.float32)
        self.violation_weight = np.zeros(self.vertex_count, dtype=np.float64)
        self.unexplained_weight = np.zeros(self.vertex_count, dtype=np.float64)
        self.measured_weight = np.zeros(self.vertex_count, dtype=np.float64)
        self.silhouette_frames = np.zeros(self.vertex_count, dtype=np.int64)

    def add(self, frame: int, faces: np.ndarray, triangles: np.ndarray, uv: np.ndarray, classes: np.ndarray,
            error: np.ndarray, max_error: float = 0.2) -> None:
        """Add the pixels of one sensor in one frame (all arrays (N,) or (N, 2) for uv)."""
        hit = triangles >= 0
        corners = np.asarray(faces)[np.where(hit, triangles, 0)]                      # (N, 3)
        weights = np.stack([1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], axis=1)   # (N, 3)
        weights = np.clip(weights, 0.0, 1.0)
        ranged = hit & np.isin(classes, (FITTED, VIOLATION, UNEXPLAINED)) & np.isfinite(error)
        if ranged.any():
            value = np.minimum(np.abs(error[ranged]), max_error)
            v = corners[ranged].ravel()
            w = weights[ranged].ravel()
            total_w = np.bincount(v, weights=w, minlength=self.vertex_count)
            total_wr = np.bincount(v, weights=w * np.repeat(value, 3), minlength=self.vertex_count)
            previous_w = self._weight[frame].astype(np.float64)
            previous_wr = np.nan_to_num(self.error[frame].astype(np.float64)) * previous_w
            weight = previous_w + total_w
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = (previous_wr + total_wr) / weight
            observed = weight > 0
            self.error[frame, observed] = mean[observed]
            self._weight[frame] = weight
            self.measured_weight += total_w
            for cls, store in ((VIOLATION, self.violation_weight), (UNEXPLAINED, self.unexplained_weight)):
                chosen = ranged.copy()
                chosen[ranged] = classes[ranged] == cls
                if chosen.any():
                    store += np.bincount(corners[chosen].ravel(), weights=weights[chosen].ravel(),
                                         minlength=self.vertex_count)
        silhouette = hit & (classes == SILHOUETTE)
        if silhouette.any():
            touched = np.zeros(self.vertex_count, dtype=bool)
            touched[np.unique(corners[silhouette].ravel())] = True
            self.silhouette_frames += touched

    def observed(self) -> np.ndarray:
        """(T, V) True where a vertex had at least one ranged pixel in the frame."""
        return self._weight > 0

    def summary(self) -> dict:
        """Per vertex over all frames: coverage (share of frames observed), median and p95 of the
        per-frame error [m] (NaN if never observed), violation and unexplained rates (share of the
        ranged pixel weight around the vertex)."""
        observed = self.observed()
        coverage = observed.mean(axis=0) if self.frame_count else np.zeros(self.vertex_count)
        median = np.full(self.vertex_count, np.nan)
        p95 = np.full(self.vertex_count, np.nan)
        seen = observed.any(axis=0)
        if seen.any():
            values = self.error[:, seen].astype(np.float64)
            with np.errstate(all="ignore"):
                median[seen] = np.nanmedian(values, axis=0)
                p95[seen] = np.nanpercentile(values, 95, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            violation = np.where(self.measured_weight > 0, self.violation_weight / self.measured_weight, np.nan)
            unexplained = np.where(self.measured_weight > 0, self.unexplained_weight / self.measured_weight, np.nan)
        return {"coverage": coverage, "median": median, "p95": p95, "violation_rate": violation,
                "unexplained_rate": unexplained,
                "silhouette_coverage": self.silhouette_frames / max(self.frame_count, 1)}


def vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(V,) area represented by each vertex: a third of the area of every triangle around it [m^2]."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    area = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    return np.bincount(f.ravel(), weights=np.repeat(area / 3.0, 3), minlength=len(v))


def part_table(summary: dict, labels: np.ndarray, areas: np.ndarray, part_names, min_coverage: float = 0.05) -> dict:
    """Per body part: area-weighted coverage, and over the vertices observed in at least min_coverage
    of the frames: area-weighted median of the vertex medians, of the vertex p95, and the violation
    and unexplained rates. Errors in millimetres."""
    table = {}
    for k, name in enumerate(part_names):
        members = labels == k
        if not members.any():
            continue
        weight = areas[members]
        coverage = float(np.sum(summary["coverage"][members] * weight) / max(weight.sum(), 1e-12))
        seen = members & (summary["coverage"] >= min_coverage) & np.isfinite(summary["median"])
        row = {"coverage": coverage, "vertices_seen": float(seen.sum() / members.sum())}
        if seen.any():
            w = areas[seen]
            row["median_mm"] = 1000.0 * _weighted_median(summary["median"][seen], w)
            row["p95_mm"] = 1000.0 * _weighted_median(summary["p95"][seen], w)
            row["violation_rate"] = float(np.nansum(summary["violation_rate"][seen] * w) / w.sum())
            row["unexplained_rate"] = float(np.nansum(summary["unexplained_rate"][seen] * w) / w.sum())
        table[name] = row
    return table


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
    return float(values[order][min(index, len(values) - 1)])


def point_distances(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(N,) distance of every measured point to the body surface [m]."""
    from bodyscan.dynamic.fitting import closest_points
    if len(points) == 0:
        return np.zeros(0)
    return closest_points(vertices, faces, points)[2]


def point_rates(distances: np.ndarray, sigma: np.ndarray, config: ClassifyConfig | None = None,
                clutter: float = 0.25, outlier: float = 0.05) -> dict:
    """Shares of measured person points within the tolerance (inliers), beyond max(tolerance, outlier)
    (outliers) and beyond 'clutter' (points that belong to nothing on the body)."""
    c = config or ClassifyConfig()
    n = len(distances)
    if n == 0:
        return {"points": 0, "inlier": np.nan, "outlier": np.nan, "clutter": np.nan}
    tolerance = np.maximum(c.sigma_factor * np.asarray(sigma, dtype=np.float64), c.min_tolerance)
    return {"points": int(n), "inlier": float(np.mean(distances <= tolerance)),
            "outlier": float(np.mean(distances > np.maximum(tolerance, outlier))),
            "clutter": float(np.mean(distances > clutter))}
