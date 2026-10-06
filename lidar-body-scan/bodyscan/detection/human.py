"""Is an object a standing person? A cascade of hand-written shape tests.

No learning: every test is a rule on the shape of the human body, organised
like the Haar cascade of Viola and Jones (2001): cheap tests come first and
each one can reject, so most objects are discarded after a few operations;
the last stage compares rectangles of a silhouette image through its integral
image (summed-area table), which is what a Haar feature is.

Every object of every frame becomes a BodyView: its points in the floor frame
(z up, z = 0 on the floor) as seen from the sensor. Its measurements are
computed on demand and cached, so an object rejected by a cheap stage never
pays for the expensive ones (normals, silhouette).

The silhouette (OccupancyImage) is a grid of occupied cells: lateral position
across the line of sight (horizontal, perpendicular to the direction from the
sensor to the object) against height above the floor. The cells grow with the
distance, as the beams spread, so that a continuous surface fills every cell
at any distance. Rectangles are given in units of the object's height (x from
the torso centre line, y up from the floor), so the same rectangles describe a
tall and a short person.

Stages (Stage subclasses; a pipeline can remove, add or replace them):

    SizeStage        top height, bottom near the floor, lateral extent, points  (hard)
    ColumnStage      points at every height from the feet to the head          (hard)
    SilhouetteStage  silhouette area; a solid torso as wide as a person's      (hard)
    SurfaceStage     not a flat panel (a door, a board, a cut-out)             (hard)
    ShapeStage       weighted mean of the features (scored):
                       Haar-like rectangle features of the silhouette: head
                       isolated on top, shoulders wider than the head, compact
                       legs, torso wider than the legs, head above the torso;
                       curved surface: few normals along the dominant directions
                       (furniture is made of flat faces)

A frame passes when every hard stage passes and the ShapeStage score reaches
score_threshold. A tracked object is a person when at least min_pass_fraction
of its frames pass, and at least min_pass_frames of them: the decision uses
all the frames, so a person seen from an awkward angle in a few frames is
still found, and furniture that looks like a person in a few frames is not.

The default thresholds describe adults standing (stature 1.4 to 2.1 m) and
were checked on the lab recordings tt13, tt14, tt15 (people, furniture,
stands, boxes) and on views of a real body mesh ray-cast at 1 to 5 m from
every side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import open3d as o3d

from bodyscan.config import param


@dataclass
class HumanConfig:
    """Shape tests of a standing adult (see docs/parameters.md for their effect)."""
    min_height: float = param(1.2, "lowest top of a person above the floor", unit="m",
                              effect="lower accepts children and people sitting, and more furniture")
    max_height: float = param(2.3, "highest top of a person", unit="m")
    max_bottom: float = param(0.5, "the object must come down to this height at least (feet near the floor)",
                              unit="m", effect="higher accepts people whose legs are hidden (behind a desk)")
    min_width: float = param(0.12, "narrowest lateral extent", unit="m")
    max_width: float = param(2.0, "widest lateral extent (arms spread)", unit="m")
    min_points: int = param(80, "fewest points of an object (after thinning)")
    min_column_fill: float = param(0.85, "fraction of the 5 cm height slices from bottom to top with points")
    max_vertical_gap: float = param(0.30, "largest height range without points", unit="m")
    min_area: float = param(0.20, "smallest silhouette area seen from the sensor", unit="m2",
                            effect="lower accepts children and partly hidden people, and thin stands, "
                                   "lamps and coat racks")
    min_torso_width: float = param(0.18, "narrowest solid torso (median run of occupied cells at 50 to 72 % of "
                                         "the height); a torso seen from the side is about 0.25 m deep", unit="m",
                                   effect="lower accepts poles and tripods")
    max_flat_fraction: float = param(0.65, "hard limit: largest fraction of points whose normal lies within "
                                           "flat_angle of one of the 3 dominant directions (panels, cut-outs)",
                                     effect="lower rejects more furniture and may reject people seen from far away")
    curved_threshold: float = param(0.45, "flat fraction at which the 'curved surface' feature scores 0.5 "
                                          "(people about 0.2 to 0.35, boxes and chairs 0.5 to 0.8)",
                                    effect="lower: furniture scores lower, so do people in stiff clothes")
    flat_angle: float = param(10.0, "half angle of a dominant normal direction", unit="deg")
    score_threshold: float = param(0.6, "a frame passes when the mean score of the rectangle features reaches "
                                        "this", effect="higher: fewer false detections, more missed people")
    min_pass_fraction: float = param(0.5, "a tracked object is a person when this fraction of its frames pass")
    min_pass_frames: int = param(2, "and at least this many frames pass")


def rise(value: float, threshold: float, softness: float) -> float:
    """Logistic step: 0 well below the threshold, 0.5 at it, 1 well above."""
    if not np.isfinite(value):
        return 0.0
    return float(1.0 / (1.0 + np.exp(-(value - threshold) / softness)))


@dataclass(frozen=True)
class ViewSampling:
    """Spacing of the samples on a surface: the larger of the beam spacing at
    that distance and the voxel the points were thinned to."""
    horizontal_rad: float
    vertical_rad: float
    voxel: float

    def spacing(self, distance: float) -> tuple[float, float]:
        """(lateral, vertical) sample spacing [m] on a surface facing the sensor at 'distance'."""
        return max(self.voxel, distance * self.horizontal_rad), max(self.voxel, distance * self.vertical_rad)


class OccupancyImage:
    """Silhouette of an object seen from the sensor, with its integral image.

    Cell (row, col) is occupied when a point falls in it; rows are heights
    above the floor, columns lateral positions across the line of sight,
    measured from the torso centre line. Rectangles are in units of the top
    height: x across (negative and positive sides are equivalent), y up."""

    def __init__(self, lateral: np.ndarray, height: np.ndarray, top: float, cell_lateral: float,
                 cell_height: float):
        self.top = float(top)
        self.cell_lateral, self.cell_height = float(cell_lateral), float(cell_height)
        torso = (height > 0.50 * top) & (height < 0.72 * top)
        self.center = float(np.median(lateral[torso] if torso.sum() >= 5 else lateral))
        u = lateral - self.center
        self.first = int(np.floor(u.min() / cell_lateral))
        cols = np.floor(u / cell_lateral).astype(int) - self.first
        rows = np.clip(np.floor(height / cell_height).astype(int), 0, None)
        self.grid = np.zeros((rows.max() + 1, cols.max() + 1), dtype=bool)
        self.grid[rows, cols] = True
        self.table = np.zeros((self.grid.shape[0] + 1, self.grid.shape[1] + 1), dtype=np.int64)
        self.table[1:, 1:] = self.grid.cumsum(axis=0).cumsum(axis=1)

    @property
    def area(self) -> float:
        """Occupied area [m2]."""
        return float(self.grid.sum()) * self.cell_lateral * self.cell_height

    def _col_range(self, x0: float, x1: float) -> tuple[int, int]:
        start = int(np.floor(x0 * self.top / self.cell_lateral)) - self.first
        stop = int(np.ceil(x1 * self.top / self.cell_lateral)) - self.first
        return start, max(stop, start + 1)

    def _row_range(self, y0: float, y1: float) -> tuple[int, int]:
        start = int(np.floor(y0 * self.top / self.cell_height))
        stop = int(np.ceil(y1 * self.top / self.cell_height))
        return start, max(stop, start + 1)

    def fraction(self, x0: float, x1: float, y0: float, y1: float) -> float:
        """Occupied fraction of the rectangle [x0, x1] x [y0, y1] (four look-ups in the
        integral image; cells outside the grid are empty)."""
        c0, c1 = self._col_range(x0, x1)
        r0, r1 = self._row_range(y0, y1)
        total = (c1 - c0) * (r1 - r0)
        rows, cols = self.grid.shape
        c0, c1 = np.clip([c0, c1], 0, cols)
        r0, r1 = np.clip([r0, r1], 0, rows)
        occupied = self.table[r1, c1] - self.table[r0, c1] - self.table[r1, c0] + self.table[r0, c0]
        return float(occupied) / total

    def _rows(self, y0: float, y1: float) -> np.ndarray:
        r0, r1 = np.clip(self._row_range(y0, y1), 0, self.grid.shape[0])
        rows = self.grid[r0:r1]
        return rows[rows.any(axis=1)]

    def width(self, y0: float, y1: float) -> float:
        """Median over the rows of the band of the occupied extent [m] (NaN if empty)."""
        rows = self._rows(y0, y1)
        if len(rows) == 0:
            return float("nan")
        extents = [np.flatnonzero(r)[-1] - np.flatnonzero(r)[0] + 1 for r in rows]
        return float(np.median(extents)) * self.cell_lateral

    def solid_width(self, y0: float, y1: float) -> float:
        """Median over the rows of the band of the longest run of occupied cells,
        single empty cells bridged [m] (0 if empty)."""
        rows = self._rows(y0, y1)
        if len(rows) == 0:
            return 0.0
        runs = []
        for r in rows:
            cols = np.flatnonzero(r)
            breaks = np.flatnonzero(np.diff(cols) > 2)
            starts = np.concatenate([[0], breaks + 1])
            stops = np.concatenate([breaks, [len(cols) - 1]])
            runs.append(int(np.max(cols[stops] - cols[starts] + 1)))
        return float(np.median(runs)) * self.cell_lateral

    def center_of(self, y0: float, y1: float) -> float:
        """Mean lateral position of the occupied cells of the band, in units of the height."""
        rows = self._rows(y0, y1)
        if len(rows) == 0:
            return float("nan")
        cols = np.nonzero(rows)[1]
        return float(((cols.mean() + 0.5 + self.first) * self.cell_lateral) / self.top)


def flat_fraction(points: np.ndarray, radius: float, angle_deg: float, faces: int = 3, seed: int = 0) -> float:
    """Fraction of the points whose normal lies within angle_deg of one of the
    'faces' dominant normal directions (found greedily). Furniture is made of
    flat faces (high); a body is curved everywhere (low)."""
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    normals = np.asarray(cloud.normals)
    if len(normals) < 10:
        return 1.0
    cosine = np.cos(np.radians(angle_deg))
    rng = np.random.default_rng(seed)
    remaining = np.ones(len(normals), dtype=bool)
    covered = 0
    for _ in range(faces):
        candidates = normals[remaining]
        if len(candidates) < 10:
            break
        probes = candidates[rng.choice(len(candidates), min(200, len(candidates)), replace=False)]
        support = (np.abs(candidates @ probes.T) > cosine).sum(axis=0)
        direction = probes[int(np.argmax(support))]
        hit = remaining & (np.abs(normals @ direction) > cosine)
        covered += int(hit.sum())
        remaining &= ~hit
    return covered / len(normals)


class BodyView:
    """One object in one frame, seen from the sensor. Measurements are cached
    properties, computed the first time a stage asks for them."""

    def __init__(self, points: np.ndarray, sensor_position: np.ndarray, sampling: ViewSampling,
                 flat_angle: float = 10.0):
        self.points = np.asarray(points, dtype=np.float64)
        self.sensor = np.asarray(sensor_position, dtype=np.float64)
        self.sampling = sampling
        self.flat_angle = flat_angle

    @property
    def count(self) -> int:
        return len(self.points)

    @cached_property
    def top(self) -> float:
        return float(np.percentile(self.points[:, 2], 99.5))

    @cached_property
    def bottom(self) -> float:
        return float(np.percentile(self.points[:, 2], 0.5))

    @cached_property
    def center_xy(self) -> np.ndarray:
        return np.median(self.points[:, :2], axis=0)

    @cached_property
    def distance(self) -> float:
        """Horizontal distance from the sensor [m]."""
        return float(max(np.linalg.norm(self.center_xy - self.sensor[:2]), 0.05))

    @cached_property
    def lateral(self) -> np.ndarray:
        """Position of every point across the line of sight [m]."""
        sight = (self.center_xy - self.sensor[:2]) / self.distance
        return (self.points[:, :2] - self.center_xy) @ np.array([-sight[1], sight[0]])

    @cached_property
    def width(self) -> float:
        """Lateral extent (2nd to 98th percentile) [m]."""
        if self.count < 10:
            return 0.0
        return float(np.percentile(self.lateral, 98) - np.percentile(self.lateral, 2))

    @cached_property
    def columns(self) -> tuple[float, float]:
        """(fraction of the height slices from bottom to top with points, largest empty
        height range [m]); slices of 5 cm, or 1.5 beam spacings when the beams are farther apart."""
        step = max(0.05, 1.5 * self.spacing[1])
        edges = np.arange(self.bottom, self.top + step, step)
        if len(edges) < 2:
            return 1.0, 0.0
        occupied = np.histogram(self.points[:, 2], bins=edges)[0] > 0
        longest, run = 0, 0
        for flag in occupied:
            run = 0 if flag else run + 1
            longest = max(longest, run)
        return float(occupied.mean()), step * longest

    @cached_property
    def spacing(self) -> tuple[float, float]:
        return self.sampling.spacing(self.distance)

    @cached_property
    def image(self) -> OccupancyImage:
        lateral_spacing, vertical_spacing = self.spacing
        return OccupancyImage(self.lateral, self.points[:, 2], self.top,
                              max(0.03, 1.5 * lateral_spacing), max(0.03, 1.5 * vertical_spacing))

    @cached_property
    def flat_fraction(self) -> float:
        # Normals from patches of 1.5 sample spacings (at least 6 cm): larger patches
        # average the curvature of the body away and make it look flat.
        radius = max(0.06, 1.5 * max(self.spacing))
        return flat_fraction(self.points, radius, self.flat_angle)

    def measurements(self) -> dict:
        """The measurements computed so far (for reports)."""
        values = {"points": self.count}
        for name in ("top", "bottom", "distance", "width", "flat_fraction"):
            if name in self.__dict__:
                values[name] = round(float(self.__dict__[name]), 4)
        if "columns" in self.__dict__:
            values["column_fill"], values["vertical_gap"] = (round(v, 3) for v in self.columns)
        if "image" in self.__dict__:
            values["area"] = round(self.image.area, 4)
            values["torso_width"] = round(self.image.solid_width(0.50, 0.72), 4)
        return values


@dataclass
class StageResult:
    score: float            # hard stages: 0 or 1; scored stages: [0, 1]
    reason: str = ""        # why it failed (empty when it passed)
    details: dict = field(default_factory=dict)


class Stage:
    """One test of the cascade. Hard stages reject; scored stages contribute a score."""
    name = "stage"
    hard = True

    def __init__(self, config: HumanConfig):
        self.config = config

    def evaluate(self, view: BodyView) -> StageResult:
        raise NotImplementedError


class SizeStage(Stage):
    name = "size"

    def evaluate(self, view):
        c = self.config
        if view.count < c.min_points:
            return StageResult(0.0, f"{view.count} points")
        if not c.min_height <= view.top <= c.max_height:
            return StageResult(0.0, f"top at {view.top:.2f} m")
        if view.bottom > c.max_bottom:
            return StageResult(0.0, f"bottom at {view.bottom:.2f} m (floating)")
        if not c.min_width <= view.width <= c.max_width:
            return StageResult(0.0, f"width {view.width:.2f} m")
        return StageResult(1.0)


class ColumnStage(Stage):
    name = "column"

    def evaluate(self, view):
        fill, gap = view.columns
        if fill < self.config.min_column_fill or gap > self.config.max_vertical_gap:
            return StageResult(0.0, f"heights filled {fill:.2f}, gap {gap:.2f} m")
        return StageResult(1.0)


class SilhouetteStage(Stage):
    name = "silhouette"

    def evaluate(self, view):
        area = view.image.area
        torso = view.image.solid_width(0.50, 0.72)
        if area < self.config.min_area:
            return StageResult(0.0, f"area {area:.2f} m2")
        if torso < self.config.min_torso_width:
            return StageResult(0.0, f"solid torso {torso:.2f} m wide")
        return StageResult(1.0)


class SurfaceStage(Stage):
    name = "surface"

    def evaluate(self, view):
        flat = view.flat_fraction
        if flat > self.config.max_flat_fraction:
            return StageResult(0.0, f"flat faces ({100 * flat:.0f} % of the points)")
        return StageResult(1.0)


class Feature:
    """A score in [0, 1] of one view of an object; 'weight' is its weight in the
    ShapeStage score."""
    name = "feature"
    weight = 1.0

    def score(self, view: BodyView) -> float:
        raise NotImplementedError


class HaarFeature(Feature):
    """A feature computed from rectangles of the silhouette (view.image)."""

    def score(self, view: BodyView) -> float:
        return self.score_image(view.image)

    def score_image(self, image: OccupancyImage) -> float:
        raise NotImplementedError


class RectangleContrast(HaarFeature):
    """The 'on' rectangles are filled (mean occupancy above on_min) and the
    'off' rectangles are empty (largest occupancy below off_max). Each test is
    a logistic step and the score is their product: both must hold.
    Rectangles: (x0, x1, y0, y1) in units of the height."""

    def __init__(self, name, on, off, on_min, off_max, softness=0.06, weight=1.0):
        self.name, self.on, self.off = name, list(on), list(off)
        self.on_min, self.off_max, self.softness, self.weight = on_min, off_max, softness, weight

    def score_image(self, image):
        on = float(np.mean([image.fraction(*r) for r in self.on]))
        off = max((image.fraction(*r) for r in self.off), default=0.0)
        return rise(on, self.on_min, self.softness) * (1.0 - rise(off, self.off_max, self.softness))


class WidthRatio(HaarFeature):
    """Silhouette width of one height band over another, through a logistic step."""

    def __init__(self, name, wide, narrow, threshold, softness, weight=1.0):
        self.name, self.wide, self.narrow = name, wide, narrow
        self.threshold, self.softness, self.weight = threshold, softness, weight

    def score_image(self, image):
        narrow = image.width(*self.narrow)
        if not np.isfinite(narrow) or narrow <= 0:
            return 0.0
        return rise(image.width(*self.wide) / narrow, self.threshold, self.softness)


class Centered(HaarFeature):
    """The band's occupied cells are centred on the torso centre line."""

    def __init__(self, name, band, max_offset, softness, weight=1.0):
        self.name, self.band, self.max_offset = name, band, max_offset
        self.softness, self.weight = softness, weight

    def score_image(self, image):
        offset = image.center_of(*self.band)
        if not np.isfinite(offset):                    # nothing in the band: no head to centre
            return 0.0
        return 1.0 - rise(abs(offset), self.max_offset, self.softness)


class CurvedSurface(Feature):
    """Few normals along the dominant directions: a body is curved everywhere,
    furniture is made of flat faces (view.flat_fraction)."""

    def __init__(self, name, threshold, softness=0.04, weight=1.0):
        self.name, self.threshold, self.softness, self.weight = name, threshold, softness, weight

    def score(self, view):
        return 1.0 - rise(view.flat_fraction, self.threshold, self.softness)


def default_features(config: HumanConfig | None = None) -> list[Feature]:
    """The features of a standing person; rectangles in units of the height
    (x across the line of sight from the torso centre line, y up from the floor)."""
    curved = (config or HumanConfig()).curved_threshold
    return [
        # The head fills a narrow box on top and nothing lies beside it (the strongest
        # cue against furniture, poles and stands: double weight).
        RectangleContrast("head isolated", on=[(-0.06, 0.06, 0.91, 0.98)],
                          off=[(0.13, 0.25, 0.91, 0.98), (-0.25, -0.13, 0.91, 0.98)],
                          on_min=0.45, off_max=0.15, weight=2.0),
        # Shoulders (or the chest, seen from the side) are wider than the head.
        WidthRatio("shoulders wider than head", wide=(0.76, 0.84), narrow=(0.91, 0.98),
                   threshold=1.15, softness=0.10),
        # The legs stand close to the centre line; nothing spreads out at their height.
        RectangleContrast("legs compact", on=[(-0.10, 0.10, 0.08, 0.35)],
                          off=[(0.17, 0.32, 0.08, 0.35), (-0.32, -0.17, 0.08, 0.35)],
                          on_min=0.30, off_max=0.12),
        # The upper body (with the arms) is at least as wide as the legs.
        WidthRatio("torso wider than legs", wide=(0.50, 0.70), narrow=(0.08, 0.35),
                   threshold=0.95, softness=0.10),
        # The head stands above the torso.
        Centered("head over torso", band=(0.88, 1.0), max_offset=0.06, softness=0.02),
        # A curved surface, not the flat faces of furniture.
        CurvedSurface("curved surface", threshold=curved),
    ]


class ShapeStage(Stage):
    """Weighted mean score of the features; fails below score_threshold."""
    name = "shape"
    hard = False

    def __init__(self, config: HumanConfig, features: list[Feature] | None = None):
        super().__init__(config)
        self.features = features if features is not None else default_features(config)

    def evaluate(self, view):
        scores = {f.name: f.score(view) for f in self.features}
        weights = [f.weight for f in self.features]
        score = float(np.average(list(scores.values()), weights=weights))
        weakest = min(scores, key=scores.get)
        reason = "" if score >= self.config.score_threshold else f"weakest: {weakest} ({scores[weakest]:.2f})"
        return StageResult(score, reason, {k: round(v, 3) for k, v in scores.items()})


@dataclass
class FrameResult:
    passed: bool
    score: float
    stage: str              # stage that rejected the frame ('' if it passed)
    reason: str
    view: BodyView
    features: dict


@dataclass
class HumanVerdict:
    human: bool
    pass_fraction: float
    mean_score: float       # mean shape score of the frames that reached the ShapeStage
    frames: int
    height_m: float
    rejections: dict        # stage -> number of frames it rejected
    examples: dict          # stage -> one rejection reason
    measurements: dict      # median of each measurement over the frames

    def as_dict(self) -> dict:
        return {"human": self.human, "pass_fraction": round(self.pass_fraction, 3),
                "mean_score": round(self.mean_score, 3), "frames": self.frames,
                "height_m": round(self.height_m, 3), "rejections": self.rejections,
                "examples": self.examples, "measurements": self.measurements}


class HumanCascade:
    """Stages in order; a frame passes when every hard stage passes and every
    scored stage reaches score_threshold."""

    def __init__(self, config: HumanConfig, sampling: ViewSampling, stages: list[Stage] | None = None):
        self.config = config
        self.sampling = sampling
        self.stages = stages if stages is not None else [
            SizeStage(config), ColumnStage(config), SilhouetteStage(config), SurfaceStage(config),
            ShapeStage(config)]

    def frame(self, points: np.ndarray, sensor_position: np.ndarray) -> FrameResult:
        view = BodyView(points, sensor_position, self.sampling, self.config.flat_angle)
        scores, features = [], {}
        for stage in self.stages:
            result = stage.evaluate(view)
            features.update(result.details)
            if stage.hard and result.score < 1.0:
                return FrameResult(False, 0.0, stage.name, result.reason, view, features)
            if not stage.hard:
                scores.append(result.score)
                if result.score < self.config.score_threshold:
                    return FrameResult(False, result.score, stage.name, result.reason, view, features)
        return FrameResult(True, float(np.mean(scores)) if scores else 1.0, "", "", view, features)

    def track(self, track) -> HumanVerdict:
        results = [self.frame(cluster.points, cluster.sensor) for cluster in track.clusters]
        passed = [r.passed for r in results]
        rejections, examples = {}, {}
        for r in results:
            if not r.passed:
                rejections[r.stage] = rejections.get(r.stage, 0) + 1
                examples.setdefault(r.stage, r.reason)
        shape_scores = [r.score for r in results if r.passed or r.stage == "shape"]
        measured = [r.view.measurements() | r.features for r in results]
        keys = sorted({k for m in measured for k in m})
        measurements = {k: round(float(np.median([m[k] for m in measured if k in m])), 4) for k in keys}
        fraction = float(np.mean(passed)) if passed else 0.0
        human = fraction >= self.config.min_pass_fraction and sum(passed) >= self.config.min_pass_frames
        heights = [r.view.top for r in results if r.passed] or [r.view.top for r in results]
        return HumanVerdict(human, fraction, float(np.mean(shape_scores)) if shape_scores else 0.0,
                            len(results), float(np.median(heights)) if heights else 0.0,
                            rejections, examples, measurements)
