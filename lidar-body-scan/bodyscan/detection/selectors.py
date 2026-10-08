"""Selectors: tests that decide which tracked objects are the ones wanted.

Every pipeline that needs "the object of interest" (the person on the
turntable, the person turning in place, the objects listed by 'bodyscan
detect') finds it the same way: foreground against the empty scene, objects
followed across frames (ObjectFinder), then a chain of selectors. No region
of the room is assumed; a selector looks at the object itself:

    NearSelector      its centre is within a distance of a point given by the user
    HumanSelector     it is shaped like a standing person (the Haar-like cascade)
    RotationSelector  it turns about a vertical axis (optionally a whole lap)

The chain is an AND, evaluated cheapest first, and stops at the first test
that fails (unless every test is asked for, as 'bodyscan detect' without
flags does to report them all). A new criterion is a Selector subclass; a
pipeline adds it to the chain without touching the others.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from bodyscan.detection.human import HumanCascade
from bodyscan.detection.rotation import RotationAnalyzer


@dataclass
class Verdict:
    """Outcome of one selector on one tracked object."""
    passed: bool
    reason: str = ""
    details: dict = field(default_factory=dict)
    axis: np.ndarray | None = None          # a vertical axis (floor frame X Y), if the test measured one


class Selector(ABC):
    """One test of the chain. 'cost' orders the chain (cheap first)."""
    name: str = "selector"
    cost: int = 0

    @abstractmethod
    def evaluate(self, track) -> Verdict:
        """Verdict for a Track (its clusters are in the floor frame)."""


class NearSelector(Selector):
    """The object's centre (median of its points over all frames) is within
    'radius' of a point given by the user: picks one object among several
    without a fixed region of interest."""
    name = "near"
    cost = 0

    def __init__(self, point, radius: float):
        self.point = np.asarray(point, dtype=np.float64)
        self.radius = radius

    def evaluate(self, track) -> Verdict:
        points = np.concatenate([cluster.points for cluster in track.clusters])
        distance = float(np.linalg.norm(np.median(points[:, :2], axis=0) - self.point))
        passed = distance <= self.radius
        reason = "" if passed else f"centre {distance:.2f} m from ({self.point[0]:.2f}, {self.point[1]:.2f})"
        return Verdict(passed, reason, {"distance_m": round(distance, 3)})


class HumanSelector(Selector):
    """The Haar-like person cascade (detection.human) over the frames of the track."""
    name = "human"
    cost = 1

    def __init__(self, cascade: HumanCascade):
        self.cascade = cascade

    def evaluate(self, track) -> Verdict:
        verdict = self.cascade.track(track)
        reason = "" if verdict.human else (
            f"{100 * verdict.pass_fraction:.0f} % of the frames pass"
            + (f" (most rejected by the {max(verdict.rejections, key=verdict.rejections.get)} stage)"
               if verdict.rejections else ""))
        return Verdict(verdict.human, reason, verdict.as_dict())


class RotationSelector(Selector):
    """Turning about a vertical axis (detection.rotation); its axis is the
    centre of the object."""
    name = "rotation"
    cost = 2

    def __init__(self, analyzer: RotationAnalyzer):
        self.analyzer = analyzer

    def evaluate(self, track) -> Verdict:
        result = self.analyzer.analyze(track)
        axis = None if result.axis is None or not result.rotating else np.asarray(result.axis, dtype=np.float64)
        return Verdict(result.rotating, "" if result.rotating else result.reason, result.as_dict(), axis)


class SelectorChain:
    """Selectors applied in order of cost; an object is selected when all pass."""

    def __init__(self, selectors: list[Selector]):
        self.selectors = sorted(selectors, key=lambda s: s.cost)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.selectors]

    def evaluate(self, track, every: bool = False) -> tuple[bool, dict[str, Verdict]]:
        """(selected, verdict per selector name). Stops at the first failure
        unless 'every' is set."""
        verdicts = {}
        selected = True
        for selector in self.selectors:
            verdict = selector.evaluate(track)
            verdicts[selector.name] = verdict
            if not verdict.passed:
                selected = False
                if not every:
                    break
        return selected, verdicts
