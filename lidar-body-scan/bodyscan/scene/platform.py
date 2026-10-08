"""The turntable platform: its ring, when found in the empty scene, refines the centre."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from bodyscan.config import param
from bodyscan.geometry import fit_circle


@dataclass
class PlatformConfig:
    """Turntable platform: where it is and how it looks in the empty scene."""
    center: tuple[float, float] | None = param(
        None, "platform centre in the floor frame (X Y) [m]; not set: the axis of the object found turning "
              "([select]), refined by the platform ring", effect="imposes the start of the axis fit and picks the "
                                                                  "object near it")
    ring_search: float = param(0.3, "the platform ring is used when its centre is within this of the axis found",
                               unit="m")
    ring_radius: float = param(0.582, "radius of the platform ring (0: no ring search)", unit="m",
                               effect="the ring is an optional refinement: a setup without it works with 0")
    platform_top: float = param(0.03, "height of the platform top above the floor", unit="m",
                                effect="z = 0 of the output is here")


@dataclass
class PlatformEstimate:
    center: np.ndarray
    radius: float
    rms: float
    inliers: int
    plausible: bool


class PlatformDetector(ABC):
    @abstractmethod
    def find(self, world: np.ndarray, start: np.ndarray) -> PlatformEstimate | None:
        """Platform in floor-frame points of the empty scene, searched near 'start'."""


class RingPlatform(PlatformDetector):
    """The low ring of the platform (5 mm to 10 cm above the floor, 0.35 to
    0.80 m from the centre) fitted with a circle; if it is not near the start
    (the sensor was moved), a Hough vote for circles of the ring radius among
    the low points within +-search."""

    def __init__(self, radius: float = 0.582, search: float = 0.8):
        self.radius = radius
        self.search = search

    def ring(self, world: np.ndarray, center: np.ndarray) -> PlatformEstimate | None:
        distance = np.linalg.norm(world[:, :2] - center, axis=1)
        ring = world[(world[:, 2] > 0.005) & (world[:, 2] < 0.10) & (distance > 0.35) & (distance < 0.80)]
        if len(ring) < 100:
            return None
        fitted_center, radius, rms, inliers = fit_circle(ring[:, :2])
        return PlatformEstimate(np.asarray(fitted_center), radius, rms, inliers,
                                abs(radius - self.radius) < 0.03 and rms < 0.025 and inliers >= 300)

    def _recentred(self, world, ring: PlatformEstimate) -> PlatformEstimate:
        """The annulus centred on the fitted centre, fitted again; kept only if still plausible."""
        again = self.ring(world, ring.center)
        return again if again is not None and again.plausible else ring

    def find(self, world: np.ndarray, start: np.ndarray) -> PlatformEstimate | None:
        start = np.asarray(start, dtype=np.float64)
        ring = self.ring(world, start)
        if ring is not None and ring.plausible:
            return self._recentred(world, ring)
        low = world[(world[:, 2] > 0.005) & (world[:, 2] < 0.10)][:, :2]
        low = low[np.all(np.abs(low - start) < self.search + self.radius + 0.1, axis=1)]
        if len(low) < 300:
            return ring
        if len(low) > 20000:
            low = low[np.random.default_rng(0).choice(len(low), 20000, replace=False)]
        angles = np.linspace(0.0, 2 * np.pi, 90, endpoint=False)
        circle = self.radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        votes = (low[:, None, :] + circle[None]).reshape(-1, 2)
        edges = [np.arange(start[c] - self.search, start[c] + self.search + 0.02, 0.02) for c in range(2)]
        histogram, ex, ey = np.histogram2d(votes[:, 0], votes[:, 1], bins=edges)
        for _ in range(3):                                              # best peaks first
            i, j = np.unravel_index(np.argmax(histogram), histogram.shape)
            if histogram[i, j] <= 0:
                break
            guess = np.array([0.5 * (ex[i] + ex[i + 1]), 0.5 * (ey[j] + ey[j + 1])])
            candidate = self.ring(world, guess)
            if candidate is not None and candidate.plausible:
                return self._recentred(world, candidate)
            histogram[max(i - 3, 0):i + 4, max(j - 3, 0):j + 4] = 0
        return ring
