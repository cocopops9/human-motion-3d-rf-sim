"""Objects followed from frame to frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bodyscan.config import param
from bodyscan.detection.segmentation import Cluster


@dataclass
class TrackingConfig:
    """Following objects across frames."""
    track_distance: float = param(0.5, "an object is the same as in an earlier frame if its centre moved less "
                                       "than this (horizontal)", unit="m",
                                  effect="larger follows faster walkers, but may swap two nearby objects")
    min_frames: int = param(3, "objects seen in fewer frames are ignored")


@dataclass
class Track:
    """One object over time: its clusters in time order."""
    id: int
    clusters: list[Cluster] = field(default_factory=list)

    @property
    def times(self) -> np.ndarray:
        return np.array([c.time for c in self.clusters])

    @property
    def centroids(self) -> np.ndarray:
        return np.array([c.centroid for c in self.clusters])

    def last(self) -> Cluster:
        return self.clusters[-1]


class Tracker:
    """Greedy nearest-centroid association between consecutive frames."""

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.tracks: list[Track] = []

    def update(self, clusters: list[Cluster]) -> None:
        open_tracks = list(self.tracks)
        pairs = []
        for t, track in enumerate(open_tracks):
            for k, cluster in enumerate(clusters):
                distance = np.linalg.norm(track.last().centroid[:2] - cluster.centroid[:2])
                if distance <= self.config.track_distance:
                    pairs.append((distance, t, k))
        used_tracks, used_clusters = set(), set()
        for _, t, k in sorted(pairs):
            if t in used_tracks or k in used_clusters:
                continue
            open_tracks[t].clusters.append(clusters[k])
            used_tracks.add(t)
            used_clusters.add(k)
        for k, cluster in enumerate(clusters):
            if k not in used_clusters:
                self.tracks.append(Track(len(self.tracks), [cluster]))

    def result(self) -> list[Track]:
        return [t for t in self.tracks if len(t.clusters) >= self.config.min_frames]
