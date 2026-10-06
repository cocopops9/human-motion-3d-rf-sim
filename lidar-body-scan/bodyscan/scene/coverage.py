"""What the sensor can see of a person: beam limits towards a point, sample spacing, smallest gap.

Numbers of the Ouster OS0-128 (firmware 3.1, 2048 x 10 Hz) used by default:
column step 360/2048 = 0.176 deg, beam step about 0.71 deg (90 deg over 128
beams), beam diameter 5 mm at the window and divergence 0.35 deg (full width
at half maximum). See docs/hardware.md for the derivation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BEAM_APERTURE_M = 0.005
BEAM_DIVERGENCE_DEG = 0.35


def beam_limits(directions: np.ndarray, world_from_sensor: np.ndarray, center_xy, half_width_deg: float = 10.0):
    """Elevation [deg, floor frame] of the highest and the lowest beam within
    +-half_width_deg of azimuth around the direction of center_xy."""
    world = directions.reshape(-1, 3) @ world_from_sensor[:3, :3].T
    horizontal = np.hypot(world[:, 0], world[:, 1])
    azimuth = np.degrees(np.arctan2(world[:, 1], world[:, 0]))
    elevation = np.degrees(np.arctan2(world[:, 2], horizontal))
    origin = world_from_sensor[:2, 3]
    target = np.degrees(np.arctan2(center_xy[1] - origin[1], center_xy[0] - origin[0]))
    near = np.abs((azimuth - target + 180.0) % 360.0 - 180.0) < half_width_deg
    return float(elevation[near].max()), float(elevation[near].min())


@dataclass
class Sampling:
    """Sampling of a surface facing the sensor at 'distance' [m]."""
    distance: float
    horizontal: float          # spacing of neighbouring columns [m]
    vertical: float            # spacing of neighbouring beams [m]
    footprint: float           # beam diameter [m]

    def smallest_gap(self, along: str = "horizontal") -> float:
        """Smallest gap between two surfaces that is seen as a gap: at least one
        beam must pass through it without touching either side (the footprint)
        and the samples on both sides must be told apart (two sample steps)."""
        step = self.horizontal if along == "horizontal" else self.vertical
        return self.footprint + 2.0 * step


def sampling_at(distance: float, horizontal_rad: float, vertical_rad: float,
                aperture: float = BEAM_APERTURE_M, divergence_deg: float = BEAM_DIVERGENCE_DEG) -> Sampling:
    footprint = aperture + distance * np.radians(divergence_deg)
    return Sampling(distance, distance * horizontal_rad, distance * vertical_rad, footprint)
