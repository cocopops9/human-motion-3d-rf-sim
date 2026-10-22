"""Candidate placements of the Ouster, the Helios and the Xtion around a capture zone.

The Ouster stands at the origin (lens at ouster_height), the zone centre at
zone_distance along +x. The other sensors are placed on circles around the zone
centre, at an angle measured from the direction of the Ouster (0 deg: beside
the Ouster; 180 deg: opposite it, behind the zone; +-120 deg: a triangle).
The Xtion looks at the zone centre at aim_height; the Helios stands upright
(no tilt), as a spinning LiDAR sees all around.

Distances follow docs/multisensor-plan.md section 2.2: the Helios needs about
3.4 m (standing) to 3.9 m (jumps) for the whole body inside its 30 deg; the
Xtion sees a standing person whole from 1.6 m in portrait and 2.2 m in
landscape, and is noisier with distance squared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bodyscan.sensors.models import helios_16, ouster_os0_128, xtion
from bodyscan.sensors.rig import Placement, Rig, looking_at, pose


@dataclass(frozen=True)
class Layout:
    """One candidate: which sensors, at which angle around the zone (deg) and distance from its centre (m)."""
    name: str
    description: str
    xtion_angle: float | None = None
    xtion_distance: float = 2.2
    xtion_orientation: str = "portrait"
    xtion_height: float = 0.95
    helios_angle: float | None = None
    helios_distance: float = 3.6
    helios_height: float = 0.95


LAYOUTS = {layout.name: layout for layout in (
    Layout("ouster", "the Ouster alone (the 13 and 14 Oct takes)"),
    Layout("ouster+xtion-opposite", "Xtion opposite the Ouster", xtion_angle=180.0),
    Layout("ouster+helios-opposite", "Helios opposite the Ouster", helios_angle=180.0),
    Layout("triangle-120", "all three about 120 deg apart", xtion_angle=120.0, helios_angle=-120.0),
    Layout("opposite-90", "Xtion opposite the Ouster, Helios at the side", xtion_angle=180.0, helios_angle=90.0),
    Layout("triangle-120-landscape", "triangle with the Xtion in landscape at 2.5 m", xtion_angle=120.0,
           xtion_distance=2.5, xtion_orientation="landscape", helios_angle=-120.0),
    Layout("opposite-90-landscape", "opposite and side, Xtion in landscape at 2.5 m", xtion_angle=180.0,
           xtion_distance=2.5, xtion_orientation="landscape", helios_angle=90.0),
)}


def _around(zone_xy: np.ndarray, angle_deg: float, distance: float) -> np.ndarray:
    """Point at 'distance' from the zone centre, 'angle_deg' from the direction of the Ouster (origin)."""
    toward_ouster = math.atan2(-zone_xy[1], -zone_xy[0])
    a = toward_ouster + math.radians(angle_deg)
    return zone_xy + distance * np.array([math.cos(a), math.sin(a)])


def build_rig(layout: Layout, zone_distance: float = 2.3, ouster_height: float = 1.03, aim_height: float = 0.9,
              ouster_mode: tuple = (1024, 20.0), helios_mode: tuple = (1800, 10.0)) -> Rig:
    """The rig of a layout in the floor frame of the Ouster (origin on the floor below it, zone along +x)."""
    zone = np.array([zone_distance, 0.0])
    placements = [Placement("ouster", ouster_os0_128(*ouster_mode), pose((0.0, 0.0, ouster_height)))]
    if layout.helios_angle is not None:
        xy = _around(zone, layout.helios_angle, layout.helios_distance)
        world = looking_at((xy[0], xy[1], layout.helios_height), (zone[0], zone[1], layout.helios_height), level=True)
        placements.append(Placement("helios", helios_16(*helios_mode), world))
    if layout.xtion_angle is not None:
        xy = _around(zone, layout.xtion_angle, layout.xtion_distance)
        world = looking_at((xy[0], xy[1], layout.xtion_height), (zone[0], zone[1], aim_height))
        placements.append(Placement("xtion", xtion(layout.xtion_orientation), world))
    return Rig(placements, "ouster", {"layout": layout.name, "description": layout.description,
                                      "zone_centre": zone.tolist()})


def layout_names() -> list[str]:
    return list(LAYOUTS)
