"""A rig: the sensors of a session, where they stand and how their clocks map to the session clock.

World frame: the floor frame of the session, z up, z = 0 on the floor. By
convention the origin is on the floor below the reference sensor (the
Ouster), as in the single-LiDAR pipeline, so a one-sensor rig of an old take
has the same coordinates as its segment-motion output.

rig.json (schema 'bodyscan.rig/1'):

    {"schema": "bodyscan.rig/1", "reference": "ouster",
     "placements": [{"name": "ouster", "serial": "...", "sensor": {...model...},
                     "world_from_sensor": [[...4 x 4...]], "clock": {"offset": 0.0, "rate": 1.0},
                     "notes": ""}, ...],
     "calibration": {...residuals, method, date...}, "created": "2026-10-21T..."}
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bodyscan.sensors.clock import ClockModel
from bodyscan.sensors.models import SensorModel, sensor_from_dict

SCHEMA = "bodyscan.rig/1"


def rotation_matrix(yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """3x3 rotation of a sensor frame (x forward, y left, z up): yaw about the world z axis, then
    pitch (positive tilts the forward axis down), then roll about the forward axis."""
    y, p, r = (math.radians(v) for v in (yaw_deg, pitch_deg, roll_deg))
    yaw = np.array([[math.cos(y), -math.sin(y), 0.0], [math.sin(y), math.cos(y), 0.0], [0.0, 0.0, 1.0]])
    pitch = np.array([[math.cos(p), 0.0, math.sin(p)], [0.0, 1.0, 0.0], [-math.sin(p), 0.0, math.cos(p)]])
    roll = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(r), -math.sin(r)], [0.0, math.sin(r), math.cos(r)]])
    return yaw @ pitch @ roll


def pose(position, yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """4x4 world_from_sensor of a sensor at 'position' (x, y, z) with the given orientation."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix(yaw_deg, pitch_deg, roll_deg)
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return matrix


def looking_at(position, target, level: bool = False) -> np.ndarray:
    """4x4 world_from_sensor of a sensor at 'position' whose forward axis points at 'target'
    (level=True: only turned about the vertical, forward axis horizontal)."""
    position = np.asarray(position, dtype=np.float64)
    toward = np.asarray(target, dtype=np.float64) - position
    yaw = math.degrees(math.atan2(toward[1], toward[0]))
    pitch = 0.0 if level else -math.degrees(math.atan2(toward[2], math.hypot(toward[0], toward[1])))
    return pose(position, yaw, pitch)


@dataclass
class Placement:
    """One sensor of the rig."""
    name: str
    sensor: SensorModel
    world_from_sensor: np.ndarray
    clock: ClockModel = field(default_factory=ClockModel)
    serial: str = ""
    notes: str = ""

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.world_from_sensor, dtype=np.float64)[:3, 3].copy()

    def to_sensor(self, world: np.ndarray) -> np.ndarray:
        """(N, 3) world points in this sensor's frame."""
        m = np.asarray(self.world_from_sensor, dtype=np.float64)
        return (np.asarray(world, dtype=np.float64) - m[:3, 3]) @ m[:3, :3]

    def to_dict(self) -> dict:
        return {"name": self.name, "serial": self.serial, "sensor": self.sensor.to_dict(),
                "world_from_sensor": np.asarray(self.world_from_sensor, dtype=np.float64).tolist(),
                "clock": self.clock.to_dict(), "notes": self.notes}

    @classmethod
    def from_dict(cls, data: dict) -> "Placement":
        return cls(str(data["name"]), sensor_from_dict(data["sensor"]),
                   np.asarray(data["world_from_sensor"], dtype=np.float64),
                   ClockModel.from_dict(data.get("clock", {})), str(data.get("serial", "")),
                   str(data.get("notes", "")))


@dataclass
class Rig:
    """The sensors of a session. 'reference' names the sensor whose clock and floor frame the others
    are calibrated against."""
    placements: list
    reference: str = ""
    calibration: dict = field(default_factory=dict)
    created: str = ""

    def __post_init__(self):
        names = [p.name for p in self.placements]
        if len(set(names)) != len(names):
            raise ValueError(f"sensor names must be unique in a rig: {names}")
        if not self.reference and self.placements:
            self.reference = self.placements[0].name
        if self.placements and self.reference not in names:
            raise ValueError(f"reference sensor {self.reference!r} is not in the rig {names}")

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.placements]

    def placement(self, name: str) -> Placement:
        for p in self.placements:
            if p.name == name:
                return p
        raise KeyError(f"no sensor {name!r} in the rig (sensors: {', '.join(self.names)})")

    def subset(self, names) -> "Rig":
        """The rig with only the named sensors (the reference is kept if present)."""
        chosen = [p for p in self.placements if p.name in set(names)]
        reference = self.reference if self.reference in set(names) else (chosen[0].name if chosen else "")
        return Rig(chosen, reference, dict(self.calibration), self.created)

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "reference": self.reference, "placements": [p.to_dict() for p in self.placements],
                "calibration": self.calibration,
                "created": self.created or datetime.now(timezone.utc).isoformat(timespec="seconds")}

    def save(self, path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict) -> "Rig":
        schema = data.get("schema", SCHEMA)
        if schema != SCHEMA:
            raise ValueError(f"unsupported rig schema {schema!r} (this version reads {SCHEMA})")
        return cls([Placement.from_dict(p) for p in data.get("placements", [])], str(data.get("reference", "")),
                   dict(data.get("calibration", {})), str(data.get("created", "")))

    @classmethod
    def load(cls, path) -> "Rig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
