"""JSON reports with numpy values."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np


def to_builtin(value):
    """json.dumps default= hook: numpy arrays and scalars, dataclasses, paths."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def write_json(path, data) -> None:
    Path(path).write_text(json.dumps(data, indent=2, default=to_builtin), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
