"""Ouster sensor or recording, read through the Ouster SDK.

This is the only module that imports the SDK (ouster-sdk), and it does so
when a source is opened: processing never needs it. Tested with ouster-sdk
1.0.1; written for the 0.11 to 0.16 API as well (the names that changed are
looked up both ways).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

RECORDING_SUFFIXES = (".pcap", ".osf", ".bag", ".mcap")


def _sdk():
    try:
        from ouster.sdk import core, open_source
    except ImportError as error:                    # pragma: no cover - depends on the installation
        raise SystemExit("the Ouster SDK is needed to talk to the sensor or read .osf/.pcap files: "
                         "install ouster-sdk (through MATLAB: pyenv / Add-On Explorer, or pip)") from error
    return core, open_source


def is_live(source: str) -> bool:
    """A hostname or IP address, as opposed to a recording on disk."""
    path = Path(source)
    return not path.exists() and path.suffix.lower() not in RECORDING_SUFFIXES


class OusterSource:
    """A live sensor (hostname or IP) or a recording (.osf, .pcap with its
    metadata json): scans, metadata, and the per-pixel geometry.

    For a live sensor the UDP destination stored in the sensor is used as it
    is; with auto_udp_dest the SDK overwrites it with the address it detects
    (on the lab PC an IPv6 link-local address the sensor cannot reach)."""

    def __init__(self, source: str, auto_udp_dest: bool = False, meta: str | None = None):
        core, open_source = _sdk()
        self.core = core
        self.name = source
        self.live = is_live(source)
        if meta is not None:
            self.handle = open_source(source, meta=[meta])
        elif self.live and not auto_udp_dest:
            self.handle = open_source(source, no_auto_udp_dest=True)
        else:
            self.handle = open_source(source)
        self.info = self.handle.sensor_info[0] if hasattr(self.handle, "sensor_info") else self.handle.metadata[0]
        self.height = int(self.info.format.pixels_per_column)
        self.width = int(self.info.format.columns_per_frame)

    def describe(self) -> str:
        info = self.info
        return (f"{info.prod_line}  fw {info.fw_rev}  mode {info.config.lidar_mode}  "
                f"{self.height}x{self.width}")

    def metadata_json(self) -> str:
        info = self.info
        return info.to_json_string() if hasattr(info, "to_json_string") else info.updated_metadata_string()

    def pixel_lut(self) -> tuple[np.ndarray, np.ndarray]:
        """Destaggered direction and offset with xyz = range_m * direction + offset.
        The SDK projection is affine in the range for every pixel, so two
        evaluations recover it exactly (checked against core.XYZLut to 1e-16 m)."""
        core = self.core
        lut = core.XYZLut(self.info)
        at_1m = lut(np.full((self.height, self.width), 1000, dtype=np.uint32))
        at_2m = lut(np.full((self.height, self.width), 2000, dtype=np.uint32))
        direction = core.destagger(self.info, at_2m - at_1m)
        offset = core.destagger(self.info, at_1m) - direction
        return direction.astype(np.float32), offset.astype(np.float32)

    def scans(self):
        """Single scans: depending on the SDK version a source yields one scan,
        a list with one scan per sensor, or a FrameSet; one sensor here."""
        for item in self.handle:
            container = isinstance(item, (list, tuple)) or type(item).__name__ == "FrameSet"
            scan = (item[0] if len(item) > 0 else None) if container else item
            if scan is not None:
                yield scan

    def field(self, scan, name: str):
        """Destaggered (H, W) field, or None if the scan does not have it."""
        field = getattr(self.core.ChanField, name, None)
        if field is None or not scan.has_field(field):
            return None
        return self.core.destagger(self.info, scan.field(field))

    def frame_arrays(self, scan, time_s: float, phase: int) -> dict:
        """The arrays of one frame file. Range as uint16 millimetres (up to
        65.5 m, beyond the OS0 range): half the disk traffic of uint32."""
        range_mm = self.field(scan, "RANGE")
        arrays = {
            "range": np.where(range_mm < 65535, range_mm, 0).astype(np.uint16),
            "frame_id": np.int64(scan.frame_id),
            "time": np.float64(time_s),
            "timestamps": np.asarray(scan.timestamp, dtype=np.uint64),
            "phase": np.int64(phase),
            "columns_ok": np.float64(self.columns_ok(scan)),
        }
        reflectivity = self.field(scan, "REFLECTIVITY")
        if reflectivity is not None:
            arrays["reflectivity"] = np.clip(reflectivity, 0, 255).astype(np.uint8)
        return arrays

    @staticmethod
    def columns_ok(scan) -> float:
        """Fraction of the columns received (lost UDP packets lower it)."""
        status = np.asarray(scan.status)
        return float(np.count_nonzero(status & 0x1)) / max(status.size, 1)

    @staticmethod
    def scan_time(scan) -> float | None:
        """Sensor time of the scan [s] (median of its column timestamps), None if unknown."""
        stamps = np.asarray(scan.timestamp, dtype=np.float64)
        stamps = stamps[stamps > 0]
        return float(np.median(stamps)) * 1e-9 if stamps.size else None

    def close(self) -> None:
        self.handle.close()
