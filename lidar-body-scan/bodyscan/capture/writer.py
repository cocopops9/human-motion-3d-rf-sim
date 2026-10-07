"""Run directory of a capture, and the writer thread that keeps the acquisition loop free."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import numpy as np


class FrameWriter:
    """Writes frames from a queue, so that the acquisition loop never waits for
    the disk (not reading the sensor would let the SDK buffer overflow)."""

    def __init__(self, compress: bool = False):
        self.queue = queue.Queue()
        self.save = np.savez_compressed if compress else np.savez
        self.error: Exception | None = None
        self.written = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            if self.error is not None:
                continue                          # after a disk error, drain the queue
            path, arrays = item
            try:
                self.save(path, **arrays)
                self.written += 1
            except Exception as error:            # disk full, folder removed: reported by put() and close()
                self.error = error

    def put(self, path, arrays: dict) -> None:
        if self.error is not None:
            raise SystemExit(f"cannot write the frames: {self.error}")
        self.queue.put((path, arrays))

    def backlog(self) -> int:
        return self.queue.qsize()

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()


class RunDirectory:
    """The layout read by bodyscan.io.NpzRecording:

        lut.npz, metadata.json, capture.json
        background/bg_XXXXX.npz      empty scene
        frames/frame_XXXXX.npz       range, reflectivity, frame_id, time, timestamps, phase, columns_ok
    """

    def __init__(self, path, compress: bool = False):
        self.path = Path(path)
        if self.path.exists() and any(self.path.iterdir()):
            raise SystemExit(f"{self.path} is not empty; choose a new directory to avoid mixing runs")
        (self.path / "background").mkdir(parents=True, exist_ok=True)
        (self.path / "frames").mkdir(exist_ok=True)
        self.writer = FrameWriter(compress)
        self.counts = {"background": 0, "frames": 0}

    def write_sensor(self, source) -> None:
        """Metadata and per-pixel geometry of the sensor (processing then needs no SDK)."""
        (self.path / "metadata.json").write_text(source.metadata_json())
        direction, offset = source.pixel_lut()
        arrays = {"direction": direction, "offset": offset}
        try:
            arrays["pixel_shift"] = source.pixel_shift()                  # pixel times of moving subjects
        except AttributeError:                                            # older SDK metadata, or a test source
            pass
        np.savez(self.path / "lut.npz", **arrays)

    def put(self, kind: str, arrays: dict) -> int:
        """Queue one frame of 'background' or 'frames'; returns its index."""
        index = self.counts[kind]
        name = f"bg_{index:05d}.npz" if kind == "background" else f"frame_{index:05d}.npz"
        self.writer.put(self.path / kind / name, arrays)
        self.counts[kind] += 1
        return index

    def backlog(self) -> int:
        return self.writer.backlog()

    def close(self, log: dict) -> None:
        """Wait for the queued frames and write capture.json (with the number of
        frames actually written, and the disk error if there was one)."""
        self.writer.close()
        log = dict(log)
        log["files_written"] = self.writer.written
        if self.writer.error is not None:
            log["write_error"] = str(self.writer.error)
        (self.path / "capture.json").write_text(json.dumps(log, indent=2))
        if self.writer.error is not None:
            raise SystemExit(f"frames could not be written: {self.writer.error} "
                             f"({self.writer.written} of {sum(self.counts.values())} written)")
