"""
Capture a person turning on the spot in front of the fixed Ouster, in one run.

    python capture_person.py --source os-122542000054.local --out person_run

Timeline of one execution (all times configurable):

    1. BACKGROUND   --background-seconds (3 s). Stay out of the capture region
                    (at the PC is fine). One low beep at the start.
    2. COUNTDOWN    --delay (15 s). Walk to the mark and take the pose. One beep
                    per second, higher pitch in the last 3 seconds.
    3. CAPTURE      --duration (140 s: two full turns at 20 to 25 degrees per step). Long beep at the start. A short "step"
                    beep every --cue-every seconds (4 s): turn by one small
                    step (20 to 30 degrees), then hold still until the next beep.
                    Two beeps at the end.

Frames are read continuously in all phases (the sensor keeps streaming; not
reading would let the SDK buffer overflow), and written by a separate thread,
so the acquisition loop never waits for the disk.

Output directory
----------------
    lut.npz              per-pixel unit direction and offset (destaggered), so that
                         xyz = range_m * direction + offset. Processing needs no SDK.
    metadata.json        sensor metadata
    background/bg_XXXXX.npz
    frames/frame_XXXXX.npz
                         range (H, W) uint16 [mm], destaggered, 0 = no return;
                         reflectivity (H, W) uint8; frame_id; time [s] since capture
                         start; columns_ok (fraction of valid columns)
    capture.json         phase times, cue times, frame counts, frame_id gaps

Then:  python fuse_person.py person_run --crop-min ... --crop-max ... --out person

Disk: about 0.8 MB per frame, 8 MB/s, about 1.1 GB for the default 140 s.
Write to a local folder outside OneDrive (--out C:\\lidar\\run1) so that the
sync does not compete with the writer; keep Ouster Studio closed.
"""

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from ouster.sdk import core, open_source

VERSION = "2026-09-30a"


# ----------------------------------------------------------------------------
# Sound cues (the person cannot watch the screen)
# ----------------------------------------------------------------------------

def beep(frequency, duration_ms):
    """Non-blocking beep: winsound.Beep blocks, so it runs in its own thread."""
    def play():
        if sys.platform.startswith("win"):
            import winsound
            winsound.Beep(int(frequency), int(duration_ms))
        else:
            print("\a", end="", flush=True)
    threading.Thread(target=play, daemon=True).start()


# ----------------------------------------------------------------------------
# Sensor
# ----------------------------------------------------------------------------

def open_sensor(source_path, auto_udp_dest=False):
    """Live sensor with the stored UDP destination (see ouster_extract.py)."""
    if Path(source_path).exists():
        source = open_source(source_path)                      # a recording, for testing
    elif auto_udp_dest:
        source = open_source(source_path)
    else:
        source = open_source(source_path, no_auto_udp_dest=True)
    info = source.sensor_info[0] if hasattr(source, "sensor_info") else source.metadata[0]
    return source, info


def iterate_scans(source):
    for item in source:
        is_container = isinstance(item, (list, tuple)) or type(item).__name__ == "FrameSet"
        scan = (item[0] if len(item) > 0 else None) if is_container else item
        if scan is not None:
            yield scan


def pixel_lut(info):
    """Destaggered direction and offset such that xyz = range_m * direction + offset.

    The SDK projection is affine in the range for every pixel, so two
    evaluations recover it exactly (checked against core.XYZLut to 1e-16 m).
    """
    height, width = info.format.pixels_per_column, info.format.columns_per_frame
    lut = core.XYZLut(info)
    at_1m = lut(np.full((height, width), 1000, dtype=np.uint32))
    at_2m = lut(np.full((height, width), 2000, dtype=np.uint32))
    direction = core.destagger(info, at_2m - at_1m)
    offset = core.destagger(info, at_1m) - direction
    return direction.astype(np.float32), offset.astype(np.float32)


def read_field(scan, info, name):
    field = getattr(core.ChanField, name, None)
    if field is None or not scan.has_field(field):
        return None
    return core.destagger(info, scan.field(field))


# ----------------------------------------------------------------------------
# Writer thread
# ----------------------------------------------------------------------------

class FrameWriter:
    """Writes frames from a queue so that the acquisition loop never blocks."""

    def __init__(self, compress):
        self.queue = queue.Queue()
        self.save = np.savez_compressed if compress else np.savez
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            path, arrays = item
            self.save(path, **arrays)

    def put(self, path, arrays):
        self.queue.put((path, arrays))

    def backlog(self):
        return self.queue.qsize()

    def close(self):
        self.queue.put(None)
        self.thread.join()


# ----------------------------------------------------------------------------
# Capture
# ----------------------------------------------------------------------------

def frame_arrays(scan, info, capture_start):
    range_mm = read_field(scan, info, "RANGE")
    # uint16 millimetres (up to 65.5 m, beyond the OS0 range) halves the disk
    # traffic; a longer range, if any, is stored as "no return".
    arrays = {
        "range": np.where(range_mm < 65535, range_mm, 0).astype(np.uint16),
        "frame_id": np.int64(scan.frame_id),
        "time": np.float64(time.monotonic() - capture_start),
        "columns_ok": np.float64(np.count_nonzero(np.asarray(scan.status) & 0x1) / info.format.columns_per_frame),
    }
    reflectivity = read_field(scan, info, "REFLECTIVITY")
    if reflectivity is not None:
        arrays["reflectivity"] = np.clip(reflectivity, 0, 255).astype(np.uint8)
    return arrays


def run_capture(source, info, args, out_dir):
    writer = FrameWriter(args.compress)
    (out_dir / "background").mkdir(exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)

    t_background = args.background_seconds
    t_capture_start = t_background + args.delay
    t_end = t_capture_start + args.duration

    log = {"version": VERSION, "background_seconds": args.background_seconds, "delay": args.delay,
           "duration": args.duration, "cue_every": args.cue_every, "cues": [],
           "frame_id_gaps": [], "low_columns_frames": []}
    counts = {"background": 0, "countdown": 0, "frames": 0}
    last_frame_id = None
    last_second = None
    next_cue = args.cue_every
    phase = None

    start = time.monotonic()
    beep(440, 300)
    print(f"[background] {args.background_seconds:g} s: stay out of the capture region")

    try:
        for scan in iterate_scans(source):
            now = time.monotonic() - start

            # frame_id is a 16-bit counter that wraps around
            if last_frame_id is not None and (scan.frame_id - last_frame_id) % 65536 != 1:
                log["frame_id_gaps"].append([int(last_frame_id), int(scan.frame_id)])
            last_frame_id = scan.frame_id

            if now < t_background:
                phase = "background"
                arrays = frame_arrays(scan, info, start + t_capture_start)
                writer.put(out_dir / "background" / f"bg_{counts['background']:05d}.npz", arrays)
                counts["background"] += 1

            elif now < t_capture_start:
                if phase != "countdown":
                    phase = "countdown"
                    print(f"[countdown] {args.delay:g} s: go to the mark and take the pose")
                remaining = int(np.ceil(t_capture_start - now))
                if remaining != last_second:
                    last_second = remaining
                    beep(1000 if remaining <= 3 else 700, 120)
                    print(f"  {remaining}", flush=True)
                counts["countdown"] += 1

            elif now < t_end:
                if phase != "capture":
                    phase = "capture"
                    beep(1200, 700)
                    print(f"[capture] {args.duration:g} s, step cue every {args.cue_every:g} s")
                elapsed = now - t_capture_start
                if args.cue_every > 0 and elapsed >= next_cue:
                    beep(1500, 150)
                    log["cues"].append(round(elapsed, 3))
                    next_cue += args.cue_every

                arrays = frame_arrays(scan, info, start + t_capture_start)
                if arrays["columns_ok"] < 0.99:
                    log["low_columns_frames"].append(counts["frames"])
                writer.put(out_dir / "frames" / f"frame_{counts['frames']:05d}.npz", arrays)
                counts["frames"] += 1
                if counts["frames"] % 50 == 0:
                    print(f"  {elapsed:5.1f} s  frames {counts['frames']}  writer backlog {writer.backlog()}",
                          flush=True)
            else:
                break
    finally:
        beep(1200, 200)
        time.sleep(0.3)
        beep(1200, 200)
        print("[done] writing the remaining frames ...")
        writer.close()

    log.update(counts)
    return log


def main():
    parser = argparse.ArgumentParser(description="Capture a person turning in front of the fixed LiDAR.")
    parser.add_argument("--source", default="os-122542000054.local", help="sensor hostname/IP (or a recording)")
    parser.add_argument("--out", required=True, help="output directory (must not exist or be empty)")
    parser.add_argument("--background-seconds", type=float, default=3.0,
                        help="empty-scene recording at the start [s]; 0 = none")
    parser.add_argument("--delay", type=float, default=15.0, help="time to walk into position [s]")
    parser.add_argument("--duration", type=float, default=140.0, help="capture length [s]")
    parser.add_argument("--cue-every", type=float, default=4.0, help="step beep period [s]; 0 = no cues")
    parser.add_argument("--compress", action="store_true",
                        help="compressed npz: about half the size, but about 0.1 s of CPU per frame, "
                             "so the writer falls behind and the backlog is written after the capture")
    parser.add_argument("--auto-udp-dest", action="store_true", help="see ouster_extract.py")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"{out_dir} is not empty; choose a new --out to avoid mixing runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"capture_person.py version {VERSION}")
    source, info = open_sensor(args.source, args.auto_udp_dest)
    print(f"sensor: {info.prod_line}  fw: {info.fw_rev}  mode: {info.config.lidar_mode}  "
          f"size: {info.format.pixels_per_column}x{info.format.columns_per_frame}")

    text = info.to_json_string() if hasattr(info, "to_json_string") else info.updated_metadata_string()
    (out_dir / "metadata.json").write_text(text)
    direction, offset = pixel_lut(info)
    np.savez(out_dir / "lut.npz", direction=direction, offset=offset)

    try:
        log = run_capture(source, info, args, out_dir)
    finally:
        source.close()

    (out_dir / "capture.json").write_text(json.dumps(log, indent=2))
    rate = log["frames"] / args.duration if args.duration > 0 else 0.0
    print(f"background frames {log['background']}, capture frames {log['frames']} ({rate:.1f} Hz), "
          f"frame_id gaps {len(log['frame_id_gaps'])}, frames with lost columns {len(log['low_columns_frames'])}")
    if log["frame_id_gaps"] or log["low_columns_frames"]:
        print("  some frames were lost or incomplete; fuse_person.py never uses incomplete frames as keyframes")


if __name__ == "__main__":
    main()
