"""
Capture a person standing still on the motorized turntable during any rotation
(a partial turn, one lap, several laps).

    python capture_turntable.py --out C:\\lidar\\tt1 --motor-port COM3 --turn-deg 1800

The script drives the platform itself, with the same serial command as
girogirotondo.m ("[T-28800]" = 800 microsteps x 36 gear ratio = one turn), so
the recording and the rotation share one clock. Rotations longer than one lap
are sent as several one-lap commands (--max-steps-per-command), each after the
previous "T done". Close MATLAB (or at least
"clear motor_ctrl") first: only one program can open COM3.

Timeline of one execution:

    0. MOTOR        open the serial port; an Arduino resets when the port opens,
                    so the script waits --motor-boot seconds before anything else.
    1. BACKGROUND   --background-seconds (3 s). Platform EMPTY (bin removed),
                    nobody within 1 m of it. One low beep.
    2. COUNTDOWN    --delay (15 s). Step onto the centre of the platform, take the
                    pose, stand still. One beep per second, higher in the last 3.
    3. HOLD         --hold (3 s) standing still before the rotation (angle 0).
    4. ROTATION     long beep; the motor commands are sent; frames are recorded
                    until the controller answers "T done" to the last one.
    5. HOLD         --hold (3 s) standing still after the rotation,
                    then two beeps: step off.

TWO-PC MODE (platform on another PC): omit --motor-port. Start
girogirotondo_timer.m on the platform PC first (it waits --delay_s, 60 s by
default, beeping in the last 10 s), then this script on the LiDAR PC:

    python capture_turntable.py --out C:\lidar\tt1 --turn-deg 1440

    Start it a few seconds AFTER girogirotondo_timer (never before it, at most
    30 s after it).

    1. BACKGROUND   3 s, platform empty.
    2. COUNTDOWN    --delay (15 s): step onto the platform.
    3. RECORD       long beep, stand still; the platform starts when its own
                    timer ends, turns turn_deg, stops. The recording stops BY
                    ITSELF, then two beeps: step off. Its length is computed
                    from --turn-deg and the measured platform speed:
                        --platform-delay (60 s, delay_s in MATLAB)
                        - background - countdown
                        + turn_deg / 360 * --lap-seconds (89.75 s per lap)
                        + 0.25 s per stop between one-lap commands
                        + --margin (8 s)
                    so you stand still for --margin seconds plus the time
                    between starting MATLAB and this script after the stop.
                    --duration N sets the length by hand; without --turn-deg
                    and --duration it records until ENTER (a helper) or Ctrl+C.
                    ENTER or Ctrl+C always stop it early, keeping the frames.
    --turn-deg is optional here: if given it is stored in capture.json and
    fuse_turntable.py compares it with the rotation it measures.
    The recording must start BEFORE the platform does: start this script at
    most (platform delay - background - countdown - 5 s) after the MATLAB
    one, e.g. within 35 s for a 60 s platform delay. fuse_turntable.py finds
    the still parts and the turn in the data.

Output directory (same layout as capture_person.py, plus timing):
    lut.npz, metadata.json
    background/bg_XXXXX.npz
    frames/frame_XXXXX.npz   range (H, W) uint16 [mm], reflectivity uint8, frame_id,
                             time [s] (host clock, since program start),
                             timestamps (W,) uint64 [ns] (sensor clock, per column),
                             phase (1 hold before, 2 rotation, 3 hold after),
                             columns_ok
    capture.json             motor command, time sent, time of "T done", phase
                             boundaries, serial replies, frame counts

Disk: about 0.8 MB per frame (10 Hz). Keep Ouster Studio closed and write
outside OneDrive.
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

from capture_person import FrameWriter, beep, iterate_scans, open_sensor, pixel_lut, read_field

VERSION = "2026-10-01c"

STEPS_PER_TURN = 800 * 36          # microsteps per motor revolution x gear ratio (girogirotondo.m)


# ----------------------------------------------------------------------------
# Motor (serial protocol of movement_control.m)
# ----------------------------------------------------------------------------

def start_enter_listener():
    """Return an Event set when ENTER is pressed (a daemon thread waits on stdin)."""
    pressed = threading.Event()

    def wait_for_enter():
        try:
            sys.stdin.readline()
        except Exception:
            return
        pressed.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()
    return pressed


class Turntable:
    """Minimal serial driver: '[T+N]' / '[T-N]' moves N steps, the controller
    answers a line containing 'T done' when the move is finished.

    movement_control.m sends '-' for a positive step count, so '[T-28800]' is
    exactly what girogirotondo.m sends for one turn.
    """

    def __init__(self, port, baudrate, boot_seconds, clock_start):
        try:
            import serial
        except ImportError:
            sys.exit("pyserial is missing: pip install pyserial")
        self.clock_start = clock_start
        self.port = serial.Serial(port, baudrate, timeout=0.2)
        time.sleep(boot_seconds)                     # Arduino reset on port opening
        self.port.reset_input_buffer()
        self.replies = []
        self.done_time = None
        self.sent_time = None
        self.stop = False
        self.reader = threading.Thread(target=self.read_loop, daemon=True)
        self.reader.start()

    def now(self):
        return time.monotonic() - self.clock_start

    def read_loop(self):
        while not self.stop:
            line = self.port.readline().decode("ascii", errors="replace").strip()
            if not line:
                continue
            stamp = self.now()
            self.replies.append([round(stamp, 3), line])
            if "T done" in line and self.sent_time is not None and self.done_time is None:
                self.done_time = stamp

    def move(self, steps):
        sign = "-" if steps > 0 else "+"
        command = f"[T{sign}{abs(int(steps))}]"
        self.done_time = None
        self.sent_time = self.now()
        self.port.write((command + "\n").encode("ascii"))
        self.port.flush()
        return command

    def close(self):
        self.stop = True
        self.reader.join(timeout=1.0)
        self.port.close()


# ----------------------------------------------------------------------------
# Capture
# ----------------------------------------------------------------------------

def frame_arrays(scan, info, clock_start, phase):
    range_mm = read_field(scan, info, "RANGE")
    arrays = {
        "range": np.where(range_mm < 65535, range_mm, 0).astype(np.uint16),
        "frame_id": np.int64(scan.frame_id),
        "time": np.float64(time.monotonic() - clock_start),
        "timestamps": np.asarray(scan.timestamp, dtype=np.uint64),
        "phase": np.int64(phase),
        "columns_ok": np.float64(np.count_nonzero(np.asarray(scan.status) & 0x1)
                                 / info.format.columns_per_frame),
    }
    reflectivity = read_field(scan, info, "REFLECTIVITY")
    if reflectivity is not None:
        arrays["reflectivity"] = np.clip(reflectivity, 0, 255).astype(np.uint8)
    return arrays


def run_capture(source, info, motor, args, out_dir, clock_start):
    writer = FrameWriter(False)
    (out_dir / "background").mkdir(exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)

    log = {"version": VERSION, "steps_per_turn": STEPS_PER_TURN, "motor": motor is not None,
           "frame_id_gaps": [], "low_columns_frames": []}
    commands = []
    if args.turn_deg is not None:
        # In two-PC mode this is only a hint for fuse_turntable.py (it checks it against the data).
        steps = int(round(args.turn_deg / 360.0 * STEPS_PER_TURN))
        log.update({"turn_deg": args.turn_deg, "steps": steps})
        remaining = steps
        while remaining > 0:
            commands.append(min(remaining, args.max_steps_per_command))
            remaining -= commands[-1]
        log["commands"], log["command_times"], log["done_times"] = [], [], []
    counts = {"background": 0, "frames": 0}

    def now():
        return time.monotonic() - clock_start

    t_background_end = now() + args.background_seconds
    t_countdown_end = t_background_end + args.delay
    t_hold_end = t_countdown_end + args.hold
    phase, last_second, last_frame_id = "background", None, None
    record_start = None
    rotation_start = rotation_end = None
    end_time = None

    beep(440, 300)
    print(f"[background] {args.background_seconds:g} s: platform empty, stay away from it")
    try:
        for scan in iterate_scans(source):
            t = now()
            if last_frame_id is not None and (scan.frame_id - last_frame_id) % 65536 != 1:
                log["frame_id_gaps"].append([int(last_frame_id), int(scan.frame_id)])
            last_frame_id = scan.frame_id

            if phase == "background":
                if t < t_background_end:
                    writer.put(out_dir / "background" / f"bg_{counts['background']:05d}.npz",
                               frame_arrays(scan, info, clock_start, 0))
                    counts["background"] += 1
                    continue
                phase = "countdown"
                print(f"[countdown] {args.delay:g} s: step onto the centre of the platform, pose, stand still")

            if phase == "countdown":
                if t < t_countdown_end:
                    remaining = int(np.ceil(t_countdown_end - t))
                    if remaining != last_second:
                        last_second = remaining
                        beep(1000 if remaining <= 3 else 700, 120)
                        print(f"  {remaining}", flush=True)
                    continue
                log["hold_before_start"] = round(t, 3)
                if motor is None:
                    # Platform driven from the other PC: record one block
                    # (standing still, the turn, standing still); the
                    # fusion finds the turn in the data.
                    phase = "record"
                    record_start = t
                    limit = args.duration if args.duration is not None else args.max_record_seconds
                    record_end = t + limit
                    enter = start_enter_listener()
                    beep(1200, 700)
                    if args.duration is not None:
                        print(f"[record] {args.duration:g} s: stand still; the platform starts when its own timer ends")
                    else:
                        print("[record] stand still; the platform starts when its own timer ends.\n"
                              "         Press ENTER (or Ctrl+C) a few seconds after the platform has STOPPED.")
                else:
                    phase = "hold_before"
                    print(f"[hold] {args.hold:g} s still (angle 0)")

            if phase == "record":
                if t >= record_end:
                    if args.duration is None:
                        print(f"  safety stop after --max-record-seconds {args.max_record_seconds:g} s")
                    break
                if enter.is_set():
                    print(f"  ENTER pressed after {t - record_start:.1f} s of recording")
                    break

            if phase == "hold_before" and t >= t_hold_end:
                phase = "rotation"
                rotation_start = t
                beep(1200, 700)
                pending = list(commands)
                log["commands"].append(motor.move(pending.pop(0)))
                log["command_times"].append(round(motor.sent_time, 3))
                print(f"[rotation] sent {log['commands'][-1]} (command 1 of {len(commands)}); waiting for 'T done'")

            if phase == "rotation":
                finished = False
                if motor.done_time is not None:
                    log["done_times"].append(round(motor.done_time, 3))
                    if pending:
                        log["commands"].append(motor.move(pending.pop(0)))
                        log["command_times"].append(round(motor.sent_time, 3))
                        print(f"  sent {log['commands'][-1]} (command {len(log['commands'])} of {len(commands)})")
                    else:
                        finished = True
                if t - rotation_start > args.max_rotation_seconds:
                    print("  WARNING: no 'T done' within --max-rotation-seconds; stopping the recording "
                          "(the platform may still be turning)")
                    finished = True
                if finished:
                    phase = "hold_after"
                    rotation_end = t
                    end_time = t + args.hold
                    print(f"[hold] rotation finished after {rotation_end - rotation_start:.1f} s; "
                          f"{args.hold:g} s still (angle {args.turn_deg:g})")

            if phase == "hold_after" and t >= end_time:
                break

            code = {"record": 0, "hold_before": 1, "rotation": 2, "hold_after": 3}[phase]
            arrays = frame_arrays(scan, info, clock_start, code)
            if arrays["columns_ok"] < 0.99:
                log["low_columns_frames"].append(counts["frames"])
            writer.put(out_dir / "frames" / f"frame_{counts['frames']:05d}.npz", arrays)
            counts["frames"] += 1
            if counts["frames"] % 50 == 0:
                print(f"  {t - log['hold_before_start']:5.1f} s  frames {counts['frames']}  "
                      f"writer backlog {writer.backlog()}", flush=True)
    except KeyboardInterrupt:
        print("  Ctrl+C: recording stopped")
    finally:
        beep(1200, 200)
        time.sleep(0.3)
        beep(1200, 200)
        print("[done] step off the platform; writing the remaining frames ...")
        writer.close()

    log.update(counts)
    if record_start is not None:
        log["record_start"] = round(record_start, 3)
        log["record_seconds"] = round(now() - record_start, 3)
    log["rotation_start"] = round(rotation_start, 3) if rotation_start is not None else None
    log["rotation_end"] = round(rotation_end, 3) if rotation_end is not None else None
    if motor is not None:
        log["serial_replies"] = motor.replies
    return log


def main():
    parser = argparse.ArgumentParser(description="Capture a person on the turntable during a rotation.")
    parser.add_argument("--source", default="os-122542000054.local", help="sensor hostname/IP")
    parser.add_argument("--out", required=True, help="output directory (new or empty)")
    parser.add_argument("--motor-port", default=None, help="serial port of the turntable, e.g. COM3; "
                                                           "omit to start the platform from MATLAB")
    parser.add_argument("--baudrate", type=int, default=9600)
    parser.add_argument("--motor-boot", type=float, default=2.5, help="wait after opening the port [s]")
    parser.add_argument("--turn-deg", type=float, default=None,
                        help="rotation [deg], any value (50, 360, 1800, ...). With --motor-port: the rotation "
                             "to command (default 360). Without: optional, the rotation set in "
                             "girogirotondo_timer.m, stored as a hint for fuse_turntable.py")
    parser.add_argument("--max-steps-per-command", type=int, default=STEPS_PER_TURN,
                        help="with --motor-port: longer rotations are sent as several commands of at most "
                             "this many steps (one lap, known to work)")
    parser.add_argument("--background-seconds", type=float, default=3.0)
    parser.add_argument("--delay", type=float, default=15.0, help="time to step onto the platform [s]")
    parser.add_argument("--hold", type=float, default=3.0, help="still time before and after the turn [s]")
    parser.add_argument("--duration", type=float, default=None,
                        help="without --motor-port: stop the recording by itself after this many seconds "
                             "(must cover the wait for the platform timer, the whole turn and a few seconds "
                             "after it). Default: record until ENTER (or Ctrl+C) is pressed")
    parser.add_argument("--platform-delay", type=float, default=60.0,
                        help="without --motor-port: delay_s of girogirotondo_timer.m [s]")
    parser.add_argument("--lap-seconds", type=float, default=89.75,
                        help="time of one lap of the platform [s] (measured 2026-10-01: 89.75 s, 4.01 deg/s)")
    parser.add_argument("--margin", type=float, default=8.0,
                        help="recording kept after the computed end of the rotation [s]")
    parser.add_argument("--max-record-seconds", type=float, default=1800.0,
                        help="without --motor-port and --duration: safety stop of the recording [s]")
    parser.add_argument("--max-rotation-seconds", type=float, default=900.0,
                        help="safety stop of the recording if 'T done' never arrives [s]")
    parser.add_argument("--auto-udp-dest", action="store_true", help="see ouster_extract.py")
    args = parser.parse_args()

    if args.motor_port and args.turn_deg is None:
        args.turn_deg = 360.0
    if not args.motor_port and args.duration is None and args.turn_deg is not None:
        commands = int(np.ceil(args.turn_deg / 360.0 * STEPS_PER_TURN / args.max_steps_per_command - 1e-9))
        rotation = args.turn_deg / 360.0 * args.lap_seconds + 0.25 * max(commands - 1, 0)
        wait = args.platform_delay - args.background_seconds - args.delay
        args.duration = wait + rotation + args.margin
        print(f"recording length {args.duration:.0f} s: platform starts within {wait:.0f} s, turns "
              f"{args.turn_deg:g} deg in about {rotation:.0f} s, then {args.margin:g} s still")
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"{out_dir} is not empty; choose a new --out to avoid mixing runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"capture_turntable.py version {VERSION}")

    clock_start = time.monotonic()
    motor = None
    if args.motor_port:
        print(f"opening {args.motor_port} ({args.motor_boot:g} s for the controller to boot)")
        motor = Turntable(args.motor_port, args.baudrate, args.motor_boot, clock_start)

    source, info = open_sensor(args.source, args.auto_udp_dest)
    print(f"sensor: {info.prod_line}  fw: {info.fw_rev}  mode: {info.config.lidar_mode}")
    text = info.to_json_string() if hasattr(info, "to_json_string") else info.updated_metadata_string()
    (out_dir / "metadata.json").write_text(text)
    direction, offset = pixel_lut(info)
    np.savez(out_dir / "lut.npz", direction=direction, offset=offset)

    try:
        log = run_capture(source, info, motor, args, out_dir, clock_start)
    finally:
        source.close()
        if motor is not None:
            motor.close()

    (out_dir / "capture.json").write_text(json.dumps(log, indent=2))
    if motor is None:
        print(f"background frames {log['background']}, frames {log['frames']} "
              f"({log['frames'] / max(log.get('record_seconds', 0), 1e-6):.1f} Hz), "
              f"recorded {log.get('record_seconds', 0):.1f} s, frame_id gaps {len(log['frame_id_gaps'])}, "
              f"frames with lost columns {len(log['low_columns_frames'])}")
        return
    duration = (log["rotation_end"] or 0) - (log["rotation_start"] or 0)
    print(f"background frames {log['background']}, frames {log['frames']}, rotation {duration:.1f} s "
          f"(average {args.turn_deg / duration if duration > 0 else 0:.2f} deg/s), "
          f"frame_id gaps {len(log['frame_id_gaps'])}, frames with lost columns {len(log['low_columns_frames'])}")


if __name__ == "__main__":
    main()
