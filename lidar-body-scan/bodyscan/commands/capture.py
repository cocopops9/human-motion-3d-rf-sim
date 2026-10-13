"""Capture commands (need the Ouster SDK; pyserial for the motor mode)."""

from __future__ import annotations

import time
from pathlib import Path

from bodyscan.capture.protocols import (InPlaceCaptureConfig, TurntableCaptureConfig, frames_per_second,
                                        inplace_phases, turntable_log, turntable_phases)
from bodyscan.commands.base import Command, ConfiguredCommand
from bodyscan.log import info


class CaptureCommand(ConfiguredCommand):
    """Common frame of the captures: output directory, sensor, phases, report.
    A subclass provides config_class and phases()."""

    def add_arguments(self, parser):
        parser.add_argument("out", help="output directory (new or empty), outside OneDrive, e.g. C:\\lidar\\tt17")
        self.add_config_arguments(parser)

    def open_devices(self, config, origin):
        """Devices other than the sensor (the motor); returns them, or None."""
        return None

    def close_devices(self, devices) -> None:
        """Release what open_devices returned."""

    def phases(self, config, devices) -> list:
        raise NotImplementedError

    def extra_log(self, config, devices) -> dict:
        return {}

    def check_source(self, config, source) -> None:
        """Checks of the opened sensor before anything is recorded (warnings only)."""

    def summary(self, log) -> str:
        return (f"background frames {log['background']}, frames {log['frames']}, frame_id gaps "
                f"{len(log['frame_id_gaps'])}, frames with lost columns {len(log['low_columns_frames'])}")

    def run(self, args):
        config = self.prepare(args)
        if config is None:
            return 0
        from bodyscan.capture import cues
        from bodyscan.capture.sensor import OusterSource
        from bodyscan.capture.session import Session
        from bodyscan.capture.writer import RunDirectory
        out = Path(args.out)
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty; choose a new directory to avoid mixing runs")
        timing = getattr(config, "timing", None)
        if timing is not None:
            cues.configure(timing.beep_min_ms, timing.beep_wake, timing.beep_backend)
        origin = time.monotonic()
        devices = self.open_devices(config, origin)
        source = run_directory = None
        log = {}
        try:
            source = OusterSource(config.sensor.source, config.sensor.auto_udp_dest)
            info(f"sensor: {source.describe()}" + ("  (replaying a recording)" if not source.live else ""))
            self.check_source(config, source)
            phases = self.phases(config, devices)              # checks the configuration before any file is made
            run_directory = RunDirectory(out, config.sensor.compress)
            run_directory.write_sensor(source)
            session = Session(source, run_directory, origin)
            session.log.update(self.extra_log(config, devices))
            log = session.run(phases)
        finally:
            if source is not None:
                source.close()
            self.close_devices(devices)
            cues.close()
            if run_directory is not None:
                run_directory.close(log)
        info(self.summary(log))
        if log.get("frame_id_gaps") or log.get("low_columns_frames"):
            info("  some frames were lost or incomplete; the fusion does not use incomplete frames")
        return 0


class CaptureTurntable(CaptureCommand):
    """Capture a person standing still on the turntable during a turn (any angle, several laps).

    Motor mode (this PC drives the platform over the serial port):
        python -m bodyscan capture-turntable C:\\lidar\\tt17 --motor-port COM3 --turn-deg 1800

        background (platform empty) -> countdown (step on, pose) -> hold -> turn
        (one command per lap, each after the 'T done' of the previous) -> hold

    Two-PC mode (girogirotondo_timer.m drives the platform from the other PC):
        start girogirotondo_timer.m first, then within about 30 s:
        python -m bodyscan capture-turntable C:\\lidar\\tt17 --turn-deg 1800

        background -> countdown -> one recording that covers the platform timer,
        the turn and --margin seconds (or until ENTER without --turn-deg)."""
    name = "capture-turntable"
    help = "record a person on the turntable (Ouster sensor, optional motor)"
    config_class = TurntableCaptureConfig

    def open_devices(self, config, origin):
        if not config.motor.motor_port:
            return None
        from bodyscan.capture.motor import SerialTurntable
        info(f"opening {config.motor.motor_port} ({config.motor.motor_boot:g} s for the controller to boot)")
        return SerialTurntable(config.motor.motor_port, config.motor.baudrate, config.motor.motor_boot,
                               clock=lambda: time.monotonic() - origin)

    def close_devices(self, devices):
        if devices is not None:
            devices.close()

    def phases(self, config, devices):
        return turntable_phases(config, devices)

    def extra_log(self, config, devices):
        return turntable_log(config, devices)

    def summary(self, log):
        text = super().summary(log)
        if log.get("motor") and log.get("rotation_start") is not None and log.get("rotation_end") is not None:
            duration = log["rotation_end"] - log["rotation_start"]          # first start to last end
            text += f"; rotation {duration:.1f} s ({log.get('turn_deg', 0) / max(duration, 1e-6):.2f} deg/s)"
        elif log.get("record_seconds"):
            text += f"; recorded {log['record_seconds']:.1f} s ({frames_per_second(log):.1f} Hz)"
        return text


class CaptureInPlace(CaptureCommand):
    """Capture a person turning on the spot by small steps (no turntable).

        python -m bodyscan capture-inplace C:\\lidar\\person1 --duration 140 --cue-every 4

        background (stay out of the region) -> countdown (go to the mark, pose) ->
        capture with a beep every --cue-every seconds: turn by one step, hold still."""
    name = "capture-inplace"
    help = "record a person turning in place by steps (Ouster sensor)"
    config_class = InPlaceCaptureConfig

    def phases(self, config, devices):
        return inplace_phases(config)

    def extra_log(self, config, devices):
        return {"duration": config.stepping.duration, "cue_every": config.stepping.cue_every}


class TestBeeps(Command):
    """Plays the cues of capture-motion, shortened, without the sensor, and
    prints each one as it plays: check that every beep is heard through the
    earbuds or speakers that will be used for the capture.

        python -m bodyscan test-beeps
        python -m bodyscan test-beeps --backend winsound     (the old way, to compare)"""
    name = "test-beeps"
    help = "play the capture cues (no sensor) to check that every beep is heard"

    def add_arguments(self, parser):
        parser.add_argument("--backend", choices=("auto", "stream", "winsound"), default="auto")
        parser.add_argument("--beep-min-ms", type=float, default=350.0)
        parser.add_argument("--no-beep-wake", action="store_true")

    def run(self, args):
        from bodyscan.capture import cues
        used = cues.configure(args.beep_min_ms, not args.no_beep_wake, args.backend)
        info(f"backend: {used}")
        time.sleep(1.0)
        steps = [("leave: 1 long low", [(400, 800)], 1.2)]
        steps += [(f"leave tick {k}", [(400, 80)], 1.0) for k in range(1, 4)]
        steps += [("empty room: 2 low", [(440, 300), (440, 300)], 2.5)]
        steps += [(f"countdown {k}", [(1000 if k <= 3 else 700, 120)], 1.0) for k in range(5, 0, -1)]
        steps += [("start: 1 long (stand still)", [(1200, 700)], 1.5),
                  ("2 short high (move)", [(1500, 150), (1500, 150)], 2.0),
                  ("1 short (change)", [(1500, 150)], 2.0),
                  ("3 medium (back to the mark)", [(1000, 250)] * 3, 2.5),
                  ("1 long (A-pose, stand still)", [(1200, 700)], 2.0),
                  ("end: 2", [(1200, 200), (1200, 200)], 0.0)]
        for number, (label, tones, wait) in enumerate(steps, 1):
            info(f"{number:2d}  {label}")
            cues.beep_pattern(*tones) if len(tones) > 1 else cues.beep(*tones[0])
            time.sleep(wait)
        cues.flush(5.0)
        cues.close()
        info(f"{len(steps)} cues played: tell which numbers you did not hear")
        return 0
