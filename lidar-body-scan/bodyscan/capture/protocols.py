"""Capture protocols: the configuration of each capture and its list of phases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bodyscan.capture.motor import STEPS_PER_TURN, split_turn
from bodyscan.capture.session import Background, Countdown, Hold, MotorTurn, Record, Stepping
from bodyscan.config import param, section
from bodyscan.log import info


@dataclass
class SensorConfig:
    """The sensor (or a recording, to test a protocol without the sensor)."""
    source: str = param("os-122542000054.local", "sensor hostname or IP address, or a .osf/.pcap recording")
    auto_udp_dest: bool = param(False, "let the SDK set the UDP destination of the sensor (the lab PC needs the "
                                       "stored one)")
    compress: bool = param(False, "compressed frame files: half the size, but about 0.1 s of CPU per frame, "
                                  "so the writer falls behind and finishes after the capture")


@dataclass
class CountdownConfig:
    """Empty scene and countdown."""
    background_seconds: float = param(3.0, "empty-scene recording at the start (nobody near the platform or "
                                           "the mark)", unit="s")
    delay: float = param(15.0, "countdown to take position and pose", unit="s")


@dataclass
class TurntableTimingConfig(CountdownConfig):
    """Empty scene, countdown, still periods."""
    hold: float = param(3.0, "standing still before and after the turn (motor mode)", unit="s")


@dataclass
class MotorConfig:
    """Platform driven by this PC over the serial port (omit motor_port to drive it from MATLAB)."""
    motor_port: str | None = param(None, "serial port of the turntable controller, e.g. COM3; not set: the "
                                         "platform is started by girogirotondo_timer.m on the other PC")
    baudrate: int = param(9600, "serial speed")
    motor_boot: float = param(2.5, "wait after opening the port (the controller resets)", unit="s")
    turn_deg: float | None = param(None, "turn of the platform (motor mode: default 360, negative turns the "
                                         "other way; two-PC mode: the turn set in MATLAB, stored for the fusion)",
                                   unit="deg")
    max_steps_per_command: int = param(STEPS_PER_TURN, "longer turns are sent as several commands of at most "
                                                       "this many steps (one lap, known to work)")
    max_rotation_seconds: float = param(300.0, "safety stop: a motor command without 'T done' within this ends "
                                               "the capture (one lap takes about 90 s)", unit="s")


@dataclass
class TwoPcConfig:
    """Platform driven by girogirotondo_timer.m on another PC: length of the recording."""
    duration: float | None = param(None, "record this long; not set: computed from turn_deg and the platform "
                                         "timing below, or until ENTER without turn_deg", unit="s")
    platform_delay: float = param(60.0, "delay_s of girogirotondo_timer.m", unit="s")
    lap_seconds: float = param(89.75, "time of one lap of the platform (measured 2026-10-01: 4.01 deg/s)",
                               unit="s")
    margin: float = param(8.0, "recording kept after the computed end of the rotation", unit="s")
    max_record_seconds: float = param(1800.0, "safety stop of a recording that waits for ENTER", unit="s")


@dataclass
class TurntableCaptureConfig:
    sensor: SensorConfig = section(SensorConfig)
    timing: TurntableTimingConfig = section(TurntableTimingConfig)
    motor: MotorConfig = section(MotorConfig)
    two_pc: TwoPcConfig = section(TwoPcConfig)


@dataclass
class SteppingConfig:
    """The person turns on the spot by steps."""
    duration: float = param(140.0, "capture length (two turns at 20 to 25 deg per step)", unit="s")
    cue_every: float = param(4.0, "a beep every this many seconds: turn by one step (20 to 30 deg), then hold "
                                  "still (0 = no cues)", unit="s")


@dataclass
class InPlaceCaptureConfig:
    sensor: SensorConfig = section(SensorConfig)
    timing: CountdownConfig = section(CountdownConfig)
    stepping: SteppingConfig = section(SteppingConfig)


def record_seconds(config: TurntableCaptureConfig) -> float | None:
    """Two-PC mode: length of the recording, so that it covers the wait for the
    platform timer, the turn and margin seconds after it."""
    two_pc, motor, timing = config.two_pc, config.motor, config.timing
    if two_pc.duration is not None:
        return two_pc.duration
    if motor.turn_deg is None:
        return None
    commands = len(split_turn(motor.turn_deg, motor.max_steps_per_command))
    rotation = abs(motor.turn_deg) / 360.0 * two_pc.lap_seconds + 0.25 * max(commands - 1, 0)
    wait = two_pc.platform_delay - timing.background_seconds - timing.delay
    info(f"recording length {wait + rotation + two_pc.margin:.0f} s: the platform starts within {wait:.0f} s, "
         f"turns {motor.turn_deg:g} deg in about {rotation:.0f} s, then {two_pc.margin:g} s still")
    return wait + rotation + two_pc.margin


def turntable_phases(config: TurntableCaptureConfig, motor=None) -> list:
    """Motor mode: background, countdown, hold, turn, hold. Two-PC mode:
    background, countdown, one recording (the fusion finds the turn in the data)."""
    t = config.timing
    phases = [Background(t.background_seconds, f"{t.background_seconds:g} s: platform empty, stay away from it"),
              Countdown(t.delay, f"{t.delay:g} s: step onto the centre of the platform, pose, stand still")]
    if motor is None:
        seconds = record_seconds(config)
        message = (f"{seconds:g} s: stand still; the platform starts when its own timer ends" if seconds is not None
                   else "stand still; the platform starts when its own timer ends. Press ENTER (or Ctrl+C) a few "
                        "seconds after the platform has STOPPED")
        phases.append(Record(seconds, config.two_pc.max_record_seconds, message))
        return phases
    turn = config.motor.turn_deg if config.motor.turn_deg is not None else 360.0
    commands = split_turn(turn, config.motor.max_steps_per_command)
    if not commands:
        raise SystemExit(f"--turn-deg {turn:g}: no motor step to send")
    phases += [Hold(t.hold, 1, f"{t.hold:g} s still (angle 0)"),
               MotorTurn(motor, commands, config.motor.max_rotation_seconds, f"turning {turn:g} deg"),
               Hold(t.hold, 3, f"{t.hold:g} s still (angle {turn:g})")]
    return phases


def inplace_phases(config: InPlaceCaptureConfig) -> list:
    t, s = config.timing, config.stepping
    return [Background(t.background_seconds, f"{t.background_seconds:g} s: stay out of the capture region"),
            Countdown(t.delay, f"{t.delay:g} s: go to the mark and take the pose"),
            Stepping(s.duration, s.cue_every, f"{s.duration:g} s: at every beep turn by one small step (20 to "
                                              f"30 deg), then hold still")]


def turntable_log(config: TurntableCaptureConfig, motor) -> dict:
    """Fields of capture.json read by the fusion (the commanded turn)."""
    log = {"motor": motor is not None, "steps_per_turn": STEPS_PER_TURN}
    turn = config.motor.turn_deg if (config.motor.turn_deg is not None or motor is None) else 360.0
    if turn is not None:
        log.update({"turn_deg": float(abs(turn)), "turn_sense": 1 if turn >= 0 else -1,
                    "steps": int(round(abs(turn) / 360.0 * STEPS_PER_TURN))})
    return log


def frames_per_second(log: dict) -> float:
    span = log.get("record_seconds") or 0.0
    return log.get("frames", 0) / span if span > 0 else float(np.nan)
