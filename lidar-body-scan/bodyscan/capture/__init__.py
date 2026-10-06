"""Recording with the Ouster sensor: sensor access, run directory, phases, protocols, motor.

Only bodyscan.capture.sensor and bodyscan.capture.diagnostics import the
Ouster SDK (when a source is opened); only bodyscan.capture.motor imports
pyserial (when the port is opened)."""

from bodyscan.capture.motor import STEPS_PER_TURN, SerialTurntable, split_turn
from bodyscan.capture.protocols import (CountdownConfig, InPlaceCaptureConfig, MotorConfig, SensorConfig,
                                        SteppingConfig, TurntableCaptureConfig, TurntableTimingConfig, TwoPcConfig,
                                        inplace_phases, record_seconds, turntable_phases)
from bodyscan.capture.session import Background, Countdown, Hold, MotorTurn, Phase, Record, Session, Stepping
from bodyscan.capture.writer import FrameWriter, RunDirectory

__all__ = ["STEPS_PER_TURN", "SerialTurntable", "split_turn", "CountdownConfig", "InPlaceCaptureConfig",
           "MotorConfig", "SensorConfig", "SteppingConfig", "TurntableCaptureConfig", "TurntableTimingConfig",
           "TwoPcConfig", "inplace_phases", "record_seconds", "turntable_phases", "Background", "Countdown", "Hold",
           "MotorTurn", "Phase", "Record", "Session", "Stepping", "FrameWriter", "RunDirectory"]
