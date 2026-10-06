"""Serial driver of the turntable (the protocol of movement_control.m and girogirotondo.m)."""

from __future__ import annotations

import threading
import time

STEPS_PER_TURN = 800 * 36          # microsteps per motor revolution x gear ratio (girogirotondo.m)


def split_turn(turn_deg: float, max_steps_per_command: int = STEPS_PER_TURN) -> list[int]:
    """Signed step counts of the commands for a turn (negative: the other way):
    rotations longer than max_steps_per_command (one lap, known to work) are
    sent as several commands."""
    if max_steps_per_command <= 0:
        raise ValueError("max_steps_per_command must be positive")
    remaining = int(round(abs(turn_deg) / 360.0 * STEPS_PER_TURN))
    sign = -1 if turn_deg < 0 else 1
    commands = []
    while remaining > 0:
        commands.append(min(remaining, max_steps_per_command))
        remaining -= commands[-1]
    return [sign * steps for steps in commands]


class SerialTurntable:
    """'[T+N]' / '[T-N]' moves N steps; the controller answers a line with
    'T done' when the move is finished. movement_control.m sends '-' for a
    positive step count, so '[T-28800]' is exactly what girogirotondo.m sends
    for one turn. An Arduino resets when the port opens: the constructor
    waits boot_seconds. Only one program can open the port: close MATLAB (or
    'clear motor_ctrl') first."""

    def __init__(self, port: str, baudrate: int = 9600, boot_seconds: float = 2.5, clock=time.monotonic):
        try:
            import serial
        except ImportError as error:
            raise SystemExit("pyserial is missing (install it through MATLAB's Python, or pip install pyserial)") \
                from error
        self.clock = clock
        self.port = serial.Serial(port, baudrate, timeout=0.2)
        time.sleep(boot_seconds)
        self.port.reset_input_buffer()
        self.replies: list = []
        self.sent_time = None
        self.done_time = None
        self._stop = False
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()

    def _read_loop(self):
        while not self._stop:
            line = self.port.readline().decode("ascii", errors="replace").strip()
            if not line:
                continue
            stamp = self.clock()
            self.replies.append([round(stamp, 3), line])
            if "T done" in line and self.sent_time is not None and self.done_time is None:
                self.done_time = stamp

    def move(self, steps: int) -> str:
        sign = "-" if steps > 0 else "+"
        command = f"[T{sign}{abs(int(steps))}]"
        self.done_time = None
        self.sent_time = self.clock()
        self.port.write((command + "\n").encode("ascii"))
        self.port.flush()
        return command

    def close(self) -> None:
        self._stop = True
        self.reader.join(timeout=1.0)
        self.port.close()
