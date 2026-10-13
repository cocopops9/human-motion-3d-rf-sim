"""A capture as a sequence of phases.

The sensor streams continuously; every scan is handed to the current phase,
which decides whether the scan is stored (as an empty-scene frame or as a
frame, with its phase label) and when the phase ends. A capture protocol is
just a list of phases, so a new protocol (another motor, another sequence of
cues) is a new list, or a new Phase subclass:

    Leave        time for the operator to leave the scene after starting the command;
                 a low tick every second, nothing stored
    Background   empty scene, stored in background/
    Countdown    time to take position; a beep every second, nothing stored
    Hold         standing still, stored with a phase label (1 before, 3 after the turn)
    Stepping     in-place capture: a cue beep every few seconds to turn by one step
    MotorTurn    the platform turns: commands sent one after the other on 'T done'
    Record       the platform is driven by another PC: record for a given time,
                 or until ENTER is pressed

Phase labels stored in every frame (read by the fusion): 0 unlabelled,
1 still before the turn, 2 turning, 3 still after the turn.

Times: seconds since the session clock origin (wall clock for a live
sensor). When a recording is replayed (testing), the sensor time of the
scans is the clock, so the phases last as long as they would live.
"""

from __future__ import annotations

import time

import numpy as np

from bodyscan import __version__
from bodyscan.capture.cues import EnterListener, beep, beep_pattern, flush
from bodyscan.log import info, warning


class Phase:
    name = "phase"
    store: str | None = "frames"        # "background", "frames", or None (not stored)
    code = 0                            # phase label stored with the frames

    def __init__(self, seconds: float | None = None, message: str = ""):
        self.seconds = seconds
        self.message = message
        self.started = 0.0

    def begin(self, session: "Session", t: float) -> None:
        self.started = t
        if self.message:
            info(f"[{self.name}] {self.message}")

    def finished(self, session: "Session", t: float) -> bool:
        return self.seconds is not None and t - self.started >= self.seconds

    def update(self, session: "Session", t: float) -> None:
        """Called for every scan of the phase (cues, motor)."""

    def end(self, session: "Session", t: float) -> None:
        """Called once when the phase is over."""


class Leave(Phase):
    """Time to walk out of the sensor's view after starting the command: one long low
    beep, then a low tick every second (higher in the last 3 s); nothing stored."""
    name = "leave"
    store = None

    def begin(self, session, t):
        beep(400, 800)
        super().begin(session, t)
        self.last = None

    def update(self, session, t):
        remaining = int(np.ceil(self.seconds - (t - self.started)))
        if remaining != self.last and remaining > 0:
            self.last = remaining
            beep(600 if remaining <= 3 else 400, 80)
            info(f"  leave the scene: {remaining}")


class Background(Phase):
    """Empty scene: two low beeps at the start, silence while it records (stay out of view)."""
    name = "background"
    store = "background"

    def begin(self, session, t):
        beep_pattern((440, 300), (440, 300))
        super().begin(session, t)


class Countdown(Phase):
    name = "countdown"
    store = None

    def begin(self, session, t):
        super().begin(session, t)
        self.last = None

    def update(self, session, t):
        remaining = int(np.ceil(self.seconds - (t - self.started)))
        if remaining != self.last and remaining > 0:
            self.last = remaining
            beep(1000 if remaining <= 3 else 700, 120)
            info(f"  {remaining}")


class Hold(Phase):
    name = "hold"

    def __init__(self, seconds, code, message=""):
        super().__init__(seconds, message)
        self.code = code


class Stepping(Phase):
    """In-place capture: a short beep every cue_every seconds means 'turn by one step, then hold still'."""
    name = "capture"

    def __init__(self, seconds, cue_every, message=""):
        super().__init__(seconds, message)
        self.cue_every = cue_every

    def begin(self, session, t):
        beep(1200, 700)
        super().begin(session, t)
        self.next_cue = self.cue_every
        session.log["cues"] = []

    def update(self, session, t):
        elapsed = t - self.started
        if self.cue_every > 0 and elapsed >= self.next_cue:
            beep(1500, 150)
            session.log["cues"].append(round(elapsed, 3))
            self.next_cue += self.cue_every


class MotorTurn(Phase):
    """Sends the step commands (signed step counts) one after the other, each
    after the 'T done' of the previous one; ends with the last 'T done'. If a
    command gets no 'T done' within max_seconds, the whole capture stops: the
    platform state is unknown, and the frames after it must not be labelled
    'still after the turn'."""
    name = "rotation"
    code = 2

    def __init__(self, motor, commands, max_seconds, message=""):
        super().__init__(None, message)
        if not commands:
            raise ValueError("MotorTurn needs at least one command")
        self.motor, self.pending, self.max_seconds = motor, list(commands), max_seconds
        self.total = len(commands)
        self.done = False
        self.sent_at = 0.0

    def _send(self, session, t):
        command = self.motor.move(self.pending.pop(0))
        self.sent_at = t
        session.log["commands"].append(command)
        session.log["command_times"].append(round(self.motor.sent_time, 3))
        info(f"  sent {command} (command {self.total - len(self.pending)} of {self.total}); waiting for 'T done'")

    def begin(self, session, t):
        beep(1200, 700)
        super().begin(session, t)
        for key in ("commands", "command_times", "done_times", "rotation_starts", "rotation_ends"):
            session.log.setdefault(key, [])
        session.log["rotation_starts"].append(round(t, 3))
        session.log.setdefault("rotation_start", round(t, 3))       # first turn (read by older tools)
        self._send(session, t)

    def update(self, session, t):
        if self.motor.done_time is not None and not self.done:
            session.log["done_times"].append(round(self.motor.done_time, 3))
            if self.pending:
                self._send(session, t)
            else:
                self.done = True

    def finished(self, session, t):
        if not self.done and t - self.sent_at > self.max_seconds:
            warning(f"no 'T done' within {self.max_seconds:g} s of the last command: the capture stops here "
                    "(the platform may still be turning)")
            session.stop_requested = True
            return True
        return self.done

    def end(self, session, t):
        session.log["rotation_ends"].append(round(t, 3))
        session.log["rotation_end"] = round(t, 3)
        info(f"  rotation finished after {t - self.started:.1f} s")


class Record(Phase):
    """Platform driven from another PC: record 'seconds', or until ENTER when
    seconds is None (with a safety stop after max_seconds); ENTER always stops."""
    name = "record"

    def __init__(self, seconds, max_seconds, message=""):
        super().__init__(seconds, message)
        self.max_seconds = max_seconds

    def begin(self, session, t):
        beep(1200, 700)
        super().begin(session, t)
        self.enter = EnterListener() if session.interactive else None
        if self.enter is not None and not self.enter.available and self.seconds is None:
            warning(f"ENTER cannot be read here (no console): the recording stops after {self.max_seconds:g} s; "
                    "give --duration or --turn-deg")
        session.log["record_start"] = round(t, 3)

    def finished(self, session, t):
        elapsed = t - self.started
        if self.enter is not None and self.enter.pressed.is_set():
            info(f"  ENTER pressed after {elapsed:.1f} s of recording")
            return True
        if self.seconds is None and elapsed >= self.max_seconds:
            info(f"  safety stop after {self.max_seconds:g} s")
            return True
        return super().finished(session, t)

    def end(self, session, t):
        session.log["record_seconds"] = round(t - self.started, 3)


class Motion(Record):
    """A take of a moving person, with sound cues (the person cannot watch the screen):

        start          one long beep              stand still in the A-pose on the mark
        after hold     two short high beeps       move
        every cue_every one short beep            change what you are doing (optional)
        end - hold - back_seconds
                       three medium beeps         go back to the mark
        end - hold     one long beep              stand still in the A-pose
        end            two beeps (the session)    done

    Without a duration (until ENTER) only the first two cues are given."""
    name = "record"

    def __init__(self, seconds, max_seconds, hold, back_seconds, cue_every, message=""):
        super().__init__(seconds, max_seconds, message)
        self.hold, self.back_seconds, self.cue_every = hold, back_seconds, cue_every

    def begin(self, session, t):
        super().begin(session, t)                         # long beep: stand still
        self.cues = [(self.hold, "move", ((1500, 150), (1500, 150)))]
        if self.seconds is not None:
            final = self.seconds - self.hold
            back = final - self.back_seconds
            if self.cue_every > 0:
                k = self.hold + self.cue_every
                while k < (back if self.back_seconds > 0 else final) - 1.0:
                    self.cues.append((k, "change", ((1500, 150),)))
                    k += self.cue_every
            if self.back_seconds > 0 and back > self.hold + 1.0:
                self.cues.append((back, "back to the mark", ((900, 200), (900, 200), (900, 200))))
            if final > self.hold:
                self.cues.append((final, "A-pose, stand still", ((1200, 700),)))
        elif self.cue_every > 0:
            self.next_change = self.hold + self.cue_every
        self.cues.sort(key=lambda cue: cue[0])
        session.log["cues"] = []
        info("  beeps: 1 long = stand still (A-pose); 2 short high = move"
             + ("; 1 short = change movement" if self.cue_every > 0 else "")
             + ("; 3 medium = go back to the mark; 1 long = A-pose, stand still" if self.seconds is not None else
                "; press ENTER after 3 s still in the A-pose")
             + "; 2 at the end = done")

    def update(self, session, t):
        elapsed = t - self.started
        while self.cues and elapsed >= self.cues[0][0]:
            at, label, tones = self.cues.pop(0)
            beep_pattern(*tones)
            session.log["cues"].append([round(elapsed, 3), label])
            info(f"  {elapsed:5.1f} s  {label}")
        if self.seconds is None and self.cue_every > 0 and elapsed >= self.next_change:
            beep(1500, 150)
            session.log["cues"].append([round(elapsed, 3), "change"])
            self.next_change += self.cue_every


class Session:
    """Reads the scans of a source and runs the phases over them."""

    def __init__(self, source, run_directory, origin: float | None = None, interactive: bool = True,
                 clock=time.monotonic):
        self.source, self.run_directory = source, run_directory
        self.clock = clock
        self.origin = clock() if origin is None else origin
        self.replay = not source.live
        self.interactive = interactive and source.live
        self.first_scan_time = None
        self.log = {"bodyscan_version": __version__, "source": source.name, "replay": self.replay,
                    "phases": [], "frame_id_gaps": [], "low_columns_frames": []}

    def now(self, scan) -> float:
        if not self.replay:
            return self.clock() - self.origin
        stamp = self.source.scan_time(scan)
        if stamp is None:
            return 0.0 if self.first_scan_time is None else self.last_time
        if self.first_scan_time is None:
            self.first_scan_time = stamp
        return stamp - self.first_scan_time

    def run(self, phases: list) -> dict:
        """Run the phases over the scans. The first phase begins with the first
        scan (connecting to the sensor and booting the motor take seconds that
        must not eat into the empty-scene phase)."""
        phases = list(phases)
        index, phase = 0, None
        last_frame_id = None
        self.last_time = 0.0
        self.stop_requested = False
        try:
            for scan in self.source.scans():
                t = self.now(scan)
                self.last_time = t
                if phase is None and index == 0:
                    phase = phases[0]
                    phase.begin(self, t)
                    self.log["phases"].append([phase.name, round(t, 3)])
                if last_frame_id is not None and (scan.frame_id - last_frame_id) % 65536 != 1:
                    self.log["frame_id_gaps"].append([int(last_frame_id), int(scan.frame_id)])
                last_frame_id = scan.frame_id
                while phase is not None and phase.finished(self, t):
                    phase.end(self, t)
                    index += 1
                    phase = phases[index] if index < len(phases) and not self.stop_requested else None
                    if phase is not None:
                        phase.begin(self, t)
                        self.log["phases"].append([phase.name, round(t, 3)])
                if phase is None:
                    break
                phase.update(self, t)
                if phase.store is None:
                    continue
                arrays = self.source.frame_arrays(scan, t, phase.code)
                number = self.run_directory.put(phase.store, arrays)
                if phase.store == "frames":
                    if arrays["columns_ok"] < 0.99:
                        self.log["low_columns_frames"].append(number)
                    if (number + 1) % 50 == 0:
                        info(f"  {t:6.1f} s  frames {number + 1}  writer backlog {self.run_directory.backlog()}")
            else:
                if phase is not None:
                    warning("the source ended before the last phase")
                    phase.end(self, self.last_time)
        except KeyboardInterrupt:
            info("  Ctrl+C: recording stopped (the frames are kept)")
            if phase is not None:
                phase.end(self, self.last_time)
        finally:
            beep_pattern((1200, 200), (1200, 200))
            flush(3.0)
            info("[done] writing the remaining frames ...")
        self.log.update(self.run_directory.counts)
        return self.log
