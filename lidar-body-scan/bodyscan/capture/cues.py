"""Sound cues (the person on the platform cannot watch the screen) and the ENTER key."""

from __future__ import annotations

import queue
import sys
import threading
import time


# Sound cues are played one after the other by a single thread (patterns of beeps never overlap),
# every beep lasts at least min_ms, and a short pulse near the upper limit of hearing is played
# first when the audio has been silent for a second: wireless earbuds go to sleep between cues
# and cut the start of the next sound, so a short beep after a pause is lost.
_REAL_AUDIO = sys.platform.startswith("win")
_settings = {"min_ms": 350.0, "wake": True}
_queue: "queue.Queue" = queue.Queue()
_state = {"thread": None, "last": 0.0}
WAKE_HZ, WAKE_MS, IDLE_S = 18000, 150, 1.0


def configure(min_ms: float = 350.0, wake: bool = True) -> None:
    """Shortest beep [ms] and whether a wake pulse precedes a beep after a pause."""
    _settings["min_ms"], _settings["wake"] = float(min_ms), bool(wake)


def _play(frequency: float, duration_ms: float) -> None:
    if _REAL_AUDIO:
        import winsound
        winsound.Beep(int(frequency), int(duration_ms))
    else:
        print("\a", end="", flush=True)


def _worker() -> None:
    while True:
        frequency, duration_ms, gap = _queue.get()
        try:
            if _settings["wake"] and _REAL_AUDIO and time.monotonic() - _state["last"] > IDLE_S:
                _play(WAKE_HZ, WAKE_MS)
            _play(frequency, max(duration_ms, _settings["min_ms"]))
            _state["last"] = time.monotonic()
            if gap and _REAL_AUDIO:
                time.sleep(gap)
        except Exception:                                   # a sound that cannot play never stops a capture
            pass
        finally:
            _queue.task_done()


def beep(frequency: float, duration_ms: float, gap_s: float = 0.0) -> None:
    """Queue one beep (non-blocking; winsound.Beep blocks, so a thread plays the queue)."""
    if _state["thread"] is None:
        _state["thread"] = threading.Thread(target=_worker, daemon=True)
        _state["thread"].start()
    _queue.put((frequency, duration_ms, gap_s))


def beep_pattern(*tones) -> None:
    """Beeps (frequency, duration_ms) one after the other, 0.15 s apart."""
    for frequency, duration_ms in tones:
        beep(frequency, duration_ms, 0.15)


def flush(timeout: float = 3.0) -> None:
    """Wait until the queued beeps have played (at most timeout seconds)."""
    end = time.monotonic() + timeout
    while _queue.unfinished_tasks and time.monotonic() < end:
        time.sleep(0.02)


class EnterListener:
    """'pressed' becomes set when ENTER is pressed (a daemon thread waits on
    stdin). Without a console (stdin redirected or closed, as under some
    launchers) nothing can be pressed: 'available' is False and 'pressed'
    never becomes set (an end of file is not a key press)."""

    def __init__(self):
        self.pressed = threading.Event()
        stream = sys.stdin
        self.available = bool(stream is not None and hasattr(stream, "isatty") and stream.isatty())
        if self.available:
            threading.Thread(target=self._wait, daemon=True).start()

    def _wait(self):
        try:
            line = sys.stdin.readline()
        except Exception:
            return
        if line:
            self.pressed.set()
