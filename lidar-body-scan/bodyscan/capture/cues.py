"""Sound cues (the person on the platform cannot watch the screen) and the ENTER key."""

from __future__ import annotations

import sys
import threading


def beep(frequency: float, duration_ms: float) -> None:
    """Non-blocking beep (winsound.Beep blocks, so it plays in its own thread);
    the terminal bell elsewhere."""
    def play():
        if sys.platform.startswith("win"):
            import winsound
            winsound.Beep(int(frequency), int(duration_ms))
        else:
            print("\a", end="", flush=True)
    threading.Thread(target=play, daemon=True).start()


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
