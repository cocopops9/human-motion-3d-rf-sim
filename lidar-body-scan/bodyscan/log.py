"""Console output: one line per event, progress lines for long stages.

Everything is printed through info() and Progress so that a caller (tests,
MATLAB, a notebook) can silence the output with set_quiet(True)."""

from __future__ import annotations

import time

_quiet = False


def set_quiet(quiet: bool) -> None:
    global _quiet
    _quiet = bool(quiet)


def info(message: str = "") -> None:
    if not _quiet:
        print(message, flush=True)


def warning(message: str) -> None:
    info(f"  WARNING: {message}")


class Progress:
    """Progress line with the elapsed and remaining time, so that a long stage is visibly alive."""

    def __init__(self, label: str, total: int):
        self.label, self.total, self.start = label, total, time.time()

    def step(self, done: int) -> None:
        elapsed = time.time() - self.start
        remaining = elapsed / done * (self.total - done) if done else 0.0
        info(f"  {self.label}: {done}/{self.total}  elapsed {elapsed:5.0f} s  remaining about {remaining:5.0f} s")

    def maybe(self, done: int, every: int) -> None:
        """Report every 'every' steps and at the end."""
        if done % every == 0 or done == self.total:
            self.step(done)
