"""Sound cues (the person cannot watch the screen) and the ENTER key.

Two ways to play a beep:

    stream    one audio output stream stays open for the whole capture and the
              beeps are mixed into it (between them it carries an inaudible
              noise floor). Bluetooth earbuds keep the link awake, so no beep
              is cut. Needs the sounddevice package (pip install sounddevice).
    winsound  winsound.Beep (Windows): the audio device is opened for every
              beep. Bluetooth earbuds suspend the link after each sound and
              need a fraction of a second to resume it, which can swallow a
              short beep entirely. A wake pulse before a beep that follows a
              pause partly compensates.

Beeps are queued and played one after the other in both cases (a pattern
never overlaps the next beep), and none is shorter than min_ms."""

from __future__ import annotations

import queue
import sys
import threading
import time

from bodyscan.log import info

_REAL_AUDIO = sys.platform.startswith("win")
_settings = {"min_ms": 350.0, "wake": True, "backend": "auto"}
_state = {"thread": None, "last": 0.0, "stream": None, "resolved": None}
_queue: "queue.Queue" = queue.Queue()
WAKE_HZ, WAKE_MS, IDLE_S = 18000, 150, 1.0
PATTERN_GAP_S = 0.15


# ---------------------------------------------------------------- stream backend

class ToneStream:
    """An output stream that is always open; beeps are appended to a buffer
    that the audio callback drains. Between beeps the callback writes a noise
    floor about 80 dB below full scale (inaudible), so that no driver or
    earbud treats the stream as idle."""

    AMPLITUDE = 0.4
    FLOOR = 1e-4
    FADE_S = 0.005

    def __init__(self, sounddevice_module=None, samplerate: float | None = None):
        import numpy as np
        if sounddevice_module is None:
            import sounddevice as sounddevice_module
        self.np = np
        self.lock = threading.Lock()
        self.pending: list = []                     # arrays of samples still to play
        self.rng = np.random.default_rng(0)
        device = sounddevice_module.query_devices(kind="output")
        self.samplerate = float(samplerate or device["default_samplerate"])
        self.device_name = device["name"]
        self.stream = sounddevice_module.OutputStream(samplerate=self.samplerate, channels=1, dtype="float32",
                                                      callback=self._callback)
        self.stream.start()

    def tone(self, frequency: float, duration_ms: float, gap_s: float = 0.0):
        np = self.np
        n = int(round(self.samplerate * duration_ms / 1000.0))
        t = np.arange(n) / self.samplerate
        samples = self.AMPLITUDE * np.sin(2.0 * np.pi * frequency * t)
        fade = min(int(self.samplerate * self.FADE_S), n // 2)
        if fade > 0:
            ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
            samples[:fade] *= ramp
            samples[n - fade:] *= ramp[::-1]
        silence = np.zeros(int(round(self.samplerate * gap_s)))
        with self.lock:
            self.pending.append(np.concatenate([samples, silence]).astype(np.float32))

    def busy(self) -> bool:
        with self.lock:
            return bool(self.pending)

    def _callback(self, outdata, frames, time_info, status):
        np = self.np
        out = (self.FLOOR * self.rng.standard_normal(frames)).astype(np.float32)
        filled = 0
        with self.lock:
            while self.pending and filled < frames:
                chunk = self.pending[0]
                take = min(len(chunk), frames - filled)
                out[filled:filled + take] += chunk[:take]
                filled += take
                if take == len(chunk):
                    self.pending.pop(0)
                else:
                    self.pending[0] = chunk[take:]
        outdata[:, 0] = out

    def close(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


# ---------------------------------------------------------------- winsound backend

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
            _play(frequency, duration_ms)
            _state["last"] = time.monotonic()
            if gap and _REAL_AUDIO:
                time.sleep(gap)
        except Exception:                                   # a sound that cannot play never stops a capture
            pass
        finally:
            _queue.task_done()


# ---------------------------------------------------------------- public interface

def configure(min_ms: float = 350.0, wake: bool = True, backend: str = "auto") -> str:
    """Shortest beep [ms], wake pulse (winsound only) and backend; opens the
    stream right away, so that earbuds are awake before the first cue.
    Returns the backend in use."""
    _settings["min_ms"], _settings["wake"], _settings["backend"] = float(min_ms), bool(wake), backend
    _state["resolved"] = None
    return _backend()


def _backend() -> str:
    if _state["resolved"] is not None:
        return _state["resolved"]
    wanted = _settings["backend"]
    resolved = "winsound"
    if wanted in ("auto", "stream") and _REAL_AUDIO:
        if _state["stream"] is None:
            try:
                _state["stream"] = ToneStream()
                info(f"beeps: audio stream open on '{_state['stream'].device_name}'")
            except Exception as error:                      # sounddevice missing or no output device
                if wanted == "stream":
                    info(f"beeps: cannot open an audio stream ({error}); falling back to winsound")
                else:
                    info("beeps: winsound (Bluetooth earbuds may cut beeps; pip install sounddevice to fix)")
        if _state["stream"] is not None:
            resolved = "stream"
    _state["resolved"] = resolved
    return resolved


def beep(frequency: float, duration_ms: float, gap_s: float = 0.0) -> None:
    """Queue one beep (non-blocking)."""
    duration_ms = max(float(duration_ms), _settings["min_ms"])
    if _backend() == "stream":
        _state["stream"].tone(frequency, duration_ms, gap_s)
        return
    if _state["thread"] is None:
        _state["thread"] = threading.Thread(target=_worker, daemon=True)
        _state["thread"].start()
    _queue.put((frequency, duration_ms, gap_s))


def beep_pattern(*tones) -> None:
    """Beeps (frequency, duration_ms) one after the other, PATTERN_GAP_S apart."""
    for frequency, duration_ms in tones:
        beep(frequency, duration_ms, PATTERN_GAP_S)


def flush(timeout: float = 3.0) -> None:
    """Wait until the queued beeps have played (at most timeout seconds)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        stream = _state["stream"] if _state["resolved"] == "stream" else None
        if not (_queue.unfinished_tasks or (stream is not None and stream.busy())):
            break
        time.sleep(0.02)
    if _state["resolved"] == "stream":
        time.sleep(0.3)                                     # the device buffer and the Bluetooth latency


def close() -> None:
    """Close the audio stream (end of the capture)."""
    if _state["stream"] is not None:
        _state["stream"].close()
    _state["stream"] = None
    _state["resolved"] = None


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
