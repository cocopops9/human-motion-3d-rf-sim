"""Capture protocols and the phase session, with a fake sensor (no SDK, no hardware)."""

import json
import unittest

import numpy as np

from bodyscan.capture.motor import STEPS_PER_TURN, split_turn
from bodyscan.capture.protocols import (InPlaceCaptureConfig, TurntableCaptureConfig, inplace_phases, record_seconds,
                                        turntable_phases)
from bodyscan.capture.session import Session
from bodyscan.capture.writer import RunDirectory
from bodyscan.io import NpzRecording
from tests.helpers import TemporaryFolder


class FakeScan:
    def __init__(self, frame_id, time_s, width):
        self.frame_id = frame_id
        self.timestamp = np.full(width, int(round((100.0 + time_s) * 1e9)), dtype=np.uint64)
        self.status = np.ones(width, dtype=np.uint32)


class FakeSource:
    """Replays 'frames' scans at 'rate' Hz; looks like bodyscan.capture.sensor.OusterSource."""
    live = False
    name = "fake"

    def __init__(self, frames, rate=10.0, height=4, width=32, skip=()):
        self.frames, self.rate, self.height, self.width, self.skip = frames, rate, height, width, set(skip)
        self.clock = 0.0

    def scans(self):
        for k in range(self.frames):
            if k in self.skip:                      # lost frame: a gap in frame_id
                continue
            self.clock = k / self.rate
            yield FakeScan(k, self.clock, self.width)

    @staticmethod
    def scan_time(scan):
        return float(np.median(scan.timestamp)) * 1e-9

    def frame_arrays(self, scan, time_s, phase):
        return {"range": np.full((self.height, self.width), 1500, dtype=np.uint16), "frame_id": np.int64(scan.frame_id),
                "time": np.float64(time_s), "timestamps": scan.timestamp, "phase": np.int64(phase),
                "columns_ok": np.float64(1.0)}

    def metadata_json(self):
        return "{}"

    def pixel_lut(self):
        directions = np.zeros((self.height, self.width, 3), dtype=np.float32)
        directions[..., 0] = 1.0
        return directions, np.zeros_like(directions)


class FakeMotor:
    """Answers 'T done' after steps / speed seconds of the fake clock."""

    def __init__(self, source, steps_per_second=STEPS_PER_TURN / 2.0):
        self.source, self.speed = source, steps_per_second
        self.sent_time, self.duration, self.commands = None, 0.0, []

    def move(self, steps):
        self.sent_time, self.duration = self.source.clock, steps / self.speed
        self.commands.append(steps)
        return f"[T-{steps}]"

    @property
    def done_time(self):
        if self.sent_time is not None and self.source.clock >= self.sent_time + self.duration:
            return self.sent_time + self.duration
        return None


class SplitTest(unittest.TestCase):
    def test_turns_are_split_into_laps(self):
        self.assertEqual(split_turn(1800), [STEPS_PER_TURN] * 5)
        self.assertEqual(split_turn(400), [STEPS_PER_TURN, 3200])
        self.assertEqual(sum(split_turn(50)), round(50 / 360 * STEPS_PER_TURN))

    def test_two_pc_recording_length(self):
        config = TurntableCaptureConfig()
        config.motor.turn_deg = 720.0
        expected = (60.0 - 3.0 - 15.0) + 2 * 89.75 + 0.25 + 8.0
        self.assertAlmostEqual(record_seconds(config), expected, places=6)
        config.two_pc.duration = 30.0
        self.assertEqual(record_seconds(config), 30.0)


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryFolder()

    def tearDown(self):
        self.folder.cleanup()

    def run_session(self, source, phases, name="run"):
        run_directory = RunDirectory(self.folder.path / name)
        run_directory.write_sensor(source)
        log = Session(source, run_directory, interactive=False).run(phases)
        run_directory.close(log)
        return log, self.folder.path / name

    def test_inplace_protocol(self):
        config = InPlaceCaptureConfig()
        config.timing.leave_seconds = 0.0
        config.timing.background_seconds, config.timing.delay = 0.5, 0.3
        config.stepping.duration, config.stepping.cue_every = 2.0, 0.5
        source = FakeSource(frames=40, skip=[15])          # one frame lost during the capture
        log, path = self.run_session(source, inplace_phases(config))
        self.assertEqual([p[0] for p in log["phases"]], ["background", "countdown", "capture"])
        self.assertEqual(log["background"], 5)
        self.assertEqual(log["frames"], 19)
        self.assertEqual(len(log["cues"]), 3)
        self.assertEqual(log["frame_id_gaps"], [[14, 16]])
        recording = NpzRecording(path)
        self.assertEqual((len(recording), recording.background_count()), (19, 5))
        self.assertEqual(json.loads((path / "capture.json").read_text())["frames"], 19)

    def test_turntable_motor_protocol(self):
        config = TurntableCaptureConfig()
        config.timing.background_seconds, config.timing.delay, config.timing.hold = 0.3, 0.2, 0.4
        config.motor.turn_deg = 720.0
        source = FakeSource(frames=200)
        motor = FakeMotor(source)
        log, path = self.run_session(source, turntable_phases(config, motor))
        self.assertEqual(motor.commands, [STEPS_PER_TURN, STEPS_PER_TURN])
        self.assertEqual([p[0] for p in log["phases"]], ["background", "countdown", "hold", "rotation", "hold"])
        recording = NpzRecording(path)
        phases = [recording.load(k).phase for k in range(len(recording))]
        self.assertEqual(sorted(set(phases)), [1, 2, 3])
        self.assertEqual(phases, sorted(phases))                       # 1 before 2 before 3
        self.assertAlmostEqual(log["rotation_end"] - log["rotation_start"], 4.0, delta=0.25)

    def test_live_capture_starts_with_the_first_scan(self):
        """Connecting to the sensor and booting the motor take seconds before the
        first scan: the empty-scene phase must still last its full time."""
        config = InPlaceCaptureConfig()
        config.timing.leave_seconds = 0.0
        config.timing.background_seconds, config.timing.delay = 0.5, 0.3
        config.stepping.duration = 1.0
        source = FakeSource(frames=40)
        source.live = True
        run_directory = RunDirectory(self.folder.path / "live")
        run_directory.write_sensor(source)
        clock = lambda: 3.5 + source.clock                                  # noqa: E731  3.5 s of setup
        log = Session(source, run_directory, origin=0.0, interactive=False, clock=clock).run(inplace_phases(config))
        run_directory.close(log)
        self.assertEqual(log["background"], 5)
        self.assertEqual(log["frames"], 10)

    def test_motor_timeout_stops_the_capture(self):
        config = TurntableCaptureConfig()
        config.timing.background_seconds, config.timing.delay, config.timing.hold = 0.2, 0.2, 0.3
        config.motor.turn_deg, config.motor.max_rotation_seconds = 360.0, 1.0
        source = FakeSource(frames=100)
        motor = FakeMotor(source, steps_per_second=1.0)                     # never finishes in time
        log, path = self.run_session(source, turntable_phases(config, motor), name="timeout")
        self.assertEqual([p[0] for p in log["phases"]], ["background", "countdown", "hold", "rotation"])
        phases = [NpzRecording(path).load(k).phase for k in range(log["frames"])]
        self.assertNotIn(3, phases)                                          # nothing labelled 'still after'

    def test_leave_phase_stores_nothing(self):
        config = InPlaceCaptureConfig()
        config.timing.leave_seconds, config.timing.background_seconds, config.timing.delay = 1.0, 0.5, 0.3
        config.stepping.duration, config.stepping.cue_every = 1.0, 0.0
        log, _ = self.run_session(FakeSource(frames=40), inplace_phases(config), name="leave")
        self.assertEqual([p[0] for p in log["phases"]], ["leave", "background", "countdown", "capture"])
        self.assertEqual(log["background"], 5)                             # the leave second is not stored
        self.assertEqual(log["frames"], 10)

    def test_motion_protocol_cues(self):
        from bodyscan.commands.motion import CaptureMotion, MotionCaptureConfig
        config = MotionCaptureConfig()
        t = config.timing
        t.leave_seconds, t.background_seconds, t.delay = 0.5, 0.5, 0.3
        t.duration, t.hold, t.back_seconds, t.cue_every = 8.0, 1.0, 2.0, 1.0
        log, _ = self.run_session(FakeSource(frames=100), CaptureMotion().phases(config, None), name="motion")
        self.assertEqual([p[0] for p in log["phases"]], ["leave", "background", "countdown", "record"])
        labels = [label for _, label in log["cues"]]
        self.assertEqual(labels, ["move", "change", "change", "back to the mark", "A-pose, stand still"])
        times = [at for at, _ in log["cues"]]
        for at, expected in zip(times, (1.0, 2.0, 3.0, 5.0, 7.0)):
            self.assertAlmostEqual(at, expected, delta=0.11)
        self.assertAlmostEqual(log["record_seconds"], 8.0, delta=0.11)

    def test_turntable_leave_shortens_the_two_pc_wait(self):
        config = TurntableCaptureConfig()
        config.motor.turn_deg = 360.0
        before = record_seconds(config)
        config.timing.leave_seconds = 10.0
        self.assertAlmostEqual(before - record_seconds(config), 10.0, places=6)

    def test_check_take_report(self):
        import contextlib
        import io
        import types
        from bodyscan.commands.motion import CheckTake
        take, seg = self.folder.path / "take", self.folder.path / "seg"
        (take / "frames").mkdir(parents=True)
        seg.mkdir()
        rng = np.random.default_rng(0)
        frames = []
        for k in range(40):
            mask = np.ones(64, dtype=np.uint8)
            if k % 4 == 0:
                mask[8:24] = 0                                                  # one lost packet
            np.savez(take / "frames" / f"frame_{k:05d}.npz", columns_ok=np.float64(mask.mean()),
                     time=np.float64(30.0 + 0.05 * k), column_ok=mask)
            if k not in (3, 4):
                frames.append({"index": k, "time": 30.0 + 0.05 * k, "distance_m": 2.0 + 0.05 * k, "points": 900})
        (seg / "segments.json").write_text(json.dumps({"frames": frames, "missing": [3, 4]}))
        (take / "capture.json").write_text(json.dumps({"record_start": 30.0,
                                                       "cues": [[0.5, "move"], [1.5, "back to the mark"]]}))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = CheckTake().run(types.SimpleNamespace(take=str(take), segments=str(seg), near=1.5, far=4.0))
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("frames below 0.99: 10 (25 %)", text)
        self.assertIn("person found in 38 of 40 frames", text)
        self.assertIn("move", text)
        self.assertIn("0 frames outside", text)

    def test_turns_in_both_directions(self):
        self.assertEqual(split_turn(-400), [-STEPS_PER_TURN, -3200])
        self.assertEqual(split_turn(0), [])

    def test_two_pc_protocol_stops_by_itself(self):
        config = TurntableCaptureConfig()
        config.timing.background_seconds, config.timing.delay = 0.2, 0.2
        config.two_pc.duration = 1.0
        log, _ = self.run_session(FakeSource(frames=100), turntable_phases(config, None))
        self.assertEqual(log["frames"], 10)
        self.assertAlmostEqual(log["record_seconds"], 1.0, delta=0.11)


if __name__ == "__main__":
    unittest.main()
