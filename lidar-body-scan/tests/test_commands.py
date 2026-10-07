"""Command line: every command is registered; configurations written by --write-config load back."""

import contextlib
import io
import unittest

from bodyscan import config as cfg
from bodyscan.commands import build_parser, main
from tests.helpers import TemporaryFolder

COMMANDS = {"fuse", "fuse-inplace", "mesh", "quality", "detect", "capture-turntable", "capture-inplace",
            "check-view", "check-sensor", "convert", "simulate", "params", "smooth",
            "capture-motion", "segment-motion", "avatar", "track", "export-motion", "review-motion",
            "simulate-motion", "evaluate-motion", "bench-motion", "make-test-body"}


class CommandLineTest(unittest.TestCase):
    def test_every_command_is_registered(self):
        parser = build_parser()
        registered = set(parser._subparsers._group_actions[0].choices)
        self.assertTrue(COMMANDS <= registered, COMMANDS - registered)

    def test_write_config_round_trip(self):
        if cfg.tomllib is None:
            self.skipTest("TOML needs Python 3.11")
        from bodyscan.pipelines.turntable import TurntableConfig
        folder = TemporaryFolder()
        try:
            path = folder.path / "tt.toml"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["fuse", "x", "--write-config", str(path), "--radius", "0.8", "--center", "1.0", "0.2"])
            loaded = TurntableConfig()
            cfg.merge(loaded, cfg.load_file(path))
            self.assertEqual(loaded.isolation.radius, 0.8)
            self.assertEqual(loaded.platform.center, (1.0, 0.2))
        finally:
            folder.cleanup()

    def test_parameter_reference(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["params"])
        text = output.getvalue()
        for heading in ("bodyscan fuse (turntable)", "bodyscan fuse-inplace", "bodyscan mesh", "bodyscan detect",
                        "bodyscan capture-turntable", "bodyscan smooth", "bodyscan capture-motion",
                        "bodyscan segment-motion", "bodyscan avatar", "bodyscan track", "bodyscan export-motion",
                        "bodyscan simulate-motion", "bodyscan bench-motion"):
            self.assertIn(heading, text)

    def test_track_options_round_trip(self):
        # two sections share parameter names (rounds, iterations): their options carry the section name
        if cfg.tomllib is None:
            self.skipTest("TOML needs Python 3.11")
        from bodyscan.dynamic.track import TrackPipelineConfig
        folder = TemporaryFolder()
        try:
            path = folder.path / "track.toml"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["track", "x", "--avatar", "a.npz", "--out", "o", "--write-config", str(path),
                      "--smoothness-weight", "0.5", "--refine-rounds", "3", "--tracking-rounds", "4",
                      "--no-rolling-shutter"])
            loaded = TrackPipelineConfig()
            cfg.merge(loaded, cfg.load_file(path))
            self.assertEqual(loaded.refine.smoothness_weight, 0.5)
            self.assertEqual(loaded.refine.rounds, 3)
            self.assertEqual(loaded.tracking.rounds, 4)
            self.assertFalse(loaded.refine.rolling_shutter)
        finally:
            folder.cleanup()

    def test_every_configuration_file_loads(self):
        # configs/<command>_*.toml must load into that command's configuration (no stale sections)
        if cfg.tomllib is None:
            self.skipTest("TOML needs Python 3.11")
        from pathlib import Path
        from bodyscan.commands.base import Command
        commands = {"turntable": "fuse", "inplace": "fuse-inplace", "capture_turntable": "capture-turntable",
                    "capture_inplace": "capture-inplace", "mesh": "mesh", "smooth": "smooth", "detect": "detect",
                    "capture_motion": "capture-motion", "segment_motion": "segment-motion", "avatar": "avatar",
                    "track": "track", "export_motion": "export-motion", "simulate_motion": "simulate-motion"}
        files = sorted((Path(__file__).resolve().parent.parent / "configs").glob("*.toml"))
        self.assertTrue(files)
        for path in files:
            prefix = max((p for p in commands if path.stem.startswith(p + "_")), key=len, default=None)
            self.assertIsNotNone(prefix, path.name)
            config_class = Command.registry[commands[prefix]]().config_class
            config = config_class()
            cfg.merge(config, cfg.load_file(path))


if __name__ == "__main__":
    unittest.main()
