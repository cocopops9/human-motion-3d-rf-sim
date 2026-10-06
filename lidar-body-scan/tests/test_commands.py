"""Command line: every command is registered; configurations written by --write-config load back."""

import contextlib
import io
import unittest

from bodyscan import config as cfg
from bodyscan.commands import build_parser, main
from tests.helpers import TemporaryFolder

COMMANDS = {"fuse", "fuse-inplace", "mesh", "quality", "detect", "capture-turntable", "capture-inplace",
            "check-view", "check-sensor", "convert", "simulate", "params", "smooth"}


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
                        "bodyscan capture-turntable", "bodyscan smooth"):
            self.assertIn(heading, text)

    def test_every_configuration_file_loads(self):
        # configs/<command>_*.toml must load into that command's configuration (no stale sections)
        if cfg.tomllib is None:
            self.skipTest("TOML needs Python 3.11")
        from pathlib import Path
        from bodyscan.commands.base import Command
        commands = {"turntable": "fuse", "inplace": "fuse-inplace", "capture_turntable": "capture-turntable",
                    "capture_inplace": "capture-inplace", "mesh": "mesh", "smooth": "smooth", "detect": "detect"}
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
