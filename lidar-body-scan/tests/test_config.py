"""Configuration sections: options, files, precedence, documentation."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from bodyscan import config as cfg
from bodyscan.config import param, section


@dataclass
class AlphaSection:
    """First section."""
    voxel: float = param(0.005, "voxel of something", unit="m", effect="smaller: denser")
    count: int = param(3, "a count of 50 % of things")
    flag: bool = param(True, "a switch")
    center: tuple[float, float] | None = param(None, "a point", unit="m")


@dataclass
class BetaSection:
    """Second section."""
    voxel: float = param(0.02, "another voxel", unit="m")
    mode: str = param("fast", "a mode", choices=("fast", "slow"))


@dataclass
class Example:
    alpha: AlphaSection = section(AlphaSection)
    beta: BetaSection = section(BetaSection, mode="slow")


def parse(argv):
    parser = argparse.ArgumentParser()
    cfg.add_arguments(parser, Example)
    return cfg.from_arguments(Example, parser.parse_args(argv))


class ConfigTest(unittest.TestCase):
    def test_defaults_and_section_overrides(self):
        config = parse([])
        self.assertEqual(config.alpha.voxel, 0.005)
        self.assertEqual(config.beta.mode, "slow")          # overridden default of this pipeline
        self.assertIsNone(config.alpha.center)

    def test_duplicated_names_get_the_section_prefix(self):
        config = parse(["--alpha-voxel", "0.01", "--beta-voxel", "0.03", "--count", "7"])
        self.assertEqual((config.alpha.voxel, config.beta.voxel, config.alpha.count), (0.01, 0.03, 7))

    def test_types_tuples_and_booleans(self):
        config = parse(["--center", "1.5", "-0.25", "--no-flag", "--mode", "fast"])
        self.assertEqual(config.alpha.center, (1.5, -0.25))
        self.assertFalse(config.alpha.flag)
        self.assertEqual(config.beta.mode, "fast")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse(["--mode", "medium"])

    def test_precedence_file_then_option_then_set(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "c.json"
            path.write_text(json.dumps({"alpha": {"count": 5, "voxel": 0.008}}))
            config = parse(["--config", str(path), "--count", "6"])
            self.assertEqual((config.alpha.count, config.alpha.voxel), (6, 0.008))
            config = parse(["--config", str(path), "--count", "6", "--set", "alpha.count=9"])
            self.assertEqual(config.alpha.count, 9)

    def test_unknown_names_are_errors(self):
        with self.assertRaises(SystemExit):
            parse(["--set", "alpha.voxl=1"])
        with self.assertRaises(SystemExit):
            cfg.merge(Example(), {"gamma": {"x": 1}})

    def test_toml_round_trip(self):
        if cfg.tomllib is None:
            self.skipTest("TOML needs Python 3.11")
        self.assertIn('center = "none"', cfg.to_toml(parse([])))      # unset optional parameter
        config = parse(["--center", "1", "2", "--count", "4"])
        text = cfg.to_toml(config)
        self.assertIn("center = [1.0, 2.0]", text)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "c.toml"
            path.write_text(text)
            loaded = Example()
            cfg.merge(loaded, cfg.load_file(path))
            self.assertEqual(cfg.to_dict(loaded), cfg.to_dict(config))

    def test_annotations_without_union_operator(self):
        """Python 3.9 reads "X | None" from the annotation text."""
        hint = cfg._hint_from_text(AlphaSection, "center")
        self.assertEqual(cfg._spec(hint)["kind"], "sequence")
        self.assertTrue(cfg._spec(hint)["optional"])
        self.assertIs(cfg._hint_from_text(AlphaSection, "voxel"), float)

    def test_help_with_percent_and_markdown(self):
        parser = argparse.ArgumentParser()
        cfg.add_arguments(parser, Example)
        self.assertIn("50 %", parser.format_help())
        markdown = cfg.to_markdown(Example, "example")
        for name in ("--alpha-voxel", "--beta-voxel", "--count", "--flag", "--center", "--mode"):
            self.assertIn(f"`{name}`", markdown)
        self.assertIn("smaller: denser", markdown)


if __name__ == "__main__":
    unittest.main()
