"""Command-line framework: one Command subclass per command, registered by name.

A new command is a subclass with a 'name' (and 'help'); defining it inside a
module imported by bodyscan.commands is enough to make it available as
python -m bodyscan <name>."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bodyscan import config as cfg
from bodyscan.log import info, set_quiet


class Command:
    name: str = ""
    help: str = ""
    aliases: tuple = ()
    registry: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            Command.registry[cls.name] = cls

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Options of the command."""

    def run(self, args: argparse.Namespace) -> int:
        raise NotImplementedError


class ConfiguredCommand(Command):
    """A command whose parameters are a configuration dataclass (config_class):
    every parameter is an option, plus --config files, --set, --write-config
    and --quiet."""
    config_class = None

    def add_config_arguments(self, parser) -> None:
        parser.add_argument("--quiet", action="store_true", help="no console output")
        parser.add_argument("--write-config", metavar="FILE",
                            help="write the effective configuration as TOML and exit (a starting point for --config)")
        cfg.add_arguments(parser, self.config_class)

    def build_config(self, args):
        return cfg.from_arguments(self.config_class, args)

    def prepare(self, args):
        """Configuration from the arguments; None when only --write-config was asked."""
        set_quiet(args.quiet)
        config = self.build_config(args)
        if args.write_config:
            path = Path(args.write_config)
            text = (json.dumps(cfg.to_dict(config), indent=2) if path.suffix.lower() == ".json"
                    else cfg.to_toml(config))
            path.write_text(text, encoding="utf-8")
            info(f"wrote {args.write_config}")
            return None
        from bodyscan import __version__
        info(f"bodyscan {__version__}: {self.name}")
        return config


class PipelineCommand(ConfiguredCommand):
    """A command that runs a Pipeline: input path, --out, and the options of
    the pipeline configuration."""
    pipeline_class = None
    input_help = "input"
    default_out = "out"

    @property
    def config_class(self):
        return self.pipeline_class.config_class

    def add_arguments(self, parser):
        parser.add_argument("input", help=self.input_help)
        parser.add_argument("--out", default=self.default_out, help="output base name (path without extension)")
        self.add_config_arguments(parser)

    def inputs(self, args) -> dict:
        return {"run_dir": Path(args.input), "out": Path(args.out)}

    def run(self, args):
        config = self.prepare(args)
        if config is not None:
            self.pipeline_class(config).run(**self.inputs(args))
        return 0
