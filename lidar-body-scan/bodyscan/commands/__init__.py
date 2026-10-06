"""Command-line interface: python -m bodyscan <command> [options] (python -m bodyscan --help)."""

from __future__ import annotations

import argparse
import importlib
import sys

from bodyscan.commands.base import Command, PipelineCommand

# Modules whose Command subclasses register themselves on import. A module
# that needs an optional dependency (ouster-sdk) imports it inside run().
_MODULES = ["fuse", "inplace", "mesh", "detect", "tools", "capture"]


def _load_commands():
    for module in _MODULES:
        try:
            importlib.import_module(f"bodyscan.commands.{module}")
        except ModuleNotFoundError as error:          # a command module not written yet, or a missing dependency
            if error.name and error.name.startswith("bodyscan.commands"):
                continue
            raise


def build_parser() -> argparse.ArgumentParser:
    _load_commands()
    parser = argparse.ArgumentParser(prog="bodyscan", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, command_class in sorted(Command.registry.items()):
        command = command_class()
        sub_parser = sub.add_parser(name, help=command_class.help, aliases=list(command_class.aliases),
                                    description=command_class.__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
        command.add_arguments(sub_parser)
        sub_parser.set_defaults(_command=command)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return int(args._command.run(args) or 0)


__all__ = ["Command", "PipelineCommand", "build_parser", "main"]
