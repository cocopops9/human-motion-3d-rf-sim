"""Configuration sections: every tunable parameter declared once.

A configuration section is a dataclass whose fields are created with param():

    @dataclass
    class FusionConfig:
        voxel: float = param(0.005, "voxel of the fused cloud", unit="m",
                             effect="smaller: denser output, slower; larger: fewer points")

A pipeline configuration is a dataclass of sections (for example
TurntableConfig with sections frames, scene, isolation, motion, ...). The same
declaration drives

    * the command-line options (add_arguments / from_arguments),
    * TOML configuration files, one table per section (load_toml / to_toml),
    * the parameter reference (to_markdown), so the documentation cannot drift
      away from the code.

Precedence when a pipeline starts: defaults, then each --config file in
order, then the command-line options, then --set section.name=value.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
import types
import typing
from pathlib import Path
from typing import Any

try:                                   # Python 3.11 and later
    import tomllib
except ModuleNotFoundError:            # pragma: no cover - older Python: JSON configs only
    tomllib = None


def param(default: Any, help: str, unit: str = "", effect: str = "", choices: tuple | None = None):
    """A documented configuration field.

    help    what the parameter is.
    unit    physical unit, shown in --help and in the parameter reference.
    effect  what changes in the result when the value goes up or down.
    choices allowed values for a string parameter."""
    metadata = {"help": help, "unit": unit, "effect": effect, "choices": choices}
    if isinstance(default, (list, dict)):
        return dataclasses.field(default_factory=lambda value=default: copy.deepcopy(value), metadata=metadata)
    return dataclasses.field(default=default, metadata=metadata)


def section(cls, **overrides):
    """Field holding a whole configuration section, created with its defaults;
    'overrides' changes some defaults for one pipeline (the in-place pipeline
    uses the slab correction of the turntable with wider bounds)."""
    return dataclasses.field(default_factory=lambda: cls(**overrides))


# ----------------------------------------------------------------------------
# Introspection
# ----------------------------------------------------------------------------

def sections(config) -> list[tuple[str, Any]]:
    """(name, section instance) pairs of a pipeline configuration."""
    return [(f.name, getattr(config, f.name)) for f in dataclasses.fields(config)]


def section_fields(section_value) -> list[dataclasses.Field]:
    return list(dataclasses.fields(section_value))


# "X | None" is types.UnionType from Python 3.10; Optional[X] is typing.Union.
UNION_TYPES = (typing.Union,) + ((types.UnionType,) if hasattr(types, "UnionType") else ())


def _hint_of(owner, name):
    cls = owner if isinstance(owner, type) else type(owner)
    try:
        return typing.get_type_hints(cls)[name]
    except TypeError:                  # Python 3.9 cannot evaluate "X | None"
        return _hint_from_text(cls, name)


def _hint_from_text(cls, name):
    """Type hint from the annotation text, with "X | None" read as Optional[X]."""
    for klass in cls.__mro__:
        text = klass.__dict__.get("__annotations__", {}).get(name)
        if text is not None:
            break
    else:
        raise KeyError(name)
    if not isinstance(text, str):
        return text
    parts = [part.strip() for part in text.split("|")]
    optional = "None" in parts
    parts = [part for part in parts if part != "None"]
    namespace = dict(vars(sys.modules[klass.__module__]))
    base = eval(parts[0], namespace)                    # noqa: S307 - annotations of our own dataclasses
    return typing.Optional[base] if optional else base


def _spec(hint) -> dict:
    """Describe a type hint for argparse and for conversion of TOML values.
    Supported: bool, int, float, str, X | None, tuple[float, ...], list[float]."""
    optional = False
    origin = typing.get_origin(hint)
    if origin in UNION_TYPES:
        arguments = [a for a in typing.get_args(hint) if a is not type(None)]
        optional = len(arguments) < len(typing.get_args(hint))
        hint = arguments[0]
        origin = typing.get_origin(hint)
    if origin in (tuple, list):
        arguments = typing.get_args(hint)
        element = arguments[0] if arguments else float
        if origin is tuple and arguments and arguments[-1] is not Ellipsis:
            nargs = len(arguments)
        else:
            nargs = "+"
        return {"kind": "sequence", "element": element, "nargs": nargs, "container": origin, "optional": optional}
    return {"kind": "scalar", "type": hint, "optional": optional}


def _convert(value, spec):
    if value is None:
        return None
    if spec["optional"] and isinstance(value, str) and value.strip().lower() in ("none", "null", "not set"):
        return None                       # TOML has no null: an optional parameter is unset with "none"
    if spec["kind"] == "sequence":
        items = [spec["element"](v) for v in value]
        return tuple(items) if spec["container"] is tuple else items
    kind = spec["type"]
    if kind is bool:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("1", "true", "yes", "on"):
                return True
            if lowered in ("0", "false", "no", "off"):
                return False
            raise ValueError(f"not a boolean: {value!r}")
        return bool(value)
    return kind(value)


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

def _flag_names(config_cls) -> dict[tuple[str, str], str]:
    """Option name per (section, field): --field-name when the name is unique
    across the sections, else --section-field-name."""
    config = config_cls()
    count: dict[str, int] = {}
    for _, value in sections(config):
        for f in section_fields(value):
            count[f.name] = count.get(f.name, 0) + 1
    names = {}
    for section_name, value in sections(config):
        for f in section_fields(value):
            base = f.name if count[f.name] == 1 else f"{section_name}_{f.name}"
            names[(section_name, f.name)] = "--" + base.replace("_", "-")
    return names


def add_arguments(parser: argparse.ArgumentParser, config_cls) -> None:
    """Add every parameter of a pipeline configuration as an option, one
    argument group per section. Options default to 'not given', so that only
    what is typed on the command line overrides the configuration files."""
    config = config_cls()
    names = _flag_names(config_cls)
    parser.add_argument("--config", action="append", default=[], metavar="FILE",
                        help="TOML (or JSON) configuration file; several are applied in order")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.NAME=VALUE",
                        help="override one parameter, e.g. --set fusion.voxel=0.004")
    for section_name, value in sections(config):
        doc = (type(value).__doc__ or "").strip().splitlines()
        group = parser.add_argument_group(f"[{section_name}]", doc[0] if doc else None)
        for f in section_fields(value):
            spec = _spec(_hint_of(value, f.name))
            unit = f.metadata.get("unit") or ""
            default = getattr(value, f.name)
            text = f.metadata.get("help", "")
            text += f" [{unit}]" if unit else ""
            text += f" (default {default!r})"
            text = text.replace("%", "%%")          # argparse formats help strings with %
            dest = f"{section_name}.{f.name}"
            kwargs: dict[str, Any] = {"dest": dest, "default": argparse.SUPPRESS, "help": text}
            if spec["kind"] == "sequence":
                kwargs.update(type=spec["element"], nargs=spec["nargs"], metavar="V")
            elif spec["type"] is bool:
                kwargs.update(action=argparse.BooleanOptionalAction)
            else:
                kwargs.update(type=spec["type"], metavar="VALUE")
                if f.metadata.get("choices"):
                    kwargs.pop("metavar")
                    kwargs["choices"] = f.metadata["choices"]
            group.add_argument(names[(section_name, f.name)], **kwargs)


def from_arguments(config_cls, arguments: argparse.Namespace):
    """Configuration from defaults, --config files, options and --set."""
    config = config_cls()
    for path in getattr(arguments, "config", []) or []:
        merge(config, load_file(path))
    for key, value in vars(arguments).items():
        if "." in key and not key.startswith("_"):
            section_name, name = key.split(".", 1)
            if hasattr(config, section_name):
                set_value(config, section_name, name, value)
    for assignment in getattr(arguments, "set", []) or []:
        if "=" not in assignment or "." not in assignment.split("=", 1)[0]:
            raise SystemExit(f"--set expects SECTION.NAME=VALUE, got {assignment!r}")
        key, text = assignment.split("=", 1)
        section_name, name = key.strip().split(".", 1)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
        set_value(config, section_name, name, value)
    return config


def set_value(config, section_name: str, name: str, value) -> None:
    if not hasattr(config, section_name):
        raise SystemExit(f"unknown configuration section [{section_name}]")
    target = getattr(config, section_name)
    if not any(f.name == name for f in dataclasses.fields(target)):
        raise SystemExit(f"unknown parameter {section_name}.{name}")
    spec = _spec(_hint_of(target, name))
    converted = _convert(value, spec)
    choices = next(f.metadata.get("choices") for f in dataclasses.fields(target) if f.name == name)
    if choices and converted is not None and converted not in choices:
        raise SystemExit(f"{section_name}.{name} = {converted!r}: choose one of {', '.join(map(str, choices))}")
    setattr(target, name, converted)


# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------

def load_file(path) -> dict:
    """A configuration file: TOML (one table per section) or JSON (same shape)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if tomllib is None:
        raise SystemExit("TOML configuration files need Python 3.11 or later; use a .json file instead")
    return tomllib.loads(text)


def merge(config, values: dict) -> None:
    """Apply a {section: {name: value}} mapping; unknown names are errors (typos)."""
    for section_name, table in values.items():
        if not isinstance(table, dict):
            raise SystemExit(f"configuration: [{section_name}] must be a table")
        for name, value in table.items():
            set_value(config, section_name, name, value)


def to_dict(config) -> dict:
    return {name: dataclasses.asdict(value) for name, value in sections(config)}


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(type(value))


def to_toml(config, with_help: bool = True) -> str:
    """The whole configuration as a TOML file (TOML has no null value: a
    parameter left unset is written as "none")."""
    lines = []
    for section_name, value in sections(config):
        doc = (type(value).__doc__ or "").strip().splitlines()
        if lines:
            lines.append("")
        if with_help and doc:
            lines.append(f"# {doc[0]}")
        lines.append(f"[{section_name}]")
        for f in section_fields(value):
            current = getattr(value, f.name)
            unit = f.metadata.get("unit") or ""
            comment = f"  # {f.metadata.get('help', '')}" + (f" [{unit}]" if unit else "") if with_help else ""
            shown = '"none"' if current is None else _toml_value(current)
            lines.append(f"{f.name} = {shown}{comment}")
    return "\n".join(lines) + "\n"


def to_markdown(config_cls, title: str) -> str:
    """Parameter reference of one pipeline configuration (docs/parameters.md)."""
    config = config_cls()
    names = _flag_names(config_cls)
    out = [f"## {title}", ""]
    for section_name, value in sections(config):
        doc = (type(value).__doc__ or "").strip()
        out += [f"### [{section_name}]", ""]
        if doc:
            out += [doc, ""]
        out += ["| Option | Default | Meaning | Effect of changing it |", "|---|---|---|---|"]
        for f in section_fields(value):
            unit = f.metadata.get("unit") or ""
            default = getattr(value, f.name)
            shown = "not set" if default is None else f"`{default}`" + (f" {unit}" if unit else "")
            meaning = f.metadata.get("help", "").replace("|", "\\|")
            effect = (f.metadata.get("effect") or "").replace("|", "\\|")
            out.append(f"| `{names[(section_name, f.name)]}` | {shown} | {meaning} | {effect} |")
        out.append("")
    return "\n".join(out)
