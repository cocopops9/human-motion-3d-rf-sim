"""Smoothing pipeline for an existing mesh ('bodyscan smooth').

    LoadMesh       .ply/.obj/.stl mesh, optional point cloud for the fidelity measure
    SmoothRuns     the smoothing (meshing.smoother) with the given parameters, or once
                   per combination of the --sweep values; writes <out>.ply (or one
                   <out>_<name><value>.ply per combination), each with <name>_quality.json
    Summary        one table: settings against facet noise, bump height, deviation

The input is normally the unsmoothed output of 'bodyscan mesh'.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from bodyscan.config import param, section, set_value
from bodyscan.jsonio import write_json
from bodyscan.log import info
from bodyscan.meshing.quality import format_quality, mesh_quality
from bodyscan.meshing.smoother import SmoothingConfig, smooth_mesh
from bodyscan.pipelines.base import Context, Pipeline, Step, StepList


@dataclass
class SweepConfig:
    """Comparison of several values (one mesh per combination)."""
    sweep: tuple[str, ...] = param((), "NAME=V1,V2,... : run once per value of a smoothing parameter and write one "
                                       "mesh per value, <out>_<name><value>.ply; several sweeps run every "
                                       "combination. Example: --sweep rounds=1,2,4,8 scale_mm=6,12",
                                   effect="nothing changes in a single result: it is for choosing the values")


@dataclass
class SmoothReportConfig:
    """Quality report."""
    frequency_ghz: float = param(60.0, "carrier frequency (roughness targets lambda / 8 and lambda / 32)", unit="GHz")
    radius: float = param(0.01, "neighbourhood of the facet-noise and bump-height measures", unit="m")


@dataclass
class SmoothMeshConfig:
    smoothing: SmoothingConfig = section(SmoothingConfig)
    compare: SweepConfig = section(SweepConfig)
    report: SmoothReportConfig = section(SmoothReportConfig)


def mesh_path(out: Path, suffix: str = "") -> Path:
    """<out><suffix>.ply, keeping a known mesh extension of 'out' and any dots in its name."""
    out = Path(out)
    if out.suffix.lower() in (".ply", ".obj", ".stl", ".off"):
        return out.with_name(out.stem + suffix + out.suffix)
    return out.with_name(out.name + suffix + ".ply")


def parse_sweeps(entries) -> list[tuple[str, list[str]]]:
    """[('rounds', ['1', '2', '4'])] from ['rounds=1,2,4']; names must be smoothing parameters."""
    names = {f.name for f in dataclasses.fields(SmoothingConfig)}
    sweeps = []
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"--sweep expects NAME=V1,V2,..., got {entry!r}")
        name, values = entry.split("=", 1)
        name = name.strip().replace("-", "_")
        if name not in names:
            raise SystemExit(f"--sweep {name}: not a smoothing parameter (choose from {', '.join(sorted(names))})")
        items = [v.strip() for v in values.split(",") if v.strip()]
        if not items:
            raise SystemExit(f"--sweep {name}: no values")
        sweeps.append((name, items))
    return sweeps


def variants(base: SmoothingConfig, sweeps) -> list[tuple[str, str, SmoothingConfig]]:
    """(file suffix, label, configuration) for every combination of the sweep values."""
    if not sweeps:
        return [("", "smoothed", base)]
    result = []
    for combination in itertools.product(*[values for _, values in sweeps]):
        holder = type("Holder", (), {})()
        holder.smoothing = dataclasses.replace(base)
        for (name, _), value in zip(sweeps, combination):
            set_value(holder, "smoothing", name, value)
        suffix = "".join(f"_{name}{value}" for (name, _), value in zip(sweeps, combination))
        label = " ".join(f"{name}={value}" for (name, _), value in zip(sweeps, combination))
        result.append((suffix, label, holder.smoothing))
    return result


class LoadMesh(Step):
    name = "load"

    def run(self, ctx: Context) -> None:
        mesh = o3d.io.read_triangle_mesh(str(ctx.mesh_path))
        if len(mesh.triangles) == 0:
            raise SystemExit(f"{ctx.mesh_path}: no triangles")
        mesh.remove_unreferenced_vertices()
        ctx.mesh = mesh
        ctx.cloud_points = None
        if getattr(ctx, "cloud_path", None):
            ctx.cloud_points = np.asarray(o3d.io.read_point_cloud(str(ctx.cloud_path)).points)
        ctx.report["input"] = str(Path(ctx.mesh_path).resolve())
        r = ctx.config.report
        before = mesh_quality(mesh, r.frequency_ghz, r.radius, ctx.cloud_points)
        info("input mesh:\n  " + format_quality(before).replace("\n", "\n  "))
        ctx.report["quality_input"] = before
        ctx.rows = [("input", before, None)]


def describe(s: SmoothingConfig) -> str:
    text = f"scale {s.scale_mm:g} mm x {s.rounds} rounds"
    if s.finish_rounds > 0 and 0 < s.finish_scale_mm < s.scale_mm:
        text += f" + {s.finish_rounds} at {s.finish_scale_mm:g} mm"
    text += f", normal sigma {s.normal_sigma:g}, keep volume {'on' if s.keep_volume else 'off'}, "
    text += f"limit {s.max_deviation_mm:g} mm" if s.max_deviation_mm > 0 else "no limit"
    return text


class SmoothRuns(Step):
    name = "smooth"

    def run(self, ctx: Context) -> None:
        r = ctx.config.report
        runs = variants(ctx.config.smoothing, parse_sweeps(ctx.config.compare.sweep))
        ctx.report["outputs"] = {}
        for suffix, label, s in runs:
            info(f"{label}: {describe(s)}")

            def progress(number, scale_mm, statistics):
                text = (f"  round {number} ({scale_mm:g} mm): distance from the input median "
                        f"{statistics['deviation_mm']['median']} mm, p99 {statistics['deviation_mm']['p99']} mm, "
                        f"max {statistics['deviation_max_mm']} mm")
                if "held_at_limit_fraction" in statistics:
                    text += f"; {100 * statistics['held_at_limit_fraction']:.1f} % of the vertices at the limit"
                info(text)

            mesh, statistics = smooth_mesh(ctx.mesh, s, progress)
            quality = mesh_quality(mesh, r.frequency_ghz, r.radius, ctx.cloud_points)
            quality["deviation_from_input"] = statistics
            path = mesh_path(ctx.out, suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_triangle_mesh(str(path), mesh)
            write_json(path.with_name(path.stem + "_quality.json"),
                       {**ctx.report, "settings": dataclasses.asdict(s), "quality": quality})
            info(f"wrote {path}")
            ctx.report["outputs"][label] = str(path)
            ctx.rows.append((label, quality, statistics))
        ctx.mesh_out = mesh


class Summary(Step):
    name = "summary"

    def run(self, ctx: Context) -> None:
        cloud = ctx.cloud_points is not None
        width = max(10, max(len(label) for label, _, _ in ctx.rows))
        header = (f"{'mesh':>{width}} | facet noise median / p90 [deg] | bump rms [mm] | "
                  f"distance from input median / p99 / max [mm]" + (" | cloud distance median [mm]" if cloud else ""))
        lines = ["", header]
        for label, quality, statistics in ctx.rows:
            facet = quality.get("facet_noise_deg", {})
            if statistics:
                d = statistics["deviation_mm"]
                deviation = f"{d['median']:.2f} / {d['p99']:.2f} / {statistics['deviation_max_mm']:.2f}"
            else:
                deviation = "0 / 0 / 0"
            median, p90 = facet.get("median", float("nan")), facet.get("p90", float("nan"))
            line = (f"{label:>{width}} | {median:>14.2f} / {p90:<13.2f} | "
                    f"{quality.get('height_rms_mm', float('nan')):>13.3f} | {deviation:>42}")
            if cloud:
                line += f" | {quality.get('fidelity_mm', {}).get('median', float('nan')):>10.2f}"
            lines.append(line)
        info("\n".join(lines))


class SmoothMeshPipeline(Pipeline):
    config_class = SmoothMeshConfig
    name = "smooth"

    def steps(self) -> StepList:
        return StepList([LoadMesh(), SmoothRuns(), Summary()])
