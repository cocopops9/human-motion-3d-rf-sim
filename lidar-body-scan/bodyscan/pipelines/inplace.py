"""In-place pipeline: a person turning on the spot by steps in front of a fixed LiDAR.

Compared with the turntable there are four difficulties, handled as follows.

    Imperfect rotation   the body drifts a few cm at every step and the turning
                         axis wanders: every keyframe gets its own pose
    Steps                frames recorded while stepping show legs and arms in
                         transit: a per-pixel motion score finds the still
                         periods, one keyframe per stop
    Micro-movements      breathing and sway: a keyframe is the per-pixel median
                         of up to keyframe_frames still frames
    Non-rigid changes    arms and legs are not exactly in the same place at
                         every stop: 4-DOF registration with a robust kernel,
                         height-slab correction, support filter

Steps:

    LoadRecording        the run directory (frames, background)
    SceneFromBackground  floor frame and background from the empty-scene frames
    LocatePerson         person region: [region] crop box, or the person found in the
                         frames with no region of the room assumed ([select]: by default
                         the person cascade; detection.finder) and the cylinder it
                         occupies about its centre ([region] center imposes the centre)
    FindStillPeriods     motion score of every frame, still periods
    BuildKeyframes       one per-pixel median keyframe per still period
    RegisterKeyframes    pose of every keyframe (registration.keyframes)
    CorrectKeyframes     height slabs (lean, head, hips)
    FuseKeyframes        support filter, voxel averaging, surface fit, confidence
    WriteOutputs         <out>.ply, <out>_confidence.ply, <out>_views.ply, <out>.json,
                         <out>_top.png, <out>_motion.png

Output frame: z up, z = 0 on the floor, origin on the floor below the centre
of the body, axes of the first keyframe.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from bodyscan.config import param, section
from bodyscan.fusion import FusionConfig, SlabConfig, SlabCorrection, SurfaceFusion
from bodyscan.geometry import evaluate, yaw_of
from bodyscan.io import write_confidence
from bodyscan.jsonio import write_json
from bodyscan.log import info, warning
from bodyscan.pipelines.base import Context, Pipeline, Step, StepList
from bodyscan.detection.finder import SelectConfig
from bodyscan.pipelines.common import FindSubject, FrameConfig, LoadRecording, SceneFromBackground, SubjectConfig
from bodyscan.plots import motion_plot, top_view
from bodyscan.registration.keyframes import KeyframeConfig, KeyframeRegistration
from bodyscan.scene import CylinderRegion, FloorConfig, ForegroundIsolator, IsolationConfig, SensorBoxRegion
from bodyscan.scene.stillness import StillConfig, keyframe_range, motion_trace, still_segments


@dataclass
class RegionConfig:
    """Where the person stands. Without a crop box, the person is found in the
    frames by the tests of [select] (the cascade of 'bodyscan detect --human'
    by default), and the region is measured on it."""
    crop_min: tuple[float, float, float] | None = param(None, "crop box, lower corner in the SENSOR frame (X Y Z)",
                                                        unit="m", effect="include the arms and the drift of the feet")
    crop_max: tuple[float, float, float] | None = param(None, "crop box, upper corner in the sensor frame (X Y Z)",
                                                        unit="m")
    center: tuple[float, float] | None = param(None, "centre of the person region in the floor frame (X Y); the "
                                                     "person near it is used to measure the region (or [isolation] "
                                                     "radius when given)", unit="m")


@dataclass
class InPlaceConfig:
    select: SelectConfig = section(SelectConfig, human=True)
    subject: SubjectConfig = section(SubjectConfig)
    frames: FrameConfig = section(FrameConfig)
    floor: FloorConfig = section(FloorConfig)
    region: RegionConfig = section(RegionConfig)
    isolation: IsolationConfig = section(IsolationConfig, min_height=0.025)
    stills: StillConfig = section(StillConfig)
    registration: KeyframeConfig = section(KeyframeConfig)
    slabs: SlabConfig = section(SlabConfig, slab_iterations=2, slab_max_turn=10.0, slab_max_shift=0.05)
    fusion: FusionConfig = section(FusionConfig, min_views=2)


class LocatePerson(FindSubject):
    """Person region: the crop box when given; else the person found in the
    frames ([select] tests, near [region] center when given) and a cylinder
    about its centre reaching its farthest points plus [subject] margin."""
    name = "region"

    def near(self, ctx: Context):
        center = ctx.config.region.center
        return None if center is None else np.array(center, dtype=np.float64)

    def run(self, ctx: Context) -> None:
        c = ctx.config
        r, iso = c.region, c.isolation
        if r.crop_min is not None and r.crop_max is not None:
            region = SensorBoxRegion(r.crop_min, r.crop_max, iso.min_height, iso.max_height)
            described = f"crop box {tuple(r.crop_min)} to {tuple(r.crop_max)} (sensor frame)"
        else:
            if r.center is not None and iso.radius is not None:
                ctx.subject, center = None, np.array(r.center, dtype=np.float64)
            else:
                super().run(ctx)
                center = np.array(r.center, dtype=np.float64) if r.center is not None else ctx.center
            radius = self.region_radius(ctx, center)
            region = CylinderRegion(center, radius, iso.min_height, iso.max_height)
            described = f"cylinder of radius {radius:.2f} m about ({center[0]:.3f}, {center[1]:.3f}) m"
            ctx.report["region_center"] = center
            ctx.report["region_radius_m"] = radius
        info(f"person region: {described}")
        ctx.isolator = ForegroundIsolator(ctx.source.sensor, ctx.background, ctx.floor, region, iso.edge_jump,
                                          iso.cluster_eps, iso.normal_radius)


class FindStillPeriods(Step):
    name = "stills"

    def run(self, ctx: Context) -> None:
        s = ctx.config.stills
        trace = motion_trace(ctx.source, ctx.isolator, s.motion_threshold)
        segments, threshold = still_segments(trace, s)
        info(f"frames: {len(trace.scores)}, person pixels median {int(np.median(trace.sizes))}, still threshold "
             f"{threshold:.3f}, still periods {len(segments)}")
        if len(segments) < 3:
            raise SystemExit("fewer than 3 still periods. Check the person region and the motion plot, raise "
                             "--still-max or --still-factor, or hold still longer at each stop.")
        ctx.trace, ctx.segments, ctx.still_threshold = trace, segments, threshold
        ctx.report.update({"still_threshold": threshold, "segments": [[int(a), int(b)] for a, b in segments]})


class BuildKeyframes(Step):
    name = "keyframes"

    def run(self, ctx: Context) -> None:
        s = ctx.config.stills
        trace = ctx.trace
        clouds, keyframes, used = [], [], []
        for n, segment in enumerate(ctx.segments):
            if n % s.keyframe_stride:
                continue
            range_m, chosen = keyframe_range(ctx.source, segment, s.keyframe_frames)
            cloud = ctx.isolator.cloud(range_m).cloud
            start, stop = segment
            if len(cloud.points) < s.min_pixels // 2:
                info(f"  still period {n} ({trace.times[start]:.1f} to {trace.times[stop - 1]:.1f} s) dropped: "
                     f"{len(cloud.points)} points")
                continue
            clouds.append(cloud)
            used.append(n)
            keyframes.append({"segment": n, "frames": [int(chosen[0]), int(chosen[-1])],
                              "time_s": [float(trace.times[start]), float(trace.times[stop - 1])],
                              "points": len(cloud.points)})
        info(f"keyframes: {len(clouds)} ({', '.join(str(k['points']) for k in keyframes)} points)")
        if len(clouds) < 3:
            raise SystemExit("fewer than 3 usable keyframes")
        ctx.keyframe_clouds, ctx.keyframes, ctx.used_segments = clouds, keyframes, used
        ctx.report["keyframes"] = keyframes


class RegisterKeyframes(Step):
    name = "registration"

    def __init__(self, registration_class=KeyframeRegistration):
        self.registration_class = registration_class

    def run(self, ctx: Context) -> None:
        result = self.registration_class(ctx.config.registration).run(ctx.keyframe_clouds)
        poses, edges, keys = result.poses, result.edges, result.keys
        yaws = np.unwrap([yaw_of(p) for p in poses])
        turn = np.degrees(yaws - yaws[0])
        if turn[-1] < 0:
            turn = -turn
        info(f"registration: {result.stats['sequential_edges']} consecutive edges, {result.stats['loop_closures']} "
             f"other edges, {result.stats['model_kept']} edges kept at the solved-turn model")
        info(f"turn per keyframe [deg]: {' '.join(f'{a:.0f}' for a in turn)}")
        coverage = float(turn.max() - turn.min())
        if coverage < 330:
            warning(f"the keyframes cover {coverage:.0f} deg only; part of the body was never seen")
        info("step table (keyframe pair: turn, walked distance of the body axis, overlap):")
        for edge in edges:
            if edge.target == edge.source + 1:
                walked = np.linalg.norm(keys.body_axis(edge.target) - keys.body_axis(edge.source))
                info(f"  {edge.source:2d}->{edge.target:2d}  turn {turn[edge.target] - turn[edge.source]:+6.1f} deg  "
                     f"walked {100 * walked:5.1f} cm  overlap {edge.fitness:.2f}")
        quality = []
        for edge in edges:
            relative = np.linalg.inv(poses[edge.target]) @ poses[edge.source]
            fitness, rmse = evaluate(keys.points[edge.source], keys.targets[edge.target], relative, keys.fine)
            quality.append({"source": edge.source, "target": edge.target, "loop": edge.uncertain,
                            "fitness": round(fitness, 3), "rmse_mm": round(1000 * rmse, 2)})
        ctx.poses, ctx.turn = poses, turn
        ctx.report.update({"turn_deg": turn, "coverage_deg": coverage, "edges": quality,
                           "registration": result.stats})
        sequential = [q for q in quality if q["target"] == q["source"] + 1]
        rmse = np.array([q["rmse_mm"] for q in quality])
        if sequential and len(rmse):
            info(f"alignment: consecutive overlap {min(q['fitness'] for q in sequential):.2f} to "
                 f"{max(q['fitness'] for q in sequential):.2f}, point-to-plane RMSE median {np.median(rmse):.1f} mm, "
                 f"max {rmse.max():.1f} mm")
        if len(sequential) < len(poses) - 1:
            warning(f"{len(poses) - 1 - len(sequential)} consecutive step(s) were dropped by the pose graph: "
                    "the chain is broken there (look at the step table and <out>_views.ply)")


class CorrectKeyframes(Step):
    """Keyframes moved by their poses, then the height-slab correction; 'extra'
    corrections (ViewCorrection objects) run at the end."""
    name = "corrections"

    def __init__(self, extra=()):
        self.extra = list(extra)

    def run(self, ctx: Context) -> None:
        c = ctx.config
        clouds = [copy.deepcopy(cloud).transform(pose) for cloud, pose in zip(ctx.keyframe_clouds, ctx.poses)]
        if c.slabs.slab_iterations > 0:
            clouds, result = SlabCorrection(c.slabs, clean_reference=False).apply(clouds)
            info(f"height-slab correction: RMS {', '.join(f'{h:.1f}' for h in result['rms_mm_per_pass'])} mm")
            ctx.report["slab_correction_rms_mm"] = result["rms_mm_per_pass"]
        for correction in self.extra:
            clouds, extra_report = correction.apply(clouds)
            ctx.report.setdefault("extra_corrections", {})[correction.name] = extra_report
        ctx.aligned = clouds


class FuseKeyframes(Step):
    name = "fusion"

    def run(self, ctx: Context) -> None:
        c = ctx.config
        fused, views, removed, confidence = SurfaceFusion(c.fusion).fuse(ctx.aligned, c.fusion.min_views)
        pieces = np.asarray(fused.voxel_down_sample(0.02).cluster_dbscan(eps=0.05, min_points=5))
        if pieces.size and pieces.max() >= 0:
            sizes = np.bincount(pieces[pieces >= 0])
            # An arm held away from the body can be a separate piece of a few per cent;
            # a misplaced group of keyframes is a large second body.
            big = int(np.sum(sizes > 0.25 * sizes.max()))
            if big > 1:
                warning(f"the fused cloud has {big} separate large pieces (a person is one): some keyframes are "
                        f"placed wrongly. Look at {ctx.out}_views.ply and {ctx.out}_top.png; a break of the chain "
                        "usually sits at a step with low overlap in the step table.")
        center = np.median(np.asarray(fused.points)[:, :2], axis=0)
        output_from_world = np.eye(4)
        output_from_world[:2, 3] = -center
        fused.transform(output_from_world)
        views.transform(output_from_world)
        top = float(np.asarray(fused.points)[:, 2].max())
        info(f"support filter (at least {c.fusion.min_views} keyframes) removed {100 * removed:.1f}% of the points")
        info(f"fused: {len(fused.points)} points, highest point {top:.3f} m above the floor")
        ctx.fused, ctx.views_cloud, ctx.confidence = fused, views, confidence
        ctx.report.update({"support_filter_removed_fraction": removed, "fused_points": len(fused.points),
                           "highest_point_m": top, "output_from_world": output_from_world,
                           "poses_output_frame": [output_from_world @ p for p in ctx.poses],
                           "confidence": {"views_median": float(np.median(confidence["views"])),
                                          "spread_median_mm": float(np.median(confidence["spread_mm"]))}})


class WriteOutputs(Step):
    name = "write"

    def run(self, ctx: Context) -> None:
        out = str(ctx.out)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(f"{out}.ply", ctx.fused)
        o3d.io.write_point_cloud(f"{out}_views.ply", ctx.views_cloud)
        wrote_confidence = write_confidence(f"{out}_confidence.ply", ctx.fused, ctx.confidence)
        top = top_view(f"{out}_top.png", np.asarray(ctx.views_cloud.points), np.asarray(ctx.views_cloud.colors))
        trace = ctx.trace
        motion = motion_plot(f"{out}_motion.png", trace.times, trace.scores, ctx.still_threshold, ctx.segments,
                             set(ctx.used_segments))
        write_json(f"{out}.json", ctx.report)
        info(f"wrote {out}.ply, {out}_views.ply" + (f", {out}_confidence.ply" if wrote_confidence else "")
             + f", {out}.json" + (f", {out}_top.png" if top else "") + (f", {out}_motion.png" if motion else ""))


class InPlacePipeline(Pipeline):
    """Fuse an in-place capture of a person into one cloud (see the module docstring)."""
    config_class = InPlaceConfig
    name = "fuse-inplace"

    def steps(self) -> StepList:
        return StepList([LoadRecording(), SceneFromBackground(), LocatePerson(), FindStillPeriods(), BuildKeyframes(),
                         RegisterKeyframes(), CorrectKeyframes(), FuseKeyframes(), WriteOutputs()])
