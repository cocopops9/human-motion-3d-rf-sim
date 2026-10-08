"""Turntable pipeline: a person standing still on the rotating platform, a fixed LiDAR.

Every frame is the same body rotated by the platform angle theta(t) about the
platform axis. Steps:

    LoadRecording       the run directory (frames, background, capture.json)
    SceneFromBackground floor frame and background from the empty-scene frames
    FindSubject         the object on the platform, found in the frames with no region of
                        the room assumed: foreground objects followed across frames,
                        selected by the tests of [select] (default: turning about a
                        vertical axis by most of a lap; --human adds the person cascade);
                        its rotation axis and the region it sweeps (detection.finder)
    LocatePlatform      start of the axis fit: [platform] center if given, else the axis
                        found, refined by the platform ring when one is found near it
    IsolatePerson       person mask in every frame, frame times on the sensor clock, usable frames
    MeasureAngles       platform angle of every frame (motion model and model-free solution)
    SelectViews         one view every view_step degrees
    BuildViews          frames turned back by their angle about the axis (per column time)
    CorrectViews        axis polish, sway, slabs, arms
    FuseSurface         support filter, voxel averaging, robust surface fit, confidence
    WriteOutputs        <out>.ply, <out>_confidence.ply, <out>_views.ply, <out>.json, <out>_angle.png

Output frame: origin on the platform axis at the platform top (z = 0 where the
feet stand), axes of the first frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from bodyscan.config import section
from bodyscan.fusion import (FusionConfig, LimbConfig, LimbCorrection, SlabConfig, SlabCorrection, SurfaceFusion,
                             SwayCorrection, ViewAngleSearch, ViewConfig, axis_error_from_shifts, build_views,
                             reference_groups, select_view_frames)
from bodyscan.geometry import yaw_transform
from bodyscan.io import write_confidence
from bodyscan.jsonio import write_json
from bodyscan.log import Progress, info, warning
from bodyscan.motion import (MotionConfig, PlatformMotionEstimator, Samples, chained_angles, consistent_frame_times,
                             constant_speed_fit, lap_times)
from bodyscan.pipelines.base import Context, Pipeline, Step, StepList
from bodyscan.detection.finder import SelectConfig
from bodyscan.pipelines.common import FindSubject, FrameConfig, LoadRecording, SceneFromBackground, SubjectConfig
from bodyscan.plots import angle_plot
from bodyscan.registration import solve_turns
from bodyscan.scene import (CylinderRegion, FloorConfig, ForegroundIsolator, IsolationConfig, PlatformConfig,
                            RingPlatform, frame_complete)

@dataclass
class TurntableConfig:
    select: SelectConfig = section(SelectConfig, rotation=True)
    subject: SubjectConfig = section(SubjectConfig, min_total_turn=320.0)
    frames: FrameConfig = section(FrameConfig)
    floor: FloorConfig = section(FloorConfig)
    platform: PlatformConfig = section(PlatformConfig)
    isolation: IsolationConfig = section(IsolationConfig)
    motion: MotionConfig = section(MotionConfig)
    views: ViewConfig = section(ViewConfig)
    slabs: SlabConfig = section(SlabConfig)
    limbs: LimbConfig = section(LimbConfig)
    fusion: FusionConfig = section(FusionConfig)


# ----------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------

class LoadTurntableRecording(LoadRecording):
    """The run directory, and the commanded turn from the capture log (or [motion] turn_deg)."""

    def run(self, ctx: Context) -> None:
        super().run(ctx)
        c = ctx.config
        commanded = c.motion.turn_deg if c.motion.turn_deg is not None else ctx.source.capture.get("turn_deg")
        ctx.commanded = None if commanded is None else float(commanded)
        ctx.report["commanded_turn_deg"] = ctx.commanded


class FindTurntableSubject(FindSubject):
    """The object on the platform. With [platform] center given, the object
    near that point; with the region radius given as well, nothing is searched."""

    def near(self, ctx: Context):
        center = ctx.config.platform.center
        return None if center is None else np.array(center, dtype=np.float64)

    def run(self, ctx: Context) -> None:
        c = ctx.config
        if c.platform.center is not None and c.isolation.radius is not None:
            ctx.subject = None
            ctx.center = np.array(c.platform.center, dtype=np.float64)
            info(f"centre and region given: ({ctx.center[0]:.3f}, {ctx.center[1]:.3f}) m, radius "
                 f"{c.isolation.radius:g} m")
            return
        super().run(ctx)


class LocatePlatform(Step):
    """Start of the axis fit: [platform] center if given; else the centre of
    the object found (its rotation axis, or its body centre), replaced by the
    centre of the platform ring when a ring of [platform] ring_radius is found
    within ring_search of it in the empty scene (an optional refinement:
    without a ring, the joint axis fit of MeasureAngles still measures the
    axis to a few mm)."""
    name = "platform"

    def __init__(self, detector=None):
        self.detector = detector

    def run(self, ctx: Context) -> None:
        p = ctx.config.platform
        given = p.center is not None
        center = np.array(p.center, dtype=np.float64) if given else np.asarray(ctx.center, dtype=np.float64)
        ring = None
        if p.ring_radius > 0:
            detector = self.detector or RingPlatform(p.ring_radius, p.ring_search)
            ring = detector.ring(ctx.background_world, center) if given else detector.find(ctx.background_world,
                                                                                             center)
        if ring is not None:
            info(f"platform ring in this background: centre ({ring.center[0]:.3f}, {ring.center[1]:.3f}) m, "
                 f"radius {ring.radius:.3f} m, rms {1000 * ring.rms:.0f} mm, {ring.inliers} points")
            offset = float(np.linalg.norm(ring.center - center))
            if not given and ring.plausible and offset <= p.ring_search:
                info(f"  start of the axis fit: the ring centre, {1000 * offset:.0f} mm from the object's centre")
                center = np.asarray(ring.center, dtype=np.float64)
            elif offset > 0.10:
                warning(f"the ring is {100 * offset:.0f} cm from the centre used ({center[0]:.3f}, "
                        f"{center[1]:.3f}): did the platform or the sensor move?")
        elif p.ring_radius > 0:
            info(f"no platform ring found; the axis fit starts at ({center[0]:.3f}, {center[1]:.3f}) m")
        ctx.center = center
        ctx.report["platform_ring"] = None if ring is None else {
            "center": ring.center, "radius": ring.radius, "rms_m": ring.rms, "plausible": ring.plausible}
        ctx.report["axis_start"] = center


class IsolatePerson(Step):
    """Person mask of every frame, its sensor time, lost packets, usable frames."""
    name = "isolate"

    def run(self, ctx: Context) -> None:
        c = ctx.config
        iso = c.isolation
        radius = FindSubject.region_radius(ctx, ctx.center)
        info(f"region of interest: radius {radius:.2f} m about ({ctx.center[0]:.3f}, {ctx.center[1]:.3f}) m, "
             f"{iso.min_height:g} to {iso.max_height:g} m above the floor")
        ctx.report["region_radius_m"] = radius
        region = CylinderRegion(ctx.center, radius, iso.min_height, iso.max_height)
        ctx.isolator = ForegroundIsolator(ctx.source.sensor, ctx.background, ctx.floor, region, iso.edge_jump,
                                          iso.cluster_eps, iso.normal_radius)
        source = ctx.source
        info(f"frames: {len(source)}; reading ...")
        person_times, sweep_times, host_times, phases, counts = [], [], [], [], []
        progress = Progress("frames", len(source))
        for n in range(len(source)):
            frame = source.load(n)
            mask, _ = ctx.isolator.mask(frame.range_m)
            person_time, sweep_time = np.nan, np.nan
            if frame.timestamps is not None:
                valid = frame.timestamps[frame.timestamps > 0]
                if valid.size:
                    sweep_time = float(np.median(valid))
                if mask.any():
                    # mean over the person's pixels: continuous even across the start of the sweep
                    stamps = np.broadcast_to(frame.timestamps[None, :], mask.shape)[mask]
                    stamps = stamps[stamps > 0]
                    if stamps.size:
                        person_time = float(np.mean(stamps))
            person_times.append(person_time)
            sweep_times.append(sweep_time)
            host_times.append(frame.host_time)
            complete = frame_complete(mask, frame.columns_ok, frame.timestamps, iso.min_columns, iso.max_person_loss)
            phases.append(frame.phase if complete else -1)
            counts.append(int(mask.sum()))
            if (n + 1) % 200 == 0 or n == len(source) - 1:
                progress.step(n + 1)
        phases, counts = np.array(phases), np.array(counts)
        times, repaired = consistent_frame_times(np.array(person_times), np.array(sweep_times), np.array(host_times))
        if repaired:
            info(f"  frame times: {repaired} frame(s) without a usable sensor timestamp of the person "
                 f"(empty platform, lost packets) put on the sensor clock")
        ctx.time_origin = float(times[0])
        ctx.times = times - ctx.time_origin
        ctx.unlabelled = c.motion.ignore_phases or not np.any(phases == 2)
        if ctx.unlabelled:
            info("  platform driven from another PC (no phase labels): the still parts and the turn "
                 "are found in the data")
            phases[phases >= 0] = 2
        ctx.phases = phases
        ctx.usable = (phases > 0) & (counts > 0.3 * np.median(counts[counts > 0]))
        info(f"  person pixels per frame: median {int(np.median(counts))}; usable frames {int(ctx.usable.sum())}")
        ctx.report.update({"frame_times_s": ctx.times, "time_origin_s": ctx.time_origin,
                           "frames_dropped_lost_packets": int(np.sum(phases == -1))})


class MeasureAngles(Step):
    """Platform angle of every frame."""
    name = "angles"

    def __init__(self, estimator_class=PlatformMotionEstimator):
        self.estimator_class = estimator_class

    def run(self, ctx: Context) -> None:
        c = ctx.config
        m = c.motion
        times, phases, usable = ctx.times, ctx.phases, ctx.usable
        frame_period = np.median(np.diff(times))
        stride = max(1, int(round(m.sample_seconds / frame_period)),
                     int(np.ceil(usable.sum() / max(m.max_samples, 10))))
        sampled = [k for k in range(0, len(times), stride) if usable[k]]
        info(f"registration samples: {len(sampled)}, one every {stride * frame_period:.2f} s")
        load = ctx.source.load
        sample_clouds = [ctx.isolator.cloud(load(k).range_m).cloud for k in sampled]
        samples = Samples(sample_clouds, m.reg_voxel, m.fine_distance)
        angles, _, measurements = chained_angles(samples, times[sampled], phases[sampled], ctx.center, m.huber_deg)
        axis = ctx.center.copy()
        sense = np.sign(angles[-1] - angles[0]) or 1.0
        sample_phases = phases[sampled].copy()
        if ctx.unlabelled:
            # Still before the turn: within 3 deg of the start; still after:
            # within 3 deg of the end. The chained angles underestimate the turn
            # by a few per cent; the motion model below measures it.
            signed = np.degrees(angles * sense)
            moving = np.flatnonzero((signed > 3.0) & (signed < signed[-1] - 3.0))
            if len(moving) >= 4:
                sample_phases[:moving[0]] = 1
                sample_phases[moving[-1] + 1:] = 3
                sample_phases[moving[0]:moving[-1] + 1] = 2
                info(f"  turn from about {times[sampled][moving[0]]:.1f} s to {times[sampled][moving[-1]]:.1f} s "
                     f"(chained estimate {signed[-1]:.0f} deg)")
                if not np.any(sample_phases == 3):
                    warning("no still frames after the turn: record longer")
            else:
                warning("no turn found in the recording")
        fit = constant_speed_fit(times[sampled], angles * sense, sample_phases)
        if fit is not None:
            info(f"constant-speed model on the chained angles: {fit['deg_per_s']:.2f} deg/s "
                 f"({360 / fit['deg_per_s']:.1f} s per lap)")

        profile = None
        if m.angle_source in ("auto", "profile", "free"):
            estimator = self.estimator_class(m)

            def load_cloud(k):
                return ctx.isolator.cloud(load(k).range_m).cloud.voxel_down_sample(m.reg_voxel)

            profile = estimator.estimate(samples, times[sampled], sample_phases, angles, axis, ctx.commanded,
                                         frame_times=times, usable=usable, local_pairs=measurements,
                                         load_cloud=load_cloud)
        use_profile = profile is not None and m.angle_source != "pairs"
        profile_only, alternatives = None, {}
        if profile is not None:
            axis = np.asarray(profile["axis"])
            info(f"  axis from the joint fit: ({axis[0]:.4f}, {axis[1]:.4f}) m, "
                 f"{1000 * np.linalg.norm(axis - ctx.center):.1f} mm from the start")
            info(f"  motion: {profile['model'].describe()}")
            info(f"  long-pair residual rms {profile['pair_rms_deg']:.2f} deg ({profile['pairs']} long pairs, "
                 f"{profile['revisits']} revisit pairs over {profile['laps_with_revisits']} laps"
                 + (f", revisit residual rms {profile['revisit_residual_rms_deg']:.2f} deg"
                    if profile["revisit_residual_rms_deg"] is not None else "")
                 + f"); systematic deviation {profile['systematic_deviation_deg']:.2f} deg")
            info(f"  registration scale {100 * profile['registration_scale']:+.2f} % "
                 + ("(calibrated by the revisit pairs)" if profile["scale_calibrated"]
                    else "(not calibrated: no revisit pairs; the angles may be short by 1 to 3 %)"))
            if ctx.commanded is not None:
                info(f"  total turn of the motor model {profile['fitted_total_deg']:.2f} deg, commanded "
                     f"{ctx.commanded:g} deg" + (" (commanded value used)" if profile["total_fixed_to_command"]
                                                 else ""))
        if use_profile and profile["systematic_deviation_deg"] > m.profile_tolerance:
            warning(f"the pairs deviate from the angles by up to {profile['systematic_deviation_deg']:.1f} deg "
                    f"after smoothing (some pairs are wrong, or the person moved). Look at the angle plot")
        if use_profile:
            frame_angles = profile["sense"] * profile["model"].angle(times)
            frame_angles -= frame_angles[sampled[0]]
            info(f"  angles: {profile['label']}")
            motor = profile["motor_model"]
            profile_only = profile["sense"] * motor.profile(times)
            profile_only -= profile_only[sampled[0]]
            for label, other in (("motor model with correction", motor), ("model-free solution",
                                                                           profile["free_model"])):
                if other is not profile["model"]:
                    angles_other = profile["sense"] * other.angle(times)
                    alternatives[label] = angles_other - angles_other[sampled[0]]
        else:
            # Per-pair solve, with the revisit pairs if the model found them.
            reference = angles
            extra = []
            if profile is not None:
                reference = profile["sense"] * profile["model"].angle(times[sampled])
                reference = reference - reference[0]
                extra = profile["revisit_pairs"]
            solved = solve_turns(len(sampled), reference, measurements + extra, np.radians(m.huber_deg))
            frame_angles = np.interp(times, times[sampled], solved)
            info("  angles from the per-pair solve")
        total = float(np.degrees(abs(frame_angles[usable].max() - frame_angles[usable].min())))
        laps = total / 360.0
        info(f"platform: total turn {total:.1f} deg = {laps:.2f} laps")
        measured_laps = lap_times(times[usable], frame_angles[usable] * sense)
        if measured_laps:
            info("  time of every lap: " + ", ".join(f"lap {lap} {duration:.1f} s ({360.0 / duration:.2f} deg/s)"
                                                     for lap, duration in measured_laps))
        if total < 300.0:
            info(f"  note: less than one lap; the cloud covers about {total + 120:.0f} deg of the body "
                 f"(the sensor sees roughly 120 deg at a time)")
        ctx.axis, ctx.sense, ctx.frame_angles = axis, sense, frame_angles
        ctx.model_angles = frame_angles.copy()
        ctx.total, ctx.laps = total, laps
        ctx.angle_plot_data = dict(sampled=sampled, chained=angles * sense, profile_only=profile_only,
                              alternatives=alternatives, label=profile["label"] if use_profile else "per-pair solve")
        ctx.report.update({
            "axis_center": axis, "total_turn_deg": total, "laps": laps, "constant_speed_fit": fit,
            "sampled_frames": sampled, "sample_times_s": times[sampled],
            "sample_angles_deg": np.degrees(angles * sense),
            "profile": None if profile is None else {k: v for k, v in profile.items()
                                                     if k not in ("model", "motor_model", "free_model",
                                                                  "revisit_pairs")},
            "angles_from_profile": bool(use_profile), "frame_angles_deg": np.degrees(frame_angles),
            "lap_times_s": [duration for _, duration in measured_laps]})


class SelectViews(Step):
    name = "views"

    def run(self, ctx: Context) -> None:
        v = ctx.config.views
        candidates = np.flatnonzero(ctx.usable)
        total, laps = ctx.total, ctx.laps
        if v.view_range is not None:
            low, high = sorted(v.view_range)
            turned = np.degrees(ctx.frame_angles[candidates]) * ctx.sense
            candidates = candidates[(turned >= low) & (turned <= high)]
            if len(candidates) == 0:
                raise SystemExit(f"--view-range {low:g} {high:g}: no frame in this part of the turn "
                                 f"(total turn {total:.1f} deg)")
            covered = float(np.degrees(np.ptp(ctx.frame_angles[candidates])))
            info(f"views restricted to {low:g} to {high:g} deg of turn: {covered:.1f} deg covered")
            if covered < 300.0:
                info("  note: less than one lap in this range; parts of the body are seen from one side only")
            total, laps = covered, covered / 360.0
        step = max(v.view_step, total / max(v.max_views, 1))
        view_frames = select_view_frames(candidates, ctx.frame_angles, ctx.sense, step)
        info(f"views: {len(view_frames)} frames, one every {step:.2f} deg of turn"
             + (f" (about {len(view_frames) / laps:.0f} per lap)" if laps >= 1.5 else ""))
        ctx.view_frames, ctx.view_step, ctx.view_laps = view_frames, step, laps
        ctx.groups = reference_groups(len(view_frames), v.reference_groups)
        # Reference budget: one lap of views (minus one group), whatever the number of laps.
        ctx.reference_views = min(len(view_frames), 360.0 / step) * (1.0 - 1.0 / max(v.reference_groups, 1))
        ctx.report.update({"view_frames": view_frames, "view_step_deg": step, "view_range_deg": v.view_range})


class BuildViews(Step):
    name = "build views"

    def run(self, ctx: Context) -> None:
        ctx.clouds = build_views(ctx.source, ctx.isolator, ctx.view_frames, ctx.frame_angles, ctx.axis, ctx.times,
                                 ctx.time_origin)


class CorrectViews(Step):
    """Optional per-view angle search, axis polish from the pattern of the
    sway corrections, sway correction (views out of bounds dropped), height
    slabs, arms. 'extra' corrections (ViewCorrection objects) run at the end."""
    name = "corrections"

    def __init__(self, extra=()):
        self.extra = list(extra)

    def run(self, ctx: Context) -> None:
        c = ctx.config
        v = c.views
        clouds, view_frames, groups = ctx.clouds, ctx.view_frames, ctx.groups
        frame_angles, axis = ctx.frame_angles, ctx.axis
        view_correction_record = []
        if v.angle_iterations > 0:
            clouds, result = ViewAngleSearch(v, np.append(axis, 0.0)).apply(clouds, groups, ctx.reference_views)
            view_turns = result["deltas"]
            frame_angles = frame_angles.copy()
            frame_angles[view_frames] -= view_turns
            view_correction_record = [[int(k), float(np.degrees(d))] for k, d in zip(view_frames, view_turns)]
            deviation = np.degrees(view_turns)
            info(f"platform angle searched view by view: deviation from the motion model rms "
                 f"{np.sqrt(np.mean(deviation ** 2)):.2f} deg, max {np.abs(deviation).max():.2f} deg")
        # Axis refinement: correct the views against each other and read the
        # axis error from the pattern of the corrections (a wrong axis
        # displaces the views along a circle as they turn; sway does not
        # follow the turn).
        for iteration in range(v.axis_iterations):
            _, result = SwayCorrection(v, f"axis check {iteration + 1}").apply(clouds, groups, ctx.reference_views)
            error = axis_error_from_shifts(result["shifts"], frame_angles[view_frames])
            if error is None:
                info("  axis polish skipped (the views do not go round the body)")
                break
            old_pivot, axis = np.append(axis, 0.0), axis + error
            info(f"  axis corrected by ({1000 * error[0]:+.1f}, {1000 * error[1]:+.1f}) mm -> "
                 f"({axis[0]:.4f}, {axis[1]:.4f}) m")
            for cloud, k in zip(clouds, view_frames):                   # re-turn about the corrected axis
                cloud.transform(yaw_transform(frame_angles[k], old_pivot))
                cloud.transform(yaw_transform(-frame_angles[k], np.append(axis, 0.0)))
            if np.linalg.norm(error) < 0.0005:
                break

        clouds, result = SwayCorrection(v).apply(clouds, groups, ctx.reference_views)
        sway_rms = result["rms_mm"]
        info(f"sway correction per view: RMS {sway_rms:.1f} mm")
        # A view whose correction is out of bounds does not fit the others: a
        # wrong angle (the platform lost steps), or the person moved more than
        # the bounds. Fused as it is, it would blur the surface, so it is
        # dropped, unless that would remove too many views (then the angles
        # themselves are suspect).
        outliers = ~np.isfinite(result["shifts"][:, 0])
        dropped_views = []
        if outliers.any():
            fraction = outliers.mean()
            if fraction <= v.max_dropped_views:
                dropped_views = [int(view_frames[k]) for k in np.flatnonzero(outliers)]
                keep_views = np.flatnonzero(~outliers)
                clouds = [clouds[k] for k in keep_views]
                view_frames = [view_frames[k] for k in keep_views]
                groups = None if groups is None else groups[keep_views]
                info(f"  {outliers.sum()} of {len(outliers)} views dropped: their correction was out of bounds "
                     f"(turn > {v.max_turn_correction:g} deg, shift > {100 * v.max_shift:g} cm or "
                     f"tilt > {v.max_tilt:g} deg)")
            else:
                warning(f"{100 * fraction:.0f}% of the views could not be corrected within the bounds; they are "
                        f"kept. The platform angles are probably off (speed changes, lost steps): look at the "
                        f"angle plot and try --angle-source pairs")
        if c.slabs.slab_iterations > 0:
            clouds, result = SlabCorrection(c.slabs, clean_reference=True).apply(clouds, groups, ctx.reference_views)
            info(f"height-slab correction: RMS {', '.join(f'{h:.1f}' for h in result['rms_mm_per_pass'])} mm")
        limb_report = None
        if c.limbs.limb_iterations > 0:
            clouds, limb_report = LimbCorrection(c.limbs).apply(clouds, groups, ctx.reference_views)
        for correction in self.extra:
            clouds, extra_report = correction.apply(clouds, groups, ctx.reference_views)
            ctx.report.setdefault("extra_corrections", {})[correction.name] = extra_report
        ctx.clouds, ctx.view_frames, ctx.groups = clouds, view_frames, groups
        ctx.frame_angles, ctx.axis = frame_angles, axis
        ctx.report.update({
            "axis_center": axis, "sway_correction_rms_mm": sway_rms,
            "dropped_view_frames": dropped_views, "view_angle_correction_deg": view_correction_record,
            "model_frame_angles_deg": np.degrees(ctx.model_angles),
            "arm_correction": None if limb_report is None else {
                "shoulder_heights_m": limb_report["tops"],
                "hand_move_median_mm": [{str(side): (float(1000 * np.nanmedian(moves)) if np.isfinite(moves).any()
                                                     else None) for side, moves in p.items()}
                                        for p in limb_report["hand_moves"]]}})


class FuseSurface(Step):
    name = "fusion"

    def run(self, ctx: Context) -> None:
        c = ctx.config
        min_views = c.fusion.min_views * max(1, int(np.floor(ctx.view_laps + 0.1)))
        fused, views, removed, confidence = SurfaceFusion(c.fusion).fuse(ctx.clouds, min_views, ctx.reference_views)
        output_from_world = np.eye(4)
        output_from_world[:3, 3] = [-ctx.axis[0], -ctx.axis[1], -c.platform.platform_top]
        fused.transform(output_from_world)
        views.transform(output_from_world)
        ctx.fused, ctx.views_cloud, ctx.confidence = fused, views, confidence
        ctx.output_from_world = output_from_world
        spread, per_point = confidence["spread_mm"], confidence["points"]
        standard_error = 1.25 * spread / np.sqrt(np.maximum(per_point, 1))   # of a median; spread includes curvature
        top = float(np.asarray(fused.points)[:, 2].max())
        info(f"support filter (at least {min_views} views) removed {100 * removed:.1f}% of the points")
        info(f"fused: {len(fused.points)} points, highest point {top:.3f} m above the platform top")
        info(f"within {1000 * c.fusion.confidence_radius:.0f} mm of an output point: median "
             f"{np.median(confidence['views']):.0f} views, {np.median(per_point):.0f} points; "
             f"spread along the normal median {np.median(spread):.1f} mm, "
             f"p90 {np.percentile(spread, 90):.1f} mm; standard error of the fitted surface median "
             f"{np.median(standard_error):.2f} mm")
        ctx.report.update({
            "support_filter_min_views": min_views, "support_filter_removed_fraction": removed,
            "confidence": {"views_median": float(np.median(confidence["views"])),
                           "points_median": float(np.median(per_point)),
                           "spread_median_mm": float(np.median(spread)),
                           "spread_p90_mm": float(np.percentile(spread, 90)),
                           "standard_error_median_mm": float(np.median(standard_error))},
            "output_from_world": output_from_world, "highest_point_m": top})


class WriteOutputs(Step):
    name = "write"

    def run(self, ctx: Context) -> None:
        out = str(ctx.out)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(f"{out}.ply", ctx.fused)
        o3d.io.write_point_cloud(f"{out}_views.ply", ctx.views_cloud)
        wrote_confidence = write_confidence(f"{out}_confidence.ply", ctx.fused, ctx.confidence)
        plot = ctx.angle_plot_data
        sense = ctx.sense
        measured = (ctx.times[ctx.view_frames], ctx.frame_angles[ctx.view_frames] * sense) \
            if ctx.config.views.angle_iterations > 0 else None
        plotted = angle_plot(f"{out}_angle.png", ctx.times[plot["sampled"]], plot["chained"], ctx.times,
                             ctx.model_angles * sense, plot["label"], measured,
                             None if plot["profile_only"] is None else plot["profile_only"] * sense,
                             {label: angles * sense for label, angles in plot["alternatives"].items()},
                             usable=ctx.usable)
        write_json(f"{out}.json", ctx.report)
        info(f"wrote {out}.ply, {out}_views.ply" + (f", {out}_confidence.ply" if wrote_confidence else "")
             + f", {out}.json" + (f", {out}_angle.png" if plotted else ""))


class TurntablePipeline(Pipeline):
    """Fuse a turntable capture of a person into one cloud (see the module docstring)."""
    config_class = TurntableConfig
    name = "fuse-turntable"

    def steps(self) -> StepList:
        return StepList([LoadTurntableRecording(), SceneFromBackground(), FindTurntableSubject(), LocatePlatform(),
                         IsolatePerson(),
                         MeasureAngles(),
                         SelectViews(), BuildViews(), CorrectViews(), FuseSurface(), WriteOutputs()])
