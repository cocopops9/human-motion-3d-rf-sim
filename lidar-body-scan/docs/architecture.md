# Code architecture

The single-file scripts of the first weeks (`legacy/`) grew to 2300 lines
each, with copied helpers and options defined in three places. The package
splits them by responsibility, so that a new setup reuses everything except
the part that changes.

## Layers

```
bodyscan/
    config.py        parameter declarations: options, files, documentation
    log.py, jsonio.py, plots.py
    geometry/        transforms, fitting (planes, circles), neighbours, clustering
    io/              recordings on disk, sensor models, PLY writers
    scene/           floor, background, platform, regions, isolation, stillness, coverage
    registration/    ICP variants, turn solver, keyframe registration
    motion/          platform angle against time: models, fitting, pairs, estimator
    fusion/          views, per-view corrections, surface fusion
    meshing/         reconstruction, cleanup, watertight remesh, smoothing, quality
    detection/       segmentation, tracking, rotation test, person cascade
    capture/         sensor access, run directory, phases, protocols, motor
    pipelines/       the steps assembled for one setup
    synthetic.py     synthetic scenes and recordings
    body/            body model in the SMPL-X format: skeleton, rotations, skinning, test body
    dynamic/         moving people: motions, simulation, segmentation, avatar, tracking,
                     evaluation, export, review, test bench
    commands/        the command line
```

A layer uses only the layers above it in this list (geometry uses nothing of
the package; pipelines use everything above them; commands only build
configurations and run pipelines or the functions of `dynamic`). Only
`capture.sensor` and `capture.diagnostics` import the Ouster SDK, and only
when a sensor or a recording is opened; only `capture.motor` imports
pyserial. The processing of static scans needs numpy and open3d.

PyTorch is needed only for moving people. `body.model` imports it; the
modules of `dynamic` import it inside the functions that need it, so the
command line (`--help`, `params`, `--write-config`), the capture and the
segmentation of moving people work without it.

## Patterns

### Parameters are declared once (`config.py`)

A configuration section is a dataclass whose fields are created with `param()`:

```python
@dataclass
class FusionConfig:
    """Fusion of the views: support filter, voxel averaging, robust surface fit."""
    voxel: float = param(0.005, "voxel of the fused cloud", unit="m",
                         effect="smaller: denser output, slower")
```

A pipeline configuration is a dataclass of sections. From that one
declaration come the command-line options (`--voxel`; when two sections
use the same name the section is added, as `--human-min-height` and
`--segmentation-min-height` of `detect`), the TOML files (`[fusion]`,
`voxel = 0.005`), `--set fusion.voxel=0.004`, `--write-config`, and the
parameter reference (`python -m bodyscan params`). The documentation cannot
drift away from the code. `section(FusionConfig, min_views=2)` reuses a
section with other defaults in another pipeline.

### Pipelines are lists of steps (`pipelines.base`)

```python
class Step(ABC):
    name = "step"
    def run(self, ctx: Context) -> None: ...      # reads and writes named entries of ctx

class TurntablePipeline(Pipeline):
    config_class = TurntableConfig
    def steps(self) -> StepList:
        return StepList([LoadTurntableRecording(), SceneFromBackground(), LocatePlatform(), IsolatePerson(),
                         MeasureAngles(), SelectViews(), BuildViews(), CorrectViews(), FuseSurface(),
                         WriteOutputs()])
```

Every step adds what it measured to `ctx.report`, which is written as the
JSON report with the full configuration. Steps shared by several pipelines
live in `pipelines.common` (loading, the scene from the empty frames, locating
the person).

### Strategies behind small interfaces

| Interface | Implementations | Used by |
|---|---|---|
| `io.FrameSource` | `NpzRecording`, `PointCloudFolder` | every pipeline |
| `io.SensorModel` | `LutSensorModel` | range-image sources |
| `scene.FloorEstimator` | `RansacFloor`, `CropBoxFloor` | `SceneFromBackground` |
| `scene.PlatformDetector` | `RingPlatform` | `LocatePlatform` |
| `scene.Region` | `CylinderRegion`, `SensorBoxRegion` | `ForegroundIsolator` |
| `motion.MotionModel` | `StepperMotion`, `FreeMotion`, `MeanMotion` | `PlatformMotionEstimator` |
| `fusion.ViewCorrection` | `SwayCorrection`, `ViewAngleSearch`, `SlabCorrection`, `LimbCorrection` | `CorrectViews`, `CorrectKeyframes` |
| `meshing.Mesher` | `PoissonMesher`, `BallPivotingMesher`, `AlphaMesher` (and `GridMesher`) | `Reconstruct` |
| `detection.Segmenter` | `BackgroundSegmenter`, `ObjectSegmenter` | `FindScene` |
| `detection.Stage` | `SizeStage`, `ColumnStage`, `SilhouetteStage`, `SurfaceStage`, `ShapeStage` | `HumanCascade` |
| `detection.Feature` | `RectangleContrast`, `WidthRatio`, `Centered` (Haar-like), `CurvedSurface` | `ShapeStage` |
| `capture.Phase` | `Background`, `Countdown`, `Hold`, `Stepping`, `MotorTurn`, `Record` | `Session` |

### Moving people (`body`, `dynamic`)

A motion is a `dynamic.motions.PoseSequence`: times, the world root rotation
and pelvis position, and the 21 SMPL-X body joint rotations (axis-angle in the
parent frames; hands optional). Everything passes motions in this form: the
procedural motions, AMASS files, the synthetic truth, the tracker output, the
export. `sample(times)` resamples it with C1 continuity, which is what keeps
the exported velocities free of steps.

A body is a `body.model.ShapedBody`: the model (`BodyModel`, any SMPL-X
format npz) with one person's shape coefficients, at a subdivision level,
with optional displacements along the rest normals (the detail of the scan).
`pose(rotations, root_rotation, root_position)` poses it in the world frame
of bodyscan (z up); `dynamic.avatar.Avatar` stores what is needed to rebuild
the body of one person.

The tracker (`dynamic.track.Tracker`) is an energy minimised with L-BFGS:
`frame_energy` holds the terms of one frame (data, silhouette and free
space, in front of the measured surface, floor), `fit_frame` adds the joint
limits, the twist prior and the continuity to the prediction, `refine` adds
over a window the smoothness of the joint motion (jerk), the no-sliding of
standing feet and the rolling shutter; `recover_limbs` refits a frame from
other starts when points stay unexplained. A new term goes into one of these
functions with its weight in `TrackConfig` or `RefineConfig`; a new starting
pose of the search into `LEG_STARTS` or `ARM_STARTS`.

### Commands register themselves (`commands.base`)

A subclass of `Command` with a `name` is a command; `PipelineCommand` adds the
input path, `--out` and every option of the pipeline configuration;
`ConfiguredCommand` does the same for commands that are not pipelines (the
captures). The module must be listed in `commands/__init__._MODULES`.

## Extending

### A variant of a setup: replace one step

A platform with an angle encoder: the angles come from the encoder log, every
other step stays.

```python
from bodyscan.pipelines.base import Step
from bodyscan.pipelines.turntable import MeasureAngles, TurntablePipeline

class AnglesFromEncoder(Step):
    name = "angles"
    def run(self, ctx):
        ...   # read the encoder log, set ctx.frame_angles, ctx.axis, ctx.sense, ctx.total, ctx.laps,
              # ctx.model_angles and ctx.angle_plot_data, as MeasureAngles does

class EncoderTurntable(TurntablePipeline):
    def steps(self):
        steps = super().steps()
        steps[steps.index_of(MeasureAngles)] = AnglesFromEncoder()
        return steps
```

### Another correction of the views

Subclass `fusion.ViewCorrection` (`apply(clouds, groups, reference_views)`
returns the corrected clouds and a report) and pass it to the step:
`CorrectViews(extra=[MyLegCorrection(config)])`.

### Another test in the person cascade

A new feature of the shape stage:

```python
from bodyscan.detection.human import Feature, HumanCascade, ShapeStage, default_features, rise
from bodyscan.pipelines.detect import Classify, DetectPipeline

class TallEnough(Feature):
    name = "tall enough"
    def score(self, view):
        return rise(view.top, 1.5, 0.05)          # 0.5 at 1.5 m

def my_cascade(config, sampling):
    cascade = HumanCascade(config, sampling)
    cascade.stages[-1] = ShapeStage(config, default_features(config) + [TallEnough()])
    return cascade

class MyDetect(DetectPipeline):
    def steps(self):
        steps = super().steps()
        steps[steps.index_of(Classify)] = Classify(cascade_factory=my_cascade)
        return steps
```

A new hard stage is a `Stage` subclass (`evaluate(view)` returns a
`StageResult`); `BodyView` computes its measurements on demand, so a stage
that needs normals or the silhouette pays for them only if the cheaper stages
passed.

### Another capture protocol

A protocol is a list of phases; a new behaviour is a `Phase` subclass
(`begin`, `update`, `finished`, `end`). For example a capture that turns the
platform by 90 deg steps and holds still in between:

```python
phases = [Background(3, "platform empty"), Countdown(15, "step on")]
for k in range(4):
    phases += [Hold(3, 1, "still"), MotorTurn(motor, [7200], 120, "turning 90 deg")]
```

### Another sensor or file format

A `FrameSource` subclass (`__len__`, `load(index) -> Frame`, optionally
`load_background`), and for range images a `SensorModel` (`xyz(range)`,
`directions`). `open_recording` chooses the source from the directory
content.

### Another body model or motion source

Any npz with the SMPL-X keys and the SMPL-X kinematic tree (55 joints) works
as a model (`body.model` checks the tree); a model with pose blend shapes
uses them. Another motion source only has to produce a `PoseSequence`
(`motions.load_amass` is an example); `simulate-motion` accepts it as
`--motion file.npz`.

### A new parameter

Add a `param()` field to the section; it appears in `--help`, in the
configuration files, in `docs/parameters.md` (`python -m bodyscan params --out
docs/parameters.md`) and in the JSON report of every run.

## Tests

`tests/` uses unittest (no extra dependency). `synthetic.py` renders scenes
with known truth: the detection tests check the axes of two rotating
platforms and the two people of a scene; the pipeline tests (slow, behind
`BODYSCAN_SLOW_TESTS=1`) fuse a synthetic turntable run and a synthetic
in-place run and compare with the truth; the capture tests run the protocols
on a fake sensor; the meshing tests check closedness, orientation and
smoothing on spheres. `test_motion.py` checks the conventions of the body
model (the sign of every documented joint rotation, subdivision, skinning),
the procedural motions (feet neither through the floor nor sliding, a
ballistic flight), the simulated recording and its segmentation against the
pixel labels, the export (velocities against differences of the meshes),
and, slow, the avatar fit and the tracking of a short walk; it needs PyTorch
and is skipped without it.
