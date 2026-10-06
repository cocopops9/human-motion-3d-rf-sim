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
    commands/        the command line
    synthetic.py     synthetic scenes and recordings
```

A layer uses only the layers above it in this list (geometry uses nothing of
the package; pipelines use everything below them; commands only build
configurations and run pipelines). Only `capture.sensor` and
`capture.diagnostics` import the Ouster SDK, and only when a sensor or a
recording is opened; only `capture.motor` imports pyserial. The processing
needs numpy and open3d.

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
smoothing on spheres.
