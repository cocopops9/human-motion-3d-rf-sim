"""Pipelines: an ordered list of steps sharing one context.

A Step does one thing (estimate the floor, measure the angles, fuse, ...)
and reads and writes named entries of the Context. A Pipeline lists its
steps; a variant of a setup is a subclass that replaces, removes or inserts
steps, without copying the others:

    class EncoderTurntable(TurntablePipeline):
        def steps(self):
            steps = super().steps()
            steps[steps.index_of(MeasureAngles)] = AnglesFromEncoder()
            return steps

Every step adds what it measured to ctx.report, which is written as the
JSON report of the run together with the full configuration.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from types import SimpleNamespace

from bodyscan import __version__
from bodyscan.config import to_dict
from bodyscan.log import info


class Context(SimpleNamespace):
    """Shared state of a pipeline run: inputs, intermediate results, report."""

    def require(self, *names):
        missing = [n for n in names if not hasattr(self, n)]
        if missing:
            raise RuntimeError(f"pipeline step needs {', '.join(missing)} from an earlier step")
        return [getattr(self, n) for n in names]


class Step(ABC):
    """One stage of a pipeline."""

    name = "step"

    @abstractmethod
    def run(self, ctx: Context) -> None:
        """Read inputs from ctx, write results to ctx (and ctx.report)."""


class StepList(list):
    """List of steps with lookup by class, for subclasses that edit it."""

    def index_of(self, step_class) -> int:
        for k, step in enumerate(self):
            if isinstance(step, step_class):
                return k
        raise ValueError(f"no {step_class.__name__} in the pipeline")


class Pipeline(ABC):
    """Assembles steps for one setup; config_class declares its parameters."""

    config_class: type = None
    name = "pipeline"

    def __init__(self, config=None):
        self.config = config if config is not None else self.config_class()

    @abstractmethod
    def steps(self) -> StepList:
        """The steps, in order."""

    def run(self, **inputs) -> Context:
        ctx = Context(config=self.config, report={"pipeline": self.name, "bodyscan_version": __version__,
                                                  "config": to_dict(self.config)}, **inputs)
        start = time.time()
        for step in self.steps():
            step.run(ctx)
        ctx.report["elapsed_s"] = round(time.time() - start, 1)
        info(f"{self.name}: done in {ctx.report['elapsed_s']:.0f} s")
        return ctx
