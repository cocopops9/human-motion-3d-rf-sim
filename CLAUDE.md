# CLAUDE.md: repository conventions

This repository is public. Keep this file free of credentials, private hostnames, account names and personal data. The owner's private context (machines, cluster, data paths, preferences) is in `C:\master_thesis\CLAUDE.md` on his laptop, outside the repository; Claude Code loads it automatically when started inside this repository on that laptop.

## What this repository is

Thesis monorepo for "3D Reconstruction of Human Motion for RF Sensing Simulations" (Università di Padova, IMDEA Networks). The main module is `lidar-body-scan`, the Python package `bodyscan`:

* static phase: LiDAR scans to closed, smoothed meshes of standing people;
* dynamic phase: personal SMPL-X avatars fitted to recordings of moving people, exported as animated meshes with per-vertex velocities for micro-Doppler simulation in NVIDIA Sionna RT.

## Current work

Read `lidar-body-scan/docs/multisensor-plan.md` first: the extension of bodyscan from one Ouster LiDAR to any combination of Ouster OS0-128, RoboSense Helios 16 and an ASUS Xtion depth camera, with new avatars, metrics, heatmaps and an interactive viewer (its section 16 gives the state). `lidar-body-scan/docs/evaluation.md` documents `evaluate-take`, `view` and `bench-rig` (1.7.0); `lidar-body-scan/docs/motion.md` documents the single-LiDAR tracker; `lidar-body-scan/docs/adr/` holds the design decisions.

## Layout

* `lidar-body-scan/bodyscan/`: layers geometry, io, sensors, scene, registration, motion, fusion, meshing, detection, capture, pipelines, body, dynamic, evaluation, viewer, commands. Each layer uses only the layers before it; keep it so.
* `lidar-body-scan/docs/`: README material, algorithms, architecture, hardware, tuning, motion, parameters (generated), Algorithm Lab, 3D viewer.
* `lidar-body-scan/tests/`: fast tests by default; slow tests behind an environment variable.
* `lidar-body-scan/configs/`: TOML configurations.
* `lidar-body-scan/models/`: body-model files, ignored by git.
* `lidar-body-scan/legacy/`, `lidar-body-scan/matlab/`: old scripts and the turntable MATLAB code.

## Commands

* Run: `python -m bodyscan <command>` from `lidar-body-scan`, or after `python -m pip install -e ".[capture]"`.
* Fast tests: `python -m pytest -q tests` (from `lidar-body-scan`; the motion tests need PyTorch and SciPy, the viewer test viser).
* Slow tests: set `BODYSCAN_SLOW_TESTS=1` (PowerShell: `$env:BODYSCAN_SLOW_TESTS = "1"`), then run pytest.
* Parameter reference: `python -m bodyscan params --out docs/parameters.md`.

## Rules

* Never commit body-model files (SMPL-X, SKEL, MHR assets), recorded data, motion results, or personal data. Licences differ and the files are large.
* Every release updates:
  * the version in `pyproject.toml`, `bodyscan/__init__.py` and the README badge;
  * `CHANGELOG.md`, with measured results, what was tried and not kept, and what is not validated;
  * the relevant docs, and `docs/parameters.md`.

  Fast tests and slow motion tests must pass before pushing to `main`.
* Synthetic comparisons are paired (same takes, same noise realisations) and use at least three noise realisations. One realisation can move the mean joint error by 13 to 34 %.
* Results label what was observed by a sensor, what was guessed and what was interpolated.
* Decisions that change an interface get an ADR in `lidar-body-scan/docs/adr/`.

## Code style

* Clear, organised, human-like code; readability before brevity.
* No emoticons in code, comments or program output.
* Type hints on public functions. Docstrings state what is computed, in which frame and in which units.
* Every parameter is declared once in a config dataclass with help text and its effect.

## Text style for docs and messages

* English, concise and formal.
* No "-" or "--" used as punctuation in prose: use commas, colons and semicolons.
* Numbers with units. Uncertainty stated as low, moderate or high confidence.

## Git

* Work on `main` of `cocopops9/human-motion-3d-rf-sim` unless a change is risky enough for a branch.
* Commit messages are short and factual and end with the Claude co-author line.
