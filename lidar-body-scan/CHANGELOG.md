# Changelog

## Documentation (2026-10-08)

- `docs/lab/index.html`, the Algorithm Lab: the static pipeline as live 2D
  simulations on slices of the real scan tt9 (background subtraction on a
  real beam row; ICP about the axis against free ICP; axis error; stepper
  model fitted to the measured angles; views turned back; support filter,
  voxel averaging and median surface fit; Poisson and marching squares;
  signed distance, ray parity and gap sealing; bilateral normal smoothing
  with reflected rays and facet-tilt metrics).
- docs/algorithms.md: an animation recorded from the Lab in every section,
  with a link to the live demo; docs/tuning.md: parameter to demo table.
- The facet tilt from vertex noise is √2 σ / L (rms), not 2 σ / L: corrected
  in docs/algorithms.md and in the docstring of `meshing.smoothing`.

## Documentation (2026-10-07)

- Story animation and tour: the wait on the platform before the motor starts
  (t = 21 to 39 s) is shown, the dial uses the measured platform angle, the
  tour shows the countdown card and has a GIF button with the 2D animation.

- README: `docs/img/lab_to_turntable.gif`, one recording from the empty lab
  to the fused cloud (run tt9: the person walking from the PC, the flight to
  the platform, one lap, the fusion). The interactive tour now starts with the
  same story, with animated camera flights between the steps, and uses the
  lab of run tt9.

- README: a visual tour from a real lab frame (`docs/img/lab_scan.gif`: the
  beam sweep and the same frame as a 128 x 2048 image), the static pipeline
  step by step (`pipeline_overview.png`, `turntable_fusion.gif` from the raw
  frames of recording tt9 and the views of its fusion person_tt11, `cloud_mesh_smooth.gif`), and an
  Architecture section. Usage, results and the rest are unchanged.
- `docs/viewer/index.html`: interactive 3D tour (three.js, data embedded):
  the lab frame, the turntable recording, the fusion view by view, the mesh
  and the smoothed mesh side by side.

## 1.2.0 (2026-10-06)

### New: moving people

Animated meshes of a person moving around the LiDAR, for Sionna RT, and the
person's body tracks. The motion comes only from the LiDAR frames; the body
model fills what one sensor cannot see. Workflow, recording protocol,
algorithms, accuracy and limits: [docs/motion.md](docs/motion.md).

- `capture-motion`: empty room, countdown, then the person moving (until
  ENTER or `--duration`); warns when the sensor is not in the 1024 x 20 mode.
- `segment-motion`: the person in every frame (points, pixels, the time of
  every point, a silhouette crop) against the empty room; pixels without a
  return where the empty room always returns count as the person (dark
  fabric).
- `avatar`: the SMPL-X body model fitted to the person's turntable scan
  (shape, pose), plus the scan's detail as displacements on a subdivided
  surface.
- `track`: the avatar fitted to every frame, coarse to fine (points to the
  visible surface, silhouette and free space, joint limits, floor,
  continuity), with a search from other starts for legs and arms when points
  remain unexplained (arms swung overhead, knees folding at a landing); then
  the whole sequence refined (smooth joint motion by a penalty on the jerk,
  standing feet that do not slide, the time of every point and pixel); parts
  the sensor hardly sees are kept smooth; frames that need a look are
  flagged.
- `review-motion`: the tracked body over the LiDAR points, as a GIF and PNG
  pictures, for human checks.
- `export-motion`: one PLY mesh per time step at any rate, per-vertex
  velocities (derivative of the smooth motion), per-part velocities, the
  joints as CSV (body tracks); optionally one mesh per body part.
- `simulate-motion`, `evaluate-motion`, `bench-motion`, `make-test-body`:
  synthetic recordings with truth (rolling shutter, beam footprint, mixed
  pixels, dropouts; walks and countermovement jumps from footprints and leg
  inverse kinematics), error metrics (joints, part velocities and their
  Doppler equivalent, accelerations, sliding feet), a test bench over
  motions, sensor modes and distances, and a procedural test body in the
  SMPL-X file format for tests without the SMPL-X files. On the bench, at
  1024 x 20 and 2 m: joints within 13 mm, body-part velocities within 0.11
  (walk) and 0.14 m/s (jump); at 2048 x 10 the jumps are three to four times
  worse, so moving people are recorded at 1024 x 20.
- The capture stores the pixel shift of every row in `lut.npz` (the time of
  every pixel of a moving person).

### Requirements of the new commands

PyTorch (`python -m pip install torch`, the CUDA build on a PC with an NVIDIA
GPU), Pillow for the GIF (`pip install bodyscan[motion]` installs both), and
the SMPL-X model files: register at https://smpl-x.is.tue.mpg.de and keep the
files out of the repository (the licence does not allow redistribution). The
capture and `segment-motion` need neither.

## 1.1.0 (2026-10-05)

### Changed

- Conversion and smoothing are separate: `mesh` converts the cloud into a
  closed mesh and no longer smooths it (the `[smoothing]` section of `mesh`
  is gone; `frequency_ghz` moved to `[quality]`, the decimation
  `target_edge_mm` to `[closing]`). `smooth` does all the smoothing.
- `smooth` takes the smoothing parameters directly: `--scale-mm` (12),
  `--rounds` (4), `--finish-rounds` and `--finish-scale-mm` (2 at 6 mm),
  `--normal-sigma` (1.5), `--normal-iterations`, `--vertex-iterations`,
  `--relax-iterations`, `--keep-volume`, `--max-deviation-mm` (0 = off).
  `--sweep NAME=V1,V2,...` writes one mesh per value and a comparison table.
  The distance from the input is always measured and reported.
- The `--level` presets are removed. They held the surface within 0.5 to
  3 mm of the input, while the visible lumps of tt11 need 2 to 7 mm, so the
  five levels differed by 0.65 mm median and looked the same.
- The automatic vertex passes also grow as (scale / 12 mm)² above 12 mm
  (at 24 mm, 15 passes left the surface dimpled).
- `configs/mesh_smoother.toml` removed (superseded by `smooth`);
  `configs/smooth_previous_mesh.toml` reproduces the smoothing of `mesh` 1.0;
  `configs/mesh_sionna_60ghz.toml` renamed `mesh_default.toml`;
  `configs/smooth_default.toml` added. The parameter reference now includes
  `smooth`.

To get the mesh of version 1.0 from version 1.1:

    python -m bodyscan mesh person.ply --out person_mesh.ply
    python -m bodyscan smooth person_mesh.ply --config configs/smooth_previous_mesh.toml --out person_mesh10.ply

## 1.0.0 (2026-10-05)

First version of the `bodyscan` package, replacing the scripts in `legacy/`.

### New

- `detect`: rotating objects (`--rotation`), people (`--human`), rotating
  people (both), with their centres. Person test: a cascade of hand-written
  shape tests in the spirit of a Haar cascade (size, vertical continuity,
  silhouette area and solid torso, flat panels, then Haar-like rectangle
  features on an integral image of the silhouette and a curved-surface
  feature), decided over all the frames of a tracked object.
- `fuse-inplace`: the in-place pipeline in the new structure; the person is
  found automatically; surface fit and confidence as on the turntable.
- `fuse` finds the platform ring around the person found in the frames
  (`--search-start` no longer defaults to the 2026-09-30 calibration).
- `capture-turntable` and `capture-inplace`: the capture scripts as protocols
  of phases; they can replay a recording for testing.
- `check-view` also prints the sample spacing and the smallest visible gap at
  the person; `convert`, `simulate`, `params`, `quality`, `check-sensor`.
- Configuration files (TOML or JSON), `--set`, `--write-config`; the parameter
  reference is generated from the code (`docs/parameters.md`).
- Tests (unittest) on synthetic scenes; MATLAB launcher `matlab/bodyscan.m`.

- `smooth`: a smoother version of an existing mesh, the shape kept within a
  distance limit of the input surface (`--level 1` to `5`, limit 0.5 to 3 mm),
  volume restored; on tt11 facet noise p90 3.8 deg to 2.0 to 3.2 deg, bump
  height 0.15 mm to 0.05 to 0.08 mm.

### Changed

- Mesh: no decimation by default. On tt16: facet normal noise p90 4.7 deg
  instead of 7.6 deg, bump height rms 0.41 mm instead of 0.61 mm, about
  400 000 triangles instead of 140 000 (`configs/mesh_budget_140k.toml`
  restores the decimation).
- Mesh smoothing: `scale_mm` 6 mm by default (it gave the lowest 90th
  percentile of facet noise on tt16) instead of twice the wavelength;
  `vertex_iterations` 0 = automatic, 15 on the 4 mm grid and
  15 x (3.6 mm / edge)^2 on other grids, so that every grid is smoothed
  over the same distance. `configs/mesh_smoother.toml` (facet noise p90
  3.3 deg, bump 0.35 mm) and `configs/mesh_detail_2mm.toml` added.
- Isolation radius 0.9 m (0.55 m cut the hands of tt16).

### Fixed

- Mesh smoothing and quality measures used at most the 64 nearest triangles
  (48 vertices for the bump height) instead of all those within the radius.
  On fine grids the smoothing shrank and `--scale-mm` did almost nothing,
  and the measures read too low (the 2 mm mesh was reported as the
  smoothest, 2.9 deg p90, and is in fact the roughest with 15 vertex
  passes, 7.6 deg). Whole radius neighbourhoods now (`RadiusSearch`), with
  grouped sources on fine meshes; the smoothing of a 2 mm mesh needs 1.2 GB
  instead of 5.3 GB.
- Mesh: about one run in ten gave a mesh with a doubled, jagged surface (more
  triangles and area, bad facets). Cause: when Open3D could not orient the
  triangles consistently, the triangles flipped one by one after the cloud
  normals stayed inconsistent, and the inside test of the watertight remesh
  followed those normals. The winding is now made consistent breadth-first
  and turned per connected piece; the inside test uses ray parity whenever
  the triangle normals disagree with it.
- Watertight remesh: the normal of a triangle decides the side only when the
  closest point lies inside it (no spikes at edges and vertices); the winding
  is checked per connected piece; the far field is tested per 2 x 2 x 2 block
  in slabs (a 2 mm grid needed more memory than a PC has).
- Captures: the empty-scene phase starts with the first scan (connecting to
  the sensor and booting the motor no longer eat into it); ENTER is read only
  from a console; a motor command without 'T done' stops the capture instead
  of labelling the following frames 'still'; negative turns; disk errors of
  the writer are reported; the output folder is created only once the sensor
  answers.
- Configuration: allowed values are checked in files and `--set` too;
  `--write-config x.json` writes JSON; Python 3.9 can build the options.
- `mesh --out person_2.5mm` keeps the dots of the name.
- In place: consecutive keyframes are always registered, even after a step
  larger than `--loop-max-angle`.
- Frame times on one clock (from `fuse_turntable.py` 2026-10-02c): frames
  without the person no longer take the host clock (the half-moon fusion of
  tt13).
- Frames with lost packets are kept unless the loss hits the person
  (2026-10-02d).
