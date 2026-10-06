# Changelog

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
