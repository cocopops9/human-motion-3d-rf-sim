# Tuning guide: how the parameters change the results

Every parameter is listed in [parameters.md](parameters.md). This guide
explains the ones that matter, starting from what you see in the outputs. The
effects quoted were measured: on the real tt16 cloud for the mesh, on the
synthetic turntable run rsE (with a known truth) for the fusion, and on the
lab recordings tt13, tt14 and tt15 for the detector.

**See the parameters act.** The [Algorithm Lab](https://raw.githack.com/cocopops9/human-motion-3d-rf-sim/main/lidar-body-scan/docs/lab/index.html) runs the algorithms
on slices of a real scan in the browser; each parameter below has a slider
there:

| Parameter | Live demo |
|---|---|
| `bg_threshold`, `bg_relative`, `radius`, `edge_jump`, `cluster_eps` | [find the person](https://raw.githack.com/cocopops9/human-motion-3d-rf-sim/main/lidar-body-scan/docs/lab/index.html#background) |
| axis error, long pairs, motor model | [platform angle](https://raw.githack.com/cocopops9/human-motion-3d-rf-sim/main/lidar-body-scan/docs/lab/index.html#angles) |
| `min_views`, `support_radius`, `voxel`, `confidence_radius` | [fusion](https://raw.githack.com/cocopops9/human-motion-3d-rf-sim/main/lidar-body-scan/docs/lab/index.html#fusion) |
| `depth` (Poisson) | [points to surface](https://raw.githack.com/cocopops9/human-motion-3d-rf-sim/main/lidar-body-scan/docs/lab/index.html#poisson) |
| `watertight` (grid) | [closed mesh](https://raw.githack.com/cocopops9/human-motion-3d-rf-sim/main/lidar-body-scan/docs/lab/index.html#watertight) |
| `scale_mm`, `rounds`, `normal_sigma`, `finish_rounds`, `max_deviation_mm`, `keep_volume` | [smoothing](https://raw.githack.com/cocopops9/human-motion-3d-rf-sim/main/lidar-body-scan/docs/lab/index.html#smoothing) (step by step, reflection lines in 3D) |
| `max_axis_spread`, `max_speed_spread`, `score_threshold` (detect) | [detection](https://raw.githack.com/cocopops9/human-motion-3d-rf-sim/main/lidar-body-scan/docs/lab/index.html#detection) |

The 2D slices show the trend of each parameter; the numbers in this guide
are the measured 3D results.

## 0. Read the outputs before changing anything

| Output | What to look at |
|---|---|
| console | the sensor height and tilt, the platform centre, warnings (ring not found, views dropped, weak alignments) |
| `<out>_angle.png` (fuse) | platform angle against time: the curves must agree; the residual panel must stay within a few degrees |
| `<out>_views.ply` | every view in its own colour: a doubled body means wrong angles or a wrong axis; a part present in a few views only means the person moved |
| `<out>_confidence.ply` | per point: `views`, `points`, `spread_mm`; colour green (0 mm spread) to red (5 mm and more) |
| `<out>.json` | everything measured and the complete configuration (the version of the code is `bodyscan_version`) |
| `<out>_quality.json` (mesh, smooth) | facet noise, angles between triangles, bump height against λ/8, distance to the cloud; for smooth also the distance from the input mesh and the settings |
| `<out>_top.png` (detect, fuse-inplace) | where the objects are; a correct in-place fusion is one closed ring per slice |

## 1. Your results of 4 October

| Run | What you saw | Cause | What to change |
|---|---|---|---|
| tt13b | the fusion is "completely messed up" | very likely fused with a version older than `fuse_turntable.py` 2026-10-02c: frames without the person (after stepping off) took the host clock, the frame times went backwards and almost every view got the angle 0 (the half-moon). A synthetic recording with 120 such frames at the end, two stalls of the platform and a recording that ends before the turn, fuses correctly with 2026-10-02d (angles 1.4 deg rms, cloud to truth median 2.0 mm) and with the package (2.2 deg rms, 2.3 mm; it also warns that the end of the turn is missing). Check `version` in `person_tt13b.json` | fuse again with `python -m bodyscan fuse` |
| tt13b | the feet are missing | field of view: lens at 1.20 m, axis at about 1.1 m; the feet need 49 deg below the horizon and the beams reach 45 | move the sensor back, or lower it (docs/hardware.md, section 4) |
| tt14 | thumb separated, head cropped | the thumb gap (30 to 50 mm) is above the resolution limit (about 11 to 17 mm at 1 m); the sensor was tilted down to bring the feet in, which pushed the head out | same as above: no tilt can fit head and feet at this distance and height |
| tt16 | hands, part of the feet and part of the head cropped | hands: the person region had a radius of 0.55 m (now 0.9 m, `--radius`); head and feet: the field of view, as above | `--radius 0.9` (default now); sensor position |

Nothing in the processing restores a part that the beams never reached. Run
`python -m bodyscan check-view` on a 20 s capture after every change of the
sensor position: it prints the heights reached by the highest and the lowest
beams at the person and the tilt that fits.

## 2. Symptom → parameters

### Parts of the body missing from the fused cloud

| Symptom | Parameter | Effect |
|---|---|---|
| hands or arms cut at a vertical line | `--radius` (0.9 m) | the person region around the platform centre; it must include the hands. Larger takes in more of the room, which the empty-scene test removes |
| the lowest centimetres of the shoes missing | `--min-height` (0.05 m), `--bg-threshold` (0.05 m) | points below `min-height` are dropped; a pixel must be closer than the empty scene by `bg-threshold`, so the edge of the sole within 5 cm of the platform is lost. 0.03 and 0.03 keep more of the shoes and more platform noise |
| thin parts (fingers, hands far from the body) thinned or missing | `--min-views` (3 per lap), `--support-radius` (0.02 m) | a point needs neighbours from this many views; thin parts are seen by fewer views. Lower `min-views` keeps them, and more ghosts |
| a hand separated from the body disappears | `--cluster-eps` (0.06 m) | only the largest cluster of every frame is kept; a hand separated by more than this from the arm is dropped |
| many frames ignored | `--min-columns`, `--max-person-loss` | frames with lost packets on the person are not used; the console reports how many |

### The fused cloud is wrong (doubled, rotated copies, smeared)

| Symptom | Parameter | Effect |
|---|---|---|
| the curves of `<out>_angle.png` disagree, or the residuals exceed a few degrees | `--angle-source` (`auto`) | `free`: model-free only (best when the motor lost steps); `profile`: motor model only; `pairs`: the old per-pair solve |
| the total turn is wrong | `--turn-deg` | the commanded turn is used when the data agree within `--loop-tolerance` (3 deg) |
| a doubled body shifted sideways | `--center X Y` | impose the platform centre (from `python -m bodyscan detect C:\lidar\tt17 --rotation --human`); the axis fit starts there |
| two-PC recording, phases mislabelled | `--ignore-phases` | the still parts and the turn are found in the data |
| many views dropped ("correction out of bounds") | `--max-turn-correction` (3 deg), `--max-shift` (0.03 m), `--max-tilt` (2 deg) | bounds of the sway correction; a dropped view did not fit the others (wrong angle, or the person moved). If more than `--max-dropped-views` (25 %) would be dropped, they are kept and the angles are suspect |
| arms blurred or thinner than in a single view | `--limb-iterations` (2), `--limb-max-shift` (0.06 m) | the per-arm correction; 0 turns it off |
| head or hips doubled | `--slab-iterations` (1), `--slab-max-turn` (5 deg), `--slab-max-shift` (0.03 m) | the per-height correction |

### The surface of the cloud is noisy

| Parameter | Effect |
|---|---|
| more laps (`--turn-deg 1440` or more at capture) | more views per patch: the surface fit becomes more precise |
| `--confidence-radius` (0.01 m) | neighbourhood of the surface fit: larger is smoother and flattens details smaller than the radius |
| `--surface-fit` | off: the noisier voxel averages |
| `--voxel` (0.005 m) | spacing of the output points |

Measured on the synthetic turntable run rsE (one lap, 1 cm range noise; distance from
the fused cloud to the true body):

| Setting | Points | Distance to the truth median / p90 / p99 [mm] |
|---|---|---|
| defaults (surface fit within 10 mm) | 98 900 | 2.1 / 6.1 / 11.3 |
| `--confidence-radius 0.006` | 127 300 | 3.8 / 10.1 / 15.8 |
| `--confidence-radius 0.015` | 82 500 | 1.5 / 4.7 / 10.1 |
| `--no-surface-fit` (voxel averages) | 193 000 | 5.5 / 13.4 / 19.2 |
| `--slab-iterations 0 --limb-iterations 0` | 104 000 | 2.4 / 6.9 / 13.1 |
| `--min-views 6` | 98 700 | 2.1 / 6.1 / 11.4 |
| `--radius 0.55` | 98 500 | 2.2 / 6.1 / 11.2 |

The surface fit is the largest effect: without it the cloud keeps the range
noise (5.5 mm median). A larger fit radius is closer to the truth here only
because the synthetic body has no detail smaller than 15 mm; on a real body it
flattens the nose, the ears and the fingers. The synthetic arms stay within
0.55 m of the axis, so the radius changes nothing here; on tt16 it cut the
hands. The angles are the same in every run (1.3 deg rms, axis within
0.4 mm): these parameters act after them.

### The mesh is not smooth enough for ray tracing

Conversion and smoothing are two commands: `mesh` turns the cloud into a
closed mesh and does not smooth it; `smooth` smooths a closed mesh, with
every parameter of the smoothing set directly. Sionna RT reflects every ray
on the plane of the triangle it hits, so the numbers that matter are the
facet normal noise (the random tilt of the triangles against the mean normal
within 1 cm; a reflected ray turns by twice the tilt) and the bump height
(Rayleigh: below λ/8 = 0.62 mm at 60 GHz the surface reflects like a
mirror). The price of smoothing is the shape change: `smooth` reports the
distance of every vertex from its input (median, 99th percentile, maximum),
and the distance from the fused cloud with `--cloud`.

#### What each parameter of `smooth` does

| Parameter | Default | What it controls | Measured effect |
|---|---|---|---|
| `--scale-mm` | 12 | the size of what is removed: every facet averages the facets within twice this distance | tt11, 4 rounds: 6 mm removes the facet noise and keeps the scan-row stripes; 12 mm removes most stripes; 16 to 24 mm also flattens clothing folds, the knees and the face (table below) |
| `--rounds` | 4 | the strength: how many times the filter is applied | tt11 at 12 mm: 1, 4, 8 rounds move the surface by 0.42, 0.83, 1.21 mm median; the stripes fade progressively |
| `--finish-rounds`, `--finish-scale-mm` | 2, 6 | rounds at a fine scale after the main ones (only if finer than `--scale-mm`) | remove the small dimples and, above about 16 mm, the fine ripples a wide filter leaves (tt11 at 24 mm: facet noise p90 5.7 to 1.6 deg); about 0.1 mm more median shape change |
| `--normal-sigma` | 1.5 | how different two facet normals may be and still be averaged (0.35: about 20 deg; 1.5: nearly isotropic) | 0.35 keeps creases (fingers, chin, folds) but turns scan defects into sharp creases and leaves flat patches (tt11: bump height 0.18 instead of 0.06 mm) |
| `--max-deviation-mm` | 0 (off) | a bound on the distance from the input surface | below about 3 mm on tt11 it also stops the smoothing of the stripes (they need 2 to 7 mm): see "Why the levels looked alike"; a tight limit can even raise the facet noise tail where vertices are held at the band edge (tt11, 1 mm, 1 round at 12 mm: p90 3.77 to 3.91 deg, against 3.59 without the limit) |
| `--keep-volume` | on | a uniform offset along the normals restores the input volume after every round | off: thin parts (arms, fingers) shrink a little at every round |
| `--normal-iterations` | 4 | normal filtering passes inside one round | more spread the average farther in one round |
| `--vertex-iterations` | 0 (automatic) | vertex update passes inside one round: 15 on 3.6 mm edges, times (3.6 / edge)², times (scale / 12 mm)² above 12 mm | too few: the vertices do not follow the filtered normals (24 mm with 15 passes: dimpled, facet noise p90 4.9 deg) |
| `--relax-iterations` | 5 | tangential relaxation before the first round | better shaped triangles; the surface does not move |
| `--sweep NAME=V1,V2,...` | | one mesh per value, `<out>_<name><value>.ply`, and a comparison table | for choosing the values |

#### Measured on your `person_tt11_mesh.ply`

The input was made with the smoothing that `mesh` applied up to version 1.0
(6 mm, normal sigma 0.35), so it is already lightly smoothed. Facet noise
in degrees, distances in mm; render in
`tt11_smooth_parameters.png` (flat shading, as the ray tracer sees it).

| `smooth` options | Facet noise median / p90 | Bump height rms | Distance from the input median / p99 / max |
|---|---|---|---|
| (input) | 1.28 / 3.77 | 0.147 | 0 / 0 / 0 |
| `--rounds 1` | 0.79 / 2.13 | 0.033 | 0.42 / 2.82 / 5.80 |
| defaults (`--scale-mm 12 --rounds 4`) | 0.62 / 1.72 | 0.030 | 0.83 / 4.93 / 8.59 |
| `--rounds 8` | 0.52 / 1.54 | 0.040 | 1.21 / 6.93 / 12.05 |
| `--scale-mm 6` | 0.81 / 2.16 | 0.032 | 0.39 / 2.80 / 5.60 |
| `--scale-mm 16` | 0.50 / 1.54 | 0.039 | 1.31 / 7.17 / 12.50 |
| `--scale-mm 24` | 0.41 / 1.61 | 0.057 | 2.24 / 11.56 / 19.00 |

Where the largest changes are (defaults): 2.3 % of the vertices move by more
than 4 mm; 28 % of those are on the rim of the flat soles (the 90 deg edge is
rounded), 26 % on the arms and hands, and the rest is spread over the legs
and the torso, where the lumps and stripes were (the intended change). With
`--rounds 8` it is 7.7 % of the vertices (20 %, 20 %, 60 %). To keep the
edges and the hands, bound the change (`--max-deviation-mm`) or protect
creases with a lower `--normal-sigma` (0.35 to 0.5; not measured on these
edges), at the cost of the bumps next to them.

#### Measured on tt16 (fused cloud, through `mesh` then `smooth`)

| Mesh | Facet noise median / p90 | Bump height rms | Distance from the input median / p99 / max | Distance from the cloud median |
|---|---|---|---|---|
| `mesh` (defaults, no smoothing) | 3.59 / 13.07 | 0.479 | 0 / 0 / 0 | 0.40 |
| `smooth --config configs/smooth_previous_mesh.toml` (what `mesh` did up to 1.0) | 1.42 / 4.74 | 0.407 | 0.22 / 1.35 / 3.16 | 0.68 |
| `smooth --rounds 1` | 0.89 / 2.68 | 0.306 | 0.60 / 3.45 / 6.95 | 1.10 |
| `smooth` (defaults) | 0.64 / 1.98 | 0.307 | 1.02 / 5.35 / 9.71 | 1.56 |
| `smooth --rounds 8` | 0.54 / 1.69 | 0.303 | 1.37 / 7.04 / 12.25 | 1.94 |

Conclusions:

- `--rounds` and `--scale-mm` are the two levers you see. Rounds make the
  same kind of bump fainter; the scale decides which bumps count as noise.
  Both move the surface farther from the measurement: on tt16 the median
  distance from the cloud goes from 0.40 mm (no smoothing) to 1.6 mm
  (defaults), below the 4.4 mm spread of the fused cloud along the normal,
  so the change stays within what the measurement can tell apart. That is
  an argument, not a proof: whether a 1 to 2 mm smoother surface changes
  the ray-traced field enough to matter has to be checked in Sionna.
- The bump height of tt16 stays at 0.30 mm whatever the rounds: what is
  left is detail at the 1 to 2 cm scale (clothing folds, the hands, the
  face), which the quadric of the measure does not absorb. It is below
  λ/8 = 0.62 mm at 60 GHz already after one round, and above λ/32 = 0.16 mm
  in every row.
- The facet noise keeps falling with the rounds; past about 4 rounds the
  gain is small and the shape change keeps growing.
- `configs/smooth_previous_mesh.toml` reproduces the meshes made before the
  conversion and the smoothing were separated (tt16: 1.42 / 4.74 deg here,
  1.42 / 4.69 deg with version 1.0; Poisson is not bit-reproducible).

#### Why the five levels of version 1.0 looked alike

Their hashes and metrics differed (facet noise p90 3.2 deg at level 1,
2.0 deg at level 5), but level 1 and level 5 were 0.65 mm apart in the
median: invisible at the scale of a body. Every level held the surface
within 0.5 to 3 mm of the input, while the visible lumps (scan-row stripes,
leg bumps) need 2 to 7 mm of movement to go, so the limit, not the level,
set the result. The levels are gone; the limit is off by default and
reported instead.

#### Conversion parameters (`mesh`)

`mesh` has no smoothing filter, but the Poisson octree (`--depth` 9, cells of
about 4 mm on a person) and the watertight grid (`--watertight`, 4 mm) set
the finest detail kept. Measured on tt16, each followed by
`smooth --config configs/smooth_previous_mesh.toml` to compare with the
earlier tables:

| `mesh` options | Triangles | Facet noise median / p90 [deg] | Bump height rms [mm] | Distance from the cloud median / p90 [mm] |
|---|---|---|---|---|
| defaults (4 mm grid) | 396 270 | 1.42 / 4.69 | 0.41 | 0.68 / 4.08 |
| `--watertight 0.006` | 175 380 | 1.55 / 4.91 | 0.36 | 0.72 / 4.20 |
| `--watertight 0.003 --depth 10` | 702 712 | 1.35 / 4.53 | 0.39 | 0.67 / 4.12 |
| `--watertight 0.002 --depth 10` | 1 588 702 | 1.29 / 4.62 | 0.41 | 0.66 / 4.07 |
| `configs/mesh_budget_140k.toml` (decimation to 6 mm) | 136 386 | 1.54 / 7.59 | 0.61 | 0.77 / 4.18 |

- A finer grid is not rougher once the vertex passes follow the grid
  (automatic): the 2 mm grid keeps gaps down to about 3 mm and costs 4 times
  the triangles, 5 minutes and 2.7 GB of memory
  (`configs/mesh_detail_2mm.toml`).
- For a triangle budget, prefer a coarser grid to decimation: decimation to
  6 mm edges leaves folds at the creases (90th percentile 7.6 deg), the 6 mm
  grid does not.

Earlier measurements of the smoothing parameters at one round (tt16, 4 mm
grid; smoothing then inside `mesh`): `--scale-mm` 4, 6, 8, 10 gave facet
noise p90 5.05, 4.69, 4.95, 5.36 deg; `--normal-sigma` 0.2 and 0.6 gave
7.50 and 4.19 deg; `--normal-iterations` 2 and 8 gave 5.56 and 4.55 deg;
`--vertex-iterations` 25 and 40 gave 4.21 and 3.93 deg, at 0.75 and 0.81 mm
from the cloud. With one round and a crease-keeping normal sigma, a larger
scale blurred creases more than it removed noise; with several rounds and
the nearly isotropic default, the larger scale wins (tables above).

A correction to the numbers given before 5 October: the smoothing and the
quality measures then used at most the 64 nearest triangles instead of all
the triangles within the radius. On the 4 mm grid this changed little, but on
finer grids the filter shrank (so `--scale-mm` did almost nothing) and the
measures read too low. Both now use whole radius neighbourhoods; the tables
above are measured with them.

### Fingers merged

See [hardware.md](hardware.md): with the sensor upright and the whole body in
view, the thumb can be separated, individual fingers only marginally. The
processing defaults fill gaps narrower than about 1 cm (fusion) and 6 mm
(watertight grid), and `smooth` rounds them (a 12 mm scale treats a finger
as a bump; `--normal-sigma 0.35` or `--max-deviation-mm` protects them). The experimental
`configs/turntable_fingers.toml` (fusion) and `configs/mesh_detail_2mm.toml`
(mesh) keep narrower gaps; the finer fusion is noisier everywhere.

## 3. Detection

| Symptom | Parameter | Effect |
|---|---|---|
| a person is missed | `--human-min-height` (1.2 m), `--min-area` (0.20 m²), `--min-torso-width` (0.18 m) | lower accepts children, seated people and partly hidden people, and more stands and poles |
| furniture reported as a person | `--score-threshold` (0.6), `--curved-threshold` (0.45), `--max-flat-fraction` (0.65) | higher threshold, lower curved threshold: fewer false detections, more missed people |
| a person is split into several objects | `--cluster-distance` (0.10 m) | larger joins the parts, and also a person with an object they touch |
| a rotating object is missed | `--max-axis-spread` (0.10 m), `--max-speed-spread` (0.35), `--min-speed` (0.5 deg/s) | looser accepts noisier rotations, and also some walking people |
| the decision is unstable | `--max-frames` (60) | more frames, spread over the recording, give steadier decisions |
| a person farther than about 6 m is missed | `--curved-threshold`, `--max-flat-fraction` | at 6 m the beams are 7.4 cm apart, the normals are estimated over larger patches and the body looks flatter; higher values accept it, and more furniture |

Measured values (people against objects) behind the defaults, on tt13, tt14,
tt15 and on a real body mesh ray-cast from 1 to 6.5 m:

| Measure | People (lab recordings) | People (body mesh, 1 to 6.5 m, every side) | Objects (lab recordings) |
|---|---|---|---|
| silhouette area | 0.47 to 0.69 m² | 0.40 to 0.70 m² | stands and poles 0.11 to 0.15 m² |
| solid torso width | 0.32 to 0.53 m | 0.32 to 0.45 m | poles, stands, a desk, a coat stand 0.09 to 0.18 m |
| flat fraction (normals within 10 deg of 3 directions) | 0.20 to 0.23 | 0.20 to 0.43 (rises with the distance) | boxes, cabinets, desks 0.48 to 0.73 |
| shape score | 0.75 to 0.94 | 0.57 to 0.95 (one view of 72 below 0.6) | 0.11 to 0.41 |

## 4. In-place capture

| Symptom | Parameter | Effect |
|---|---|---|
| "fewer than 3 still periods" | `--still-max` (0.25), `--still-factor` (2.5), `--min-still` (5 frames) | how still a stop must be; look at `<out>_motion.png` |
| a wrong turn at one step (low overlap in the step table) | `--max-step` (90 deg), `--hypothesis-step` (10 deg) | the yaw search between stops |
| slow registration | `--keyframe-stride 2` | half the keyframes, about a quarter of the time |
| head and feet do not align | `--max-tilt` (5 deg) | the lean allowed between stops; 0 cannot align head and feet together |
