# Evaluating tracked motions: numbers, heatmaps, viewer, placement bench

bodyscan 1.7.0 adds three commands that replace the review videos with numbers and an interactive viewer, and a bench that compares sensor layouts before any tripod is moved:

| Command | What it answers |
|---|---|
| `evaluate-take` | How well does the tracked body agree with the measurements, per frame, per body part and per vertex? |
| `view` | What does the take look like: points, body, guessed limbs, heatmaps, at any frame? |
| `bench-rig` | How much of the body would each candidate layout of the Ouster, Helios and Xtion see? |

## 1. What can be measured without a reference

No MoCap is available, so on real data every error is a consistency with the measurements, never an accuracy. A body part that no sensor observed has no error value: it is drawn grey in the heatmaps and marked "not observed" in the tables, never green. Accuracy against a known truth comes only from the synthetic bench (`bench-motion`), and, with several sensors, from residuals against a sensor that the fit did not use (planned, see docs/multisensor-plan.md section 8.3).

## 2. evaluate-take

```
python -m bodyscan evaluate-take <take>\motion.npz --segments C:\lidar\open\<take>_seg
    --avatar C:\lidar\s02_avatar.npz --model SMPLX_MALE.npz --out C:\lidar\evaluation_v160\<take>
```

### 2.1 Pixel classes

For every pixel of the silhouette crop of segment-motion, the body is posed at that pixel's measurement time (the crop is split into blocks of 8 columns, 0.4 ms at 1024x20, so the rolling shutter is respected) and ray cast along the pixel's ray. With e = measured range minus body range and the tolerance max(3 sigma, 2 cm) (sigma: `--sigma`, 1 cm by default):

| Class | Condition | Meaning |
|---|---|---|
| fitted | person pixel, body hit, \|e\| within the tolerance | the body is where the sensor measured the person |
| violation | body hit and e above the tolerance (person or room behind) | the body stands where the sensor saw through: a misplaced or phantom part |
| unexplained | person pixel without body surface within the tolerance (no hit, or the body behind the measurement), or a dark pixel without body | a missing or misplaced part |
| silhouette | dark pixel (no return, light blocked next to the person) with the body on it | consistent, no range |
| unobserved | body hit, but the pixel says nothing (no return without the dark label, an object in front, a room surface touching the body such as the floor under a sole) | no information |

The shares reported are over the ranged pixels (fitted plus violation plus unexplained). "Frames with more than 5 % violations" corresponds to the "no data above 5 %" measure of the earlier video reviews.

### 2.2 Points

Every person point is measured against the nearest body surface: inliers (within the tolerance), outliers (beyond the tolerance and 5 cm) and clutter (beyond 25 cm). This nearest-surface distance understates a displacement along the ray (on the synthetic test of section 6, a body moved 8 cm towards the sensor gives a median point distance of only 27.7 mm, while 92.5 % of the pixels become violations), so the pixel classes are the sharper measure.

### 2.3 Per-vertex map and heatmaps

Every ranged pixel that hits the body adds its absolute range error (capped at 20 cm) to the three corners of the triangle it hits, weighted by the barycentric weights. Per frame a vertex gets the weighted mean of its pixels, or nothing. Over the take, per vertex: coverage (share of frames observed), median and p95 of the per-frame errors, violation and unexplained rates. Written as coloured meshes in the pose of the first evaluated frame (the start A-pose of a take), moved to the origin:

| File | Colour scale |
|---|---|
| heatmap_median.ply | dark green 0 mm, yellow 20 mm, red 50 mm and above; grey: observed in fewer than 5 % of the frames |
| heatmap_p95.ply | dark green 0 mm, yellow 32 mm, red 80 mm and above; grey as above |
| heatmap_violation.ply | dark green 0 %, red 50 % and above; grey as above |
| heatmap_coverage.ply | dark green: observed in every frame; red: never |

Open them in MeshLab, CloudCompare or Blender (vertex colours), or in the viewer.

### 2.4 Per-part table

For each body part (the 24 parts of skeleton.PART_NAMES): area-weighted coverage, and over the vertices observed in at least 5 % of the frames, the area-weighted median of the vertex medians and of the vertex p95, and the violation and unexplained rates.

### 2.5 Kinematics and plausibility

From the tracked joints and rotations: speed p95 and acceleration RMS per main joint; the share of the velocity power above 6 Hz (`--cutoff-hz`), a measure of jitter, which becomes false micro-Doppler in the radio simulation; frames with the root turning faster than 720 deg/s; elbows and knees out of their plane of flexion (beyond the ranges of bodyscan.body.anatomy plus 0.5 deg); joints faster than the anatomical speed limits; the share of frames with each tracker flag and with an arm more than 2 cm inside the trunk.

### 2.6 Output files

| File | Content |
|---|---|
| metrics.json | the summary: pixel classes, points, coverage, per-part table, kinematics, flags, settings |
| metrics.npz | per frame: frame index, time, class counts, point rates; per vertex: coverage, median, p95, violation and unexplained rates, part labels; faces and display pose; the per-frame per-vertex error in float16 millimetres (NaN: not observed), used by the viewer |
| heatmap_*.ply | section 2.3 |
| report.md | the summary as text, with the legend |

### 2.7 All takes of a series (PowerShell on the laptop)

```
$results = "C:\lidar\cluster_results_v160\results"
Get-ChildItem $results -Directory | Where-Object { Test-Path "$($_.FullName)\motion.npz" } | ForEach-Object {
    python -m bodyscan evaluate-take "$($_.FullName)\motion.npz" --segments "C:\lidar\open\$($_.Name)_seg" `
        --avatar C:\lidar\s02_avatar.npz --model SMPLX_MALE.npz --out "C:\lidar\evaluation_v160\$($_.Name)" --quiet
}
```

On the cluster the same command runs in a SLURM array (one task per take), as the tracking does.

## 3. view

```
python -m pip install viser
python -m bodyscan view --results C:\lidar\cluster_results_v160\results --segments C:\lidar\open
    --avatar C:\lidar\s02_avatar.npz --model SMPLX_MALE.npz --evaluations C:\lidar\evaluation_v160
```

Open http://localhost:8080. The panel holds:

* **Take**: the motion type (the take name without its repetition number) and the repetition, then Load.
* **Playback**: frame slider, Play, speed, previous and next flagged frame.
* **Show**: the person's points, coloured by sensor or by distance to the body (dark green 0, red 5 cm). The body is drawn plain, with guessed limbs in violet, or as a heatmap: this frame's error, the take's median or p95 error, violation rate or coverage. The skeleton and the sensor can also be shown.
* **Numbers**: the frame (time, flags, guessed limbs, pixel classes) and the take (evaluation summary, worst parts by p95).
* **Curves**: violation and unexplained shares per frame over the take.

The viewer needs an evaluation folder only for the heatmaps, the numbers and the curves; without it the points, the body and the guessed limbs are shown. On a cluster node, forward the port (`ssh -L 8080:<node>:8080 dauriagius@login.dei.unipd.it`) and open it locally.

## 4. bench-rig

```
python -m bodyscan bench-rig --out bench_rig
python -m bodyscan bench-rig --avatar C:\lidar\s02_avatar.npz --model SMPLX_MALE.npz --out bench_rig_s02
```

For every layout (bodyscan.sensors.layouts) and bench motion (walk across the line of sight of the Ouster, walk towards it, jumps facing it, jumps side-on), the body is posed at 10 Hz and, for every sensor, a vertex counts as visible when it is in the field of view and range limits, faces the sensor, and is not hidden by the body itself. The table gives the mean share of the body surface visible to at least one sensor; bench_rig.json holds, per part and per sensor, the visible share and the sample spacing on the visible surface.

First results with the procedural test body (1.76 m), zone centre 2.3 m from the Ouster, Helios at 3.6 m and 0.95 m high, Xtion at 0.95 m aimed at 0.9 m:

| layout | walk across | walk radial | jump facing | jump side-on |
|---|---|---|---|---|
| Ouster alone | 38 % | 46 % | 46 % | 31 % |
| Ouster + Xtion opposite (portrait, 2.2 m) | 57 % | 87 % | 89 % | 61 % |
| Ouster + Helios opposite | 70 % | 89 % | 90 % | 62 % |
| triangle 120 deg (Xtion portrait, 2.2 m) | 85 % | 90 % | 94 % | 91 % |
| Xtion opposite, Helios at the side (portrait, 2.2 m) | 81 % | 91 % | 92 % | 79 % |
| triangle 120 deg (Xtion landscape, 2.5 m) | 91 % | 91 % | 94 % | 91 % |
| Xtion opposite, Helios at the side (landscape, 2.5 m) | 90 % | 91 % | 92 % | 79 % |

Reading: only the 120 deg triangle keeps about 91 % of the body visible in the side-on jumps, the motion on which the single-LiDAR tracker failed. With the triangle, the Xtion in landscape at 2.5 m sees the walk across better than in portrait at 2.2 m (91 % against 85 %), because the wider view keeps the walker in frame; the price is more depth noise (about 10 mm against 8 mm, scaled from Kinect v1 measurements). On the visible surface the Xtion samples every 3 to 6 mm, the Ouster every 25 to 37 mm and the Helios every 39 to 64 mm: the Helios tells where a part is, not its shape.

Limits: geometry only (no noise, dropouts or tracking); the test body, not the subject; two-second motions; no furniture or other occluders. Rerun with the subject's avatar before choosing; the tracking errors of a layout follow when the multi-sensor tracker exists.

## 5. Synthetic check of evaluate-take (tests/test_evaluation.py)

A simulated standing person (test body, 512 x 20 LiDAR at 2 m) evaluated with its true motion and with the same motion moved 8 cm towards the sensor:

| Motion | fitted | violation | unexplained | point distance median | per-part median error |
|---|---|---|---|---|---|
| true | 90.4 % | 2.4 % | 7.2 % | 4.8 mm | about the noise |
| moved 8 cm | 1.2 % | 92.5 % | 6.3 % | 27.7 mm | 79 to 89 mm on the trunk, limbs and head |

The per-vertex errors recover the 8 cm displacement along the rays; the nearest-surface distance does not.
