# Moving people: animated meshes from one LiDAR

Version 1.2 adds a second use of the sensor: a person walks, jumps or moves
in front of the Ouster OS0-128, and the package produces an **animated
mesh** of that person (one mesh per time step, with the same triangles
throughout and the velocity of every vertex) and the person's **body
tracks** (the 55 SMPL-X joints over time), for radio simulations in Sionna RT
such as micro-Doppler spectrograms.

The motion comes only from the LiDAR frames of the recording. No motion is
generated or learned: a body model of the person (from the turntable scan)
is fitted to every frame, and only generic physical and anatomical rules
(joint limits, the floor, feet that do not slide while standing, smooth
motion) fill what one sensor cannot see. Section 7 says what that
leaves uncertain.

## 1. The workflow in eight commands

```
python -m bodyscan capture-turntable C:\lidar\s01_tt --turn-deg 1800          scan of the person (as before)
python -m bodyscan fuse C:\lidar\s01_tt --out C:\lidar\s01_scan
python -m bodyscan avatar C:\lidar\s01_scan.ply --model C:\smplx\SMPLX_NEUTRAL.npz --out C:\lidar\s01_avatar

python -m bodyscan capture-motion C:\lidar\s01_walk01 --duration 60            a take of the person moving
python -m bodyscan segment-motion C:\lidar\s01_walk01 --out C:\lidar\s01_walk01_seg
python -m bodyscan track C:\lidar\s01_walk01_seg --avatar C:\lidar\s01_avatar.npz --out C:\lidar\s01_walk01_motion
python -m bodyscan review-motion C:\lidar\s01_walk01_motion.npz --segments C:\lidar\s01_walk01_seg
    --avatar C:\lidar\s01_avatar.npz --out C:\lidar\s01_walk01_review
python -m bodyscan export-motion C:\lidar\s01_walk01_motion.npz --avatar C:\lidar\s01_avatar.npz
    --out C:\lidar\s01_walk01_meshes --rate 200
```

| Command | Input | Output | Needs |
|---|---|---|---|
| `avatar` | the fused cloud (or mesh) of the person, standing in the A-pose | `<out>.npz` (the avatar), `<out>_rest.ply`, `<out>_scan_pose.ply`, `<out>.json` | PyTorch, SMPL-X |
| `capture-motion` | the sensor | a capture directory (empty room first) | Ouster SDK |
| `segment-motion` | a capture directory | `scene.npz`, `frames/person_XXXXX.npz`, `segments.json` | numpy, Open3D |
| `track` | a segmentation folder (or a capture directory) and the avatar | `<out>.npz` (the motion), `<out>.json` (summary) | PyTorch |
| `review-motion` | the motion, the segmentation, the avatar | `review.gif`, `review_XXXXX.png` | Pillow for the GIF |
| `export-motion` | the motion and the avatar | `frames/mesh_XXXXXX.ply`, `frames/velocity_XXXXXX.npy`, `sequence.npz`, `joints.csv`, `export.json` | PyTorch |

Every command takes `--write-config`, `--config` and `--set` like the
others; `configs/*_default.toml` hold the defaults, `docs/parameters.md`
every parameter.

## 2. What you need

| Item | How | Why |
|---|---|---|
| PyTorch | `python -m pip install torch` (on the processing PC with an NVIDIA GPU, the CUDA build from pytorch.org) | the body model and the fits; the capture PC does not need it |
| Pillow | `python -m pip install pillow` | the review GIF (without it only PNG pictures) |
| SMPL-X model | register at https://smpl-x.is.tue.mpg.de, download the npz models (`SMPLX_NEUTRAL.npz`, or male/female) | the body model. The licence allows non-commercial research and forbids redistribution: keep the files outside the repository (`.gitignore` excludes `SMPLX_*`, folders named `smplx` and every `.npz`) |

Without SMPL-X, `python -m bodyscan make-test-body testbody.npz` writes a
procedural body in the same file format (capsules and ellipsoids, 55 joints,
10 shape directions). It is good for trying the commands and for the tests,
not for real people.

Processing time, measured on a 2-core cloud CPU without a GPU: about 0.85 s
per frame for the frame-by-frame fit and 1.4 s per frame for the refinement,
so a 60 s take at 20 frames per second (1,200 frames) takes about 45 min; a
frame that needs the limb search (section 4) takes 5 to 10 s more, a few
frames per jump. On a GPU it has not been measured.

## 3. Recording

### Sensor

| Setting | Recommendation | Why |
|---|---|---|
| Mode | **1024 x 20** (`capture-motion` warns otherwise) | the same columns per second as 2048 x 10, but twice the frames: the Nyquist limit of the motion is 10 Hz instead of 5 Hz. The horizontal spacing it gives up (12 mm instead of 6 mm at 2 m) is finer than the spacing of the beams (25 mm at 2 m) anyway. On the test bench (section 6) jumps are three to four times more accurate in this mode |
| Lens height | 1.0 m, upright and level | feet visible beyond 1.0 m, the head of a 1.85 m person beyond 0.85 m (vertical field of view ±45 deg, docs/hardware.md) |
| Distance of the motion | 1.5 to 3.5 m | at 2 m a 1.8 m person spans 73 beams (25 mm apart), at 3 m 49 (37 mm), at 5 m 29 (62 mm) |
| Connector | turned towards a wall, away from where people move | every frame starts and ends in the direction of the sensor's cable connector ([Ouster coordinate system](https://docs.ouster.com/sensor-docs/coordinate-system)): a person there is measured partly at the start and partly at the end of the frame, 50 ms apart. The tracker corrects for the time of every pixel, but a person elsewhere is measured within a few milliseconds |
| Room | a free strip of 6 x 2.5 m or more; furniture may stay but must not move during the take; nobody else in the view | the person is cut out against the empty room |

### Clothes

Tight stretch suits are right: the LiDAR measures the surface of the cloth,
and the avatar is fitted to the scan in the same suit. **Black is a risk**:
many black dyes absorb near infrared (the sensor works at 865 nm), so dark
fabric returns little light, mostly at grazing angles, and leaves holes in the
body. The segmentation counts a pixel without a return where the empty room
always returns, next to the person, as part of the person's silhouette, which
helps the tracker, but missing points are missing data. Test the suits
before a session: record the person standing still at 1.5, 2.5 and 3.5 m for
a few seconds each, run `segment-motion`, and compare the `dark_pixels` of a
frame with its `points` in `segments.json`: more than about 20 % dark pixels
means the suit is too dark for that distance (threshold chosen by judgement,
not measured).

### A take

`capture-motion` records 10 s of empty room (`--background-seconds`; nobody in
the sensor's view), counts down 10 s (`--delay`), then records until ENTER or
for `--duration` seconds.

1. Start on a floor mark in the A-pose (arms about 40 deg from the body,
   palms forward), still for 3 s: the tracker starts from this pose.
2. The motion.
3. End in the A-pose, still for 3 s.

Walking: straight passes 5 to 6 m long, 1.5 to 3.5 m from the sensor, in
both directions (each side of the body faces the sensor in one direction);
at least 10 gait cycles per direction, so 4 or 5 passes. Avoid circles around
the sensor for accuracy: they always show the same side. Jumping: in place,
facing the sensor and then side-on, 10 jumps per set. Several takes per
motion are cheap and let you discard a bad one.

The person's turntable scan (`capture-turntable`, then `fuse`; `mesh` is
optional) should be made the same day in the same suit.

## 4. Processing

### `avatar`: the person's body model

The SMPL-X model is fitted to the scan in two stages. Pose and shape: the
root, the 21 body joint rotations and the first `--num-betas` shape
coefficients, so that the scan points lie on the model surface (robust
point-to-plane distance) and the model lies on the scan where the scan has
data, with the soles on the floor; the facing direction is taken from the
shoulders and feet of the scan, or tried from several starts. Detail: the
model surface is subdivided `--level` times (level 2: about 3.6 mm edges for
SMPL-X) and every vertex moves along its normal by a displacement that brings
the surface onto the scan, kept smooth by a penalty on its slope, which also
fills the holes of the scan (soles, top of the head). The hands keep the
model's relaxed hand: the LiDAR does not resolve fingers (docs/hardware.md).

Check `<out>.json`: the scan-to-avatar distance (median and 90th percentile),
and open `<out>_scan_pose.ply` together with the scan.

### `segment-motion`: the person in every frame

1. Foreground: pixels closer than the empty room by more than `--bg-threshold`
   (8 cm) or 1.5 % of the range, minus mixed pixels at edges, in the floor
   frame found from the empty room.
2. The person: the foreground clusters within `--gate` (0.9 m) of where the
   person is predicted to be (constant velocity from the previous frames); in
   the first frame, or after `--lost-after` frames without the person, the
   largest cluster at least 0.8 m tall.
3. The time of every point (the sensor turns during a frame) from the column
   timestamps, and a crop of the range image around the person: person
   pixels, dark pixels, measured and empty-room ranges.

### `track`: the avatar in every frame

Stage 1 fits each frame in turn, starting from the prediction of the
previous frames (constant velocity). The unknowns are the root (rotation and
pelvis position) and the 21 body joint rotations; the shape is the avatar's.
The energy is minimised with L-BFGS, with correspondences renewed
`--tracking-rounds` times, coarse to fine: the robust scale of the data term
starts at `--coarse-scale` times `--robust-scale` and halves every round, so
that a limb that moved far since the last frame is still pulled to its
points. The terms:

| Term | What it asks |
|---|---|
| data | every point of the person close to the surface of the body that the sensor can see (robust point-to-plane distance plus a little point-to-point; only faces facing the sensor and not hidden by other parts take correspondences) |
| silhouette, free space | no part of the body where the sensor saw through to the background: a vertex that projects onto a pixel whose measured range lies beyond it (outside the crop: where the empty room lies beyond it and no person was found) is pulled towards the person's silhouette (a distance map of the crop, in angle, times the range). A vertex behind furniture is not pulled |
| in front | no part of the body in front of the measured surface |
| limits | the anatomical joint limits (bodyscan.body.skeleton: knees and elbows bend one way, and so on) |
| twist | the twist of the spine and the rotation of the thighs about their own axis stay small (weak, `--twist-weight`): otherwise the pelvis can turn while the trunk and the legs turn back by as much, which the points hardly tell apart |
| floor | no foot vertex below the floor |
| continuity | joint rotations and root close to the prediction; weak for what the sensor sees, up to 11 times stronger for a body part seen less than 30 % (`--hidden-continuity`), so that a hidden limb keeps its motion instead of jumping wherever a few points pull it |

The first frame is fitted from 8 facing directions; a frame whose points lie
more than `--lost-residual` (5 cm, median) from the body is fitted again from
several directions. When more than `--recover-fraction` (2 %, and at least
10) of the points remain farther than `--recover-distance` (6 cm) from the
body after the fit, a limb is probably in the wrong place: in a jump the arms
swing overhead in 0.3 s and the knees fold at the landing, faster than the
prediction follows. The frame is then fitted again from other starts: the
previous frame's pose, both legs from three poses (standing, half squat, deep
squat), each arm from eight (A-pose, hanging, elbow bent, out to the side,
forward, up in front, up at the side, back). Each start first moves only its
limb, and is kept when it explains the points better (mean point distance
capped at 15 cm, plus a penalty for body in free space, 2 mm better at
least); an arm only when the sensor sees at least 20 % of it there, since no
point can justify moving an arm the sensor does not see. Without this search
a lost limb can stay wrong for the rest of the take (an arm hidden behind the
body, legs landing feet forward instead of knees forward), because nothing
pulls it back.

Stage 2 refines the whole sequence in windows of 40 frames overlapping by 10:
the same terms for every frame (coarse to fine again), plus

| Term | What it asks |
|---|---|
| smoothness | the accelerations of the joints change smoothly: a Charbonnier penalty on the jerk (weight `--smoothness-weight` 0.5; quadratic below `--smoothness-scale`, 200 m/s³, linear above), which removes frame-to-frame jitter and keeps sharp real events such as a landing; up to 11 times stronger for a body part seen less than 30 % (`--hidden-smoothness`). A penalty on the acceleration itself (`--smoothness acceleration`) would also pull a jump's flight, a free fall at 9.81 m/s², towards a straight line |
| contact | a foot found standing (lowest vertex below 3.5 cm and ankle slower than 0.35 m/s in stage 1) does not slide |
| rolling shutter | every point and every silhouette pixel is compared with the body at the time it was measured (the body is moved along its motion between the neighbouring frames) |

Output `<out>.npz`: for every tracked frame its time (sensor clock), the
root, the body joint rotations (a PoseSequence: bodyscan.dynamic.motions),
the 55 joints, the foot contacts, per body part the fraction the sensor
observed, the point residuals and the review flags. `<out>.json` summarises
it.

Flags (a frame can have several): `residual` (points more than 3 cm from the
body, median), `unexplained` (more than 5 % of the points farther than 5 cm
from the body: a limb in the wrong place), `free_space` (more than 5 % of the
visible vertices where the sensor saw through), `below_floor` (a foot 2 cm
below the floor), `sliding` (a standing foot moving faster than 0.3 m/s),
`joint_limits` (a limit exceeded by more than 10 deg), `acceleration` (a
joint accelerating more than 120 m/s²).

### `review-motion`: human supervision

The tracked body seen from the sensor and from the side, with the person's
LiDAR points coloured by their distance to the body (green on the surface,
yellow at 2 cm, red at 5 cm and more). `review.gif` shows every `--every`-th
frame; every flagged frame gets a PNG. Look at the flagged frames first, then
at the GIF for unnatural motion (a hidden arm that freezes or swings
oddly). There is no automatic reference: this review is the check.

### `export-motion`: meshes for Sionna RT

| File | Content |
|---|---|
| `frames/mesh_XXXXXX.ply` | the body at step XXXXXX (binary PLY, metres, the same triangles at every step) |
| `frames/velocity_XXXXXX.npy` | the velocity of every vertex at that step (float32, m/s) |
| `frames/vertices_XXXXXX.npy` | with `--positions`: the vertices of that step (float32, m), for programs that update a mesh in place |
| `sequence.npz` | times, triangles, a body part label per vertex, the 55 joints per step, the mean velocity of every body part per step |
| `joints.csv` | the body tracks: time, then x, y, z of the 55 joints |
| `parts/<part>/mesh_XXXXXX.ply` | with `--split-parts`: one mesh per body part (20 parts) |
| `export.json` | rate, level, frame, units, how the velocities were obtained |

The frame is the floor frame of the recording: z up, z = 0 on the floor,
origin on the floor below the LiDAR, x and y along the LiDAR's axes. The
motion between tracked frames is the C1 interpolation of the tracked poses
(positions and velocities continuous), so `--rate` can be anything; the
velocities are the derivative of that motion, not differences between steps,
so they are exact for it at any rate.

Size: with SMPL-X at level 2 (167,285 vertices, 334,528 triangles) a step
takes 6.4 MB of PLY and 2.0 MB of velocities, so 60 s at 200 steps per second
is about 100 GB. `--level 1` divides it by about 4, `--no-velocities` drops
the velocity files, `--start` and `--stop` choose a span.

How to use it in Sionna RT. Sionna RT gives one velocity per scene object and
has no documented way to deform a mesh (version 2.2, checked on 2026-10-06;
the vertex positions can probably be rewritten through Mitsuba's scene
parameters, untested), so the Doppler of a path comes from the velocities of
the objects it touches. Two ways follow:

1. Body parts as objects: `--split-parts`, load the 20 part meshes of a step
   as 20 objects and give each the mean velocity of its part from
   `sequence.npz` (`part_velocity`); recompute the paths at the export rate
   (100 to 200 steps per second is enough for the geometry). A forearm or a
   shank also rotates, so one velocity per part is an approximation.
2. Paths at every step: replace the person's mesh at every step and take the
   phase change of every path between steps. The steps must then be shorter
   than half the period of the largest Doppler shift: at 60 GHz a hand at
   4 m/s gives 1.6 kHz, so more than 3,200 steps per second.

The per-vertex velocities serve a custom Doppler computation (the velocity of
the surface at the point a ray hits).

## 5. Synthetic recordings and evaluation

`simulate-motion` scans a body playing a motion with a simulated OS0-128
(exact beam layout, the sensor turning during a frame, block-wise posing of
the body every 8 columns, beam footprint with sub-rays and mixed ranges at
edges, Gaussian range noise of 8 mm, dropouts on the body that grow at
grazing angles) in a furnished room, and writes a capture directory that
every command reads, plus the truth: `truth.npz` (the motion),
`truth_avatar.npz` (the exact body), `labels.npz` (which pixels hit the
person) and `scan.ply` (a simulated turntable scan of the body, the input of
`avatar`). Motions: `walk`, `walk-circle`, `jump`, `jump-forward`, `stand`,
or an AMASS file (real motion capture, registration at
https://amass.is.tue.mpg.de), at `--distance` from the sensor in the
direction `--azimuth` (90 deg by default; at 0 deg every frame starts and
ends on the person).

The procedural motions are test material, not output: the feet follow
planned footprints and the legs come from inverse kinematics on the body
model, so that feet on the floor neither slide nor go through it; jumps
follow a ballistic flight (0.25 m rise by default, 0.45 s in the air).

`evaluate-motion` compares a tracked motion with the truth:

| Metric | Meaning |
|---|---|
| MPJPE | mean distance of the 22 main joints (pelvis to wrists), per frame |
| seen / hidden joints | the same, split by whether the joint's body part was observed (at least 30 % of the part) or hidden (less than 5 %) |
| vertices | mean distance of corresponding vertices (when the avatar has the topology of the truth) |
| part velocity | per body part, the RMS error of the mean velocity of its vertices (what sets the Doppler shift of that part), and its Doppler equivalent 2 Δv / λ at 60 GHz |
| acceleration | mean error of the joint accelerations (jitter shows here) |
| standing feet | the estimated ankle speed while the true foot stands on the floor |

`bench-motion` runs simulate, segment, track and evaluate over a grid of
motions, sensor modes and distances and writes `bench.md`; with
`--avatar-from-scan` the avatar is fitted to the simulated turntable scan, as
it would be for a person, so the numbers include the avatar error.

## 6. Accuracy on the synthetic bench

Setup: the procedural test body (1.76 m) in two motions: a walk of 4 s at
1.15 m/s across the line of sight (2 s standing before, 1 s after), and two
countermovement jumps in place facing the sensor (pelvis rising 0.25 m, 0.45 s
in the air, arms swinging overhead), at 2 and 3.5 m from a simulated OS0-128
in its two modes. The avatar is fitted to a simulated turntable scan of the
same body, so the numbers include the avatar's own error; every setting is
the default. Errors are means over the whole take, standing phases included
(about 40 % of a walk take). Command: `python -m bodyscan bench-motion <folder>
--model testbody.npz --avatar-from-scan`.

| Motion | Mode | Distance | Joints, mean | Joints, p90 | Seen joints | Hidden joints (share) | Part velocity (Doppler at 60 GHz) | Acceleration error | Standing feet |
|---|---|---|---|---|---|---|---|---|---|
| jump | 1024 x 20 | 2 m | 11.6 mm | 19.1 mm | 11.8 mm | 8 mm (5 %) | 0.14 m/s (57 Hz) | 1.1 m/s² | 6 mm/s |
| jump | 1024 x 20 | 3.5 m | 13.9 mm | 20.3 mm | 13.2 mm | 12 mm (5 %) | 0.18 m/s (70 Hz) | 1.4 m/s² | 7 mm/s |
| jump | 2048 x 10 | 2 m | 35.8 mm | 100.9 mm | 27.2 mm | 104 mm (6 %) | 0.62 m/s (248 Hz) | 1.9 m/s² | 7 mm/s |
| jump | 2048 x 10 | 3.5 m | 56.1 mm | 128.3 mm | 44.3 mm | 134 mm (7 %) | 0.63 m/s (251 Hz) | 2.1 m/s² | 6 mm/s |
| walk | 1024 x 20 | 2 m | 13.3 mm | 20.5 mm | 9.3 mm | 42 mm (12 %) | 0.11 m/s (43 Hz) | 0.9 m/s² | 6 mm/s |
| walk | 1024 x 20 | 3.5 m | 19.5 mm | 27.3 mm | 12.4 mm | 44 mm (20 %) | 0.27 m/s (109 Hz) | 2.2 m/s² | 10 mm/s |
| walk | 2048 x 10 | 2 m | 16.9 mm | 26.4 mm | 12.0 mm | 55 mm (11 %) | 0.12 m/s (49 Hz) | 0.6 m/s² | 5 mm/s |
| walk | 2048 x 10 | 3.5 m | 20.4 mm | 29.4 mm | 14.7 mm | 48 mm (16 %) | 0.13 m/s (53 Hz) | 0.7 m/s² | 4 mm/s |

How to read it:

1. **1024 x 20 is the mode for jumps**: at 10 frames per second the push-off
   (0.3 s) and the flight (0.45 s) are 3 or 4 frames, and the joint error
   triples or quadruples. For walks 1024 x 20 is better at 2 m and mixed at
   3.5 m (better joints, worse velocities, from the hidden arm below). Use
   1024 x 20.
2. **What the sensor sees is accurate**: joints of observed body parts are
   within 9 to 13 mm in every 1024 x 20 take.
3. **What it does not see is not**: the arm on the far side of a walk is 42
   to 55 mm off on average, and at 3.5 m its hand moves with an error of
   0.8 m/s (true speed 0.7 m/s), which is most of the 0.27 m/s of that row.
4. **Velocities**: 0.11 to 0.14 m/s of error per body part at 2 m, 43 to
   57 Hz at 60 GHz. For scale, a torso walking at 1.2 m/s shifts by 480 Hz,
   a swinging hand at 3 m/s by 1.2 kHz.
5. **No foot sliding**: standing feet move 4 to 10 mm/s (a foot dragged on the
   floor would be 100 mm/s and more).

Where the frame starts (the connector side of the sensor) a person is
measured at the start and at the end of the frame. The same jump, with the
exact body: 11.6 mm and 0.135 m/s of velocity error side-on, 14.9 mm and
0.330 m/s at the seam (hands 0.65 to 0.83 m/s). Turn the connector away from
the people (section 3).

How the refinement was chosen (walk and jump at 2 m, 1024 x 20, during
development): the frame-by-frame fit alone leaves frame-to-frame jitter
(estimated joint accelerations 1.5 to 1.8 times the true ones) and standing
feet that slide by 5 to 6 cm/s. A penalty on the jerk (weight 0.5) removed
nearly all the sliding and a quarter to a third of the acceleration error,
and lowered the velocity error by 5 to 7 %; a penalty on the acceleration did
about as well on these takes, but it also pulls a free fall towards a
straight line, so the jerk is the default. The stronger smoothing of the parts
the sensor hardly sees took the velocity error of the walk at 3.5 m from 0.36
to 0.28 m/s (the far hand from 1.2 to 0.9 m/s), and the limb search fixed the
landings (jump at 2 m, exact body: joint error from 21 to 12 mm, worst frame
from 120 to 40 mm).

## 7. Limits

**One sensor sees one side.** The body parts turned away from the LiDAR (the
far arm while walking past it, the back while facing it) give no points. The
tracker keeps them plausible (limits, the silhouette and free space, smooth
accelerations), but their motion is not measured: the bench shows how much
worse hidden joints are than seen ones. `<out>.npz` stores per frame and per
body part the fraction that was observed, so that a radio study can tell
measured from inferred motion. If the radar will stand next to the LiDAR, the
parts the LiDAR misses are mostly the parts the radar does not illuminate
directly (moderate confidence: reflections from walls reach the far side).
A second LiDAR on the other side would measure them.

**Fingers.** The LiDAR does not resolve fingers at these distances (beam
footprint 14 to 26 mm and beams 19 to 43 mm apart at 1.5 to 3.5 m; a finger
is about 17 mm wide; docs/hardware.md). The hands are positioned by the
forearms and keep the relaxed hand pose of the model. Finger motion needs
cameras or gloves.

**Time resolution.** 20 frames per second sample the motion up to 10 Hz.
Gait has content up to about 8 to 15 Hz at the feet, so the fastest parts of
a step and the impact of a landing are smoothed. Every frame is itself
spread over the 50 ms of a sensor turn; the rolling shutter term uses the
time of every point, which corrects the position of the parts, not motion
faster than the frame rate.

**Rigid skin.** The avatar moves by linear blend skinning: muscles, breathing,
the belly and the cloth do not deform. Tight suits keep the cloth on the
body.

**The numbers are synthetic.** The bench uses the procedural test body and
procedural motions with a simulated sensor. On real recordings there is no
reference; the review flags and pictures are the check. A comparison with
motion capture (markers or inertial) would give real error numbers.

**No learned prior.** Following Marco's preference for motion captured in
the real world, no motion is generated: no network or motion library
constrains the poses. A learned pose or motion prior
(trained on real motion capture such as AMASS) would make the hidden parts
more natural; it would be one more term in the tracker's energy.

## 8. Using the code

```python
import numpy as np
from bodyscan.dynamic.avatar import Avatar
from bodyscan.dynamic.motions import PoseSequence

motion = PoseSequence.load("s01_walk01_motion.npz")          # times (sensor clock), root, joint rotations
avatar = Avatar.load("s01_avatar.npz")
model = avatar.model("C:/smplx/SMPLX_NEUTRAL.npz", "cpu")
body = avatar.shaped(model)                                  # the person's body, at the avatar's level
times = motion.times[0] + np.array([3.0, 3.005])             # 3 s into the take, and 5 ms later
vertices, joints = motion.sample(times).posed(body)          # (2, V, 3) and (2, 55, 3), metres
```

Joint rotations are axis-angle vectors in the parent joint's frame, as in
SMPL-X; the root rotation and position are in the world (floor) frame, with
the pelvis as the root position. The signs of the joint rotations are
documented in bodyscan/body/skeleton.py and checked by the tests.
