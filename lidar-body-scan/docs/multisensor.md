# Moving people with several sensors

Version 1.8 tracks a moving person with the Ouster OS0-128 and any number of additional sensors seen in
the same take: on the recording of 3 Nov 2026 a RoboSense Helios 16P (a second spinning LiDAR) and an ASUS
Xtion PRO Live (a depth camera). It also fits the hands (pose model `full55`, docs/adr/0002-full-pose-55.md)
and chooses, frame by frame, among several tracked runs of the same take (`select-motion`).

This page is the user guide of that tracker. The single-LiDAR tracker it extends is described in
[motion.md](motion.md) (every term of the energy, the hidden-limb coupling, the rules of a human body); the
design and the stages of the multi-sensor work are in [multisensor-plan.md](multisensor-plan.md); the numbers
and the viewer in [evaluation.md](evaluation.md).

To see what the tracker does with the takes of 3 Nov 2026: the animations and figures of the section
[Moving people](../README.md#moving-people) of the README, and the interactive viewer
([docs/motion-viewer/index.html](motion-viewer/index.html), opened from the README).

Everything below was measured on one recorded session (subject s02, 8 takes of 50 to 80 s, 10,800 frames at
20 Hz) without a ground truth: the measures are how well the body explains the points of each sensor,
including points held out of the fit, and how much two runs that should agree disagree.

## 1. A take with several sensors

The Ouster stays the reference: its segmentation folder (`segment-motion`, `<take>_seg`) gives the floor
frame, the timing and the frame numbering. The other sensors come as one more folder per take:

```
<take>_extra/
    views.json                        the sensors: name, kind (spinning or pinhole), noise relative to the
                                      Ouster (sigma), free-space weight, range of use
    helios/scene.npz                  pose in the Ouster's floor frame, image geometry, empty-scene ranges
    helios/frames/person_XXXXX.npz    the person in that sensor's image closest in time to Ouster frame XXXXX
    xtion/scene.npz
    xtion/frames/person_XXXXX.npz
```

A person frame of an additional sensor has the arrays of the Ouster's: the person's points in the floor
frame with their times, their pixels, and a crop of the image around the person (mask, measured and
empty-scene ranges, pixel times). `bodyscan.dynamic.views` reads the folder; `bodyscan.dynamic.extra_segment`
(numpy only) finds the person in an additional sensor's images: closest image in time, background
subtraction, a gate around the Ouster's person, a silhouette crop that keeps dark pixels (a suit the Helios
hardly sees must not look like free space), optional voxel thinning of a dense depth camera.

The decoding of the raw recordings, the extrinsic calibration and the time offsets are made by the session
tools of the three-sensor rig, which are not part of this repository yet. For the 3 Nov session they found
(measured against the avatar fitted to the Ouster alone):

| | Helios 16P | Xtion PRO Live |
|---|---|---|
| calibration | static, on the empty room (ICP inlier median 6.1 mm); the person seen about 8 cm short at 4 m, corrected by a range correction of the person's pixels only (the room agrees with the Ouster within 6 mm) | static calibration about 35 cm off sideways; corrected on the person by a yaw and a shift in the floor plane; inverse-depth offset -0.0031 per m; floor returns below 6 cm dropped |
| time offset | 2 to 3 ms, consistent with zero | 33 ms late (residual lag after correction 0.5 ms median) |
| residual on the person | median 14 mm | median 20 mm; depth steps of about 2.6 cm at 3.2 m |

## 2. The recipe

The commands of one take, with the settings that 1.8.0 recommends (section 7 explains the Xtion's range):

```
set SENSORS=--extra-sensors s02_walk01_extra --extra-sensor-noise xtion:2.0:0.26 --extra-sensor-range xtion:1.3:3.4,helios:3.4:5.2 --toe-weight 0.3

rem 1. body pose, several runs: forward and backward in time, and the sensor combinations
python -m bodyscan track s02_walk01_seg --avatar s02_avatar.npz --model SMPLX_MALE.npz --pose-model body21 --init-full 8 %SENSORS% --out runs\forward\motion
python -m bodyscan track s02_walk01_seg --avatar s02_avatar.npz --model SMPLX_MALE.npz --pose-model body21 --init-full 8 %SENSORS% --reverse --out runs\backward\motion
python -m bodyscan track s02_walk01_seg --avatar s02_avatar.npz --model SMPLX_MALE.npz --pose-model body21 --init-full 8 --extra-sensors s02_walk01_extra --exclude-sensors helios,xtion --toe-weight 0.3 --out runs\ouster\motion

rem 2. the best of them frame by frame, then a last pass from the chosen poses
python -m bodyscan select-motion runs\forward\motion.npz runs\backward\motion.npz runs\ouster\motion.npz --segments s02_walk01_seg --extra-sensors s02_walk01_extra --avatar s02_avatar.npz --extra-sensor-range xtion:1.3:3.4,helios:3.4:5.2 --out selected.npz
python -m bodyscan track s02_walk01_seg --avatar s02_avatar.npz --model SMPLX_MALE.npz --pose-model body21 %SENSORS% --init-motion selected.npz --out selected_motion

rem 3. the whole pose with the hands, from the selected motion
python -m bodyscan track s02_walk01_seg --avatar s02_avatar.npz --model SMPLX_MALE.npz --pose-model full55 --hand-pca SMPLX_MALE_v11.npz --hand-components 6 --no-flat-hand-mean --face-fixed %SENSORS% --init-motion selected_motion.npz --out final

rem 4. look at it
python -m bodyscan review-motion final.npz --segments s02_walk01_seg --avatar s02_avatar.npz --extra-sensors s02_walk01_extra --out review
python -m bodyscan viewer-pack final.npz --segments s02_walk01_seg --extra-sensors s02_walk01_extra --avatar s02_avatar.npz --hand-pca SMPLX_MALE_v11.npz --out viewer_packs\final\s02_walk01 --uncertainty-step 5
```

`--holdout-fraction 0.1` (the same points in every run with the same `--holdout-seed`) keeps a tenth of
every sensor's points out of every fit and of the selection, to measure the fit on points it did not use.
Each run of step 1 is independent: on a cluster they run side by side. On the 3 Nov session one body pass
took 3.6 to 5.8 h per take with 4 CPU threads, the full-pose pass from a warm start 3.0 to 4.3 h.

Why the steps: the hands are fitted only from a warm start, because a cold full-pose fit of a real walking
window left 13.9 mm of median point distance against 6.9 mm for the body pose alone, with limb searches in
26 to 29 of 32 frames; from the body-pose motion it left 6.7 mm. The start pose comes from whole fits from 8
facings (`--init-full 8`), because the earlier start (quick fits ranked by energy) tracked 4 of 16 reference
windows facing backwards. The jaw and the eyes stay at rest (`--face-fixed`): they are not observable at 2
to 5 m.

## 3. How the sensors enter the fit

| Term or measure | With several sensors |
|---|---|
| data | the points of every sensor, each distance divided by its sensor's noise relative to the Ouster's (`sigma` in views.json, or `--extra-sensor-noise name:sigma:quadratic`: max(sigma, quadratic r^2) at range r); the sum over all points is divided by the number of the Ouster's points (`--reference-count`, default on), so that another sensor adds evidence instead of diluting the Ouster's |
| free space, in front | per sensor, on its own image (`RangeImageGeometry` for spinning LiDARs, `PinholeGeometry` for depth cameras; for the sparse beams of the Helios the free space is tested only near a beam); weight `--extra-sensor-weight` |
| visible faces | per sensor; a body part counts as observed when any sensor sees it |
| time | stage 1 moves every point of an additional sensor to the Ouster frame's time with the velocity of the last two frames, each point by its own time; stage 2 compares every point and pixel with the body at its own time, as for the Ouster's rolling shutter |
| range of use | `--extra-sensor-range name:nearest:farthest`: the frames of a sensor with the person outside this range are dropped (the log says how many) |
| searches | the misfit measures that start the limb searches (unexplained share, score) are counted per Ouster point too, with every distance in units of its sensor's noise; with raw distances the noisier sensors set off searches in 11 of 64 frames, whitened in 1 of 32 |
| held out | `--exclude-sensors a,b`: those sensors are loaded but not fitted; their points check the pose (held-out-sensor validation) |
| toes | `--toe-weight 0.3` holds the toe joints within about 2 deg of rest; without it they drifted upwards to medians of 15 to 41 deg per take and to the 60 deg limit in up to 22 % of the frames |

Every motion stores per sensor the frames it was seen, its points per frame and its residual, fitted and
held out (`motion.json`, `motion.npz`).

## 4. Choosing among runs: `select-motion`

Where an arm moves fast, the fit of a frame has several local minima and the track keeps whichever it fell
into. Two runs of the same configuration that differ only in rounding (the robust scale changed by 3e-8 of
its value) put an arm joint more than 10 cm apart in 11 to 25 of the 37 frames of a jump of s02_jump01
(frames 225 to 261). One run is therefore a sample, not a result, and two configurations cannot be ranked
on one run each when their difference is smaller than that spread.

`select-motion` (`bodyscan.dynamic.select`) takes several motions of the same take, scores each frame of
each with the tracker's own score (mean point distance capped at 15 cm, plus 15 cm times the share of the
visible body in free space; points of every sensor in use, in units of their noise), and chooses a path through the
motions by dynamic programming: the sum of the scores plus, at every change of motion, 0.05 per metre of the
largest joint distance between the two motions beyond 3 cm. Then each arm is chosen on its own the same way.
The chosen poses are the warm start of a last pass (`track --init-motion`), which fits every frame from them
and refines the windows, so that the joins are smooth.

Ouster points 5 to 25 cm from the body, summed over the window (lower is better):

| s02_jump01, frames 225 to 261 | points off | arm more than 10 cm from the repeat |
|---|---|---|
| single forward runs (two) | 1057, 737 | 13 of 37 frames |
| selection among two forward runs | 503 | |
| selection among a forward and a backward run (two disjoint pairs) | 335, 461 | 2 of 37 frames |
| selection among all four | 442 | |

On a walk (s02_walk01, frames 225 to 319, no fast arm motion) the selections are within the spread of the
runs: 713 among two forward runs, 702 and 721 among a forward and a backward run.

## 5. Other paths through the same energy

`--reverse` runs stage 1 from the last frame to the first: the prediction, the gaps, the turn limit, the
facing checks and the spreading of a turn run backwards in time; the motion is written in the order of time
and stage 2 runs forwards. A limb lost in fast motion is lost at other frames than forwards: in the jump
above the forward runs lose the arms in the take-off and the flight (frames 236 to 246, up to 164 points off
per frame), the backward runs keep them (at most 29) and fail at frames 247 and 248 instead (43 to 50).
Single backward runs: 463 and 517 points off, no arm more than 10 cm apart between the two. On the walk
they are as good as the forward runs (770 and 763 against 747 and 764). Forward runs are unchanged to the
last bit by the option.

`--beam-width N` (`bodyscan.dynamic.beam`) carries N tracks from frame to frame, each continued from its
constant-velocity prediction and from its last pose held still; the cheapest distinct N carry on (the
explain() score plus the same 0.05 per metre for leaving the prediction) and the motion is the cheapest track
at the end. With N = 3: 614 and 611 points off in the jump, at 3.5 times the time of a forward run. It is
dominated by a backward run plus the selection, which cost less and do better; it stays as an option.

Not kept: more correspondence rounds in stage 2 (5 of 16 iterations instead of 2 of 40) changed nothing
beyond 2 mm (1046 and 733): stage 2 does not undo the decisions of stage 1.

## 6. Results on the session of 3 Nov

All 8 takes, each tracked with every configuration (10 % of the points held out), then selected and fitted
with the hands. Medians over the 8 takes; Ouster columns in mm:

| Run | Ouster residual | Ouster held out | worst take, held out | unexplained | limb uncertainty | acceleration p95 | toe |
|---|---|---|---|---|---|---|---|
| production of 10 Nov (mean over all points, full pose, flat hands) | 7.09 | | | 1.88 % | 13.2 mm | 47.4 m/s² | -21.2° |
| old weighting (mean over all points) | 7.17 | 7.36 | 8.14 | 1.64 % | 12.4 mm | 43.8 m/s² | -20.1° |
| reference count | 6.93 | 7.05 | 7.76 | 1.17 % | 11.3 mm | 37.1 m/s² | -24.4° |
| all changes of 1.8 (noise by range, ranges, toes) | 7.23 | 7.31 | 11.02 | 1.24 % | 10.5 mm | 37.7 m/s² | -0.1° |
| the same, repeated (rounding) | 7.09 | 7.25 | 7.60 | 1.23 % | 10.6 mm | 37.2 m/s² | -0.1° |
| Ouster alone | 6.56 | 6.74 | 11.85 | 1.16 % | 34.2 mm | 31.9 m/s² | 0.0° |
| Ouster and Helios | 6.51 | 6.67 | 11.10 | 1.15 % | 17.1 mm | 34.9 m/s² | 0.0° |
| Ouster and Xtion | 7.07 | 7.29 | 7.69 | 1.33 % | 11.8 mm | 36.8 m/s² | -0.1° |
| selection among the runs above, last pass | 6.72 | 6.90 | 7.33 | 1.04 % | 11.0 mm | 36.1 m/s² | -0.1° |
| the selection with the hands (final) | 6.67 | 6.84 | 7.25 | 1.02 % | 10.9 mm | 35.7 m/s² | -0.1° |

Unexplained: share of the Ouster's points 5 to 25 cm from the body. Limb uncertainty: the largest
standard deviation of the knees, ankles, elbows and wrists from the Gauss-Newton covariance of the fit,
median over the frames. Acceleration: 95th percentile of the joint accelerations.

How to read it:

* Three of the eight single runs lost a whole take (the worst take column: 11.02, 11.85 and 11.10 mm
  against 8.14 mm or less in every other case); which ones is partly chance, since the repeat of the first,
  which differs from it only in rounding, lost none. The selection never lost a take (worst take 7.33 mm).
* The Ouster's held-out points favour fits that use the Ouster alone or mostly: the Ouster with the Helios
  has the lowest held-out distance in 6 of 8 takes. They cannot judge what only the other sensors see:
  without the Xtion the limb uncertainty triples (34.2 mm with the Ouster alone, 11.8 mm with the Xtion).
* Against the production of 10 Nov the final motion explains more of the points (1.02 % unexplained
  against 1.88 %), moves less abruptly (35.7 against 47.4 m/s²), and keeps the toes at rest.

## 7. Findings and limits

**The Xtion's useful range is 1.3 to 3.4 m.** Its points against the Ouster-only fits (the Xtion held out),
by the person's distance from it: median distance about 20 mm and signed bias within 4 mm from 1.8 to
3.4 m; 34 mm and +31 mm at 3.4 to 3.8 m; 54 mm and +54 mm at 3.8 to 4.2 m. The camera is specified for 0.8
to 3.5 m. With the range at 4.2 m these points turned the body in the turns at the far end of
s02_walk03_radial: 8 to 15 % of the points unexplained against 1 % with the Ouster alone, the pelvis
turning at up to 510 deg/s. On frames 180 to 244 of that take: 1646 and 1624 Ouster points off with the
Xtion up to 4.2 m, 694 and 689 up to 3.4 m, 502 with the Ouster alone; a selection among the 4.2 m runs and
the Ouster-only run, scored and refitted with the Xtion up to 3.4 m, 598. Closer than about 1.3 m the
Xtion returned a fraction of the body with errors of 10 to 30 cm (s02_walk01: an arm followed those points
over the head for 3 s).

**Fingers are not measured.** At 2.5 to 3 m the points do not tell an open hand from a fist: in a boxing
window of s02_natural01 (frames 585 to 629) the hand search from starts between open and closed
(`--hand-starts 4`, also in warm-start frames since 1.8.0) kept the flat hand when the prior is the flat
hand, the fingers bent about 34 deg with the relaxed hand as prior, and the distances of the points near
the fingers differed by at most 1 mm between these. The hand shown is the prior's: `--no-flat-hand-mean`
(the relaxed hand) is the closer default for most activities. A fist would have to be imposed, not
measured.

**An arm no sensor sees well stays ambiguous.** In s02_walk02 (frames 666 to 680) the Ouster sees almost
nothing of the right arm and the Xtion sees it from 3.4 to 2.8 m: a natural swing and an arm turned inwards explain
the points equally (the scores of the two differ by about 0.1 mm per frame), and in four runs of that window
three kept the natural swing and one turned it inwards. Neither the selection nor more runs can decide
this; a motion prior for unobserved limbs, or a sensor placed to see that side, would.

**Other limits.** No ground truth: every number above is the agreement of the body with the measurements or
between runs. The calibration of the additional sensors is made on the person with the Ouster as the
reference, so a bias of the Ouster or of the avatar is not seen. The Helios person bias may depend on
reflectivity (not modelled). The noise model of the Xtion is fitted (2.0 times the Ouster's, 0.26 r^2 where
larger), not measured on a target.

Not validated: the prior weights of the hands and the face (`hand_prior_weight` 0.002, `face_weight` 10)
are first guesses; the beam and the backward tracking were measured on two windows of two takes; the
recommended Xtion range was measured on one session with one camera.

## 8. Looking at the results

`review-motion --extra-sensors` draws every sensor's points in its own colour, with a third view from
behind. `viewer-pack` packs a tracked take for an interactive viewer: the points of every sensor with
their distances to the body (also the points of sensors the run did not fit, as an independent check), the
per-frame metrics, and with `--uncertainty-step` the joint position uncertainty
([evaluation.md](evaluation.md), section 5).
