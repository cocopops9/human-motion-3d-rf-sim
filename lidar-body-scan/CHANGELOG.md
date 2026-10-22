# Changelog

## 1.7.1: no file left open on Windows (2026-10-22)

The first run of the fast tests on the recording laptop (Windows, Python 3.14) found two files left
open. Windows cannot delete or overwrite an open file, so the temporary folders of two tests could not
be removed; in use, the viewer locked every motion.npz and metrics.npz it had loaded until it was closed.

Fixed:

- `view`: the take data copies the arrays of motion.npz and metrics.npz into memory and closes the
  files (it kept both open for as long as the take was shown).
- `evaluate-take`: motion.npz is read and closed at the start (it stayed open for the whole evaluation).
- `flag_shares` accepts the npz or a dict of its arrays.
- tests/test_motion.py: the export test closes sequence.npz before removing its folder.

Added:

- Two tests that the viewer and `evaluate-take` leave no file open: they delete the motion, the
  metrics and the avatar right after loading them (`evaluate-take` while it runs), and check that
  every npz that numpy opened is closed, which also catches a leak on Linux.

## 1.7.0: numbers instead of videos, a viewer, and the first multi-sensor building blocks (2026-10-21)

The goal changed on 21 Oct: reconstruct motion and shape as precisely as possible with up to three
sensors (the Ouster OS0-128, a RoboSense Helios 16 and an ASUS Xtion depth camera), and judge results
by numbers and an interactive viewer instead of review videos. Plan: docs/multisensor-plan.md. This
version adds what does not depend on the multi-sensor tracker; `track` is unchanged.

Added:

- `bodyscan.sensors`: nominal models of the Ouster OS0-128, the Helios 16 (16 beams, +-15 deg, 2 deg
  apart; 0.2 deg at 10 Hz) and the Xtion (640 x 480, 58 x 45 deg, 0.8 to 3.5 m, depth noise
  0.0016 z^2, landscape or portrait); `RangeImage`, the frame type of every sensor (rays, ranges,
  valid / no return / unknown, times, noise); a ray-casting renderer; clocks (sensor time to session
  time from the lower envelope of the arrival times, robust to stalls of the receiving PC); placements
  and rig.json; candidate layouts around the capture zone.
- `bodyscan.evaluation` and `evaluate-take`: fit quality of a tracked take without ground truth.
  Every pixel near the person is compared, along its ray, with the body posed at the pixel's time:
  fitted, violation (the body where the sensor saw through), unexplained, silhouette, unobserved.
  Per-vertex error map (regions no sensor saw have no value), per-part tables, point inlier, outlier
  and clutter rates, kinematics (jitter as the velocity power above 6 Hz, turns faster than
  720 deg/s, hinges out of plane, joint speeds over the limits), tracker flag shares; metrics.json,
  metrics.npz, coloured heatmap meshes (median, p95, violation, coverage) and report.md.
- `view`: an interactive viewer in the browser (Viser, optional: `pip install viser` or the `viewer`
  extra): take by motion type, playback, points by sensor or by distance to the body, the body plain,
  with guessed limbs in violet or as a heatmap, skeleton, sensor, numbers of the frame and of the
  take, curves.
- `bench-rig`: coverage of the body by candidate layouts (visible share per part and per sensor,
  sample spacing).
- docs/evaluation.md, docs/multisensor-plan.md, docs/adr/0001-sensor-abstraction.md.

Fixed:

- SciPy was used by the left-right coupling of `track` (bodyscan.dynamic.coupling) since 1.5.0 but
  not declared: it is now in the `motion` extra.

Measured (tests and runs in a 1-CPU sandbox, test body):

- evaluate-take on a simulated standing person with its true motion: fitted 90.4 %, violation 2.4 %,
  unexplained 7.2 % of the ranged pixels, point distance median 4.8 mm; with the motion moved 8 cm
  towards the sensor: violation 92.5 %, per-part median errors 79 to 89 mm (the 8 cm along the rays),
  while the nearest-surface point distance rises only to 27.7 mm (it understates displacements along
  the ray).
- bench-rig, mean share of the body visible to some sensor (walk across, walk radial, jumps facing,
  jumps side-on): Ouster alone 38, 46, 46, 31 %; 120 deg triangle with the Xtion in landscape at
  2.5 m 91, 91, 94, 91 %; Xtion opposite and Helios at the side 90, 91, 92, 79 %. Sample spacing on
  the visible surface: Xtion 3 to 6 mm, Ouster 25 to 37 mm, Helios 39 to 64 mm.
- Fast tests: the 107 existing ones (with SciPy installed) plus 14 sensor, 12 evaluation and 3 viewer
  tests: 136 pass, 6 slow ones skipped.

Not validated:

- evaluate-take, view and bench-rig have not yet run on the real s02 takes or with the s02 avatar.
- The viewer was tested headless (server, controls and every drawing mode); it has not been looked at
  in a browser.
- The Xtion noise model is scaled from Kinect v1 measurements (same sensor class), not measured on this
  unit; its frame time is assumed global (no rolling shutter).
- Coverage is geometry only: no noise, dropouts, occluders or tracking.

## 1.6.0: a human body: hinges, physical turns and speeds, a steady chest, the hidden arm mirrored at the chest and kept out of sight (2026-10-20)

The review videos of the 27 s02 takes tracked with 1.5.0 were better than
with 1.4.1 while the person was side-on, but not natural: unseen parts
contorted, elbows and knees bent sideways, arms a little inside the trunk,
the far arm stiff and not following the near one (s02_jump02: "a crazy
worm"), and the body turning faster than a person can. Measured on the
1.5.0 motion files (no point clouds needed):

- elbows bent out of their plane of flexion beyond what an elbow allows
  (5 deg towards the body, 12 deg away from it) in 53 to 100 % of the frames
  of every take (an arm hanging with the palm forward was the upper arm
  turned out by 75 to 90 deg and the elbow bent sideways by 20 deg, the edge
  of its box limit); knees by more than 6 deg in up to 38 % of the frames
  (the side-on jumps);
- in 5 takes the body turned by 50 to 177 deg from one frame to the next (up
  to 3500 deg/s), all from the facing check of 1.5.0; in s02_run01_3 it then
  ran backwards;
- the chest twisted against the pelvis by 23 to 28 deg (p95) at up to
  280 deg/s in the side-on jumps;
- the far arm of the side-on jumps, mirrored in the frame of the pelvis'
  facing, did not mirror the near one relative to the chest: the far wrist
  6 to 14 cm (median) from the mirror of the near one; the guess was blended
  by adding rotation vectors;
- arms more than 2 cm inside the trunk in 14 to 51 % of the frames of the
  moving takes, up to 12 cm;
- single frames where a joint jumped by 30 to 110 deg and back (faster than
  any joint of a person in these motions: the far collar of the side-on
  jumps, the arms after the data gap of natural01_3), in up to 3.6 % of the
  frames.

Changes to `track` (each can be switched back with its option):

- Hinges (new module bodyscan.body.anatomy): elbows and knees bend in one
  plane. The forearm stays within 5 deg of the elbow's plane of flexion
  towards the body and 12 deg away from it (the carrying angle), the shank
  within 6 deg of the knee's; flexion -5 to 150 deg (elbow), -8 to 155 deg
  (knee). A penalty in stage 1 and 2 (`--hinge-weight` 20) and a projection
  of the final pose (the bone's direction clamped, its twist kept). The
  hanging start of the arms has the palm towards the thigh, not forward by
  the upper arm turned out.
- Wrist limits as a wrist: bending towards the palm and back ±80 deg,
  sideways ±35 deg, twist ±40 deg (the forearm turns the hand); the box had
  bending and sideways swapped.
- Turns as a body turns: a facing found by the facing check that the body
  cannot reach from the second before at `--max-turn-rate` is kept only if
  the frames before, fitted again backwards from it, explain their points as
  well (`--connect-tolerance` 3 mm) and meet the earlier track within
  `--connect-frames` (30): the turn is then spread over them; otherwise it is
  refused. A facing that would make the body move backwards (pelvis faster
  than 0.6 m/s, more than 120 deg from the facing) is refused.
- The twist of the chest against the pelvis (the sum of the three spine
  joints' twists) is held (`--trunk-twist-weight` 1).
- Hidden limbs: the partner's joint rotations, each relative to its parent,
  mirrored: the guessed arm hangs from the chest as its partner does. Version
  1.5 rejected this because a chest tracked 10 deg twisted put the far wrist
  10 cm off; with the chest held, the mirror at the chest is right, while the
  mirror in the frame of the pelvis doubled every twist of the chest. Offsets
  and blends are combined as rotations (quaternion products, spherical
  interpolation). No turn about the vertical any more (`--shadow-turn`
  removed).
- The last step, after the completion of the hidden limbs
  (`Tracker.finish_pose`):
  - a guessed arm standing where the sensor saw something else (empty space
    beyond it, a surface behind it) is moved out of its sight by the least
    turn of its shoulder and elbow, smoothed over 0.15 s, twice
    (`--hide-guessed-limbs`);
  - no joint turns faster than a person's (bodyscan.body.anatomy.SPEED_LIMITS:
    400 deg/s for each spine joint, 500 for the neck and the head, 600 for the
    collars, 1000 to 1500 for the limbs; `--joint-speed-scale` 1, 0 for none):
    a faster change between two frames is spread over the frames around it,
    as many as the joint needs; a joint that jumped for one frame and came
    back takes the mean of its neighbours;
  - the hinges are projected;
  - every arm still deeper than the margin inside the trunk is pushed out by
    the least turn of its shoulder and elbow, spread over the neighbouring
    frames.
- Arms out of the trunk in stage 1 and 2: `--penetration-margin` 1 cm
  (2 cm), the squared excess depths summed per ten arm samples (averaged over
  all of them, so that a forearm 5 cm inside cost little).
- New summary fields: facing_refused, turn_frames_refitted,
  guessed_arms_hidden, joint_speeds_limited, hinges_projected,
  arms_pushed_out. The `joint_limits` flag includes the hinges.

On the synthetic bench (test body, 1024 x 20, default settings), the whole
pipeline of 1.5.0 as released and of 1.6.0 on the same takes (side-on jumps:
mean over three noise realisations, with the mean paired difference and its
range; the others one realisation each; joint error, body-part velocity
error, error of the far wrist and elbow, velocity error of the far hand):

| Motion | Joint error | Body-part velocity error | Far wrist | Far elbow | Far hand velocity |
|---|---|---|---|---|---|
| jumps side-on, 2.3 m | 22.3 then 23.4 mm (+1.1, +0.5 to +1.7) | 0.332 then 0.274 m/s (-0.058, -0.096 to -0.030) | 85 then 78 mm | 61 then 68 mm | 1.07 then 0.75 m/s |
| jog across, 2.4 m | 19.1 then 16.2 mm | 0.310 then 0.289 m/s | 139 then 104 mm | 38 then 43 mm | 1.13 then 1.04 m/s |
| walk across, 2.4 m | 11.2 then 11.0 mm | 0.099 then 0.099 m/s | 28 then 33 mm | 25 then 23 mm | 0.22 then 0.24 m/s |
| jumps facing the sensor | 10.6 then 10.0 mm | 0.130 then 0.124 m/s | 14 then 12 mm | 15 then 14 mm | 0.21 then 0.18 m/s |
| jog towards the sensor | 16.0 then 16.0 mm | 0.693 then 0.615 m/s | 67 then 77 mm | 36 then 35 mm | 2.01 then 1.80 m/s |

The body is the more human one: elbows out of their plane in 20 to 61 % of
the frames with 1.5.0, never with 1.6.0; knees in up to 19 %, never; the
chest twisted against the pelvis by 16 deg (p95) in the side-on jumps, whose
true chest does not twist at all, 1 deg with 1.6.0; arms in the trunk up to
29 mm (jog towards the sensor), 12 mm; the guessed arm in free space as
often as with 1.5.0 or less (except the side-on jumps: visible body in free
space 2.4 then 2.5 %). Worse: the joint error of the side-on jumps, higher
in each of the three realisations (the joints the sensor does not see 44
then 47 mm, the far elbow 61 then 68 mm, the velocity of the near hand 0.24
then 0.27 m/s), and the far wrist of the walk (28 then 33 mm); the chest is now
stiffer than a real one in a walk or a jog (synthetic truth: 5 deg of twist,
p95; 1.6.0 tracks 0.7 to 1.5 deg, 1.5.0 7 to 12 deg).

On the hidden arms of the side-on jumps (first realisation), the steps one
by one (same stage 1 and 2): the plain mirror at the chest 22.7 mm, far wrist 94 mm, far elbow
80 mm, far hand velocity 0.76 m/s, visible body in free space 3.4 %; then
out of the sensor's sight 21.6 mm, 78 mm, 72 mm, 0.70 m/s, 2.3 %. The speed
limits changed no frame of these takes but two of the jog towards the
sensor, where they took the body-part velocity error from 0.666 to
0.615 m/s.

Tried and not kept:
- swing-twist cones for the shoulders: the twist of a swing-twist
  decomposition depends on the path (an arm swung forward and up has a twist
  of minus the swing), so a cone on it refuses ordinary poses;
- moving a guessed arm out of the sensor's sight frame by frame without
  smoothing: the far hand's velocity error of the jog went from 1.04 to
  1.49 m/s (0.1 s of smoothing: 1.07; 0.15 s: 1.04);
- the push of the arms out of the trunk with a change weight of 10: an arm
  put 9 cm inside came out to 4.3 cm; with 2 and a second round to 1.0 to
  1.2 cm.

Not validated yet: everything on real recordings (re-run the s02 takes and
send back the motion files: the anatomy, turns, speeds and arm depths are
measured from them); more than one noise realisation of the motions other
than the side-on jumps.

## 1.5.0: hidden limbs guessed from their partner; arms out of the trunk; the facing guarded (2026-10-18)

The review videos of the 27 s02 takes tracked with 1.4.1 were good while the
person faced the sensor and poor while the person was side-on: the far arm,
which the sensor does not see, kept still while the other one swung, took
arbitrary poses or sank into the trunk; heads turned towards the sensor; in
s02_jump02 the body turned around at the first take-off and stayed so.
Changes to `track` (each can be switched back with its option):

- Hidden limbs are guessed from their partner (new module
  bodyscan.dynamic.coupling, `[coupling]` options). Around every frame the
  kind of motion is estimated from what the sensor sees (the observed arm and
  leg, the height of the pelvis): alternating (walking, running: the partner
  half a cycle later) or together (jumps, squats: at the same time), or
  unknown (no coupling). After stage 2 a limb the sensor does not see takes
  its partner's motion, mirrored at the body's midplane in the frame of the
  facing direction (the world orientations of the segments, not their
  rotations relative to the trunk, which doubled a trunk twist). Where the
  limb shows (half of an arm, a fifth of a leg seen with points) it keeps
  its tracked pose, and the offset from the guess seen there is carried into
  the hidden frames: linearly between two observations up to 3 s apart, so
  the limb meets both, fading over `--offset-memory` (0.4 s) elsewhere. The
  guess is turned about the vertical (0 to +-32 deg, `--shadow-turn`) where
  it would lie in the sensor's sight or in the trunk: the facing of a person
  seen side-on is uncertain by 5 to 10 deg and the mirror doubles it. The
  plan, the weights and the turns change smoothly in time. The hidden limbs
  also start stage 2 from the guess (`--start-from-guess`).
- In stage 2 the free-space test also covers the limb vertices that the
  model hides behind itself (`--free-space-hidden`): the measured ranges, not
  the model, tell what hides a limb.
- Arms are kept out of the trunk in stage 1 and 2 (`--penetration-weight` 2,
  `--penetration-margin` 2 cm); new review flag `arm_in_trunk` (deeper than
  4 cm) and its depth per frame in the npz.
- The facing is guarded: a fit or a search that turns the body faster than
  `--max-turn-rate` (240 deg/s, plus 40 deg) against the second before is
  refused; a frame poorly explained five frames in a row is fitted again from
  the opposite and the side-on facings and the better fit is kept
  (`--facing-check`).
- Stage 2 optimises each window with the 10 frames after it (`--lookahead`).
- `--head-weight` 0.05 to 0.15: heads of people seen side-on turned towards
  the sensor by 30 to 40 deg at times.
- `review-motion` draws guessed limbs pale violet in the side view; the npz
  holds `coupling_mode`, `coupling_confidence`, `guessed` and `guessed_limbs`.
- The `free_space` flag counts the visible vertices only, as before.
- The options `--enabled` and `--window` of the refinement are now
  `--refine-enabled` and `--refine-window` (the coupling has options of the
  same names).

On the synthetic bench (test body, 1024 x 20, default settings), stage 2 and
the completion against stage 2 of 1.4.1 on the same stage 1 (paired; mean
over three noise realisations, then the mean paired difference and its
range):

| Motion | Joint error | Body-part velocity error | Far wrist | Far hand velocity | Near hand velocity | Legs velocity |
|---|---|---|---|---|---|---|
| jumps side-on, 2.3 m | 28.8 then 22.7 mm (-6.1, -6.8 to -5.1) | 0.81 then 0.33 m/s (-0.48, -0.53 to -0.40) | 170 then 87 mm | 2.92 then 1.08 m/s | 0.33 then 0.24 m/s | 0.19 then 0.16 m/s |
| jog across, 2.4 m | 25.9 then 20.4 mm (-5.5, -6.6 to -4.8) | 0.44 then 0.31 m/s (-0.13, -0.19 to -0.04) | 215 then 134 mm | 1.56 then 1.12 m/s | 0.28 then 0.25 m/s | 0.13 then 0.13 m/s |
| walk across, 2.4 m | 18.1 then 13.8 mm (-4.3, -4.8 to -3.5) | 0.106 then 0.106 m/s (0.000, -0.003 to +0.003) | 93 then 34 mm | 0.26 then 0.25 m/s | 0.134 then 0.128 m/s | 0.094 then 0.094 m/s |

The synthetic people move their two sides alike, which favours the
coupling. With one arm swinging 30 % less than the other, 0.06 s later and
its elbow bent 15 deg more (one realisation each): side-on jumps 19.9 then
18.1 mm, 0.221 then 0.207 m/s, far wrist 75 then 54 mm; jog 23.0 then
19.9 mm, 0.37 then 0.31 m/s; walk 14.5 then 11.4 mm, 0.097 then 0.094 m/s,
far wrist 82 then 39 mm. Worse: the velocity of the far forearm in walks
(0.147 then 0.169 m/s; asymmetric walk 0.107 then 0.120), where the 1.4.1
far arm, partly seen, already followed the swing.

The whole pipeline, stage 1 included, against 1.4.1 (one realisation each;
joint error, body-part velocity error, far wrist): jumps after a turn in
place, as in s02_jump02, 24.2 then 18.5 mm, 0.73 then 0.29 m/s, 152 then
69 mm; side-on jumps 27.8 then 22.1 mm, 0.80 then 0.37 m/s, 154 then 86 mm;
jog 26.0 then 19.8 mm, 0.48 then 0.32 m/s, 218 then 140 mm; walk 16.3 then
12.7 mm, 0.103 then 0.104 m/s, 85 then 39 mm. The turn guard and the facing
check did not act on these takes (none of them flipped with 1.4.1 either).
The turn-and-jumps take took 1681 s against 1436 s with 1.4.1; stage 2 with
the completion alone takes 1.3 to 1.5 times as long as stage 2 of 1.4.1
(single thread).

Tried and not kept:
- the coupling as a term of stage 2: with no points on a limb the optimiser
  hardly moves it (a target turned by 20 deg, or a ten times larger weight,
  changed the far wrist by 2 mm);
- mirroring the rotations relative to the trunk: a trunk tracked 10 deg
  twisted put the far wrist about 10 cm off;
- blending the guess by the time to the nearest observation: an arm that
  moves less than its partner (synthetic asymmetric take) got its far hand
  velocity error from 0.51 to 1.22 m/s;
- arms anchored from a quarter seen: far wrist 51 to 87 mm on that take;
- the free-space test of hidden limb vertices in stage 1 too: the backs of
  the limbs along the edges of their silhouette, a pixel outside it, made the
  limb searches noisy (side-on jumps: 86 searches instead of 52, velocity
  error of the legs 0.38 instead of 0.28 m/s after stage 1);
- a plan estimated window by window without smoothing: where a window cut by
  the end of a take missed the cycle, the plan switched between alternating
  and unknown from one frame to the next and the guessed hand jumped by 25 cm
  (walk: body-part velocity error 0.10 to 0.24 m/s); likewise a guess that
  switched from the partner half a cycle later to the mean of before and
  after where the time half a cycle before entered the take (jog: the far
  hand jumped by 30 cm; velocity error of the take 0.43 against 0.33 m/s
  once smooth).

Not validated yet: everything on real recordings (re-run the s02 takes); the
turn guard and the facing check on a real flip (no synthetic take flipped);
the head weight on real heads.

## 1.4.1: the right facing from the first frame; results over three noise realisations (2026-10-17)

- `track` chooses the facing of the first frame, and of a frame where the
  person is lost, by fitting the body as a whole from the best start of each
  distinct facing (`--init-facings` 4, more than 45 deg apart) and keeping
  the fit that explains the points best. Before, the lowest energy of quick
  fits of the root decided, and for a person seen side-on it hardly tells
  front from back: with 1.4.0 one of three noise realisations of a synthetic
  jog across the line of sight was tracked facing backwards for the whole
  take (joint error 274 mm). 1.3.0 picked the right facing on that take by
  about 10 % of the energy, so it could fail the same way. The whole fits
  separate the two clearly (point score 6.5 mm against 18 mm). It costs a few
  seconds per initialisation.
- The help of `--gap-frames` no longer claims a limb search after a gap; a
  test of the free-space penalties.
- Tried and not kept: smoothing the jumps of joints the sensor does not see
  more strongly in stage 2 (quadratic up to 11 times the scale), or treating
  a part without nearby points as unseen. Both moved velocity error from the
  far arm to the arm the sensor sees (side-on jumps: near hand and forearm
  +0.2 m/s in every realisation).

The synthetic numbers quoted for 1.4.0 were single runs. Another noise
realisation of the same take changes the mean joint error by 13 to 34 %
(the largest minus the smallest of three, over their mean), so the versions
are compared on the same realisations. Over three realisations of each take
(test body, 1024 x 20, default settings; mean over the three, then the mean
paired difference with its range):

| Motion | Joint error, 1.3.0 then 1.4.1 | Body-part velocity error | Neck and head angle, p95 | Also |
|---|---|---|---|---|
| walk across, 2.4 m | 18.5 then 18.1 mm (-0.4, range -0.5 to -0.3) | 0.123 then 0.106 m/s | 8.6 then 4.6 deg | trunk twist p95 16 then 8 deg; hidden joints +3 mm |
| jumps facing the sensor, 2.3 m | 16.3 then 13.5 mm (-2.8, -3.1 to -2.6) | 0.184 then 0.138 m/s | 32 then 7.5 deg | legs 0.19 then 0.10 m/s |
| jumps side-on, 2.3 m | 38.1 then 28.8 mm (-9.3, -11.0 to -8.0) | 0.78 then 0.81 m/s (+0.03, -0.02 to +0.05) | 77 then 8.8 deg | trunk 0.195 then 0.144 m/s; far hand and forearm 2.28 then 2.42 m/s |
| jog across, 2.4 m | 28.1 then 25.9 mm (-2.2, -2.8 to -1.5) | 0.41 then 0.44 m/s (+0.03, -0.01 to +0.10) | 10.0 then 7.3 deg | legs 0.16 then 0.13 m/s; far hand and forearm 1.14 then 1.26 m/s; hidden joints +7 mm |
| jog towards the sensor, 4 to 2 m | 29.0 then 21.5 mm (-7.5, -7.9 to -6.9) | 0.68 then 0.54 m/s | 7.5 then 4.3 deg | hidden joints 107 then 38 mm |

The arm on the far side of the body, which the sensor does not see, stays a
guess: its velocity error is about that of an arm held still against the
trunk (hand 1.3 m/s in the jog across, 2.3 m/s in the side-on jumps), and in
those two motions it decides the total velocity error. Tracking takes 7 to
47 % longer than with 1.3.0 (single thread; most for the side-on jumps).

## 1.4.0: tracking without twisted trunks, heads and phantom limbs (2026-10-16)

The review videos of the 27 s02 takes (tracked with 1.3.0 on the DEI
cluster) showed bodies twisted out of shape when the person was side-on
(s02_jump02: 12 to 22 % of the body seen from the LiDAR where the sensor
measured nothing), heads and arms at odd angles, forearms at waist height
not followed in running, and a leg thrown away after a 0.65 s data gap.
Changes to `track` (all of them can be switched back with their options):

- The frame fit runs 30 L-BFGS iterations per round instead of 12, which
  stopped before convergence (`--iterations`).
- Spine, neck and head keep adult ranges of motion (bodyscan.body.skeleton):
  trunk twist about +-40 deg and side bend about +-45 deg in all (was up to
  +-95 and +-80), head turn about +-80 deg (was +-130).
- New weak priors towards neutral: neck and head (`--head-weight` 0.05),
  forearm twist and wrists (`--forearm-twist-weight` 0.02).
- Free space: the penalty adds a term at four times the silhouette scale, so
  that a limb 20 to 50 cm out in empty space is still pulled back; its bound
  is unchanged, so leaving the points never pays (`--free-space-loss wide`;
  `bounded` is the old one). Columns lost with a UDP packet are unknown, no
  longer empty space.
- The search of a poorly fitted frame also runs when more than 3 % of the
  visible body lies in free space (`--recover-free-space`), also tries the
  trunk, neck and head untwisted and the body turned by +-35 deg
  (`--torso-search`, `--yaw-search`), searches only the limbs near the misfit,
  has four new arm starts (elbow bent with the forearm forward, running swing
  forward and back, swing forward and up), moves the limbs coarse to fine,
  and accepts a mostly hidden arm when it explains points or leaves empty
  space.
- After a gap of more than 2.5 frame periods the prediction keeps the last
  pose (`--gap-frames`).
- The `free_space` review flag fires at 3 % of the visible body (was 5 %).

Synthetic takes with truth (test body, 2-core CPU, joint error MPJPE mean;
1.3.0 then 1.4.0): jumps side-on with arm swings 35.8 then 27.8 mm (p90 76
then 49 mm, head turned up to 82 then 10 deg), jumps facing the sensor 14.7
then 11.6 mm (head 34 then 9 deg), walking towards the sensor with jogging arms
27.7 then 19.8 mm, the same across the line of sight 26.3 then 24.1 mm, a
walk across the line of sight 16.8 then 16.3 mm (trunk twist p95 16 then 8
deg). The arm hidden behind the body stays a guess (hanging) in both.

## Tracking: progress of stage 2, precision option (2026-10-14)

- The progress of the whole-sequence refinement counted correspondence rounds
  instead of windows ("2/5" printed for every window, with a wrong remaining
  time); it now counts windows.
- `track` prints which frames it tracks (`--start`, `--stop`) and the precision.
- New `--precision auto|float32|float64` (auto: float32 on a GPU, float64 on a
  CPU). On a CPU float32 made the refinement about 30 % faster on a synthetic
  walk (79 frames), with the same mean joint error (18.9 mm both).

## Body models live with the code (2026-10-14)

- `C:\lidar` is data only. The SMPL-X files go in the project's `models`
  folder (ignored by git, never committed); `--model SMPLX_MALE.npz` finds
  them by file name, also when the avatar remembers another path.
- Documentation examples use `--model SMPLX_NEUTRAL.npz`.

## Beeps through one open audio stream (2026-10-13)

- The previous fix (longer beeps, wake pulse) did not make the earbuds play
  every beep. Beeps are now mixed into one output stream (sounddevice) that
  stays open for the whole capture, so Bluetooth earbuds never suspend the
  link; `--beep-backend auto|stream|winsound`. sounddevice is added to the
  capture extras; without it the old winsound path is used.
- New command `test-beeps`: plays the capture cues without the sensor.

## Audible beeps through earbuds (2026-10-13)

- Beeps are queued and played one after the other, each at least 350 ms
  (`--beep-min-ms`); a faint wake pulse precedes a beep after a pause
  (`--no-beep-wake` to disable). Hypothesis: wireless earbuds clip the start
  of short sounds; not yet confirmed on the user's earbuds.

## check-take, lost columns per packet (2026-10-13)

- New command `check-take TAKE SEGMENTS`: columns received per frame, which
  UDP packets are lost and whether the same ones are lost in consecutive
  frames, frames without a person, distance of the person between the beeps.
- Frames now store `column_ok` (which of the columns arrived), so that the
  cause of lost columns (random: receiving PC or network; fixed packets:
  sensor or cable) can be told from the data. Older takes only have the
  fraction.

## Capture: time to leave the scene, sound cues for motion takes (2026-10-13)

- `capture-motion`, `capture-inplace`: `--leave-seconds` (default 10 s) before
  the empty-scene recording, so the operator can walk out of the sensor's view
  after starting the command (one long low beep, then low ticks every second;
  nothing stored). `capture-turntable` has the option too (default 0); in
  two-PC mode it is subtracted from the wait for the platform timer.
- The empty scene starts with two low beeps.
- `capture-motion`: beeps during the take: one long (stand still in the
  A-pose, `--hold` 3 s), two short high (move), one short every `--cue-every`
  s (change movement, off by default), three medium `--back-seconds` (8 s)
  before the end (go back to the mark), one long (A-pose for the last
  `--hold` s), two at the end. The cues are logged in capture.json.

## Tracking: clutter and hidden arms (2026-10-10)

- `track`: points farther than `max_correspondence` (25 cm) from the body are
  clutter: they no longer count as unexplained, trigger no limb search and no
  `unexplained` flag; their mean fraction is `clutter_fraction_mean` in the
  summary and `clutter` in the motion file. On a real take with debris in the
  segmentation the old count made the tracker bend limbs towards it.
- `track`: tracking starts with the arms hanging instead of the scan's A-pose,
  and an arm seen less than 30 % is pulled slowly towards hanging
  (`--relax-hidden-arms`, default 0.1, 0 to switch off): a hidden arm no
  longer stays raised for the whole take.
- Synthetic benchmark (test body, default settings): walk MPJPE mean
  25.6 to 21.2 mm, max 53.7 to 41.6 mm; jump unchanged (14.0 and 14.3 mm).

## Documentation (2026-10-08, third pass)

- Algorithm Lab reorganised in the order the commands run, eight chapters:
  1 place the sensor (`check-view`), 2 find the object (foreground, the
  tests: rotation, people, the cascade step by step), 3 isolate it in every
  frame, 4 platform angle, 5 fuse, 6 Poisson, 7 watertight, 8 smooth. A
  pipeline map grouped by command at the top; every chapter opens with the
  same card (question, command, code and parameters), lettered tabs, and a
  link to the next step.
- New chapter 1, `check-view`: the real fan of beams of the scan on the real
  side profile of the body (head and feet in view, the verdict check-view
  prints, as height, distance, tilt and stature change), and the smallest
  visible gap (footprint and sample steps against a gap between two
  surfaces, vertical and horizontal). Animations `lab_checkview.gif`,
  `lab_gap.gif`.
- docs/algorithms.md in the same order (new section 2, check-view; the finder
  before the isolation), with a table mapping every step to its command,
  section and Lab chapter; tuning.md and README follow the chapters.

## 1.3.0 (2026-10-08)

### Changed: one mechanism finds the object of interest, with no region of the room assumed

- New `detection.finder` (`ObjectFinder`, `FoundObject`, `SelectConfig`) and
  `detection.selectors` (`Selector`, `NearSelector`, `HumanSelector`,
  `RotationSelector`, `SelectorChain`): foreground against the empty scene,
  objects followed across frames, then a chain of tests chosen by flags.
  `detect`, `fuse` and `fuse-inplace` all use it, with the same flags
  `--rotation`, `--human`, `--near X Y` (section `[select]`).
- `fuse`: the object on the platform is the one turning about a vertical
  axis by at least `--min-total-turn` (320 deg) by default; its rotation axis
  starts the axis fit. The platform ring is an optional refinement near that
  axis (`--ring-radius 0` switches it off; `--ring-search` now 0.3 m). The
  fallback to the platform centre of the 2026-09-30 calibration and
  `--search-start` are removed: if no object passes the tests, `fuse` stops
  with the table of objects and the reason each one failed.
- `fuse` and `fuse-inplace`: the region of interest is measured on the
  object found (its reach over the frames plus `--margin`, 0.15 m) instead of
  a fixed radius of 0.9 m; `--radius` (now not set by default) still imposes
  one. New section `[subject]`: `--search-frames`, `--margin`,
  `--body-depth`, `--min-total-turn`.
- `fuse-inplace`: the person is found with `--human` by default; `--near X Y`
  chooses among several people; `--center` and the crop box still impose
  the region.
- Rotation test: pairs slower than `min_speed` or turning less than
  `still_turn` (1 deg) are still pairs, kept out of the statistics (a
  turntable that waits before turning is now recognised as rotating); the
  turn covered (`turned_deg`) can be required with `min_total_turn`. Long
  pairs are searched around the sum of the neighbouring turns they span.
- `detect`: the steps `SegmentAndTrack` and `Classify` become `FindObjects`;
  the frames are read one at a time; the flags are configuration options
  (`--rotation/--no-rotation`, TOML `[select]`). `pipelines.common.locate_person`
  is replaced by `find_subject` and the step `FindSubject`; `check-view` uses
  it too.
- Tests: `--near`, every object reported without flags, a turntable track
  with still parts before and after a lap, the whole-lap requirement, the
  turntable finding its turning person and measured region.

### Documentation

- Algorithm Lab: a new first tab in chapter 2, "Frames → angles → fused
  cloud", on the real scan: a frame sees one side, ICP measures a turn, the
  chained turns drift, the motor model fitted to all pairs, every frame
  turned back by its angle, and the same views with chained or constant
  speed angles (smeared); the least-squares simulation is now the last tab,
  without overlapping labels; every step of the watertight remesh animates
  and plays when selected; a new tab animates the person cascade stage by
  stage (height slices, occupancy image, integral image, the four look-ups
  of every Haar-like rectangle, the scores); chapters 1 and 7 describe the
  finder and the measured region.
- algorithms.md, architecture.md, tuning.md, README: the finder, the
  selectors and the flags; new animations `lab_story.gif` and `lab_haar.gif`.

## Documentation (2026-10-08, second pass)

- Algorithm Lab: the whole isolation chain on a real frame (why background
  subtraction alone leaves false alarms and which test removes them); how
  the registrations and the motor model work together (simulation on a known
  truth: chained, long and revisit pairs, motor model); the watertight
  remesh step by step (grid, distance, sign by ray parity with an animated
  sweep, extraction, a hole and the majority of 5 rays); smoothing step by
  step with the filter window and weights, and reflection lines on a real
  chest patch smoothed live in 3D; a new chapter on `detect`: the rotation
  test and the person cascade, computed by the package on real data.
- docs/algorithms.md: the same additions as text and animations.

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
