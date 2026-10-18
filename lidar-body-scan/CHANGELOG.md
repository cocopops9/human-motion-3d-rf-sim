# Changelog

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
far arm, partly seen, already followed the swing. Stage 2 and the
completion take 1.3 to 1.5 times as long as the stage 2 of 1.4.1 (single
thread).

END-TO-END

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
turn guard on a real flip (the synthetic turn-and-jumps take did not flip
with 1.4.1 either); the head weight on real heads. The synthetic people
move their two sides symmetrically, which favours the coupling; the
asymmetric takes (one arm swinging less, later and more bent) check the
other case.

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
