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
python -m bodyscan avatar C:\lidar\s01_scan.ply --model SMPLX_NEUTRAL.npz --out C:\lidar\s01_avatar

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
| SMPL-X model | register at https://smpl-x.is.tue.mpg.de, download the npz models (`SMPLX_NEUTRAL.npz`, or male/female) | the body model. The licence allows non-commercial research and forbids redistribution: put the files in the project's `models` folder, not in the data folder `C:\lidar`; commands find them by file name (`--model SMPLX_MALE.npz`), and `.gitignore` excludes `SMPLX_*` and every `.npz`, so they are never committed |

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

`capture-motion` gives 10 s to walk out of the sensor's view after the
command is started (`--leave-seconds`), records 10 s of empty room
(`--background-seconds`; nobody in view), counts down 10 s (`--delay`), then
records until ENTER or for `--duration` seconds. The person cannot watch the
screen, so every step has its beeps:

| Beeps | When | What to do |
|---|---|---|
| 1 long low, then a low tick every second (higher in the last 3) | leave (`--leave-seconds`) | walk out of the sensor's view |
| 2 low | empty room starts (`--background-seconds`) | stay out of view, silence until the countdown |
| 1 every second (higher in the last 3) | countdown (`--delay`) | walk to the floor mark, take the A-pose |
| 1 long | recording starts | stand still in the A-pose for `--hold` s (3 s) |
| 2 short high | after the hold | move |
| 1 short | every `--cue-every` s (off by default) | change movement |
| 3 medium | `--back-seconds` (8 s) before the final hold | go back to the floor mark |
| 1 long | the last `--hold` s | A-pose, stand still |
| 2 | end | done |

Beeps are played one after the other and each lasts at least `--beep-min-ms` (350 ms).
With Bluetooth earbuds, install sounddevice (`pip install sounddevice`): one audio
stream then stays open for the whole capture and the earbuds never suspend the
link between beeps (`--beep-backend stream`). Without it, each beep opens the audio
device again and Bluetooth earbuds can swallow the first beeps. Check before a
session, without the sensor:

    python -m bodyscan test-beeps

It plays the cues one by one with their numbers on screen.

Without `--duration` (recording until ENTER) only the cues up to "move" are
given: stand 3 s in the A-pose before pressing ENTER. The empty room must
really be empty: a person in it becomes part of the background and is erased
from the take (the person2 take of 2026-09-30 lost the torso this way).

After every take, check it before leaving the site:

```
python -m bodyscan segment-motion C:\lidar\open\s01_walk01 --out C:\lidar\open\s01_walk01_seg
python -m bodyscan check-take C:\lidar\open\s01_walk01 C:\lidar\open\s01_walk01_seg
```

`check-take` reports the columns received (a lost UDP packet is 16 columns,
a hole 5.6 deg wide: 0.25 m at 2.5 m; `segment-motion` drops frames below 90 %
of the columns), which packets are lost and whether the same ones are lost
again in the next frame (random: the receiving PC or network; fixed: a
sensor or cable cause), the frames without a person, and the distance of the
person in every window between the beeps (1.5 to 4 m is the working range).
Takes recorded before 2026-10-13 have no per-column record: the report then
gives only the fraction of columns.

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
| silhouette, free space | no part of the body where the sensor saw through to the background: a vertex that projects onto a pixel whose measured range lies beyond it (outside the crop: where the empty room lies beyond it and no person was found) is pulled towards the person's silhouette (a distance map of the crop, in angle, times the range). A vertex behind furniture is not pulled. Columns lost with a UDP packet (no return in the whole crop height where the empty room returns) are unknown, not empty. The penalty (`--free-space-loss wide`) is a robust term at `--silhouette-scale` (6 cm) plus one at four times that scale: a limb 20 to 50 cm out in empty space is still pulled back, while the bound stays that of version 1.3, so leaving the points altogether never pays (`bounded`: the robust term alone, version 1.3) |
| in front | no part of the body in front of the measured surface (same penalty) |
| limits | the anatomical joint limits (bodyscan.body.skeleton: knees and elbows bend one way, and so on). Since version 1.4 the spine, neck and head keep adult ranges: trunk twist about ±40 deg and side bend about ±45 deg in all, head turn about ±80 deg, tilt ±45 deg |
| twist | the twist of the spine and the rotation of the thighs about their own axis stay small (weak, `--twist-weight`): otherwise the pelvis can turn while the trunk and the legs turn back by as much, which the points hardly tell apart |
| head | the neck and head rotations stay near upright and facing forward (weak, `--head-weight`): the head is round, its points hardly tell how it is turned |
| forearms | the twist of the forearms about their axis and the wrist rotations stay near rest (weak, `--forearm-twist-weight`) |
| floor | no foot vertex below the floor |
| continuity | joint rotations and root close to the prediction; weak for what the sensor sees, up to 11 times stronger for a body part seen less than 30 % (`--hidden-continuity`), so that a hidden limb keeps its motion instead of jumping wherever a few points pull it |
| hidden arms | an arm seen less than 30 % is pulled slowly towards hanging (`--relax-hidden-arms`, 0 to switch off): an arm nobody sees is most likely hanging; tracking starts with the arms hanging, so that the scan's A-pose does not stay on an arm the sensor never sees |
| trunk | no arm inside the trunk (since version 1.5, `--penetration-weight` 2): samples of the forearms, the hands and the lower halves of the upper arms deeper than `--penetration-margin` (2 cm: soft tissue, clothes, an avatar a little wider than the person) inside the trunk are pushed out. The depth of a sample is its largest distance to the tangent planes of its four nearest trunk samples (pelvis to chest, without the armpits), negative inside. Nothing else kept a hidden arm out of the body: the sensor cannot see inside it, so the points never objected |

Each frame is fitted with `--iterations` (30) L-BFGS iterations per round;
version 1.3 stopped at 12, before the fit converged: on synthetic jumps the
joint error was about 14 mm instead of 11 mm, the head turned by up to 20 deg
for nothing, and the unconverged frames needed limb searches.

The first frame is fitted from 8 facing directions (`--init-yaws`): a quick
fit of the root from each, then the best start of each distinct facing (the
`--init-facings` 4 with the lowest energies, more than 45 deg apart) is fitted
as a whole, and the fit that explains the points best is kept. Before version
1.4.1 the lowest energy of the quick fits decided, and for a person seen
side-on it hardly tells front from back: a synthetic jog was tracked facing
backwards for the whole take (joint error 27 cm), while the whole fits
separate the two clearly (score of the search below: 6.5 mm facing the right
way, 18 mm facing backwards, of which 9 mm for 6 % of the body in free
space). A frame whose points lie
more than `--lost-residual` (5 cm, median) from the body is fitted again from
several directions in the same way. After a gap of more than `--gap-frames`
(2.5) frame periods (frames dropped by the segmentation) the prediction keeps the last
pose instead of extrapolating its velocity, which threw a moving leg away
after a gap of 0.65 s. When more than `--recover-fraction` (2 %, and at
least 10) of the points remain farther than `--recover-distance` (6 cm) from
the body after the fit, a limb is probably in the wrong place: in a jump the
arms swing overhead in 0.3 s and the knees fold at the landing, faster than
the prediction follows. Likewise when more than `--recover-free-space` (3 %)
of the visible body lies where the sensor saw empty space: an arm stretched
into the air, or the trunk turned the wrong way when the person stands
side-on. The frame is then fitted again from other starts: the previous
frame's pose; the trunk, neck and head untwisted, with the whole body as it
is and turned by ±`--yaw-search` (35 deg) (`--torso-search`); both legs from
three poses (standing, half squat, deep squat); each arm from twelve
(A-pose, hanging, elbow bent sideways, out to the side, forward, up in
front, up at the side, back, and the arm carriages of running and jumping:
elbow bent with the forearm forward, swung forward, swung back, swung forward
and up). Only the limbs near the misfit are searched: each unexplained point
counts for the limb with the nearest elbow, wrist, knee, ankle or foot, each
vertex in free space for its own limb. Each start first moves only its limb,
coarse to fine, and is kept when it explains the points better (mean point
distance capped at 15 cm, plus a penalty for body in free space, 2 mm better
at least); an arm only when the sensor sees at least 20 % of it there, or
when the move explains points left unexplained (the hand of the far arm over
the head) or takes the arm out of empty space, since no point can justify
moving an arm the sensor does not see. Without this search a lost limb can
stay wrong for the rest of the take (an arm hidden behind the body, legs
landing feet forward instead of knees forward), because nothing pulls it
back.

The facing of the body is guarded since version 1.5. A person does not turn
around between two frames, but a search or a fit from several directions on
a frame whose points are hard to explain can: in the review of s02_jump02
(version 1.4.1) the body turns by about 180 deg within 0.2 s at the first
take-off and is tracked facing backwards, contorted, until the end. A result
that turns the body faster than `--max-turn-rate` (240 deg/s, plus 40 deg)
against any frame of the second before is refused, and the plain fit from
the prediction is kept. The other way round, a body that does face the wrong
way stays poorly explained: when a frame is still poorly explained after the
searches five frames in a row (more than twice `--recover-fraction` of the
points unexplained, or more than twice `--recover-free-space` of the visible
body in free space), it is fitted afresh from the opposite facing and the two
side-on ones (`--facing-check`, at most every ten frames), its limbs searched
where points stay unexplained, and that fit is kept, turn or not, when it
explains the points clearly better (score 3 mm lower).

Stage 2 refines the whole sequence in windows of 40 frames overlapping by 10,
each optimised together with the `--lookahead` 10 frames after it (refined
again with the next window), so that the end of a window sees what follows:
the same terms for every frame (coarse to fine again), plus

| Term | What it asks |
|---|---|
| smoothness | the accelerations of the joints change smoothly: a Charbonnier penalty on the jerk (weight `--smoothness-weight` 0.5; quadratic below `--smoothness-scale`, 200 m/s³, linear above), which removes frame-to-frame jitter and keeps sharp real events such as a landing; up to 11 times stronger for a body part seen less than 30 % (`--hidden-smoothness`). A penalty on the acceleration itself (`--smoothness acceleration`) would also pull a jump's flight, a free fall at 9.81 m/s², towards a straight line |
| contact | a foot found standing (lowest vertex below 3.5 cm and ankle slower than 0.35 m/s in stage 1) does not slide |
| rolling shutter | every point and every silhouette pixel is compared with the body at the time it was measured (the body is moved along its motion between the neighbouring frames) |

After stage 2, the limbs the sensor does not see are completed from their
partner on the other side (since version 1.5, below).

#### Hidden limbs: the left-right coupling

A person seen side-on shows the sensor one arm and mostly one leg. Until
version 1.4 the far arm was held by continuity and smoothness and pulled
slowly towards hanging: it kept still while the other one swung, drifted to
the side, or sank into the trunk, which the sensor cannot see into. Since
version 1.5 it is guessed from its partner (bodyscan.dynamic.coupling). Most
activities move the two arms, and the two legs, in one of two ways:
*alternating* (walking, running, running on the spot: each limb repeats its
partner's motion, mirrored, half a cycle later; the pelvis rises and falls
twice per cycle, once per step) or *together* (jumps, squats, both arms
raised: each limb mirrors its partner at the same time; the pelvis rises and
falls once per cycle). Which one, and the cycle, is estimated around every
frame (`--coupling-window`, ±2 s) from what the sensor does see: the
best-observed leg and arm repeat their cycle (the autocorrelation of their
angles peaks at the cycle T, or, for a burst of three strides, the times
between their peaks); the autocorrelation of the pelvis height at T/2 is then
positive for alternating and negative for together; when both arms or both
legs are observed, their correlation says it directly; a squat or a single
jump (the pelvis drops while the hips flex) counts as together. Where none of
this is clear (standing, gestures, a turn), no limb is coupled. The summary
of `track` gives the share of frames of each kind. The plan changes smoothly:
near the ends of a take the window is moved inwards rather than cut short,
the cycle is held steady (a running median over a second), and the
confidence fades out over 0.25 s before every change of mode and in after it.
A first version let the plan switch between alternating and unknown from one
frame to the next where a cut window missed the cycle, and the guessed hand
jumped by 25 cm there (5 m/s in a synthetic walk).

The guess for a hidden limb is its partner at the same time (together) or
half a cycle before and after (alternating; the two blended by how well the
partner was observed then, changing smoothly, also where the time half a
cycle before enters the take), mirrored at the body's midplane in the frame
of the facing direction. The world orientation of the
partner's segments is mirrored, not their rotations relative to the trunk: a
trunk tracked 10 deg twisted would otherwise put the far arm 20 deg off (the
far wrist about 10 cm away in a first version). Where the limb does show (a
hand appearing in front of the trunk, the far leg between the legs; half of
the limb seen with points) it keeps its tracked pose, and the offset from the
guess seen there is carried into the hidden frames: between two
observations up to 3 s apart it changes linearly from one to the other, so
the limb meets both and in between moves like its partner (in a short gap
that is almost the plain interpolation of the two observations; across a
long one the partner's swing); before the first observation, after the last
and in longer gaps it fades with exp(-dt / `--offset-memory`, 0.4 s), so far
from any observation the limb moves like its partner. A person whose two
arms move differently (one swinging less) keeps that difference where the
far arm was seen. The facing of a person seen side-on
is uncertain by 5 to 10 deg (the points of the side of a trunk hardly tell:
on the synthetic side-on jumps the tracked facing is 5 deg off in the median,
and a pose turned to the true facing does not explain the points better),
and the mirror doubles it, which puts the far hand 10 to 20 cm out of the
shadow of the body. So the guess is turned about the vertical where needed:
of 0, ±8, ±16 ... ±`--shadow-turn` (32 deg; legs half), the turn that puts
the fewest of the limb's vertices where the sensor would have seen them
(empty space beyond, or a measured surface behind), with an arm not inside
the trunk and a small turn preferred. The turns of successive frames are
chosen together, as the path of least cost with a cost for every change, and
smoothed over 0.2 s: a hand 0.6 m from the shoulder moves 1 cm per degree of
turn, and turns chosen frame by frame made it jitter. Where the plan is
unsure the guess is blended in partly.

The guess is applied after stage 2, not as a term of it. With no points on
a limb, the energy of a window is dominated by the data of the rest of the
body and the optimiser hardly moves the limb (on a synthetic side-on jump a
target turned by 20 deg, or a ten times larger weight, changed the far
wrist by 2 mm). The hidden limbs do start stage 2 from the guess, so that a
far hand that appears in front of the trunk is near its points. The
free-space term of stage 1 and 2 now also covers the limb vertices that the
body model hides behind itself (`--free-space-hidden`): what hides a limb
from the sensor is told by the measured ranges, not by the model.

Output `<out>.npz`: for every tracked frame its time (sensor clock), the
root, the body joint rotations (a PoseSequence: bodyscan.dynamic.motions),
the 55 joints, the foot contacts, per body part the fraction the sensor
observed, the point residuals and the review flags. `<out>.json` summarises
it.

Flags (a frame can have several): `residual` (points more than 3 cm from the
body, median), `unexplained` (more than 5 % of the points between 5 cm
and `max_correspondence`, 25 cm, from the body: a limb in the wrong place;
farther points are clutter no limb can reach, reported as
`clutter_fraction_mean` in the summary and never a reason to search the
limbs again), `free_space` (more than 3 % of the
visible vertices where the sensor saw through; 5 % before version 1.4, which
let a whole arm in empty space pass unflagged; the hidden limb vertices that
the fit also tests since version 1.5 do not count here, since the back of
every limb along the edge of its silhouette falls a pixel outside it),
`below_floor` (a foot 2 cm
below the floor), `sliding` (a standing foot moving faster than 0.3 m/s),
`joint_limits` (a limit exceeded by more than 10 deg), `acceleration` (a
joint accelerating more than 120 m/s²), `arm_in_trunk` (an arm more than
4 cm inside the trunk; the depth per frame is `arm_in_trunk` in the npz).
The npz also holds the coupling of every frame (`coupling_mode`: 1
together, -1 alternating, 0 none; `coupling_confidence`) and how much of
each limb was guessed from its partner (`guessed`, 0 to 1, limbs in
`guessed_limbs`); `review-motion` draws a limb guessed more than half pale
violet in the side view.

### `review-motion`: human supervision

The tracked body seen from the sensor and from the side, with the person's
LiDAR points coloured by their distance to the body (green on the surface,
yellow at 2 cm, red at 5 cm and more). `review.gif` shows every `--every`-th
frame; every flagged frame gets a PNG. Look at the flagged frames first, then
at the GIF for unnatural motion (a hidden arm that freezes or swings
oddly). In the side view a limb guessed from its partner (the sensor did not
see it, version 1.5) is pale violet: its motion is inferred, not measured.
There is no automatic reference: this review is the check.

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

The table above was measured with version 1.3. Version 1.4.1 against 1.3.0 on
five synthetic motions with the exact body, each over three noise
realisations (the table is in CHANGELOG.md): the joint error is lower in all
five (by 0.4 mm in a walk up to 9 mm in side-on jumps); the body-part
velocity error is lower by 14 to 25 % in a walk, in jumps facing the sensor
and in a jog towards it, and unchanged within the spread in side-on jumps and
in a jog across the line of sight, where the far arm decides it; neck and
head stay within 9 deg (p95) instead of turning by up to 77 deg. One run is
not enough to compare settings: another noise realisation of the same take
changes the mean joint error by 13 to 34 %, so compare settings on the same
realisations and over several of them.

## 7. Limits

**One sensor sees one side.** The body parts turned away from the LiDAR (the
far arm while walking past it, the back while facing it) give no points. The
tracker keeps them plausible (limits, the silhouette and free space, smooth
accelerations, arms out of the trunk) and, since version 1.5, guesses a
hidden arm or leg from its partner on the other side (section 4); but their
motion is not measured. The guess assumes that the two sides move alike,
mirrored, at the same time or half a cycle apart: right for walking, running
and jumping; for a gesture of one arm the plan is unknown and the hidden arm
is not coupled; a person who swings one arm less keeps the difference only
where that arm was seen. The bench shows how much worse hidden joints are than
seen ones. `<out>.npz` stores per frame and per body part the fraction that
was observed, and per limb how much of it was guessed, so that a radio study
can tell measured from inferred motion. If the radar will stand next to the LiDAR, the
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
model = avatar.model("SMPLX_NEUTRAL.npz", "cpu")
body = avatar.shaped(model)                                  # the person's body, at the avatar's level
times = motion.times[0] + np.array([3.0, 3.005])             # 3 s into the take, and 5 ms later
vertices, joints = motion.sample(times).posed(body)          # (2, V, 3) and (2, 55, 3), metres
```

Joint rotations are axis-angle vectors in the parent joint's frame, as in
SMPL-X; the root rotation and position are in the world (floor) frame, with
the pelvis as the root position. The signs of the joint rotations are
documented in bodyscan/body/skeleton.py and checked by the tests.
