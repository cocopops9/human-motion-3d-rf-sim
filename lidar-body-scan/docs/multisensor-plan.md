# Multi-sensor capture and reconstruction: plan for bodyscan 2

Status: accepted, 2026-10-21; first part implemented in bodyscan 1.7.0 (section 16). Owner: Giuseppe D'Auria (Università di Padova, IMDEA Networks). Executor: Claude Code on the recording laptop. Where this plan and docs/motion.md disagree, this plan wins for new work; docs/motion.md stays the reference for the single-LiDAR tracker 1.6.0.

## 0. Summary

The single Ouster cannot see the far side of a moving person, so bodyscan 1.6.0 guesses the hidden limbs. The lab now has two more sensors: a RoboSense Helios 16 LiDAR and an ASUS Xtion depth camera. This plan turns bodyscan into a sensor-agnostic system:

* any combination of these sensors records one session;
* the sensors are calibrated and synchronised;
* one tracker fits one personal avatar to all observations;
* every result carries quantitative error measures and a per-vertex error heatmap;
* an interactive viewer replaces the review videos.

The order of work is: hardware bring-up, recorder, synthetic bench to choose the placement, calibration and synchronisation, the multi-sensor tracker, a better avatar, then the recording day. Metrics and the viewer can start at once on the existing single-LiDAR data.

## 1. Requirements and constraints

### 1.1 Functional requirements

| Id | Requirement |
|---|---|
| F1 | Reconstruct the motion and the body shape of a moving person as accurately as the sensors allow |
| F2 | Work with any subset of {Ouster OS0-128, Helios 16, Xtion}: 1 LiDAR, 2 LiDARs, 1 LiDAR plus depth camera, all three. New sensor types (for example a true time-of-flight camera) must be addable without touching the tracker |
| F3 | With only the Ouster on the existing s02 takes, reproduce the 1.6.0 tracks (regression test) |
| F4 | Personal avatars that fit the person's shape better than SMPL-X plus displacements, with wrist rotation and better joint placement where the data support it |
| F5 | Quantitative metrics of fit quality, outliers and plausibility, per take, per frame and per body part, plus a body heatmap of the error |
| F6 | Interactive viewer: choose a take, toggle points per sensor, avatar, heatmap and plots |
| F7 | Unchanged output for Sionna RT: animated meshes with per-vertex velocities and body tracks (`export-motion`) |

### 1.2 Non-functional requirements

| Id | Requirement |
|---|---|
| N1 | Computation time is not a constraint; use the UNIPD DEI cluster (SLURM, A40, RTX and L40S GPUs) and, if available, an IMDEA cluster |
| N2 | Storage is not a constraint; raw data goes to the NAS, which has a cable link to the cluster and is reachable from the laptop (to be confirmed) |
| N3 | Clean, documented, tested code (layered architecture, typed interfaces, unit and regression tests, changelog with measured results) |
| N4 | Recording must be reliable on Windows laptops; packet or frame loss must be measured, not assumed |
| N5 | Every result states what was measured and what was inferred; hidden body parts are labelled as such everywhere |

### 1.3 Constraints and decisions stated by the user (21 Oct 2026)

* No ground truth will be available (no MoCap, no IMUs). Validation is synthetic plus held-out-sensor residuals (section 8).
* The radar is not used during motion takes; it will be used later for static people only. Placement is therefore free.
* The recording happens in the large open space, with enough room; the same motion protocol as on 13 and 14 Oct (walk01, walk02, walk03 radial, run01, runspot, jump01, jump02, natural01; three takes each).
* All takes of a session are recorded on one day, so one calibration serves them all.
* Six turntable poses per subject, two to three laps each, are acceptable.
* Publication is a paper only; research licences (SMPL-X, SKEL) are acceptable, as is MHR (Apache 2.0).
* Fingers would be a plus, not a requirement. Only depth is needed from the ASUS camera.
* The recording PC is the Lenovo laptop (Windows), with a NOVOO NH08S-C800 USB-C hub providing the Ethernet port and two USB 3.0 ports; a second or third Windows PC can be added if needed.
* Helios 16 power: the original 12 V adapter is missing; the owners already run the Helios from a 19 V, 2.0 A (38 W) adapter, within its 9 to 32 V rating.
* Claude Code may act autonomously on the laptop and the cluster.

## 2. Sensors

### 2.1 Specifications

| Item | Ouster OS0-128-SR | RoboSense Helios 16 (H16, S/N 1007BCE00015) | ASUS Xtion (most likely PRO LIVE) |
|---|---|---|---|
| Principle | spinning LiDAR, 865 nm | spinning LiDAR, near-infrared (905 nm is typical for RoboSense; not verified for this unit) | structured light (PrimeSense PS1080 class), infrared dot projector (about 830 nm, moderate confidence); not time of flight |
| Channels and field of view | 128 rings, about 90° vertical (±45°), 360° | 16 rings, 30° vertical (−15° to +15°), 2° spacing, 360° | 640 × 480 depth pixels, 58° H × 45° V |
| Horizontal sampling | 1024 columns at 20 Hz (0.352°), 2048 at 10 Hz | 0.1°, 0.2°, 0.4° at 5, 10, 20 Hz | 30 fps |
| Range | used 1.5 to 4 m | 0.2 to 150 m | 0.8 to 3.5 m |
| Noise | precision ±0.8 to ±4 cm (datasheet); 5 to 10 mm fit residuals on the body | precision 1 cm typical; accuracy ±2 cm (1 to 100 m) | grows with distance squared: about 6 mm at 2 m, 10 mm at 2.5 m, 14 mm at 3 m (scaled from Kinect v1 studies, moderate confidence); quantisation about 10 mm at 2 m |
| Time stamps | per column, sensor clock; PTP capable | per packet, sensor clock; PTP, gPTP, GPS plus PPS capable | per frame, device clock; no trigger input; latency about 1.5 frames |
| Interface | Ethernet (gigabit), UDP 7502 (lidar), 7503 (IMU) | Ethernet via interface box (100 Mbit), UDP MSOP 6699, DIFOP 7788 | USB 2.0, OpenNI2 |
| Power | interface box | 9 to 32 V, 11 W typical | USB |
| Data rate (raw) | about 32 MB/s (115 GB/h) at 1024x20 | about 1 to 2 MB/s (estimate, to be measured) | 18.4 MB/s uncompressed (66 GB/h), less with lossless compression |

### 2.2 Sampling geometry

Spacing between neighbouring samples (mm), from the angular steps above:

| Distance | Ouster ring | Ouster column (1024) | Helios ring | Helios column 0.2° / 0.4° | Xtion pixel |
|---|---|---|---|---|---|
| 1.8 m | 22 | 11 | 63 | 6 / 13 | 3.1 |
| 2.3 m | 28 | 14 | 80 | 8 / 16 | 4.0 |
| 3.0 m | 37 | 18 | 105 | 10 / 21 | 5.2 |
| 3.5 m | 43 | 21 | 122 | 12 / 24 | 6.1 |
| 3.9 m | 48 | 24 | 136 | 14 / 27 | (out of range) |

Full-body coverage constraints:

* **Helios.** Distance d must satisfy d ≥ max(h, H − h) / tan 15°, with h the sensor height and H the top of the head. For H = 1.80 m the best height is 0.90 m, giving d ≥ 3.36 m (15 rings on the body, 117 mm apart). For jumps (H = 2.06 m) the best height is 1.03 m, giving d ≥ 3.84 m (13 rings, 134 mm apart).
* **Xtion, landscape.** Full height of a standing person from 2.17 m (2.49 m for jumps). The view there is 2.41 m (2.76 m) wide.
* **Xtion, portrait (rotated 90°).** Full height from 1.62 m (1.86 m for jumps), with a view 1.35 m (1.54 m) wide there; at 2.5 m the view is 2.07 m wide by 2.77 m high.
* **Body pixels.** A body seen by the Xtion (frontal area about 0.55 m²) covers about 45,000 depth pixels at 2.0 m and 20,000 at 3.0 m. The Ouster gives about 1,800 points at 2 m, and the Helios a few hundred.

### 2.3 What each sensor contributes (expected, to be confirmed by the bench)

* **Ouster.** Dense 360° geometry with per-column timing, the reference sensor of the rig and of the calibration. Unchanged role.
* **Xtion.** By far the densest surface sampling and the only sensor that resolves hand orientation (a hand spans about 25 by 50 pixels at 2 m). Its limits: narrow field of view, noise growing with distance squared, edge artefacts, no hardware synchronisation, and possibly weak returns from the black suit at about 830 nm. Best placed where the Ouster sees least.
* **Helios.** Coarse rings (about 12 cm apart at its working distance) over the whole height from a third direction. It constrains where the torso and the limbs are on the side the other two miss; it cannot give surface detail. Thin limbs can fall between rings.
* **Combined.** Three viewpoints reduce the unobserved body surface from roughly half to roughly 5 to 15 % per frame (low confidence; the bench measures it). Self-occlusion remains: inner arms against the trunk, inner thighs, armpits, hands in front of the body.

## 3. Hardware setup

### 3.1 Topology

```
Option A (one PC, preferred if the loss test passes)

  Ouster ──── interface box ──┐
                              ├── D-Link switch (gigabit) ── NOVOO hub RJ45 ── laptop
  Helios ──── interface box ──┘
  Xtion  ──── USB 2.0 (active repeater if longer than 5 m) ── laptop USB port (not the hub)

Option B (two PCs, if one PC loses data)

  Ouster, Helios ── switch ── laptop (records both LiDARs)
  Xtion ───────────────────── second PC next to the camera
```

Ethernet runs up to 100 m per cable, so the sensor positions never require separate PCs for the LiDARs. Option A keeps all host timestamps on one clock. Option B needs the event and motion-based clock alignment of section 4. The Ouster's 32 MB/s stream (about 256 Mbit/s) already passed through the hub's Ethernet port with 0.27 % of packets lost, which a 100 Mbit link could not carry, so the path is gigabit.

### 3.2 Power of the Helios

The Helios accepts 9 to 32 V and draws about 11 W. The owners of the equipment already run it from this 19 V, 38 W adapter, so the plug and the polarity are proven in use; no further check is needed.

### 3.3 Network plan

* **Ouster:** sensor 192.168.33.50, destination 192.168.33.30 (laptop adapter), ports 7502 and 7503. Unchanged.
* **Helios:** its current IP and destination must be read first. The RoboSense factory default is believed to be lidar 192.168.1.200 sending to 192.168.1.102 (moderate confidence).
  1. Give the laptop adapter a second address 192.168.1.102/24.
  2. Listen on UDP 7788 (DIFOP) to confirm the sensor IP.
  3. Then either keep both subnets on the adapter, or set the Helios to 192.168.33.60 with destination 192.168.33.30 through its web page.
* **Ports:** none clash (7502, 7503, 6699, 7788).

### 3.4 Mounts and heights

* **Ouster:** as before, lens at about 1.03 m, level.
* **Helios:** on its own tripod (70 cm), raised to an optical centre of about 0.90 to 1.03 m; the bench gives the value.
* **Xtion:** on a tripod or stand with a 90° bracket if portrait is chosen; optical centre at about half the body height.
* **All:** tape the tripod feet to the floor and photograph each mount; a moved sensor invalidates the calibration (section 5).

### 3.5 Placement hypothesis (fixed by Stage 2)

* **Distances.** Ouster at 2.3 m from the zone centre (as before); Xtion at 1.9 to 2.5 m, portrait; Helios at 3.4 to 3.9 m.
* **Angles.** Two families will be compared:
  * (a) three sensors about 120° apart;
  * (b) the Xtion opposite the Ouster (180°), with the Helios at 90°.

  The 120° layout keeps every surface normal within 60° of some sensor. The opposite layout gives the densest sensor the side the Ouster never sees.

  First evidence (`bench-rig`, test body, geometry only; section 16): the 120° triangle keeps about 91 % of the body visible in every bench motion, including the side-on jumps (79 % with the opposite layout); with the triangle, the Xtion in landscape at 2.5 m beats portrait at 2.2 m for walking across (91 % against 85 %). Leading candidate: triangle, Xtion landscape at about 2.5 m (moderate confidence; to be rerun with the s02 avatar).
* **Capture zone.** The zone is the floor region where every sensor sees the whole body. The Xtion's field of view makes it small (about 1.5 to 2.1 m wide at the Xtion's distance), so the walking lines of the protocol will be shorter than on 13 and 14 Oct. The bench outputs the zone polygon and the new floor marks.

### 3.6 Turntable scans on the recording day

The turntable scan uses the Ouster at its turntable position (about 1.6 m from the platform axis, lens near 1.0 m). Two options:

1. **Scan first, then place the motion rig.** Record the turntable scans first, then move the Ouster to the motion position and calibrate. This is the default because it uses the existing fusion code unchanged.
2. **Turntable inside the capture zone.** All three sensors see the rotating subject and nothing is moved. This improves head-top and foot coverage, but needs multi-sensor turntable fusion. It is a later option.

## 4. Time synchronisation

Target: offsets between sensor clocks below 1 ms for the LiDARs and below 5 ms for the Xtion after refinement. A 1 ms error equals 8 mm on a foot moving at 8 m/s.

1. **Host time.** Every recorder process stamps each packet or frame with `time.perf_counter_ns()` on arrival. On Windows this reads the system-wide performance counter, so processes on one PC share one clock.
2. **Clock model per sensor.** Session time = a + b × sensor time, fitted robustly to the lower envelope of (sensor time, arrival time). Arrival is the true time plus a positive delay, so the lower envelope removes most of the network and scheduling jitter (expected about 1 ms, moderate confidence; measured in Stage 0).
3. **Event check.** The jump at the start of every take is seen by all sensors. The landing instant gives an independent offset estimate per sensor, accurate to a few milliseconds after interpolation.
4. **Refinement in tracking.** A constant offset per non-reference sensor (bounded to ±50 ms) is a free parameter of stage 2. The moving body makes it identifiable. This also absorbs the Xtion's unknown internal latency and, in Option B, the offset between the two PCs.

PTP is not used: both LiDARs support it, but a Windows laptop cannot act as a PTP master without extra hardware.

## 5. Extrinsic calibration

All poses are expressed in one world frame: the floor plane is z = 0, with the origin below the Ouster.

1. **Floor (3 degrees of freedom per sensor).** Each sensor fits the floor plane in its empty-room recording, which gives pitch, roll and height.
2. **Target (3 remaining degrees of freedom per sensor).** A rigid object with several flat faces at different orientations: at least 0.6 m per face, matte, not black (for example a large box on a stool, or a tilted table against a box). It is placed at 5 to 8 positions and heights across the zone for a few seconds each. Each sensor segments it against its background and fits its faces. Yaw and horizontal position of every sensor relative to the Ouster follow from a joint point-to-plane optimisation over all placements. Large flat walls of the open space, if visible to all sensors, add constraints for free.
3. **Body refinement.** A two-minute calibration walk through the zone is tracked with the Ouster first. The extrinsics of the other sensors are then refined (bounded to ±2 cm and ±1°) together with their time offsets. The bound prevents avatar errors from leaking into the calibration.
4. **Validation.** Report the target residuals per sensor (aim: 5 mm rms or better for the LiDARs, 10 mm for the Xtion at 2.5 m). Record the target again at the end of the day: a difference above a few millimetres or about 0.1° flags a moved sensor and the affected takes.
5. **Xtion intrinsics and depth bias (optional).** Nominal focal length is about 577 px for 640 px over 58°. Kinect-class cameras have systematic depth distortion. A flat wall recorded by the Xtion and the Ouster at three distances gives a per-pixel depth correction.

The result is one file per session, `rig.json`: sensor type, serial, intrinsics or beam model, extrinsics, clock model, noise model, calibration residuals and date.

## 6. Software architecture

### 6.1 Principles

* **Sensors are adapters.** Everything above them sees only one observation type. The tracker never knows which device produced a point.
* **Layering.** The current layering stays (geometry, io, scene, registration, motion, fusion, meshing, detection, capture, pipelines, body, dynamic, commands; each uses only the layers above it). New code goes into new subpackages or into the existing layer it belongs to.
* **Data.** Pure functions and frozen dataclasses for data; configuration only through the existing config dataclasses (one declaration per parameter, with help text, TOML or JSON, `--set`).
* **Determinism.** Fixed seeds, single-thread mode for regression tests, paired comparisons over several noise realisations.
* **Labelling.** Every derived quantity carries its provenance: observed by which sensors, guessed, or interpolated.

### 6.2 New and changed modules

| Module | Content |
|---|---|
| `bodyscan.sensors` (new, 1.7.0) | sensor models `SpinningLidar` and `DepthCamera` (beam or pinhole geometry, field of view, noise σ(range), pixel timing) with the presets `ouster_os0_128`, `helios_16`, `xtion`; `RangeImage` and its renderer; `clock` (fit_clock); `rig` (Placement, Rig, rig.json); `layouts` (candidate placements). Still to come: the device readers that turn recordings into `RangeImage` frames |
| `bodyscan.sensors.rig` and `bodyscan.calibration` (later) | `Rig` and rig.json exist (1.7.0); calibration (floor, target, body refinement) and the event check are still to come |
| `bodyscan.session` (new) | On-disk layout of a session, manifest, per-sensor raw streams, takes, backgrounds; legacy adapter for single-Ouster takes |
| `bodyscan.capture` (extended) | One recorder process per sensor, a controller process for cues and markers, `check-session` |
| `bodyscan.dynamic` (extended) | Segmentation per sensor and association in the world frame; tracker energy terms generalised to N sensors; time offsets; coupling used only for parts no sensor sees |
| `bodyscan.body` (extended) | `BodyModel` interface (SMPL-X now; MHR and others behind the same interface); personal pose correctives |
| `bodyscan.evaluation` (new, 1.7.0) | pixel classes, per-vertex error map, part tables, kinematics, heatmaps (`evaluate-take`), coverage of the body by a rig (`bench-rig`); still to come: held-out-sensor evaluation and the tracking part of the multi-sensor bench |
| `bodyscan.viewer` (new, 1.7.0) | interactive viewer (section 9), `view` |

### 6.3 Core data types

* **`RangeImage`.** One frame of one sensor:
  * ray origin and direction per pixel in the sensor frame;
  * measured range per pixel;
  * validity per pixel: valid, no return, or unknown (lost packet, invalid depth);
  * time per pixel or per column;
  * σ per pixel.

  LiDAR frames and depth frames both map onto it, which makes the free-space test and the heatmap uniform across sensors.
* **`Observation`.** The world-frame foreground points of one `RangeImage`, each with its sensor id, time, σ and ray, plus the mask of unknown pixels.
* **`Session`.** A rig plus takes; a take is a time interval plus per-sensor stream references.
* **`MotionTrack`.** The tracker output: the current motion.npz contents, plus per-sensor and per-part observed fractions and the time offsets.

### 6.4 Data flow

```
recorders (one process per sensor)
   → raw streams + host timestamps          session/<sensor_id>/raw/
check-session   → loss and timing report    session/reports/
calibrate-rig   → rig.json                  session/rig.json
segment-session → observations per take     session/takes/<take>/observations/
track           → MotionTrack               session/takes/<take>/<version>/motion.npz
evaluate-take   → metrics.json, metrics.npz session/takes/<take>/<version>/
view            → interactive browser viewer
export-motion   → meshes and velocities for Sionna RT
```

### 6.5 Tracker generalisation

* **State.** Avatar pose at a fixed state rate (to be chosen on the bench, for example 30 or 60 Hz). Every point is evaluated at its own time by interpolating between the two neighbouring states (linear for translation, geodesic for rotations). This generalises the current per-point rolling-shutter timing to asynchronous sensors.
* **Data term.** Robust point-to-surface residual over all foreground points of all sensors, whitened by the point's σ. Each sensor is first voxel-subsampled to a common spacing (1 to 1.5 cm, tuned on the bench), so the dense Xtion does not dominate by count through correlated errors.
* **Free space, silhouette, in front.** Per sensor, in its own `RangeImage`, with unknown pixels skipped. This replaces the Ouster-only versions and fixes the lost-column issue by construction.
* **Visibility.** Per part and per frame, the union over sensors. The 1.6.0 left-right coupling and completion run only for parts that no sensor observes. With three sensors this should be the exception.
* **Time offsets.** One per non-reference sensor, refined in stage 2.
* **Initialisation.** Multiple viewpoints remove most of the front and back ambiguity of the first frame; the 1.6.0 facing search stays as a fallback.
* **Unchanged.** Anatomy (hinges, joint speed limits, trunk twist prior), contacts, floor and smoothness carry over.

### 6.6 Backward compatibility

A legacy adapter wraps a capture-motion folder (and its _seg folder) as a one-sensor session: an Ouster rig with an identity clock and the floor frame from segment-motion. With the 1.6.0 options this path must reproduce the 1.6.0 motion files of three short excerpts (for example frames 0 to 79 of walk01_2, jump01 and jump02_2) within floating-point tolerance, single thread. This is a slow regression test. Improvements to the single-sensor case are allowed only behind options, with before and after metrics in the changelog.

### 6.7 Command line

Existing commands keep their names and behaviour. New commands:

* `capture-session`: record any set of sensors with cues.
* `check-session`: loss, timing and clock report.
* `calibrate-rig`: produce `rig.json`.
* `segment-session`: per-take observations.
* `evaluate-take`: metrics and heatmaps.
* `view`: the viewer.
* `bench-rig`: synthetic multi-sensor bench.

`track` accepts either a session take or a legacy _seg folder.

### 6.8 Quality rules

* Type hints on all new public functions; docstrings saying what the function measures or computes and in which frame and unit.
* Tests:
  * unit tests per module;
  * synthetic end-to-end tests with truth;
  * the regression test of 6.6;
  * fast tests (`python -m pytest -q tests`) on every commit;
  * slow tests (`BODYSCAN_SLOW_TESTS=1`) before every release.
* Static checks (ruff, and mypy on the new packages) and fast tests in GitHub Actions on push to main. The repository is public, so this is free.
* Versioning: semantic. The multi-sensor tracker is 2.0.0, because `track` gains a new input type; old command lines stay valid.
* Every version updates CHANGELOG.md (measured results, what was tried and not kept, what is not validated), docs, `docs/parameters.md` and the version string.
* An ADR in `docs/adr/` for each decision of this plan that changes an interface (rig format, observation type, body model interface, viewer technology).

## 7. Avatar

### 7.1 Diagnosis of the current avatar

The current avatar is SMPL-X fitted to one turntable scan in A-pose, plus per-vertex displacements on a twice-subdivided mesh. Its limitations:

1. **One pose only.** The displacements are learned in one pose, so pose-dependent deformation (shoulders, hips, elbows, knees) comes only from SMPL-X's generic correctives.
2. **Joint centres.** They are regressed from the surface by SMPL-X, not measured on the person.
3. **Hands looked fixed.** The SMPL-X wrist has three rotation axes. The hands looked fixed because the Ouster sees a hand as a blob with no usable orientation, so the priors kept the wrist near neutral. This is an observability problem, which the Xtion addresses, not a model problem.

More joints help only where the data constrain them: with sparse data, extra degrees of freedom add ambiguity, not accuracy (high confidence). The choice of model must therefore come from measurements on our data (7.4).

### 7.2 Candidates

| Candidate | What it is | Strengths for this project | Weaknesses | Licence |
|---|---|---|---|---|
| A. SMPL-X plus displacements (current) | SMPL-X with personal per-vertex offsets | known, motion priors, AMASS compatible, 55 joints with hands | one-pose personalisation | research, registration per user |
| B. SMPL-X plus personal pose correctives | A, plus displacement fields learned from the six scan poses, sparse per joint region (shoulder, elbow, hip, knee, spine), ridge-regularised | uses the six-pose scans directly; keeps all SMPL-X tooling | six poses limit what can be learned; must be validated pose by pose | as SMPL-X |
| C. MHR (Meta, 2025) | Decoupled skeleton and surface (ATLAS paradigm), non-linear pose correctives, several levels of detail, finger articulation; PyTorch integration | lower fitting error than SMPL and SMPL-X on unseen identities, per its paper; better shoulders, elbows and knees; skeleton fitted independently of the surface | young ecosystem, few motion priors; our tracker priors must be re-tuned | Apache 2.0 |
| D. SKEL (MPI, 2023) | SMPL mesh re-rigged with a biomechanical skeleton: fewer, realistic degrees of freedom (knee hinge, forearm supination, scapula sliding) | joint locations closer to anatomy; its constrained kinematics match our need for natural motion | SMPL resolution (6,890 vertices), no finger articulation, research code | research |
| E. SUPR (MPI, 2022; from prior knowledge, not re-verified in this session) | Part-based model with articulated feet and toes | foot contact deformation | research code; less tooling than SMPL-X | research |

Recommended first comparison: A against B against C. D is a second step if joint placement turns out to be the dominant error, and E only if foot contact matters for the radar.

### 7.3 Multi-pose scan protocol (turntable, same suit as the motion day, same day)

Six poses, two to three laps each (about 90 s per lap): A-pose, T-pose, arms raised forward to about 90°, a half squat, a stride (one foot forward), elbows bent at 90°. Holding the arms up for three laps is tiring and blurs the fusion, so the arm poses get two laps. Each pose is fused with the existing turntable pipeline.

### 7.4 Selection protocol

1. **Shape at an unseen pose.** For each candidate and each pose p: fit the shape on the other five poses, then fit only the pose to scan p. Report the scan-to-model distance (median, p95, per body part) at the unseen pose. This measures shape generalisation to poses not used for fitting.
2. **Joint plausibility.** Report the bone lengths in each pose (they must not change), joint centres against the scan's limb axes, and the hinge axes.
3. **Tracking on the synthetic bench.** Use the winning avatar as truth, with the same rig and motions as Stage 2. Report joint error, surface error and body-part velocity error, paired over three noise realisations.
4. **Real data.** Report held-out-sensor residuals (section 8.3) on the recorded takes.

The candidate with the lowest unseen-pose and held-out errors wins. Ties go to the simplest model.

## 8. Metrics and heatmap

### 8.1 Principles

Without ground truth, real-data metrics measure consistency with measurements, not accuracy. Three kinds of evidence are kept separate in every report:

* **In-fit residuals:** data the fit used.
* **Held-out residuals:** a sensor the fit did not use. These are the closest substitute for accuracy.
* **Synthetic errors:** against true motion on the bench.

A body region nobody observed has no error value. It is shown as unobserved, never as accurate.

### 8.2 Definitions (per take, per tracker version)

For sensor s, frame k and pixel i with a valid measured range r and ray u, let r̂ be the range at which u hits the posed avatar at the pixel's time (ray cast), and e = r − r̂.

| Pixel class | Condition | Meaning |
|---|---|---|
| Fitted | measurement on the body and model hit, \|e\| ≤ τ_s | normal residual; \|e\| is the range residual |
| Free-space violation | model hit and e > τ_s | the avatar stands where the sensor saw through |
| Unexplained | foreground point and no model hit within τ_s | a body part missing or misplaced in the model |
| Unobserved | model hit and pixel unknown or occluded | no information |

τ_s = 3σ_s(r), with σ_s from the noise model measured in Stage 0.

* **Per vertex.** Every model-hit pixel is attributed to the vertices of the hit triangle by barycentric weights. Over the frames where it is observed, a vertex accumulates: median and p95 of \|e\|, violation rate, unexplained rate nearby, and coverage (share of frames observed). The heatmap colours the median or p95 of \|e\| on vertices with coverage of at least 5 %. Others are drawn grey and hatched. A second heatmap shows coverage itself.
* **Per body part.** The same statistics over the vertices of each part (24 parts of the SMPL-X segmentation, or the chosen model's own): per take, per sensor and over all sensors.
* **Per point.** Point-to-surface distance of every foreground point: inlier rate (≤ 3σ_s), outlier rate (> 3σ_s and > 5 cm), clutter rate (> 25 cm, the 1.6.0 definition).
* **Temporal.**
  * Joint and part velocity and acceleration.
  * High-frequency energy: position power above 8 Hz per joint, a jitter proxy.
  * Part velocity spectra, the quantity that becomes micro-Doppler.
* **Plausibility (from the 1.5.0 and 1.6.0 reviews).** Turns above 720°/s; elbows and knees out of plane; arm depth inside the trunk; chest twist and its rate; joints over the speed limits; floor penetration; foot sliding during contact.
* **Calibration and synchronisation.** Target residuals per sensor, per-sensor time offsets and their change over the day.

### 8.3 Held-out-sensor validation

For each take, the tracker runs once with all sensors and once per sensor left out (four runs with three sensors). The residuals of the left-out sensor, by part and over time, are the main real-data error estimate. The gap between in-fit and held-out residuals measures overfitting. Its limitation must be stated in every report: a body part seen only by the left-out sensor is unconstrained in that run, so held-out residuals overestimate the error of the full fit there.

### 8.4 Synthetic multi-sensor bench

`simulate-motion` gains a rig description with three sensor models:

* **Ouster:** the existing model.
* **Helios:** 16 elevations at 2° from −15°, azimuth steps 0.2° or 0.4°, per-packet timing, 1 cm noise.
* **Xtion:** pinhole 640 × 480; σ(z) = k z²; quantisation; dropout at depth edges and on dark surfaces (rate from Stage 0); frame time offset.

Motions are the existing synthetic set (walk, jog across and towards, jumps facing and side-on, the jump02-like take), plus AMASS sequences if licensed. Errors are reported per part against the truth, paired over three noise realisations.

### 8.5 Stability (optional, cluster)

Ten fits of one take, each with a different random 20 % of points removed. The spread per joint and per vertex gives an uncertainty map, which costs only cluster time.

### 8.6 Outputs

Next to every motion.npz:

* `metrics.json`: scalars and per-part tables.
* `metrics.npz`: per-frame and per-vertex arrays.
* A `report.md` summary with the figures that the viewer also shows.

## 9. Viewer

### 9.1 Requirements

1. Choose session, take and tracker version from lists.
2. Timeline: play, pause, speed, frame slider, jump to flagged frames.
3. Toggles:
   * points per sensor, coloured by sensor or by residual;
   * avatar, solid or wireframe;
   * skeleton;
   * guessed limbs highlighted;
   * sensor positions and fields of view.
4. Heatmap with a metric selector: median, p95, violation rate, coverage, uncertainty.
5. Plots synchronised with the timeline: residuals per part, part velocities, flags, time offsets.
6. Click a body part to show its time series.
7. Side by side comparison of two versions or two sensor subsets (for example Ouster only against all three).

### 9.2 Technology (decision to confirm by a one-day spike)

Viser (Python, free and open source). It draws point clouds and meshes in a web browser and has Python-configurable buttons, checkboxes, sliders and dropdowns. It runs on the laptop (`python -m bodyscan view`), or on a cluster node through an SSH tunnel. Per-vertex colour support for the heatmap must be confirmed in the spike. If it is missing, the heatmap is drawn as a coloured point layer on the vertices.

Fallback: Open3D's GUI, already installed on the laptop, which has checkboxes, combo boxes, sliders and per-vertex colours, but runs only locally.

Per take, a preprocessing step writes `view_cache.npz` (decimated points per frame, mesh vertices per frame in float16, per-vertex metrics), so playback stays fluid.

## 10. Compute and storage

* **Raw sessions on the NAS.** The Ouster writes about 115 GB/h and the Xtion up to 66 GB/h. The cluster home is limited to about 26 GB and must never hold raw data. The NAS mount paths on the laptop and on the cluster are to be discovered by Claude Code in Stage 0.
* **Cluster.**
  * Never compute on login.dei.unipd.it.
  * Jobs are cancelled below 10 % GPU efficiency or below 60 % CPU efficiency.
  * The multi-sensor tracker should run on GPU in float32 with batched evaluation (all points of a stage 2 window at once, several held-out runs per job), so that GPU use stays well above the threshold. Otherwise it uses CPU nodes as for 1.4.1.
  * One array task per take and configuration; results go to versioned folders (results_vXXX).
* **IMDEA cluster.** It can share the load if access exists (scheduler and GPUs unknown). The job scripts are written against a small scheduler abstraction, so the same take list can be submitted to either cluster.

## 11. Stages

| Stage | Content | Depends on | Done when |
|---|---|---|---|
| S0 Hardware bring-up | Helios power check, network setup and first point cloud; Xtion driver (OpenNI2 on Windows 11), USB ids, depth stream and timestamps; per-sensor noise model from a static scene; interference test (each sensor alone against all together); suit visibility for the Xtion at 1.6, 2.0, 2.5, 3.0 m; NAS paths; SSH key to the cluster | none | each sensor streams to disk on the laptop; noise and interference numbers recorded in docs/hardware.md |
| S1 Recorder | `capture-session` (one process per sensor, large socket buffers, raw packets or frames with host timestamps, lossless depth compression in a writer thread), cues as in capture-motion with a longer final hold, `check-session` | S0 | a 2-minute three-sensor test shows below 1 % lost frames per sensor on the chosen topology (A or B) |
| S2 Bench and placement | Helios and Xtion simulators; `bench-rig` over 8 to 12 candidate rigs (angles, distances, heights, Xtion orientation, sensor modes) and the synthetic motions, three noise realisations | none (parallel to S0 and S1) | a recommended rig with per-part coverage and error tables, the capture zone polygon and the floor-mark plan |
| S3 Calibration and synchronisation | `calibrate-rig` (floor, target, body refinement), clock models, event check | S1 | on a test day: target residuals within the aims of section 5; clock offsets stable over 30 minutes |
| S4 Multi-sensor tracker (2.0.0) | `RangeImage` and `Observation` types, per-sensor segmentation, generalised energy terms, time offsets, legacy adapter and regression test | S2, S3 (synthetic first) | bench errors with three sensors below the single-Ouster errors on every motion; regression test passes |
| S5 Avatar | multi-pose scans, candidates A, B, C fitted, selection protocol of 7.4 | S4 for step 3 of 7.4 | a chosen model with unseen-pose and bench numbers |
| S6 Metrics and heatmap | `evaluate-take`, section 8 | none (starts on the 1.6.0 data) | metrics and heatmaps for the 27 s02 takes of 1.6.0 |
| S7 Viewer | `view`, section 9 | S6 | all requirements of 9.1 work on the 27 s02 takes |
| S8 Cluster and NAS pipeline | scheduler abstraction, GPU batching, held-out runs, versioned results | S4 | a full session processed with held-out runs |
| S9 Recording day | section 12 | S1, S2, S3, S5 scans | all takes recorded, calibration repeated at the end |
| S10 Processing and evaluation | tracking, held-out validation, review in the viewer, comparison against single-Ouster tracks of the same takes | S8, S9 | per-take reports; accepted takes listed |
| S11 Sionna RT export | `export-motion` of accepted takes | S10 | animated meshes loaded in Sionna RT |

S6 and S7 can start immediately, in parallel with S0 to S2.

## 12. Recording day protocol (draft)

1. **Morning: scans.** Turntable scans of the subject: six poses, two to three laps each (about 35 minutes).
2. **Place the rig.** Place and level all sensors per the S2 plan, tape the tripod feet, photograph the mounts, measure heights and distances with a tape (initial guess for the calibration).
3. **Checks.** `check-session` live test, 2 minutes with all sensors.
4. **Empty room.** 30 s for every sensor's background.
5. **Calibration (start).** Target at 5 to 8 positions, then the 2-minute calibration walk.
6. **Takes.** The protocol of 13 and 14 Oct on the new floor marks, three takes each. A start jump in every take and a final hold of at least 5 s. Nothing else runs on the recording PCs.
7. **Calibration (end).** Target again, then 30 s of empty room.
8. **Copy.** Raw data to the NAS before leaving; keep the laptop copy until the NAS copy is verified (checksums).

## 13. Risks

| Risk | Likelihood | Effect | Mitigation |
|---|---|---|---|
| Xtion driver does not work on Windows 11 | moderate | no depth camera data | test in S0; second PC with an older Windows or Linux; OpenNI2 builds from other vendors |
| Packet or frame loss with three streams on one laptop | moderate to high | gaps, as on 14 Oct | one process per sensor, large socket buffers, Option B |
| Black suit returns weakly at about 830 nm | unknown | holes in Xtion depth on the body | measure in S0; another suit colour for the turntable and motion days if needed (the suit matters for both) |
| Infrared interference between sensors | low to moderate | extra noise or dropouts | measured in S0; change distances or angles |
| A sensor is moved during the day | moderate | wrong calibration for later takes | taped feet, end-of-day calibration, flagged takes |
| Helios adds too little to justify its placement constraints | moderate | wasted space | the bench quantifies its contribution; if small, place it for coverage of the feet and torso only |
| Avatar candidates overfit six poses | moderate | worse shape at unseen poses | leave-one-pose-out selection |
| Body-based calibration absorbs avatar errors | low to moderate | biased extrinsics | bounded refinement, comparison with the target-only calibration |

## 14. Open items

* Exact Xtion model (PRO or PRO LIVE): read from the USB product id in S0; only depth is needed either way.
* Which second PC, if Option B is needed.
* NAS mount paths on the laptop and the cluster.
* Access, scheduler and GPUs of an IMDEA cluster, if any.
* A stand for the Xtion (and a 90° bracket if portrait is chosen).

## 15. Sources

* RoboSense Helios series specifications (vertical field of view, resolution, voltage, power, range precision): https://store.robosense.ai/products/helios-series
* RS-Helios-16P listing quoting the user manual (ports 6699 and 7788, time synchronisation, voltage): https://openelab.com/products/robosense-rs-helios-16p-lidar
* ASUS Xtion PRO LIVE specifications (structured light, 0.8 to 3.5 m, 640 × 480, 30 fps, 58° × 45°, USB 2.0): https://rosindustrial.org/news/2016/1/13/3d-camera-survey
* PrimeSense depth resolution at 2 m: https://electronicdesign.com/embedded/how-microsoft-s-primesense-based-kinect-really-works
* Depth noise of PrimeSense sensors growing with distance squared: K. Khoshelham and S. Oude Elberink, Accuracy and resolution of Kinect depth data for indoor mapping applications, Sensors 12(2), 2012 (cited from prior knowledge)
* MHR: https://arxiv.org/abs/2511.15586 and https://github.com/facebookresearch/MHR
* SKEL: https://skel.is.tue.mpg.de
* Viser: https://pypi.org/project/viser
* Ouster time synchronisation: https://docs.ouster.com/sensor-docs/time-synchronization

## 16. Status on 21 Oct 2026 (bodyscan 1.7.0)

| Stage | State |
|---|---|
| S2 bench and placement | coverage part done (`bench-rig`); tracking part waits for S4 |
| S6 metrics and heatmap | done for the single-Ouster takes (`evaluate-take`); held-out-sensor residuals wait for S4 |
| S7 viewer | first version (`view`), tested headless, not yet looked at in a browser |
| S0, S1, S3, S4, S5, S8 to S11 | not started |

Coverage of the body (mean share visible to some sensor; test body, zone centre 2.3 m from the Ouster, Helios at 3.6 m and 0.95 m, Xtion at 0.95 m aimed at 0.9 m; details in docs/evaluation.md section 4):

| layout | walk across | walk radial | jump facing | jump side-on |
|---|---|---|---|---|
| Ouster alone | 38 % | 46 % | 46 % | 31 % |
| Ouster + Xtion opposite (portrait, 2.2 m) | 57 % | 87 % | 89 % | 61 % |
| Ouster + Helios opposite | 70 % | 89 % | 90 % | 62 % |
| triangle 120° (Xtion portrait, 2.2 m) | 85 % | 90 % | 94 % | 91 % |
| Xtion opposite, Helios at the side (portrait, 2.2 m) | 81 % | 91 % | 92 % | 79 % |
| triangle 120° (Xtion landscape, 2.5 m) | 91 % | 91 % | 94 % | 91 % |
| Xtion opposite, Helios at the side (landscape, 2.5 m) | 90 % | 91 % | 92 % | 79 % |

Next: run `evaluate-take` on the 27 s02 takes of 1.6.0 and look at them with `view`; rerun `bench-rig` with the s02 avatar; then S0 (hardware bring-up) and S1 (recorders).
