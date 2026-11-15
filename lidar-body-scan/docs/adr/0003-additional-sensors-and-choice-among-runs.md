# ADR 0003: additional sensors as a folder per take, counted per Ouster point; the final motion chosen among runs

**Status:** Accepted in 1.8.0
**Date:** 2026-11-15
**Deciders:** Giuseppe D'Auria

## Context

The session of 3 Nov 2026 recorded each take with the Ouster OS0-128, a RoboSense Helios 16P and an ASUS
Xtion PRO Live. The tracker of 1.7 knew the Ouster only. ADR 0001 planned a `RangeImage` for every sensor
and a rig file; the recordings, the calibration on the person and the time offsets were made by session
tools outside the package, and the tracker had to use their output without breaking the single-Ouster
takes (with the Ouster alone a run must give the motion of 1.7 bit for bit).

Two observations on the first three-sensor production shaped the decisions below:

* with the data term a mean over the points of all sensors, the Xtion (up to 2800 points per frame at
  1.5 m against about 1300 of the Ouster) diluted the Ouster and hid misplaced limbs from the searches;
* two runs of the same configuration that differ only in rounding put an arm more than 10 cm apart in 11
  to 25 of 37 frames of a jump: where a limb moves fast, a single run is a sample of the local minima.

## Decision

1. **Input.** A take keeps its Ouster segmentation folder (floor frame, timing, frame numbering) and gains
   a folder of additional sensors: `views.json` and, per sensor, `scene.npz` (pose in the floor frame,
   image geometry, empty scene) and `frames/person_XXXXX.npz` matched to the Ouster frames
   (`track --extra-sensors`). The image geometry is `RangeImageGeometry` for spinning LiDARs and
   `PinholeGeometry` for depth cameras; every term is computed per sensor on its own image.
2. **Weighting.** Every point is divided by its sensor's noise relative to the Ouster's (constant, or
   growing with the range squared), and the sum over all points is divided by the Ouster's point count
   (`reference_count`), so that another sensor adds evidence. Frames of a sensor with the person outside
   its useful range are dropped (`extra_sensor_range`).
3. **The Ouster is the reference.** The floor frame, the timing and the calibration of the other sensors
   on the person are relative to it; held-out validation leaves the other sensors out
   (`--exclude-sensors`).
4. **The final motion is chosen among runs.** Several runs of a take (other sensors, other rounding,
   backwards in time) are scored frame by frame with the tracker's own score, and the path of least score
   plus a switching cost is the warm start of a last pass (`select-motion`, then `track --init-motion`).

## Options considered

### Weighting: per Ouster point (chosen), mean over all points, per sensor equal shares

The mean over all points made the densest sensor dominate (Ouster points 5 to 25 cm off the body in a jump
window: 2206 and 1794 with the mean, 806 to 865 per Ouster point). Equal shares per sensor would give a
sensor with a few points in a frame the weight of the Ouster. Counting per Ouster point keeps the
single-Ouster objective unchanged and lets every additional point add evidence in proportion to its noise.

### Final motion: one run (rejected), a best configuration chosen once per take, a choice per frame (chosen)

One run is a sample. A configuration chosen per take still fails where another succeeds: in the comparison
of 3 Nov three of the eight single runs lost a whole take (Ouster held-out distance 11.0 to 11.9 mm against
8.1 mm or less in every other case), which ones is partly chance (a run that differs from one of them only in
rounding did not), and the choice per frame never lost one (worst take 7.33 mm).

### Several hypotheses inside the tracker (beam search, kept as an option)

A beam of three tracks per frame improved a jump window less than one backward run chosen with a forward
one, at 3.5 times the time of a forward run (docs/multisensor.md, section 5).

## Consequences

* motion.npz and motion.json gain per-sensor frames, points and residuals (fitted and held out); body21
  motions of a single Ouster keep their keys and values.
* A take costs several tracking runs plus a selection and a last pass; on a cluster the runs are
  independent.
* The calibration and the segmentation of the additional sensors live outside the package for now; the
  folder format of decision 1 is the interface.
* Not decided: a motion prior for limbs that no sensor sees well (the choice among runs cannot resolve
  them, docs/multisensor.md section 7).
