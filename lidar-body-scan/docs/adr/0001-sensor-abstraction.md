# ADR 0001: one frame type for every sensor, and a rig file

**Status:** Accepted for 1.7.0 (models, frame type, clocks, rig file); the tracker follows in 2.0.0
**Date:** 2026-10-21
**Deciders:** Giuseppe D'Auria

## Context

The tracker of 1.6.0 reads one Ouster through its range image (RangeImageGeometry, segment-motion crops). The lab now has a RoboSense Helios 16 (16 beams over 30 deg) and an ASUS Xtion (a structured-light depth camera), placed around the capture zone, recorded by one or two Windows PCs without hardware synchronisation. The code must work with any subset of the sensors, including the old single-Ouster takes, and new sensor types (for example a true time-of-flight camera) must not require changes to the tracker.

## Decision

1. Every sensor frame becomes a `RangeImage`: (H, W) ray directions and origins in the sensor frame, measured range, a per-pixel state (valid, no return, unknown), per-pixel times and per-pixel noise, plus world_from_sensor. A spinning LiDAR and a depth camera differ only in how they produce it (`SpinningLidar`, `DepthCamera` in bodyscan.sensors.models).
2. Downstream code (segmentation, tracking, evaluation) consumes rays and pixel states, never device types. The free-space test and the heatmap treat "unknown" pixels as missing, never as seen through; this also removes the lost-column defect of 1.3.0 by construction.
3. A session is described by `rig.json` (schema bodyscan.rig/1): per sensor its model, serial, world_from_sensor and clock model (session = offset + rate x sensor time), plus calibration residuals. World frame: the floor frame, origin below the reference sensor (the Ouster), as in 1.6.0, so a one-sensor rig of an old take keeps its coordinates.
4. Clocks are fitted to the lower envelope of (sensor time, arrival time) on the host clock; constant residual offsets between sensors are estimated from the motion (start jump, tracker time offset).

## Options considered

### A. One frame type for all sensors (chosen)

| Dimension | Assessment |
|---|---|
| Complexity | medium: one adapter per device |
| Extensibility | high: a new device needs a model and a reader only |
| Fit to the tracker | high: the 1.6.0 terms (data, silhouette, free space) are already ray based |

Pros: uniform free-space and visibility tests; the evaluation and the viewer work for every sensor. Cons: a depth camera's per-pixel rays use more memory than its intrinsics (mitigated by cropping around the person).

### B. Point clouds only, sensor type ignored downstream

Pros: simplest. Cons: loses the free-space information (where the sensor saw through), which the 1.6.0 tracker needs against phantom limbs, and the per-pixel timing of spinning LiDARs.

### C. Device-specific code paths in the tracker

Pros: each sensor optimally modelled. Cons: every new device touches the tracker; combinations multiply the test matrix.

## Consequences

* Easier: adding a sensor; evaluating any combination; held-out-sensor validation (drop one sensor's frames).
* Harder: the tracker must evaluate residuals per sensor with its own noise and times (2.0.0 work).
* To revisit: memory per frame for the Xtion if whole images are kept; the global-shutter assumption for the Xtion (to be measured).

## Action items

1. [x] Sensor models, RangeImage, renderer, clocks, rig file, layouts (1.7.0).
2. [x] Evaluation and viewer on the legacy single-Ouster takes (1.7.0).
3. [ ] Legacy adapter from a segment-motion folder to a one-sensor session, with the regression test against 1.6.0 (2.0.0).
4. [ ] Recorders per sensor writing raw streams with host times; check-session (S1 of the plan).
5. [ ] Tracker on RangeImages of N sensors (2.0.0).
