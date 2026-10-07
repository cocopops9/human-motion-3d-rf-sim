# human-motion-3d-rf-sim
3D reconstruction of human motion for RF sensing simulations. Multi-sensor acquisition (LiDAR, camera, MoCap) and accurate 3D human models, used to simulate realistic people in Sionna RT and reproduce micro-Doppler signatures, with radio data as ground truth.

## Contents

| Folder | What |
|---|---|
| [lidar-body-scan](lidar-body-scan) | the `bodyscan` package for the Ouster OS0-128: closed, smoothed meshes of a person standing on a turntable, and (version 1.2) animated meshes with per-vertex velocities of a person walking or jumping around the sensor ([docs/motion.md](lidar-body-scan/docs/motion.md)) |
