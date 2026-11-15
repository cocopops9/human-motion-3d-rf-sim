# human-motion-3d-rf-sim
3D reconstruction of human motion for RF sensing simulations. Multi-sensor acquisition (LiDAR, camera, MoCap) and accurate 3D human models, used to simulate realistic people in Sionna RT and reproduce micro-Doppler signatures, with radio data as ground truth.

## Contents

| Folder | What |
|---|---|
| [lidar-body-scan](lidar-body-scan) | the `bodyscan` package (version 1.8.0): closed, smoothed meshes of a person standing on a turntable scanned by an Ouster OS0-128; animated meshes with per-vertex velocities of a person walking, running or jumping, tracked with a personal SMPL-X avatar ([docs/motion.md](lidar-body-scan/docs/motion.md)); since version 1.8 with several sensors at once (the Ouster, a RoboSense Helios 16 and an ASUS Xtion depth camera), with the hands, and with the final motion chosen among several runs ([docs/multisensor.md](lidar-body-scan/docs/multisensor.md)) |
