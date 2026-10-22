"""Sensors of a multi-sensor rig: models, range images, clocks, placements and candidate layouts.

    models    SpinningLidar (Ouster OS0-128, RoboSense Helios 16) and DepthCamera (ASUS Xtion):
              ray directions, pixel times, field of view and range noise
    image     RangeImage, the frame type of every sensor, and render() for simulation by ray casting
    clock     sensor clock to session clock (lower envelope of the arrival times)
    rig       Placement and Rig (rig.json): where each sensor stands, its clock, its calibration
    layouts   candidate placements of the three sensors around a capture zone, for planning

This layer uses geometry and io only (numpy and Open3D; no PyTorch).
"""

from bodyscan.sensors.clock import ClockModel, fit_clock
from bodyscan.sensors.image import NO_RETURN, UNKNOWN, VALID, RangeImage, cast, mesh_scene, render
from bodyscan.sensors.models import (DepthCamera, RangeNoise, SensorModel, SpinningLidar, helios_16, ouster_os0_128,
                                     preset, sensor_from_dict, xtion)
from bodyscan.sensors.rig import Placement, Rig, looking_at, pose, rotation_matrix

__all__ = ["ClockModel", "fit_clock", "NO_RETURN", "UNKNOWN", "VALID", "RangeImage", "cast", "mesh_scene", "render",
           "DepthCamera", "RangeNoise", "SensorModel", "SpinningLidar", "helios_16", "ouster_os0_128", "preset",
           "sensor_from_dict", "xtion", "Placement", "Rig", "looking_at", "pose", "rotation_matrix"]
