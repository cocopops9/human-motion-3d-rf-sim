"""Platform angle versus time."""

from bodyscan.motion.models import FreeMotion, MeanMotion, Motion, MotionModel, StepperMotion, profile_angle
from bodyscan.motion.fitting import (constant_speed_fit, fit_correction, fit_free, fit_motion, lap_times,
                                     measurement_classes, plausible_sequence, robust_cost)
from bodyscan.motion.pairs import Samples, boundary_pairs, chained_angles, joint_axis_fit, revisit_pairs
from bodyscan.motion.estimator import MotionConfig, PlatformMotionEstimator
from bodyscan.motion.timing import consistent_frame_times

__all__ = ["FreeMotion", "MeanMotion", "Motion", "MotionModel", "StepperMotion", "profile_angle",
           "constant_speed_fit", "fit_correction", "fit_free", "fit_motion", "lap_times", "measurement_classes",
           "plausible_sequence", "robust_cost", "Samples", "boundary_pairs", "chained_angles", "joint_axis_fit",
           "revisit_pairs", "MotionConfig", "PlatformMotionEstimator", "consistent_frame_times"]
