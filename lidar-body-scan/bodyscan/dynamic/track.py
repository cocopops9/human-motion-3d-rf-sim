"""Tracking: the avatar fitted to every frame of a segmented recording.

The motion comes from the LiDAR: every frame the avatar is posed so that its
surface passes through the person's points. What the sensor cannot see (the
far side of the body, an arm behind the torso) is filled by constraints that
keep the motion human, never by a learned or recorded motion:

    data          every point of the person close to the visible surface of the
                  body (robust point-to-plane distance, a little point-to-point;
                  only faces the sensor can see take correspondences)
    silhouette    no visible part of the body where the sensor saw through to
                  the background: a vertex that projects onto a pixel whose
                  measured range lies beyond it is pulled towards the person's
                  silhouette (distance map of the silhouette crop, in angle,
                  times the range); a vertex in front of the measured surface
                  inside the silhouette is pushed back to it
    limits        anatomical joint limits (bodyscan.body.skeleton.LIMITS_DEG)
    floor         no vertex of the feet below the floor
    continuity    frame by frame: close to the motion predicted from the
                  previous frames (constant velocity)

Each frame is fitted coarse to fine (the robust scale shrinks from
coarse_scale x robust_scale to robust_scale over the rounds), so that a limb
that moved fast is still caught. When points of the person remain far from
the body after the fit (an arm swung overhead in a jump, knees folding at a
landing), the frame is fitted again from other starts (the previous pose,
the legs and each arm from a few poses) and the start that explains the
points best is kept. Body parts the sensor hardly sees get a stronger
continuity (stage 1) and smoothness (stage 2), so that they keep their
motion instead of jumping where a few points pull them.

Stage 2 refines the whole sequence in overlapping windows: the same data,
silhouette, limit and floor terms for every frame, plus

    smoothness    the accelerations of the joints change smoothly (a penalty
                  on the jerk, Charbonnier: small jitter is removed, real sharp
                  events such as a heel strike or a landing stay; a body in
                  free fall or turning steadily costs nothing, which a penalty
                  on the acceleration itself would flatten)
    contact       a foot found standing on the floor (low and slow in stage 1)
                  does not slide
    rolling shutter  each point, and each pixel of the silhouette terms, is
                  compared with the body at the time it was measured (the
                  sensor needs one turn per frame: a person spans a few
                  milliseconds of it, or a whole turn when standing where the
                  frame starts)

Output motion.npz: the pose of every frame (times on the sensor clock,
root rotation and position, body joint rotations: a PoseSequence), the
joints, the foot contacts, per frame and per body part the fraction of the
part that the sensor actually observed, and quality numbers that flag frames
for review.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bodyscan.body import skeleton
from bodyscan.config import param, section
from bodyscan.dynamic import coupling
from bodyscan.dynamic.coupling import CouplingConfig
from bodyscan.log import Progress, info, warning


@dataclass
class TrackConfig:
    """Fitting the avatar to every frame (stage 1, frame by frame)."""
    level: int = param(0, "subdivision level of the avatar used for tracking (0: model resolution, the "
                          "fastest; the export can use a finer level)")
    device: str = param("auto", "PyTorch device: auto (CUDA when available), cpu, cuda")
    precision: str = param("auto", "floating point precision: auto (float32 on a GPU, float64 on a CPU), "
                                   "float32, float64", choices=("auto", "float32", "float64"),
                           effect="float32 on a CPU: about 30 % faster refinement, same mean joint error on a "
                                  "synthetic walk")
    robust_scale: float = param(0.03, "distance at which a point stops counting fully (Geman-McClure)", unit="m",
                                effect="larger: smoother but more biased by stray points; smaller: needs a "
                                       "good start")
    point_to_point: float = param(0.1, "weight of the point-to-point distance next to the point-to-plane one")
    max_correspondence: float = param(0.25, "points farther than this from the visible body are ignored", unit="m")
    visibility_tolerance: float = param(0.03, "a vertex behind another by more than this is hidden", unit="m")
    silhouette_weight: float = param(2.0, "weight of the silhouette and free-space term",
                                     effect="larger keeps hidden limbs out of the free space more firmly")
    silhouette_scale: float = param(0.06, "robust scale of the silhouette distances", unit="m")
    free_space_loss: str = param("wide", "how the free-space terms grow with the distance of a vertex from "
                                         "where the sensor allows it: wide (a robust term at the scale plus one "
                                         "at 4 times the scale: a limb 20 to 50 cm out in empty space is still "
                                         "pulled back) or bounded (the robust term alone: the pull fades beyond "
                                         "about two scales; version 1.3). Both are bounded, so that leaving the "
                                         "points altogether never pays", choices=("wide", "bounded"),
                                 effect="bounded lets a limb the sensor sees as empty space stay there "
                                        "(arms stretched into the air, a twisted trunk)")
    free_space_margin: float = param(0.06, "a vertex counts as in free space when the sensor saw this much "
                                           "beyond it", unit="m")
    limit_weight: float = param(5.0, "weight of the anatomical joint limits")
    twist_weight: float = param(0.05, "weight pulling the twist of the spine and the rotation of the hips about "
                                      "the legs towards zero [per rad^2]: the pelvis can turn and the trunk and "
                                      "legs turn back by as much, which the points hardly tell apart")
    trunk_twist_weight: float = param(1.0, "weight pulling the twist of the chest against the pelvis (the sum of the "
                                           "three spine joints' twists) towards zero [per rad^2]: seen side-on, the "
                                           "points hardly tell it, and an arm swinging in front of the chest pulls it "
                                           "round",
                                      effect="0 (version 1.5): chests of the s02 side-on jumps twisted back and forth "
                                             "by about 25 deg (p95) at up to 280 deg/s, and the guessed far arm, "
                                             "hanging from the chest, swung with them")
    head_weight: float = param(0.15, "weight pulling the neck and head rotations towards upright and facing "
                                     "forward [per rad^2]: the head is round, its points hardly tell how it is "
                                     "turned or tilted",
                               effect="0: the head turns and tilts freely (the weird head angles of version 1.3); "
                                      "0.05 (version 1.4): heads of people seen side-on still turned towards the "
                                      "sensor by 30 to 40 deg at times")
    forearm_twist_weight: float = param(0.02, "weight pulling the twist of the forearms (about their long axis) "
                                              "and the wrist rotations towards rest [per rad^2]: the points of a "
                                              "forearm and of a relaxed hand hardly show them")
    floor_weight: float = param(5.0, "weight of the no-foot-below-the-floor term")
    pose_continuity: float = param(0.05, "weight of the joint rotations staying near the prediction "
                                         "(constant velocity) [per rad^2]",
                                   effect="larger: steadier hidden limbs, slower to follow fast motion")
    root_continuity: float = param(0.02, "weight of the root staying near the prediction")
    hidden_continuity: float = param(10.0, "the continuity weight of a joint whose body part is less than 30 % "
                                           "visible is multiplied by up to 1 + this (fully hidden): a hidden limb "
                                           "keeps its motion instead of jumping where a few points pull it")
    relax_hidden_arms: float = param(0.1, "weight pulling the shoulder and elbow of an arm the sensor does not see "
                                          "towards hanging [per rad^2], times how hidden it is (0 when 30 % or "
                                          "more of it is seen): an arm nobody sees is most likely hanging",
                                     effect="larger: a hidden arm drops faster; 0: it keeps its last pose")
    rounds: int = param(3, "correspondence rounds per frame")
    iterations: int = param(30, "L-BFGS iterations per round",
                            effect="12 (version 1.3) stops before the fit converges: on synthetic jumps the joint "
                                   "error rose from about 11 to 14 mm, the head turned by up to 20 deg, and the "
                                   "unconverged frames needed limb searches")
    coarse_scale: float = param(4.0, "robust scale of the first round, as a multiple of robust_scale (halved every "
                                     "round down to robust_scale)",
                                effect="larger: catches faster limbs, but stray points pull more in the first round")
    init_yaws: int = param(8, "facing directions tried on the first frame and after the person is lost")
    init_facings: int = param(4, "of the facings the init_yaws quick fits end in, the ones with the lowest "
                                 "energies (at most this many, more than 45 deg apart) are fitted as a whole "
                                 "and compared on the points",
                              effect="1: the facing of the lowest quick-fit energy, as before version 1.4.1 (a "
                                     "person seen side-on can be tracked facing backwards)")
    lost_residual: float = param(0.05, "median point distance above which a frame is fitted again from several "
                                       "starts", unit="m")
    recover_distance: float = param(0.06, "a point farther than this from the visible body is unexplained", unit="m")
    recover_fraction: float = param(0.02, "fraction of unexplained points above which the frame is fitted again "
                                          "from other starts: the previous pose, legs, arms (0: never)",
                                    effect="smaller: more searches (about 5 s per frame on a CPU)")
    recover_free_space: float = param(0.03, "fraction of the visible body lying where the sensor saw empty space "
                                            "above which the frame is also fitted again from other starts (0: "
                                            "never)",
                                      effect="0: a limb or a turned trunk placed in empty space is never searched "
                                             "again (version 1.3)")
    torso_search: bool = param(True, "the search of a poorly fitted frame also tries the trunk, neck and head "
                                     "untwisted, and the whole body turned by +-yaw_search")
    yaw_search: float = param(35.0, "turn of the body tried by the trunk search", unit="deg")
    gap_frames: float = param(2.5, "after more than this many frame periods without a tracked frame (frames "
                                   "dropped by the segmentation) the prediction keeps the last pose instead of "
                                   "extrapolating its velocity",
                              effect="large: a limb in motion before a gap is thrown on along its velocity")
    max_turn_rate: float = param(240.0, "fastest turn of the whole body believed: a fit or a search that turns "
                                        "the body faster than this (plus 40 deg) against any frame of the last second "
                                        "is refused, and the plain fit from the prediction is kept (0: no limit)",
                                 unit="deg/s",
                                 effect="0: a frame whose points are hard to explain can turn the body around (in "
                                        "the review of s02_jump02, version 1.4.1, the body turns by about 180 deg "
                                        "within 0.2 s at the first take-off and stays so)")
    facing_check: bool = param(True, "a frame still poorly explained after the searches, five frames in a row, is "
                                     "fitted again from the opposite facing and from the two side-on ones (at most "
                                     "every ten frames); a fit that explains the points clearly better (3 mm) is kept "
                                     "when the body can turn there physically: within max_turn_rate of the frames "
                                     "before, or by a turn spread over them (connect_frames)",
                               effect="false: a body tracked facing the wrong way stays so")
    connect_frames: int = param(30, "a better facing that the body cannot reach from the frames before at "
                                    "max_turn_rate is kept only if the frames before, fitted again backwards from it "
                                    "(at most this many), explain their points as well as before (connect_tolerance) "
                                    "and meet the earlier track: the turn is then spread over them; otherwise it is "
                                    "refused (0: refused always)",
                              effect="version 1.5 kept such a facing at once: in 5 of the 27 s02 takes the body turned "
                                     "by 50 to 177 deg between two frames (up to 3500 deg/s), in run01_3 it then ran "
                                     "backwards")
    connect_tolerance: float = param(0.003, "how much worse (score of explain(), about the mean point distance) a "
                                            "frame fitted again backwards may explain its points than before", unit="m")
    hinge_weight: float = param(20.0, "weight keeping the elbows and knees in their plane of flexion "
                                      "(bodyscan.body.anatomy: out of plane at most 5 deg varus and 12 deg valgus at the "
                                      "elbow, 6 deg at the knee) [per rad^2]",
                                effect="0: elbows bent sideways at the edge of the joint box (20 deg) instead of the "
                                       "upper arm turned (versions 1.3 to 1.5: elbows out of plane by more than 5 deg in "
                                       "30 to 100 % of the frames of the s02 takes, knees by 8 deg in up to half the "
                                       "frames of the side-on jumps)")
    joint_speed_scale: float = param(1.0, "scale of the fastest rotation each joint may make between two frames "
                                          "(bodyscan.body.anatomy.SPEED_LIMITS: 400 deg/s for each spine joint, 600 "
                                          "for the collars, 1000 to 1500 for the limbs); a faster change is spread "
                                          "over the frames around it. Raise it for throwing or kicking; 0: no limit",
                                     effect="0 (version 1.5): single frames where a joint jumped by 30 to 110 deg and "
                                            "back, in up to 3.6 % of the frames of the s02 takes")
    penetration_weight: float = param(2.0, "weight keeping the arms out of the trunk: a vertex of a forearm, a "
                                           "hand or the lower half of an upper arm deeper than penetration_margin "
                                           "inside the trunk is pushed out (stage 1 and 2; summed over the vertices "
                                           "inside, per ten of them)",
                                      effect="0: an arm the sensor does not see can sink into the trunk (version "
                                             "1.4); version 1.5 averaged over all the arm samples, so that a forearm "
                                             "5 cm inside cost little (s02 takes: arms more than 2 cm inside in 15 to "
                                             "50 % of the frames)")
    penetration_margin: float = param(0.01, "depth of an arm inside the trunk that is not penalised (soft tissue, "
                                            "clothes, an avatar slightly wider than the person)", unit="m",
                                      effect="0.02 (version 1.5): arms visibly inside the trunk")


@dataclass
class RefineConfig:
    """Whole-sequence refinement (stage 2): natural motion."""
    enabled: bool = param(True, "refine the whole sequence after the frame-by-frame fit")
    window: int = param(40, "frames optimised together")
    overlap: int = param(10, "frames shared with the previous window (held, for continuity)")
    lookahead: int = param(10, "frames after each window optimised with it and refined again with the next "
                               "window: the last frames of a window see what comes after them",
                           effect="0: the end of each window is refined without the frames that follow (version "
                                  "1.4); larger: slower")
    free_space_hidden: bool = param(True, "the free-space term also covers the limb vertices the body model hides "
                                          "behind itself: a hidden limb placed where the sensor saw empty space (out "
                                          "of the shadow of the trunk) is pulled back like a visible one; the "
                                          "measured ranges tell what really hid it. Stage 2 only: in stage 1 the "
                                          "backs of the limbs along the silhouette edges (a pixel out) made the "
                                          "limb searches noisy",
                                    effect="false: only vertices the model shows the sensor are tested (version 1.4)")
    rounds: int = param(2, "correspondence rounds per window")
    iterations: int = param(40, "L-BFGS iterations per round")
    smoothness: str = param("jerk", "what the smoothness term penalises: the jerk (change of acceleration) or "
                                    "the acceleration of the joints", choices=("jerk", "acceleration"),
                            effect="acceleration also pulls a jump's flight (free fall, 9.81 m/s2) flatter")
    smoothness_weight: float = param(0.5, "weight of the smoothness term",
                                     effect="larger: smoother motion, sharp events (landings) softened")
    hidden_smoothness: float = param(10.0, "the smoothness weight of a joint whose body part is less than 30 % "
                                           "visible is multiplied by up to 1 + this (fully hidden)")
    smoothness_scale: float = param(200.0, "Charbonnier scale of the smoothness term: below it smoothed like a "
                                           "spring, above it penalised linearly (m/s3 for jerk, m/s2 for "
                                           "acceleration)")
    contact_height: float = param(0.035, "a foot whose lowest vertex is below this is on the floor if also slow",
                                  unit="m")
    contact_speed: float = param(0.35, "... and its ankle moves slower than this", unit="m/s")
    skating_weight: float = param(2.0, "weight of the no-sliding of feet on the floor")
    rolling_shutter: bool = param(True, "compare every point with the body at the time it was measured")


@dataclass
class TrackPipelineConfig:
    tracking: TrackConfig = section(TrackConfig)
    refine: RefineConfig = section(RefineConfig)
    coupling: CouplingConfig = section(CouplingConfig)


# Starting poses of an arm for the search of recover_limbs: (elevation, swing,
# twist, elbow) [deg] as in bodyscan.dynamic.motions.Angles (elevation 0:
# hanging, 90: horizontal at the side; swing: forward; twist 90: palm forward).
ARM_STARTS = (
    (40, 0, 90, 10),        # A-pose
    (10, 0, 0, 15),         # hanging relaxed: palm towards the thigh, forearm a little forward (version 1.5: palm
                            # forward by the upper arm turned out 90 deg, so the elbow bent sideways)
    (10, 0, 90, 90),        # hanging, elbow bent
    (90, 0, 90, 10),        # horizontal at the side
    (15, 90, 0, 10),        # horizontal in front
    (15, 160, 0, 10),       # up, in front
    (165, 0, 90, 10),       # up, at the side
    (15, -40, 0, 15),       # back
    (10, 0, 0, 90),         # hanging, elbow bent, forearm forward (jogging, carrying)
    (10, 35, 0, 95),        # running, swung forward: forearm forward and up
    (10, -35, 0, 95),       # running, swung back: forearm forward, elbow behind the trunk
    (20, 60, 0, 30),        # swinging forward and up (a jump's take-off)
)


# Joints whose rotation about the long axis (y in the T-pose frames: the spine,
# the thighs) trades with a turn of the pelvis (indices into the 21 body joints).
TWIST_JOINTS = [skeleton.JOINT[name] - 1 for name in ("spine1", "spine2", "spine3", "left_hip", "right_hip")]

# Neck and head (indices into the 21 body joints): pulled towards upright by head_weight.
HEAD_JOINTS = [skeleton.JOINT[name] - 1 for name in ("neck", "head")]

# Forearm twist: the x component of the elbow rotation (the elbow bends about y); and the
# wrists, all three components: pulled towards rest by forearm_twist_weight.
ELBOW_JOINTS = [skeleton.JOINT[name] - 1 for name in ("left_elbow", "right_elbow")]
WRIST_JOINTS = [skeleton.JOINT[name] - 1 for name in ("left_wrist", "right_wrist")]

# The trunk, neck and head joints reset by the trunk search (twist and side bend only; the
# forward bend stays), and the limbs of the searches (indices into the 21 body joints).
TRUNK_JOINTS = [skeleton.JOINT[name] - 1 for name in ("spine1", "spine2", "spine3")]
LIMB_JOINT_NAMES = {
    "legs": [f"{side}_{name}" for side in ("left", "right") for name in ("hip", "knee", "ankle")],
    "left_arm": [f"left_{name}" for name in ("collar", "shoulder", "elbow", "wrist")],
    "right_arm": [f"right_{name}" for name in ("collar", "shoulder", "elbow", "wrist")],
}
# Body parts (skeleton.PART_NAMES) of each limb, to tell which limb a misfit belongs to.
LIMB_PARTS = {
    "legs": ("left_thigh", "right_thigh", "left_shank", "right_shank", "left_foot", "right_foot"),
    "left_arm": ("left_upper_arm", "left_forearm", "left_hand"),
    "right_arm": ("right_upper_arm", "right_forearm", "right_hand"),
}
# Joints (full skeleton) that stand for each limb when a point is assigned to the nearest limb.
LIMB_JOINTS = {
    "legs": ("left_knee", "right_knee", "left_ankle", "right_ankle", "left_foot", "right_foot"),
    "left_arm": ("left_elbow", "left_wrist"),
    "right_arm": ("right_elbow", "right_wrist"),
    "trunk": ("pelvis", "spine2", "spine3", "neck", "head", "left_shoulder", "right_shoulder", "left_hip",
              "right_hip"),
}


def neutral_priors(body, config) -> object:
    """Soft priors of a body pose (torch, (..., 21, 3)) towards neutral where the points tell
    little: spine and hip twist, neck and head, forearm twist and wrists."""
    energy = config.twist_weight * (body[..., TWIST_JOINTS, 1] ** 2).sum()
    if config.trunk_twist_weight > 0:
        energy = energy + config.trunk_twist_weight * (body[..., TRUNK_JOINTS, 1].sum(-1) ** 2).sum()
    if config.head_weight > 0:
        energy = energy + config.head_weight * (body[..., HEAD_JOINTS, :] ** 2).sum()
    if config.forearm_twist_weight > 0:
        energy = energy + config.forearm_twist_weight * ((body[..., ELBOW_JOINTS, 0] ** 2).sum()
                                                          + (body[..., WRIST_JOINTS, :] ** 2).sum())
    return energy


def free_space_loss(distance, scale: float, kind: str):
    """Penalty of a free-space distance [m] (torch), in the units of geman_mcclure(d, scale)
    (bounded by scale^2). 'bounded': Geman-McClure at the scale. 'wide': 0.6 of it plus 0.4 of a
    Geman-McClure at 4 times the scale (rescaled to the same bound): the same bound, so leaving
    the points for empty space never pays more than before, but a pull that reaches about four
    times farther (a limb 20 to 50 cm out in empty space is pulled back)."""
    from bodyscan.dynamic.fitting import geman_mcclure
    if kind == "bounded":
        return geman_mcclure(distance, scale)
    wide = 4.0 * scale
    return 0.6 * geman_mcclure(distance, scale) + 0.4 * (scale / wide) ** 2 * geman_mcclure(distance, wide)

# A limb counts as observed (keeps its tracked pose in complete_hidden_limbs) from this fraction of its
# full view seen with points (coupling.limb_observation). A far leg seen side-on is mostly seen in part
# (between the legs, below the near knee) and its tracked pose is better than the guess there (synthetic
# side-on jumps: far foot 35 mm tracked, 52 mm guessed where a quarter to a half of it showed; a jog that
# starts from standing, the far leg seen at 21 to 25 %: far ankle 16 mm anchored from a quarter, 12 mm
# from a fifth); a far arm seen in part (a hand behind the near arm) is not (an arm moving unlike its
# partner: far wrist 51 mm with the arm anchored from half seen, 87 mm from a quarter).
OBSERVED_LIMB = {"arm": 0.5, "leg": 0.2}

# A guessed arm is moved out of the sensor's sight (Tracker.hide_guessed_limbs) where more than this
# much of it is guessed, and the changes are smoothed over this time [s] (synthetic jog across the line
# of sight: per frame alone, the far hand's velocity error rose from 1.04 to 1.49 m/s; smoothed over
# 0.1 s 1.07, over 0.15 s 1.04, with the far wrist 104 mm from the truth instead of 120).
HIDE_FROM = 0.3
HIDE_SMOOTHING = 0.15


# Facings closer than this are one candidate of the initialisation (Tracker.initialise).
INIT_FACING_GAP = np.radians(45.0)

# Starting poses of both legs for the same search: (hip flexion, knee flexion,
# ankle dorsiflexion) [deg]: standing, half squat, deep squat (a landing).
LEG_STARTS = ((0, 3, 0), (35, 55, 15), (65, 95, 25))


def _leg_rotations(hip, knee, ankle) -> dict:
    """Hip, knee and ankle rotations of both legs: {index in the (21, 3) body pose: axis-angle}."""
    from bodyscan.dynamic.motions import Angles, angles_to_body
    angles = Angles(1)
    for side in ("left", "right"):
        for name, value in (("hip", hip), ("knee", knee), ("ankle", ankle)):
            getattr(angles, f"{side}_{name}")[:] = value
    body = angles_to_body(angles)[0]
    return {skeleton.JOINT[f"{side}_{name}"] - 1: body[skeleton.JOINT[f"{side}_{name}"] - 1]
            for side in ("left", "right") for name in ("hip", "knee", "ankle")}


def _arm_rotations(side: str, elevation, swing, twist, elbow) -> dict:
    """Shoulder and elbow rotations of one arm: {index in the (21, 3) body pose: axis-angle}."""
    from bodyscan.dynamic.motions import Angles, angles_to_body
    angles = Angles(1)
    for name, value in (("elevation", elevation), ("swing", swing), ("twist", twist), ("elbow", elbow)):
        getattr(angles, f"{side}_{name}")[:] = value
    body = angles_to_body(angles)[0]
    return {index: body[index] for index in (skeleton.JOINT[f"{side}_shoulder"] - 1,
                                               skeleton.JOINT[f"{side}_elbow"] - 1)}


# ----------------------------------------------------------------------------
# Observations of one frame
# ----------------------------------------------------------------------------

class FrameData:
    """One segmented frame on the device: the person's points and the silhouette crop."""

    def __init__(self, person, geometry, device, dtype):
        import torch
        from bodyscan.dynamic.fitting import distance_map
        self.index, self.time = person.index, person.time
        self.points_np = person.points
        self.points = torch.as_tensor(person.points, dtype=dtype, device=device)
        self.point_times = person.point_times
        self.crop_origin = person.crop_origin
        self.mask = person.crop_mask
        self.crop_range = person.crop_range
        self.crop_background = person.crop_background
        self.crop_times = person.crop_times                     # None for segmentations of version 1.2.0 beta
        h, w = self.mask.shape
        self.reference_column = person.crop_origin[1] + w / 2.0
        # one pixel around the person is uncertain: the edge filter of the segmentation removed mixed
        # pixels there, so the sensor did not clearly see through it
        from bodyscan.dynamic.segment import _dilate
        self.uncertain = _dilate(self.mask > 0, 1)
        # columns lost with a UDP packet: no return in the whole crop height although the empty
        # scene returns there; the sensor saw nothing of them, so they are not free space
        with np.errstate(invalid="ignore"):
            self.lost_columns = (np.all(~np.isfinite(self.crop_range), axis=0)
                                 & (np.mean(np.isfinite(self.crop_background), axis=0) > 0.5))
        self.uncertain[:, self.lost_columns] = True
        silhouette = distance_map(self.uncertain, geometry.vertical_step, geometry.column_step)
        self.distance = torch.as_tensor(silhouette, dtype=dtype, device=device)
        self.center = person.center
        self._tree = None

    def tree(self):
        """k-d tree of the person's points (built on first use)."""
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.points_np)
        return self._tree


class TrunkCollision:
    """Arms kept out of the trunk. Samples of the arm surface (forearms, hands and the lower halves
    of the upper arms: the upper end of an upper arm always touches the trunk) and of the trunk
    surface (pelvis to chest, without the armpits); the signed distance of an arm sample to the
    trunk is taken as the largest distance to the tangent planes of its nearest trunk samples
    (negative inside: exact for a convex trunk, close for a real one at the depths that matter)."""

    NEAREST = 4

    def __init__(self, shaped, part_labels: np.ndarray, arm_samples: int = 120, trunk_samples: int = 500):
        rest = shaped.vertices.detach().cpu().numpy().astype(np.float64)
        joints = shaped.rest_joints.detach().cpu().numpy().astype(np.float64)
        names = np.array(skeleton.PART_NAMES)[part_labels]
        J = skeleton.JOINT
        arm = []
        for side in ("left", "right"):
            shoulder, elbow = joints[J[f"{side}_shoulder"]], joints[J[f"{side}_elbow"]]
            axis = elbow - shoulder
            along = (rest - shoulder) @ axis / max(float(axis @ axis), 1e-9)
            lower_upper_arm = (names == f"{side}_upper_arm") & (along > 0.5)
            members = np.flatnonzero(lower_upper_arm | np.isin(names, [f"{side}_forearm", f"{side}_hand"]))
            arm.append(self._spread(members, arm_samples))
        self.arm_of = {"left": arm[0], "right": arm[1]}
        self.arm = np.concatenate(arm)
        trunk = np.isin(names, ["pelvis", "torso_lower", "torso_middle", "torso_upper"])
        for side in ("left", "right"):
            trunk &= np.linalg.norm(rest - joints[J[f"{side}_shoulder"]], axis=1) > 0.12
        self.trunk = self._spread(np.flatnonzero(trunk), trunk_samples)

    @staticmethod
    def _spread(indices: np.ndarray, count: int) -> np.ndarray:
        """About 'count' of the indices, evenly over the list (vertex order follows the surface)."""
        if len(indices) <= count:
            return indices
        return indices[np.round(np.linspace(0, len(indices) - 1, count)).astype(int)]

    # the energy counts the squared excess depth of the arm samples per this many of them: a forearm
    # inside the trunk costs about as much whether few or many of its samples are (version 1.5 took the
    # mean over all the samples of both arms, so a forearm 5 cm inside cost little)
    PER_SAMPLES = 10

    def energy(self, vertices, trunk_normals, margin: float, scale: float = 0.02, chunk: int = 16):
        """Sum over frames of the squared excess depth (in units of 'scale') of the arm samples
        inside the trunk, per PER_SAMPLES samples. vertices (F, V, 3) torch; trunk_normals (F, trunk
        samples, 3), detached."""
        return self.energy_of(vertices[:, self.arm], vertices[:, self.trunk], trunk_normals, margin, scale, chunk)

    def energy_of(self, arm, trunk, trunk_normals, margin: float, scale: float = 0.02, chunk: int = 16):
        """energy() from the arm samples (F, A, 3) and the trunk samples (F, T, 3) themselves."""
        import torch
        with torch.no_grad():
            nearest = torch.cat([torch.cdist(arm[i:i + chunk].detach(), trunk[i:i + chunk].detach())
                                 .topk(self.NEAREST, dim=-1, largest=False).indices
                                 for i in range(0, arm.shape[0], chunk)])                   # (F, A, K)
        frames = torch.arange(arm.shape[0], device=arm.device)[:, None, None]
        points = trunk[frames, nearest]                                                     # (F, A, K, 3)
        normals = trunk_normals[frames, nearest]
        signed = ((arm[:, :, None] - points) * normals).sum(-1).max(dim=-1).values         # (F, A)
        excess = torch.relu(-signed - margin) / scale
        return (excess ** 2).sum() / self.PER_SAMPLES

    def depth(self, vertices: np.ndarray, normals: np.ndarray) -> float:
        """Deepest arm sample inside the trunk [m] (0 when none) of one posed body (numpy)."""
        return self.depth_of(vertices[self.arm], vertices[self.trunk], normals[self.trunk])

    @classmethod
    def depth_of(cls, arm: np.ndarray, trunk: np.ndarray, trunk_normals: np.ndarray) -> float:
        """Deepest of the points 'arm' inside the trunk given by its samples and their normals [m]."""
        from scipy.spatial import cKDTree
        _, nearest = cKDTree(trunk).query(arm, k=cls.NEAREST)
        signed = ((arm[:, None] - trunk[nearest]) * trunk_normals[nearest]).sum(-1).max(axis=1)
        return float(max(-signed.min(), 0.0))


@dataclass
class FrameMatch:
    """Correspondences of one frame for one optimisation round."""
    point_index: object
    face_ids: object
    bary: object
    silhouette: object            # vertex indices in free space
    silhouette_pixels: tuple      # their pixels (rows, columns) for the projection
    front: object                 # vertex indices in front of the measured surface
    front_pixels: tuple
    front_range: object           # measured ranges at those vertices
    visible: np.ndarray           # boolean per vertex
    visible_count: int
    silhouette_offsets: np.ndarray = None     # time of their pixel minus the frame time [s] (float64)
    front_offsets: np.ndarray = None
    distance: np.ndarray = None               # every point's distance to the visible body [m]


# ----------------------------------------------------------------------------
# Tracker
# ----------------------------------------------------------------------------

class Tracker:
    def __init__(self, avatar, model, segmented, config: TrackPipelineConfig | None = None):
        import torch
        from bodyscan.dynamic.motions import foot_vertices
        self.config = config or TrackPipelineConfig()
        c = self.config.tracking
        self.model = model
        self.avatar = avatar
        self.segmented = segmented
        self.geometry = segmented.geometry
        self.device, self.dtype = model.device, model.dtype
        self.shaped = avatar.shaped(model, c.level)
        self.faces_np = self.shaped.faces_np
        self.faces = self.shaped.faces
        self.tables = self.geometry.torch_tables(self.device, self.dtype)
        self.sensor = segmented.sensor_position
        self.feet = torch.as_tensor(foot_vertices(self.shaped), device=self.device)
        self.feet_np = foot_vertices(self.shaped)
        self.hands = model.rotations(count=1)[0, skeleton.LEFT_HAND.start:]          # relaxed hands, fixed
        labels = skeleton.part_labels(self.shaped.weights.detach().cpu().numpy())
        self.part_labels = labels
        self.part_count = np.bincount(labels, minlength=len(skeleton.PART_NAMES))
        lower, upper = skeleton.limit_arrays()
        # start from the scan pose with the arms hanging: the scan's A-pose would stay on an arm
        # the sensor never sees
        self.hanging = {}
        for side in ("left", "right"):
            self.hanging.update(_arm_rotations(side, *ARM_STARTS[1]))
        start = np.array(avatar.scan_body, dtype=np.float64).reshape(21, 3).copy()
        for index, value in self.hanging.items():
            start[index] = value
        self.start_body = np.clip(start, lower[skeleton.BODY], upper[skeleton.BODY])
        self.hanging_index = np.array(sorted(self.hanging))
        self.hanging_body = np.array([self.hanging[i] for i in self.hanging_index])
        segmentation = segmented.summary.get("config", {})
        self.foreground_margin = float(segmentation.get("bg_threshold", 0.08)) + 0.02
        self.background_range = segmented.background_range                # (H, W) empty room [m], NaN: unknown
        frame_times = segmented.times()
        steps = np.diff(frame_times) if len(frame_times) > 1 else np.zeros(0)
        self.frame_period = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 0.05
        self.collision = TrunkCollision(self.shaped, labels)
        limb_parts = [skeleton.PART_NAMES.index(f"{side}_{part}") for side in ("left", "right")
                      for part in ("upper_arm", "forearm", "hand", "thigh", "shank", "foot")]
        self.limb_vertices = np.isin(labels, limb_parts)
        self.coupling_plan = None

    # -- posing ---------------------------------------------------------------------------------
    def rotations(self, body):
        """(F, 55, 3, 3) from body pose (F, 21, 3) (torch), hands relaxed."""
        import torch
        from bodyscan.body.rotations import axis_angle_to_matrix
        f = body.shape[0]
        head = torch.zeros(f, 1, 3, dtype=self.dtype, device=self.device)
        tail = torch.zeros(f, 3, 3, dtype=self.dtype, device=self.device)
        main = axis_angle_to_matrix(torch.cat([head, body, tail], dim=1))
        return torch.cat([main, self.hands.expand(f, -1, -1, -1)], dim=1)

    def pose(self, omega, reference, position, body, subset=None):
        """World vertices and joints of F frames: root rotation exp(omega) . reference."""
        from bodyscan.body.rotations import axis_angle_to_matrix
        root = axis_angle_to_matrix(omega) @ reference
        posed = self.shaped.pose(self.rotations(body), root, position, subset=subset)
        return posed.vertices, posed.joints

    # -- correspondences ------------------------------------------------------------------------
    def match(self, vertices: np.ndarray, normals: np.ndarray, frame: FrameData,
              max_distance: float | None = None, hidden: bool = False) -> FrameMatch:
        """Correspondences of the points with the visible body, and the vertices in free space or in
        front of the measured surface: of the visible vertices, and with 'hidden' of the limbs wherever
        they are (stage 2, refine.free_space_hidden)."""
        import torch
        from bodyscan.dynamic.fitting import closest_points, visible_faces, visible_vertices
        c = self.config.tracking
        g = self.geometry
        visible = visible_vertices(vertices, normals, g, self.sensor, c.visibility_tolerance)
        faces = visible_faces(self.faces_np, visible)
        face_ids, bary, distance = closest_points(vertices, self.faces_np, frame.points_np, faces)
        valid = np.flatnonzero(distance < (c.max_correspondence if max_distance is None else max_distance))
        # the free-space and in-front tests: the visible vertices, and with 'hidden' the limbs wherever
        # they are (behind the measured surface a vertex is neither free nor in front, so the measured
        # ranges decide what hides it, not the model)
        tested = visible | self.limb_vertices if hidden else visible
        index = np.flatnonzero(tested)
        rows, cols, ranges = g.project(vertices[index])
        unwrapped = g.column_distance_unwrapped(cols, frame.reference_column)
        r0, c0 = frame.crop_origin
        h, w = frame.mask.shape
        rr = np.round(rows - r0).astype(int)
        cc = np.round(unwrapped - c0).astype(int)
        in_view = (rows > -0.5) & (rows < g.height - 0.5)
        in_crop = in_view & (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rr_c, cc_c = np.clip(rr, 0, h - 1), np.clip(cc, 0, w - 1)
        pixel_rows = np.clip(np.round(rows).astype(int), 0, g.height - 1)
        pixel_cols = np.mod(np.round(cols).astype(int), g.width)
        measured = np.where(in_crop, frame.crop_range[rr_c, cc_c], np.nan)
        # outside the crop only the empty room is known: no person was found there
        background = np.where(in_crop, frame.crop_background[rr_c, cc_c],
                              self.background_range[pixel_rows, pixel_cols])
        person = in_crop & frame.uncertain[rr_c, cc_c]
        margin = max(c.free_space_margin, self.foreground_margin)
        with np.errstate(invalid="ignore"):
            # free space: the sensor saw (or, outside the crop, would have seen) beyond the vertex, and the
            # vertex would have been segmented as the person had it been there (closer than the empty
            # scene by the segmentation margin); a vertex behind furniture is not free
            nearest = np.fmin(measured, background)
            beyond = np.isfinite(nearest) & (ranges < nearest - margin)
            free = in_view & ~person & beyond
            front = person & np.isfinite(measured) & (ranges < measured - margin)
        offsets = np.zeros(len(index))
        if frame.crop_times is not None:
            offsets = np.where(in_crop, frame.crop_times[rr_c, cc_c] - frame.time, 0.0)

        def tensor(values, dtype=None):
            return torch.as_tensor(values, device=self.device, dtype=dtype)

        return FrameMatch(tensor(valid), tensor(face_ids[valid]), tensor(bary[valid], self.dtype), tensor(index[free]),
                          (tensor(pixel_rows[free]), tensor(pixel_cols[free])), tensor(index[front]),
                          (tensor(pixel_rows[front]), tensor(pixel_cols[front])), tensor(measured[front], self.dtype),
                          visible, max(int(visible.sum()), 1), offsets[free], offsets[front], distance)

    def in_sight(self, vertices: np.ndarray, frame: FrameData) -> np.ndarray:
        """Per vertex (N, 3): where the sensor measured something else, so it would have seen the
        vertex there: empty space beyond it outside the person, or a surface behind it inside (the
        free-space and in-front tests of match() for any vertices, visible or not)."""
        c = self.config.tracking
        g = self.geometry
        rows, cols, ranges = g.project(vertices)
        unwrapped = g.column_distance_unwrapped(cols, frame.reference_column)
        r0, c0 = frame.crop_origin
        h, w = frame.mask.shape
        rr = np.round(rows - r0).astype(int)
        cc = np.round(unwrapped - c0).astype(int)
        in_view = (rows > -0.5) & (rows < g.height - 0.5)
        in_crop = in_view & (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rr_c, cc_c = np.clip(rr, 0, h - 1), np.clip(cc, 0, w - 1)
        pixel_rows = np.clip(np.round(rows).astype(int), 0, g.height - 1)
        pixel_cols = np.mod(np.round(cols).astype(int), g.width)
        measured = np.where(in_crop, frame.crop_range[rr_c, cc_c], np.nan)
        background = np.where(in_crop, frame.crop_background[rr_c, cc_c],
                              self.background_range[pixel_rows, pixel_cols])
        person = in_crop & frame.uncertain[rr_c, cc_c]
        margin = max(c.free_space_margin, self.foreground_margin)
        with np.errstate(invalid="ignore"):
            nearest = np.fmin(measured, background)
            free = in_view & ~person & np.isfinite(nearest) & (ranges < nearest - margin)
            front = person & np.isfinite(measured) & (ranges < measured - margin)
        return free | front

    # -- energy of one frame --------------------------------------------------------------------
    def frame_energy(self, vertices, frame: FrameData, match: FrameMatch, shifted_points=None,
                     shifted_silhouette=None, shifted_front=None, scale: float | None = None):
        """Data, silhouette, front and floor terms of one frame (vertices (V, 3) torch).
        shifted_points: optional (M, 3) model points already moved to the point times;
        shifted_silhouette, shifted_front: the vertices of those terms moved to the times of their pixels;
        scale: robust scale of the data term (default robust_scale; the normalisation stays that of
        robust_scale, so a larger scale widens the reach without changing the weight of small residuals)."""
        import torch
        from bodyscan.dynamic.fitting import face_normals, geman_mcclure, sample_map, surface_points
        c = self.config.tracking
        s2 = c.robust_scale ** 2
        data_scale = c.robust_scale if scale is None else scale
        energy = vertices.new_zeros(())
        if len(match.point_index):
            points = frame.points[match.point_index]
            model_points = (surface_points(vertices, self.faces, match.face_ids, match.bary)
                            if shifted_points is None else shifted_points)
            normals = face_normals(vertices, self.faces, match.face_ids)
            diff = points - model_points
            plane = (diff * normals).sum(-1)
            energy = energy + (geman_mcclure(plane, data_scale).mean()
                               + c.point_to_point * geman_mcclure(diff.norm(dim=-1), data_scale).mean()) / s2
        g = self.geometry
        r0, c0 = frame.crop_origin
        h, w = frame.mask.shape
        if len(match.silhouette):
            moved = vertices[match.silhouette] if shifted_silhouette is None else shifted_silhouette
            rows, cols, ranges = g.project_torch(moved, self.tables, frame.reference_column, match.silhouette_pixels)
            rr, cc = rows - r0, cols - c0
            inside = sample_map(frame.distance, rr, cc)
            out_r = torch.relu(-rr) + torch.relu(rr - (h - 1))
            out_c = torch.relu(-cc) + torch.relu(cc - (w - 1))
            angle = inside + torch.sqrt((out_r * g.vertical_step) ** 2 + (out_c * g.column_step) ** 2 + 1e-12)
            lateral = angle * ranges
            scale = c.silhouette_scale
            energy = energy + (c.silhouette_weight * free_space_loss(lateral, scale, c.free_space_loss).sum()
                               / scale ** 2 / match.visible_count)
        if len(match.front):
            moved = vertices[match.front] if shifted_front is None else shifted_front
            _, _, ranges = g.project_torch(moved, self.tables, frame.reference_column, match.front_pixels)
            excess = torch.relu(match.front_range - ranges - c.free_space_margin)
            energy = energy + (c.silhouette_weight * free_space_loss(excess, c.robust_scale, c.free_space_loss).sum()
                               / s2 / match.visible_count)
        feet_z = vertices[self.feet, 2]
        energy = energy + c.floor_weight * (torch.relu(-feet_z) ** 2).mean() / s2
        return energy

    # -- stage 1 ----------------------------------------------------------------------------------
    def fit_frame(self, frame: FrameData, reference, position, body, prediction, rounds=None, iterations=None,
                  free_body=True, coarse=True, joints=None):
        """Fit one frame from a start (reference rotation (3, 3), position (3,), body (21, 3) numpy).
        prediction: (reference, position, body) the continuity term pulls towards, or None.
        coarse: start the rounds at the coarse robust scale; joints: indices (into the 21 body
        joints) of the only joints that move (the root stays), for searches of one limb.
        Returns (rotation matrix, position, body, energy, match)."""
        import torch
        from bodyscan.body.anatomy import hinge_penalty
        from bodyscan.body.model import vertex_normals
        from bodyscan.dynamic.fitting import limit_penalty
        c = self.config.tracking
        rounds = c.rounds if rounds is None else rounds
        iterations = c.iterations if iterations is None else iterations
        dev, dt = self.device, self.dtype
        reference_t = torch.as_tensor(reference, dtype=dt, device=dev).reshape(1, 3, 3)
        omega = torch.zeros(1, 3, dtype=dt, device=dev, requires_grad=True)
        pos = torch.as_tensor(position, dtype=dt, device=dev).reshape(1, 3).clone().requires_grad_(True)
        b = torch.as_tensor(body, dtype=dt, device=dev).reshape(1, 21, 3).clone().requires_grad_(free_body)
        if prediction is not None:
            from bodyscan.body.rotations import matrix_to_axis_angle_np
            pred_ref, pred_pos, pred_body = prediction
            pred_omega = torch.as_tensor(matrix_to_axis_angle_np(pred_ref @ reference.T), dtype=dt, device=dev)
            pred_pos = torch.as_tensor(pred_pos, dtype=dt, device=dev)
            pred_body = torch.as_tensor(pred_body, dtype=dt, device=dev).reshape(1, 21, 3)
        tensors = [omega, pos] + ([b] if free_body else [])
        if joints is not None:
            mask = torch.zeros(1, 21, 1, dtype=dt, device=dev)
            mask[0, list(joints)] = 1.0
            b.register_hook(lambda grad: grad * mask)
            tensors = [b]
        energy_value, match = None, None
        for number in range(rounds):
            # coarse to fine: the robust scale halves every round down to robust_scale
            scale = (c.robust_scale * max(c.coarse_scale / 2.0 ** number, 1.0) if coarse and number < rounds - 1
                     else c.robust_scale)
            with torch.no_grad():
                vertices, _ = self.pose(omega, reference_t, pos, b)
                normals = vertex_normals(vertices, self.faces)
                trunk_normals = normals[:, self.collision.trunk]
            match = self.match(vertices[0].cpu().numpy().astype(np.float64),
                               normals[0].cpu().numpy().astype(np.float64), frame,
                               max(c.max_correspondence, 3.0 * scale))
            optimizer = torch.optim.LBFGS(tensors, lr=1.0, max_iter=iterations, history_size=10,
                                          line_search_fn="strong_wolfe")
            seen = self.joint_seen(match.visible)[1:]                  # the 21 body joints
            continuity = torch.as_tensor(1.0 + c.hidden_continuity * self.hidden_factor(seen), dtype=dt,
                                         device=dev).reshape(1, 21, 1)
            relax = torch.as_tensor(c.relax_hidden_arms * self.hidden_factor(seen)[self.hanging_index], dtype=dt,
                                    device=dev).reshape(1, -1, 1)
            hanging = torch.as_tensor(self.hanging_body, dtype=dt, device=dev).reshape(1, -1, 3)

            def total():
                vertices, _ = self.pose(omega, reference_t, pos, b)
                energy = self.frame_energy(vertices[0], frame, match, scale=scale)
                energy = energy + c.limit_weight * limit_penalty(b)
                if c.hinge_weight > 0:
                    energy = energy + c.hinge_weight * hinge_penalty(b)
                energy = energy + neutral_priors(b, c)
                if c.penetration_weight > 0:
                    energy = energy + c.penetration_weight * self.collision.energy(vertices, trunk_normals,
                                                                                   c.penetration_margin)
                if c.relax_hidden_arms > 0:
                    energy = energy + (((b[:, self.hanging_index] - hanging) ** 2) * relax).sum()
                if prediction is not None:
                    energy = energy + c.pose_continuity * (((b - pred_body) ** 2) * continuity).sum()
                    energy = energy + c.root_continuity * (((omega[0] - pred_omega) ** 2).sum()
                                                           + ((pos[0] - pred_pos) ** 2).sum() / 0.05 ** 2)
                return energy

            def closure():
                optimizer.zero_grad()
                energy = total()
                energy.backward()
                return energy

            optimizer.step(closure)
            with torch.no_grad():
                energy_value = float(total().item())
        from bodyscan.body.rotations import axis_angle_to_matrix
        with torch.no_grad():
            rotation = (axis_angle_to_matrix(omega) @ reference_t)[0].cpu().numpy().astype(np.float64)
        return (rotation, pos.detach()[0].cpu().numpy().astype(np.float64),
                b.detach()[0].cpu().numpy().astype(np.float64), energy_value, match)

    def residual(self, rotation, position, body, frame: FrameData) -> float:
        """Median distance [m] from the frame's points to the visible body surface."""
        return float(np.median(self.explain(rotation, position, body, frame)["distance"]))

    def explain(self, rotation, position, body, frame: FrameData) -> dict:
        """How well a pose explains a frame: the distance of every point to the visible body, the
        fraction of points between recover_distance and max_correspondence ('unexplained': farther
        points are clutter no limb can reach), and a score (lower is
        better): the mean point distance, each capped at 0.15 m, plus 0.15 m times the fraction of
        the visible vertices that lie where the sensor saw through."""
        import torch
        from bodyscan.body.model import vertex_normals
        with torch.no_grad():
            vertices, _ = self.pose(torch.zeros(1, 3, dtype=self.dtype, device=self.device),
                                    torch.as_tensor(rotation, dtype=self.dtype, device=self.device)[None],
                                    torch.as_tensor(position, dtype=self.dtype, device=self.device)[None],
                                    torch.as_tensor(body, dtype=self.dtype, device=self.device)[None])
            normals = vertex_normals(vertices, self.faces)
        match = self.match(vertices[0].cpu().numpy().astype(np.float64), normals[0].cpu().numpy().astype(np.float64),
                           frame)
        distance = match.distance
        c = self.config.tracking
        # points beyond max_correspondence are clutter (no limb can reach them): never a reason
        # to search the limbs again
        far = (distance > c.recover_distance) & (distance < c.max_correspondence)
        free = match.silhouette.cpu().numpy() if len(match.silhouette) else np.zeros(0, dtype=np.int64)
        return {"distance": distance, "unexplained": float(np.mean(far)), "unexplained_count": int(far.sum()),
                "score": float(np.mean(np.minimum(distance, 0.15)) + 0.15 * len(match.silhouette) / match.visible_count),
                "visible": match.visible, "far": far, "free": len(match.silhouette) / match.visible_count,
                "free_vertices": free,
                "observed_parts": self.observed_parts(vertices[0].cpu().numpy().astype(np.float64), match.visible,
                                                      frame)}

    def observed_parts(self, vertices: np.ndarray, visible: np.ndarray, frame: FrameData) -> np.ndarray:
        """Per body part (skeleton.PART_NAMES) the fraction of its vertices that the sensor sees with
        a point of the person within two robust scales (as in assess)."""
        parts = len(skeleton.PART_NAMES)
        index = np.flatnonzero(visible)
        if len(index) == 0 or len(frame.points_np) == 0:
            return np.zeros(parts)
        distance, _ = frame.tree().query(vertices[index], k=1,
                                         distance_upper_bound=2 * self.config.tracking.robust_scale)
        seen = index[np.isfinite(distance)]
        return np.bincount(self.part_labels[seen], minlength=parts) / np.maximum(self.part_count, 1)

    def limbs_to_search(self, rotation, position, body, frame: FrameData, explained: dict) -> set:
        """The limbs ('legs', 'left_arm', 'right_arm') and 'trunk' near the misfit of a frame: each
        unexplained point counts for the limb with the nearest of its joints (LIMB_JOINTS), each
        vertex in free space for the limb it belongs to."""
        import torch
        counts = {name: 0 for name in LIMB_JOINTS}
        if explained["unexplained_count"]:
            with torch.no_grad():
                _, joints = self.pose(torch.zeros(1, 3, dtype=self.dtype, device=self.device),
                                      torch.as_tensor(rotation, dtype=self.dtype, device=self.device)[None],
                                      torch.as_tensor(position, dtype=self.dtype, device=self.device)[None],
                                      torch.as_tensor(body, dtype=self.dtype, device=self.device)[None])
            joints = joints[0].cpu().numpy()
            names = list(LIMB_JOINTS)
            anchors = np.concatenate([joints[[skeleton.JOINT[j] for j in LIMB_JOINTS[n]]] for n in names])
            owner = np.concatenate([[k] * len(LIMB_JOINTS[n]) for k, n in enumerate(names)])
            points = frame.points_np[explained["far"]]
            nearest = np.argmin(np.linalg.norm(points[:, None] - anchors[None], axis=2), axis=1)
            for k in owner[nearest]:
                counts[names[k]] += 1
        if len(explained["free_vertices"]):
            parts = np.array(skeleton.PART_NAMES)[self.part_labels[explained["free_vertices"]]]
            for limb, members in LIMB_PARTS.items():
                counts[limb] += int(np.isin(parts, members).sum()) // 4      # vertices are denser than points
            counts["trunk"] += int((~np.isin(parts, sum(LIMB_PARTS.values(), ()))).sum()) // 4
        return {name for name, count in counts.items() if count >= 5}

    def joint_seen(self, visible: np.ndarray) -> np.ndarray:
        """(22,) fraction of the body part of each main joint (pelvis to wrists) that the sensor sees."""
        seen = np.zeros(len(skeleton.MAIN_JOINTS))
        for k, joint in enumerate(skeleton.MAIN_JOINTS):
            members = self.part_labels == skeleton.PART_NAMES.index(skeleton.PART_OF_JOINT[joint])
            seen[k] = visible[members].mean() if members.any() else 1.0
        return seen

    @staticmethod
    def hidden_factor(seen: np.ndarray) -> np.ndarray:
        """0 for a body part seen at 30 % or more, rising linearly to 1 for a part not seen at all."""
        return np.clip(1.0 - np.asarray(seen) / 0.3, 0.0, 1.0)

    def _arm_seen(self, side: str, visible: np.ndarray) -> float:
        """Fraction of the vertices of one arm (upper arm, forearm, hand) that the sensor sees."""
        parts = [skeleton.PART_NAMES.index(f"{side}_{name}") for name in ("upper_arm", "forearm", "hand")]
        members = np.isin(self.part_labels, parts)
        return float(visible[members].mean()) if members.any() else 0.0

    def recover_limbs(self, frame: FrameData, result, prediction, previous=None, limbs=None, trunk=False):
        """Fit the frame again from other starts: the previous frame's pose (without the
        prediction's velocity, which overshoots when a limb stops or turns); with 'trunk' the
        trunk, neck and head untwisted with the body as it is and turned by +-yaw_search (a trunk
        turned the wrong way puts the far half of the body where the sensor sees empty space);
        both legs from every pose of LEG_STARTS (a landing: knees forward, not feet forward), then
        each arm from every pose of ARM_STARTS. 'limbs' (a set of 'legs', 'left_arm', 'right_arm',
        None for all) limits the searches to the limbs near the misfit. Each limb start first
        moves only its limb (the rest of the fit stays), coarse to fine, and any start is kept
        when it lowers the score of explain() by 2 mm; an arm only when the sensor sees that arm in
        the result (at least 20 % of it), since no point can justify moving an arm the sensor does
        not see, or when the move explains points left unexplained (5 and 30 % at least: the hand
        of the far arm over the head) or takes the arm out of empty space (an arm in front of the
        body where the sensor saw nothing goes behind it). The pose kept is then fitted once more
        as a whole."""
        from bodyscan.body.rotations import rotation_z_np
        c = self.config.tracking
        best = result
        state = self.explain(best[0], best[1], best[2], frame)
        improved = False

        def attempt(trial, side=None):
            # an arm the trial leaves mostly hidden is accepted when it explains points that were
            # unexplained (the hand of the far arm over the head) or takes the arm out of empty
            # space (fewer vertices in free space): then the sensor does tell where it is
            nonlocal best, state, improved
            trial_state = self.explain(trial[0], trial[1], trial[2], frame)
            explains = trial_state["unexplained_count"] <= min(state["unexplained_count"] - 5,
                                                               0.7 * state["unexplained_count"])
            if trial_state["score"] < state["score"] - 0.002 and (
                    side is None or self._arm_seen(side, trial_state["visible"]) >= 0.2 or explains
                    or trial_state["free"] < state["free"] - 0.01):
                best, state, improved = trial, trial_state, True

        if previous is not None:
            attempt(self.fit_frame(frame, previous[0], previous[1], previous[2], prediction, coarse=False))
        if trunk and c.torso_search:
            untwisted = best[2].copy()
            untwisted[TRUNK_JOINTS, 1:] = 0.0                       # twist and side bend; the bend stays
            untwisted[HEAD_JOINTS] = 0.0
            rotation, position = best[0], best[1]
            for yaw in (0.0, c.yaw_search, -c.yaw_search):
                attempt(self.fit_frame(frame, rotation_z_np(np.radians(yaw)) @ rotation, position, untwisted,
                                       prediction, rounds=2, iterations=10))
        if limbs is None or "legs" in limbs:
            legs = [skeleton.JOINT[name] - 1 for name in LIMB_JOINT_NAMES["legs"]]
            for start in LEG_STARTS:
                body = best[2].copy()
                for index, value in _leg_rotations(*start).items():
                    body[index] = value
                attempt(self.fit_frame(frame, best[0], best[1], body, prediction, rounds=2, iterations=10,
                                       joints=legs))
        for side in ("left", "right"):
            if limbs is not None and f"{side}_arm" not in limbs:
                continue
            arm = [skeleton.JOINT[name] - 1 for name in LIMB_JOINT_NAMES[f"{side}_arm"]]
            for start in ARM_STARTS:
                body = best[2].copy()
                for index, value in _arm_rotations(side, *start).items():
                    body[index] = value
                attempt(self.fit_frame(frame, best[0], best[1], body, prediction, rounds=2, iterations=10,
                                       joints=arm), side)
        if improved:
            best = self.fit_frame(frame, best[0], best[1], best[2], prediction, rounds=2, coarse=False)
        return best

    def initialise(self, frame: FrameData, body=None, yaws=None, position=None):
        """Several facing directions from the start pose: a quick fit of the root from each, then
        the best start of every distinct facing (more than INIT_FACING_GAP apart, the init_facings
        lowest energies) fitted as a whole; keeps the fit that explains the points best. The quick
        fits' energies hardly tell front from back for a person seen side-on (a synthetic jog was
        tracked facing backwards for the whole take); the whole fits do (point score about 6 mm
        against 18 mm there)."""
        from bodyscan.body.rotations import rotation_z_np
        c = self.config.tracking
        body = self.start_body if body is None else body
        if position is None:
            center = np.median(frame.points_np[:, :2], axis=0)
            away = center - self.sensor[:2]
            center = center + 0.12 * away / max(np.linalg.norm(away), 1e-6)
            position = np.array([center[0], center[1], self.avatar.standing_height])
        yaws = np.linspace(0.0, 2 * np.pi, max(c.init_yaws, 1), endpoint=False) if yaws is None else yaws
        quick = sorted((self.fit_frame(frame, rotation_z_np(yaw), position, body, None, rounds=2, iterations=10,
                                       free_body=False) for yaw in yaws), key=lambda result: result[3])
        facings, starts = [], []
        for result in quick:
            facing = np.arctan2(result[0][1, 0], result[0][0, 0])
            if all(abs(np.angle(np.exp(1j * (facing - other)))) > INIT_FACING_GAP for other in facings):
                facings.append(facing)
                starts.append(result)
        best, best_score = None, np.inf
        for rotation, start_position, start_body, _, _ in starts[:max(c.init_facings, 1)]:
            fitted = self.fit_frame(frame, rotation, start_position, start_body, None)
            score = self.explain(fitted[0], fitted[1], fitted[2], frame)["score"]
            if score < best_score:
                best, best_score = fitted, score
        return best

    def track(self, start: int = 0, stop: int | None = None) -> dict:
        """Stage 1 over the segmented frames; returns arrays per frame."""
        from bodyscan.body.rotations import matrix_to_axis_angle_np
        c = self.config.tracking
        count = len(self.segmented)
        stop = count if stop is None else min(stop, count)
        rotations, positions, bodies, times, indices, energies, residuals = [], [], [], [], [], [], []
        slots, restarts, observed, scores = [], [], [], []
        searched = refused = turned = connected = facing_refused = 0
        poor_run, last_check, since = 0, -10 ** 9, 0
        prediction = None
        progress = Progress("tracked frames", stop - start)
        for k in range(start, stop):
            person = self.segmented.load(k)
            frame = FrameData(person, self.geometry, self.device, self.dtype)
            restarted = False
            if not rotations:
                result = self.initialise(frame)
                restarted = True
                explained = self.explain(result[0], result[1], result[2], frame)
            else:
                # after a restart the last two poses do not give a velocity; after a gap (frames
                # dropped) neither: a limb in motion before it would be thrown on along its velocity
                gap = c.gap_frames > 0 and frame.time - times[-1] > c.gap_frames * self.frame_period
                last = slice(-1, None) if restarts[-1] or gap else slice(None)
                prediction = self.predict(rotations[last], positions[last], bodies[last], times[last], frame.time)
                result = self.fit_frame(frame, prediction[0], prediction[1], prediction[2], prediction)
                explained = self.explain(result[0], result[1], result[2], frame)
                plain = (result, explained)                # the fit from the prediction, kept if a search turns
                residual = float(np.median(explained["distance"]))
                if residual > c.lost_residual:
                    yaw = np.arctan2(rotations[-1][1, 0], rotations[-1][0, 0])
                    retry = self.initialise(frame, bodies[-1], yaw + np.radians([0, -45, 45, -90, 90, 180]),
                                            positions[-1])
                    retried = self.explain(retry[0], retry[1], retry[2], frame)
                    # compared on the points: the retry's energy has no continuity terms
                    if np.median(retried["distance"]) < residual:
                        result, restarted, explained = retry, True, retried
            misfit = 0 < c.recover_fraction < explained["unexplained"] and explained["unexplained_count"] >= 10
            in_free_space = 0 < c.recover_free_space < explained["free"]
            if misfit or in_free_space:
                near = self.limbs_to_search(result[0], result[1], result[2], frame, explained)
                limbs = (near - {"trunk"}) or None                     # nothing pinned down: every limb
                trunk = in_free_space or "trunk" in near or explained["unexplained"] > 3 * c.recover_fraction
                result = self.recover_limbs(frame, result, None if restarted else prediction,
                                           None if restarted else (rotations[-1], positions[-1], bodies[-1]),
                                           limbs=limbs, trunk=trunk)
                explained = self.explain(result[0], result[1], result[2], frame)
                searched += 1
            if rotations and not self.plausible_turn(result[0], frame.time, rotations[since:], times[since:]):
                # people do not turn around between two frames: a search that did is wrong
                result, explained = plain
                restarted = False
                refused += 1
            poorly = ((explained["unexplained"] > 2 * c.recover_fraction and explained["unexplained_count"] >= 10)
                      or explained["free"] > 2 * c.recover_free_space)
            poor_run = poor_run + 1 if poorly else 0
            if c.facing_check and rotations and poor_run >= 5 and k - last_check >= 10:
                last_check = k
                retry, retried = self.facing_retry(frame, result)
                if (retried["score"] < explained["score"] - 0.003
                        and not self.runs_backwards(retry[0], positions, times, frame.time)):
                    if self.plausible_turn(retry[0], frame.time, rotations[since:], times[since:]):
                        result, explained = retry, retried
                        turned += 1
                        poor_run = 0
                    else:
                        # the body cannot turn so far since the frames before: either they were tracked
                        # facing wrongly (then, fitted again backwards from the new facing, they explain their
                        # points as well and the turn spreads over them) or the new facing is wrong
                        refits = self.connect_turn(retry, frame.time, rotations, positions, bodies, times, slots,
                                                   scores, since)
                        if refits is None:
                            facing_refused += 1
                        else:
                            for i, fitted, state in refits:
                                rotations[i], positions[i], bodies[i], energies[i] = fitted[0], fitted[1], fitted[2], fitted[3]
                                residuals[i] = float(np.median(state["distance"]))
                                observed[i] = state["observed_parts"]
                                scores[i] = state["score"]
                            result, explained = retry, retried
                            turned += 1
                            connected += len(refits)
                            poor_run = 0
            if restarted:
                since = len(rotations)          # the turn limit counts from the frames after a new start
            rotation, position, body, energy, _ = result
            rotations.append(rotation)
            positions.append(position)
            bodies.append(body)
            times.append(frame.time)
            indices.append(person.index)
            slots.append(k)
            restarts.append(restarted)
            energies.append(energy)
            residuals.append(float(np.median(explained["distance"])))
            observed.append(explained["observed_parts"])
            scores.append(explained["score"])
            progress.maybe(k - start + 1, 25)
        root = np.array([matrix_to_axis_angle_np(r) for r in rotations])
        if searched:
            info(f"  limbs searched again in {searched} frames (points left unexplained by the fit)")
        if refused or turned or facing_refused:
            info(f"  turns of the whole body refused in {refused} frames (faster than {c.max_turn_rate:g} deg/s); "
                 f"facing changed after a poor fit in {turned} frames ({connected} frames before fitted again to "
                 f"spread the turn), refused in {facing_refused} (no physical turn there)")
        return {"times": np.array(times), "frame_index": np.array(indices), "slot": np.array(slots, dtype=np.int64),
                "restart": np.array(restarts, dtype=bool), "searches": searched, "turns_refused": refused,
                "facing_changes": turned, "facing_refused": facing_refused, "turn_frames_refitted": connected,
                "root_rotation": root,
                "root_matrix": np.array(rotations), "root_position": np.array(positions), "body": np.array(bodies),
                "energy": np.array(energies), "residual": np.array(residuals),
                "observed": np.array(observed).reshape(len(times), len(skeleton.PART_NAMES))}

    @staticmethod
    def yaw(rotation) -> float:
        """Facing direction [rad] of a root rotation (the forward axis +x of the upright frame)."""
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def plausible_turn(self, rotation, time: float, rotations, times) -> bool:
        """Whether the facing of 'rotation' at 'time' is within max_turn_rate (plus 40 deg) of the facing
        of every frame tracked in the second before."""
        c = self.config.tracking
        if c.max_turn_rate <= 0:
            return True
        yaw, rate, slack = self.yaw(rotation), np.radians(c.max_turn_rate), np.radians(40.0)
        for j in range(len(times) - 1, -1, -1):
            elapsed = time - times[j]
            if elapsed > 1.0:
                break
            if abs(np.angle(np.exp(1j * (yaw - self.yaw(rotations[j]))))) > rate * max(elapsed, 0.0) + slack:
                return False
        return True

    def runs_backwards(self, rotation, positions, times, time: float) -> bool:
        """Whether a body facing as 'rotation' at 'time' would be moving backwards: its pelvis moved over
        the last half second faster than 0.6 m/s, in a direction more than 120 deg from that facing
        (people walk and run forwards; a person seen side-on and tracked facing backwards does not)."""
        recent = [i for i in range(max(len(times) - 20, 0), len(times)) if 0.0 < time - times[i] <= 0.5]
        if len(recent) < 3:
            return False
        first, last = recent[0], recent[-1]
        elapsed = times[last] - times[first]
        if elapsed <= 0:
            return False
        velocity = (np.asarray(positions[last][:2]) - np.asarray(positions[first][:2])) / elapsed
        speed = float(np.linalg.norm(velocity))
        if speed < 0.6:
            return False
        facing = np.asarray(rotation)[:2, 0]
        facing = facing / max(float(np.linalg.norm(facing)), 1e-9)
        return float(facing @ velocity) / speed < np.cos(np.radians(120.0))

    def connect_turn(self, candidate, time: float, rotations, positions, bodies, times, slots, scores, since: int):
        """The frames before a facing the body cannot reach from them, fitted again backwards from it
        (each from the one after it, with the continuity of stage 1 run backwards), as long as each
        explains its points about as well as before (connect_tolerance), turns physically from the one
        after it (max_turn_rate plus 20 deg) and has not yet met the earlier track (within 15 deg of its
        facing): [(index, fit, explain()), ...] when it meets the earlier track within connect_frames,
        None otherwise (the new facing is then refused)."""
        c = self.config.tracking
        if c.connect_frames <= 0:
            return None
        rate, slack, meet = np.radians(c.max_turn_rate), np.radians(20.0), np.radians(15.0)
        chain = [(candidate[0], candidate[1], candidate[2], time)]          # the latest first
        refits = []
        last = max(len(rotations) - c.connect_frames, since, 0)
        for i in range(len(rotations) - 1, last - 1, -1):
            data = FrameData(self.segmented.load(int(slots[i])), self.geometry, self.device, self.dtype)
            later = chain[-2:]                                               # in the order of time, reversed
            prediction = self.predict([s[0] for s in later], [s[1] for s in later], [s[2] for s in later],
                                      [s[3] for s in later], times[i])
            fitted = self.fit_frame(data, prediction[0], prediction[1], prediction[2], prediction)
            state = self.explain(fitted[0], fitted[1], fitted[2], data)
            if state["score"] > scores[i] + c.connect_tolerance:
                return None
            step = abs(np.angle(np.exp(1j * (self.yaw(fitted[0]) - self.yaw(chain[-1][0])))))
            if step > rate * max(chain[-1][3] - times[i], 0.0) + slack:
                return None
            refits.append((i, fitted, state))
            if abs(np.angle(np.exp(1j * (self.yaw(fitted[0]) - self.yaw(rotations[i]))))) < meet:
                return refits
            chain.append((fitted[0], fitted[1], fitted[2], times[i]))
        return None

    def facing_retry(self, frame: FrameData, result):
        """The frame fitted afresh from the opposite facing and the two side-on ones (the start pose,
        initialise), the limbs searched where its points stay unexplained: (result, explain())."""
        c = self.config.tracking
        yaw = self.yaw(result[0])
        retry = self.initialise(frame, None, yaw + np.radians([180.0, 90.0, -90.0]), result[1])
        retried = self.explain(retry[0], retry[1], retry[2], frame)
        if retried["unexplained"] > c.recover_fraction or retried["free"] > c.recover_free_space:
            near = self.limbs_to_search(retry[0], retry[1], retry[2], frame, retried)
            retry = self.recover_limbs(frame, retry, None, None, limbs=(near - {"trunk"}) or None, trunk=True)
            retried = self.explain(retry[0], retry[1], retry[2], frame)
        return retry, retried

    @staticmethod
    def predict(rotations, positions, bodies, times, t):
        """Constant-velocity prediction of the next frame (rotation, position, body)."""
        from bodyscan.body.rotations import axis_angle_to_matrix_np, matrix_to_axis_angle_np
        if len(rotations) < 2 or times[-1] <= times[-2]:
            return rotations[-1], positions[-1], bodies[-1]
        ratio = np.clip((t - times[-1]) / (times[-1] - times[-2]), 0.0, 3.0)
        delta = matrix_to_axis_angle_np(rotations[-1] @ rotations[-2].T)
        rotation = axis_angle_to_matrix_np(delta * ratio) @ rotations[-1]
        position = positions[-1] + (positions[-1] - positions[-2]) * ratio
        body = bodies[-1] + 0.5 * (bodies[-1] - bodies[-2]) * ratio
        return rotation, position, body

    # -- left-right coupling of hidden limbs (bodyscan.dynamic.coupling) ---------------------------
    def joints_of(self, result: dict, batch: int = 64) -> np.ndarray:
        """(F, 55, 3) world joints of a result."""
        import torch
        frames = len(result["times"])
        out = np.zeros((frames, skeleton.NUM_JOINTS, 3))
        for first in range(0, frames, batch):
            span = slice(first, min(first + batch, frames))

            def tensor(values):
                return torch.as_tensor(values, dtype=self.dtype, device=self.device)

            with torch.no_grad():
                zero = torch.zeros(span.stop - span.start, 3, dtype=self.dtype, device=self.device)
                _, joints = self.pose(zero, tensor(result["root_matrix"][span]), tensor(result["root_position"][span]),
                                      tensor(result["body"][span]))
            out[span] = joints.cpu().numpy()
        return out

    def plan_coupling(self, result: dict) -> dict | None:
        """The kind of motion around every frame (alternating, together, unknown) and its cycle,
        from the observed limbs and the pelvis of the stage-1 result; None when switched off."""
        k = self.config.coupling
        if not k.enabled or len(result["times"]) < 8 or "observed" not in result:
            return None
        joints = self.joints_of(result)
        signals = coupling.signals(joints, result["root_matrix"], result["body"])
        observation = coupling.limb_observation(result["observed"])
        plan = coupling.estimate(result["times"], signals, observation, k, step=self.frame_period)
        limbs = coupling.LIMBS if k.couple_legs else ("left_arm", "right_arm")
        return {"plan": plan, "observation": observation, "limbs": limbs}

    def start_hidden_limbs(self, result: dict, prepared: dict) -> int:
        """With coupling.start_from_guess, the hidden limbs start stage 2 from their guess (as
        complete_hidden_limbs places them): a far arm left where stage 1 put it (still, in the trunk, out to
        the side) disturbs the refinement of the rest of the body (synthetic side-on jumps: velocity error of
        the near hand 0.24 m/s with the guessed start, 0.34 without; probably through the joint smoothness,
        which ties the far arm to the trunk). Returns the number of limb-frames changed."""
        if not self.config.coupling.start_from_guess:
            return 0
        changed = self.guess_hidden_limbs(result, prepared)
        result.pop("guessed", None)
        result.pop("guessed_limbs", None)
        return int(sum(changed.values()))

    def complete_hidden_limbs(self, result: dict, prepared: dict) -> dict:
        """After stage 2, every limb the sensor does not see takes its partner's refined motion, mirrored
        relative to the trunk (coupling.joint_guesses: at the same time or half a cycle away), turned by the
        rotation from it observed where the limb did show: between two observations up to 3 s apart that
        rotation turns from the one to the other (coupling.carried_offsets), so the limb meets both and
        moves like its partner in between; before the first and after the last, and in longer gaps, it fades
        over offset_memory. Where the limb was observed (OBSERVED_LIMB of it seen with points) it keeps its
        tracked pose; where the plan is unsure (weight under 0.5) the guess is blended in partly (spherical
        interpolation). (The guess is applied after stage 2, not as a term of it: with no points on the
        limb, the energy of a window is dominated by the data of the rest of the body and the optimiser
        hardly moves the limb.) The joints are then kept anatomical and out of the trunk (finish_pose).
        Returns {limb: frames changed}."""
        return self.guess_hidden_limbs(result, prepared)

    def guess_hidden_limbs(self, result: dict, prepared: dict) -> dict:
        """The hidden limbs of 'result' replaced by their guess (complete_hidden_limbs), in place. Returns
        {limb: frames changed}."""
        from bodyscan.body.rotations import (axis_angle_to_quaternion_np, quaternion_conjugate_np,
                                             quaternion_multiply_np, quaternion_to_axis_angle_np, slerp_np)
        k = self.config.coupling
        times = result["times"]
        tracked = result["body"]
        body = tracked.copy()
        plan, observation = prepared["plan"], prepared["observation"]
        guesses = coupling.joint_guesses(times, tracked, plan, observation, prepared["limbs"])
        changed = {}
        result["guessed_limbs"] = list(prepared["limbs"])
        result["guessed"] = np.zeros((len(times), len(prepared["limbs"])))
        for number, limb in enumerate(prepared["limbs"]):
            guess = guesses[limb]
            index = coupling.body_indices(limb)
            seen = observation[limb]
            # where the limb was observed (OBSERVED_LIMB of its full view seen with points: the points place
            # it) it keeps its tracked pose: the guess turned by the rotation observed there; between
            # observations that rotation turns from one to the next, so the limb meets them on both sides and
            # moves like its partner in between. The blend follows the weight of the guess, which changes
            # smoothly (coupling.estimate, joint_guesses); at an anchored frame the guess turned by its offset
            # is the tracked pose, so the limb stays continuous where the anchors start and end.
            anchored = (seen >= OBSERVED_LIMB[limb.split("_")[1]]).astype(np.float64)
            blend = np.clip(2.0 * guess.weight, 0.0, 1.0) * (1.0 - anchored)
            result["guessed"][:, number] = blend * (1.0 - seen)
            if not np.any(blend > 0.05):
                continue
            target = axis_angle_to_quaternion_np(guess.rotations)                 # (F, J, 4)
            observed_pose = axis_angle_to_quaternion_np(tracked[:, index])
            offsets = quaternion_multiply_np(quaternion_conjugate_np(target), observed_pose)
            carried = coupling.carried_offsets(times, offsets, anchored, k.offset_memory)
            guessed = quaternion_multiply_np(target, carried)
            final = slerp_np(observed_pose, guessed, np.broadcast_to(blend[:, None], guessed.shape[:-1]))
            body[:, index] = quaternion_to_axis_angle_np(final)
            changed[limb] = int(np.sum(blend > 0.05))
        result["body"] = body
        return changed

    # -- the last step: a human body -------------------------------------------------------------------
    def finish_pose(self, result: dict) -> dict:
        """The pose of every frame made that of a human body, in place, after the refinement and the
        completion of the hidden limbs: a guessed arm moved out of the sensor's sight
        (hide_guessed_limbs), no joint faster than a person's (anatomy.limit_speeds), the elbows and
        knees moved into their range of flexion and out of plane (anatomy.project_hinges: their twist
        kept; the penalty of stage 1 and 2 keeps them close, the completion may not), then the arms
        pushed out of the trunk (push_arms_out). Returns the numbers of frames changed: {"hidden": limb
        frames, "speeds": ..., "hinges": ..., "arms": ...}."""
        from bodyscan.body.anatomy import hinge_excess, limit_speeds, project_hinges
        hidden = self.hide_guessed_limbs(result) if self.config.coupling.hide_guessed_limbs else 0
        body, fast = limit_speeds(result["times"], result["body"], self.config.tracking.joint_speed_scale)
        excess = hinge_excess(body)
        result["body"] = project_hinges(body)
        pushed = self.push_arms_out(result)
        return {"hidden": hidden, "speeds": int(fast.any(axis=1).sum()), "hinges": int(np.sum(excess > 0.5)),
                "arms": pushed}

    def _sight_energy(self, vertices, frame: FrameData, match: FrameMatch, members: np.ndarray):
        """The free-space and in-front terms of frame_energy for the vertices 'members' of one limb only,
        per vertex of the limb (vertices (V, 3) torch; match from match(..., hidden=True))."""
        import torch
        from bodyscan.dynamic.fitting import sample_map
        c = self.config.tracking
        g = self.geometry
        count = max(len(members), 1)
        energy = vertices.new_zeros(())
        free = np.isin(match.silhouette.cpu().numpy(), members)
        front = np.isin(match.front.cpu().numpy(), members)
        if free.any():
            keep = torch.as_tensor(free, device=self.device)
            index = match.silhouette[keep]
            pixels = (match.silhouette_pixels[0][keep], match.silhouette_pixels[1][keep])
            rows, cols, ranges = g.project_torch(vertices[index], self.tables, frame.reference_column, pixels)
            r0, c0 = frame.crop_origin
            h, w = frame.mask.shape
            rr, cc = rows - r0, cols - c0
            out_r = torch.relu(-rr) + torch.relu(rr - (h - 1))
            out_c = torch.relu(-cc) + torch.relu(cc - (w - 1))
            angle = sample_map(frame.distance, rr, cc) + torch.sqrt((out_r * g.vertical_step) ** 2
                                                                    + (out_c * g.column_step) ** 2 + 1e-12)
            scale = c.silhouette_scale
            energy = energy + free_space_loss(angle * ranges, scale, c.free_space_loss).sum() / scale ** 2 / count
        if front.any():
            keep = torch.as_tensor(front, device=self.device)
            index = match.front[keep]
            pixels = (match.front_pixels[0][keep], match.front_pixels[1][keep])
            _, _, ranges = g.project_torch(vertices[index], self.tables, frame.reference_column, pixels)
            excess = torch.relu(match.front_range[keep] - ranges - c.free_space_margin)
            energy = energy + (free_space_loss(excess, c.robust_scale, c.free_space_loss).sum()
                               / c.robust_scale ** 2 / count)
        return energy

    def hide_guessed_limbs(self, result: dict, rounds: int = 2, batch: int = 16) -> int:
        """Guessed arms out of the sensor's sight, in place. The guess of a hidden arm (its partner,
        mirrored) can stand where the sensor saw something else: empty space beyond it, or a surface
        behind it; there the arm cannot be. In every frame where more than HIDE_FROM of the arm is
        guessed, its shoulder and elbow are turned by the least that takes it out of the sensor's sight
        (a short optimisation of the frame: the free-space and in-front terms of the arm's vertices
        against the change of the two joints, with the joint limits, the hinges and the trunk kept); the
        changes are smoothed over HIDE_SMOOTHING, so that the arm does not shake from one frame to the
        next, and the whole is done 'rounds' times (the second round takes what the smoothing left).
        Version 1.5 turned the guess about the vertical for this, which pulled the far arm out to the
        side. Legs are left as guessed: moving a hip or a knee would move a foot standing on the floor.
        Returns the number of arm-frames changed."""
        import torch
        from scipy.ndimage import gaussian_filter1d
        from bodyscan.body.anatomy import hinge_penalty, project_hinges
        from bodyscan.body.model import vertex_normals
        from bodyscan.dynamic.fitting import limit_penalty
        limbs = list(result.get("guessed_limbs", []))
        guessed = result.get("guessed")
        if not limbs or guessed is None:
            return 0
        c = self.config.tracking
        dev, dt = self.device, self.dtype
        sigma = max(HIDE_SMOOTHING / self.frame_period, 0.5)
        plan = []
        for number, limb in enumerate(limbs):
            side, kind = limb.split("_")
            frames = np.flatnonzero(guessed[:, number] > HIDE_FROM)
            if kind != "arm" or len(frames) == 0:
                continue
            joints = [skeleton.JOINT[f"{side}_{name}"] - 1 for name in ("shoulder", "elbow")]
            parts = [skeleton.PART_NAMES.index(part) for part in coupling.LIMB_PARTS[limb]]
            plan.append((number, joints, np.flatnonzero(np.isin(self.part_labels, parts)), frames))
        if not plan:
            return 0
        changed = np.zeros((len(result["times"]), len(limbs)), dtype=bool)
        for _ in range(max(rounds, 1)):
            body = result["body"]
            change = np.zeros_like(body)
            for number, joints, members, frames in plan:
                for first in range(0, len(frames), batch):
                    chunk = frames[first:first + batch]
                    data = [FrameData(self.segmented.load(int(result["slot"][i])), self.geometry, dev, dt)
                            for i in chunk]
                    zero = torch.zeros(len(chunk), 3, dtype=dt, device=dev)
                    root = torch.as_tensor(result["root_matrix"][chunk], dtype=dt, device=dev)
                    position = torch.as_tensor(result["root_position"][chunk], dtype=dt, device=dev)
                    start = torch.as_tensor(body[chunk], dtype=dt, device=dev)
                    with torch.no_grad():
                        vertices, _ = self.pose(zero, root, position, start)
                        normals = vertex_normals(vertices, self.faces)
                        trunk_normals = normals[:, self.collision.trunk]
                    vertices_np = vertices.cpu().numpy().astype(np.float64)
                    normals_np = normals.cpu().numpy().astype(np.float64)
                    matches = [self.match(vertices_np[k], normals_np[k], data[k], hidden=True) for k in range(len(chunk))]
                    with torch.no_grad():
                        active = [k for k in range(len(chunk))
                                  if float(self._sight_energy(vertices[k], data[k], matches[k], members)) > 1e-6]
                    if not active:
                        continue
                    delta = torch.zeros(len(chunk), len(joints), 3, dtype=dt, device=dev, requires_grad=True)
                    optimizer = torch.optim.LBFGS([delta], lr=1.0, max_iter=25, history_size=10,
                                                  line_search_fn="strong_wolfe")

                    def closure():
                        optimizer.zero_grad()
                        moved = start.clone()
                        moved[:, joints] = start[:, joints] + delta
                        posed, _ = self.pose(zero, root, position, moved)
                        energy = sum(self._sight_energy(posed[k], data[k], matches[k], members) for k in active)
                        energy = energy + 0.5 * (delta ** 2).sum()
                        energy = energy + c.limit_weight * limit_penalty(moved) + c.hinge_weight * hinge_penalty(moved)
                        if c.penetration_weight > 0:
                            energy = energy + c.penetration_weight * self.collision.energy(posed, trunk_normals,
                                                                                           c.penetration_margin)
                        energy.backward()
                        return energy

                    optimizer.step(closure)
                    found = delta.detach().cpu().numpy()
                    for k in active:
                        change[chunk[k], joints] = found[k]
            # smoothed over HIDE_SMOOTHING, and faded out where the arm is guessed less
            weight = np.zeros(body.shape[:2])
            for number, joints, members, frames in plan:
                weight[:, joints] = np.clip(guessed[:, number] / HIDE_FROM, 0.0, 1.0)[:, None]
            change = gaussian_filter1d(change, sigma, axis=0, mode="nearest") * weight[..., None]
            if np.abs(change).max() < 1e-6:
                break
            result["body"] = project_hinges(body + change)
            for number, joints, members, frames in plan:
                changed[:, number] |= np.abs(change[:, joints]).max(axis=(1, 2)) > np.radians(0.5)
        return int(changed.sum())

    def arm_depths(self, result: dict, batch: int = 32) -> np.ndarray:
        """(F, 2) the deepest sample of the left and the right arm inside the trunk [m] (0 when none)."""
        import torch
        from bodyscan.body.model import vertex_normals
        frames = len(result["times"])
        depth = np.zeros((frames, 2))
        for first in range(0, frames, batch):
            span = np.arange(first, min(first + batch, frames))
            with torch.no_grad():
                zero = torch.zeros(len(span), 3, dtype=self.dtype, device=self.device)
                vertices, _ = self.pose(zero, torch.as_tensor(result["root_matrix"][span], dtype=self.dtype,
                                                              device=self.device),
                                        torch.as_tensor(result["root_position"][span], dtype=self.dtype,
                                                        device=self.device),
                                        torch.as_tensor(result["body"][span], dtype=self.dtype, device=self.device))
                normals = vertex_normals(vertices, self.faces)
            v = vertices.cpu().numpy().astype(np.float64)
            n = normals.cpu().numpy().astype(np.float64)
            for i, k in enumerate(span):
                trunk, trunk_normals = v[i, self.collision.trunk], n[i, self.collision.trunk]
                for side, name in enumerate(("left", "right")):
                    depth[k, side] = TrunkCollision.depth_of(v[i, self.collision.arm_of[name]], trunk, trunk_normals)
        return depth

    def push_arms_out(self, result: dict, rounds: int = 2) -> int:
        """Arms out of the trunk, in place: in every frame where an arm reaches deeper into the trunk than
        penetration_margin (plus 2 mm), its shoulder and elbow are turned by the least that brings it out
        (a short optimisation of the frame: the excess depth against the change of the two joints, the
        joint limits kept; the trunk does not move), then the changes are spread over the neighbouring
        frames (each frame keeps the larger of its own change and the smoothed one), so that the arm does
        not jump; a second round takes what is left (an arm 9 cm inside came out to 1 to 2 cm in one
        round). Returns the number of frames changed."""
        changed = np.zeros(len(result["times"]), dtype=bool)
        for _ in range(max(rounds, 1)):
            moved = self._push_arms_out_once(result)
            if not moved.any():
                break
            changed |= moved
        return int(changed.sum())

    def _push_arms_out_once(self, result: dict, batch: int = 24) -> np.ndarray:
        """One round of push_arms_out: (F,) the frames changed."""
        import torch
        from bodyscan.body.anatomy import hinge_penalty, project_hinges
        from bodyscan.body.model import vertex_normals
        from bodyscan.dynamic.fitting import limit_penalty
        from scipy.ndimage import gaussian_filter1d
        c = self.config.tracking
        frames = len(result["times"])
        if c.penetration_weight <= 0:
            return np.zeros(frames, dtype=bool)
        depth = self.arm_depths(result)
        bad = np.flatnonzero(depth.max(axis=1) > c.penetration_margin + 0.002)
        if len(bad) == 0:
            return np.zeros(frames, dtype=bool)
        dev, dt = self.device, self.dtype
        body = result["body"]
        joints = [skeleton.JOINT[f"{side}_{name}"] - 1 for side in ("left", "right") for name in ("shoulder", "elbow")]
        change = np.zeros((len(body), len(joints), 3))
        arm_samples = self.collision.arm
        for first in range(0, len(bad), batch):
            chunk = bad[first:first + batch]
            zero = torch.zeros(len(chunk), 3, dtype=dt, device=dev)
            root = torch.as_tensor(result["root_matrix"][chunk], dtype=dt, device=dev)
            position = torch.as_tensor(result["root_position"][chunk], dtype=dt, device=dev)
            start = torch.as_tensor(body[chunk], dtype=dt, device=dev)
            with torch.no_grad():
                vertices, _ = self.pose(zero, root, position, start)
                normals = vertex_normals(vertices, self.faces)
                trunk = vertices[:, self.collision.trunk]
                trunk_normals = normals[:, self.collision.trunk]
            delta = torch.zeros(len(chunk), len(joints), 3, dtype=dt, device=dev, requires_grad=True)
            optimizer = torch.optim.LBFGS([delta], lr=1.0, max_iter=40, history_size=10, line_search_fn="strong_wolfe")

            def total():
                moved = start.clone()
                moved[:, joints] = start[:, joints] + delta
                arm, _ = self.pose(zero, root, position, moved, subset=arm_samples)
                energy = self.collision.energy_of(arm, trunk, trunk_normals, c.penetration_margin)
                energy = energy + 2.0 * (delta ** 2).sum()
                energy = energy + c.limit_weight * limit_penalty(moved) + c.hinge_weight * hinge_penalty(moved)
                return energy

            def closure():
                optimizer.zero_grad()
                energy = total()
                energy.backward()
                return energy

            optimizer.step(closure)
            change[chunk] = delta.detach().cpu().numpy()
        # spread over the neighbouring frames: each frame keeps the larger of its own change and the smoothed one
        smooth = gaussian_filter1d(change, 1.5, axis=0, mode="nearest")
        own, spread = np.linalg.norm(change, axis=2), np.linalg.norm(smooth, axis=2)
        change = np.where((own >= spread)[..., None], change, smooth)
        moved = body.copy()
        moved[:, joints] = body[:, joints] + change
        result["body"] = project_hinges(moved)
        return np.linalg.norm(change, axis=(1, 2)) > 1e-3

    # -- stage 2 ----------------------------------------------------------------------------------
    def contacts(self, result: dict) -> np.ndarray:
        """(F, 2) feet on the floor: lowest foot vertex low and ankle slow."""
        import torch
        r = self.config.refine
        frames = len(result["times"])
        if frames == 0:
            return np.zeros((0, 2), dtype=bool)
        with torch.no_grad():
            zero = torch.zeros(frames, 3, dtype=self.dtype, device=self.device)
            vertices, joints = self.pose(zero, torch.as_tensor(result["root_matrix"], dtype=self.dtype,
                                                               device=self.device),
                                         torch.as_tensor(result["root_position"], dtype=self.dtype, device=self.device),
                                         torch.as_tensor(result["body"], dtype=self.dtype, device=self.device),
                                         subset=self.feet)
        vertices = vertices.cpu().numpy()
        joints = joints.cpu().numpy()
        labels = self.part_labels[self.feet_np]
        times = result["times"]
        contact = np.zeros((frames, 2), dtype=bool)
        for side, name in enumerate(("left", "right")):
            mine = labels == skeleton.PART_NAMES.index(f"{name}_foot")
            low = vertices[:, mine, 2].min(axis=1)
            ankle = joints[:, skeleton.JOINT[f"{name}_ankle"], :2]
            speed = np.zeros(frames)
            if frames > 1:
                step = np.linalg.norm(np.diff(ankle, axis=0), axis=1) / np.maximum(np.diff(times), 1e-6)
                speed[1:] = step
                speed[:-1] = np.minimum(speed[:-1], step) if frames > 2 else step
                speed[0] = step[0]
            contact[:, side] = (low < r.contact_height) & (speed < r.contact_speed)
        # remove single-frame flickers
        for side in range(2):
            flags = contact[:, side].copy()
            for k in range(1, frames - 1):
                if flags[k - 1] == flags[k + 1] != flags[k]:
                    contact[k, side] = flags[k - 1]
        return contact

    def refine(self, result: dict, contacts: np.ndarray) -> dict:
        """Stage 2: overlapping windows over the whole sequence."""
        import torch
        from bodyscan.body.model import vertex_normals
        from bodyscan.body.rotations import axis_angle_to_matrix, matrix_to_axis_angle_np
        from bodyscan.body.anatomy import hinge_penalty
        from bodyscan.dynamic.fitting import charbonnier, geman_mcclure, limit_penalty, surface_points
        r = self.config.refine
        c = self.config.tracking
        dev, dt = self.device, self.dtype
        frames = len(result["times"])
        times = result["times"]
        reference = result["root_matrix"].copy()
        position = result["root_position"].copy()
        body = result["body"].copy()
        data = {}
        main = list(skeleton.MAIN_JOINTS)
        feet_joints = {0: [skeleton.JOINT["left_ankle"], skeleton.JOINT["left_foot"]],
                       1: [skeleton.JOINT["right_ankle"], skeleton.JOINT["right_foot"]]}
        step = max(r.window - r.overlap, 1)
        starts = list(range(0, max(frames - r.overlap, 1), step))
        progress = Progress("refined windows", len(starts))
        for window_number, first in enumerate(starts):
            # the look-ahead frames are optimised with the window and refined again with the next one
            last = min(first + r.window + max(r.lookahead, 0), frames)
            held = r.overlap if first > 0 else 0
            span = np.arange(first, last)
            count = len(span)
            if count < 3:
                continue
            for k in span:
                if k not in data:
                    data[k] = FrameData(self.segmented.load(int(result["slot"][k])), self.geometry, dev, dt)
            ref_t = torch.as_tensor(reference[span], dtype=dt, device=dev)
            omega = torch.zeros(count, 3, dtype=dt, device=dev)
            pos = torch.as_tensor(position[span], dtype=dt, device=dev)
            b = torch.as_tensor(body[span], dtype=dt, device=dev)
            omega_f = omega[held:].clone().requires_grad_(True)
            pos_f = pos[held:].clone().requires_grad_(True)
            b_f = b[held:].clone().requires_grad_(True)
            # relative times: sensor clocks run to 1e9 s, beyond the resolution of float32 (GPU)
            t_span = torch.as_tensor(times[span] - times[span[0]], dtype=dt, device=dev)
            window_contacts = contacts[span]

            def assemble():
                return (torch.cat([omega[:held], omega_f]), torch.cat([pos[:held], pos_f]), torch.cat([b[:held], b_f]))

            for round_number in range(r.rounds):
                # coarse to fine as in stage 1, starting from half the coarse scale (stage 1 is close)
                scale = (c.robust_scale * max(c.coarse_scale / 2.0 ** (round_number + 1), 1.0)
                         if round_number < r.rounds - 1
                         else c.robust_scale)
                with torch.no_grad():
                    o, p, bb = assemble()
                    vertices, _ = self.pose(o, ref_t, p, bb)
                    normals = vertex_normals(vertices, self.faces)
                    trunk_normals = normals[:, self.collision.trunk]
                vertices_np = vertices.cpu().numpy().astype(np.float64)
                matches = [self.match(vertices_np[i], normals[i].cpu().numpy().astype(np.float64), data[k],
                                      max(c.max_correspondence, 3.0 * scale), hidden=r.free_space_hidden)
                           for i, k in enumerate(span)]
                seen = np.stack([self.joint_seen(m.visible) for m in matches])          # (count, 22)
                hidden_weight = 1.0 + r.hidden_smoothness * self.hidden_factor(seen)
                shutter = []
                for i, k in enumerate(span):
                    frame, match = data[k], matches[i]
                    before, after = max(i - 1, 0), min(i + 1, count - 1)
                    span_t = float(times[span[after]] - times[span[before]])
                    if not r.rolling_shutter or span_t <= 0:
                        shutter.append(None)
                        continue

                    def factor(offsets):              # time offsets [s] (float64) as fractions of span_t
                        return torch.as_tensor(np.asarray(offsets, dtype=np.float64) / span_t, dtype=dt, device=dev)

                    point_offsets = frame.point_times[match.point_index.cpu().numpy()] - times[k]
                    shutter.append({"before": before, "after": after, "points": factor(point_offsets),
                                    "silhouette": factor(match.silhouette_offsets),
                                    "front": factor(match.front_offsets)})
                tensors = [omega_f, pos_f, b_f]
                optimizer = torch.optim.LBFGS(tensors, lr=1.0, max_iter=r.iterations, history_size=10,
                                              line_search_fn="strong_wolfe")

                def total():
                    o, p, bb = assemble()
                    vertices, joints = self.pose(o, ref_t, p, bb)
                    energy = vertices.new_zeros(())
                    for i, k in enumerate(span):
                        match = matches[i]
                        moved = {}
                        if shutter[i] is not None:
                            # the body at the time of each point and pixel: moved along its motion
                            # between the neighbouring frames
                            s = shutter[i]
                            step = vertices[s["after"]] - vertices[s["before"]]
                            if len(match.point_index):
                                moved["shifted_points"] = (
                                    surface_points(vertices[i], self.faces, match.face_ids, match.bary)
                                    + s["points"][:, None] * surface_points(step, self.faces, match.face_ids,
                                                                            match.bary))
                            if len(match.silhouette):
                                moved["shifted_silhouette"] = (vertices[i][match.silhouette]
                                                               + s["silhouette"][:, None] * step[match.silhouette])
                            if len(match.front):
                                moved["shifted_front"] = (vertices[i][match.front]
                                                          + s["front"][:, None] * step[match.front])
                        energy = energy + self.frame_energy(vertices[i], data[k], match, scale=scale, **moved)
                    energy = energy / count + c.limit_weight * limit_penalty(bb) / count
                    if c.hinge_weight > 0:
                        energy = energy + c.hinge_weight * hinge_penalty(bb) / count
                    energy = energy + neutral_priors(bb, c) / count
                    dt1 = (t_span[1:] - t_span[:-1]).clamp_min(1e-4)
                    if c.penetration_weight > 0:
                        energy = energy + c.penetration_weight * self.collision.energy(
                            vertices, trunk_normals, c.penetration_margin) / count
                    # smoothness of the joint motion (non-uniform frame times): accelerations at the inner
                    # frames, jerk between them; normalised by the scale (about |x| / scale when large)
                    j = joints[:, main]
                    velocity = (j[1:] - j[:-1]) / dt1[:, None, None]
                    accel = 2.0 * (velocity[1:] - velocity[:-1]) / (t_span[2:] - t_span[:-2]).clamp_min(1e-4)[:, None, None]
                    if r.smoothness == "jerk" and count > 3:
                        rough = (accel[1:] - accel[:-1]) / dt1[1:-1, None, None]
                        weight = 0.5 * (hidden_weight[1:-2] + hidden_weight[2:-1])
                    else:
                        rough = accel
                        weight = hidden_weight[1:-1]
                    weight = torch.as_tensor(weight, dtype=dt, device=dev)
                    energy = energy + r.smoothness_weight * (weight * charbonnier(rough.norm(dim=-1),
                                                                                  r.smoothness_scale)).mean() \
                        / r.smoothness_scale
                    # feet that stand on the floor do not slide
                    for side, ids in feet_joints.items():
                        both = torch.as_tensor(window_contacts[1:, side] & window_contacts[:-1, side], device=dev)
                        if bool(both.any()):
                            slide = (joints[1:, ids, :2] - joints[:-1, ids, :2]).norm(dim=-1) / dt1[:, None]
                            energy = energy + r.skating_weight * geman_mcclure(slide[both], 0.1).mean() / 0.01
                    return energy

                def closure():
                    optimizer.zero_grad()
                    energy = total()
                    energy.backward()
                    return energy

                optimizer.step(closure)
            with torch.no_grad():
                o, p, bb = assemble()
                final = (axis_angle_to_matrix(o) @ ref_t).cpu().numpy().astype(np.float64)
            reference[span] = final
            position[span] = p.detach().cpu().numpy()
            body[span] = bb.detach().cpu().numpy()
            for k in list(data):
                if k < first + step:
                    del data[k]
            progress.maybe(window_number + 1, 1)
        refined = dict(result)
        refined["root_matrix"] = reference
        refined["root_rotation"] = np.array([matrix_to_axis_angle_np(m) for m in reference])
        refined["root_position"] = position
        refined["body"] = body
        return refined

    # -- quality ----------------------------------------------------------------------------------
    def assess(self, result: dict) -> dict:
        """Per frame: point residual, silhouette violations, observed fraction per body part,
        floor penetration, joint-limit excess; and the joints."""
        import torch
        from bodyscan.body.model import vertex_normals
        from bodyscan.dynamic.fitting import closest_points, visible_faces
        frames = len(result["times"])
        parts = len(skeleton.PART_NAMES)
        observed = np.zeros((frames, parts))
        residual_median = np.zeros(frames)
        residual_p90 = np.zeros(frames)
        unexplained = np.zeros(frames)
        clutter = np.zeros(frames)
        violation = np.zeros(frames)
        floor = np.zeros(frames)
        arm_in_trunk = np.zeros(frames)
        joints_out = np.zeros((frames, skeleton.NUM_JOINTS, 3), dtype=np.float32)
        from bodyscan.body.anatomy import hinge_excess
        lower, upper = skeleton.limit_arrays()
        excess = np.degrees(np.maximum(lower[skeleton.BODY] - result["body"], 0)
                            + np.maximum(result["body"] - upper[skeleton.BODY], 0)).max(axis=(1, 2))
        excess = np.maximum(excess, hinge_excess(result["body"]))           # elbows and knees out of their plane
        batch = 32
        for first in range(0, frames, batch):
            span = np.arange(first, min(first + batch, frames))
            with torch.no_grad():
                zero = torch.zeros(len(span), 3, dtype=self.dtype, device=self.device)
                vertices, joints = self.pose(zero, torch.as_tensor(result["root_matrix"][span], dtype=self.dtype,
                                                                   device=self.device),
                                             torch.as_tensor(result["root_position"][span], dtype=self.dtype,
                                                             device=self.device),
                                             torch.as_tensor(result["body"][span], dtype=self.dtype, device=self.device))
                normals = vertex_normals(vertices, self.faces)
            joints_out[span] = joints.cpu().numpy()
            for i, k in enumerate(span):
                frame = FrameData(self.segmented.load(int(result["slot"][k])), self.geometry, self.device, self.dtype)
                v = vertices[i].cpu().numpy().astype(np.float64)
                match = self.match(v, normals[i].cpu().numpy().astype(np.float64), frame)
                _, _, distance = closest_points(v, self.faces_np, frame.points_np,
                                                visible_faces(self.faces_np, match.visible))
                residual_median[k] = np.median(distance)
                residual_p90[k] = np.percentile(distance, 90)
                reach = self.config.tracking.max_correspondence
                unexplained[k] = np.mean((distance > 0.05) & (distance < reach))
                clutter[k] = np.mean(distance >= reach)
                # of the vertices the model shows the sensor (match without 'hidden'): the hidden limb vertices
                # that stage 2 also tests include the back of every limb along the edge of its silhouette, a
                # pixel out by the discretisation (about 1 % of the visible count on the truth of synthetic
                # takes), which would flag most frames
                violation[k] = len(match.silhouette) / match.visible_count
                floor[k] = v[self.feet_np, 2].min()
                arm_in_trunk[k] = self.collision.depth(v, normals[i].cpu().numpy().astype(np.float64))
                # observed: visible vertices with a point of the person within 2 robust scales
                visible = np.flatnonzero(match.visible)
                if len(visible):
                    from bodyscan.dynamic.avatar import _nearest
                    _, near = _nearest(frame.points_np, v[visible])
                    seen = visible[near < 2 * self.config.tracking.robust_scale]
                    observed[k] = np.bincount(self.part_labels[seen], minlength=parts) / np.maximum(self.part_count, 1)
        observed[:, self.part_count == 0] = np.nan                     # parts without vertices in this body
        return {"joints": joints_out, "residual_median": residual_median, "residual_p90": residual_p90,
                "unexplained": unexplained, "clutter": clutter, "silhouette_violation": violation, "floor": floor,
                "limit_excess_deg": excess, "observed": observed, "arm_in_trunk": arm_in_trunk}


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

def flag_frames(quality: dict, contacts: np.ndarray, times: np.ndarray) -> dict:
    """Frames that need a look: points far from the body (most of them, or more
    than 5 % of them within reach of a limb: a limb in the wrong place; points
    beyond max_correspondence are clutter and not counted), body in free space, feet
    below the floor or sliding while standing, joint limits exceeded, sudden
    accelerations."""
    joints = quality["joints"][:, list(skeleton.MAIN_JOINTS)].astype(np.float64)
    frames = len(times)
    accel = np.zeros(frames)
    if frames > 2:
        dt1 = np.maximum(np.diff(times), 1e-4)
        velocity = np.diff(joints, axis=0) / dt1[:, None, None]
        a = 2 * np.diff(velocity, axis=0) / np.maximum(times[2:] - times[:-2], 1e-4)[:, None, None]
        accel[1:-1] = np.linalg.norm(a, axis=2).max(axis=1)
    sliding = np.zeros(frames)
    for side, name in enumerate(("left", "right")):
        ankle = quality["joints"][:, skeleton.JOINT[f"{name}_ankle"], :2]
        if frames > 1:
            speed = np.linalg.norm(np.diff(ankle, axis=0), axis=1) / np.maximum(np.diff(times), 1e-4)
            both = contacts[1:, side] & contacts[:-1, side]
            sliding[1:] = np.maximum(sliding[1:], np.where(both, speed, 0.0))
    checks = {
        "residual": quality["residual_median"] > 0.03,
        "unexplained": quality["unexplained"] > 0.05,
        "free_space": quality["silhouette_violation"] > 0.03,
        "below_floor": quality["floor"] < -0.02,
        "sliding": sliding > 0.3,
        "joint_limits": quality["limit_excess_deg"] > 10.0,
        "acceleration": accel > 120.0,
        "arm_in_trunk": quality.get("arm_in_trunk", np.zeros(frames)) > 0.04,
    }
    flags = np.zeros(frames, dtype=np.int64)
    for bit, (name, mask) in enumerate(checks.items()):
        flags |= mask.astype(np.int64) << bit
    return {"flags": flags, "flag_names": list(checks), "acceleration_max": accel, "sliding": sliding,
            "counts": {name: int(mask.sum()) for name, mask in checks.items()}}


def save_motion(path, result: dict, quality: dict, contacts: np.ndarray, flags: dict, avatar_path, avatar,
                config: TrackPipelineConfig, extra: dict | None = None) -> dict:
    observed = quality["observed"]
    summary = {
        "frames": int(len(result["times"])),
        "duration_s": float(result["times"][-1] - result["times"][0]) if len(result["times"]) > 1 else 0.0,
        "residual_median_mm": float(1000 * np.median(quality["residual_median"])),
        "residual_p90_mm": float(1000 * np.median(quality["residual_p90"])),
        "observed_fraction_per_part": {name: float(np.nanmean(observed[:, k])) for k, name in
                                       enumerate(skeleton.PART_NAMES) if np.isfinite(observed[:, k]).any()},
        "flagged_frames": flags["counts"],
        "clutter_fraction_mean": float(np.mean(quality["clutter"])) if "clutter" in quality else 0.0,
        "contact_fraction": {"left": float(contacts[:, 0].mean()), "right": float(contacts[:, 1].mean())},
        **(extra or {}),
    }
    frames = len(result["times"])
    np.savez_compressed(path, times=result["times"], root_rotation=result["root_rotation"],
                        root_position=result["root_position"], body=result["body"],
                        frame_index=result["frame_index"], joints=quality["joints"], contacts=contacts,
                        observed=observed.astype(np.float32), residual_median=quality["residual_median"],
                        residual_p90=quality["residual_p90"], unexplained=quality["unexplained"],
                        clutter=quality.get("clutter", np.zeros_like(quality["unexplained"])),
                        silhouette_violation=quality["silhouette_violation"],
                        floor=quality["floor"], limit_excess_deg=quality["limit_excess_deg"],
                        arm_in_trunk=quality.get("arm_in_trunk", np.zeros(frames)), flags=flags["flags"],
                        flag_names=np.array(flags["flag_names"]), avatar=str(avatar_path), model=avatar.model_path,
                        model_sha1=avatar.model_sha1, level=np.int64(config.tracking.level),
                        coupling_mode=result.get("coupling_mode", np.zeros(frames, dtype=np.int64)),
                        coupling_confidence=result.get("coupling_confidence", np.zeros(frames)),
                        guessed=result.get("guessed", np.zeros((frames, 0))).astype(np.float32),
                        guessed_limbs=np.array(result.get("guessed_limbs", []), dtype=str),
                        config=json.dumps({"tracking": asdict(config.tracking), "refine": asdict(config.refine),
                                           "coupling": asdict(config.coupling)}),
                        summary=json.dumps(summary))
    Path(str(path).replace(".npz", "") + ".json").write_text(json.dumps(summary, indent=2))
    return summary


def summary_text(summary: dict) -> str:
    lines = [f"tracked {summary['frames']} frames over {summary['duration_s']:.1f} s; points to body: median "
             f"{summary['residual_median_mm']:.1f} mm, p90 {summary['residual_p90_mm']:.1f} mm (median of frames)"]
    observed = summary["observed_fraction_per_part"]
    weakest = sorted(observed.items(), key=lambda item: item[1])[:4]
    lines.append("  least observed parts (fraction of the part seen, mean over frames): "
                 + ", ".join(f"{name} {value:.0%}" for name, value in weakest))
    flagged = {k: v for k, v in summary["flagged_frames"].items() if v}
    lines.append("  frames to review: " + (", ".join(f"{k} {v}" for k, v in flagged.items()) if flagged else "none"))
    return "\n".join(lines)


def run_tracking(segment_folder, avatar_path, out, config: TrackPipelineConfig | None = None, model_path=None,
                 start: int = 0, stop: int | None = None) -> dict:
    """Stage 1, contacts, stage 2, quality; writes <out>.npz and <out>.json."""
    import torch
    from bodyscan.dynamic.avatar import Avatar
    from bodyscan.dynamic.segment import SegmentedRecording
    config = config or TrackPipelineConfig()
    device = config.tracking.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = config.tracking.precision
    if precision == "auto":
        precision = "float32" if device.startswith("cuda") else "float64"
    dtype = torch.float32 if precision == "float32" else torch.float64
    avatar = Avatar.load(avatar_path)
    model = avatar.model(model_path, device, dtype)
    segmented = SegmentedRecording(segment_folder)
    tracker = Tracker(avatar, model, segmented, config)
    first, last = start, len(segmented) if stop is None else min(stop, len(segmented))
    info(f"tracking frames {first} to {last - 1} of {len(segmented)} on {device} ({str(dtype).split('.')[-1]})")
    result = tracker.track(start, stop)
    prepared = None
    if config.refine.enabled and len(result["times"]) >= 3:
        prepared = tracker.plan_coupling(result)
        if prepared is not None:
            changed = tracker.start_hidden_limbs(result, prepared)
            share = prepared["plan"].summary()
            info(f"  hidden limbs coupled to their partner: motion alternating in {share['alternating']:.0%} of the "
                 f"frames, together in {share['together']:.0%}, unknown in {share['unknown']:.0%}"
                 + (f"; {changed} limb-frames start from the partner's motion" if changed else ""))
    contacts = tracker.contacts(result)
    if config.refine.enabled and len(result["times"]) >= 3:
        info("refining the whole sequence")
        result = tracker.refine(result, contacts)
        if prepared is not None:
            completed = tracker.complete_hidden_limbs(result, prepared)
            if completed:
                info("  hidden limbs completed from their partner: "
                     + ", ".join(f"{limb.replace('_', ' ')} {count} frames" for limb, count in completed.items()))
    info("making every pose that of a human body")
    finished = tracker.finish_pose(result)
    info(f"  guessed arms moved out of the sensor's sight in {finished['hidden']} arm-frames, joints slowed to a "
         f"person's speed in {finished['speeds']} frames, elbows and knees put back into their plane in "
         f"{finished['hinges']}, arms pushed out of the trunk in {finished['arms']}")
    contacts = tracker.contacts(result)
    quality = tracker.assess(result)
    flags = flag_frames(quality, contacts, result["times"])
    out = Path(out)
    if out.suffix != ".npz":
        out = out.with_suffix(".npz")
    extra = {"limb_searches": int(result.get("searches", 0)), "restarts": int(np.sum(result.get("restart", []))),
             "turns_refused": int(result.get("turns_refused", 0)), "facing_changes": int(result.get("facing_changes", 0)),
             "facing_refused": int(result.get("facing_refused", 0)),
             "turn_frames_refitted": int(result.get("turn_frames_refitted", 0)),
             "guessed_arms_hidden": finished["hidden"], "joint_speeds_limited": finished["speeds"],
             "hinges_projected": finished["hinges"], "arms_pushed_out": finished["arms"]}
    if prepared is not None:
        extra["coupling"] = prepared["plan"].summary()
        result["coupling_mode"] = prepared["plan"].mode
        result["coupling_confidence"] = prepared["plan"].confidence
    summary = save_motion(out, result, quality, contacts, flags, avatar_path, avatar, config, extra)
    info(summary_text(summary))
    if any(flags["counts"].values()):
        warning("some frames are flagged: look at them with 'bodyscan review-motion'")
    return summary
