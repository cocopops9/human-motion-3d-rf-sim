"""The SMPL-X skeleton: joint names, parents, body parts, anatomical limits.

SMPL-X model space: y up, z forward (the body faces +z), x towards the left
of the body. Every joint rotation is expressed in the frame of its parent;
in the rest pose (T-pose: arms along +x and -x, palms down, legs along -y)
all the joint frames are aligned with the model axes, so the sign of a
rotation about x, y or z has the same meaning for every joint:

    knee bend (heel towards the buttock)          +x
    hip flexion (thigh forward)                   -x
    ankle dorsiflexion (toes up)                  -x
    spine and neck flexion (bend forward)         +x
    left arm down to the side                     -z      right arm down    +z
    left arm forward                              -y      right arm forward +y
    left elbow bend (palm-down T-pose)            -y      right elbow bend  +y
    left hip abduction (thigh outwards)           +z      right             -z

These follow from rotating the rest direction of each bone (thigh -y, foot
+z, left arm +x) by the right-hand rule; the tests check them on a model.
"""

from __future__ import annotations

import numpy as np

JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
    "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw",
    "left_eye", "right_eye",
    "left_index1", "left_index2", "left_index3", "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3", "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
    "right_index1", "right_index2", "right_index3", "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1", "right_ring2", "right_ring3",
    "right_thumb1", "right_thumb2", "right_thumb3")

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15,
           20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
           21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53)

NUM_JOINTS = 55
NUM_BODY_JOINTS = 21                # joints 1..21, the 'body_pose' of SMPL-X (63 numbers)
BODY = slice(1, 22)
JAW, LEFT_EYE, RIGHT_EYE = 22, 23, 24
LEFT_HAND = slice(25, 40)
RIGHT_HAND = slice(40, 55)
JOINT = {name: index for index, name in enumerate(JOINT_NAMES)}

# The 22 joints of the main body (pelvis to wrists): the joints compared by the
# usual pose error (MPJPE) and the joints the LiDAR can actually constrain.
MAIN_JOINTS = tuple(range(22))

FEET = {"left": (JOINT["left_ankle"], JOINT["left_foot"]), "right": (JOINT["right_ankle"], JOINT["right_foot"])}

# Body parts for per-vertex labels (from the joint that moves each vertex most).
PART_OF_JOINT = {
    0: "pelvis", 1: "left_thigh", 2: "right_thigh", 3: "torso_lower", 4: "left_shank", 5: "right_shank",
    6: "torso_middle", 7: "left_foot", 8: "right_foot", 9: "torso_upper", 10: "left_foot", 11: "right_foot",
    12: "neck", 13: "left_shoulder", 14: "right_shoulder", 15: "head", 16: "left_upper_arm",
    17: "right_upper_arm", 18: "left_forearm", 19: "right_forearm", 20: "left_hand", 21: "right_hand",
    22: "head", 23: "head", 24: "head"}
for _j in range(25, 40):
    PART_OF_JOINT[_j] = "left_hand"
for _j in range(40, 55):
    PART_OF_JOINT[_j] = "right_hand"
PART_NAMES = tuple(dict.fromkeys(PART_OF_JOINT[j] for j in range(NUM_JOINTS)))

# Soft anatomical limits of the body joints, per axis-angle component in the
# parent frame [deg]: ((x min, x max), (y min, y max), (z min, z max)). They are
# deliberately wide: their job is to forbid what no person can do (a knee or an
# elbow bent backwards, a foot turned around), not to model the exact range of
# motion. A box on axis-angle components is exact for one-axis rotations and an
# approximation for combined ones.
LIMITS_DEG = {
    1: ((-130, 35), (-50, 50), (-35, 65)),      # left hip: flexion is -x, abduction +z
    2: ((-130, 35), (-50, 50), (-65, 35)),      # right hip: abduction -z
    3: ((-30, 55), (-35, 35), (-30, 30)),       # spine1
    4: ((-6, 155), (-25, 25), (-12, 12)),       # left knee: bends +x only
    5: ((-6, 155), (-25, 25), (-12, 12)),       # right knee
    6: ((-25, 35), (-30, 30), (-25, 25)),       # spine2
    7: ((-35, 55), (-35, 35), (-30, 30)),       # left ankle: dorsiflexion -x
    8: ((-35, 55), (-35, 35), (-30, 30)),       # right ankle
    9: ((-25, 35), (-30, 30), (-25, 25)),       # spine3
    10: ((-60, 30), (-15, 15), (-15, 15)),      # left toes
    11: ((-60, 30), (-15, 15), (-15, 15)),      # right toes
    12: ((-45, 55), (-60, 60), (-45, 45)),      # neck
    13: ((-25, 25), (-35, 35), (-20, 45)),      # left collar: shoulder up is +z
    14: ((-25, 25), (-35, 35), (-45, 20)),      # right collar
    15: ((-45, 45), (-70, 70), (-35, 35)),      # head
    16: ((-110, 110), (-170, 70), (-115, 95)),  # left shoulder: forward -y, down -z
    17: ((-110, 110), (-70, 170), (-95, 115)),  # right shoulder: forward +y, down +z
    18: ((-100, 100), (-160, 6), (-20, 20)),    # left elbow: bends -y only (twist on x)
    19: ((-100, 100), (-6, 160), (-20, 20)),    # right elbow: bends +y only
    20: ((-100, 100), (-80, 80), (-50, 50)),    # left wrist
    21: ((-100, 100), (-80, 80), (-50, 50)),    # right wrist
}


def limit_arrays(dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    """(55, 3) lower and upper limits in radians; joints without limits get +-pi."""
    lower = np.full((NUM_JOINTS, 3), -np.pi, dtype=dtype)
    upper = np.full((NUM_JOINTS, 3), np.pi, dtype=dtype)
    for joint, bounds in LIMITS_DEG.items():
        for axis, (low, high) in enumerate(bounds):
            lower[joint, axis] = np.radians(low)
            upper[joint, axis] = np.radians(high)
    return lower, upper


def children_of(parents=PARENTS) -> list[list[int]]:
    children = [[] for _ in parents]
    for joint, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(joint)
    return children


def part_labels(weights: np.ndarray) -> np.ndarray:
    """Per-vertex part index (into PART_NAMES) from the skinning weights (V, 55)."""
    joint = np.argmax(weights, axis=1)
    lookup = np.array([PART_NAMES.index(PART_OF_JOINT[j]) for j in range(weights.shape[1])])
    return lookup[joint]


# Model space (SMPL-X: y up, z forward, x left) to the upright world frame used
# by bodyscan (z up). In the upright world frame a body in its rest
# orientation faces +x and its left is +y; the root orientation of a frame is
# then a world rotation (a yaw turns the facing direction about z).
WORLD_FROM_MODEL = np.array([[0.0, 0.0, 1.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0]])
