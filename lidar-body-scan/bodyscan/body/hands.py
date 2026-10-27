"""Hand pose in the MANO PCA space of SMPL-X.

SMPL-X poses each hand with 15 joints (45 axis-angle numbers). The official
model files also carry the MANO principal components of real hand poses
(hands_componentsl, hands_componentsr: (45, 45), one component per row) and
the mean hand pose (hands_meanl, hands_meanr: (45,), a relaxed hand). A hand
described by K coefficients c is

    hand pose (45,) = hands_mean + c . hands_components[:K]

(the SMPL-X convention with flat_hand_mean false). Some distributions of the
model leave these arrays out; then they are read from another file of the
same model (HandPCA.load). That file need not have an identical body, but it
must be compatible (check_compatibility): the same mesh, pose correctives,
joint regressor, skinning weights and kinematic tree, and template vertices
that differ only far from the hands. The body is always posed and shaped
with the avatar's own model file, so the avatar stays valid.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HAND_KEYS = ("hands_componentsl", "hands_componentsr", "hands_meanl", "hands_meanr")
SIDES = ("left", "right")


def _array(value) -> np.ndarray:
    """npz entries: plain arrays, or 0-d object arrays wrapping a scipy sparse matrix."""
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        value = value.item()
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value)


def file_sha1(path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Arrays that two compatible model files must have identical (the kinematic tree after normalising its root
# parent entry, which is stored as -1 or as the uint32 4294967295 depending on the distribution).
EQUAL_KEYS = ("f", "posedirs", "J_regressor", "weights")
# Joints of the hands in SMPL-X order: the wrists and the 30 finger joints.
HAND_JOINTS = (20, 21) + tuple(range(25, 55))
# Smallest distance in the rest pose [m] from a template vertex that differs between the two files to the
# nearest hand joint.
MIN_HAND_DISTANCE = 0.30
# Template vertices further apart than this [m] (Euclidean) count as differing.
VERTEX_TOLERANCE = 1e-6


def _kintree(table: np.ndarray) -> np.ndarray:
    """Kinematic tree as int64 with the root's parent written as -1."""
    table = np.asarray(table).astype(np.int64)
    if table.size and table[0, 0] in (-1, 2**32 - 1):
        table[0, 0] = -1
    return table


def check_compatibility(body_arrays: dict, hand_arrays: dict,
                        min_hand_distance: float = MIN_HAND_DISTANCE) -> dict:
    """Whether the hands of one SMPL-X model file may be used with the body of another.

    body_arrays: v_template, f, posedirs, J_regressor, weights, kintree_table of the body model (the file the
    avatar was fitted with); hand_arrays: the same arrays of the file that gives the hands, plus HAND_KEYS.
    Required: f, posedirs, J_regressor and weights equal; kintree_table equal after normalising the root
    parent entry to -1; hands_components (45, 45) and hands_mean (45,) per hand; every template vertex that
    moved by more than VERTEX_TOLERANCE lies at least min_hand_distance [m] from the nearest hand joint
    (HAND_JOINTS) of the body model in the rest pose (template, zero shape). Returns
        compatible                    bool
        errors                        list of the failed requirements
        differing_template_vertices   count
        max_template_difference_mm    largest distance between the two templates' vertices [mm]
        min_hand_distance_m           rest-pose distance from a differing vertex to the nearest hand joint
                                      [m] (None when no vertex differs)
    Shape components (shapedirs) are not compared: the avatar's shape is applied with the body model's."""
    errors = []
    for key in EQUAL_KEYS:
        if key not in body_arrays or key not in hand_arrays:
            errors.append(f"{key} missing")
        elif not np.array_equal(_array(body_arrays[key]), _array(hand_arrays[key])):
            errors.append(f"{key} differs")
    if "kintree_table" not in body_arrays or "kintree_table" not in hand_arrays:
        errors.append("kintree_table missing")
    elif not np.array_equal(_kintree(body_arrays["kintree_table"]), _kintree(hand_arrays["kintree_table"])):
        errors.append("kintree_table differs")
    for key in HAND_KEYS:
        expected = (45, 45) if key.startswith("hands_components") else (45,)
        shape = _array(hand_arrays[key]).shape if key in hand_arrays else None
        if shape != expected:
            errors.append(f"{key} has shape {shape}, expected {expected}")
    report = {"differing_template_vertices": 0, "max_template_difference_mm": 0.0,
              "min_hand_distance_m": None}
    template = _array(body_arrays["v_template"]).astype(np.float64)
    other = _array(hand_arrays["v_template"]).astype(np.float64)
    if template.shape != other.shape:
        errors.append(f"v_template has shape {other.shape}, expected {template.shape}")
    else:
        gap = np.linalg.norm(template - other, axis=1)
        differing = np.flatnonzero(gap > VERTEX_TOLERANCE)
        report["differing_template_vertices"] = int(len(differing))
        report["max_template_difference_mm"] = float(1000 * gap.max()) if len(gap) else 0.0
        if len(differing) and "J_regressor" in body_arrays:
            joints = _array(body_arrays["J_regressor"]).astype(np.float64) @ template
            hands = joints[list(HAND_JOINTS)]
            distance = np.linalg.norm(template[differing, None, :] - hands[None], axis=2)
            report["min_hand_distance_m"] = float(distance.min())
            if report["min_hand_distance_m"] < min_hand_distance:
                errors.append(f"template vertices differ {report['min_hand_distance_m']:.3f} m from a hand joint "
                              f"(at least {min_hand_distance:.2f} m required)")
    return {"compatible": not errors, "errors": errors, **report}


def compatibility_of_files(body_model, hand_file, min_hand_distance: float = MIN_HAND_DISTANCE) -> dict:
    """check_compatibility of two .npz files, with the paths and sha1 of both."""
    keys = ("v_template", "kintree_table") + EQUAL_KEYS
    with np.load(body_model, allow_pickle=True) as a, np.load(hand_file, allow_pickle=True) as b:
        body = {key: _array(a[key]) for key in keys if key in a.files}
        hands = {key: _array(b[key]) for key in keys + HAND_KEYS if key in b.files}
    report = check_compatibility(body, hands, min_hand_distance)
    return {**report, "body_model": str(body_model), "body_model_sha1": file_sha1(body_model),
            "hand_file": str(hand_file), "hand_file_sha1": file_sha1(hand_file)}


@dataclass
class HandPCA:
    """The first 'count' MANO components of both hands (count 0: the hands stay at the mean hand).

    components     (2, K, 45) rows of hands_components (left, right), radians per unit coefficient
    mean           (2, 45) relaxed hand pose (axis-angle of the 15 joints, flattened)
    path, sha1     the file they were read from
    compatibility  compatibility_of_files with the body model (empty when none was given)
    flat_mean      the mean was set to zeros (the flat hand) instead of the file's hands_mean"""
    components: np.ndarray
    mean: np.ndarray
    path: str
    sha1: str
    compatibility: dict = field(default_factory=dict)
    flat_mean: bool = False

    @property
    def count(self) -> int:
        return int(self.components.shape[1])

    @classmethod
    def load(cls, path, count: int, body_model=None, flat_mean: bool = False) -> "HandPCA":
        """Read the hand components and means from an SMPL-X .npz. With 'body_model' (the model file
        the avatar was fitted with), the two files must be compatible (check_compatibility): otherwise
        the hands would belong to another body and the program stops. flat_mean: the mean is the flat
        hand (zeros; SMPL-X flat_hand_mean true), the components are those of the file."""
        path = Path(path)
        if not path.exists():
            raise SystemExit(f"hand PCA file not found: {path}")
        with np.load(path, allow_pickle=True) as data:
            missing = [key for key in HAND_KEYS if key not in data.files]
            if missing:
                raise SystemExit(f"{path} has no MANO hand components (missing {', '.join(missing)})")
            arrays = {key: _array(data[key]) for key in HAND_KEYS}
        compatibility = {}
        if body_model is not None:
            compatibility = compatibility_of_files(body_model, path)
            if not compatibility["compatible"]:
                raise SystemExit(f"{path} is not compatible with the body of {body_model}: "
                                 f"{'; '.join(compatibility['errors'])}; its hand components cannot be used "
                                 "with this avatar")
        components = np.stack([arrays["hands_componentsl"], arrays["hands_componentsr"]])
        mean = np.stack([arrays["hands_meanl"].reshape(45), arrays["hands_meanr"].reshape(45)])
        if components.ndim != 3 or components.shape[2] != 45:
            raise SystemExit(f"{path}: hand components of shape {components.shape[1:]}, expected (n, 45)")
        if not 0 <= count <= components.shape[1]:
            raise SystemExit(f"{count} hand components asked, {path} has {components.shape[1]}")
        if flat_mean:
            mean = np.zeros_like(mean)
        return cls(components[:, :count].astype(np.float64), mean.astype(np.float64), str(path), file_sha1(path),
                   compatibility, flat_mean)

    def axis_angle_np(self, coefficients: np.ndarray) -> np.ndarray:
        """(..., 2, K) coefficients -> (..., 2, 15, 3) hand joint rotations (axis-angle)."""
        pose = self.mean + np.einsum("...hk,hkn->...hn", coefficients, self.components)
        return pose.reshape(*pose.shape[:-1], 15, 3)
