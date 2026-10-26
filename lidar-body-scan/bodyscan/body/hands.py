"""Hand pose in the MANO PCA space of SMPL-X.

SMPL-X poses each hand with 15 joints (45 axis-angle numbers). The official
model files also carry the MANO principal components of real hand poses
(hands_componentsl, hands_componentsr: (45, 45), one component per row) and
the mean hand pose (hands_meanl, hands_meanr: (45,), a relaxed hand). A hand
described by K coefficients c is

    hand pose (45,) = hands_mean + c . hands_components[:K]

(the SMPL-X convention with flat_hand_mean false). Some distributions of the
model leave these arrays out; then they are read from another file of the
same model (HandPCA.load), after checking that its body arrays are the same,
so that an avatar fitted with the first file stays valid.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The arrays that define the body: two model files with these equal pose and skin a body identically.
BODY_KEYS = ("v_template", "f", "shapedirs", "posedirs", "J_regressor", "weights", "kintree_table")
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


def compare_body_arrays(path_a, path_b, tolerance: float = 0.0) -> list[str]:
    """Names of the BODY_KEYS arrays that differ between two model files (missing in one, other shape,
    or values further apart than 'tolerance'); an empty list when the bodies are the same."""
    differ = []
    with np.load(path_a, allow_pickle=True) as a, np.load(path_b, allow_pickle=True) as b:
        arrays = {key: (_array(a[key]), _array(b[key])) for key in BODY_KEYS if key in a.files and key in b.files}
    for key in BODY_KEYS:
        if key not in arrays:
            differ.append(f"{key} (missing)")
            continue
        x, y = arrays[key]
        if x.shape != y.shape:
            differ.append(f"{key} (shape {x.shape} against {y.shape})")
        elif np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
            gap = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
            if gap > tolerance:
                differ.append(f"{key} (largest difference {gap:.3g})")
        elif not np.array_equal(x, y):
            differ.append(key)
    return differ


@dataclass
class HandPCA:
    """The first 'count' MANO components of both hands (count 0: the hands stay at the mean hand).

    components  (2, K, 45) rows of hands_components (left, right), radians per unit coefficient
    mean        (2, 45) relaxed hand pose (axis-angle of the 15 joints, flattened)
    path, sha1  the file they were read from"""
    components: np.ndarray
    mean: np.ndarray
    path: str
    sha1: str

    @property
    def count(self) -> int:
        return int(self.components.shape[1])

    @classmethod
    def load(cls, path, count: int, body_model=None) -> "HandPCA":
        """Read the hand components and means from an SMPL-X .npz. With 'body_model' (the model file
        the avatar was fitted with), the two files must have the same body arrays (BODY_KEYS):
        otherwise the hands would belong to another body and the program stops."""
        path = Path(path)
        if not path.exists():
            raise SystemExit(f"hand PCA file not found: {path}")
        with np.load(path, allow_pickle=True) as data:
            missing = [key for key in HAND_KEYS if key not in data.files]
            if missing:
                raise SystemExit(f"{path} has no MANO hand components (missing {', '.join(missing)})")
            arrays = {key: _array(data[key]) for key in HAND_KEYS}
        if body_model is not None:
            differ = compare_body_arrays(body_model, path)
            if differ:
                raise SystemExit(f"{path} is not the same body as {body_model}: {', '.join(differ)} differ; "
                                 "its hand components cannot be used with this avatar")
        components = np.stack([arrays["hands_componentsl"], arrays["hands_componentsr"]])
        mean = np.stack([arrays["hands_meanl"].reshape(45), arrays["hands_meanr"].reshape(45)])
        if components.ndim != 3 or components.shape[2] != 45:
            raise SystemExit(f"{path}: hand components of shape {components.shape[1:]}, expected (n, 45)")
        if not 0 <= count <= components.shape[1]:
            raise SystemExit(f"{count} hand components asked, {path} has {components.shape[1]}")
        return cls(components[:, :count].astype(np.float64), mean.astype(np.float64), str(path), file_sha1(path))

    def axis_angle_np(self, coefficients: np.ndarray) -> np.ndarray:
        """(..., 2, K) coefficients -> (..., 2, 15, 3) hand joint rotations (axis-angle)."""
        pose = self.mean + np.einsum("...hk,hkn->...hn", coefficients, self.components)
        return pose.reshape(*pose.shape[:-1], 15, 3)
