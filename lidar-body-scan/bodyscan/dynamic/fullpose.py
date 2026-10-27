"""The pose beyond the 21 body joints, for the tracker (pose model full55).

The tracker fits the root and the 21 body joints of every frame (pose model
body21: the jaw and the eyes stay at rest, the hands at the mean hand of the
model file). With full55 it also fits, per frame, an 'extra' vector:

    jaw           (3)     axis-angle
    eyes          (2, 3)  axis-angle of the left and the right eye
    hands         (2, K)  MANO PCA coefficients of the left and the right hand
                          (bodyscan.body.hands: hand pose = mean + c . components)

so that all 55 joints of SMPL-X move. The extra vector is affine in the joint
rotations (axis-angle), so its priors and its continuity are written on the
rotations themselves: radians, comparable with those of the body joints.

With face_fixed the jaw and the eyes stay at rest and are not part of the
vector (it holds the hand coefficients only), so that the objective is that of
body21 plus the hand terms: with no hand coefficients and the flat mean hand
it is the objective of body21.
"""

from __future__ import annotations

import numpy as np

from bodyscan.body.hands import HandPCA

FACE_SIZE = 9          # jaw (3) and eyes (2 x 3) at the start of the vector, unless face_fixed


class PoseLayout:
    """Layout of the extra vector of one pose model, on the model's device."""

    def __init__(self, model, pose_model: str = "body21", hands: HandPCA | None = None, face_fixed: bool = False):
        import torch
        if pose_model not in ("body21", "full55"):
            raise ValueError(f"unknown pose model {pose_model!r} (body21 or full55)")
        if pose_model == "full55" and hands is None:
            raise ValueError("pose model full55 needs the hand PCA")
        self.pose_model = pose_model
        self.full = pose_model == "full55"
        self.hands = hands if self.full else None
        self.hand_count = hands.count if self.full else 0
        self.face_fixed = bool(face_fixed) and self.full
        self.face_size = FACE_SIZE if self.full and not self.face_fixed else 0
        self.face = slice(0, self.face_size)              # jaw then eyes; empty with face_fixed
        self.size = self.face_size + 2 * self.hand_count if self.full else 0
        self.device, self.dtype = model.device, model.dtype
        if self.full:
            self.components = torch.as_tensor(hands.components, dtype=self.dtype, device=self.device)   # (2, K, 45)
            self.mean = torch.as_tensor(hands.mean, dtype=self.dtype, device=self.device)               # (2, 45)

    def neutral(self) -> np.ndarray:
        """(E,) jaw and eyes at rest, hands at the PCA mean."""
        return np.zeros(self.size)

    def coefficients(self, extra):
        """(..., E) -> (..., 2, K) hand coefficients (torch or numpy)."""
        return extra[..., self.face_size:].reshape(*extra.shape[:-1], 2, self.hand_count)

    def hand_offsets(self, extra):
        """(F, E) torch -> (F, 2, 45) hand rotations away from the mean hand [rad]."""
        import torch
        return torch.einsum("fhk,hkn->fhn", self.coefficients(extra), self.components)

    def tail(self, extra):
        """(F, E) torch -> (F, 33, 3) axis-angle of joints 22 to 54: jaw, eyes, left hand, right hand."""
        import torch
        f = extra.shape[0]
        hands = (self.mean + self.hand_offsets(extra)).reshape(f, 30, 3)
        face = (torch.zeros(f, 3, 3, dtype=self.dtype, device=self.device) if self.face_fixed
                else extra[:, self.face].reshape(f, 3, 3))
        return torch.cat([face, hands], dim=1)

    def split(self, extra: np.ndarray) -> dict:
        """(F, E) numpy -> the parts of the pose for the output: jaw (F, 3), eyes (F, 2, 3), hand
        coefficients (F, 2, K), left_hand and right_hand (F, 15, 3) axis-angle, and the rotation
        magnitudes of the jaw (F,) and of each eye (F, 2) [deg]."""
        extra = np.asarray(extra, dtype=np.float64)
        extra = extra.reshape(1, self.size) if extra.ndim == 1 else extra.reshape(len(extra), self.size)
        coefficients = self.coefficients(extra)
        hands = self.hands.axis_angle_np(coefficients)
        face = np.zeros((len(extra), FACE_SIZE)) if self.face_fixed else extra[:, self.face]
        jaw = face[:, 0:3]
        eyes = face[:, 3:9].reshape(-1, 2, 3)
        return {"jaw": jaw, "eyes": eyes, "hand_coefficients": coefficients, "left_hand": hands[:, 0],
                "right_hand": hands[:, 1], "jaw_deg": np.degrees(np.linalg.norm(jaw, axis=-1)),
                "eye_deg": np.degrees(np.linalg.norm(eyes, axis=-1))}
