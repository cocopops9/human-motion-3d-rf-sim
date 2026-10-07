"""SMPL-X compatible body model: shape, pose, linear blend skinning (PyTorch).

The model file is the SMPL-X .npz of the official distribution
(SMPLX_NEUTRAL.npz, SMPLX_MALE.npz, SMPLX_FEMALE.npz; register at
https://smpl-x.is.tue.mpg.de, the licence does not allow redistribution, so
the file never goes into the repository), or the procedural test body of
bodyscan.body.testbody, which uses the same keys.

    v_template   (V, 3)       rest mesh (T-pose), metres, model space y up, z forward
    f            (F, 3)       triangles
    shapedirs    (V, 3, S)    shape (and expression) blend shapes
    posedirs     (V, 3, P)    pose blend shapes, P = 9 x 54
    J_regressor  (55, V)      joints from vertices
    weights      (V, 55)      skinning weights
    kintree_table (2, 55)     parents in the first row
    hands_meanl, hands_meanr  (45,) relaxed hand pose

A body is posed in two stages. The model stage is SMPL-X proper: shape
blend shapes, pose blend shapes, forward kinematics and linear blend
skinning in model space, with the root joint rotation usually left at the
identity. The world stage maps model space to the upright world frame of
bodyscan (z up; skeleton.WORLD_FROM_MODEL), turns it by the world root
rotation about the pelvis and puts the pelvis at the root position:

    world = R_root . WORLD_FROM_MODEL . (model - pelvis) + root_position

so the root position is the pelvis itself (not the SMPL 'transl', which
depends on the shape) and the root rotation is a plain world rotation.

Finer meshes: ShapedBody at level L > 0 subdivides every triangle into 4,
L times (midpoint subdivision); every per-vertex quantity of a new vertex is
the mean of the two ends of its edge, so the coarse vertices keep their
indices and a level-L body is the level-0 body plus detail. Per-vertex
displacements (along the rest normals) carry the detail of a scan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bodyscan.body import skeleton

try:
    import torch
except ImportError as error:                       # pragma: no cover - depends on the installation
    raise ImportError("the body model needs PyTorch: python -m pip install torch "
                      "(CUDA build on a PC with an NVIDIA GPU, see pytorch.org)") from error


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class Subdivision:
    """Midpoint subdivision of a triangle mesh, 'levels' times."""

    def __init__(self, faces: np.ndarray, vertex_count: int, levels: int):
        self.edges = []
        self.counts = [int(vertex_count)]
        f = np.asarray(faces, dtype=np.int64)
        for _ in range(levels):
            n = self.counts[-1]
            pairs = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
            pairs.sort(axis=1)
            unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
            inverse = inverse.reshape(-1)
            m = n + inverse.reshape(3, -1).T                          # (F, 3): midpoints of edges 01, 12, 20
            a, b, c = f[:, 0], f[:, 1], f[:, 2]
            m01, m12, m20 = m[:, 0], m[:, 1], m[:, 2]
            f = np.concatenate([np.stack([a, m01, m20], 1), np.stack([m01, b, m12], 1),
                                np.stack([m20, m12, c], 1), np.stack([m01, m12, m20], 1)])
            self.edges.append(unique)
            self.counts.append(n + len(unique))
        self.faces = f
        self._edge_tensors = {}

    @property
    def levels(self) -> int:
        return len(self.edges)

    def apply(self, values):
        """(..., V0, C) per-vertex values (numpy or torch) -> (..., VL, C)."""
        is_torch = isinstance(values, torch.Tensor)
        for k, edges in enumerate(self.edges):
            if is_torch:
                key = (k, values.device)
                if key not in self._edge_tensors:
                    self._edge_tensors[key] = torch.as_tensor(edges, device=values.device)
                e = self._edge_tensors[key]
                values = torch.cat([values, 0.5 * (values[..., e[:, 0], :] + values[..., e[:, 1], :])], dim=-2)
            else:
                values = np.concatenate([values, 0.5 * (values[..., edges[:, 0], :] + values[..., edges[:, 1], :])],
                                        axis=-2)
        return values


def vertex_normals(vertices, faces):
    """Area-weighted unit vertex normals of (..., V, 3) vertices (torch), faces (F, 3) long tensor."""
    v0 = vertices[..., faces[:, 0], :]
    v1 = vertices[..., faces[:, 1], :]
    v2 = vertices[..., faces[:, 2], :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)                 # length = 2 x area
    normals = torch.zeros_like(vertices)
    for k in range(3):
        normals.index_add_(-2, faces[:, k], face_normals)
    return torch.nn.functional.normalize(normals, dim=-1, eps=1e-12)


def _as_array(value):
    """npz entries: plain arrays, or 0-d object arrays wrapping a scipy sparse matrix (J_regressor)."""
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        value = value.item()
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value)


@dataclass
class Posed:
    """A posed body (world frame): vertices (N, V, 3), joints (N, 55, 3), and the
    rotation of every joint in the world (N, 55, 3, 3)."""
    vertices: "torch.Tensor"
    joints: "torch.Tensor"
    joint_rotations: "torch.Tensor"


class BodyModel:
    """An SMPL-X model file (see the module docstring) on a device."""

    def __init__(self, path, num_betas: int = 10, device: str | None = None, dtype=torch.float32):
        self.path = Path(path)
        if not self.path.exists():
            raise SystemExit(f"body model {self.path} not found. Download SMPL-X (npz) from "
                             "https://smpl-x.is.tue.mpg.de after registering, or make the test body with "
                             "'python -m bodyscan make-test-body'")
        data = np.load(self.path, allow_pickle=True)
        missing = [k for k in ("v_template", "f", "shapedirs", "J_regressor", "weights", "kintree_table")
                   if k not in data.files]
        if missing:
            raise SystemExit(f"{self.path} is not an SMPL-X npz model (missing {', '.join(missing)})")
        self.device = torch.device(device or default_device())
        self.dtype = dtype
        v_template = _as_array(data["v_template"]).astype(np.float64)
        faces = _as_array(data["f"]).astype(np.int64)
        shapedirs = _as_array(data["shapedirs"]).astype(np.float64)
        regressor = _as_array(data["J_regressor"]).astype(np.float64)
        weights = _as_array(data["weights"]).astype(np.float64)
        kintree = _as_array(data["kintree_table"]).astype(np.int64)
        if weights.shape[1] != skeleton.NUM_JOINTS or regressor.shape[0] != skeleton.NUM_JOINTS:
            raise SystemExit(f"{self.path}: {weights.shape[1]} joints; only SMPL-X (55 joints) is supported")
        parents = kintree[0].copy()
        parents[0] = -1
        parents[parents >= skeleton.NUM_JOINTS] = -1
        if tuple(int(p) for p in parents) != skeleton.PARENTS:
            raise SystemExit(f"{self.path}: the kinematic tree is not the SMPL-X one")
        self.parents = skeleton.PARENTS
        # Shape space: SMPL-X 1.1 has 300 shape + 100 expression components, 1.0 has 10 + 10.
        total = shapedirs.shape[2]
        shape_count = 300 if total >= 400 else total // 2
        self.num_betas = int(min(num_betas, shape_count))
        self.faces_np = faces
        self.vertex_count = len(v_template)
        self.gender = str(data["gender"]) if "gender" in data.files else "neutral"
        self.v_template = torch.as_tensor(v_template, dtype=dtype, device=self.device)
        self.shapedirs = torch.as_tensor(shapedirs[:, :, :self.num_betas], dtype=dtype, device=self.device)
        self.regressor = torch.as_tensor(regressor, dtype=dtype, device=self.device)
        self.weights = torch.as_tensor(weights, dtype=dtype, device=self.device)
        self.faces = torch.as_tensor(faces, device=self.device)
        posedirs = _as_array(data["posedirs"]).astype(np.float64) if "posedirs" in data.files else None
        self.has_pose_blendshapes = posedirs is not None and bool(np.any(posedirs != 0))
        if self.has_pose_blendshapes:
            flat = posedirs.reshape(self.vertex_count * 3, -1).T                 # (P, V*3)
            self.posedirs = torch.as_tensor(flat, dtype=dtype, device=self.device)
        else:
            self.posedirs = None
        mean_left = _as_array(data["hands_meanl"]) if "hands_meanl" in data.files else np.zeros(45)
        mean_right = _as_array(data["hands_meanr"]) if "hands_meanr" in data.files else np.zeros(45)
        self.hand_mean = {"left": mean_left.reshape(15, 3).astype(np.float64),
                          "right": mean_right.reshape(15, 3).astype(np.float64)}
        self.sha1 = _file_sha1(self.path)
        self._subdivisions: dict[int, Subdivision] = {}

    # ------------------------------------------------------------------
    def subdivision(self, level: int) -> Subdivision | None:
        if level <= 0:
            return None
        if level not in self._subdivisions:
            self._subdivisions[level] = Subdivision(self.faces_np, self.vertex_count, level)
        return self._subdivisions[level]

    def tensor(self, value, dtype=None):
        return torch.as_tensor(np.asarray(value), dtype=dtype or self.dtype, device=self.device)

    def shaped(self, betas=None, level: int = 0, normal_displacement=None, displacement=None) -> "ShapedBody":
        """The body of one person in the rest pose: shape blend shapes ('betas'),
        then, at subdivision 'level', displacements along the rest normals
        ((VL,) metres) and/or free displacements ((VL, 3))."""
        betas_t = (torch.zeros(self.num_betas, dtype=self.dtype, device=self.device) if betas is None
                   else torch.as_tensor(betas, dtype=self.dtype, device=self.device)[:self.num_betas])
        if betas_t.numel() < self.num_betas:
            betas_t = torch.cat([betas_t, torch.zeros(self.num_betas - betas_t.numel(), dtype=self.dtype,
                                                      device=self.device)])
        base = self.v_template + torch.einsum("vcs,s->vc", self.shapedirs, betas_t)
        rest_joints = self.regressor @ base
        return ShapedBody(self, base, rest_joints, level, normal_displacement, displacement)

    # ------------------------------------------------------------------
    def rotations(self, root=None, body=None, jaw=None, eyes=None, left_hand=None, right_hand=None, count=1):
        """(N, 55, 3, 3) joint rotations from axis-angle parts (tensors or arrays):
        root (N, 3) model-space root rotation (normally left out: identity),
        body (N, 21, 3), jaw (N, 3), eyes (N, 2, 3), hands (N, 15, 3); a part
        left out is the identity, except the hands, which take the relaxed mean pose."""
        from bodyscan.body.rotations import axis_angle_to_matrix
        parts = {"root": root, "body": body, "jaw": jaw, "eyes": eyes, "left_hand": left_hand,
                 "right_hand": right_hand}
        n = count
        for value in parts.values():
            if value is not None:
                n = value.shape[0]
                break
        full = torch.zeros((n, skeleton.NUM_JOINTS, 3), dtype=self.dtype, device=self.device)

        def put(index, value):
            full[:, index] = torch.as_tensor(value, dtype=self.dtype, device=self.device).reshape(
                full[:, index].shape)

        if root is not None:
            put(0, root)
        if body is not None:
            put(skeleton.BODY, body)
        if jaw is not None:
            put(skeleton.JAW, jaw)
        if eyes is not None:
            put(slice(skeleton.LEFT_EYE, skeleton.RIGHT_EYE + 1), eyes)
        put(skeleton.LEFT_HAND, left_hand if left_hand is not None else np.tile(self.hand_mean["left"], (n, 1, 1)))
        put(skeleton.RIGHT_HAND, right_hand if right_hand is not None else np.tile(self.hand_mean["right"], (n, 1, 1)))
        return axis_angle_to_matrix(full)


class ShapedBody:
    """One person's body (shape and displacements fixed) at one subdivision level."""

    def __init__(self, model: BodyModel, base_vertices, rest_joints, level: int = 0, normal_displacement=None,
                 displacement=None):
        self.model = model
        self.level = int(level)
        self.subdivision = model.subdivision(self.level)
        self.base_vertices = base_vertices                   # (V0, 3) shaped, no displacement
        self.rest_joints = rest_joints                       # (55, 3)
        if self.subdivision is not None:
            vertices = self.subdivision.apply(base_vertices)
            self.faces_np = self.subdivision.faces
            self.weights = self.subdivision.apply(model.weights)
        else:
            vertices = base_vertices
            self.faces_np = model.faces_np
            self.weights = model.weights
        self.faces = torch.as_tensor(self.faces_np, device=model.device)
        self.rest_normals = vertex_normals(vertices, self.faces)
        if normal_displacement is not None:
            d = torch.as_tensor(normal_displacement, dtype=model.dtype, device=model.device).reshape(-1, 1)
            if d.shape[0] != vertices.shape[0]:
                raise ValueError(f"normal displacement has {d.shape[0]} values for {vertices.shape[0]} vertices "
                                 f"(level {self.level})")
            vertices = vertices + d * self.rest_normals
        if displacement is not None:
            vertices = vertices + torch.as_tensor(displacement, dtype=model.dtype, device=model.device)
        self.vertices = vertices                              # (VL, 3) rest pose with detail
        self.world_from_model = torch.as_tensor(skeleton.WORLD_FROM_MODEL, dtype=model.dtype, device=model.device)

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    def with_vertices(self, vertices) -> "ShapedBody":
        """Same body with other rest vertices (used by the displacement fit)."""
        clone = object.__new__(ShapedBody)
        clone.__dict__.update(self.__dict__)
        clone.vertices = vertices
        return clone

    def pose(self, rotations, root_rotation=None, root_position=None, vertices=None, subset=None) -> Posed:
        """Pose the body.

        rotations      (N, 55, 3, 3) joint rotations (model stage; usually identity at joint 0)
        root_rotation  (N, 3, 3) world rotation (None: identity)
        root_position  (N, 3) world position of the pelvis (None: the rest pelvis mapped to the world)
        vertices       optional (VL, 3) rest vertices replacing self.vertices (differentiable)
        subset         optional vertex indices: only these vertices are posed (feet, a body part)"""
        model = self.model
        rest = self.vertices if vertices is None else vertices
        weights = self.weights
        if subset is not None:
            subset = torch.as_tensor(subset, device=model.device)
            rest, weights = rest[subset], weights[subset]
        n = rotations.shape[0]
        eye = torch.eye(3, dtype=model.dtype, device=model.device)
        if model.has_pose_blendshapes:
            feature = (rotations[:, 1:] - eye).reshape(n, -1)
            offsets = (feature @ model.posedirs).reshape(n, model.vertex_count, 3)
            if self.subdivision is not None:
                offsets = self.subdivision.apply(offsets)
            if subset is not None:
                offsets = offsets[:, subset]
            posed_rest = rest.unsqueeze(0) + offsets
        else:
            posed_rest = rest.unsqueeze(0).expand(n, -1, -1)
        rot, pos = self.kinematics(rotations)
        joints = self.rest_joints
        # skinning transforms A_j = [R_j | p_j - R_j J_j]
        translation = pos - (rot @ joints.reshape(1, -1, 3, 1)).squeeze(-1)
        transforms = torch.cat([rot, translation.unsqueeze(-1)], dim=-1).reshape(n, skeleton.NUM_JOINTS, 12)
        blended = (weights @ transforms).reshape(n, -1, 3, 4)            # (N, V, 3, 4)
        model_vertices = (blended[..., :3] @ posed_rest.unsqueeze(-1)).squeeze(-1) + blended[..., 3]
        # world stage
        to_world = self.world_from_model
        if root_rotation is not None:
            to_world = root_rotation @ to_world                          # (N, 3, 3)
        else:
            to_world = to_world.expand(n, 3, 3)
        pelvis = pos[:, :1]                                              # (N, 1, 3)
        if root_position is None:
            root_position = (self.world_from_model @ joints[0].reshape(3, 1)).reshape(1, 3).expand(n, 3)
        origin = root_position.reshape(n, 1, 3)
        vertices_world = ((model_vertices - pelvis) @ to_world.transpose(1, 2)) + origin
        joints_world = ((pos - pelvis) @ to_world.transpose(1, 2)) + origin
        joint_rotations = to_world.unsqueeze(1) @ rot
        return Posed(vertices_world, joints_world, joint_rotations)

    def normals(self, vertices):
        return vertex_normals(vertices, self.faces)

    def kinematics(self, rotations):
        """Forward kinematics in model space: (N, 55, 3, 3) joint rotations in
        their parent frames -> (N, 55, 3, 3) global rotations and (N, 55, 3)
        positions. The joints are processed by depth in the tree (about ten
        batched steps instead of one step per joint)."""
        levels, parent_slots, order = _tree_levels(self.model.parents, rotations.device)
        joints = self.rest_joints
        n = rotations.shape[0]
        offsets = joints - joints[list(max(p, 0) for p in self.model.parents)]
        offsets = torch.cat([joints[:1], offsets[1:]])                     # root: its rest position
        level_rot = [rotations[:, levels[0]]]
        level_pos = [offsets[levels[0]].expand(n, -1, -1)]
        for depth in range(1, len(levels)):
            previous_rot = torch.cat(level_rot, dim=1)
            previous_pos = torch.cat(level_pos, dim=1)
            parent_rot = previous_rot[:, parent_slots[depth]]
            parent_pos = previous_pos[:, parent_slots[depth]]
            level_rot.append(parent_rot @ rotations[:, levels[depth]])
            level_pos.append(parent_pos + (parent_rot @ offsets[levels[depth]].unsqueeze(-1)).squeeze(-1))
        rot = torch.cat(level_rot, dim=1)[:, order]
        pos = torch.cat(level_pos, dim=1)[:, order]
        return rot, pos

    def joints_world(self, rotations, root_rotation=None, root_position=None):
        """(N, 55, 3) world joint positions only (no skinning: cheap)."""
        rot, pos = self.kinematics(rotations)
        n = rotations.shape[0]
        to_world = self.world_from_model if root_rotation is None else root_rotation @ self.world_from_model
        to_world = to_world.expand(n, 3, 3)
        if root_position is None:
            root_position = (self.world_from_model @ self.rest_joints[0].reshape(3, 1)).reshape(1, 3).expand(n, 3)
        return ((pos - pos[:, :1]) @ to_world.transpose(1, 2)) + root_position.reshape(n, 1, 3)


_LEVELS_CACHE: dict = {}


def _tree_levels(parents, device):
    """Joints grouped by depth; for every depth > 0 the position of each joint's
    parent in the concatenation of the previous depths; and the permutation
    that puts the concatenated joints back in model order."""
    key = (tuple(parents), str(device))
    if key not in _LEVELS_CACHE:
        depth = np.zeros(len(parents), dtype=np.int64)
        for j, p in enumerate(parents):
            depth[j] = 0 if p < 0 else depth[p] + 1
        levels = [np.flatnonzero(depth == d) for d in range(depth.max() + 1)]
        slot = {}
        position = 0
        parent_slots = [None]
        for d, members in enumerate(levels):
            if d > 0:
                parent_slots.append(np.array([slot[parents[j]] for j in members]))
            for j in members:
                slot[j] = position
                position += 1
        concatenated = np.concatenate(levels)
        order = np.argsort(concatenated)
        _LEVELS_CACHE[key] = ([torch.as_tensor(m, device=device) for m in levels],
                              [None] + [torch.as_tensor(p, device=device) for p in parent_slots[1:]],
                              torch.as_tensor(order, device=device))
    return _LEVELS_CACHE[key]


def _file_sha1(path: Path) -> str:
    import hashlib
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
