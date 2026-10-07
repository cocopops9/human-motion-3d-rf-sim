"""A procedural body in the SMPL-X file format, for tests and demonstrations.

SMPL-X itself cannot be shipped (its licence asks every user to register),
so the tests, the synthetic recordings and the examples use this body: an
adult of about 1.76 m built from tapered capsules and ellipsoids around the
55 SMPL-X joints, in the SMPL-X rest pose (T-pose, palms down, y up, facing
+z) and with the SMPL-X joint order and kinematic tree. It is written as an
.npz with the keys of the official model, so every function that takes an
SMPL-X model takes this file as well:

    surface       zero level set of the union of the primitives (marching
                  cubes on a 4 mm grid), decimated to about 10,000 vertices
                  (the size of SMPL-X)
    weights       from the distance to the bones, blended over a band
                  proportional to the limb radius, smoothed on the mesh
    J_regressor   every joint is an exact affine combination of the
                  vertices around it, so the joints follow shape changes
    shapedirs     10 hand-made shape directions (stature, girth, leg and arm
                  length, shoulder, belly, hips, chest, head, neck) and 10
                  empty expression directions
    posedirs      zero (no pose blend shapes)
    hands_mean    fingers slightly bent (a relaxed hand)

It is not anatomically accurate and not meant to be: it exercises the same
code paths as SMPL-X with a body of human size and proportions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bodyscan.body import skeleton

# ----------------------------------------------------------------------------
# Skeleton of the test body (model space, metres, feet on y = 0, T-pose)
# ----------------------------------------------------------------------------

_FINGERS = {                      # z offset of the knuckle from the wrist, segment lengths, radius
    "index": (0.039, (0.095, 0.042, 0.025, 0.022), 0.0085),
    "middle": (0.013, (0.100, 0.046, 0.028, 0.024), 0.0085),
    "ring": (-0.013, (0.095, 0.043, 0.026, 0.023), 0.0082),
    "pinky": (-0.039, (0.085, 0.034, 0.020, 0.020), 0.0075),
}
_THUMB = ((0.025, -0.012, 0.025), (0.040, 0.032, 0.025), 0.0100)


def rest_joints() -> tuple[np.ndarray, dict]:
    """(55, 3) joint positions and the end points of the leaf bones (fingertips, toes, head top)."""
    j = np.zeros((skeleton.NUM_JOINTS, 3))

    def put(name, xyz):
        j[skeleton.JOINT[name]] = xyz

    put("pelvis", (0.0, 0.95, 0.0))
    put("spine1", (0.0, 1.06, -0.015))
    put("spine2", (0.0, 1.19, -0.02))
    put("spine3", (0.0, 1.32, -0.01))
    put("neck", (0.0, 1.50, -0.015))
    put("head", (0.0, 1.60, 0.0))
    put("jaw", (0.0, 1.575, 0.055))
    ends = {"head": np.array([0.0, 1.79, 0.01])}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        put(f"{side}_hip", (sign * 0.085, 0.88, 0.0))
        put(f"{side}_knee", (sign * 0.095, 0.49, 0.01))
        put(f"{side}_ankle", (sign * 0.10, 0.085, -0.02))
        put(f"{side}_foot", (sign * 0.11, 0.025, 0.11))
        put(f"{side}_collar", (sign * 0.075, 1.42, -0.01))
        put(f"{side}_shoulder", (sign * 0.185, 1.42, -0.02))
        put(f"{side}_elbow", (sign * 0.45, 1.42, -0.03))
        put(f"{side}_wrist", (sign * 0.70, 1.42, -0.025))
        put(f"{side}_eye", (sign * 0.032, 1.655, 0.075))
        ends[f"{side}_foot"] = np.array([sign * 0.11, 0.03, 0.18])
        wrist = j[skeleton.JOINT[f"{side}_wrist"]]
        for finger, (z, lengths, _) in _FINGERS.items():
            spread = np.radians(8.0) * (z / 0.039)                    # fingers fanned a little
            direction = np.array([sign * np.cos(spread), 0.0, np.sin(spread)])
            point = wrist + np.array([sign * lengths[0], -0.004, z])
            for k in range(3):
                put(f"{side}_{finger}{k + 1}", point)
                point = point + lengths[k + 1] * direction
            ends[f"{side}_{finger}3"] = point
        offset, lengths, _ = _THUMB
        direction = np.array([sign * 1.0, -0.2, 0.9])
        direction /= np.linalg.norm(direction)
        point = wrist + np.array([sign * offset[0], offset[1], offset[2]])
        for k in range(3):
            put(f"{side}_thumb{k + 1}", point)
            point = point + lengths[k] * direction
        ends[f"{side}_thumb3"] = point
    return j, ends


# ----------------------------------------------------------------------------
# Primitives (approximate signed distances, exact enough near the surface)
# ----------------------------------------------------------------------------

@dataclass
class Capsule:
    a: np.ndarray
    b: np.ndarray
    ra: float
    rb: float
    flat_bottom: bool = False

    def bounds(self):
        r = max(self.ra, self.rb) + 0.01
        return np.minimum(self.a, self.b) - r, np.maximum(self.a, self.b) + r

    def sdf(self, p):
        ab = self.b - self.a
        t = np.clip(((p - self.a) @ ab) / (ab @ ab), 0.0, 1.0)
        closest = self.a + t[..., None] * ab
        value = np.linalg.norm(p - closest, axis=-1) - (self.ra + t * (self.rb - self.ra))
        if self.flat_bottom:
            value = np.maximum(value, -p[..., 1])
        return value


@dataclass
class Ellipsoid:
    center: np.ndarray
    radii: np.ndarray

    def bounds(self):
        return self.center - self.radii - 0.01, self.center + self.radii + 0.01

    def sdf(self, p):
        q = (p - self.center) / self.radii
        return (np.linalg.norm(q, axis=-1) - 1.0) * self.radii.min()


def primitives(joints: np.ndarray, ends: dict) -> list:
    j = {name: joints[index] for name, index in skeleton.JOINT.items()}
    v = np.array
    parts = [Ellipsoid(v([0.0, 0.93, -0.01]), v([0.165, 0.12, 0.11])),          # pelvis
             Ellipsoid(v([0.0, 1.10, 0.0]), v([0.150, 0.16, 0.10])),            # abdomen
             Ellipsoid(v([0.0, 1.30, 0.0]), v([0.165, 0.15, 0.11])),            # chest
             Capsule(v([-0.17, 1.40, -0.02]), v([0.17, 1.40, -0.02]), 0.06, 0.06),   # shoulders
             Capsule(v([0.0, 1.42, -0.015]), v([0.0, 1.58, -0.005]), 0.052, 0.050),  # neck
             Ellipsoid(v([0.0, 1.68, 0.01]), v([0.077, 0.11, 0.095])),          # head
             Ellipsoid(v([0.0, 1.655, 0.10]), v([0.012, 0.02, 0.015]))]         # nose
    for side, sign in (("left", 1.0), ("right", -1.0)):
        parts += [Capsule(j[f"{side}_shoulder"], j[f"{side}_elbow"], 0.048, 0.040),
                  Capsule(j[f"{side}_elbow"], j[f"{side}_wrist"], 0.040, 0.030),
                  Ellipsoid(j[f"{side}_wrist"] + v([sign * 0.05, -0.003, 0.0]), v([0.055, 0.016, 0.045])),
                  Capsule(j[f"{side}_hip"] + v([0, 0.03, 0]), j[f"{side}_knee"], 0.080, 0.055),
                  Capsule(j[f"{side}_knee"], j[f"{side}_ankle"], 0.050, 0.036),
                  Capsule(v([sign * 0.10, 0.045, -0.055]), ends[f"{side}_foot"], 0.042, 0.034, flat_bottom=True)]
        for finger, (_, _, radius) in _FINGERS.items():
            chain = [j[f"{side}_{finger}{k}"] for k in (1, 2, 3)] + [ends[f"{side}_{finger}3"]]
            parts += [Capsule(chain[k], chain[k + 1], radius, radius * 0.95) for k in range(3)]
        _, _, radius = _THUMB
        chain = [j[f"{side}_thumb{k}"] for k in (1, 2, 3)] + [ends[f"{side}_thumb3"]]
        parts += [Capsule(chain[k], chain[k + 1], radius, radius * 0.92) for k in range(3)]
    return parts


def bones(joints: np.ndarray, ends: dict) -> list[tuple[int, np.ndarray, np.ndarray, float]]:
    """(owner joint, start, end, radius): the joint whose rotation moves the segment."""
    J = skeleton.JOINT
    j = joints
    result = []

    def add(owner, a, b, radius):
        result.append((J[owner], np.asarray(a, dtype=float), np.asarray(b, dtype=float), radius))

    add("pelvis", j[J["pelvis"]], j[J["spine1"]], 0.14)
    add("spine1", j[J["spine1"]], j[J["spine2"]], 0.14)
    add("spine2", j[J["spine2"]], j[J["spine3"]], 0.14)
    add("spine3", j[J["spine3"]], j[J["neck"]], 0.12)
    add("neck", j[J["neck"]], j[J["head"]], 0.05)
    add("head", j[J["head"]], ends["head"], 0.09)
    for side in ("left", "right"):
        add("pelvis", j[J["pelvis"]], j[J[f"{side}_hip"]], 0.12)
        add("spine3", j[J["spine3"]], j[J[f"{side}_collar"]], 0.10)
        add(f"{side}_collar", j[J[f"{side}_collar"]], j[J[f"{side}_shoulder"]], 0.06)
        add(f"{side}_shoulder", j[J[f"{side}_shoulder"]], j[J[f"{side}_elbow"]], 0.045)
        add(f"{side}_elbow", j[J[f"{side}_elbow"]], j[J[f"{side}_wrist"]], 0.035)
        add(f"{side}_hip", j[J[f"{side}_hip"]], j[J[f"{side}_knee"]], 0.07)
        add(f"{side}_knee", j[J[f"{side}_knee"]], j[J[f"{side}_ankle"]], 0.045)
        add(f"{side}_ankle", j[J[f"{side}_ankle"]], j[J[f"{side}_foot"]], 0.04)
        add(f"{side}_foot", j[J[f"{side}_foot"]], ends[f"{side}_foot"], 0.035)
        for finger in ("index", "middle", "ring", "pinky", "thumb"):
            radius = _THUMB[2] if finger == "thumb" else _FINGERS[finger][2]
            add(f"{side}_wrist", j[J[f"{side}_wrist"]], j[J[f"{side}_{finger}1"]], 0.03)
            chain = [j[J[f"{side}_{finger}{k}"]] for k in (1, 2, 3)] + [ends[f"{side}_{finger}3"]]
            for k in range(3):
                add(f"{side}_{finger}{k + 1}", chain[k], chain[k + 1], radius)
    return result


def _segment_distance(points, a, b):
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    closest = a + t[:, None] * ab
    return np.linalg.norm(points - closest, axis=1), closest


# ----------------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------------

def surface(voxel: float = 0.004, target_triangles: int = 20000):
    """Mesh of the union of the primitives: (vertices, faces) in model space."""
    import open3d as o3d
    from bodyscan.meshing.marching import marching_cubes
    joints, ends = rest_joints()
    shapes = primitives(joints, ends)
    lower = np.array([-0.99, -0.012, -0.18])
    upper = np.array([0.99, 1.83, 0.26])
    shape = np.ceil((upper - lower) / voxel).astype(int) + 1
    sdf = np.full(shape, 1.0, dtype=np.float32)
    axes = [lower[k] + voxel * np.arange(shape[k]) for k in range(3)]
    for primitive in shapes:
        low, high = primitive.bounds()
        i0 = np.clip(np.floor((low - lower) / voxel).astype(int), 0, shape - 1)
        i1 = np.clip(np.ceil((high - lower) / voxel).astype(int) + 1, 0, shape)
        grid = np.stack(np.meshgrid(axes[0][i0[0]:i1[0]], axes[1][i0[1]:i1[1]], axes[2][i0[2]:i1[2]],
                                    indexing="ij"), axis=-1)
        block = sdf[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
        np.minimum(block, primitive.sdf(grid).astype(np.float32), out=block)
    for axis in range(3):                                   # the grid border is outside
        sdf[(slice(None),) * axis + (0,)] = 1.0
        sdf[(slice(None),) * axis + (-1,)] = 1.0
    sdf[np.abs(sdf) < 1e-7] = 1e-7
    vertices, faces = marching_cubes(sdf, voxel)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices + lower), o3d.utility.Vector3iVector(faces))
    mesh = mesh.simplify_quadric_decimation(target_triangles)
    mesh.remove_duplicated_vertices()
    from bodyscan.meshing.smoothing import tangential_relaxation      # better shaped triangles
    relaxed = tangential_relaxation(np.asarray(mesh.vertices), np.asarray(mesh.triangles).astype(np.int64), 10)
    mesh.vertices = o3d.utility.Vector3dVector(relaxed)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    labels, sizes, _ = mesh.cluster_connected_triangles()
    labels, sizes = np.asarray(labels), np.asarray(sizes)
    mesh.remove_triangles_by_mask(labels != int(np.argmax(sizes)))
    mesh.remove_unreferenced_vertices()
    v, f = np.asarray(mesh.vertices).copy(), np.asarray(mesh.triangles).copy()
    if np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() < 0:
        f = f[:, ::-1].copy()                                   # outward orientation
    return v, f


def skinning_weights(vertices: np.ndarray, faces: np.ndarray, joints: np.ndarray, ends: dict,
                     smoothing: int = 10, influences: int = 4) -> np.ndarray:
    """(V, 55) weights: per bone exp(-(d - d_min) / (0.5 r)) among the bones
    within d_min + 1.5 r of the vertex (r: radius of the nearest bone), summed
    per owner joint, smoothed over the mesh, cut to the largest 'influences'."""
    segments = bones(joints, ends)
    distances = np.stack([_segment_distance(vertices, a, b)[0] for _, a, b, _ in segments], axis=1)
    radii = np.array([r for *_, r in segments])
    owners = np.array([o for o, *_ in segments])
    nearest = np.argmin(distances, axis=1)
    d_min = distances[np.arange(len(vertices)), nearest]
    r_near = radii[nearest]
    band = distances <= (d_min + 1.5 * r_near)[:, None]
    bone_weight = np.where(band, np.exp(-(distances - d_min[:, None]) / (0.5 * r_near[:, None])), 0.0)
    weights = np.zeros((len(vertices), skeleton.NUM_JOINTS))
    for k, owner in enumerate(owners):
        weights[:, owner] += bone_weight[:, k]
    weights /= weights.sum(axis=1, keepdims=True)
    # smoothing over the mesh neighbours (never across a gap: fingers stay apart)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])
    degree = np.bincount(edges[:, 0], minlength=len(vertices)).astype(np.float64)
    for _ in range(smoothing):
        summed = np.zeros_like(weights)
        np.add.at(summed, edges[:, 0], weights[edges[:, 1]])
        weights = 0.5 * weights + 0.5 * summed / np.maximum(degree, 1.0)[:, None]
    order = np.argsort(-weights, axis=1)
    keep = np.zeros_like(weights, dtype=bool)
    np.put_along_axis(keep, order[:, :influences], True, axis=1)
    weights = np.where(keep, weights, 0.0)
    weights[weights < 1e-4] = 0.0
    return weights / weights.sum(axis=1, keepdims=True)


def joint_regressor(vertices: np.ndarray, joints: np.ndarray, neighbours: int = 16) -> np.ndarray:
    """(55, V): each joint as the minimum-norm affine combination of its
    nearest vertices (exact: J_regressor @ vertices == joints)."""
    regressor = np.zeros((len(joints), len(vertices)))
    for index, joint in enumerate(joints):
        order = np.argsort(np.linalg.norm(vertices - joint, axis=1))[:neighbours]
        a = np.vstack([vertices[order].T, np.ones(len(order))])            # (4, K)
        b = np.append(joint, 1.0)
        w = a.T @ np.linalg.solve(a @ a.T + 1e-12 * np.eye(4), b)
        regressor[index, order] = w
    return regressor


def shape_directions(vertices: np.ndarray, joints: np.ndarray, ends: dict) -> np.ndarray:
    """(V, 3, 10) shape directions; one unit of a coefficient is roughly one
    standard deviation of adult variation."""
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    dirs = np.zeros(vertices.shape + (10,))
    hands = np.abs(x) > 0.70
    head = (y > 1.55) & (np.abs(x) < 0.13)
    dirs[:, :, 0] = 0.05 * vertices                                          # stature (about the feet)
    segments = bones(joints, ends)
    distance = np.full(len(vertices), np.inf)
    radial = np.zeros_like(vertices)
    for _, a, b, _ in segments:
        d, closest = _segment_distance(vertices, a, b)
        better = d < distance
        distance[better] = d[better]
        radial[better] = vertices[better] - closest[better]
    girth = np.where(hands, 0.0, np.where(head, 0.3, 1.0))
    dirs[:, :, 1] = 0.15 * radial * girth[:, None]                           # girth
    legs = y < 0.88
    dirs[legs, 1, 2] = -0.05 * (0.88 - y[legs])                              # leg length
    arms = np.abs(x) > 0.185
    dirs[arms, 0, 3] = np.sign(x[arms]) * 0.05 * (np.abs(x[arms]) - 0.185)   # arm length
    ramp = np.clip((np.abs(x) - 0.10) / 0.085, 0.0, 1.0) * (y > 1.25)
    dirs[:, 0, 4] = np.sign(x) * 0.02 * ramp                                 # shoulder width
    belly = (z > 0) & (y > 0.95) & (y < 1.25) & (np.abs(x) < 0.17)
    dirs[belly, 2, 5] = 0.04 * np.sin(np.pi * (y[belly] - 0.95) / 0.30) * z[belly] / 0.10
    hips = (y > 0.75) & (y < 1.0) & (np.abs(x) < 0.2)
    dirs[hips, 0, 6] = 0.03 * x[hips] / 0.17                                 # hip width
    chest = (y > 1.2) & (y < 1.42) & (np.abs(x) < 0.18)
    dirs[chest, 2, 7] = 0.03 * z[chest] / 0.11                               # chest depth
    dirs[head, :, 8] = 0.06 * (vertices[head] - np.array([0.0, 1.68, 0.01]))  # head size
    neck = (y > 1.5) & (np.abs(x) < 0.15)
    dirs[neck, 1, 9] = 0.02                                                  # neck length
    return dirs


def relaxed_hands() -> tuple[np.ndarray, np.ndarray]:
    """Mean hand pose (45,) per side: every finger joint bent by 14 deg
    (the bend axis is -z for the left hand, +z for the right, palm down)."""
    left = np.zeros((15, 3))
    right = np.zeros((15, 3))
    bend = np.radians(14.0)
    for k in range(15):
        thumb = k >= 12
        left[k, 2] = -bend * (0.5 if thumb else 1.0)
        right[k, 2] = bend * (0.5 if thumb else 1.0)
    return left.reshape(-1), right.reshape(-1)


def build(path, voxel: float = 0.004, target_triangles: int = 20000) -> Path:
    """Write the test body as an SMPL-X style npz; returns the path."""
    path = Path(path)
    joints, ends = rest_joints()
    vertices, faces = surface(voxel, target_triangles)
    weights = skinning_weights(vertices, faces, joints, ends)
    regressor = joint_regressor(vertices, joints)
    shape = shape_directions(vertices, joints, ends)
    shapedirs = np.concatenate([shape, np.zeros_like(shape)], axis=2)                  # 10 shape + 10 expression
    posedirs = np.zeros((len(vertices), 3, 9 * (skeleton.NUM_JOINTS - 1)), dtype=np.float32)
    kintree = np.array([skeleton.PARENTS, list(range(skeleton.NUM_JOINTS))], dtype=np.int64)
    left, right = relaxed_hands()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, v_template=vertices.astype(np.float64), f=faces.astype(np.int64),
                        shapedirs=shapedirs.astype(np.float64), posedirs=posedirs,
                        J_regressor=regressor.astype(np.float64), weights=weights.astype(np.float64),
                        kintree_table=kintree, hands_meanl=left, hands_meanr=right,
                        hands_componentsl=np.eye(45), hands_componentsr=np.eye(45),
                        gender=np.array("neutral"), joint_names=np.array(skeleton.JOINT_NAMES),
                        testbody=np.array(True))
    return path


def cached(folder=None) -> Path:
    """The test body in a cache folder (built on first use, about 10 s). The
    file name carries a checksum of this module, so a changed body is rebuilt."""
    import hashlib
    import tempfile
    folder = Path(folder) if folder is not None else Path(tempfile.gettempdir()) / "bodyscan_cache"
    version = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:10]
    path = folder / f"testbody_smplx_{version}.npz"
    if not path.exists():
        temporary = folder / f"testbody_smplx_{np.random.default_rng().integers(1 << 30)}.npz"
        build(temporary)
        temporary.replace(path)
    return path
