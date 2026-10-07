"""Rotations: axis-angle, matrices and quaternions, in numpy and in PyTorch.

Conventions: an axis-angle vector is the rotation axis times the angle in
radians (SMPL-X stores every joint rotation this way); quaternions are
(w, x, y, z) with w >= 0 after normalisation where a sign has to be chosen.

The numpy functions serve the simulation, the export and the evaluation (no
gradient needed); the PyTorch functions serve the optimisation. PyTorch is
imported only by the functions that use it, so the numpy part works on a PC
without PyTorch.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# numpy
# ----------------------------------------------------------------------------


def axis_angle_to_matrix_np(axis_angle: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrices (Rodrigues)."""
    a = np.asarray(axis_angle, dtype=np.float64)
    angle = np.linalg.norm(a, axis=-1, keepdims=True)
    small = angle < 1e-8
    axis = np.where(small, np.array([1.0, 0.0, 0.0]), a / np.where(small, 1.0, angle))
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c, s = np.cos(angle[..., 0]), np.sin(angle[..., 0])
    t = 1.0 - c
    matrix = np.stack([
        np.stack([c + x * x * t, x * y * t - z * s, x * z * t + y * s], axis=-1),
        np.stack([y * x * t + z * s, c + y * y * t, y * z * t - x * s], axis=-1),
        np.stack([z * x * t - y * s, z * y * t + x * s, c + z * z * t], axis=-1)], axis=-2)
    return matrix


def matrix_to_quaternion_np(matrix: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) unit quaternions (w, x, y, z), w >= 0 (Shepperd's method)."""
    m = np.asarray(matrix, dtype=np.float64)
    shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    candidates = np.stack([trace, m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], axis=1)
    choice = np.argmax(candidates, axis=1)
    q = np.empty((len(m), 4))
    for k in range(4):
        rows = choice == k
        if not rows.any():
            continue
        r = m[rows]
        if k == 0:
            s = np.sqrt(1.0 + trace[rows]) * 2.0
            q[rows] = np.stack([0.25 * s, (r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s,
                                (r[:, 1, 0] - r[:, 0, 1]) / s], axis=1)
        elif k == 1:
            s = np.sqrt(1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2]) * 2.0
            q[rows] = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / s, 0.25 * s, (r[:, 0, 1] + r[:, 1, 0]) / s,
                                (r[:, 0, 2] + r[:, 2, 0]) / s], axis=1)
        elif k == 2:
            s = np.sqrt(1.0 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2]) * 2.0
            q[rows] = np.stack([(r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 0, 1] + r[:, 1, 0]) / s, 0.25 * s,
                                (r[:, 1, 2] + r[:, 2, 1]) / s], axis=1)
        else:
            s = np.sqrt(1.0 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1]) * 2.0
            q[rows] = np.stack([(r[:, 1, 0] - r[:, 0, 1]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s,
                                (r[:, 1, 2] + r[:, 2, 1]) / s, 0.25 * s], axis=1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    return q.reshape(shape + (4,))


def quaternion_to_matrix_np(quaternion: np.ndarray) -> np.ndarray:
    """(..., 4) (w, x, y, z) -> (..., 3, 3); the quaternion need not be normalised."""
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1)], axis=-2)


def quaternion_to_axis_angle_np(quaternion: np.ndarray) -> np.ndarray:
    """(..., 4) -> (..., 3) axis-angle with angle in [0, pi]."""
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = q * np.where(q[..., :1] < 0, -1.0, 1.0)
    vector = q[..., 1:]
    sine = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sine, q[..., :1])
    scale = np.where(sine > 1e-12, angle / np.where(sine > 1e-12, sine, 1.0), 2.0)
    return vector * scale


def axis_angle_to_quaternion_np(axis_angle: np.ndarray) -> np.ndarray:
    a = np.asarray(axis_angle, dtype=np.float64)
    angle = np.linalg.norm(a, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.where(angle > 1e-12, np.sin(half) / np.where(angle > 1e-12, angle, 1.0), 0.5)
    return np.concatenate([np.cos(half), a * scale], axis=-1)


def matrix_to_axis_angle_np(matrix: np.ndarray) -> np.ndarray:
    return quaternion_to_axis_angle_np(matrix_to_quaternion_np(matrix))


def align_quaternion_signs(quaternions: np.ndarray, axis: int = 0) -> np.ndarray:
    """Flip signs along 'axis' (time) so that consecutive quaternions have a
    positive dot product: q and -q are the same rotation, and interpolation
    between them must take the short way."""
    q = np.array(quaternions, dtype=np.float64, copy=True)
    q = np.moveaxis(q, axis, 0)
    for k in range(1, len(q)):
        flip = np.sum(q[k] * q[k - 1], axis=-1, keepdims=True) < 0
        q[k] = np.where(flip, -q[k], q[k])
    return np.moveaxis(q, 0, axis)


def catmull_rom_weights(times: np.ndarray, query: np.ndarray):
    """Indices and cubic Hermite weights to evaluate samples given at 'times'
    (increasing, not necessarily uniform) at 'query' times, with Catmull-Rom
    tangents (finite differences of the neighbours). The result is C1:
    positions and velocities are continuous, which keeps the Doppler shift of
    an animated mesh free of steps at the original frame boundaries. Outside
    the sample range the ends are held. Returns (index (Q, 4), weight (Q, 4))
    such that value(query) = sum_k weight[:, k] * samples[index[:, k]]."""
    t = np.asarray(times, dtype=np.float64)
    s = np.clip(np.asarray(query, dtype=np.float64), t[0], t[-1])
    n = len(t)
    if n == 1:
        return np.zeros((len(s), 4), dtype=np.int64), np.tile([0.0, 1.0, 0.0, 0.0], (len(s), 1))
    i = np.clip(np.searchsorted(t, s, side="right") - 1, 0, n - 2)
    i0, i1, i2, i3 = np.clip(i - 1, 0, n - 1), i, i + 1, np.clip(i + 2, 0, n - 1)
    h = t[i2] - t[i1]
    u = (s - t[i1]) / h
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    # tangents m1 = (p2 - p0) / (t2 - t0), m2 = (p3 - p1) / (t3 - t1) (one-sided at the ends)
    d1 = np.where(i0 == i1, t[i2] - t[i1], t[i2] - t[i0])
    d2 = np.where(i3 == i2, t[i2] - t[i1], t[i3] - t[i1])
    w = np.zeros((len(s), 4))
    # value = h00 p1 + h10 h m1 + h01 p2 + h11 h m2
    a1 = h10 * h / d1                      # coefficient of (p2 - p0) or (p2 - p1)
    a2 = h11 * h / d2                      # coefficient of (p3 - p1) or (p2 - p1)
    w[:, 1] += h00
    w[:, 2] += h01
    # m1 contribution
    end1 = i0 == i1
    w[:, 2] += a1
    w[:, 0] += np.where(end1, 0.0, -a1)
    w[:, 1] += np.where(end1, -a1, 0.0)
    # m2 contribution
    end2 = i3 == i2
    w[:, 3] += np.where(end2, 0.0, a2)
    w[:, 2] += np.where(end2, a2, 0.0)
    w[:, 1] -= a2
    index = np.stack([i0, i1, i2, i3], axis=1)
    return index, w


def interpolate_positions(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """C1 cubic interpolation of (T, ...) samples at the query times."""
    index, weight = catmull_rom_weights(times, query)
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(len(values), -1)
    result = np.einsum("qk,qkc->qc", weight, flat[index])
    return result.reshape((len(index),) + values.shape[1:])


def interpolate_rotations(times: np.ndarray, axis_angles: np.ndarray, query: np.ndarray) -> np.ndarray:
    """C1 interpolation of (T, ..., 3) axis-angle rotations: cubic interpolation
    of sign-aligned quaternions, renormalised. For frames 50 ms apart and joint
    speeds of a few rad/s the result is within a fraction of a degree of the
    spherical (SQUAD) interpolation, and its angular velocity is continuous."""
    q = align_quaternion_signs(axis_angle_to_quaternion_np(axis_angles), axis=0)
    result = interpolate_positions(times, q, query)
    return quaternion_to_axis_angle_np(result)


def slerp_np(q0: np.ndarray, q1: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Spherical linear interpolation between unit quaternions (..., 4) at fraction u (...)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    u = np.asarray(u, dtype=np.float64)[..., None]
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sine = np.sin(theta)
    near = sine < 1e-6
    w0 = np.where(near, 1.0 - u, np.sin((1.0 - u) * theta) / np.where(near, 1.0, sine))
    w1 = np.where(near, u, np.sin(u * theta) / np.where(near, 1.0, sine))
    result = w0 * q0 + w1 * q1
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def rotation_z_np(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of_matrix_np(matrix: np.ndarray) -> np.ndarray:
    """Heading (radians) of the x axis of rotation matrices (..., 3, 3), projected on the floor."""
    m = np.asarray(matrix)
    return np.arctan2(m[..., 1, 0], m[..., 0, 0])


# ----------------------------------------------------------------------------
# PyTorch
# ----------------------------------------------------------------------------


def axis_angle_to_matrix(axis_angle):
    """(..., 3) tensor -> (..., 3, 3), differentiable everywhere (also at zero)."""
    import torch
    a = axis_angle
    angle_sq = (a * a).sum(-1, keepdim=True)
    angle = torch.sqrt(angle_sq + 1e-16)
    # sin(t)/t and (1 - cos t)/t^2 with series near zero
    small = angle_sq < 1e-8
    sin_over = torch.where(small, 1.0 - angle_sq / 6.0, torch.sin(angle) / angle)
    cos_term = torch.where(small, 0.5 - angle_sq / 24.0, (1.0 - torch.cos(angle)) / (angle_sq + 1e-30))
    x, y, z = a[..., 0:1], a[..., 1:2], a[..., 2:3]
    zero = torch.zeros_like(x)
    k = torch.cat([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(a.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=a.dtype, device=a.device).expand(a.shape[:-1] + (3, 3))
    return eye + sin_over[..., None] * k + cos_term[..., None] * (k @ k)


def matrix_to_axis_angle(matrix):
    """(..., 3, 3) tensor -> (..., 3); not used inside the optimisation (no gradient through it)."""
    import torch
    q = matrix_to_axis_angle_np(matrix.detach().cpu().numpy())
    return torch.as_tensor(q, dtype=matrix.dtype, device=matrix.device)


def rotation_6d_to_matrix(d6):
    """Continuous 6D representation (Zhou et al. 2019) -> rotation matrix (Gram-Schmidt)."""
    import torch
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rotation_6d(matrix):
    return matrix[..., :, :2].transpose(-1, -2).reshape(matrix.shape[:-2] + (6,))
