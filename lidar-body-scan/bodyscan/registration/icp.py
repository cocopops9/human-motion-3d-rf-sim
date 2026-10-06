"""Point-to-plane ICP variants with robust (Tukey) weights.

    icp_turn_about   ONE unknown: the turn about a known vertical axis (turntable)
    icp_robust       turn about the vertical + 3D shift (4 DOF), optionally with a bounded tilt (6 DOF)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from bodyscan.geometry import Target, evaluate, tilt_degrees, transform_points, yaw_transform


@dataclass
class Pair:
    """Registration of 'source' onto 'target' (indices of clouds)."""
    source: int
    target: int
    transform: np.ndarray
    fitness: float
    rmse: float
    uncertain: bool
    information: np.ndarray | None = field(default=None, repr=False)


def tukey_weights(residual: np.ndarray, distance: float) -> np.ndarray:
    return np.where(np.abs(residual) < distance, (1.0 - (residual / distance) ** 2) ** 2, 0.0)


def icp_turn_about(source: np.ndarray, target: Target, angle: float, pivot: np.ndarray,
                   distances=(0.05, 0.03, 0.015), iterations=(25, 20, 15)):
    """Point-to-plane ICP with ONE unknown: the turn about the vertical line through 'pivot'.

    On the platform the body does not translate, so fixing the axis removes the
    ambiguity between a small turn and a sideways shift of a partial view.
    Returns (angle [rad], overlap, rmse)."""
    distance = distances[-1]
    for distance, count in zip(distances, iterations):
        for _ in range(count):
            moved = transform_points(yaw_transform(angle, pivot), source)
            index, gap = target.nearest(moved)
            use = gap < distance
            if use.sum() < 30:
                return angle, 0.0, float("inf")
            p, q, n = moved[use], target.points[index[use]], target.normals[index[use]]
            residual = np.einsum("ij,ij->i", n, p - q)
            weight = tukey_weights(residual, distance)
            lever = p - pivot
            jacobian = n[:, 1] * lever[:, 0] - n[:, 0] * lever[:, 1]
            step = -np.sum(weight * jacobian * residual) / (np.sum(weight * jacobian ** 2) + 1e-12)
            angle += float(np.clip(step, -0.05, 0.05))
            if abs(step) < 1e-6:
                break
    fitness, rmse = evaluate(source, target, yaw_transform(angle, pivot), distance)
    return angle, fitness, rmse


def icp_robust(source: np.ndarray, target: Target, initial: np.ndarray, distances=(0.06, 0.035, 0.02),
               iterations=(30, 25, 20), max_tilt_deg: float = 0.0) -> np.ndarray:
    """Point-to-plane ICP with a Tukey kernel, 4 or 6 degrees of freedom.

    max_tilt_deg = 0: turn about the vertical and a 3D shift (4 DOF).
    max_tilt_deg > 0: the body may also tilt, up to that angle in total. A
    standing person sways like an inverted pendulum about the ankles; at head
    height 1 degree of lean is 3 cm, far more than the sensor noise, so a
    registration without tilt cannot align head and feet at the same time.
    The robust kernel down-weights points whose residual is large, i.e.
    limbs that moved."""
    transform = initial.copy()
    max_turn = np.radians(3.0)
    free_tilt = max_tilt_deg > 0
    for distance, count in zip(distances, iterations):
        for _ in range(count):
            moved = transform_points(transform, source)
            index, gap = target.nearest(moved)
            use = gap < distance
            if use.sum() < max(30, 0.1 * len(source)):
                break                     # lost the surface: a wrong start, stop here
            p, q, n = moved[use], target.points[index[use]], target.normals[index[use]]
            residual = np.einsum("ij,ij->i", n, p - q)
            weight = tukey_weights(residual, distance)
            center = p.mean(axis=0)
            lever = p - center
            # d/dw of n . (w x lever) = (lever x n) . w ; columns: w_x, w_y, w_z, t_x, t_y, t_z
            jacobian = np.column_stack([np.cross(lever, n), n])
            columns = [0, 1, 2, 3, 4, 5] if free_tilt else [2, 3, 4, 5]
            reduced = jacobian[:, columns]
            normal_matrix = reduced.T @ (reduced * weight[:, None]) + 1e-9 * np.eye(len(columns))
            solution = np.linalg.solve(normal_matrix, -reduced.T @ (weight * residual))
            step = np.zeros(6)
            step[columns] = solution
            # Trust region: a poorly constrained yaw (a side view is almost a
            # cylinder) must not jump across the body in one iteration.
            step[:3] = np.clip(step[:3], -max_turn, max_turn)
            length = np.linalg.norm(step[3:])
            if length > 0.5 * distance:
                step[3:] *= 0.5 * distance / length
            update = np.eye(4)
            update[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(step[:3])
            update[:3, 3] = center - update[:3, :3] @ center + step[3:]
            candidate = update @ transform
            if free_tilt and tilt_degrees(candidate) > max_tilt_deg:
                step[:2] = 0.0            # tilt bound reached: continue without tilting further
                update[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(step[:3])
                update[:3, 3] = center - update[:3, :3] @ center + step[3:]
                candidate = update @ transform
            transform = candidate
            if np.linalg.norm(step[:3]) < 1e-5 and np.linalg.norm(step[3:]) < 1e-5:
                break
    return transform


def icp_4dof(source, target, initial, distances=(0.06, 0.035, 0.02), iterations=(30, 25, 20)) -> np.ndarray:
    return icp_robust(source, target, initial, distances, iterations, 0.0)


def information_matrix(source_cloud, target_cloud, distance, transform) -> np.ndarray:
    return o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source_cloud, target_cloud, distance, transform)
