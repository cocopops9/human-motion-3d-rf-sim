"""Rigid transforms, mostly turns about a vertical axis (4x4 homogeneous matrices)."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def yaw_transform(angle: float, pivot, shift=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Rotate by 'angle' [rad] about the vertical line through 'pivot', then shift."""
    c, s = np.cos(angle), np.sin(angle)
    transform = np.eye(4)
    transform[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    pivot = np.asarray(pivot, dtype=np.float64)
    transform[:3, 3] = pivot - transform[:3, :3] @ pivot + np.asarray(shift, dtype=np.float64)
    return transform


def yaw_of(transform: np.ndarray) -> float:
    """Turn about the vertical [rad] contained in a transform."""
    return float(np.arctan2(transform[1, 0], transform[0, 0]))


def tilt_degrees(transform: np.ndarray) -> float:
    """Angle between the transformed vertical and the vertical."""
    return float(np.degrees(np.arccos(np.clip(transform[2, 2], -1.0, 1.0))))


def project_to_4dof(transform: np.ndarray) -> np.ndarray:
    """Drop any residual tilt: keep the yaw and the translation."""
    result = yaw_transform(yaw_of(transform), (0.0, 0.0, 0.0))
    result[:3, 3] = transform[:3, 3]
    return result


def rotation_to_z(normal: np.ndarray) -> np.ndarray:
    """Smallest rotation taking the unit vector 'normal' to +z."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, z)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.eye(3) if normal @ z > 0 else np.diag([1.0, -1.0, -1.0])
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis / s * np.arctan2(s, normal @ z))


def wrapped_degrees(angle_rad) -> np.ndarray:
    """Angle [rad] to degrees in [-180, 180)."""
    return (np.degrees(angle_rad) + 180.0) % 360.0 - 180.0


def turn_points(points: np.ndarray, angles, pivot) -> np.ndarray:
    """Turn every point by its own angle [rad] about the vertical through 'pivot'
    (angles: scalar or one per point)."""
    pivot = np.asarray(pivot, dtype=np.float64)
    c, s = np.cos(angles), np.sin(angles)
    relative = points - pivot
    turned = np.column_stack([c * relative[:, 0] - s * relative[:, 1], s * relative[:, 0] + c * relative[:, 1],
                              relative[:, 2]])
    return turned + pivot


def turn_vectors(vectors: np.ndarray, angles) -> np.ndarray:
    """Turn direction vectors (normals) about the vertical, one angle per vector or one for all."""
    c, s = np.cos(angles), np.sin(angles)
    return np.column_stack([c * vectors[:, 0] - s * vectors[:, 1], s * vectors[:, 0] + c * vectors[:, 1],
                            vectors[:, 2]])
