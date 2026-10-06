"""Views: frames turned back by the platform angle into the frame of the body."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from bodyscan.geometry import turn_points, turn_vectors, yaw_transform
from bodyscan.log import Progress


def select_view_frames(candidates, frame_angles, sense, step_deg) -> list[int]:
    """One frame every 'step_deg' degrees of turn, in time order."""
    view_frames, last = [], None
    for k in candidates:
        a = np.degrees(frame_angles[k]) * sense
        if last is None or abs(a - last) >= step_deg:
            view_frames.append(int(k))
            last = a
    return view_frames


def reference_groups(count: int, groups: int):
    """Group label per view for the leave-group-out references: exact leave
    one out for few views, otherwise views spread over 'groups' groups at
    random (a fixed seed), so that no group misses one direction or one lap."""
    if groups <= 0 or count <= groups:
        return None
    return np.random.default_rng(0).permutation(count) % groups


def build_views(source, isolator, view_frames, frame_angles, axis, times, time_origin) -> list:
    """Person cloud of every view frame, turned back by the platform angle
    about the axis: per point, at the time of its column (sensor timestamps),
    so that the turn during the frame is undone too; per frame without them.
    'times' must be strictly increasing (they are interpolated)."""
    pivot = np.append(axis, 0.0)
    clouds = []
    progress = Progress("views", len(view_frames))
    for n, k in enumerate(view_frames):
        frame = source.load(k)
        isolated = isolator.cloud(frame.range_m, frame.timestamps)
        cloud, point_times = isolated.cloud, isolated.point_times
        if point_times is None or len(cloud.points) == 0:
            cloud.transform(yaw_transform(-frame_angles[k], pivot))
        else:
            angle = -np.interp(point_times - time_origin, times, frame_angles)
            cloud.points = o3d.utility.Vector3dVector(turn_points(np.asarray(cloud.points), angle, pivot))
            cloud.normals = o3d.utility.Vector3dVector(turn_vectors(np.asarray(cloud.normals), angle))
        clouds.append(cloud)
        if (n + 1) % 50 == 0 or n == len(view_frames) - 1:
            progress.step(n + 1)
    return clouds
