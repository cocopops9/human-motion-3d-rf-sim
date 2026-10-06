"""Check that the LiDAR sees the whole person on the turntable (feet AND head).

Run it on a SHORT capture made right after moving or tilting the sensor, with
the person standing on the platform, for example:

    python capture_turntable.py --out C:\\lidar\\check1 --duration 20
    python check_view.py C:\\lidar\\check1 --person-height 1.81

It reports:
    - sensor height and tilt (from the floor in the background frames),
    - platform centre and its distance from the sensor,
    - the highest and lowest beam towards the platform, and the heights they
      reach at the near side of the person,
    - the height range of the person actually seen,
    - how many degrees to tilt the sensor up or down so that both the feet and
      the top of the head are inside the field of view, with margin.

Version 2026-10-02a. Needs fuse_turntable.py in the same folder.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

import fuse_turntable as ft


def median_range(run, paths):
    stack = np.stack([run.load(p)[0] for p in paths])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # pixels with no return in any frame
        return np.nanmedian(stack, axis=0)


def to_world(transform, points):
    return points @ transform[:3, :3].T + transform[:3, 3]


def beam_limits(run, transform, center, half_width_deg=10.0):
    """Elevation of the highest and lowest beam within +-half_width_deg of
    azimuth around the direction of the platform centre (floor frame)."""
    directions = run.direction.reshape(-1, 3) @ transform[:3, :3].T
    horizontal = np.hypot(directions[:, 0], directions[:, 1])
    azimuth = np.degrees(np.arctan2(directions[:, 1], directions[:, 0]))
    elevation = np.degrees(np.arctan2(directions[:, 2], horizontal))
    target = np.degrees(np.arctan2(center[1], center[0]))
    near = np.abs((azimuth - target + 180.0) % 360.0 - 180.0) < half_width_deg
    return float(elevation[near].max()), float(elevation[near].min())


def main():
    parser = argparse.ArgumentParser(description="Check the field of view on the person before a long capture.")
    parser.add_argument("run", help="capture_turntable.py output directory (short capture, person on the platform)")
    parser.add_argument("--person-height", type=float, default=1.85, help="height of the person [m]")
    parser.add_argument("--center", type=float, nargs=2, default=None, metavar=("X", "Y"),
                        help="platform centre in the floor frame, if it is not found automatically")
    parser.add_argument("--frames", type=int, default=20, help="frames used for the person")
    parser.add_argument("--margin", type=float, default=3.0, help="wanted margin at both ends [deg]")
    parser.add_argument("--min-range", type=float, default=0.3)
    parser.add_argument("--max-range", type=float, default=10.0)
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not (run_dir / "background").exists():
        (run_dir / "background").mkdir()
    run = ft.Run.__new__(ft.Run)
    run.directory = run_dir
    lut = np.load(run_dir / "lut.npz")
    run.direction = lut["direction"].astype(np.float64)
    run.offset = lut["offset"].astype(np.float64)
    run.args = args
    run.background_paths = sorted((run_dir / "background").glob("*.npz"), key=ft.natural_key)
    run.frame_paths = sorted((run_dir / "frames").glob("*.npz"), key=ft.natural_key)
    if not run.frame_paths:
        sys.exit(f"no frames in {run_dir / 'frames'}")

    # 1. Floor (background if present, else the frames themselves: the floor is visible in both).
    floor_paths = run.background_paths or run.frame_paths[:10]
    scene = run.xyz(median_range(run, floor_paths))
    scene = scene[np.isfinite(scene[..., 0])]
    transform, height, tilt = ft.fit_floor(scene)
    world = to_world(transform, scene)
    print(f"sensor: {height:.3f} m above the floor, z axis tilted {tilt:.1f} deg from the vertical")

    # 2. Platform centre.
    if args.center is not None:
        center = np.array(args.center, dtype=np.float64)
        print(f"platform centre (given): ({center[0]:.3f}, {center[1]:.3f}) m")
    else:
        ring = None
        for start in ([1.1, 0.0], ft.PLATFORM_CENTER, [1.4, 0.0], [0.9, 0.0]):
            ring = ft.find_platform(world, np.array(start, dtype=np.float64), args, search=0.8)
            if ring is not None:
                break
        if ring is None:
            sys.exit("platform ring not found; pass --center X Y")
        center = np.asarray(ring[0])
        print(f"platform ring: centre ({center[0]:.3f}, {center[1]:.3f}) m, radius {ring[1]:.3f} m, "
              f"rms {1000 * ring[2]:.0f} mm, {ring[3]} points")
    distance = float(np.hypot(*center))
    print(f"platform centre {distance:.2f} m from the sensor (horizontal)")

    # 3. Beams towards the platform.
    top, bottom = beam_limits(run, transform, center)
    feet_distance = distance - 0.15          # front of the feet when they face the sensor
    head_distance = distance - 0.10          # front of the head
    feet_height = 0.03
    need_bottom = -np.degrees(np.arctan2(height - feet_height, feet_distance))
    need_top = np.degrees(np.arctan2(args.person_height - height, head_distance))
    print(f"towards the platform: highest beam {top:+.1f} deg, lowest beam {bottom:+.1f} deg")
    print(f"  highest beam reaches {height + np.tan(np.radians(top)) * head_distance:.2f} m at the front of the head "
          f"({head_distance:.2f} m); needed {args.person_height:.2f} m")
    print(f"  lowest beam reaches {height + np.tan(np.radians(bottom)) * feet_distance:.2f} m at the front of the feet "
          f"({feet_distance:.2f} m); needed {feet_height:.2f} m")

    # 4. Person actually seen.
    picks = np.linspace(0, len(run.frame_paths) - 1, min(args.frames, len(run.frame_paths))).astype(int)
    tops, lows = [], []
    for k in picks:
        points = run.xyz(run.load(run.frame_paths[k])[0])
        points = to_world(transform, points[np.isfinite(points[..., 0])])
        on = (np.hypot(*(points[:, :2] - center).T) < 0.5) & (points[:, 2] > 0.02) & (points[:, 2] < 2.5)
        if on.sum() > 200:
            tops.append(np.percentile(points[on, 2], 99.9))
            lows.append(np.percentile(points[on, 2], 0.1))
    if tops:
        print(f"person seen from {np.median(lows):.2f} m to {np.max(tops):.2f} m "
              f"(over {len(tops)} frames)")

    # 5. Advice: rotate the beam fan by delta (positive = tilt the sensor so it looks more upward).
    low_delta = need_top + args.margin - top           # smallest upward rotation that shows the head
    high_delta = need_bottom - args.margin - bottom     # largest upward rotation that still shows the feet
    print()
    if low_delta > high_delta:
        print(f"the person does not fit in the field of view with {args.margin:g} deg margins at this "
              f"distance and height: move the sensor back or raise it")
    elif low_delta <= 0 <= high_delta:
        print(f"OK: head and feet are both inside the field of view (margins "
              f"{top - need_top:.1f} deg at the head, {need_bottom - bottom:.1f} deg at the feet)")
    else:
        delta = 0.5 * (low_delta + high_delta)
        direction = "UP (less downward tilt)" if delta > 0 else "DOWN"
        print(f"tilt the sensor about {abs(delta):.0f} deg {direction}; any value between "
              f"{min(abs(low_delta), abs(high_delta)):.0f} and {max(abs(low_delta), abs(high_delta)):.0f} deg works")


if __name__ == "__main__":
    main()
