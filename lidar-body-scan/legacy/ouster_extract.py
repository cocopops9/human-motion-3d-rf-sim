"""
Extract point clouds from an Ouster sensor (live) or from a recording (pcap/osf).

Tested with ouster-sdk 1.0.1 on a synthetic recording. Also written to work
with the 0.15 / 0.16 API. Versions older than 0.11 used `ouster.client`
instead of `ouster.sdk`; upgrade the SDK if imports fail.

Usage examples
--------------
Live sensor, save 50 frames:
    python ouster_extract.py --source os-122542000054.local --frames 50 --out frames

Recorded file (metadata json must sit next to the pcap, or be passed explicitly):
    python ouster_extract.py --source capture.pcap --meta capture.json --out frames

Accumulate a static scene over 100 frames into one averaged cloud:
    python ouster_extract.py --source capture.pcap --frames 100 --accumulate --out static

Output per frame
----------------
    frame_XXXXX.npz : xyz (H, W, 3) in meters, range (H, W) in mm,
                      reflectivity, signal, near_ir (H, W), timestamps (W,)
    frame_XXXXX.ply : valid points only, colored by reflectivity (if --ply)
"""

import argparse
from pathlib import Path

import numpy as np

from ouster.sdk import core, open_source


# ----------------------------------------------------------------------------
# Source handling
# ----------------------------------------------------------------------------

RECORDING_SUFFIXES = (".pcap", ".osf", ".bag", ".mcap")


def is_live_sensor(source_path):
    """A hostname or IP address, as opposed to a recording on disk."""
    path = Path(source_path)
    return not path.exists() and path.suffix.lower() not in RECORDING_SUFFIXES


def open_scan_source(source_path, meta_path=None, auto_udp_dest=False):
    """Open a live sensor or a recorded file and return (source, sensor_info).

    For a live sensor the UDP destination stored in the sensor (set once with
    setup_ip.py) is used as it is. With auto_udp_dest=True the SDK instead
    overwrites it with the address it detects itself, which on this PC is an
    IPv6 link-local address the sensor cannot reach.
    """
    if meta_path is not None:
        source = open_source(source_path, meta=[meta_path])
    elif is_live_sensor(source_path) and not auto_udp_dest:
        source = open_source(source_path, no_auto_udp_dest=True)
    else:
        source = open_source(source_path)

    # The attribute name changed across SDK versions.
    if hasattr(source, "sensor_info"):
        info = source.sensor_info[0]
    else:
        info = source.metadata[0]

    return source, info


def iterate_scans(source):
    """Yield single LidarScan (LidarFrame in SDK >= 1.0) objects.

    Depending on the SDK version, iterating a source yields one LidarScan,
    a list with one LidarScan per sensor (0.11 to 0.16), or a FrameSet
    (1.0 and later). We only have one sensor, so we take the first element.
    """
    for item in source:
        is_container = isinstance(item, (list, tuple)) or type(item).__name__ == "FrameSet"
        if is_container:
            scan = item[0] if len(item) > 0 else None
        else:
            scan = item
        if scan is not None:
            yield scan


# ----------------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------------

def read_field(scan, info, field_name):
    """Return a destaggered field as a (H, W) array, or None if absent."""
    field = getattr(core.ChanField, field_name, None)
    if field is None or not scan.has_field(field):
        return None
    return core.destagger(info, scan.field(field))


def scan_to_arrays(scan, info, xyz_lut):
    """Convert one LidarScan into destaggered numpy arrays."""
    xyz = core.destagger(info, xyz_lut(scan))            # (H, W, 3), meters
    arrays = {
        "xyz": xyz.astype(np.float32),
        "range": read_field(scan, info, "RANGE"),         # millimeters
        "reflectivity": read_field(scan, info, "REFLECTIVITY"),
        "signal": read_field(scan, info, "SIGNAL"),
        "near_ir": read_field(scan, info, "NEAR_IR"),
        "timestamps": np.asarray(scan.timestamp),         # per column, ns
        "frame_id": np.int64(scan.frame_id),
    }
    return {key: value for key, value in arrays.items() if value is not None}


def valid_mask(range_mm, min_range_m, max_range_m):
    """Points with a real return inside the chosen range window."""
    return (range_mm > min_range_m * 1000.0) & (range_mm < max_range_m * 1000.0)


def check_column_completeness(scan):
    """Fraction of valid columns; below 1.0 means lost UDP packets."""
    status = np.asarray(scan.status)
    return float(np.count_nonzero(status & 0x1)) / status.size


# ----------------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------------

def save_ply(path, points, intensity=None):
    """Write a point cloud to PLY using Open3D (optional dependency)."""
    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    if intensity is not None:
        normalized = intensity.astype(np.float64)
        normalized = normalized / max(normalized.max(), 1.0)
        cloud.colors = o3d.utility.Vector3dVector(np.repeat(normalized[:, None], 3, axis=1))
    o3d.io.write_point_cloud(str(path), cloud)


def save_metadata(info, out_dir):
    """Keep the sensor metadata with the data; recordings are useless without it."""
    text = info.to_json_string() if hasattr(info, "to_json_string") else info.updated_metadata_string()
    (out_dir / "metadata.json").write_text(text)


# ----------------------------------------------------------------------------
# Main modes
# ----------------------------------------------------------------------------

def export_frames(source, info, args, out_dir):
    """Save each frame separately."""
    xyz_lut = core.XYZLut(info)

    for index, scan in enumerate(iterate_scans(source)):
        if index >= args.frames:
            break

        arrays = scan_to_arrays(scan, info, xyz_lut)
        completeness = check_column_completeness(scan)
        np.savez_compressed(out_dir / f"frame_{index:05d}.npz", **arrays)

        mask = valid_mask(arrays["range"], args.min_range, args.max_range)
        if args.ply:
            intensity = arrays.get("reflectivity")
            save_ply(
                out_dir / f"frame_{index:05d}.ply",
                arrays["xyz"][mask],
                intensity[mask] if intensity is not None else None,
            )

        print(f"frame {index:5d}  id={arrays['frame_id']}  "
              f"points={int(mask.sum()):7d}  columns_ok={completeness:.3f}")


def export_accumulated(source, info, args, out_dir):
    """Average a static scene pixel by pixel over many frames.

    Valid only if nothing moves: each pixel is the same beam direction in
    every frame, so averaging the range reduces random noise roughly as
    1/sqrt(N). It does not remove systematic range bias.
    """
    xyz_lut = core.XYZLut(info)
    xyz_sum = None
    count = None
    used_frames = 0

    for index, scan in enumerate(iterate_scans(source)):
        if index >= args.frames:
            break

        arrays = scan_to_arrays(scan, info, xyz_lut)
        mask = valid_mask(arrays["range"], args.min_range, args.max_range)

        if xyz_sum is None:
            xyz_sum = np.zeros_like(arrays["xyz"], dtype=np.float64)
            count = np.zeros(mask.shape, dtype=np.int32)

        xyz_sum[mask] += arrays["xyz"][mask]
        count[mask] += 1
        used_frames += 1

    # Keep pixels that returned in most frames, to discard flickering edges.
    stable = count >= max(1, int(args.min_hit_ratio * used_frames))
    points = xyz_sum[stable] / count[stable][:, None]

    np.savez_compressed(out_dir / "accumulated.npz", points=points.astype(np.float32),
                        hits=count[stable], frames=used_frames)
    if args.ply:
        save_ply(out_dir / "accumulated.ply", points)

    print(f"accumulated {used_frames} frames into {points.shape[0]} points")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract point clouds from an Ouster source.")
    parser.add_argument("--source", required=True,
                        help="sensor hostname/IP, or path to a .pcap / .osf recording")
    parser.add_argument("--meta", default=None, help="metadata json for a pcap recording")
    parser.add_argument("--out", default="frames", help="output directory")
    parser.add_argument("--frames", type=int, default=20, help="number of frames to process")
    parser.add_argument("--min-range", type=float, default=0.3, help="meters")
    parser.add_argument("--max-range", type=float, default=10.0, help="meters")
    parser.add_argument("--auto-udp-dest", action="store_true",
                        help="live sensor: let the SDK overwrite the sensor's UDP destination with "
                             "the address it detects (default: keep the destination stored in the "
                             "sensor, configured once with setup_ip.py)")
    parser.add_argument("--ply", action="store_true", help="also write PLY files (needs open3d)")
    parser.add_argument("--accumulate", action="store_true",
                        help="average a static scene into one cloud")
    parser.add_argument("--min-hit-ratio", type=float, default=0.8,
                        help="accumulate mode: fraction of frames a pixel must be valid in")
    return parser.parse_args()


def main():
    args = parse_arguments()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    source, info = open_scan_source(args.source, args.meta, args.auto_udp_dest)
    print(f"sensor: {info.prod_line}  fw: {info.fw_rev}  "
          f"mode: {info.config.lidar_mode}  "
          f"size: {info.format.pixels_per_column}x{info.format.columns_per_frame}")
    save_metadata(info, out_dir)

    try:
        if args.accumulate:
            export_accumulated(source, info, args, out_dir)
        else:
            export_frames(source, info, args, out_dir)
    finally:
        source.close()


if __name__ == "__main__":
    main()
