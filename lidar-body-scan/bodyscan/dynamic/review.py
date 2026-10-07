"""Pictures of a tracked motion for human review.

Every rendered frame shows two views of the tracked body (shaded mesh), seen
from the LiDAR and from the side, with the person's LiDAR points drawn over
it, coloured by their distance to the body (green on the surface, yellow at
2 cm, red at 5 cm and more). Points hidden behind the body in the view are
not drawn. A strip at the top gives the time, the frame number and the
flags of the tracker (free space, sliding, ...).

Output: review_XXXXX.png for the chosen frames (flagged ones first) and
review.gif with every 'every'-th frame (needs Pillow; without it only the
PNG files are written).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from bodyscan.log import info, warning

BACKGROUND = np.array([13, 21, 20], dtype=np.float32)
DARK = np.array([22, 70, 66], dtype=np.float32)
LIGHT = np.array([196, 245, 232], dtype=np.float32)


def _camera(direction_xy):
    view = np.array([direction_xy[0], direction_xy[1], 0.0])
    view /= max(np.linalg.norm(view), 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, view)
    return view, right, up


def render(vertices, faces, points, residuals, center, direction_xy, half_width=0.9, half_height=1.05,
           pixels=360) -> np.ndarray:
    """(H, W, 3) uint8 orthographic view along direction_xy, centred on 'center'."""
    view, right, up = _camera(direction_xy)
    h = pixels
    w = int(round(pixels * half_width / half_height))
    u, v = np.meshgrid(np.linspace(-half_width, half_width, w), np.linspace(half_height, -half_height, h))
    origins = center[None] + u.reshape(-1, 1) * right + v.reshape(-1, 1) * up - 4.0 * view
    rays = np.hstack([origins, np.tile(view, (len(origins), 1))]).astype(np.float32)
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(np.asarray(vertices, dtype=np.float32))
    mesh.triangle.indices = o3d.core.Tensor(np.asarray(faces, dtype=np.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    hit = scene.cast_rays(o3d.core.Tensor(rays))
    depth = hit["t_hit"].numpy().reshape(h, w)
    normals = hit["primitive_normals"].numpy().reshape(h, w, 3)
    light = -view + np.array([0.3, 0.5, 0.6])
    light /= np.linalg.norm(light)
    shade = 0.15 + 0.85 * np.clip(np.abs(normals @ light), 0, 1)
    image = np.empty((h, w, 3), dtype=np.float32)
    image[:] = BACKGROUND
    on_body = np.isfinite(depth)
    image[on_body] = DARK + (LIGHT - DARK) * shade[on_body, None]
    # the floor line
    floor_row = int(round((half_height - (0.0 - center[2])) / (2 * half_height) * (h - 1)))
    if 0 <= floor_row < h:
        image[floor_row:floor_row + 1] = (70, 95, 90)
    if points is not None and len(points):
        relative = points - center
        px = np.round((relative @ right + half_width) / (2 * half_width) * (w - 1)).astype(int)
        py = np.round((half_height - relative @ up) / (2 * half_height) * (h - 1)).astype(int)
        pd = relative @ view + 4.0
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        px, py, pd, res = px[inside], py[inside], pd[inside], residuals[inside]
        visible = ~(depth[py, px] < pd - 0.03)                         # not behind the body in this view
        level = np.clip(res / 0.05, 0.0, 1.0)
        colors = np.stack([255 * np.clip(2 * level, 0, 1), 255 * np.clip(2 - 2 * level, 0, 1),
                           40 * np.ones_like(level)], axis=1)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            x, y = np.clip(px + dx, 0, w - 1), np.clip(py + dy, 0, h - 1)
            image[y[visible], x[visible]] = colors[visible]
    return image.astype(np.uint8)


def _label(image, lines):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return image
    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture, "RGBA")
    draw.rectangle((0, 0, picture.size[0], 16 * len(lines) + 6), fill=(13, 21, 20, 200))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for k, line in enumerate(lines):
        draw.text((8, 4 + 16 * k), line, fill=(229, 238, 236), font=font)
    return np.asarray(picture)


def review(motion_path, segment_folder, avatar_path, out, every: int = 2, frames=None, pixels: int = 320,
           model_path=None) -> list[Path]:
    import torch
    from bodyscan.dynamic.avatar import Avatar
    from bodyscan.dynamic.fitting import closest_points
    from bodyscan.dynamic.motions import PoseSequence
    from bodyscan.dynamic.segment import SegmentedRecording
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(motion_path, allow_pickle=False)
    motion = PoseSequence.load(motion_path)
    flags = data["flags"] if "flags" in data.files else np.zeros(len(motion), dtype=np.int64)
    names = [str(n) for n in data["flag_names"]] if "flag_names" in data.files else []
    frame_index = data["frame_index"] if "frame_index" in data.files else np.arange(len(motion))
    avatar = Avatar.load(avatar_path)
    model = avatar.model(model_path, "cpu", torch.float64)
    shaped = avatar.shaped(model, 0)
    segmented = SegmentedRecording(segment_folder)
    by_index = {}
    for k, path in enumerate(segmented.paths):
        by_index[int(path.stem.split("_")[-1])] = k
    sensor = segmented.sensor_position
    chosen = list(range(0, len(motion), max(every, 1))) if frames is None else list(frames)
    flagged = [k for k in range(len(motion)) if flags[k]]
    written, gif_frames = [], []
    for k in sorted(set(chosen) | set(flagged[:50])):
        part = motion.subset(slice(k, k + 1))
        vertices, _ = part.posed(shaped)
        vertices = vertices[0].astype(np.float64)
        person = segmented.load(by_index[int(frame_index[k])]) if int(frame_index[k]) in by_index else None
        points = person.points if person is not None else np.zeros((0, 3))
        residual = (closest_points(vertices, shaped.faces_np, points)[2] if len(points) else np.zeros(0))
        center = np.array([part.root_position[0, 0], part.root_position[0, 1], 1.0])
        toward = center[:2] - sensor[:2]
        views = [render(vertices, shaped.faces_np, points, residual, center, toward, pixels=pixels),
                 render(vertices, shaped.faces_np, points, residual, center, np.array([-toward[1], toward[0]]),
                        pixels=pixels)]
        image = np.concatenate(views, axis=1)
        active = [names[b] for b in range(len(names)) if flags[k] >> b & 1]
        lines = [f"frame {int(frame_index[k])}  t = {motion.times[k] - motion.times[0]:.2f} s   "
                 f"left: from the LiDAR, right: from the side",
                 "flags: " + (", ".join(active) if active else "none") +
                 (f"   points to body median {1000 * np.median(residual):.0f} mm" if len(residual) else "")]
        image = _label(image, lines)
        if k in chosen:
            gif_frames.append(image)
        if flags[k] or frames is not None:
            path = out / f"review_{int(frame_index[k]):05d}.png"
            o3d.io.write_image(str(path), o3d.geometry.Image(np.ascontiguousarray(image)))
            written.append(path)
    try:
        from PIL import Image
        if gif_frames:
            pictures = [Image.fromarray(f) for f in gif_frames]
            duration = int(1000 * (motion.times[-1] - motion.times[0]) / max(len(motion) - 1, 1) * max(every, 1))
            pictures[0].save(out / "review.gif", save_all=True, append_images=pictures[1:], duration=duration, loop=0)
            written.append(out / "review.gif")
    except ImportError:
        warning("Pillow is not installed: no GIF (python -m pip install pillow)")
    info(f"wrote {len(written)} review files to {out}")
    return written
