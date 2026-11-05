"""Additional sensors in the tracker: the pinhole image geometry, the cut of the person out of an additional
sensor's images (extra_segment), the folder read by the tracker (views.ExtraSensors) and the frame data
built from several sensors.

    python -m unittest tests.test_multisensor
"""

from __future__ import annotations

import importlib.util
import json
import unittest

import numpy as np

from tests.helpers import TemporaryFolder

HAS_TORCH = importlib.util.find_spec("torch") is not None


def look_at(origin, target) -> np.ndarray:
    """world_from_sensor of a camera (optical frame x right, y down, z forward) at 'origin' looking at 'target'."""
    forward = np.asarray(target, dtype=np.float64) - origin
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    T = np.eye(4)
    T[:3, :3] = np.stack([right, down, forward], axis=1)
    T[:3, 3] = origin
    return T


def cylinder_scene(geometry, center_xy, radius=0.2, height=1.7, wall=4.0):
    """Ranges [m] of a vertical cylinder (the 'person') in front of a wall at distance 'wall' along the
    camera axis, ray-cast analytically for every pixel of a PinholeGeometry."""
    rays = geometry.ray_directions() @ geometry.rotation.T                      # world directions
    origin = geometry.translation
    forward = geometry.rotation[:, 2]
    wall_range = wall / np.maximum(rays @ forward, 1e-6)
    d = rays[..., :2]
    o = origin[:2] - np.asarray(center_xy)
    a = (d ** 2).sum(-1)
    b = 2 * (d * o).sum(-1)
    c = (o ** 2).sum() - radius ** 2
    disc = b ** 2 - 4 * a * c
    with np.errstate(invalid="ignore"):
        hit = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-12))
    z = origin[2] + hit * rays[..., 2]
    body = (disc > 0) & (hit > 0) & (z > 0.02) & (z < height)
    return np.where(body, hit, wall_range), body


class PinholeGeometryTest(unittest.TestCase):
    def test_projection_returns_the_pixel_of_every_ray(self):
        from bodyscan.dynamic.views import PinholeGeometry
        T = look_at(np.array([0.0, 0.0, 1.0]), [3.0, 1.0, 0.8])
        g = PinholeGeometry(64, 48, 57.5, 57.5, 31.5, 23.5, T)
        rays = g.ray_directions()
        world = (rays * 2.5) @ g.rotation.T + g.translation
        rows, cols, ranges = g.project(world.reshape(-1, 3))
        v, u = np.mgrid[0:48, 0:64]
        self.assertTrue(np.allclose(rows, v.ravel(), atol=1e-6))
        self.assertTrue(np.allclose(cols, u.ravel(), atol=1e-6))
        self.assertTrue(np.allclose(ranges, 2.5, atol=1e-9))
        behind = g.translation - 1.0 * g.rotation[:, 2]
        self.assertLess(g.project(behind[None])[0][0], -100)                 # outside every image

    @unittest.skipUnless(HAS_TORCH, "needs PyTorch")
    def test_torch_projection_matches_numpy(self):
        import torch
        from bodyscan.dynamic.views import PinholeGeometry
        g = PinholeGeometry(64, 48, 57.5, 57.5, 31.5, 23.5, look_at(np.array([0.0, 0.0, 1.0]), [3.0, 1.0, 0.8]))
        points = np.array([[3.0, 1.0, 0.8], [2.5, 0.6, 1.2], [3.4, 1.5, 0.3]])
        rows, cols, ranges = g.project(points)
        tables = g.torch_tables("cpu", torch.float64)
        rt, ct, dt = g.project_torch(torch.as_tensor(points), tables, 0.0)
        self.assertTrue(np.allclose(rt.numpy(), rows) and np.allclose(ct.numpy(), cols) and np.allclose(dt.numpy(), ranges))


class ExtraSegmentTest(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryFolder()

    def tearDown(self):
        self.folder.cleanup()

    def test_the_person_is_cut_out_of_a_depth_camera_and_read_back(self):
        from bodyscan.dynamic.extra_segment import ExtraSegmentationConfig, ExtraSegmenter, SensorImages
        from bodyscan.dynamic.views import ExtraSensors, PinholeGeometry
        T = look_at(np.array([0.0, 0.0, 1.0]), [3.0, 0.0, 0.9])
        g = PinholeGeometry(160, 120, 143.9, 143.9, 79.5, 59.5, T)
        empty, _ = cylinder_scene(g, (30.0, 30.0))                             # cylinder far away: empty scene
        frames, times, masks = [], [], []
        for k in range(5):
            r, body = cylinder_scene(g, (3.0, -0.2 + 0.1 * k))
            frames.append(r)
            masks.append(body)
            times.append(np.full(r.shape, 10.0 + k / 30.0))
        images = SensorImages("pinhole", g, empty, frames, times, [10.0 + k / 30.0 for k in range(5)], 1 / 30.0)
        track = [{"index": k, "time": 10.0 + k / 30.0 + 0.004, "center": [3.0, -0.2 + 0.1 * k]} for k in range(5)]
        summary = ExtraSegmenter(images, ExtraSegmentationConfig(min_points=20)).run(track, self.folder.path, "cam", 1.5)
        self.assertEqual(len(summary["frames"]), 5)
        extra = ExtraSensors(self.folder.path)
        self.assertEqual([v.name for v in extra.views], ["cam"])
        number, person = extra.frames_for(2)[0]
        # every point lies on the cylinder of frame 2, every person pixel is in the crop's mask
        radial = np.linalg.norm(person.points[:, :2] - np.array([3.0, 0.0]), axis=1)
        self.assertLess(np.abs(radial - 0.2).max(), 1e-6)
        # the person's pixels are the cylinder's, less the mixed-pixel filter's edges at grazing angles
        r0, c0 = person.crop_origin
        h, w = person.crop_mask.shape
        cut = masks[2][r0:r0 + h, c0:c0 + w]
        self.assertTrue(np.all(cut[person.crop_mask == 1]))
        self.assertGreater((person.crop_mask == 1).sum(), 0.85 * masks[2].sum())
        self.assertAlmostEqual(float(np.median(person.point_times)), 10.0 + 2 / 30.0, places=9)

    @unittest.skipUnless(HAS_TORCH, "needs PyTorch")
    def test_frame_data_puts_every_sensor_after_the_ouster_and_keeps_its_hold_out(self):
        import torch
        from bodyscan.dynamic.segment import PersonFrame
        from bodyscan.dynamic.track import FrameData
        from bodyscan.dynamic.views import PinholeGeometry, SensorView

        def person(index, count, seed):
            rng = np.random.default_rng(seed)
            points = rng.normal(size=(count, 3))
            pixels = np.stack([rng.integers(0, 8, count), rng.integers(0, 8, count)], axis=1)
            crop = np.full((8, 8), 3.0)
            return PersonFrame(index, 1.0, points, np.full(count, 1.0), pixels, np.full(count, 3.0), (0, 0),
                               np.ones((8, 8), dtype=np.uint8), crop, crop + 1, np.zeros(2), {}, np.full((8, 8), 1.0))

        g = PinholeGeometry(8, 8, 8.0, 8.0, 3.5, 3.5, np.eye(4))
        views = [SensorView("ouster", g, np.full((8, 8), 4.0)), SensorView("cam", g, np.full((8, 8), 4.0), sigma=2.0)]
        ouster, cam = person(7, 100, 1), person(7, 40, 2)
        alone = FrameData(ouster, g, "cpu", torch.float64, holdout=0.2, seed=3, views=views[:1])
        both = FrameData(ouster, g, "cpu", torch.float64, holdout=0.2, seed=3, views=views, extra=[(1, cam)])
        self.assertTrue(np.array_equal(alone.points_np, both.points_np[:len(alone.points_np)]))
        self.assertEqual(len(both.points_np), len(alone.points_np) + both.views[1].count)
        self.assertTrue(np.all(both.point_view[len(alone.points_np):] == 1))
        self.assertTrue(np.allclose(both.point_sigma.numpy()[len(alone.points_np):], 2.0))
        views[1].fitted = False                                                 # held-out sensor
        held = FrameData(ouster, g, "cpu", torch.float64, holdout=0.2, seed=3, views=views, extra=[(1, cam)])
        self.assertEqual(len(held.points_np), len(alone.points_np))
        self.assertEqual(int((held.heldout_view == 1).sum()), 40)


if __name__ == "__main__":
    unittest.main()
