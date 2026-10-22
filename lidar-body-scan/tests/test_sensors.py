"""Sensor models, range images, clock fit and rig files (numpy and Open3D only)."""

from __future__ import annotations

import math
import unittest

import numpy as np
import open3d as o3d

from tests.helpers import TemporaryFolder

from bodyscan.sensors import (NO_RETURN, VALID, ClockModel, Placement, Rig, fit_clock, helios_16, looking_at,
                              ouster_os0_128, pose, render, rotation_matrix, sensor_from_dict, xtion)


def _wall(x: float, size: float = 8.0):
    """A vertical square wall facing -x at distance x (two triangles)."""
    h = size / 2.0
    vertices = np.array([[x, -h, -h], [x, h, -h], [x, h, h], [x, -h, h]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return vertices, faces


class SensorModelTest(unittest.TestCase):
    def test_helios_beams_and_columns(self):
        helios = helios_16()
        self.assertEqual(helios.shape, (16, 1800))
        vertical, horizontal = helios.angular_steps_deg()
        self.assertAlmostEqual(vertical, 2.0, places=9)
        self.assertAlmostEqual(horizontal, 0.2, places=9)
        self.assertEqual(helios_16(900, 20.0).mode, "900x20")
        directions = helios.directions()
        self.assertTrue(np.allclose(np.linalg.norm(directions, axis=-1), 1.0))
        top = np.degrees(np.arcsin(directions[0, 0, 2]))
        self.assertAlmostEqual(top, 15.0, places=6)
        # column 0 looks along +x; columns turn counter-clockwise
        self.assertTrue(np.allclose(directions[8, 0, :2] / np.linalg.norm(directions[8, 0, :2]), (1.0, 0.0)))
        self.assertGreater(directions[8, 450, 1], 0.99 * np.cos(np.radians(1.0)))

    def test_pixel_times_follow_the_rotation(self):
        ouster = ouster_os0_128(1024, 20.0)
        times = ouster.pixel_times(100.0)
        self.assertEqual(times.shape, (128, 1024))
        self.assertAlmostEqual(times[0, 0], 100.0)
        self.assertAlmostEqual(times[5, -1], 100.0 + 1023 / (1024 * 20.0))

    def test_xtion_field_of_view_and_orientation(self):
        landscape, portrait = xtion("landscape"), xtion("portrait")
        horizontal, vertical = landscape.field_of_view_deg()
        self.assertAlmostEqual(horizontal, 58.0, places=6)
        self.assertAlmostEqual(vertical, 45.0, places=6)
        self.assertEqual(tuple(round(v, 6) for v in portrait.field_of_view_deg()), (45.0, 58.0))
        d = landscape.directions()
        self.assertTrue(np.allclose(np.linalg.norm(d, axis=-1), 1.0))
        centre = d[240, 320]
        self.assertGreater(centre[0], 0.9999)                              # looks along +x
        self.assertGreater(d[240, 0, 1], 0.0)                               # left image edge looks to +y (left)
        self.assertGreater(d[0, 320, 2], 0.0)                               # top image row looks up
        dp = portrait.directions()
        # portrait: the image's vertical extent is the 58 deg side
        top = np.degrees(np.arctan2(dp[..., 2], dp[..., 0])).max()
        self.assertAlmostEqual(top, 29.0, delta=0.2)

    def test_xtion_noise_grows_with_depth_squared(self):
        camera = xtion()
        axis = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(float(camera.range_sigma(2.0, axis)), 0.0064, places=6)
        self.assertAlmostEqual(float(camera.range_sigma(3.0, axis)), 0.0144, places=6)

    def test_in_view(self):
        helios = helios_16()
        points = np.array([[3.0, 0.0, 0.0], [3.0, 0.0, 3.0 * math.tan(math.radians(14.0))],
                           [3.0, 0.0, 3.0 * math.tan(math.radians(20.0))], [0.1, 0.0, 0.0]])
        self.assertEqual(helios.in_view(points).tolist(), [True, True, False, False])
        camera = xtion()
        points = np.array([[2.0, 0.0, 0.0], [2.0, 2.0 * math.tan(math.radians(28.0)), 0.0],
                           [2.0, 2.0 * math.tan(math.radians(31.0)), 0.0], [0.5, 0.0, 0.0], [4.0, 0.0, 0.0]])
        self.assertEqual(camera.in_view(points).tolist(), [True, True, False, False, False])

    def test_round_trip_through_dicts(self):
        for model in (ouster_os0_128(), helios_16(900, 20.0), xtion("portrait")):
            again = sensor_from_dict(model.to_dict())
            self.assertEqual(again, model)


class RenderTest(unittest.TestCase):
    def test_lidar_sees_a_wall_at_its_distance(self):
        helios = helios_16()
        image = render(helios, pose((0.0, 0.0, 1.0)), [_wall(3.0)], rng=np.random.default_rng(1))
        forward = image.state[:, 0] == VALID
        self.assertTrue(forward.all())
        expected = 3.0 / np.cos(np.radians(np.asarray(helios.elevations_deg)))
        self.assertLess(np.abs(image.range[:, 0] - expected).max(), 0.05)       # 1 cm noise, 5 sigma
        self.assertTrue((image.state[:, 900] == NO_RETURN).all())                # behind the sensor: nothing
        points = image.world_points(image.state == VALID)
        self.assertLess(np.abs(points[:, 0] - 3.0).max(), 0.06)

    def test_depth_camera_measures_depth_and_respects_its_range(self):
        camera = xtion()
        near = render(camera, pose((0.0, 0.0, 1.0)), [_wall(2.0)])
        self.assertTrue((near.state == VALID).all())
        depth = camera.depth_of(near.range, near.directions)
        self.assertTrue(np.allclose(depth, 2.0, atol=1e-4))
        far = render(camera, pose((0.0, 0.0, 1.0)), [_wall(4.0)])
        self.assertTrue((far.state == NO_RETURN).all())                          # beyond 3.5 m

    def test_placement_turns_the_rays(self):
        camera = xtion()
        target = np.array([0.0, 3.0, 1.0])
        world_from_sensor = looking_at((0.0, 0.0, 1.0), target)
        image = render(camera, world_from_sensor, [_wall(3.0)])                   # wall at x = 3, not in view
        self.assertTrue((image.state == NO_RETURN).all())
        wall = (np.array([[-4.0, 3.0, -4.0], [4.0, 3.0, -4.0], [4.0, 3.0, 6.0], [-4.0, 3.0, 6.0]]),
                np.array([[0, 1, 2], [0, 2, 3]]))
        image = render(camera, world_from_sensor, [wall])
        self.assertTrue((image.state == VALID).all())
        self.assertTrue(np.allclose(image.world_points()[:, 1], 3.0, atol=1e-4))


class ClockTest(unittest.TestCase):
    def test_lower_envelope_ignores_delays_and_stalls(self):
        rng = np.random.default_rng(3)
        sensor = np.sort(rng.uniform(0.0, 60.0, 20000))
        offset, rate, floor = 12.5, 1.0 + 35e-6, 0.0004
        delay = floor + rng.exponential(0.0008, len(sensor))
        stalls = rng.random(len(sensor)) < 0.02
        delay[stalls] += rng.uniform(0.01, 0.1, stalls.sum())                  # the receiving PC stalls
        arrival = offset + rate * sensor + delay
        arrival[[100, 5000]] -= 0.5                                             # two corrupted timestamps
        model, report = fit_clock(sensor, arrival)
        self.assertEqual(report["rejected"], 2)
        self.assertAlmostEqual(model.rate, rate, delta=2e-6)
        # the line sits on the fastest arrivals: offset plus the smallest delay of the link
        predicted = model.to_session(sensor) - (offset + rate * sensor + floor)
        self.assertLess(np.abs(predicted).max(), 1e-4)
        self.assertLess(report["delay_median_ms"], 1.0)

    def test_round_trip_and_inverse(self):
        clock = ClockModel(3.0, 1.0001)
        self.assertAlmostEqual(float(clock.to_sensor(clock.to_session(17.0))), 17.0)
        self.assertEqual(ClockModel.from_dict(clock.to_dict()), clock)
        self.assertAlmostEqual(clock.shifted(0.25).offset, 3.25)


class RigTest(unittest.TestCase):
    def test_rotations_follow_the_conventions(self):
        forward = rotation_matrix(90.0) @ np.array([1.0, 0.0, 0.0])
        self.assertTrue(np.allclose(forward, (0.0, 1.0, 0.0)))
        tilted = rotation_matrix(0.0, 10.0) @ np.array([1.0, 0.0, 0.0])
        self.assertLess(tilted[2], 0.0)                                          # positive pitch looks down
        m = looking_at((0.0, 0.0, 1.0), (2.0, 2.0, 0.0))
        direction = m[:3, :3] @ np.array([1.0, 0.0, 0.0])
        expected = np.array([2.0, 2.0, -1.0]) / 3.0
        self.assertTrue(np.allclose(direction, expected))

    def test_rig_file_round_trip(self):
        folder = TemporaryFolder()
        try:
            rig = Rig([Placement("ouster", ouster_os0_128(), pose((0.0, 0.0, 1.03)), serial="122542000054"),
                       Placement("helios", helios_16(), pose((5.9, 0.0, 0.95), 180.0), ClockModel(0.002, 1.0)),
                       Placement("xtion", xtion("portrait"), looking_at((2.3, 2.2, 0.95), (2.3, 0.0, 0.9)))],
                      reference="ouster", calibration={"method": "test"})
            path = rig.save(folder.path / "rig.json")
            again = Rig.load(path)
            self.assertEqual(again.names, ["ouster", "helios", "xtion"])
            self.assertEqual(again.reference, "ouster")
            for a, b in zip(rig.placements, again.placements):
                self.assertEqual(a.sensor, b.sensor)
                self.assertTrue(np.allclose(a.world_from_sensor, b.world_from_sensor))
                self.assertEqual(a.clock, b.clock)
            self.assertEqual(again.subset(["helios", "xtion"]).reference, "helios")
            with self.assertRaises(ValueError):
                Rig([Placement("a", helios_16(), np.eye(4)), Placement("a", xtion(), np.eye(4))])
        finally:
            folder.cleanup()

    def test_to_sensor_inverts_the_placement(self):
        placement = Placement("xtion", xtion(), looking_at((1.0, 2.0, 1.0), (3.0, 0.0, 0.5)))
        world = np.array([[3.0, 0.0, 0.5]])
        local = placement.to_sensor(world)
        self.assertAlmostEqual(local[0, 1], 0.0, places=9)
        self.assertAlmostEqual(local[0, 2], 0.0, places=9)
        self.assertGreater(local[0, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
