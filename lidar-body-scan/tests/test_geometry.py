"""Geometry helpers: transforms, fits, neighbourhoods."""

import unittest

import numpy as np
import open3d as o3d

from bodyscan.geometry import (RadiusSearch, Target, evaluate, fit_circle, project_to_4dof, ransac_planes,
                               rotation_to_z, support_filter, tilt_degrees, transform_points, turn_points,
                               wrapped_degrees, yaw_of, yaw_transform)
import tests.helpers  # noqa: F401  (quiet console output)


class TransformTest(unittest.TestCase):
    def test_yaw_about_a_pivot(self):
        pivot = np.array([1.0, 2.0, 0.0])
        transform = yaw_transform(np.radians(30.0), pivot)
        self.assertAlmostEqual(np.degrees(yaw_of(transform)), 30.0, places=9)
        np.testing.assert_allclose(transform_points(transform, pivot[None])[0], pivot, atol=1e-12)
        self.assertAlmostEqual(tilt_degrees(transform), 0.0, places=9)

    def test_turn_points_matches_the_transform(self):
        rng = np.random.default_rng(1)
        points = rng.normal(size=(50, 3))
        pivot = np.array([0.3, -0.2, 0.0])
        expected = transform_points(yaw_transform(0.4, pivot), points)
        np.testing.assert_allclose(turn_points(points, np.full(50, 0.4), pivot), expected, atol=1e-12)

    def test_projection_to_4dof_removes_the_tilt(self):
        tilted = np.eye(4)
        tilted[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0.05, -0.03, 0.7))
        projected = project_to_4dof(tilted)
        self.assertLess(tilt_degrees(projected), 1e-9)
        self.assertAlmostEqual(yaw_of(projected), yaw_of(tilted), places=2)

    def test_rotation_to_z_and_wrapping(self):
        normal = np.array([0.1, -0.2, 0.97])
        normal /= np.linalg.norm(normal)
        np.testing.assert_allclose(rotation_to_z(normal) @ normal, [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(wrapped_degrees(np.radians([190.0, -190.0, 10.0])), [-170.0, 170.0, 10.0])


class FittingTest(unittest.TestCase):
    def test_circle(self):
        rng = np.random.default_rng(2)
        angles = rng.uniform(0, 2 * np.pi, 400)
        xy = np.column_stack([1.2 + 0.58 * np.cos(angles), 0.3 + 0.58 * np.sin(angles)])
        xy += rng.normal(0, 0.003, xy.shape)
        center, radius, rms, inliers = fit_circle(xy)
        np.testing.assert_allclose(center, [1.2, 0.3], atol=0.003)
        self.assertAlmostEqual(radius, 0.58, delta=0.003)
        self.assertGreater(inliers, 380)

    def test_planes(self):
        rng = np.random.default_rng(3)
        floor = np.column_stack([rng.uniform(-2, 2, 5000), rng.uniform(-2, 2, 5000), rng.normal(0, 0.003, 5000)])
        wall = np.column_stack([np.full(3000, 2.0) + rng.normal(0, 0.003, 3000), rng.uniform(-2, 2, 3000),
                                rng.uniform(0, 2, 3000)])
        planes = ransac_planes(np.vstack([floor, wall]), count=3, min_inliers=1000)
        normals = sorted(tuple(np.round(np.abs(p.normal), 1)) for p in planes)
        self.assertIn((0.0, 0.0, 1.0), normals)
        self.assertIn((1.0, 0.0, 0.0), normals)


class PlatformTest(unittest.TestCase):
    def ring_scene(self, center):
        rng = np.random.default_rng(5)
        angles = rng.uniform(0, 2 * np.pi, 3000)
        ring = np.column_stack([center[0] + 0.582 * np.cos(angles), center[1] + 0.582 * np.sin(angles),
                                rng.uniform(0.01, 0.04, 3000)])
        floor = np.column_stack([rng.uniform(-3, 3, 20000), rng.uniform(-3, 3, 20000), rng.normal(0, 0.002, 20000)])
        return np.vstack([ring, floor])

    def test_ring_near_the_start_and_far_from_it(self):
        from bodyscan.scene import RingPlatform
        world = self.ring_scene((1.2, 0.1))
        for start in ((1.25, 0.05), (1.6, 0.5)):                 # near, and 0.57 m away (Hough search)
            found = RingPlatform().find(world, np.array(start))
            self.assertTrue(found.plausible)
            np.testing.assert_allclose(found.center, (1.2, 0.1), atol=0.01)


class NeighbourTest(unittest.TestCase):
    def test_support_filter_keeps_points_seen_by_several_views(self):
        rng = np.random.default_rng(4)
        common = rng.uniform(0, 1, (300, 3))
        ghost = rng.uniform(5, 6, (50, 3))
        points = np.vstack([common, common + 0.001, ghost])
        labels = np.concatenate([np.zeros(300), np.ones(300), np.zeros(50)]).astype(int)
        keep = support_filter(points, labels, 0.01, 2)
        self.assertTrue(keep[:600].all())
        self.assertFalse(keep[600:].any())

    def test_radius_search_returns_every_point_within_the_radius(self):
        # 1 mm grid, 10.5 mm radius (no point on the boundary): about 350 points per
        # neighbourhood, far more than the initial cap of 64
        u, v = np.meshgrid(np.arange(60) * 0.001, np.arange(60) * 0.001)
        points = np.column_stack([u.ravel(), v.ravel(), np.zeros(u.size)])
        queries = points[::97]
        search = RadiusSearch(points, 0.0105)
        found = np.concatenate([f for _, f, _ in search.chunks(queries, entries=20000)])
        truth = (np.linalg.norm(queries[:, None, :] - points[None, :, :], axis=2) <= 0.0105).sum(axis=1)
        np.testing.assert_array_equal((found >= 0).sum(axis=1), truth)
        self.assertGreater(truth.max(), 300)

    def test_overlap_of_a_cloud_with_itself(self):
        cloud = o3d.geometry.TriangleMesh.create_sphere(0.3).sample_points_uniformly(3000)
        cloud.estimate_normals()
        target = Target(cloud)
        points = np.asarray(cloud.points)
        fitness, rmse = evaluate(points, target, np.eye(4), 0.01)
        self.assertGreater(fitness, 0.99)
        self.assertLess(rmse, 1e-6)
        fitness, _ = evaluate(points, target, yaw_transform(0.0, np.zeros(3), (0.5, 0.0, 0.0)), 0.01)
        self.assertLess(fitness, 0.5)


if __name__ == "__main__":
    unittest.main()
