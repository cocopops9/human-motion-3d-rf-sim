"""Meshing: marching cubes, orientation, watertight remesh, smoothing, quality."""

import unittest

import numpy as np
import open3d as o3d

from bodyscan.meshing import bilateral_smooth, marching_cubes, mesh_quality, watertight_remesh
from bodyscan.meshing.cleanup import orient_by_cloud_normals, orient_consistently
from bodyscan.meshing.smoother import DeviationLimit, SmoothingConfig, smooth_mesh
from bodyscan.meshing.quality import facet_noise
from bodyscan.meshing.smoothing import bilateral_normals, face_geometry, tangential_relaxation, vertex_passes
from tests.helpers import sphere_mesh


def closed_and_manifold(mesh) -> bool:
    return (len(mesh.get_non_manifold_edges(allow_boundary_edges=False)) == 0
            and bool(mesh.is_vertex_manifold()))


def consistent(triangles) -> bool:
    """Every directed edge appears once: neighbours run along shared edges in opposite directions."""
    directed = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    keys = directed[:, 0] * (triangles.max() + 1) + directed[:, 1]
    return len(np.unique(keys)) == len(keys)


def signed_volume(mesh) -> float:
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    return float(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6.0)


class MarchingCubesTest(unittest.TestCase):
    def test_sphere_is_closed_manifold_and_outward(self):
        voxel, radius = 0.01, 0.3
        axis = np.arange(-0.4, 0.4 + voxel / 2, voxel)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        sdf = np.sqrt(x ** 2 + y ** 2 + z ** 2) - radius
        vertices, faces = marching_cubes(sdf.astype(np.float32), voxel)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
        self.assertTrue(closed_and_manifold(mesh))
        self.assertTrue(consistent(np.asarray(mesh.triangles)))
        self.assertAlmostEqual(mesh.get_surface_area(), 4 * np.pi * radius ** 2, delta=0.02 * 4 * np.pi * radius ** 2)
        self.assertGreater(signed_volume(mesh), 0.0)


class OrientationTest(unittest.TestCase):
    def test_random_flips_are_undone(self):
        mesh = sphere_mesh()
        triangles = np.asarray(mesh.triangles).copy()
        flip = np.random.default_rng(0).random(len(triangles)) < 0.3
        triangles[flip] = triangles[flip][:, [0, 2, 1]]
        fixed, conflicts = orient_consistently(triangles)
        self.assertEqual(conflicts, 0)
        self.assertTrue(consistent(fixed))

    def test_cloud_normals_choose_the_outward_side(self):
        mesh = sphere_mesh()
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, [0, 2, 1]])   # inside out
        cloud = sphere_mesh().sample_points_uniformly(5000)
        cloud.normals = o3d.utility.Vector3dVector(np.asarray(cloud.points) / 0.3)
        oriented = orient_by_cloud_normals(mesh, cloud)
        self.assertGreater(signed_volume(oriented), 0.0)


class WatertightTest(unittest.TestCase):
    def test_inconsistent_winding_does_not_add_sheets(self):
        reference = watertight_remesh(sphere_mesh(), 0.01)
        mesh = sphere_mesh()
        triangles = np.asarray(mesh.triangles).copy()
        flip = np.random.default_rng(1).random(len(triangles)) < 0.3
        triangles[flip] = triangles[flip][:, [0, 2, 1]]
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        result = watertight_remesh(mesh, 0.01)
        self.assertTrue(closed_and_manifold(result))
        self.assertAlmostEqual(result.get_surface_area(), reference.get_surface_area(),
                               delta=0.01 * reference.get_surface_area())
        self.assertGreater(signed_volume(result), 0.0)

    def test_a_small_piece_wound_inside_out(self):
        """A piece under 1 % of the surface, wound inside out, next to a correct one."""
        big = sphere_mesh(0.3, resolution=60)
        small = sphere_mesh(0.028, resolution=20).translate((0.45, 0.0, 0.0))
        small.triangles = o3d.utility.Vector3iVector(np.asarray(small.triangles)[:, [0, 2, 1]])
        result = watertight_remesh(big + small, 0.006, min_fraction=1e-4)
        self.assertTrue(closed_and_manifold(result))
        labels, sizes, _ = result.cluster_connected_triangles()
        self.assertEqual(len(sizes), 2)
        expected = 4 * np.pi * (0.3 ** 2 + 0.028 ** 2)
        self.assertAlmostEqual(result.get_surface_area(), expected, delta=0.02 * expected)

    def test_noisy_surface_without_spikes(self):
        """The normal of one triangle must not decide the side at its edges and vertices."""
        noisy = sphere_mesh(noise=0.002, resolution=60)
        result = watertight_remesh(noisy, 0.004)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(noisy))
        distance = scene.compute_distance(o3d.core.Tensor(np.asarray(result.vertices, dtype=np.float32))).numpy()
        self.assertEqual(int(np.sum(distance > 0.6 * 0.004)), 0)

    def test_clip_below_makes_a_flat_closed_cut(self):
        result = watertight_remesh(sphere_mesh(), 0.01, clip_below=0.0)
        self.assertTrue(closed_and_manifold(result))
        self.assertGreater(np.asarray(result.vertices)[:, 2].min(), -0.011)


class SmoothingTest(unittest.TestCase):
    def test_bilateral_smoothing_reduces_facet_noise_and_keeps_the_size(self):
        noisy = watertight_remesh(sphere_mesh(noise=0.002), 0.008)
        before = facet_noise(np.asarray(noisy.vertices), np.asarray(noisy.triangles).astype(np.int64), 0.02)
        smoothed = bilateral_smooth(noisy, 0.015)
        after = facet_noise(np.asarray(smoothed.vertices), np.asarray(smoothed.triangles).astype(np.int64), 0.02)
        self.assertLess(np.median(after), 0.6 * np.median(before))
        radius = np.linalg.norm(np.asarray(smoothed.vertices), axis=1)
        self.assertAlmostEqual(float(np.median(radius)), 0.3, delta=0.003)

    @staticmethod
    def noisy_plane(spacing=0.001, size=120, tilt=0.05, seed=1):
        """Faces on a plane (centroids on a grid) whose normals carry a random tilt."""
        u, v = np.meshgrid(np.arange(size) * spacing, np.arange(size) * spacing)
        centroids = np.column_stack([u.ravel(), v.ravel(), np.zeros(u.size)])
        rng = np.random.default_rng(seed)
        normals = np.column_stack([rng.normal(0, tilt, u.size), rng.normal(0, tilt, u.size), np.ones(u.size)])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        inner = np.all((centroids[:, :2] > 0.025) & (centroids[:, :2] < size * spacing - 0.025), axis=1)
        return centroids, normals, np.full(u.size, spacing ** 2), inner

    @staticmethod
    def residual_tilt(normals, inner):
        return float(np.sqrt(np.mean(normals[inner, 0] ** 2 + normals[inner, 1] ** 2)))

    def test_the_scale_sets_the_smoothing_whatever_the_tessellation(self):
        # 1 mm faces: a support of 2 x 6 mm holds about 450 of them. A fixed cap of 64
        # neighbours cut it to about 4.5 mm, and the scale stopped mattering.
        centroids, normals, areas, inner = self.noisy_plane()
        fine = bilateral_normals(centroids, normals, areas, 0.002, 10.0, 1)
        broad = bilateral_normals(centroids, normals, areas, 0.006, 10.0, 1)
        self.assertGreater(self.residual_tilt(fine, inner) / self.residual_tilt(broad, inner), 2.2)

    def test_cell_sources_average_like_the_faces(self):
        centroids, normals, areas, inner = self.noisy_plane()
        exact = bilateral_normals(centroids, normals, areas, 0.006, 10.0, 1, max_sources=10 ** 6)
        cells = bilateral_normals(centroids, normals, areas, 0.006, 10.0, 1, max_sources=24)
        self.assertLess(self.residual_tilt(cells, inner), 1.5 * self.residual_tilt(exact, inner))

    def test_cell_sources_keep_a_crease(self):
        # two faces of a right-angle fold; cells of about 4 mm straddle the crease
        u, v = np.meshgrid(np.arange(60) * 0.001, np.arange(60) * 0.001)
        flat = np.column_stack([u.ravel(), v.ravel(), np.zeros(u.size)])
        wall = np.column_stack([u.ravel(), np.zeros(u.size), v.ravel() + 0.0005])
        centroids = np.vstack([flat + [0, 0.0005, 0], wall])
        normals = np.vstack([np.tile([0.0, 0.0, 1.0], (u.size, 1)), np.tile([0.0, 1.0, 0.0], (u.size, 1))])
        filtered = bilateral_normals(centroids, normals, np.full(len(centroids), 1e-6), 0.006, 0.35, 4,
                                     max_sources=24)
        away = np.concatenate([flat[:, 1] > 0.006, wall[:, 2] > 0.006])
        angle = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", filtered, normals), -1, 1)))
        self.assertLess(float(angle[away].max()), 2.0)

    def test_automatic_vertex_passes_follow_the_grid(self):
        # the vertex update spreads by about one ring per pass: half the edge needs four times the passes
        counts = {}
        for grid in (0.004, 0.002):
            mesh = watertight_remesh(sphere_mesh(radius=0.1, resolution=60), grid)
            triangles = np.asarray(mesh.triangles).astype(np.int64)
            relaxed = tangential_relaxation(np.asarray(mesh.vertices), triangles, 5)    # as bilateral_smooth does
            counts[grid] = vertex_passes(0, relaxed, triangles)
        self.assertAlmostEqual(counts[0.004], 15, delta=2)
        self.assertAlmostEqual(counts[0.002] / counts[0.004], 4.0, delta=0.6)
        self.assertEqual(vertex_passes(7, np.zeros((3, 3)), np.array([[0, 1, 2]])), 7)

    def test_automatic_vertex_passes_follow_a_wide_scale(self):
        mesh = watertight_remesh(sphere_mesh(radius=0.1, resolution=60), 0.004)
        vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles).astype(np.int64)
        at_6, at_12, at_24 = (vertex_passes(0, vertices, triangles, scale) for scale in (6.0, 12.0, 24.0))
        self.assertEqual(at_6, at_12)                                    # unchanged up to 12 mm
        self.assertAlmostEqual(at_24 / at_12, 4.0, delta=0.3)            # (24 / 12)^2

    def test_facet_noise_uses_every_face_within_the_radius(self):
        mesh = sphere_mesh(radius=0.1, resolution=120, noise=0.0004)
        vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles).astype(np.int64)
        centroids, areas, normals = face_geometry(vertices, triangles)
        measured = facet_noise(vertices, triangles, 0.015, sample=300, seed=2)
        chosen = np.sort(np.random.default_rng(2).choice(len(triangles), 300, replace=False))
        inside = np.linalg.norm(centroids[chosen][:, None, :] - centroids[None, :, :], axis=2) <= 0.015
        self.assertGreater(int(inside.sum(axis=1).min()), 100)
        mean = (inside * areas[None, :]) @ normals
        mean /= np.linalg.norm(mean, axis=1, keepdims=True)
        truth = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", mean, normals[chosen]), -1, 1)))
        np.testing.assert_allclose(measured, truth, atol=1e-6)

    def test_quality_report_of_a_smooth_sphere(self):
        report = mesh_quality(sphere_mesh(resolution=80), 60.0, 0.02)
        self.assertTrue(report["closed"])
        self.assertAlmostEqual(report["volume_l"], 4 / 3 * np.pi * 0.3 ** 3 * 1000, delta=3.0)
        self.assertTrue(report["rf"]["smooth_by_rayleigh"])


class MeshSmoothingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # a sphere of 10 cm with 0.4 mm of noise, remeshed on a 3 mm grid
        cls.noisy = watertight_remesh(sphere_mesh(radius=0.1, resolution=120, noise=0.0004), 0.003)
        cls.vertices = np.asarray(cls.noisy.vertices).copy()
        cls.triangles = np.asarray(cls.noisy.triangles).astype(np.int64)
        cls.noise_before = float(np.median(facet_noise(cls.vertices, cls.triangles, 0.01)))

    def smooth(self, **values):
        return smooth_mesh(self.noisy, SmoothingConfig(**values))

    def noise(self, mesh):
        return float(np.median(facet_noise(np.asarray(mesh.vertices), self.triangles, 0.01)))

    def test_the_deviation_limit_holds(self):
        mesh, statistics = self.smooth(scale_mm=6.0, rounds=2, max_deviation_mm=0.2)
        distance = DeviationLimit(self.noisy, 0.0002).distance(np.asarray(mesh.vertices))
        self.assertLessEqual(float(distance.max()), 0.0002 + 1e-6)
        self.assertLessEqual(statistics["deviation_max_mm"], 0.2 + 1e-3)
        self.assertIn("held_at_limit_fraction", statistics)

    def test_smoother_with_the_volume_and_the_shape_kept(self):
        mesh, statistics = self.smooth()
        self.assertLess(self.noise(mesh), 0.6 * self.noise_before)
        self.assertLess(abs(statistics["volume_change_percent"]), 0.2)
        radius = np.linalg.norm(np.asarray(mesh.vertices), axis=1)
        self.assertAlmostEqual(float(np.median(radius)), 0.1, delta=0.0005)
        self.assertTrue(closed_and_manifold(mesh))
        self.assertNotIn("held_at_limit_fraction", statistics)          # no limit by default

    def test_more_rounds_smooth_more(self):
        # the strength must follow the parameter: stripes 1 mm high and 3 cm apart (like the scan-row
        # stripes of tt11) fade round after round, and the surface moves farther from the input
        sphere = sphere_mesh(radius=0.1, resolution=120)
        vertices = np.asarray(sphere.vertices)
        height = np.arcsin(np.clip(vertices[:, 2] / 0.1, -1, 1)) * 0.1          # arc length from the equator
        vertices *= (1.0 + 0.001 * np.sin(2 * np.pi * height / 0.03) / 0.1)[:, None]
        sphere.vertices = o3d.utility.Vector3dVector(vertices)
        striped = watertight_remesh(sphere, 0.003)

        def stripes(mesh):
            return float(np.std(np.linalg.norm(np.asarray(mesh.vertices), axis=1)))

        results = [smooth_mesh(striped, SmoothingConfig(rounds=rounds, finish_rounds=0)) for rounds in (1, 2, 4)]
        amplitude = [stripes(striped)] + [stripes(mesh) for mesh, _ in results]
        moved = [statistics["deviation_mm"]["median"] for _, statistics in results]
        self.assertTrue(all(b < 0.85 * a for a, b in zip(amplitude, amplitude[1:])), amplitude)
        self.assertTrue(all(b > a for a, b in zip(moved, moved[1:])), moved)

    def test_the_schedule_is_what_the_parameters_say(self):
        from bodyscan.meshing.smoother import MeshSmoother
        schedule = lambda **values: MeshSmoother(self.noisy, SmoothingConfig(**values)).schedule()
        self.assertEqual(schedule(), [12.0] * 4 + [6.0] * 2)
        self.assertEqual(schedule(scale_mm=6.0), [6.0] * 4)              # finishing only below the main scale
        self.assertEqual(schedule(rounds=2, finish_rounds=0), [12.0] * 2)
        mesh, statistics = self.smooth(rounds=0, finish_rounds=0)
        np.testing.assert_array_equal(np.asarray(mesh.vertices), self.vertices)
        self.assertEqual(statistics["deviation_max_mm"], 0.0)


if __name__ == "__main__":
    unittest.main()
