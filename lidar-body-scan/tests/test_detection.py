"""Person cascade and the detect pipeline on synthetic scenes."""

import unittest

import numpy as np
import open3d as o3d

from bodyscan.detection.human import HumanCascade, HumanConfig, OccupancyImage, ViewSampling
from bodyscan.pipelines.detect import DetectConfig, DetectPipeline
from bodyscan.synthetic import SyntheticSensor, box, chair, person, scenario, stand
from tests.helpers import TemporaryFolder


def view_of(mesh, position, yaw_deg, sensor: SyntheticSensor):
    """Points of a mesh seen by the sensor (floor frame), thinned like the detector does."""
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.radians(yaw_deg))), center=(0, 0, 0))
    mesh.translate((position[0], position[1], 0.0))
    range_mm = sensor.scan([mesh]).astype(np.float64) / 1000.0
    directions = sensor.directions @ sensor.world_from_sensor.T
    points = (directions * range_mm[..., None])[range_mm > 0] + [0.0, 0.0, sensor.height]
    points = points[points[:, 2] > 0.10]
    return np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(0.02).points)


class OccupancyImageTest(unittest.TestCase):
    def test_rectangles_and_widths(self):
        lateral, height = np.meshgrid(np.arange(-0.2, 0.2, 0.01), np.arange(0.0, 1.8, 0.01))
        image = OccupancyImage(lateral.ravel(), height.ravel(), 1.8, 0.03, 0.03)
        self.assertGreater(image.fraction(-0.05, 0.05, 0.2, 0.8), 0.95)          # inside
        self.assertEqual(image.fraction(0.2, 0.4, 0.2, 0.8), 0.0)                 # beside
        self.assertAlmostEqual(image.width(0.4, 0.6), 0.42, delta=0.04)
        self.assertAlmostEqual(image.area, 0.4 * 1.8, delta=0.08)


class CascadeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sensor = SyntheticSensor(rows=128, columns=2048, height=1.15, noise=0.004)
        horizontal = 2 * np.pi / cls.sensor.columns
        vertical = np.radians(cls.sensor.vertical_fov_deg / (cls.sensor.rows - 1))
        cls.cascade = HumanCascade(HumanConfig(), ViewSampling(horizontal, vertical, 0.02))
        cls.origin = np.array([0.0, 0.0, cls.sensor.height])

    def passes(self, mesh, position, yaw):
        return self.cascade.frame(view_of(mesh, position, yaw, self.sensor), self.origin).passed

    def test_people_pass_from_every_side(self):
        results = [self.passes(person(), (distance, 0.2), yaw) for distance in (1.2, 3.0) for yaw in range(0, 360, 60)]
        self.assertGreaterEqual(sum(results), len(results) - 1)

    def test_furniture_and_stands_fail(self):
        self.assertFalse(self.passes(chair(), (1.5, 0.0), 30))
        self.assertFalse(self.passes(stand(), (1.5, 0.0), 0))
        self.assertFalse(self.passes(box((0.5, 0.4, 1.5), (0.0, 0.0)), (1.5, 0.0), 20))
        self.assertFalse(self.passes(box((1.2, 0.6, 0.75), (0.0, 0.0)), (1.5, 0.0), 0))


class RotationWithStillPartsTest(unittest.TestCase):
    """A turntable recording: still, one lap at 6 deg/s, still again."""

    @classmethod
    def setUpClass(cls):
        from bodyscan.detection.segmentation import Cluster
        from bodyscan.detection.tracking import Track
        sensor = SyntheticSensor(rows=128, columns=1024, height=1.15, noise=0.003)
        times = np.linspace(0.0, 80.0, 36)
        angles = np.clip(times - 10.0, 0.0, 60.0) * 6.0
        origin = np.array([0.0, 0.0, sensor.height])
        clusters = [Cluster(k, t, view_of(person(), (1.5, 0.3), a, sensor), origin)
                    for k, (t, a) in enumerate(zip(times, angles))]
        cls.track = Track(0, clusters)

    def test_still_pairs_do_not_hide_the_rotation(self):
        from bodyscan.detection.rotation import RotationAnalyzer, RotationConfig
        result = RotationAnalyzer(RotationConfig()).analyze(self.track)
        self.assertTrue(result.rotating, result.reason)
        self.assertGreater(result.still_pairs, 0)
        np.testing.assert_allclose(result.axis, (1.5, 0.3), atol=0.02)
        self.assertAlmostEqual(abs(result.speed_deg_s), 6.0, delta=0.3)
        self.assertAlmostEqual(result.turned_deg, 360.0, delta=45.0)

    def test_a_whole_lap_can_be_required(self):
        from bodyscan.detection.rotation import RotationAnalyzer, RotationConfig
        self.assertTrue(RotationAnalyzer(RotationConfig(min_total_turn=320.0)).analyze(self.track).rotating)
        result = RotationAnalyzer(RotationConfig(min_total_turn=720.0)).analyze(self.track)
        self.assertFalse(result.rotating)
        self.assertIn("turned", result.reason)


class DetectPipelineTest(unittest.TestCase):
    """Synthetic scene: a person and a chair turning on two platforms, a person
    standing still, a desk, a cabinet and a stand."""

    @classmethod
    def setUpClass(cls):
        cls.folder = TemporaryFolder()
        cls.run_dir = scenario("detect", cls.folder.path / "scene", frames=40,
                               sensor=SyntheticSensor(rows=128, columns=1024))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def detect(self, rotation, human, near=None):
        config = DetectConfig()
        config.select.near = near
        config.frames.max_frames = 30
        config.select.rotation, config.select.human = rotation, human
        ctx = DetectPipeline(config).run(input_dir=self.run_dir, out=self.folder.path / "det")
        return [r for r in ctx.results if r["selected"]]

    def test_rotating_objects_and_their_axes(self):
        selected = self.detect(True, False)
        centres = sorted([tuple(np.round(r["center"], 2)) for r in selected])
        self.assertEqual(len(centres), 2)
        np.testing.assert_allclose(centres[0], (0.0, -2.0), atol=0.02)      # chair B
        np.testing.assert_allclose(centres[1], (1.2, 0.1), atol=0.02)       # person A

    def test_people(self):
        selected = self.detect(False, True)
        centres = sorted([tuple(np.round(r["center"], 2)) for r in selected])
        self.assertEqual(len(centres), 2)
        np.testing.assert_allclose(centres[0], (-1.0, 1.0), atol=0.08)      # person C, standing still
        np.testing.assert_allclose(centres[1], (1.2, 0.1), atol=0.08)       # person A

    def test_rotating_people(self):
        selected = self.detect(True, True)
        self.assertEqual(len(selected), 1)
        np.testing.assert_allclose(selected[0]["center"], (1.2, 0.1), atol=0.02)

    def test_near_a_point(self):
        selected = self.detect(False, True, near=(-1.0, 1.0))
        self.assertEqual(len(selected), 1)                                  # person C only
        np.testing.assert_allclose(selected[0]["center"], (-1.0, 1.0), atol=0.08)

    def test_without_tests_every_object_is_reported_with_both(self):
        config = DetectConfig()
        config.frames.max_frames = 30
        ctx = DetectPipeline(config).run(input_dir=self.run_dir, out=self.folder.path / "det_all")
        self.assertTrue(all(r["selected"] for r in ctx.results))
        self.assertTrue(all(r["human"] is not None and r["rotation"] is not None for r in ctx.results))
        people = sorted(tuple(np.round(r["center"], 1)) for r in ctx.results if r["human"]["passed"])
        self.assertEqual(len(people), 2)

    def test_point_cloud_folder_with_empty_scene(self):
        """The same scene as one point cloud per frame (no range images), with background/ clouds."""
        from bodyscan.io import NpzRecording
        recording = NpzRecording(self.run_dir)
        folder = self.folder.path / "clouds"
        for kind, count, load in (("frames", len(recording), recording.load),
                                  ("background", recording.background_count(), recording.load_background)):
            (folder / kind).mkdir(parents=True, exist_ok=True)
            for k in range(count):
                np.savez(folder / kind / f"{kind}_{k:05d}.npz", points=recording.points(load(k)))
        config = DetectConfig()
        config.frames.max_frames = 30
        config.select.human = True
        ctx = DetectPipeline(config).run(input_dir=folder, out=self.folder.path / "det_clouds")
        self.assertIn("point clouds", ctx.report["segmentation"])
        centres = sorted(tuple(np.round(r["center"], 2)) for r in ctx.results if r["selected"])
        self.assertEqual(len(centres), 2)
        np.testing.assert_allclose(centres[0], (-1.0, 1.0), atol=0.08)


if __name__ == "__main__":
    unittest.main()
