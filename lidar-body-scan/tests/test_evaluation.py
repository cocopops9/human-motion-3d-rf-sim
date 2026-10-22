"""Fit-quality evaluation without ground truth: pixel classes, the per-vertex map, kinematics,
heatmaps, and a synthetic take evaluated with its true motion and with a displaced one."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import TemporaryFolder

from bodyscan.evaluation import (DARK, EMPTY, FITTED, NONE, OTHER, PERSON, SILHOUETTE, UNEXPLAINED, UNKNOWN_KIND,
                                 UNOBSERVED, VIOLATION, ClassifyConfig, RayBatch, VertexMap, classify, part_table,
                                 vertex_areas)

HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = unittest.skipUnless(HAS_TORCH, "needs PyTorch (python -m pip install torch)")
_MODEL = {}


def load_test_body():
    if "model" not in _MODEL:
        import torch
        from bodyscan.body.model import BodyModel
        from bodyscan.body.testbody import cached
        _MODEL["model"] = BodyModel(cached(), device="cpu", dtype=torch.float64)
    return _MODEL["model"]


def _batch(measured, kind):
    n = len(measured)
    return RayBatch(np.zeros((n, 3)), np.tile([1.0, 0.0, 0.0], (n, 1)), np.asarray(measured, dtype=np.float64),
                    np.asarray(kind, dtype=np.uint8), np.full(n, 0.01), np.zeros(n))


class ClassifyTest(unittest.TestCase):
    def test_every_case(self):
        cases = [  # (measured, kind, body distance, expected class)
            (2.00, PERSON, 2.01, FITTED),
            (2.00, PERSON, 1.90, VIOLATION),        # the ray went on past the body surface
            (2.00, PERSON, 2.10, UNEXPLAINED),      # measured in front of the body
            (2.00, PERSON, np.inf, UNEXPLAINED),    # measured person, no body along the ray
            (5.00, OTHER, 2.00, VIOLATION),         # the room seen through the body
            (1.00, OTHER, 2.00, UNOBSERVED),        # an object in front of the body
            (2.01, OTHER, 2.00, UNOBSERVED),        # a room surface touching the body (a sole on the floor)
            (np.nan, DARK, 2.00, SILHOUETTE),
            (np.nan, DARK, np.inf, UNEXPLAINED),
            (np.nan, NONE, 2.00, UNOBSERVED),
            (np.nan, UNKNOWN_KIND, 2.00, UNOBSERVED),
            (np.nan, NONE, np.inf, EMPTY),
            (5.00, OTHER, np.inf, EMPTY),
        ]
        batch = _batch([c[0] for c in cases], [c[1] for c in cases])
        result = classify(batch, np.array([c[2] for c in cases]), ClassifyConfig(3.0, 0.02))
        self.assertEqual(result["classes"].tolist(), [c[3] for c in cases])
        self.assertAlmostEqual(result["error"][1], 0.10)
        self.assertTrue(np.isnan(result["error"][3]))

    def test_tolerance_follows_the_noise(self):
        batch = _batch([3.0, 3.0], [PERSON, PERSON])
        batch.sigma[:] = (0.005, 0.03)                          # 3 sigma: 2 cm (minimum) and 9 cm
        result = classify(batch, np.array([2.95, 2.95]), ClassifyConfig(3.0, 0.02))
        self.assertEqual(result["classes"].tolist(), [VIOLATION, FITTED])


class VertexMapTest(unittest.TestCase):
    def test_errors_go_to_the_corners_by_barycentric_weight(self):
        faces = np.array([[0, 1, 2]])
        vertex_map = VertexMap(4, 2)
        # frame 0: one pixel at the first corner (error 1 cm), one at the centre (error 4 cm)
        vertex_map.add(0, faces, np.array([0, 0]), np.array([[0.0, 0.0], [1 / 3, 1 / 3]]),
                       np.array([FITTED, VIOLATION]), np.array([0.01, 0.04]))
        e = vertex_map.error[0]
        self.assertAlmostEqual(float(e[0]), (0.01 * 1.0 + 0.04 / 3) / (1.0 + 1 / 3), places=6)
        self.assertAlmostEqual(float(e[1]), 0.04, places=6)
        self.assertTrue(np.isnan(e[3]))                                   # vertex 3 is in no triangle hit
        summary = vertex_map.summary()
        self.assertEqual(summary["coverage"].tolist(), [0.5, 0.5, 0.5, 0.0])
        self.assertTrue(np.isnan(summary["median"][3]))
        self.assertAlmostEqual(float(summary["violation_rate"][1]), 1.0)
        self.assertAlmostEqual(float(summary["violation_rate"][0]), (1 / 3) / (1.0 + 1 / 3))

    def test_unranged_pixels_do_not_count_as_errors(self):
        faces = np.array([[0, 1, 2]])
        vertex_map = VertexMap(3, 1)
        vertex_map.add(0, faces, np.array([0]), np.array([[0.2, 0.2]]), np.array([SILHOUETTE]), np.array([np.nan]))
        self.assertTrue(np.isnan(vertex_map.error[0]).all())
        self.assertEqual(vertex_map.summary()["silhouette_coverage"].tolist(), [1.0, 1.0, 1.0])

    def test_part_table_weights_by_area_and_skips_unobserved_parts(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 0], [6, 5, 0], [5, 6, 0]], dtype=float)
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        areas = vertex_areas(vertices, faces)
        self.assertTrue(np.allclose(areas, 0.5 / 3))
        summary = {"coverage": np.array([1, 1, 1, 0, 0, 0.0]), "median": np.array([0.01, 0.02, 0.03] + [np.nan] * 3),
                   "p95": np.array([0.02, 0.03, 0.04] + [np.nan] * 3), "violation_rate": np.zeros(6),
                   "unexplained_rate": np.zeros(6)}
        table = part_table(summary, np.array([0, 0, 0, 1, 1, 1]), areas, ("seen", "hidden"))
        self.assertAlmostEqual(table["seen"]["median_mm"], 20.0)
        self.assertNotIn("median_mm", table["hidden"])
        self.assertEqual(table["hidden"]["coverage"], 0.0)


class KinematicsTest(unittest.TestCase):
    def test_jitter_and_turns(self):
        from bodyscan.evaluation.kinematics import kinematic_metrics
        times = np.arange(0.0, 10.0, 0.05)
        joints = np.zeros((len(times), 22, 3))
        joints[:, :, 0] = 0.3 * np.sin(2 * np.pi * 1.0 * times)[:, None]      # a smooth 1 Hz swing
        smooth = kinematic_metrics(times, joints)
        rng = np.random.default_rng(0)
        noisy = kinematic_metrics(times, joints + rng.normal(0.0, 0.01, joints.shape))
        self.assertLess(smooth["high_frequency_share_mean"], 0.01)
        self.assertGreater(noisy["high_frequency_share_mean"], 5 * smooth["high_frequency_share_mean"])
        yaw = np.where(times < 5.0, 0.0, np.pi)                                # a half turn in one frame
        root = np.stack([np.zeros_like(yaw), np.zeros_like(yaw), yaw], axis=1)
        turned = kinematic_metrics(times, joints, root_rotation=root)
        self.assertEqual(turned["turns"]["frames_over_limit"], 1)
        self.assertGreater(turned["turns"]["max_deg_s"], 3000.0)

    def test_hinges_and_joint_speeds(self):
        from bodyscan.body import skeleton
        from bodyscan.evaluation.kinematics import kinematic_metrics
        times = np.arange(0.0, 1.0, 0.05)
        body = np.zeros((len(times), 21, 3))
        elbow = skeleton.JOINT["left_elbow"] - 1
        body[10:, elbow, 2] = np.radians(100.0)                   # bent out of its plane at 2000 deg/s
        result = kinematic_metrics(times, np.zeros((len(times), 22, 3)), np.zeros((len(times), 3)), body)
        self.assertGreater(max(result["hinges_out_of_plane"].values()), 0.4)
        self.assertGreater(result["joint_speed_over_limit"]["left_elbow"], 0.0)
        self.assertEqual(result["joint_speed_over_limit"]["right_knee"], 0.0)


class HeatmapTest(unittest.TestCase):
    def test_colours_and_ply(self):
        import open3d as o3d
        from bodyscan.evaluation.heatmap import colours, write_coloured_ply
        rgb = colours(np.array([0.0, 0.05, 1.0, np.nan]), 0.0, 0.05)
        self.assertEqual(rgb[0].tolist(), [0, 104, 55])
        self.assertEqual(rgb[1].tolist(), [190, 20, 30])
        self.assertEqual(rgb[2].tolist(), [190, 20, 30])                     # capped at the top of the scale
        self.assertEqual(rgb[3].tolist(), [128, 128, 128])                   # no value: grey
        folder = TemporaryFolder()
        try:
            vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
            path = write_coloured_ply(folder.path / "t.ply", vertices, np.array([[0, 1, 2]]), rgb[:3])
            mesh = o3d.io.read_triangle_mesh(str(path))
            self.assertEqual(len(mesh.triangles), 1)
            self.assertTrue(np.allclose(np.asarray(mesh.vertex_colors)[0], np.array([0, 104, 55]) / 255.0, atol=1e-3))
        finally:
            folder.cleanup()


def make_synthetic_take(folder) -> dict:
    """A simulated standing person (test body, 512x20 LiDAR at the origin), its segmentation, the true
    motion as a tracked motion, the same motion moved 8 cm towards the sensor, and the avatar.
    Returns the paths (keys: take, segments, true_motion, moved_motion, avatar)."""
    from bodyscan.body.testbody import cached
    from bodyscan.dynamic import motions
    from bodyscan.dynamic.avatar import Avatar
    from bodyscan.dynamic.motions import PoseSequence
    from bodyscan.dynamic.segment import MotionSegmenter, SegmentedRecording
    from bodyscan.dynamic.simulate import MotionSensor, Room, load_truth, record
    from bodyscan.io import NpzRecording
    folder = Path(folder)
    model = load_test_body()
    shaped = model.shaped()
    motion = motions.standing(shaped, duration=0.7, xy=(2.0, 0.0), heading_deg=180.0)
    take = folder / "take"
    record(take, shaped, motion, MotionSensor(columns=512, frame_rate=20.0), Room(), background_frames=5, quiet=True)
    segments = folder / "segments"
    MotionSegmenter(NpzRecording(take)).run(segments)
    segmented = SegmentedRecording(segments)
    frames = [segmented.load(k) for k in range(len(segmented))]
    times = np.array([f.time for f in frames])
    index = np.array([f.index for f in frames])
    truth, _ = load_truth(take)
    tracked = truth.sample(times)
    _, joints = tracked.posed(shaped)
    tracked.save(folder / "true_motion.npz", frame_index=index, joints=joints)
    shift = np.array([-0.08, 0.0, 0.0])                                 # towards the sensor at the origin
    moved = PoseSequence(times, tracked.root_rotation, tracked.root_position + shift, tracked.body)
    moved.save(folder / "moved_motion.npz", frame_index=index, joints=joints + shift)
    avatar = Avatar.from_model(model)
    avatar.model_path = str(cached())
    avatar.save(folder / "avatar.npz")
    return {"take": take, "segments": segments, "true_motion": folder / "true_motion.npz",
            "moved_motion": folder / "moved_motion.npz", "avatar": folder / "avatar.npz"}


@needs_torch
class TakeEvaluationTest(unittest.TestCase):
    """A simulated standing person tracked perfectly (the true motion), and the same motion moved 8 cm
    towards the sensor: the evaluation must tell them apart from the measurements alone."""

    @classmethod
    def setUpClass(cls):
        cls.folder = TemporaryFolder()
        cls.paths = make_synthetic_take(cls.folder.path)
        cls.segments = cls.paths["segments"]

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def _evaluate(self, name):
        from bodyscan.evaluation.take import TakeEvaluationConfig, evaluate_take
        return evaluate_take(self.folder.path / name, self.segments, self.folder.path / "avatar.npz",
                             config=TakeEvaluationConfig(sigma=0.008))

    def test_true_motion_fits_and_displaced_motion_violates(self):
        good = self._evaluate("true_motion.npz")
        bad = self._evaluate("moved_motion.npz")
        g, b = good["shares_of_ranged_pixels"], bad["shares_of_ranged_pixels"]
        self.assertGreater(g["fitted"], 0.85)
        self.assertLess(g["violation"], 0.05)
        self.assertGreater(b["violation"], 0.4)
        self.assertLess(good["points"]["distance_median_mm"], 12.0)
        # the nearest-surface distance of a point understates a displacement along the ray (oblique
        # surfaces, other parts nearby); the ray classes above are the sharper test
        self.assertGreater(bad["points"]["distance_median_mm"], 2.0 * good["points"]["distance_median_mm"])
        # one sensor sees one side: part of the body is never observed and has no error value
        coverage = good["arrays"]["coverage"]
        self.assertGreater((coverage == 0).mean(), 0.2)
        self.assertGreater((coverage > 0.5).mean(), 0.1)            # 512 columns: a sparse sensor
        self.assertTrue(np.all(np.isnan(good["arrays"]["median"][coverage == 0])))
        self.assertLess(np.nanmedian(good["arrays"]["median"]), 0.015)

    def test_results_are_written(self):
        import open3d as o3d
        from bodyscan.evaluation.take import write_take_results
        result = self._evaluate("true_motion.npz")
        out = self.folder.path / "evaluation"
        paths = write_take_results(result, out)
        summary = json.loads(paths["json"].read_text())
        self.assertEqual(summary["frames_evaluated"], result["frames_evaluated"])
        self.assertIn("pelvis", summary["parts"])
        arrays = np.load(paths["npz"])
        self.assertEqual(arrays["frame_error_mm"].shape, (result["frames_evaluated"], len(arrays["coverage"])))
        mesh = o3d.io.read_triangle_mesh(str(paths["median"]))
        self.assertEqual(len(mesh.vertices), len(arrays["coverage"]))
        self.assertTrue(mesh.has_vertex_colors())
        self.assertIn("grey", paths["report"].read_text())


if __name__ == "__main__":
    unittest.main()


class LayoutTest(unittest.TestCase):
    def test_layouts_place_the_sensors_around_the_zone(self):
        from bodyscan.sensors.layouts import LAYOUTS, build_rig
        rig = build_rig(LAYOUTS["triangle-120"], zone_distance=2.3)
        self.assertEqual(rig.names, ["ouster", "helios", "xtion"])
        zone = np.array([2.3, 0.0])
        helios, camera = rig.placement("helios"), rig.placement("xtion")
        self.assertAlmostEqual(float(np.linalg.norm(helios.position[:2] - zone)), 3.6, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(camera.position[:2] - zone)), 2.2, places=6)
        # 120 deg apart, seen from the zone centre
        a = np.arctan2(*(np.array([0.0, 0.0]) - zone)[::-1])
        b = np.arctan2(*(camera.position[:2] - zone)[::-1])
        self.assertAlmostEqual(abs(np.degrees(np.angle(np.exp(1j * (b - a))))), 120.0, places=6)
        # the camera looks at the zone centre at the aim height
        forward = camera.world_from_sensor[:3, :3] @ np.array([1.0, 0.0, 0.0])
        toward = np.array([2.3, 0.0, 0.9]) - camera.position
        self.assertGreater(float(forward @ toward / np.linalg.norm(toward)), 0.9999)
        # the Helios stands level
        self.assertAlmostEqual(float((helios.world_from_sensor[:3, :3] @ np.array([0.0, 0.0, 1.0]))[2]), 1.0)
        self.assertEqual(build_rig(LAYOUTS["ouster"]).names, ["ouster"])


@needs_torch
class CoverageTest(unittest.TestCase):
    def test_more_viewpoints_see_more_of_the_body(self):
        from bodyscan.dynamic import motions
        from bodyscan.evaluation.coverage import CoverageConfig, rig_coverage
        from bodyscan.sensors.layouts import LAYOUTS, build_rig
        shaped = load_test_body().shaped()
        motion = motions.standing(shaped, duration=1.0, xy=(2.3, 0.0), heading_deg=90.0)     # side-on
        config = CoverageConfig(rate=2.0)
        alone = rig_coverage(build_rig(LAYOUTS["ouster"]), shaped, motion, config)
        three = rig_coverage(build_rig(LAYOUTS["triangle-120"]), shaped, motion, config)
        a = alone["body_visible_share"]["union"]["mean"]
        t = three["body_visible_share"]
        self.assertLess(a, 0.6)
        self.assertGreater(t["union"]["mean"], a + 0.3)
        self.assertGreaterEqual(t["union"]["mean"], max(t[s]["mean"] for s in ("ouster", "helios", "xtion")))
        # the Xtion samples the body far more densely than the Helios
        torso = three["parts"]["torso_upper"]
        self.assertLess(torso["spacing_mm_xtion"], 0.25 * torso["spacing_mm_helios"])
