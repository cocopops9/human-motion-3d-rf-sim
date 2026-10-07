"""Moving people (bodyscan.body, bodyscan.dynamic): rotations, the conventions of
the body model, the procedural motions, the simulated recording and its
segmentation, the export; and, slow, the avatar fit and the tracking.

The body tests use the procedural test body (built once, about 10 s, and
cached in the temporary folder) and need PyTorch; they are skipped without it.

    python -m unittest tests.test_motion
    BODYSCAN_SLOW_TESTS=1 python -m unittest tests.test_motion
"""

from __future__ import annotations

import importlib.util
import json
import unittest

import numpy as np

from bodyscan.body import skeleton
from bodyscan.body.rotations import (axis_angle_to_matrix_np, interpolate_positions, interpolate_rotations,
                                     matrix_to_axis_angle_np, matrix_to_quaternion_np, quaternion_to_matrix_np,
                                     rotation_z_np)
from tests.helpers import TemporaryFolder, slow

HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = unittest.skipUnless(HAS_TORCH, "needs PyTorch (python -m pip install torch)")

_MODEL = {}


def load_test_body():
    """The test body as a BodyModel on the CPU in float64 (loaded once per test run)."""
    if "model" not in _MODEL:
        import torch
        from bodyscan.body.model import BodyModel
        from bodyscan.body.testbody import cached
        _MODEL["model"] = BodyModel(cached(), device="cpu", dtype=torch.float64)
    return _MODEL["model"]


J = skeleton.JOINT


# ----------------------------------------------------------------------------
# Rotations and interpolation (numpy only)
# ----------------------------------------------------------------------------

class RotationTest(unittest.TestCase):
    def test_round_trips(self):
        rng = np.random.default_rng(0)
        axis = rng.normal(size=(300, 3))
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        angle = rng.uniform(0.0, np.pi, 300)
        angle[:3] = (0.0, 1e-9, np.pi - 1e-7)
        matrices = axis_angle_to_matrix_np(axis * angle[:, None])
        self.assertTrue(np.allclose(matrices @ matrices.transpose(0, 2, 1), np.eye(3), atol=1e-12))
        self.assertTrue(np.allclose(np.linalg.det(matrices), 1.0))
        self.assertTrue(np.allclose(axis_angle_to_matrix_np(matrix_to_axis_angle_np(matrices)), matrices, atol=1e-6))
        self.assertTrue(np.allclose(quaternion_to_matrix_np(matrix_to_quaternion_np(matrices)), matrices, atol=1e-9))

    def test_interpolation_passes_through_the_samples_with_continuous_velocity(self):
        rng = np.random.default_rng(1)
        times = np.cumsum(np.r_[0.0, rng.uniform(0.03, 0.07, 30)])            # uneven frame times
        values = np.stack([np.sin(3 * times), -2 * np.cos(2 * times)], axis=1)
        self.assertTrue(np.allclose(interpolate_positions(times, values, times), values, atol=1e-12))
        h = 1e-6
        for k in range(2, 28):
            at = interpolate_positions(times, values, [times[k]])
            left = (at - interpolate_positions(times, values, [times[k] - h])) / h
            right = (interpolate_positions(times, values, [times[k] + h]) - at) / h
            self.assertTrue(np.allclose(left, right, atol=1e-3), k)          # C1: no velocity step at a sample

    def test_rotation_interpolation_follows_a_steady_turn(self):
        times = np.arange(0.0, 2.0, 0.05)
        turn = np.zeros((len(times), 3))
        turn[:, 2] = 2.0 * times                                            # 2 rad/s about z, beyond pi
        query = times[1:-2] + 0.025
        matrices = axis_angle_to_matrix_np(interpolate_rotations(times, turn, query))
        yaw = np.unwrap(np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0]))
        self.assertLess(np.abs(yaw - 2.0 * query).max(), np.radians(0.2))


# ----------------------------------------------------------------------------
# Body model
# ----------------------------------------------------------------------------

@needs_torch
class BodyModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_test_body()
        cls.shaped = cls.model.shaped()

    def joints(self, **degrees):
        """World joints (55, 3), pelvis at the origin, with some joint rotations (axis-angle in degrees)."""
        import torch
        body = np.zeros((1, 21, 3))
        for name, value in degrees.items():
            body[0, J[name] - 1] = np.radians(value)
        with torch.no_grad():
            posed = self.shaped.pose(self.model.rotations(body=body), None, self.model.tensor(np.zeros((1, 3))))
        return posed.joints[0].numpy()

    def test_documented_sign_conventions(self):
        # world frame: the body faces +x, its left side is +y, z is up (bodyscan.body.skeleton)
        rest = self.joints()
        self.assertTrue(np.allclose(rest[J["pelvis"]], 0.0))
        self.assertGreater(rest[J["left_hip"], 1], rest[J["right_hip"], 1])
        self.assertGreater(rest[J["head"], 2], 0.4)

        knee = self.joints(left_knee=(90, 0, 0))
        self.assertLess(knee[J["left_ankle"], 0] - knee[J["left_knee"], 0], -0.3)       # heel to the buttock
        hip = self.joints(left_hip=(-90, 0, 0))
        self.assertGreater(hip[J["left_knee"], 0] - hip[J["left_hip"], 0], 0.3)        # thigh forward
        ankle = self.joints(left_ankle=(-30, 0, 0))
        self.assertGreater(ankle[J["left_foot"], 2], rest[J["left_foot"], 2] + 0.02)    # toes up
        spine = self.joints(spine1=(40, 0, 0))
        self.assertGreater(spine[J["neck"], 0], rest[J["neck"], 0] + 0.1)               # bend forward
        abduction = self.joints(left_hip=(0, 0, 30))
        self.assertGreater(abduction[J["left_knee"], 1], rest[J["left_knee"], 1] + 0.1)  # thigh outwards
        right_abduction = self.joints(right_hip=(0, 0, -30))
        self.assertLess(right_abduction[J["right_knee"], 1], rest[J["right_knee"], 1] - 0.1)

        down = self.joints(left_shoulder=(0, 0, -60), right_shoulder=(0, 0, 60))
        self.assertLess(down[J["left_elbow"], 2], rest[J["left_elbow"], 2] - 0.1)       # arms down
        self.assertLess(down[J["right_elbow"], 2], rest[J["right_elbow"], 2] - 0.1)
        forward = self.joints(left_shoulder=(0, -60, 0), right_shoulder=(0, 60, 0))
        self.assertGreater(forward[J["left_elbow"], 0], rest[J["left_elbow"], 0] + 0.1)  # arms forward
        self.assertGreater(forward[J["right_elbow"], 0], rest[J["right_elbow"], 0] + 0.1)
        elbows = self.joints(left_elbow=(0, -90, 0), right_elbow=(0, 90, 0))
        self.assertGreater(elbows[J["left_wrist"], 0], rest[J["left_wrist"], 0] + 0.15)   # elbows bend
        self.assertGreater(elbows[J["right_wrist"], 0], rest[J["right_wrist"], 0] + 0.15)

    def test_limits_allow_the_anatomical_bend_only(self):
        # the limits must agree with the signs above: knees bend +x, the left elbow -y, the right elbow +y
        lower, upper = skeleton.limit_arrays()
        for knee in ("left_knee", "right_knee"):
            self.assertGreater(upper[J[knee], 0], np.radians(120))
            self.assertGreater(lower[J[knee], 0], np.radians(-10))
        self.assertLess(lower[J["left_elbow"], 1], np.radians(-120))
        self.assertLess(upper[J["left_elbow"], 1], np.radians(10))
        self.assertGreater(upper[J["right_elbow"], 1], np.radians(120))
        self.assertGreater(lower[J["right_elbow"], 1], np.radians(-10))
        self.assertLess(lower[J["left_hip"], 0], np.radians(-100))                  # hip flexion is -x

    def test_root_rotation_and_position(self):
        import torch
        rotation = self.model.tensor(rotation_z_np(np.pi / 2)[None])            # facing +y
        with torch.no_grad():
            posed = self.shaped.pose(self.model.rotations(count=1), rotation, self.model.tensor([[1.0, 2.0, 0.9]]))
        joints = posed.joints[0].numpy()
        self.assertTrue(np.allclose(joints[J["pelvis"]], (1.0, 2.0, 0.9)))
        self.assertLess(joints[J["left_hip"], 0], 1.0 - 0.03)                  # the left side now points to -x

    def test_regressor_is_affine_and_subdivision_keeps_the_coarse_body(self):
        import torch
        rows = self.model.regressor.sum(dim=1)
        self.assertTrue(torch.allclose(rows, torch.ones_like(rows), atol=1e-6))    # joints move with the mesh
        fine = self.model.shaped(level=1)
        coarse_count = self.shaped.vertex_count
        self.assertEqual(len(fine.faces_np), 4 * len(self.shaped.faces_np))
        self.assertTrue(torch.allclose(fine.vertices[:coarse_count], self.shaped.vertices))
        body = np.radians(np.random.default_rng(2).uniform(-20, 20, (1, 21, 3)))
        rotations = self.model.rotations(body=body)
        position = self.model.tensor(np.zeros((1, 3)))
        with torch.no_grad():
            a = self.shaped.pose(rotations, None, position).vertices[0]
            b = fine.pose(rotations, None, position).vertices[0, :coarse_count]
        self.assertTrue(torch.allclose(a, b, atol=1e-10))
        # every new vertex is the midpoint of an edge of the coarse mesh
        edges = fine.subdivision.edges[0]
        midpoints = 0.5 * (self.shaped.vertices[edges[:, 0]] + self.shaped.vertices[edges[:, 1]])
        self.assertTrue(torch.allclose(fine.vertices[coarse_count:], midpoints))


# ----------------------------------------------------------------------------
# Procedural motions
# ----------------------------------------------------------------------------

@needs_torch
class MotionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bodyscan.dynamic import motions
        cls.motions = motions
        cls.model = load_test_body()
        cls.shaped = cls.model.shaped()
        cls.feet = motions.foot_vertices(cls.shaped)
        cls.walk = motions.walk(cls.shaped, duration=1.6, speed=1.2, stand_before=0.5, stand_after=0.4, rate=120,
                                solve_rate=40)
        cls.jump = motions.jump(cls.shaped, count=1, stand_before=0.5, pause=0.2, stand_after=0.3, rate=240)

    def soles(self, sequence):
        vertices, _ = sequence.posed(self.shaped, subset=self.feet)
        return vertices.astype(np.float64)

    def test_walk_stays_on_the_floor_without_slipping(self):
        soles = self.soles(self.walk)
        self.assertGreater(soles[:, :, 2].min(), -0.005)                       # no foot through the floor
        self.assertLess(np.median(soles[:, :, 2].min(axis=1)), 0.005)          # and a foot on it
        # sole vertices touching the floor in two successive samples do not move (no slip)
        dt = np.diff(self.walk.times)[:, None]
        touching = (soles[1:, :, 2] < 0.002) & (soles[:-1, :, 2] < 0.002)
        speed = np.linalg.norm(np.diff(soles[:, :, :2], axis=0), axis=2) / dt
        self.assertLess(np.median(speed[touching]), 0.05)

    def test_walk_moves_forward_at_about_the_set_speed(self):
        t = self.walk.times
        x = self.walk.root_position[:, 0]
        self.assertGreater(x[-1] - x[0], 0.8)                                  # forward along the heading (+x)
        middle = (t > 0.5 + 0.7) & (t < 0.5 + 1.6 - 0.5)                        # after the speed ramp
        speed = np.diff(x[middle]) / np.diff(t[middle])
        self.assertLess(abs(np.mean(speed) - 1.2) / 1.2, 0.35)
        self.assertLess(np.abs(self.walk.root_position[:, 1]).max(), 0.05)     # straight line

    def test_jump_flight_is_ballistic(self):
        t = self.jump.times
        soles = self.soles(self.jump)
        lowest = soles[:, :, 2].min(axis=1)
        self.assertGreater(lowest.max(), 0.2)                                   # feet well off the floor
        self.assertGreater(lowest.min(), -0.005)
        self.assertLess(lowest[-1], 0.005)                                      # and back on it
        airborne = np.flatnonzero(lowest > 0.01)
        flight = t[airborne[-1]] - t[airborne[0]]
        self.assertAlmostEqual(flight, 2.0 * np.sqrt(2.0 * 9.81 * 0.25) / 9.81, delta=0.05)   # 2 v / g
        z = self.jump.root_position[:, 2]
        middle = airborne[2 * len(airborne) // 5: 3 * len(airborne) // 5]   # away from the floor lift
        accel = (z[middle + 1] - 2 * z[middle] + z[middle - 1]) / (t[middle + 1] - t[middle]) ** 2
        self.assertTrue(np.allclose(accel, -9.81, atol=0.3))                    # free fall

    def test_sequence_file_round_trip_and_resampling(self):
        folder = TemporaryFolder()
        try:
            path = folder.path / "walk.npz"
            self.walk.save(path)
            loaded = self.motions.PoseSequence.load(path)
            self.assertTrue(np.allclose(loaded.body, self.walk.body))
            self.assertTrue(np.allclose(loaded.root_position, self.walk.root_position))
            again = loaded.sample(loaded.times[10:20])
            self.assertTrue(np.allclose(again.root_position, loaded.root_position[10:20]))
            self.assertTrue(np.allclose(axis_angle_to_matrix_np(again.body),
                                        axis_angle_to_matrix_np(loaded.body[10:20]), atol=1e-9))
        finally:
            folder.cleanup()


# ----------------------------------------------------------------------------
# Simulated recording, segmentation, image geometry
# ----------------------------------------------------------------------------

@needs_torch
class RecordingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bodyscan.dynamic import motions
        from bodyscan.dynamic.segment import MotionSegmenter, SegmentedRecording
        from bodyscan.dynamic.simulate import MotionSensor, Room, default_furniture, record
        from bodyscan.io import NpzRecording
        cls.folder = TemporaryFolder()
        model = load_test_body()
        shaped = model.shaped()
        motion = motions.standing(shaped, duration=0.8, xy=(2.0, 0.6), heading_deg=150.0)
        sensor = MotionSensor(columns=512, frame_rate=20.0, yaw_deg=10.0, position=(0.3, -0.2))
        cls.take = cls.folder.path / "take"
        record(cls.take, shaped, motion, sensor, Room(furniture=default_furniture()), background_frames=5, quiet=True)
        cls.segments = cls.folder.path / "segments"
        cls.summary = MotionSegmenter(NpzRecording(cls.take)).run(cls.segments)
        cls.segmented = SegmentedRecording(cls.segments)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_run_directory_and_truth(self):
        from bodyscan.dynamic.simulate import load_truth
        for name in ("lut.npz", "capture.json", "truth.npz", "labels.npz"):
            self.assertTrue((self.take / name).exists(), name)
        truth, meta = load_truth(self.take)
        self.assertEqual(meta["sensor_pose"]["yaw_deg"], 10.0)
        self.assertGreater(truth.times[0], 999.0)                               # sensor clock

    def test_segmentation_matches_the_labels(self):
        from bodyscan.dynamic.simulate import load_labels
        labels = load_labels(self.take)
        self.assertEqual(len(self.segmented), len(labels))
        precision, recall = [], []
        for k in range(len(self.segmented)):
            person = self.segmented.load(k)
            rows, columns = person.pixels[:, 0], person.pixels[:, 1]
            truth = labels[person.index]
            found = np.zeros_like(truth)
            found[rows, columns] = True
            precision.append((found & truth).sum() / max(found.sum(), 1))
            recall.append((found & truth).sum() / max(truth.sum(), 1))
        self.assertGreater(np.mean(precision), 0.97)
        self.assertGreater(np.mean(recall), 0.85)

    def test_points_project_back_to_their_pixels(self):
        person = self.segmented.load(3)
        rows, columns = self.segmented.geometry.pixels_of(person.points)
        same = (rows == person.pixels[:, 0]) & (columns == person.pixels[:, 1])
        self.assertGreater(same.mean(), 0.97)
        # the floor frame puts the sensor at the measured height above the floor, below it the origin
        self.assertAlmostEqual(self.segmented.sensor_height, 1.0, delta=0.01)
        self.assertTrue(np.allclose(self.segmented.sensor_position[:2], 0.0, atol=1e-6))

    def test_pixel_times_follow_the_columns(self):
        frame_stamps = 1000.0 + np.arange(512) / (512 * 20.0)
        times = self.segmented.geometry.pixel_times(frame_stamps)
        self.assertEqual(times.shape, (128, 512))
        self.assertTrue(np.all(np.diff(times[0]) > 0))


# ----------------------------------------------------------------------------
# Export of the animated mesh
# ----------------------------------------------------------------------------

@needs_torch
class ExportTest(unittest.TestCase):
    def test_meshes_velocities_and_tracks(self):
        import open3d as o3d
        from bodyscan.dynamic import motions
        from bodyscan.dynamic.avatar import Avatar
        from bodyscan.dynamic.export import ExportConfig, export_motion
        from bodyscan.body.testbody import cached
        folder = TemporaryFolder()
        try:
            model = load_test_body()
            avatar = Avatar.from_model(model)
            avatar.model_path = str(cached())
            avatar.save(folder.path / "avatar.npz")
            # a body gliding along +x at 1 m/s while the left arm swings forward
            times = np.arange(0.0, 1.0, 1.0 / 30)
            body = np.tile(motions.angles_to_body(motions.a_pose(1)), (len(times), 1, 1))
            body[:, J["left_shoulder"] - 1, 1] -= np.radians(40) * np.sin(2 * np.pi * times)
            position = np.stack([times, np.zeros_like(times), np.full_like(times, 0.95)], axis=1)
            motion = motions.PoseSequence(times, np.zeros((len(times), 3)), position, body)
            motion.save(folder.path / "motion.npz")
            config = ExportConfig(rate=100.0, start=0.3, stop=0.6)
            result = export_motion(folder.path / "motion.npz", folder.path / "avatar.npz", folder.path / "out",
                                   config, device="cpu")
            steps = result["steps"]
            self.assertEqual(steps, 31)
            out = folder.path / "out"
            mesh = o3d.io.read_triangle_mesh(str(out / "frames" / "mesh_000010.ply"))
            self.assertEqual(len(mesh.vertices), result["vertices"])
            self.assertEqual(len(mesh.triangles), result["triangles"])
            # the per-vertex velocity agrees with the difference of the meshes around it
            before = np.asarray(o3d.io.read_triangle_mesh(str(out / "frames" / "mesh_000009.ply")).vertices)
            after = np.asarray(o3d.io.read_triangle_mesh(str(out / "frames" / "mesh_000011.ply")).vertices)
            velocity = np.load(out / "frames" / "velocity_000010.npy")
            finite = (after - before) / 0.02
            self.assertLess(np.percentile(np.linalg.norm(velocity - finite, axis=1), 95), 0.05)
            sequence = np.load(out / "sequence.npz")
            pelvis = list(sequence["part_names"]).index("pelvis")
            self.assertTrue(np.allclose(sequence["part_velocity"][:, pelvis], (1.0, 0.0, 0.0), atol=0.02))
            arm = list(sequence["part_names"]).index("left_hand")
            self.assertGreater(np.abs(sequence["part_velocity"][:, arm, 0] - 1.0).max(), 0.3)   # the arm swings
            table = np.loadtxt(out / "joints.csv", delimiter=",", skiprows=1)
            self.assertEqual(table.shape, (steps, 1 + 3 * skeleton.NUM_JOINTS))
            self.assertTrue(np.allclose(table[:, 1], table[:, 0] - table[0, 0] + 0.3, atol=1e-3))   # pelvis x = t
            info = json.loads((out / "export.json").read_text())
            self.assertEqual(info["steps"], steps)
            # the whole motion: the velocity is right up to its first and last instants
            whole = export_motion(folder.path / "motion.npz", folder.path / "avatar.npz", folder.path / "whole",
                                  ExportConfig(rate=30.0, ply=False, velocities=False), device="cpu")
            ends = np.load(folder.path / "whole" / "sequence.npz")["part_velocity"][[0, whole["steps"] - 1], pelvis]
            self.assertTrue(np.allclose(ends, (1.0, 0.0, 0.0), atol=0.05), ends)
        finally:
            folder.cleanup()


# ----------------------------------------------------------------------------
# Avatar and tracking (slow)
# ----------------------------------------------------------------------------

@needs_torch
@slow
class AvatarFitTest(unittest.TestCase):
    def test_fit_to_a_turntable_scan(self):
        from bodyscan.dynamic.avatar import AvatarConfig, AvatarFitter
        from bodyscan.dynamic.bench import turntable_scan
        model = load_test_body()
        truth = model.shaped(np.array([0.8, -0.5, 0.3, 0, 0, 0, 0, 0, 0, 0]))
        points, normals = turntable_scan(truth, yaw_deg=-70.0)
        avatar = AvatarFitter(model, AvatarConfig(level=1)).fit(points, normals)
        report = avatar.report
        self.assertLess(report["shape_fit_mm"]["median"], 3.0)
        self.assertLess(report["detail_fit_mm"]["median"], 2.5)
        rotation = axis_angle_to_matrix_np(avatar.scan_root_rotation)
        yaw = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
        self.assertLess(abs((yaw + 70.0 + 180.0) % 360.0 - 180.0), 5.0)          # facing found
        stature = points[:, 2].max()
        self.assertAlmostEqual(report["stature_m"], stature, delta=0.03)


@needs_torch
@slow
class TrackingTest(unittest.TestCase):
    def test_tracking_a_short_walk(self):
        from bodyscan.dynamic import evaluate
        from bodyscan.dynamic.avatar import Avatar
        from bodyscan.dynamic.bench import SimulateMotionConfig, make_take
        from bodyscan.dynamic.segment import MotionSegmenter
        from bodyscan.dynamic.track import TrackPipelineConfig, run_tracking
        from bodyscan.body.testbody import cached
        from bodyscan.io import NpzRecording
        folder = TemporaryFolder()
        try:
            config = SimulateMotionConfig()
            config.scenario.duration, config.scenario.distance, config.scenario.scan = 1.5, 2.2, False
            take = make_take(folder.path / "take", cached(), config, quiet=True)
            MotionSegmenter(NpzRecording(take)).run(folder.path / "segments")
            tracking = TrackPipelineConfig()
            summary = run_tracking(folder.path / "segments", take / "truth_avatar.npz", folder.path / "motion.npz",
                                   tracking, start=30)
            self.assertLess(summary["residual_median_mm"], 8.0)
            self.assertLess(summary["flagged_frames"]["free_space"], 0.1 * summary["frames"])
            model = load_test_body()
            body = Avatar.load(take / "truth_avatar.npz").shaped(model, 0)
            result = evaluate.evaluate(folder.path / "motion.npz", take, model, body, body)
            self.assertLess(result["mpjpe_mm"]["mean"], 40.0)
            self.assertLess(result["seen_joint_mm"], 20.0)
            self.assertLess(result["yaw_deg"], 8.0)        # the pelvis may turn a little against the trunk
            self.assertLess(result["velocity_error_rms_m_s"], 0.3)
        finally:
            folder.cleanup()

    def test_tracking_a_jump_keeps_the_arms(self):
        # the arms swing overhead in 0.3 s: without the arm search the tracker lost them for good
        from bodyscan.dynamic import evaluate
        from bodyscan.dynamic.avatar import Avatar
        from bodyscan.dynamic.bench import SimulateMotionConfig, make_take
        from bodyscan.dynamic.segment import MotionSegmenter
        from bodyscan.dynamic.track import TrackPipelineConfig, run_tracking
        from bodyscan.body.testbody import cached
        from bodyscan.io import NpzRecording
        folder = TemporaryFolder()
        try:
            config = SimulateMotionConfig()
            config.scenario.motion, config.scenario.jumps, config.scenario.distance = "jump", 1, 2.0
            config.scenario.scan = False
            take = make_take(folder.path / "take", cached(), config, quiet=True)
            MotionSegmenter(NpzRecording(take)).run(folder.path / "segments")
            run_tracking(folder.path / "segments", take / "truth_avatar.npz", folder.path / "motion.npz",
                         TrackPipelineConfig(), start=30)
            model = load_test_body()
            body = Avatar.load(take / "truth_avatar.npz").shaped(model, 0)
            result = evaluate.evaluate(folder.path / "motion.npz", take, model, body, body)
            self.assertLess(result["mpjpe_mm"]["mean"], 30.0)
            self.assertLess(result["mpjpe_mm"]["max"], 120.0)
            for hand in ("left_hand", "right_hand"):
                self.assertLess(result["velocity_error"][hand]["rms_m_s"], 1.0)
        finally:
            folder.cleanup()


if __name__ == "__main__":
    unittest.main()
