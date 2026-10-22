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
            with np.load(out / "sequence.npz") as sequence:     # closed, or Windows cannot delete the folder
                part_names = list(sequence["part_names"])
                part_velocity = sequence["part_velocity"]
            pelvis = part_names.index("pelvis")
            self.assertTrue(np.allclose(part_velocity[:, pelvis], (1.0, 0.0, 0.0), atol=0.02))
            arm = part_names.index("left_hand")
            self.assertGreater(np.abs(part_velocity[:, arm, 0] - 1.0).max(), 0.3)   # the arm swings
            table = np.loadtxt(out / "joints.csv", delimiter=",", skiprows=1)
            self.assertEqual(table.shape, (steps, 1 + 3 * skeleton.NUM_JOINTS))
            self.assertTrue(np.allclose(table[:, 1], table[:, 0] - table[0, 0] + 0.3, atol=1e-3))   # pelvis x = t
            info = json.loads((out / "export.json").read_text())
            self.assertEqual(info["steps"], steps)
            # the whole motion: the velocity is right up to its first and last instants
            whole = export_motion(folder.path / "motion.npz", folder.path / "avatar.npz", folder.path / "whole",
                                  ExportConfig(rate=30.0, ply=False, velocities=False), device="cpu")
            with np.load(folder.path / "whole" / "sequence.npz") as sequence:
                ends = sequence["part_velocity"][[0, whole["steps"] - 1], pelvis]
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


@needs_torch
@slow
class HiddenLimbTest(unittest.TestCase):
    def test_the_far_arm_of_side_on_jumps_moves_like_its_partner(self):
        # seen side-on, the far arm is behind the trunk or the near arm; tracked alone it keeps still
        # or drifts, completed from the near arm (jumps: together, mirrored) it swings with it
        import torch
        from bodyscan.body.testbody import cached
        from bodyscan.dynamic import coupling, motions
        from bodyscan.dynamic.avatar import Avatar
        from bodyscan.dynamic.bench import SimulateMotionConfig, make_take
        from bodyscan.dynamic.segment import MotionSegmenter, SegmentedRecording
        from bodyscan.dynamic.simulate import load_truth, to_sensor_floor_frame
        from bodyscan.dynamic.track import Tracker, TrackPipelineConfig
        from bodyscan.io import NpzRecording
        folder = TemporaryFolder()
        try:
            model = load_test_body()
            shaped = model.shaped()
            # heading 90 deg: the left side towards the sensor (180 deg would face it, both arms in view)
            motion = motions.jump(shaped, count=2, height=0.3, heading_deg=90.0, stand_before=1.6, pause=0.6,
                                  stand_after=0.6)
            motion.save(folder.path / "motion.npz")
            config = SimulateMotionConfig()
            config.scenario.motion, config.scenario.distance = str(folder.path / "motion.npz"), 2.3
            config.scenario.scan = False
            take = make_take(folder.path / "take", cached(), config, quiet=True)
            MotionSegmenter(NpzRecording(take)).run(folder.path / "segments")
            avatar = Avatar.load(take / "truth_avatar.npz")
            tracker = Tracker(avatar, avatar.model(cached(), "cpu", torch.float64),
                              SegmentedRecording(folder.path / "segments"), TrackPipelineConfig())
            result = tracker.track(30, None)
            prepared = tracker.plan_coupling(result)
            self.assertGreater(np.mean(prepared["plan"].mode == coupling.TOGETHER), 0.7)
            alone = dict(result)                      # the far arm as the frame-by-frame fit leaves it
            alone["body"] = result["body"].copy()
            tracker.start_hidden_limbs(result, prepared)
            refined = tracker.refine(result, tracker.contacts(result))
            completed_result = dict(refined)
            completed = tracker.complete_hidden_limbs(completed_result, prepared)
            finished_result = dict(completed_result)
            finished_result["body"] = completed_result["body"].copy()
            finished = tracker.finish_pose(finished_result)
            body = avatar.shaped(model, 0)
            truth, meta = load_truth(take)
            seen = coupling.limb_observation(result["observed"])
            far = "left" if seen["left_arm"].mean() < seen["right_arm"].mean() else "right"
            self.assertIn(f"{far}_arm", completed)
            self.assertEqual(set(finished), {"hidden", "speeds", "hinges", "arms"})
            errors = {}
            for name, source in (("alone", alone), ("completed", completed_result), ("finished", finished_result)):
                estimate = motions.PoseSequence(result["times"], source["root_rotation"], source["root_position"],
                                                source["body"])
                _, joints = estimate.posed(body)
                _, true_joints = truth.sample(result["times"]).posed(body)
                true_joints = to_sensor_floor_frame(true_joints.reshape(-1, 3), meta["sensor_pose"]).reshape(
                    true_joints.shape)
                wrist = J[f"{far}_wrist"]
                errors[name] = float(np.linalg.norm(joints[:, wrist] - true_joints[:, wrist], axis=1).mean())
            self.assertLess(errors["completed"], 0.75 * errors["alone"], errors)
            self.assertLess(errors["completed"], 0.12, errors)
            # the last step (out of the sensor's sight, speeds, hinges, out of the trunk) keeps the guess
            self.assertLess(errors["finished"], 1.1 * errors["completed"], errors)
        finally:
            folder.cleanup()


class CouplingTest(unittest.TestCase):
    """Left-right coupling of hidden limbs (bodyscan.dynamic.coupling), numpy only."""

    STEP = 0.05

    def plan(self, legs, arms, z, observation):
        from bodyscan.dynamic import coupling
        times = np.arange(len(z)) * self.STEP
        signals = {"legs": np.stack(legs, axis=1), "arms": np.stack(arms, axis=1), "pelvis_z": z}
        return coupling.estimate(times, signals, observation, step=self.STEP)

    @staticmethod
    def observation(n, left_arm=1.0, right_arm=1.0, left_leg=1.0, right_leg=1.0):
        return {"left_arm": np.full(n, left_arm), "right_arm": np.full(n, right_arm),
                "left_leg": np.full(n, left_leg), "right_leg": np.full(n, right_leg)}

    def test_mirror_maps_a_left_limb_pose_onto_the_same_right_limb_pose(self):
        from bodyscan.dynamic import coupling, motions
        a = motions.Angles(1)
        for side in ("left", "right"):
            for name, value in (("elevation", 25.0), ("swing", 40.0), ("twist", 30.0), ("elbow", 70.0),
                                ("hip", 35.0), ("knee", 60.0), ("ankle", 10.0), ("abduction", 8.0)):
                getattr(a, f"{side}_{name}")[:] = value
        body = motions.angles_to_body(a)[0]
        for limb in ("left_arm", "left_leg"):
            left = body[coupling.body_indices(limb)]
            right = body[coupling.body_indices(coupling.PARTNER[limb])]
            self.assertTrue(np.allclose(coupling.mirror(left), right, atol=1e-9), limb)

    def test_plan_tells_an_alternating_gait_from_jumps_together(self):
        from bodyscan.dynamic import coupling
        rng = np.random.default_rng(1)
        t = np.arange(0.0, 8.0, self.STEP)
        cycle = 0.9
        phase = 2 * np.pi * t / cycle
        # running seen side-on: the right arm and most of the right leg hidden; the pelvis bobs once per step
        legs = [0.4 * np.sin(phase), -0.4 * np.sin(phase)]
        arms = [-0.5 * np.sin(phase), 0.5 * np.sin(phase)]
        z = 0.95 + 0.03 * np.cos(2 * phase) + 0.001 * rng.normal(size=len(t))
        plan = self.plan(legs, arms, z, self.observation(len(t), right_arm=0.0, right_leg=0.3))
        middle = slice(len(t) // 4, 3 * len(t) // 4)
        self.assertGreater(np.mean(plan.mode[middle] == coupling.ALTERNATING), 0.95)
        self.assertLess(abs(np.median(plan.cycle[middle]) - cycle), 0.05)
        self.assertTrue(np.allclose(plan.shift[middle], 0.5 * plan.cycle[middle]))
        # jumps one after the other: hips flex as the pelvis drops, both arms swing together
        cycle = 1.3
        phase = 2 * np.pi * t / cycle
        z = 0.95 + 0.15 * np.sin(phase)
        hip = 0.6 * (1 - np.sin(phase)) / 2
        arm = 1.2 * np.sin(phase + 0.3)
        plan = self.plan([hip, hip], [arm, arm], z, self.observation(len(t), right_arm=0.0, right_leg=0.2))
        self.assertGreater(np.mean(plan.mode[middle] == coupling.TOGETHER), 0.95)
        self.assertTrue(np.all(plan.shift[middle] == 0.0))
        # standing: nothing to tell
        still = [0.01 * rng.normal(size=len(t)) for _ in range(2)]
        plan = self.plan(still, still, 0.95 + 0.001 * rng.normal(size=len(t)), self.observation(len(t), right_arm=0.0))
        self.assertTrue(np.all(plan.mode == coupling.UNKNOWN))
        self.assertTrue(np.all(plan.confidence == 0.0))

    def test_plan_holds_to_the_end_of_a_take_and_fades_at_a_change_of_mode(self):
        from bodyscan.dynamic import coupling
        rng = np.random.default_rng(2)
        # a take of 5.4 s that stands, walks from 0.8 to 3.8 s (a cycle of 1.1 s) and stands again (as the
        # synthetic walk of the bench): a window cut short by the end of the take missed the cycle, and
        # the plan switched from alternating to unknown, with the confidence from 1 to 0, between two frames
        t = np.arange(0.0, 5.4, self.STEP)
        envelope = np.clip(np.minimum((t - 0.8) / 0.6, (3.8 - t) / 0.6), 0.0, 1.0)
        phase = 2 * np.pi * np.cumsum(envelope) * self.STEP / 1.1
        legs = [0.35 * envelope * np.sin(phase) + 0.02 * rng.normal(size=len(t)),
                -0.35 * envelope * np.sin(phase) + 0.02 * rng.normal(size=len(t))]
        arms = [-0.3 * envelope * np.sin(phase) + 0.02 * rng.normal(size=len(t)),
                0.3 * envelope * np.sin(phase) + 0.02 * rng.normal(size=len(t))]
        z = 0.95 + 0.02 * envelope * np.cos(2 * phase) + 0.002 * rng.normal(size=len(t))
        plan = self.plan(legs, arms, z, self.observation(len(t), right_arm=0.0, right_leg=0.3))
        self.assertGreater(np.mean(plan.mode == coupling.ALTERNATING), 0.9)
        self.assertLess(np.abs(np.diff(plan.confidence)).max(), 1.5 * self.STEP / coupling.FADE)
        alternating = plan.mode == coupling.ALTERNATING
        self.assertLess(np.abs(np.diff(plan.cycle[alternating])).max(), 0.02)     # the cycle changes slowly
        # running, then jumping: the confidence fades out before the change and in after it
        t = np.arange(0.0, 10.0, self.STEP)
        run = t < 5.0
        phase = 2 * np.pi * t / np.where(run, 0.9, 1.3)
        legs = [np.where(run, 0.4 * np.sin(phase), 0.3 * (1 - np.sin(phase))), np.where(run, -0.4 * np.sin(phase),
                                                                                       0.3 * (1 - np.sin(phase)))]
        arms = [np.where(run, -0.5 * np.sin(phase), 1.2 * np.sin(phase + 0.3)),
                np.where(run, 0.5 * np.sin(phase), 1.2 * np.sin(phase + 0.3))]
        z = 0.95 + np.where(run, 0.03 * np.cos(2 * phase), 0.15 * np.sin(phase)) + 0.001 * rng.normal(size=len(t))
        plan = self.plan(legs, arms, z, self.observation(len(t), right_arm=0.0, right_leg=0.3))
        self.assertIn(coupling.ALTERNATING, plan.mode[:80])
        self.assertIn(coupling.TOGETHER, plan.mode[120:])
        # never more than a frame period over FADE (plus the smoothing) between two frames
        self.assertLess(np.abs(np.diff(plan.confidence)).max(), 1.5 * self.STEP / coupling.FADE)
        changes = np.flatnonzero(plan.mode[1:] != plan.mode[:-1])
        for k in changes:
            self.assertLess(max(plan.confidence[k], plan.confidence[k + 1]), 0.05)

    def test_target_is_the_mirrored_partner_half_a_cycle_away(self):
        from bodyscan.dynamic import coupling
        t = np.arange(0.0, 4.0, self.STEP)
        cycle = 1.0
        body = np.zeros((len(t), 21, 3))
        left = coupling.body_indices("left_arm")
        body[:, left[1], 1] = 0.5 * np.sin(2 * np.pi * t / cycle)               # left shoulder swings
        body[:, left[1], 2] = -1.2                                              # left arm down
        plan = coupling.CouplingPlan(np.full(len(t), coupling.ALTERNATING), np.ones(len(t)), np.full(len(t), cycle),
                                     np.full(len(t), 0.5 * cycle))
        guess = coupling.joint_guesses(t, body, plan, self.observation(len(t), right_arm=0.0), ("right_arm",))
        target, weight = guess["right_arm"].rotations, guess["right_arm"].weight
        inner = slice(20, 60)
        expected = coupling.mirror(body[:, left[1]])
        expected[:, 1] *= -1.0                                  # half a cycle away: the swing reversed
        self.assertTrue(np.allclose(target[inner, 1], expected[inner], atol=1e-3))
        self.assertTrue(np.allclose(weight[inner], 1.0))

    def test_guess_does_not_jump_where_half_a_cycle_before_enters_the_take(self):
        from bodyscan.dynamic import coupling
        # the partner raises its arm, holds it and then swings: half a cycle after the start of the take
        # the guess gets its partner half a cycle before as well as after, and used to switch from the one
        # to the mean of the two between two frames (here 0.3 rad of the shoulder)
        t = np.arange(0.0, 4.0, self.STEP)
        body = np.zeros((len(t), 21, 3))
        left = coupling.body_indices("left_arm")
        body[:, left[1], 1] = np.where(t < 1.5, 0.6 * np.clip((t - 0.2) / 0.25, 0.0, 1.0),
                                       0.6 * np.cos(2 * np.pi * (t - 1.5)))
        body[:, left[1], 2] = -1.2
        plan = coupling.CouplingPlan(np.full(len(t), coupling.ALTERNATING), np.ones(len(t)), np.ones(len(t)),
                                     np.full(len(t), 0.5))
        guess = coupling.joint_guesses(t, body, plan, self.observation(len(t), right_arm=0.0), ("right_arm",))
        shoulder = guess["right_arm"].rotations[:, 1]
        steps = np.linalg.norm(np.diff(shoulder, axis=0), axis=1)
        self.assertLess(steps[:20].max(), 0.1)

    def test_guess_hangs_from_the_chest_as_its_partner_does(self):
        # the chest turned against the pelvis: the guessed arm is the mirror image of its partner relative
        # to the chest (version 1.5 mirrored the world orientations at the plane of the pelvis' facing, and
        # the far arm of the s02 side-on jumps swung out to the side while the near arm swung forward)
        from bodyscan.dynamic import coupling, motions
        from bodyscan.body.rotations import axis_angle_to_matrix_np
        a = motions.Angles(2)
        a.left_swing[:], a.left_elevation[:], a.left_elbow[:], a.left_twist[:] = 40.0, 20.0, 60.0, 30.0
        a.trunk_twist[:] = 15.0
        body = motions.angles_to_body(a)
        plan = coupling.CouplingPlan(np.full(2, coupling.TOGETHER), np.ones(2), np.full(2, np.nan), np.zeros(2))
        guess = coupling.joint_guesses(np.arange(2.0), body, plan, self.observation(2, right_arm=0.0), ("right_arm",))
        mirrored = guess["right_arm"].rotations
        # the arm direction relative to the chest, mirrored at its midplane (x -> -x in the model)
        for k, name in enumerate(("shoulder",)):
            left_dir = axis_angle_to_matrix_np(body[0, J["left_shoulder"] - 1]) @ np.array([1.0, 0.0, 0.0])
            right_dir = axis_angle_to_matrix_np(mirrored[0, 1]) @ np.array([-1.0, 0.0, 0.0])
            self.assertTrue(np.allclose(right_dir, left_dir * np.array([-1.0, 1.0, 1.0]), atol=1e-9))

    def test_offsets_seen_carry_on_and_fade(self):
        from bodyscan.dynamic import coupling
        from bodyscan.body.rotations import axis_angle_to_quaternion_np, quaternion_to_axis_angle_np
        t = np.arange(0.0, 3.0, self.STEP)
        angle = np.zeros(len(t))
        anchored = np.zeros(len(t))
        angle[:10], anchored[:10] = 1.0, 1.0                      # observed until 0.45 s
        angle[30:32], anchored[30:32] = -1.0, 1.0                 # briefly at 1.5 s
        offsets = axis_angle_to_quaternion_np(np.stack([angle, 0 * angle, 0 * angle], axis=1)[:, None])
        carried = quaternion_to_axis_angle_np(coupling.carried_offsets(t, offsets, anchored, memory=0.4))[:, 0, 0]
        self.assertAlmostEqual(carried[5], 1.0)
        # between two observations 1.05 s apart: turned from one to the other
        self.assertAlmostEqual(carried[20], 1.0 + (-2.0) * (20 - 9) / 21.0, delta=1e-6)
        self.assertTrue(np.all(np.diff(carried[9:31]) < 0))
        # after the last one: fades towards no offset
        self.assertAlmostEqual(carried[40], -np.exp(-(t[40] - t[31]) / 0.4), delta=1e-6)
        self.assertLess(abs(carried[59]), 0.05)


class HingeTest(unittest.TestCase):
    """Elbows and knees bend in one plane (bodyscan.body.anatomy)."""

    def test_an_elbow_bent_sideways_is_put_back_into_its_plane_with_its_twist(self):
        from bodyscan.body import anatomy
        from bodyscan.dynamic import motions
        a = motions.Angles(3)
        for side in ("left", "right"):
            getattr(a, f"{side}_elevation")[:] = 30.0
            getattr(a, f"{side}_elbow")[:] = [10.0, 60.0, 120.0]
            getattr(a, f"{side}_knee")[:] = [5.0, 40.0, 90.0]
        body = motions.angles_to_body(a)
        self.assertTrue(np.all(anatomy.hinge_excess(body) < 1e-6))
        self.assertTrue(np.allclose(anatomy.project_hinges(body), body))      # nothing to do
        bent = body.copy()
        bent[:, J["left_elbow"] - 1, 2] = np.radians(-20.0)                    # varus 20 deg (limit 5)
        bent[:, J["right_knee"] - 1, 2] = np.radians(15.0)                     # knee sideways 15 deg (limit 6)
        flexion, off = anatomy.hinge_angles(bent)
        self.assertTrue(np.all(np.degrees(off[:, 0]) < -7.0))
        self.assertTrue(np.all(np.abs(np.degrees(off[:, 3])) > 8.0))
        self.assertTrue(np.all(anatomy.hinge_excess(bent) > 2.0))
        fixed = anatomy.project_hinges(bent)
        self.assertTrue(np.all(anatomy.hinge_excess(fixed) < 1e-6))
        flexion_fixed, off_fixed = anatomy.hinge_angles(fixed)
        self.assertTrue(np.allclose(np.degrees(off_fixed[:, 0]), -5.0, atol=1e-6))
        self.assertTrue(np.allclose(flexion_fixed[:, 0], flexion[:, 0], atol=np.radians(3.0)))
        unchanged = [i for i in range(21) if i not in (J["left_elbow"] - 1, J["right_knee"] - 1)]
        self.assertTrue(np.allclose(fixed[:, unchanged], bent[:, unchanged]))

    def test_the_penalty_is_zero_in_range_and_grows_out_of_it(self):
        from bodyscan.body import anatomy
        try:
            import torch
        except ImportError:
            self.skipTest("torch")
        body = np.zeros((1, 21, 3))
        body[0, J["left_elbow"] - 1, 1] = np.radians(-90.0)                    # bent 90 deg in its plane
        self.assertEqual(float(anatomy.hinge_penalty(torch.as_tensor(body))), 0.0)
        body[0, J["left_elbow"] - 1, 2] = np.radians(-15.0)
        small = float(anatomy.hinge_penalty(torch.as_tensor(body)))
        body[0, J["left_elbow"] - 1, 2] = np.radians(-30.0)
        large = float(anatomy.hinge_penalty(torch.as_tensor(body)))
        self.assertGreater(small, 0.0)
        self.assertGreater(large, 3.0 * small)


class SpeedLimitTest(unittest.TestCase):
    """No joint turns faster than a person's joint (bodyscan.body.anatomy.limit_speeds)."""

    @staticmethod
    def speeds(times, body, joint):
        R = axis_angle_to_matrix_np(body[:, joint])
        relative = np.einsum("fji,fjk->fik", R[:-1], R[1:])
        cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        return np.degrees(np.arccos(cosine)) / np.diff(times)

    def test_a_joint_that_jumps_for_one_frame_takes_the_mean_of_its_neighbours(self):
        from bodyscan.body import anatomy
        times = np.arange(20) * 0.05
        body = np.zeros((20, 21, 3))
        elbow = J["left_elbow"] - 1
        body[:, elbow, 1] = -np.radians(np.linspace(10.0, 40.0, 20))           # 30 deg in 0.95 s
        body[10, elbow, 1] = np.radians(60.0)                                   # one frame 85 deg off, and back
        limited, changed = anatomy.limit_speeds(times, body)
        self.assertEqual(np.argwhere(changed).tolist(), [[10, elbow]])
        self.assertAlmostEqual(limited[10, elbow, 1], 0.5 * (body[9, elbow, 1] + body[11, elbow, 1]), places=9)
        self.assertTrue(np.allclose(np.delete(limited, 10, axis=0), np.delete(body, 10, axis=0)))
        self.assertLessEqual(self.speeds(times, limited, elbow).max(), anatomy.SPEED_LIMITS["left_elbow"])

    def test_a_turn_made_in_one_frame_is_spread_over_as_many_as_the_joint_needs(self):
        from bodyscan.body import anatomy
        times = np.arange(30) * 0.05
        body = np.zeros((30, 21, 3))
        spine = J["spine1"] - 1
        body[15:, spine, 1] = np.radians(60.0)                                  # 60 deg in 50 ms: 1200 deg/s
        limited, changed = anatomy.limit_speeds(times, body)
        speed = self.speeds(times, limited, spine)
        self.assertLessEqual(speed.max(), anatomy.SPEED_LIMITS["spine1"] * (1 + 1e-6))
        self.assertGreaterEqual(int(changed[:, spine].sum()), 2)               # 60 deg at 400 deg/s: 0.15 s
        self.assertTrue(np.allclose(limited[:12], body[:12]) and np.allclose(limited[18:], body[18:]))
        rows = np.flatnonzero(changed[:, spine])
        self.assertLessEqual(abs((rows.mean() - 14.5)), 0.5)                    # centred on the turn
        self.assertFalse(np.any(changed[:, [i for i in range(21) if i != spine]]))

    def test_motion_within_the_limits_is_left_alone(self):
        from bodyscan.body import anatomy
        from bodyscan.dynamic import motions
        times = np.arange(40) * 0.05
        a = motions.Angles(40)
        swing = 40.0 * np.sin(2 * np.pi * times / 0.8)                         # arms swinging at 1.25 Hz
        for side, sign in (("left", 1.0), ("right", -1.0)):
            getattr(a, f"{side}_swing")[:] = sign * swing
            getattr(a, f"{side}_elbow")[:] = 30.0 + 0.5 * swing
        body = motions.angles_to_body(a)
        limited, changed = anatomy.limit_speeds(times, body)
        self.assertFalse(changed.any())
        self.assertTrue(np.array_equal(limited, body))


class ReviewRenderTest(unittest.TestCase):
    def test_guessed_limbs_are_drawn_violet(self):
        from bodyscan.dynamic.review import render
        # a square facing the camera (camera looks along +x)
        vertices = np.array([[0.0, -0.3, 0.7], [0.0, 0.3, 0.7], [0.0, 0.3, 1.3], [0.0, -0.3, 1.3]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        center = np.array([0.0, 0.0, 1.0])
        plain = render(vertices, faces, None, None, center, np.array([1.0, 0.0]), pixels=60).astype(int)
        tinted = render(vertices, faces, None, None, center, np.array([1.0, 0.0]), pixels=60,
                        tinted=np.ones(4, dtype=bool)).astype(int)
        middle = plain[30, plain.shape[1] // 2], tinted[30, tinted.shape[1] // 2]
        self.assertGreater(middle[0][1], middle[0][2])           # body: green above blue
        self.assertGreater(middle[1][2], middle[1][1])           # guessed: blue above green


class TurnLimitTest(unittest.TestCase):
    def test_a_body_turned_around_between_frames_is_refused_and_a_real_turn_is_not(self):
        from types import SimpleNamespace
        from bodyscan.body.rotations import rotation_z_np
        from bodyscan.dynamic.track import Tracker, TrackPipelineConfig
        tracker = SimpleNamespace(config=TrackPipelineConfig(), yaw=Tracker.yaw)
        times = list(np.arange(0.0, 1.0, 0.05))
        rotations = [rotation_z_np(np.radians(90.0 * t)) for t in times]       # turning at 90 deg/s
        check = Tracker.plausible_turn
        self.assertTrue(check(tracker, rotation_z_np(np.radians(94.5)), 1.05, rotations, times))
        self.assertTrue(check(tracker, rotation_z_np(np.radians(130.0)), 1.05, rotations, times))   # +35 deg search
        self.assertFalse(check(tracker, rotation_z_np(np.radians(90.0 + 180.0)), 1.05, rotations, times))
        # five 35 deg steps in a row add up to a turn around: refused against the frames before
        self.assertFalse(check(tracker, rotation_z_np(np.radians(90.0 + 150.0)), 1.25, rotations, times))
        tracker.config.tracking.max_turn_rate = 0.0
        self.assertTrue(check(tracker, rotation_z_np(np.radians(90.0 + 180.0)), 1.05, rotations, times))

    def test_a_late_turn_is_spread_over_the_frames_before_and_a_false_one_refused(self):
        from types import SimpleNamespace
        from unittest import mock
        from bodyscan.body.rotations import rotation_z_np
        from bodyscan.dynamic import track
        times = list(0.05 * np.arange(20))

        def run(true_yaw):
            # a stand-in for the fit: started within 60 deg of the true facing it finds it, otherwise it
            # stays where it started; the score grows with the error of the facing
            def fit_frame(data, reference, position, body, prediction):
                start = Tracker.yaw(reference)
                error = np.angle(np.exp(1j * (true_yaw[data] - start)))
                yaw = true_yaw[data] if abs(error) < np.radians(60.0) else start
                return rotation_z_np(yaw), position, body, 0.0, None

            def explain(rotation, position, body, data):
                error = abs(np.angle(np.exp(1j * (true_yaw[data] - Tracker.yaw(rotation)))))
                return {"score": 0.005 + 0.02 * error / np.pi, "distance": np.zeros(1), "observed_parts": None}

            from bodyscan.dynamic.track import Tracker, TrackPipelineConfig
            stub = SimpleNamespace(config=TrackPipelineConfig(), segmented=SimpleNamespace(load=lambda slot: slot),
                                   geometry=None, device=None, dtype=None, predict=Tracker.predict, yaw=Tracker.yaw,
                                   fit_frame=fit_frame, explain=explain)
            # the track so far faced 0 deg in every frame; the facing check at 1.0 s finds 180 deg
            rotations = [rotation_z_np(0.0) for _ in times]
            positions = [np.zeros(3) for _ in times]
            bodies = [np.zeros((21, 3)) for _ in times]
            scores = [explain(r, None, None, i)["score"] for i, r in enumerate(rotations)]
            candidate = (rotation_z_np(np.pi), np.zeros(3), np.zeros((21, 3)))
            with mock.patch.object(track, "FrameData", lambda person, *args: person):
                return Tracker.connect_turn(stub, candidate, 1.0, rotations, positions, bodies, times,
                                            list(range(len(times))), scores, 0)

        from bodyscan.dynamic.track import Tracker
        # the person turned around from 0.5 s to 1.0 s and the track lagged behind
        turning = np.radians(180.0 * np.clip((np.array(times + [1.0]) - 0.5) / 0.5, 0.0, 1.0))
        refits = run(turning)
        self.assertIsNotNone(refits)
        indices = [i for i, _, _ in refits]
        self.assertEqual(indices[0], 19)
        self.assertLessEqual(indices[-1], 11)                      # back to where the turn began
        for i, fitted, _ in refits:
            self.assertAlmostEqual(abs(np.angle(np.exp(1j * (Tracker.yaw(fitted[0]) - turning[i])))), 0.0, places=6)
        # the person never turned: the frames before cannot be explained facing the other way
        self.assertIsNone(run(np.zeros(len(times) + 1)))

    def test_a_facing_that_would_run_backwards_is_refused(self):
        from bodyscan.body.rotations import rotation_z_np
        from bodyscan.dynamic.track import Tracker
        times = list(np.arange(0.0, 1.0, 0.05))
        positions = [np.array([1.5 * t, 0.0, 1.0]) for t in times]             # running along +x at 1.5 m/s
        backwards = Tracker.runs_backwards
        self.assertFalse(backwards(None, rotation_z_np(0.0), positions, times, 1.0))
        self.assertFalse(backwards(None, rotation_z_np(np.radians(100.0)), positions, times, 1.0))   # side-step
        self.assertTrue(backwards(None, rotation_z_np(np.pi), positions, times, 1.0))
        standing = [np.array([0.0, 0.0, 1.0]) for _ in times]
        self.assertFalse(backwards(None, rotation_z_np(np.pi), standing, times, 1.0))


@needs_torch
class TrunkCollisionTest(unittest.TestCase):
    def test_an_arm_through_the_trunk_is_found_and_one_at_the_side_is_not(self):
        import torch
        from bodyscan.body.model import vertex_normals
        from bodyscan.dynamic import motions
        from bodyscan.dynamic.track import TrunkCollision
        model = load_test_body()
        shaped = model.shaped()
        labels = skeleton.part_labels(shaped.weights.detach().cpu().numpy())
        collision = TrunkCollision(shaped, labels)
        a = motions.a_pose(1, elevation=12.0)
        sequence = motions.PoseSequence(np.zeros(1), np.zeros((1, 3)), np.zeros((1, 3)), motions.angles_to_body(a))
        hanging, _ = sequence.posed(shaped, dtype=np.float64)
        # the same arm moved 12 cm sideways into the trunk
        names = np.array(skeleton.PART_NAMES)[labels]
        arm = np.isin(names, ["left_upper_arm", "left_forearm", "left_hand"])
        inwards = hanging[0][names == "torso_lower"].mean(axis=0) - hanging[0][arm].mean(axis=0)
        inwards[2] = 0.0
        inside = hanging.copy()
        inside[0][arm] += 0.12 * inwards / np.linalg.norm(inwards)
        vertices = np.concatenate([hanging, inside])
        normals = vertex_normals(torch.as_tensor(vertices), shaped.faces).numpy()
        self.assertLess(collision.depth(vertices[0], normals[0]), 0.02)        # hanging at the side
        self.assertGreater(collision.depth(vertices[1], normals[1]), 0.04)     # in the trunk
        # the energy pushes the arm out: zero at the side, positive inside
        v = torch.tensor(vertices, requires_grad=True)
        n = torch.as_tensor(normals[:, collision.trunk])
        self.assertEqual(float(collision.energy(v[:1], n[:1], 0.02).detach()), 0.0)
        energy = collision.energy(v[1:], n[1:], 0.02)
        self.assertGreater(float(energy.detach()), 0.0)
        energy.backward()
        push = v.grad[1].numpy()[arm].sum(axis=0)              # the gradient points into the trunk
        self.assertGreater(float(push @ inwards), 0.0)


@needs_torch
class TrackingPenaltyTest(unittest.TestCase):
    def test_free_space_penalties_are_bounded_and_the_wide_one_reaches_farther(self):
        import torch
        from bodyscan.dynamic.track import free_space_loss
        scale = 0.06
        distance = torch.linspace(0.0, 2.0, 401, dtype=torch.float64)
        for kind in ("bounded", "wide"):
            # bounded by scale^2: leaving the points for empty space never pays more than a miss
            self.assertLessEqual(float(free_space_loss(distance, scale, kind).max()), scale ** 2 + 1e-12)
        pull = {}
        for kind in ("bounded", "wide"):
            at = torch.tensor([4 * scale, 6 * scale], dtype=torch.float64, requires_grad=True)
            free_space_loss(at, scale, kind).sum().backward()
            pull[kind] = at.grad.numpy()
        # a limb 24 or 36 cm out in empty space is still pulled back (2.4 and 3.8 times the bounded pull)
        self.assertGreater(pull["wide"][0], 2.0 * pull["bounded"][0])
        self.assertGreater(pull["wide"][1], 3.0 * pull["bounded"][1])


if __name__ == "__main__":
    unittest.main()


def test_find_model_falls_back_to_the_project_models_folder(tmp_path, monkeypatch):
    from bodyscan.body import model as body_model
    models = tmp_path / "models"
    models.mkdir()
    (models / "SMPLX_MALE.npz").write_bytes(b"x")
    monkeypatch.setattr(body_model, "MODEL_FOLDERS", (models, tmp_path))
    assert body_model.find_model("C:/lidar/SMPLX_MALE.npz") == models / "SMPLX_MALE.npz"   # data folder: not there
    assert body_model.find_model("SMPLX_MALE.npz") == models / "SMPLX_MALE.npz"
    explicit = tmp_path / "other.npz"
    explicit.write_bytes(b"x")
    assert body_model.find_model(explicit) == explicit
    import pytest
    with pytest.raises(SystemExit):
        body_model.find_model("SMPLX_FEMALE.npz")
