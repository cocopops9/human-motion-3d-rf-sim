"""The full SMPL-X pose in the tracker (pose model full55): the hand PCA file and its check against the
body model, the layout of the extra pose, the hold-out of points, the jaw and eyes of a PoseSequence,
and a short warm-started tracking of a synthetic take with both pose models.

    python -m unittest tests.test_fullpose
    BODYSCAN_SLOW_TESTS=1 python -m unittest tests.test_fullpose     (with the tracking test)
"""

from __future__ import annotations

import importlib.util
import json
import unittest
from types import SimpleNamespace

import numpy as np

from tests.helpers import TemporaryFolder, slow

HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = unittest.skipUnless(HAS_TORCH, "needs PyTorch (python -m pip install torch)")


def write_hand_pca_file(body_path, out, seed: int = 0, change_body: bool = False):
    """A copy of the model file 'body_path' with MANO-like hand components (random orthogonal rows,
    scaled down like the spread of real hand poses) and, with change_body, another v_template."""
    with np.load(body_path, allow_pickle=True) as file:
        data = {key: file[key] for key in file.files}
    rng = np.random.default_rng(seed)
    for side in ("l", "r"):
        q, _ = np.linalg.qr(rng.normal(size=(45, 45)))
        data[f"hands_components{side}"] = q * np.linspace(0.6, 0.05, 45)[:, None]
        data.setdefault(f"hands_mean{side}", rng.normal(0.0, 0.1, 45))
    if change_body:
        data["v_template"] = data["v_template"] + 1e-3
    np.savez(out, **data)
    return out


@needs_torch
class HandPCATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bodyscan.body.testbody import cached
        cls.folder = TemporaryFolder()
        cls.body = cached()
        cls.pca = write_hand_pca_file(cls.body, cls.folder.path / "hands.npz")

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_same_body_loads_and_maps_coefficients_to_rotations(self):
        from bodyscan.body.hands import HandPCA
        hands = HandPCA.load(self.pca, 6, body_model=self.body)
        self.assertEqual(hands.components.shape, (2, 6, 45))
        self.assertEqual(len(hands.sha1), 40)
        zero = hands.axis_angle_np(np.zeros((3, 2, 6)))
        np.testing.assert_allclose(zero[1, 0].reshape(45), hands.mean[0])
        coefficients = np.zeros((1, 2, 6))
        coefficients[0, 1, 2] = 2.0
        pose = hands.axis_angle_np(coefficients)[0]
        np.testing.assert_allclose(pose[1].reshape(45), hands.mean[1] + 2.0 * hands.components[1, 2])
        np.testing.assert_allclose(pose[0].reshape(45), hands.mean[0])

    def test_another_body_or_missing_components_stop(self):
        from bodyscan.body.hands import HandPCA
        other = write_hand_pca_file(self.body, self.folder.path / "other.npz", change_body=True)
        with self.assertRaises(SystemExit):
            HandPCA.load(other, 6, body_model=self.body)
        with np.load(self.body, allow_pickle=True) as file:
            stripped = {key: file[key] for key in file.files if not key.startswith("hands_")}
        np.savez(self.folder.path / "stripped.npz", **stripped)     # like the s02 model file: no hand arrays
        with self.assertRaises(SystemExit):
            HandPCA.load(self.folder.path / "stripped.npz", 6)
        with self.assertRaises(SystemExit):
            HandPCA.load(self.pca, 46)


@needs_torch
class PoseLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from bodyscan.body.hands import HandPCA
        from bodyscan.body.model import BodyModel
        from bodyscan.body.testbody import cached
        cls.folder = TemporaryFolder()
        cls.model = BodyModel(cached(), device="cpu", dtype=torch.float64)
        cls.hands = HandPCA.load(write_hand_pca_file(cached(), cls.folder.path / "hands.npz"), 6,
                                 body_model=cached())

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def stub(self, layout):
        """The parts of a Tracker that rotations() uses."""
        from bodyscan.dynamic.track import Tracker
        stub = SimpleNamespace(layout=layout, dtype=self.model.dtype, device=self.model.device,
                               hands=self.model.rotations(count=1)[0, 25:])
        return lambda body, extra=None: Tracker.rotations(stub, body, extra)

    def test_rest_extra_is_the_body21_pose_with_the_pca_mean_hands(self):
        import torch
        from bodyscan.dynamic.fullpose import PoseLayout
        full = PoseLayout(self.model, "full55", self.hands)
        self.assertEqual(full.size, 9 + 12)
        body = torch.as_tensor(np.random.default_rng(1).normal(0.0, 0.2, (4, 21, 3)))
        rotations = self.stub(full)(body, torch.zeros(4, full.size, dtype=torch.float64))
        # the body joints and the face as body21 poses them; the hands at the PCA mean
        body21 = self.stub(PoseLayout(self.model, "body21"))(body)
        torch.testing.assert_close(rotations[:, :25], body21[:, :25])
        expected = self.model.rotations(body=body, left_hand=np.tile(self.hands.mean[0].reshape(15, 3), (4, 1, 1)),
                                        right_hand=np.tile(self.hands.mean[1].reshape(15, 3), (4, 1, 1)))
        torch.testing.assert_close(rotations, expected)

    def test_extra_moves_jaw_eyes_and_hands_and_splits_back(self):
        import torch
        from bodyscan.dynamic.fullpose import PoseLayout
        from bodyscan.body.rotations import matrix_to_axis_angle_np
        layout = PoseLayout(self.model, "full55", self.hands)
        extra = np.zeros((2, layout.size))
        extra[1, 0:3] = [0.1, 0.0, 0.0]                   # jaw open by 0.1 rad
        extra[1, 6:9] = [0.0, 0.05, 0.0]                  # right eye
        extra[1, 9 + 6 + 1] = 1.5                         # second coefficient of the right hand
        rotations = self.stub(layout)(torch.zeros(2, 21, 3, dtype=torch.float64), torch.as_tensor(extra))
        angles = matrix_to_axis_angle_np(rotations.numpy())
        np.testing.assert_allclose(angles[1, 22], [0.1, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(angles[1, 24], [0.0, 0.05, 0.0], atol=1e-9)
        parts = layout.split(extra)
        np.testing.assert_allclose(parts["jaw_deg"], [0.0, np.degrees(0.1)])
        np.testing.assert_allclose(parts["eye_deg"][1], [0.0, np.degrees(0.05)])
        np.testing.assert_allclose(parts["right_hand"][1].reshape(45),
                                   self.hands.mean[1] + 1.5 * self.hands.components[1, 1])
        np.testing.assert_allclose(parts["left_hand"][1].reshape(45), self.hands.mean[0])

    def test_priors_are_zero_at_rest_and_grow_away_from_it(self):
        import torch
        from bodyscan.dynamic.fullpose import PoseLayout
        from bodyscan.dynamic.track import TrackConfig, extra_priors
        layout = PoseLayout(self.model, "full55", self.hands)
        config = TrackConfig()
        rest = torch.zeros(1, layout.size, dtype=torch.float64)
        self.assertEqual(float(extra_priors(layout, rest, config, rest)), 0.0)
        jaw = rest.clone()
        jaw[0, 0] = 0.1
        self.assertAlmostEqual(float(extra_priors(layout, jaw, config)), config.face_weight * 0.01)
        hand = rest.clone()
        hand[0, 9] = 1.0
        self.assertGreater(float(extra_priors(layout, hand, config, rest)), float(extra_priors(layout, hand, config)))


class HoldoutTest(unittest.TestCase):
    def frame(self, holdout, seed=0, index=7):
        import torch
        from bodyscan.dynamic.segment import PersonFrame
        from bodyscan.dynamic.track import FrameData
        rows, columns = np.meshgrid(np.arange(2, 12), np.arange(1020, 1024 + 6))
        pixels = np.stack([rows.ravel(), np.mod(columns.ravel(), 1024)], axis=1)
        n = len(pixels)
        crop_range = np.full((14, 16), np.nan)
        origin = (0, 1018)
        crop_range[pixels[:, 0] - origin[0], np.mod(pixels[:, 1] - origin[1], 1024)] = 3.0
        mask = np.isfinite(crop_range).astype(np.uint8)
        person = PersonFrame(index, 1.0, np.random.default_rng(3).normal(size=(n, 3)), np.zeros(n), pixels,
                             np.full(n, 3.0), origin, mask, crop_range, np.full(crop_range.shape, 6.0), np.zeros(2))
        geometry = SimpleNamespace(width=1024, vertical_step=0.006, column_step=0.006)
        return person, FrameData(person, geometry, "cpu", torch.float64, holdout, seed)

    @needs_torch
    def test_a_fixed_share_is_held_out_the_same_way_every_time(self):
        person, frame = self.frame(0.2)
        n = len(person.points)
        self.assertEqual(len(frame.points_np) + len(frame.heldout_np), n)
        self.assertGreater(len(frame.heldout_np), 0.1 * n)
        self.assertLess(len(frame.heldout_np), 0.3 * n)
        _, again = self.frame(0.2)
        np.testing.assert_array_equal(again.heldout_np, frame.heldout_np)
        _, other = self.frame(0.2, index=8)
        self.assertFalse(np.array_equal(other.heldout_np, frame.heldout_np))
        # the ranges of the held-out pixels are gone (the free-space and in-front terms cannot use them),
        # the others stay; the silhouette mask keeps them, so they count as unknown, not as free space
        self.assertEqual(int(np.isfinite(frame.crop_range).sum()), len(frame.points_np))
        self.assertTrue(np.all(person.crop_mask == frame.mask))
        _, whole = self.frame(0.0)
        self.assertEqual(len(whole.heldout_np), 0)
        self.assertEqual(int(np.isfinite(whole.crop_range).sum()), n)


@needs_torch
class PoseSequenceFaceTest(unittest.TestCase):
    def test_jaw_and_eyes_round_trip_resample_and_pose(self):
        from bodyscan.body.rotations import matrix_to_axis_angle_np
        from bodyscan.dynamic.motions import PoseSequence
        folder = TemporaryFolder()
        try:
            times = np.array([0.0, 0.1, 0.2])
            jaw = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
            eyes = np.zeros((3, 2, 3))
            sequence = PoseSequence(times, np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 21, 3)), jaw=jaw, eyes=eyes)
            sequence.save(folder.path / "m.npz")
            loaded = PoseSequence.load(folder.path / "m.npz")
            np.testing.assert_allclose(loaded.jaw, jaw)
            self.assertIsNone(loaded.left_hand)
            np.testing.assert_allclose(loaded.sample([0.05]).jaw[0], [0.05, 0.0, 0.0], atol=1e-3)
            from tests.test_motion import load_test_body
            rotations = loaded.rotations(load_test_body())
            np.testing.assert_allclose(matrix_to_axis_angle_np(rotations.numpy())[2, 22], [0.2, 0.0, 0.0], atol=1e-9)
        finally:
            folder.cleanup()


@needs_torch
@slow
class FullPoseTrackingTest(unittest.TestCase):
    """A few frames of a synthetic walk: body21 from scratch, then body21 and full55 warm-started from it
    with 20 % of the points held out (short settings: this checks the plumbing, not the accuracy; about
    11 min on 2 CPU threads, so slow)."""

    @classmethod
    def setUpClass(cls):
        from bodyscan.body.testbody import cached
        from bodyscan.dynamic.bench import SimulateMotionConfig, make_take
        from bodyscan.dynamic.segment import MotionSegmenter
        from bodyscan.dynamic.track import TrackPipelineConfig, run_tracking
        from bodyscan.io import NpzRecording
        cls.folder = TemporaryFolder()
        config = SimulateMotionConfig()
        config.scenario.duration, config.scenario.distance, config.scenario.scan = 1.0, 2.2, False
        cls.take = make_take(cls.folder.path / "take", cached(), config, quiet=True)
        cls.segments = cls.folder.path / "segments"
        MotionSegmenter(NpzRecording(cls.take)).run(cls.segments)
        cls.avatar = cls.take / "truth_avatar.npz"
        cls.pca = write_hand_pca_file(cached(), cls.folder.path / "hands.npz")
        cls.start, cls.stop = 20, 26

        def short(pose_model="body21", init="", holdout=0.0):
            tracking = TrackPipelineConfig()
            tracking.tracking.rounds, tracking.tracking.iterations = 2, 8
            tracking.refine.rounds, tracking.refine.iterations, tracking.refine.window = 1, 6, 10
            tracking.tracking.pose_model, tracking.tracking.hand_pca = pose_model, str(cls.pca)
            tracking.tracking.init_motion, tracking.tracking.holdout_fraction = init, holdout
            return tracking

        cls.base = run_tracking(cls.segments, cls.avatar, cls.folder.path / "base.npz", short(), start=cls.start,
                                stop=cls.stop)
        warm = str(cls.folder.path / "base.npz")
        cls.body21 = run_tracking(cls.segments, cls.avatar, cls.folder.path / "body21.npz",
                                  short("body21", warm, 0.2), start=cls.start, stop=cls.stop)
        cls.full55 = run_tracking(cls.segments, cls.avatar, cls.folder.path / "full55.npz",
                                  short("full55", warm, 0.2), start=cls.start, stop=cls.stop,
                                  joints_csv=cls.folder.path / "joints.csv")

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_both_pose_models_report_the_same_points(self):
        for summary in (self.body21, self.full55):
            self.assertEqual(summary["frames"], self.stop - self.start)
            self.assertEqual(summary["warm_frames"], self.stop - self.start)
            self.assertTrue(np.isfinite(summary["heldout"]["residual_pooled"]["median_mm"]))
        # same held-out points, same hand neighbourhoods (from the warm start)
        self.assertEqual(self.body21["heldout"]["residual_pooled"]["points"],
                         self.full55["heldout"]["residual_pooled"]["points"])
        self.assertEqual(self.body21["hand_points_per_frame"], self.full55["hand_points_per_frame"])
        self.assertEqual(self.body21["pose_model"], "body21")
        self.assertNotIn("heldout", self.base)

    def test_full55_saves_the_extra_pose_and_its_provenance(self):
        from bodyscan.body.hands import file_sha1
        from bodyscan.dynamic.motions import PoseSequence
        with np.load(self.folder.path / "full55.npz", allow_pickle=False) as file:
            data = {key: file[key] for key in file.files}
        frames = self.stop - self.start
        self.assertEqual(str(data["pose_model"]), "full55")
        self.assertEqual(data["hand_coefficients"].shape, (frames, 2, 6))
        self.assertEqual(data["left_hand"].shape, (frames, 15, 3))
        self.assertEqual(data["jaw_deg"].shape, (frames,))
        self.assertEqual(data["eye_deg"].shape, (frames, 2))
        self.assertEqual(data["hand_points"].shape, (frames, 2))
        self.assertEqual(str(data["hand_pca_sha1"]), file_sha1(self.pca))
        self.assertEqual(str(data["model_sha1"]), json.loads(str(data["summary"]))["model_sha1"])
        self.assertEqual(str(data["init_motion_sha1"]), file_sha1(self.folder.path / "base.npz"))
        # the strong prior keeps the jaw and the eyes near rest: the sensor cannot see them
        self.assertLess(float(np.max(data["jaw_deg"])), 5.0)
        self.assertLess(float(np.max(data["eye_deg"])), 5.0)
        sequence = PoseSequence.load(self.folder.path / "full55.npz")
        self.assertIsNotNone(sequence.left_hand)
        self.assertIsNotNone(sequence.jaw)
        table = np.loadtxt(self.folder.path / "joints.csv", delimiter=",", skiprows=1)
        self.assertEqual(table.shape, (frames, 1 + 55 * 3))
        np.testing.assert_allclose(table[:, 1:4], data["joints"][:, 0], atol=1e-4)

    def test_body21_motion_keeps_its_old_keys_only(self):
        with np.load(self.folder.path / "body21.npz", allow_pickle=False) as data:
            keys = set(data.files)
        for key in ("left_hand", "jaw", "extra", "hand_pca_sha1"):
            self.assertNotIn(key, keys)

    def test_review_draws_the_hands(self):
        from bodyscan.dynamic.review import review
        written = review(self.folder.path / "full55.npz", self.segments, self.avatar, self.folder.path / "review",
                         every=2, focus="hands")
        self.assertTrue(any(path.name == "review.gif" for path in written))
