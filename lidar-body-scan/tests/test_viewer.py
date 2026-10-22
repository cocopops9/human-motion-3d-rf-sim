"""The viewer's data layer and, when viser is installed, a headless run of the application."""

from __future__ import annotations

import importlib.util
import shutil
import unittest

import numpy as np

from tests.helpers import TemporaryFolder
from tests.test_evaluation import make_synthetic_take, needs_torch

HAS_VISER = importlib.util.find_spec("viser") is not None


@needs_torch
class ViewerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bodyscan.evaluation.take import TakeEvaluationConfig, evaluate_take, write_take_results
        cls.folder = TemporaryFolder()
        paths = make_synthetic_take(cls.folder.path / "synthetic")
        root = cls.folder.path
        for name in ("s02_stand", "s02_stand_2"):
            (root / "results" / name).mkdir(parents=True)
            shutil.copy(paths["true_motion"], root / "results" / name / "motion.npz")
            shutil.copytree(paths["segments"], root / "segments" / f"{name}_seg")
        result = evaluate_take(paths["true_motion"], paths["segments"], paths["avatar"],
                               config=TakeEvaluationConfig(sigma=0.008))
        write_take_results(result, root / "evaluations" / "s02_stand")
        cls.root, cls.avatar = root, paths["avatar"]

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_catalogue_groups_the_repetitions(self):
        from bodyscan.viewer.data import catalogue, motion_type
        self.assertEqual(motion_type("s02_walk03_2_radial"), "s02_walk03_radial")
        self.assertEqual(motion_type("s02_jump02_3"), "s02_jump02")
        groups = catalogue(self.root / "results", self.root / "segments", self.root / "evaluations")
        self.assertEqual(list(groups), ["s02_stand"])
        entries = groups["s02_stand"]
        self.assertEqual([e.name for e in entries], ["s02_stand", "s02_stand_2"])
        self.assertIsNotNone(entries[0].evaluation)
        self.assertIsNone(entries[1].evaluation)

    def test_take_data_frame_by_frame(self):
        from bodyscan.viewer.data import VERTEX_MODES, TakeData, catalogue
        entry = catalogue(self.root / "results", self.root / "segments", self.root / "evaluations")["s02_stand"][0]
        take = TakeData(entry.motion, entry.segments, self.avatar, None, entry.evaluation)
        self.assertGreater(len(take), 5)
        vertices = take.vertices(2)
        self.assertEqual(vertices.shape, (len(take.labels), 3))
        self.assertIs(take.vertices(2), vertices)                        # cached
        points = take.points(2)
        self.assertGreater(len(points), 50)
        self.assertLess(np.median(take.point_distances(2)), 0.015)
        for mode in VERTEX_MODES[2:]:
            values = take.vertex_values(2, mode)
            self.assertEqual(values.shape, (len(take.labels),))
            if mode == "coverage":
                self.assertTrue((values == 0).any())                     # the back is never seen
            else:
                self.assertTrue(np.isnan(values).any())                  # ... so it has no error value
        self.assertIn("Frame", take.frame_text(2))
        self.assertIn("Pixels", take.take_text())
        series = take.series()
        self.assertEqual(len(series["violation_share"]), len(take))

    @unittest.skipUnless(HAS_VISER, "needs viser (python -m pip install viser)")
    def test_application_draws_every_mode(self):
        from bodyscan.viewer.app import ViewerSession
        from bodyscan.viewer.data import VERTEX_MODES, catalogue
        groups = catalogue(self.root / "results", self.root / "segments", self.root / "evaluations")
        session = ViewerSession(groups, self.avatar, None, port=18000 + np.random.default_rng().integers(0, 2000),
                                host="127.0.0.1")
        try:
            session.load("s02_stand", "s02_stand")
            for mode in VERTEX_MODES:
                session.body_mode.value = mode
                session.show(3)
            session.point_mode.value = "by sensor"
            session.show_skeleton.value = True
            session.show_body.value = False
            session.show(4)
            self.assertEqual(session.frame, 4)
            self.assertIn("Frame", session.frame_text.content)
            session.load("s02_stand", "s02_stand_2")                     # a take without an evaluation
            session.body_mode.value = "median error"
            session.show_body.value = True
            session.show(1)
            self.assertIn("evaluate-take", session.legend.content)
        finally:
            session.stop()


if __name__ == "__main__":
    unittest.main()
