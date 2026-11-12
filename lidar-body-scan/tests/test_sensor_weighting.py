"""Weighting of additional sensors in the tracker (bodyscan 1.8): noise that grows with range, the range of
the person at which a sensor's frames are used, the parsing of their options, the toe prior and the count
the data term is divided by.

    python -m unittest tests.test_sensor_weighting
"""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None


class SensorOptionsTest(unittest.TestCase):
    def test_options_are_parsed_per_sensor(self):
        from bodyscan.dynamic.views import parse_sensor_options
        self.assertEqual(parse_sensor_options("xtion:2.0:0.26, helios:3.4:5.2", 2, "x"),
                         {"xtion": (2.0, 0.26), "helios": (3.4, 5.2)})
        self.assertEqual(parse_sensor_options("", 2, "x"), {})

    def test_a_malformed_option_stops(self):
        from bodyscan.dynamic.views import parse_sensor_options
        with self.assertRaises(SystemExit):
            parse_sensor_options("xtion:2.0", 2, "extra_sensor_noise")
        with self.assertRaises(SystemExit):
            parse_sensor_options("xtion:a:b", 2, "extra_sensor_noise")


class SensorViewNoiseTest(unittest.TestCase):
    def view(self, **options):
        from bodyscan.dynamic.views import SensorView
        geometry = SimpleNamespace(translation=np.array([1.0, 2.0, 1.0]))
        return SensorView("xtion", geometry, None, **options)

    def test_constant_noise_without_a_quadratic_term(self):
        v = self.view(sigma=3.21)
        np.testing.assert_allclose(v.point_sigma(np.zeros((4, 3))), 3.21)

    def test_noise_grows_with_the_square_of_the_range(self):
        v = self.view(sigma=2.0, sigma_quadratic=0.26)
        position = v.position
        points = position + np.array([[1.0, 0, 0], [2.0, 0, 0], [3.25, 0, 0], [4.0, 0, 0]])
        np.testing.assert_allclose(v.point_sigma(points), [2.0, 2.0, 0.26 * 3.25 ** 2, 0.26 * 16.0])

    def test_frames_are_used_only_with_the_person_in_range(self):
        v = self.view(person_range=(1.3, 4.2))
        self.assertTrue(v.in_range(np.array([2.0, 2.5, 3.0])))
        self.assertFalse(v.in_range(np.array([0.9, 1.0, 1.1])))
        self.assertFalse(v.in_range(np.array([4.5, 4.6])))
        self.assertTrue(v.in_range(np.zeros(0)))
        self.assertTrue(self.view().in_range(np.array([0.5])))


@unittest.skipUnless(HAS_TORCH, "needs PyTorch")
class ToePriorTest(unittest.TestCase):
    def test_the_toe_prior_pulls_only_the_toes(self):
        import torch
        from bodyscan.body import skeleton
        from bodyscan.dynamic.track import TrackConfig, neutral_priors
        body = torch.zeros(1, 21, 3, dtype=torch.float64)
        config = TrackConfig()
        config.toe_weight = 0.0
        base = float(neutral_priors(body, config))
        body[0, skeleton.JOINT["left_foot"] - 1, 0] = -1.0
        self.assertAlmostEqual(float(neutral_priors(body, config)), base)
        config.toe_weight = 0.3
        self.assertAlmostEqual(float(neutral_priors(body, config)) - base, 0.3)


class DataCountTest(unittest.TestCase):
    def test_the_data_term_is_divided_by_the_ouster_points(self):
        from bodyscan.dynamic.track import Tracker
        import torch
        tracker = SimpleNamespace(config=SimpleNamespace(tracking=SimpleNamespace(reference_count=True)))
        frame = SimpleNamespace(reference_points=100)
        # 60 matched Ouster points (indices < 100) and 40 of another sensor
        match = SimpleNamespace(point_index=torch.cat([torch.arange(60), torch.arange(100, 140)]))
        self.assertEqual(Tracker.data_count(tracker, frame, match), 60)
        tracker.config.tracking.reference_count = False
        self.assertEqual(Tracker.data_count(tracker, frame, match), 100)

    def test_without_ouster_points_all_points_count(self):
        from bodyscan.dynamic.track import Tracker
        import torch
        tracker = SimpleNamespace(config=SimpleNamespace(tracking=SimpleNamespace(reference_count=True)))
        frame = SimpleNamespace(reference_points=0)
        match = SimpleNamespace(point_index=torch.arange(100, 130))
        self.assertEqual(Tracker.data_count(tracker, frame, match), 30)


if __name__ == "__main__":
    unittest.main()
