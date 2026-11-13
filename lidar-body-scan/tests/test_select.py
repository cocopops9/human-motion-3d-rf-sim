"""Selection among several tracked motions (bodyscan.dynamic.select): the path of least cost and the pose put
together from the motions.

    python -m unittest tests.test_select
"""

from __future__ import annotations

import unittest

import numpy as np

from bodyscan.dynamic.select import ARM_JOINTS, Composite, least_cost_path


def positions(offsets, frames):
    """(K, F, 1, 3): motion k stands 'offsets[k]' metres along x at every frame."""
    return np.array([[[[offset, 0.0, 0.0]]] * frames for offset in offsets])


class LeastCostPathTest(unittest.TestCase):
    def test_the_better_motion_is_chosen_frame_by_frame_where_the_motions_coincide(self):
        scores = np.array([[1.0, 5.0, 1.0, 5.0], [5.0, 1.0, 5.0, 1.0]])
        path = least_cost_path(scores, positions([0.0, 0.01], 4), weight=0.05, free=0.03)
        self.assertEqual(path.tolist(), [0, 1, 0, 1])

    def test_a_change_between_distant_motions_must_pay_for_itself(self):
        # motion 1 is worse except in one frame, where it is better by 0.002: not worth a jump of 0.5 m
        # there and back
        scores = np.full((2, 9), 0.010)
        scores[1] = 0.011
        scores[1, 4] = 0.008
        path = least_cost_path(scores, positions([0.0, 0.5], 9), weight=0.05, free=0.03)
        self.assertEqual(path.tolist(), [0] * 9)
        # worse by 0.010 in every frame of the first half, better by 0.008 in the second: one change
        scores[1, :4] = 0.020
        scores[1, 4:] = 0.002
        path = least_cost_path(scores, positions([0.0, 0.5], 9), weight=0.05, free=0.03)
        self.assertEqual(path.tolist(), [0] * 4 + [1] * 5)

    def test_one_motion_alone_is_its_own_path(self):
        path = least_cost_path(np.ones((1, 5)), positions([0.0], 5), weight=0.05, free=0.03)
        self.assertEqual(path.tolist(), [0] * 5)


class CompositeTest(unittest.TestCase):
    def motion(self, value):
        return {"root_matrix": np.repeat(np.eye(3)[None] * value, 3, axis=0), "root_position": np.full((3, 3), value),
                "body": np.full((3, 21, 3), value)}

    def test_each_arm_comes_from_its_own_motion(self):
        composite = Composite([self.motion(1.0), self.motion(2.0), self.motion(3.0)], rest=np.array([0, 0, 1]))
        composite.paths["left"] = np.array([2, 0, 1])
        rotation, position, body = composite.pose(0)
        self.assertEqual(float(position[0]), 1.0)
        self.assertTrue(np.all(body[ARM_JOINTS["left"]] == 3.0))
        others = [j for j in range(21) if j not in ARM_JOINTS["left"]]
        self.assertTrue(np.all(body[others] == 1.0))
        # an override replaces one part for that call only
        _, _, tried = composite.pose(0, ("right", 1))
        self.assertTrue(np.all(tried[ARM_JOINTS["right"]] == 2.0))
        self.assertTrue(np.all(composite.pose(0)[2][ARM_JOINTS["right"]] == 1.0))
        self.assertEqual(float(composite.pose(2)[1][0]), 2.0)


if __name__ == "__main__":
    unittest.main()
