"""End-to-end runs of the pipelines on synthetic data (the fusion tests are slow:
BODYSCAN_SLOW_TESTS=1 python -m unittest discover -s tests)."""

import json
import unittest

import numpy as np
import open3d as o3d

from bodyscan.pipelines.inplace import InPlaceConfig, InPlacePipeline
from bodyscan.pipelines.mesh import MeshConfig, MeshPipeline
from bodyscan.pipelines.smooth import SmoothMeshConfig, SmoothMeshPipeline, parse_sweeps, variants
from bodyscan.pipelines.turntable import TurntableConfig, TurntablePipeline
from bodyscan.synthetic import SyntheticSensor, person, scenario
from tests.helpers import TemporaryFolder, slow


def distance_to(mesh, points) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(np.asarray(points, dtype=np.float32))).numpy()


class MeshPipelineTest(unittest.TestCase):
    def test_closed_mesh_then_smoothing_from_a_noisy_cloud(self):
        folder = TemporaryFolder()
        try:
            truth = o3d.geometry.TriangleMesh.create_sphere(0.25, resolution=60).translate((0, 0, 0.25))
            cloud = truth.sample_points_uniformly(40000)
            points = np.asarray(cloud.points)
            normals = (points - [0, 0, 0.25]) / 0.25
            points += np.random.default_rng(0).normal(0, 0.002, len(points))[:, None] * normals
            cloud.points, cloud.normals = o3d.utility.Vector3dVector(points), o3d.utility.Vector3dVector(normals)
            o3d.io.write_point_cloud(str(folder.path / "ball.ply"), cloud)
            ctx = MeshPipeline(MeshConfig()).run(cloud_path=folder.path / "ball.ply", out=folder.path / "ball_mesh.ply")
            mesh = o3d.io.read_triangle_mesh(str(folder.path / "ball_mesh.ply"))
            report = json.loads((folder.path / "ball_mesh_quality.json").read_text())
            self.assertTrue(report["quality"]["closed"])
            vertices = np.asarray(mesh.vertices)
            above_cut = vertices[vertices[:, 2] > 0.01]                 # the flat cut at z = 0 is not on the ball
            self.assertLess(np.median(distance_to(truth, above_cut)), 0.002)
            self.assertNotIn("smoothing", ctx.report.get("config", {}))             # conversion only

            config = SmoothMeshConfig()
            config.compare.sweep = ("rounds=1,2",)
            smoothed = SmoothMeshPipeline(config).run(mesh_path=folder.path / "ball_mesh.ply",
                                                      out=folder.path / "ball_smooth.ply")
            for rounds in (1, 2):
                path = folder.path / f"ball_smooth_rounds{rounds}.ply"
                self.assertTrue(path.exists())
                written = json.loads((folder.path / f"ball_smooth_rounds{rounds}_quality.json").read_text())
                self.assertEqual(written["settings"]["rounds"], rounds)
                self.assertTrue(written["quality"]["closed"])
            facet = [row[1]["facet_noise_deg"]["median"] for row in smoothed.rows]
            self.assertLess(facet[2], facet[1])                         # 2 rounds smoother than 1
            self.assertLess(facet[1], facet[0])                         # 1 round smoother than the input
        finally:
            folder.cleanup()


class SweepTest(unittest.TestCase):
    def test_sweep_values_become_configurations(self):
        from bodyscan.meshing.smoother import SmoothingConfig
        runs = variants(SmoothingConfig(), parse_sweeps(["rounds=1,8", "scale-mm=6,12"]))
        self.assertEqual([suffix for suffix, _, _ in runs], ["_rounds1_scale_mm6", "_rounds1_scale_mm12",
                                                             "_rounds8_scale_mm6", "_rounds8_scale_mm12"])
        self.assertEqual(runs[2][1], "rounds=8 scale_mm=6")
        self.assertEqual((runs[2][2].rounds, runs[2][2].scale_mm), (8, 6.0))
        self.assertEqual(variants(SmoothingConfig(), [])[0][:2], ("", "smoothed"))
        for wrong in (["level=3"], ["rounds"], ["rounds="]):
            with self.assertRaises(SystemExit):
                parse_sweeps(wrong)
        with self.assertRaises((SystemExit, ValueError)):
            variants(SmoothingConfig(), parse_sweeps(["rounds=many"]))


class FirstStepsTest(unittest.TestCase):
    """The first steps of the fusion pipelines on short synthetic recordings (fast)."""

    @classmethod
    def setUpClass(cls):
        cls.folder = TemporaryFolder()
        sensor = SyntheticSensor(rows=128, columns=1024)
        cls.turntable = scenario("turntable", cls.folder.path / "tt", frames=30, sensor=sensor)
        cls.inplace = scenario("inplace", cls.folder.path / "ip", frames=130, sensor=SyntheticSensor(rows=128,
                                                                                                    columns=1024))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_turntable_platform_and_person(self):
        from bodyscan.pipelines.base import Context
        from bodyscan.pipelines.turntable import IsolatePerson, LoadTurntableRecording, LocatePlatform
        from bodyscan.pipelines.common import SceneFromBackground
        ctx = Context(config=TurntableConfig(), report={}, run_dir=self.turntable, out=self.folder.path / "x")
        for step in (LoadTurntableRecording(), SceneFromBackground(), LocatePlatform(), IsolatePerson()):
            step.run(ctx)
        np.testing.assert_allclose(ctx.center, (1.2, 0.1), atol=0.05)
        self.assertEqual(int(ctx.usable.sum()), 30)

    def test_inplace_person_and_still_periods(self):
        from bodyscan.pipelines.base import Context
        from bodyscan.pipelines.common import LoadRecording, SceneFromBackground
        from bodyscan.pipelines.inplace import FindStillPeriods, LocatePerson
        ctx = Context(config=InPlaceConfig(), report={}, run_dir=self.inplace, out=self.folder.path / "y")
        for step in (LoadRecording(), SceneFromBackground(), LocatePerson(), FindStillPeriods()):
            step.run(ctx)
        np.testing.assert_allclose(ctx.report["region_center"], (1.6, 0.2), atol=0.08)
        self.assertGreaterEqual(len(ctx.segments), 3)                   # 13 s: stops every 4 s


@slow
class TurntableFusionTest(unittest.TestCase):
    def test_person_on_the_platform(self):
        folder = TemporaryFolder()
        try:
            run = scenario("turntable", folder.path / "tt", sensor=SyntheticSensor(rows=128, columns=2048))
            truth = json.loads((run / "truth.json").read_text())
            config = TurntableConfig()
            config.motion.ignore_phases = True
            ctx = TurntablePipeline(config).run(run_dir=run, out=folder.path / "person")
            np.testing.assert_allclose(ctx.axis, truth["axis"], atol=0.01)
            fused = np.asarray(o3d.io.read_point_cloud(str(folder.path / "person.ply")).points)
            body = person().translate((0.02, -0.01, 0.0))           # on the platform, axis at the origin
            # Output frame: axes of the first frame, origin on the axis at the platform top.
            distance = distance_to(body, fused)
            self.assertLess(np.median(distance), 0.006)
        finally:
            folder.cleanup()


@slow
class InPlaceFusionTest(unittest.TestCase):
    def test_person_turning_by_steps(self):
        folder = TemporaryFolder()
        try:
            # 60 s: 15 steps of 25 deg, one every 4 s
            run = scenario("inplace", folder.path / "ip", frames=600, sensor=SyntheticSensor(rows=128, columns=2048))
            ctx = InPlacePipeline(InPlaceConfig()).run(run_dir=run, out=folder.path / "person")
            self.assertGreater(ctx.report["coverage_deg"], 330.0)
            steps = np.diff(ctx.report["turn_deg"])
            self.assertLess(np.abs(steps - 25.0).max(), 1.5)
        finally:
            folder.cleanup()


if __name__ == "__main__":
    unittest.main()
