"""Shared helpers of the tests (python -m unittest discover -s tests from the repository folder)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from bodyscan.log import set_quiet

set_quiet(True)

SLOW = os.environ.get("BODYSCAN_SLOW_TESTS", "") not in ("", "0")
slow = unittest.skipUnless(SLOW, "slow test: set BODYSCAN_SLOW_TESTS=1 to run it")


def sphere_mesh(radius=0.3, resolution=40, noise=0.0, seed=0) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=resolution)
    if noise > 0:
        vertices = np.asarray(mesh.vertices)
        rng = np.random.default_rng(seed)
        vertices += rng.normal(0.0, noise, vertices.shape[0])[:, None] * vertices / radius
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    return mesh


class TemporaryFolder:
    """A temporary folder kept for the whole test class (class-level fixtures)."""

    def __init__(self):
        self.handle = tempfile.TemporaryDirectory(prefix="bodyscan_test_")
        self.path = Path(self.handle.name)

    def cleanup(self):
        self.handle.cleanup()
