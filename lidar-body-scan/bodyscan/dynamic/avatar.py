"""The avatar: one person's body model, fitted to their static scan from the turntable.

Input: the fused cloud (bodyscan fuse) or the mesh (bodyscan mesh) of the
person standing in the A-pose, palms forward, in the output frame of the
turntable pipeline (z up, z = 0 on the platform top, any facing direction).

Stage 1, pose and shape. The model (SMPL-X) is posed and shaped to the scan:
root rotation and position, the first num_betas shape coefficients and the 21
body joint rotations. The loss sums, with a robust (Geman-McClure) penalty,
the distance from every scan point to the model surface (point to plane,
plus a little point to point) and from every model vertex to the scan where
the scan has data near it; plus a small penalty on the shape coefficients,
the anatomical limits, and soles on the floor. Correspondences are recomputed
between rounds of L-BFGS. The facing direction is unknown: the fit starts
from yaw_starts directions and keeps the best.

Stage 2, detail. The rest mesh is subdivided 'level' times and every vertex
moves along its rest normal by a displacement d (metres) that brings the
posed surface onto the scan; a Laplacian term keeps the displacements
smooth and fills the holes of the scan (soles, top of the head). Posing is
linear in d for a fixed pose (posed vertex = A + d B), so this stage needs
no skinning per iteration. The hands keep the model's shape (the LiDAR does
not resolve fingers) unless keep_hands is off.

The avatar file (.npz) stores the model path and checksum, the shape
coefficients, the level, the displacements, the hand pose and the fitted scan
pose; bodyscan.dynamic.avatar.Avatar.shaped() rebuilds the body for the
tracker (any level up to the fitted one: the coarse vertices keep their
indices, so a level-0 body takes the first displacements).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d

from bodyscan.body import skeleton
from bodyscan.config import param
from bodyscan.log import info


@dataclass
class AvatarConfig:
    """Fitting the body model to the static scan of a person (turntable)."""
    num_betas: int = param(10, "shape coefficients fitted (SMPL-X has up to 300)",
                           effect="more follow the scan more closely; the detail stage adds the rest")
    level: int = param(2, "subdivisions of the detailed surface (0: model resolution, about 10,000 vertices "
                          "for SMPL-X; each level multiplies the triangles by 4: level 2 gives about 3.6 mm edges)")
    sample_points: int = param(30000, "scan points used by the pose and shape fit")
    detail_points: int = param(150000, "scan points used by the detail fit")
    yaw_starts: int = param(4, "facing directions tried when the shoulders and feet do not tell the front")
    rounds: int = param(8, "correspondence rounds of the pose and shape fit")
    iterations: int = param(25, "L-BFGS iterations per round")
    robust_scale: float = param(0.02, "distance at which a correspondence stops counting fully (Geman-McClure)",
                                unit="m")
    coverage: float = param(0.04, "a model vertex is pulled to the scan only if a scan point lies within this",
                            unit="m")
    beta_weight: float = param(1e-3, "weight of the shape coefficients (keeps the shape plausible)")
    pose_weight: float = param(1e-2, "weight keeping the joint rotations near the A-pose")
    detail_rounds: int = param(4, "correspondence rounds of the detail fit")
    smoothness: float = param(0.5, "weight of the slope of the displacements between neighbouring vertices "
                                   "(larger: smoother detail, smaller: follows the scan noise)")
    max_displacement: float = param(0.05, "displacements are limited to this", unit="m")
    keep_hands: bool = param(True, "keep the model's hands and fingers (the LiDAR does not resolve fingers)")
    device: str = param("auto", "PyTorch device: auto (CUDA when available), cpu, cuda")


@dataclass
class Avatar:
    model_path: str
    model_sha1: str
    num_betas: int
    betas: np.ndarray
    level: int
    normal_displacement: np.ndarray
    scan_root_rotation: np.ndarray
    scan_root_position: np.ndarray
    scan_body: np.ndarray
    standing_height: float
    report: dict = field(default_factory=dict)

    def save(self, path) -> None:
        np.savez_compressed(path, model_path=self.model_path, model_sha1=self.model_sha1,
                            num_betas=np.int64(self.num_betas), betas=self.betas, level=np.int64(self.level),
                            normal_displacement=self.normal_displacement.astype(np.float32),
                            scan_root_rotation=self.scan_root_rotation, scan_root_position=self.scan_root_position,
                            scan_body=self.scan_body, standing_height=np.float64(self.standing_height),
                            report=json.dumps(self.report))

    @classmethod
    def load(cls, path) -> "Avatar":
        d = np.load(path, allow_pickle=False)
        return cls(str(d["model_path"]), str(d["model_sha1"]), int(d["num_betas"]), d["betas"], int(d["level"]),
                   d["normal_displacement"].astype(np.float64), d["scan_root_rotation"], d["scan_root_position"],
                   d["scan_body"], float(d["standing_height"]), json.loads(str(d["report"])))

    @classmethod
    def from_model(cls, model, betas=None, level: int = 0, normal_displacement=None, body=None) -> "Avatar":
        """An avatar without a scan: the model with given shape coefficients (synthetic
        tests, or a person without a turntable scan), standing in the A-pose."""
        import torch
        from bodyscan.dynamic.motions import a_pose, angles_to_body, foot_vertices
        betas = np.zeros(model.num_betas) if betas is None else np.asarray(betas, dtype=np.float64)
        body = angles_to_body(a_pose(1))[0] if body is None else np.asarray(body, dtype=np.float64)
        shaped = model.shaped(betas)
        with torch.no_grad():
            soles = shaped.pose(model.rotations(body=body[None]), None, model.tensor(np.zeros((1, 3))),
                                subset=foot_vertices(shaped)).vertices[0, :, 2]
        height = -float(soles.min().item())
        displacement = np.zeros(0) if normal_displacement is None else np.asarray(normal_displacement)
        return cls(str(model.path), model.sha1, model.num_betas, betas, level, displacement, np.zeros(3),
                   np.array([0.0, 0.0, height]), body, height, {"source": "model"})

    def model(self, path=None, device=None, dtype=None):
        """The body model of the avatar (path: where the model file is on this PC)."""
        import torch
        from bodyscan.body.model import BodyModel
        model = BodyModel(path or self.model_path, self.num_betas, device, dtype or torch.float32)
        if self.model_sha1 and model.sha1 != self.model_sha1:
            raise SystemExit(f"{model.path} is not the model file the avatar was fitted with (checksum differs)")
        return model

    def shaped(self, model, level: int | None = None):
        """The avatar's body at a subdivision level (default: the fitted one)."""
        level = self.level if level is None else min(level, self.level)
        displacement = None
        if self.normal_displacement.size:
            count = model.subdivision(level).counts[-1] if level > 0 else model.vertex_count
            displacement = self.normal_displacement[:count]
        return model.shaped(self.betas, level, displacement)


def _is_mesh_file(path: Path) -> bool:
    """True for a file with triangles: .obj, .stl, .off, or a .ply whose header declares faces."""
    suffix = path.suffix.lower()
    if suffix in (".obj", ".stl", ".off"):
        return True
    if suffix != ".ply":
        return False
    with open(path, "rb") as handle:
        for _ in range(200):
            line = handle.readline().decode("ascii", errors="replace").strip()
            if line.startswith("element face"):
                return int(line.split()[-1]) > 0
            if line == "end_header" or not line:
                return False
    return False


def load_scan(path, max_points: int | None = None, seed: int = 0):
    """Points and normals of a scan: a point cloud (.ply/.pcd/.xyz) or a mesh (.ply/.obj with triangles,
    sampled uniformly)."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path}: no such file")
    if _is_mesh_file(path):
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles) == 0:
            raise SystemExit(f"{path}: no triangles")
        from bodyscan.dynamic.fitting import sample_surface
        return sample_surface(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.triangles),
                              max_points or 200000, np.random.default_rng(seed))
    cloud = o3d.io.read_point_cloud(str(path))
    if not cloud.has_points():
        raise SystemExit(f"{path}: no points")
    if not cloud.has_normals():
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
        cloud.orient_normals_consistent_tangent_plane(20)
        _orient_outwards(cloud)
    points = np.asarray(cloud.points, dtype=np.float64)
    normals = np.asarray(cloud.normals, dtype=np.float64)
    if max_points and len(points) > max_points:
        pick = np.random.default_rng(seed).choice(len(points), max_points, replace=False)
        points, normals = points[pick], normals[pick]
    return points, normals


def _orient_outwards(cloud) -> None:
    """Normals pointing away from the vertical axis of the body (a standing person is roughly a cylinder)."""
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    axis = np.median(points[:, :2], axis=0)
    outward = points[:, :2] - axis
    flip = np.einsum("ij,ij->i", normals[:, :2], outward) < 0
    normals[flip] *= -1.0
    cloud.normals = o3d.utility.Vector3dVector(normals)


def facing_from_scan(points: np.ndarray) -> tuple[float, bool]:
    """Facing direction (yaw, radians; 0 = +x) of a person standing upright,
    and whether it is reliable. The shoulders give the left-right axis (the
    widest horizontal direction at shoulder height, arms included); the feet,
    which reach forward of the shins, give the sign."""
    z = points[:, 2]
    top = np.percentile(z, 99.5)
    shoulders = points[(z > 0.70 * top) & (z < 0.84 * top), :2]
    centered = shoulders - shoulders.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    across = axes[0]
    forward = np.array([-across[1], across[0]])
    feet = points[z < 0.05 * top, :2]
    shins = points[(z > 0.12 * top) & (z < 0.25 * top), :2]
    reliable = len(feet) > 50 and len(shins) > 50
    if reliable:
        reach = (feet.mean(axis=0) - shins.mean(axis=0)) @ forward
        if reach < 0:
            forward = -forward
        reliable = abs(reach) > 0.01
    return float(np.arctan2(forward[1], forward[0])), bool(reliable)


def _nearest(points: np.ndarray, queries: np.ndarray):
    """Index of and distance to the nearest of 'points' for every query."""
    search = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(points.astype(np.float32)))
    search.knn_index()
    index, squared = search.knn_search(o3d.core.Tensor(queries.astype(np.float32)), 1)
    return index.numpy()[:, 0].astype(np.int64), np.sqrt(squared.numpy()[:, 0].astype(np.float64))


class AvatarFitter:
    def __init__(self, model, config: AvatarConfig | None = None):
        self.model = model
        self.config = config or AvatarConfig()

    # ------------------------------------------------------------------------------------------
    def fit(self, points: np.ndarray, normals: np.ndarray) -> Avatar:
        import torch
        from bodyscan.body.rotations import axis_angle_to_matrix
        from bodyscan.dynamic.fitting import closest_points, face_normals, geman_mcclure, limit_penalty, surface_points
        from bodyscan.dynamic.motions import a_pose, angles_to_body, foot_vertices
        c = self.config
        model = self.model
        dtype, device = model.dtype, model.device
        rng = np.random.default_rng(0)
        sample = rng.choice(len(points), min(c.sample_points, len(points)), replace=False)
        p_np = points[sample]
        p_t = torch.as_tensor(p_np, dtype=dtype, device=device)
        stature = float(np.percentile(points[:, 2], 99.8) - np.percentile(points[:, 2], 0.2))
        center = np.median(points[:, :2], axis=0)
        base = model.shaped()
        apose = angles_to_body(a_pose(1))[0]
        # pelvis height of the rest model standing, scaled to the scan's stature
        rest = base.pose(model.rotations(body=apose[None]), None, torch.zeros(1, 3, dtype=dtype, device=device))
        model_height = float((rest.vertices[0, :, 2].max() - rest.vertices[0, :, 2].min()).item())
        pelvis_up = -float(rest.vertices[0, :, 2].min().item())
        height0 = pelvis_up * stature / model_height
        hand_parts = [skeleton.PART_NAMES.index("left_hand"), skeleton.PART_NAMES.index("right_hand")]
        labels = skeleton.part_labels(model.weights.cpu().numpy())
        not_hands = torch.as_tensor(~np.isin(labels, hand_parts) if c.keep_hands else np.ones(len(labels), bool),
                                    device=device)
        soles = torch.as_tensor(foot_vertices(base), device=device)
        faces_t = model.faces
        faces_np = model.faces_np

        def build(params):
            """Posed world vertices of the current parameters (differentiable)."""
            from bodyscan.body.model import ShapedBody
            v_shaped = model.v_template + torch.einsum("vcs,s->vc", model.shapedirs, params["betas"])
            shaped = ShapedBody(model, v_shaped, model.regressor @ v_shaped, 0)
            rotations = model.rotations(body=params["body"].reshape(1, 21, 3))
            posed = shaped.pose(rotations, axis_angle_to_matrix(params["root_rotation"].reshape(1, 3)),
                                params["root_position"].reshape(1, 3))
            return posed.vertices[0]

        def energy(params, corr, scale):
            """Data terms in units of the robust scale (each correspondence counts 0 to 1), plus priors."""
            vertices = build(params)
            face_ids, bary, model_index, scan_index = corr
            on_surface = surface_points(vertices, faces_t, face_ids, bary)
            normal = face_normals(vertices, faces_t, face_ids)
            diff = p_t - on_surface
            plane = (diff * normal).sum(-1)
            scale2 = scale ** 2
            loss = (geman_mcclure(plane, scale).mean() + 0.1 * geman_mcclure(diff.norm(dim=-1), scale).mean()) / scale2
            if len(model_index):
                to_scan = p_t[scan_index] - vertices[model_index]
                loss = loss + 0.5 * geman_mcclure(to_scan.norm(dim=-1), scale).mean() / scale2
            loss = loss + c.beta_weight * (params["betas"] ** 2).sum()
            loss = loss + c.pose_weight * ((params["body"].reshape(21, 3) - apose_t) ** 2).sum()
            loss = loss + limit_penalty(params["body"].reshape(1, 21, 3))
            sole_z = vertices[soles, 2]
            loss = loss + (10.0 * (torch.relu(-sole_z) ** 2).mean() + 0.2 * sole_z.min() ** 2) / scale2
            return loss

        apose_t = torch.as_tensor(apose, dtype=dtype, device=device)

        def correspondences(params, scale):
            with torch.no_grad():
                vertices = build(params).cpu().numpy().astype(np.float64)
            face_ids, bary, _ = closest_points(vertices, faces_np, p_np)
            index, distance = _nearest(p_np, vertices)
            keep = (distance < max(c.coverage, 2.0 * scale)) & not_hands.cpu().numpy()
            model_index = np.flatnonzero(keep)
            return (torch.as_tensor(face_ids, device=device), torch.as_tensor(bary, dtype=dtype, device=device),
                    torch.as_tensor(model_index, device=device), torch.as_tensor(index[model_index], device=device))

        def run(params, scales, iterations, free):
            """Rounds of correspondences then L-BFGS, one round per robust scale (coarse to fine)."""
            tensors = [params[name] for name in free]
            for tensor in params.values():
                tensor.requires_grad_(False)
            for tensor in tensors:
                tensor.requires_grad_(True)
            loss_value = None
            for scale in scales:
                corr = correspondences(params, scale)
                optimizer = torch.optim.LBFGS(tensors, lr=1.0, max_iter=iterations, history_size=20,
                                              line_search_fn="strong_wolfe")

                def closure():
                    optimizer.zero_grad()
                    loss = energy(params, corr, scale)
                    loss.backward()
                    return loss

                optimizer.step(closure)
                with torch.no_grad():
                    loss_value = float(energy(params, corr, c.robust_scale).item())
            return loss_value

        def start(yaw):
            return {"root_rotation": torch.tensor([0.0, 0.0, yaw], dtype=dtype, device=device),
                    "root_position": torch.tensor([center[0], center[1], height0], dtype=dtype, device=device),
                    "betas": torch.zeros(model.num_betas, dtype=dtype, device=device),
                    "body": apose_t.clone().reshape(-1)}

        yaw0, reliable = facing_from_scan(points)
        if reliable:
            starts = [yaw0 + np.radians(offset) for offset in (0.0, -25.0, 25.0)]
            info(f"  facing {np.degrees(yaw0):.0f} deg (shoulders and feet)")
        else:
            starts = [2 * np.pi * k / max(c.yaw_starts, 1) for k in range(max(c.yaw_starts, 1))]
            info("  facing direction unclear from the scan: trying several")
        best = None
        for yaw in starts:
            params = start(yaw)
            run(params, [0.12, 0.08], 10, ["root_rotation", "root_position"])
            loss = run(params, [0.08, 0.06, 0.04], 15, ["root_rotation", "root_position", "betas"])
            info(f"  start facing {np.degrees(yaw) % 360:5.0f} deg: loss {loss:.3e}")
            if best is None or loss < best[0]:
                best = (loss, params)
        params = best[1]
        fine = np.geomspace(0.06, c.robust_scale, max(c.rounds - 2, 1)).tolist() + [c.robust_scale] * 2
        run(params, fine, c.iterations, ["root_rotation", "root_position", "betas", "body"])
        with torch.no_grad():
            vertices = build(params).cpu().numpy()
        _, _, distance = closest_points(vertices, faces_np, points)
        report = {"stature_m": stature, "shape_fit_mm": _stats(1000 * distance)}
        info(f"  pose and shape: scan to model {report['shape_fit_mm']['median']:.1f} mm median, "
             f"{report['shape_fit_mm']['p90']:.1f} mm p90")
        betas = params["betas"].detach().cpu().numpy()
        body = params["body"].detach().cpu().numpy().reshape(21, 3)
        root_rotation = params["root_rotation"].detach().cpu().numpy()
        root_position = params["root_position"].detach().cpu().numpy()
        displacement = np.zeros(0)
        if c.level >= 0 and c.detail_rounds > 0:
            displacement, detail = self.fit_detail(points, betas, body, root_rotation, root_position)
            report.update(detail)
        sole_index = foot_vertices(model.shaped(betas))
        with torch.no_grad():
            standing = model.shaped(betas).pose(model.rotations(body=body[None]), None,
                                                torch.zeros(1, 3, dtype=dtype, device=device), subset=sole_index)
        standing_height = -float(standing.vertices[0, :, 2].min().item())
        return Avatar(str(model.path), model.sha1, model.num_betas, betas, c.level, displacement, root_rotation,
                      root_position, body, standing_height, report)

    # ------------------------------------------------------------------------------------------
    def fit_detail(self, points, betas, body, root_rotation, root_position):
        """Normal displacements at the configured level for the fitted pose."""
        import torch
        from bodyscan.body.rotations import axis_angle_to_matrix
        from bodyscan.dynamic.fitting import closest_points, face_normals, geman_mcclure, surface_points
        c = self.config
        model = self.model
        dtype, device = model.dtype, model.device
        shaped = model.shaped(betas, c.level)
        rotations = model.rotations(body=body[None])
        root = axis_angle_to_matrix(torch.as_tensor(root_rotation, dtype=dtype, device=device).reshape(1, 3))
        position = torch.as_tensor(root_position, dtype=dtype, device=device).reshape(1, 3)
        with torch.no_grad():
            plain = shaped.pose(rotations, root, position).vertices[0]
            moved = shaped.pose(rotations, root, position, vertices=shaped.vertices + shaped.rest_normals).vertices[0]
        direction = moved - plain                                           # posed rest normal (unit, rotated)
        faces_t, faces_np = shaped.faces, shaped.faces_np
        count = shaped.vertex_count
        labels = skeleton.part_labels(shaped.weights.cpu().numpy())
        hand_parts = [skeleton.PART_NAMES.index("left_hand"), skeleton.PART_NAMES.index("right_hand")]
        free = ~np.isin(labels, hand_parts) if c.keep_hands else np.ones(count, dtype=bool)
        free_t = torch.as_tensor(free, dtype=dtype, device=device)
        rng = np.random.default_rng(1)
        sample = rng.choice(len(points), min(c.detail_points, len(points)), replace=False)
        p_np = points[sample]
        p_t = torch.as_tensor(p_np, dtype=dtype, device=device)
        edges = np.unique(np.sort(np.concatenate([faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]]]),
                                  axis=1), axis=0)
        edges_t = torch.as_tensor(edges, device=device)
        d = torch.zeros(count, dtype=dtype, device=device, requires_grad=True)
        # smoothness: slope of the displacement along every edge (dimensionless, so it does not depend on the level)
        edge_length = (plain[edges_t[:, 0]] - plain[edges_t[:, 1]]).norm(dim=-1).clamp_min(1e-4)
        scale2 = c.robust_scale ** 2
        for _ in range(c.detail_rounds):
            with torch.no_grad():
                current = (plain + (d * free_t)[:, None] * direction).cpu().numpy().astype(np.float64)
            face_ids, bary, _ = closest_points(current, faces_np, p_np)
            face_ids_t = torch.as_tensor(face_ids, device=device)
            bary_t = torch.as_tensor(bary, dtype=dtype, device=device)
            optimizer = torch.optim.LBFGS([d], lr=1.0, max_iter=40, history_size=20, line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad()
                vertices = plain + (d * free_t)[:, None] * direction
                diff = p_t - surface_points(vertices, faces_t, face_ids_t, bary_t)
                normal = face_normals(vertices, faces_t, face_ids_t)
                plane = (diff * normal).sum(-1)
                loss = geman_mcclure(plane, c.robust_scale).mean() / scale2
                slope = (d[edges_t[:, 0]] - d[edges_t[:, 1]]) / edge_length
                loss = loss + c.smoothness * (slope ** 2).mean()
                loss = loss + (torch.relu(d.abs() - c.max_displacement) ** 2).sum() / scale2
                loss.backward()
                return loss

            optimizer.step(closure)
        result = (d * free_t).detach().clamp(-c.max_displacement, c.max_displacement).cpu().numpy()
        final = (plain + torch.as_tensor(result, dtype=dtype, device=device)[:, None] * direction).cpu().numpy()
        _, _, distance = closest_points(final.astype(np.float64), faces_np, points)
        report = {"detail_fit_mm": _stats(1000 * distance), "displacement_mm": _stats(1000 * np.abs(result[free])),
                  "level": c.level, "vertices": count}
        info(f"  detail (level {c.level}, {count} vertices): scan to avatar {report['detail_fit_mm']['median']:.1f} mm "
             f"median, {report['detail_fit_mm']['p90']:.1f} mm p90")
        return result, report


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
            "p99": float(np.percentile(values, 99)), "mean": float(values.mean())}


def write_meshes(avatar: Avatar, model, out_base) -> list[Path]:
    """<out>_rest.ply (T-pose with detail) and <out>_scan_pose.ply (as fitted to the scan)."""
    import torch
    from bodyscan.body.rotations import axis_angle_to_matrix_np
    shaped = avatar.shaped(model)
    paths = []
    with torch.no_grad():
        rest = shaped.pose(model.rotations(count=1), None, None).vertices[0].cpu().numpy()
        posed = shaped.pose(model.rotations(body=avatar.scan_body[None]),
                            model.tensor(axis_angle_to_matrix_np(avatar.scan_root_rotation)[None]),
                            model.tensor(avatar.scan_root_position[None])).vertices[0].cpu().numpy()
    for suffix, vertices in (("_rest.ply", rest), ("_scan_pose.ply", posed)):
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices.astype(np.float64)),
                                         o3d.utility.Vector3iVector(shaped.faces_np.astype(np.int32)))
        mesh.compute_vertex_normals()
        path = Path(str(out_base) + suffix)
        o3d.io.write_triangle_mesh(str(path), mesh)
        paths.append(path)
    return paths


def report_text(avatar: Avatar) -> str:
    r = avatar.report
    lines = [f"avatar: {avatar.num_betas} shape coefficients, level {avatar.level}",
             f"  scan stature {r.get('stature_m', float('nan')):.3f} m; pelvis {avatar.standing_height:.3f} m above "
             "the floor when standing"]
    if "shape_fit_mm" in r:
        s = r["shape_fit_mm"]
        lines.append(f"  pose and shape: scan to model median {s['median']:.1f} mm, p90 {s['p90']:.1f} mm")
    if "detail_fit_mm" in r:
        s = r["detail_fit_mm"]
        lines.append(f"  with detail: scan to avatar median {s['median']:.1f} mm, p90 {s['p90']:.1f} mm, "
                     f"p99 {s['p99']:.1f} mm")
    return "\n".join(lines)


