"""Tracking: the avatar fitted to every frame of a segmented recording.

The motion comes from the LiDAR: every frame the avatar is posed so that its
surface passes through the person's points. What the sensor cannot see (the
far side of the body, an arm behind the torso) is filled by constraints that
keep the motion human, never by a learned or recorded motion:

    data          every point of the person close to the visible surface of the
                  body (robust point-to-plane distance, a little point-to-point;
                  only faces the sensor can see take correspondences)
    silhouette    no visible part of the body where the sensor saw through to
                  the background: a vertex that projects onto a pixel whose
                  measured range lies beyond it is pulled towards the person's
                  silhouette (distance map of the silhouette crop, in angle,
                  times the range); a vertex in front of the measured surface
                  inside the silhouette is pushed back to it
    limits        anatomical joint limits (bodyscan.body.skeleton.LIMITS_DEG)
    floor         no vertex of the feet below the floor
    continuity    frame by frame: close to the motion predicted from the
                  previous frames (constant velocity)

Each frame is fitted coarse to fine (the robust scale shrinks from
coarse_scale x robust_scale to robust_scale over the rounds), so that a limb
that moved fast is still caught. When points of the person remain far from
the body after the fit (an arm swung overhead in a jump, knees folding at a
landing), the frame is fitted again from other starts (the previous pose,
the legs and each arm from a few poses) and the start that explains the
points best is kept. Body parts the sensor hardly sees get a stronger
continuity (stage 1) and smoothness (stage 2), so that they keep their
motion instead of jumping where a few points pull them.

Stage 2 refines the whole sequence in overlapping windows: the same data,
silhouette, limit and floor terms for every frame, plus

    smoothness    the accelerations of the joints change smoothly (a penalty
                  on the jerk, Charbonnier: small jitter is removed, real sharp
                  events such as a heel strike or a landing stay; a body in
                  free fall or turning steadily costs nothing, which a penalty
                  on the acceleration itself would flatten)
    contact       a foot found standing on the floor (low and slow in stage 1)
                  does not slide
    rolling shutter  each point, and each pixel of the silhouette terms, is
                  compared with the body at the time it was measured (the
                  sensor needs one turn per frame: a person spans a few
                  milliseconds of it, or a whole turn when standing where the
                  frame starts)

Output motion.npz: the pose of every frame (times on the sensor clock,
root rotation and position, body joint rotations: a PoseSequence), the
joints, the foot contacts, per frame and per body part the fraction of the
part that the sensor actually observed, and quality numbers that flag frames
for review.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bodyscan.body import skeleton
from bodyscan.config import param, section
from bodyscan.log import Progress, info, warning


@dataclass
class TrackConfig:
    """Fitting the avatar to every frame (stage 1, frame by frame)."""
    level: int = param(0, "subdivision level of the avatar used for tracking (0: model resolution, the "
                          "fastest; the export can use a finer level)")
    device: str = param("auto", "PyTorch device: auto (CUDA when available), cpu, cuda")
    robust_scale: float = param(0.03, "distance at which a point stops counting fully (Geman-McClure)", unit="m",
                                effect="larger: smoother but more biased by stray points; smaller: needs a "
                                       "good start")
    point_to_point: float = param(0.1, "weight of the point-to-point distance next to the point-to-plane one")
    max_correspondence: float = param(0.25, "points farther than this from the visible body are ignored", unit="m")
    visibility_tolerance: float = param(0.03, "a vertex behind another by more than this is hidden", unit="m")
    silhouette_weight: float = param(2.0, "weight of the silhouette and free-space term",
                                     effect="larger keeps hidden limbs out of the free space more firmly")
    silhouette_scale: float = param(0.06, "robust scale of the silhouette distances", unit="m")
    free_space_margin: float = param(0.06, "a vertex counts as in free space when the sensor saw this much "
                                           "beyond it", unit="m")
    limit_weight: float = param(5.0, "weight of the anatomical joint limits")
    twist_weight: float = param(0.05, "weight pulling the twist of the spine and the rotation of the hips about "
                                      "the legs towards zero [per rad^2]: the pelvis can turn and the trunk and "
                                      "legs turn back by as much, which the points hardly tell apart")
    floor_weight: float = param(5.0, "weight of the no-foot-below-the-floor term")
    pose_continuity: float = param(0.05, "weight of the joint rotations staying near the prediction "
                                         "(constant velocity) [per rad^2]",
                                   effect="larger: steadier hidden limbs, slower to follow fast motion")
    root_continuity: float = param(0.02, "weight of the root staying near the prediction")
    hidden_continuity: float = param(10.0, "the continuity weight of a joint whose body part is less than 30 % "
                                           "visible is multiplied by up to 1 + this (fully hidden): a hidden limb "
                                           "keeps its motion instead of jumping where a few points pull it")
    rounds: int = param(3, "correspondence rounds per frame")
    iterations: int = param(12, "L-BFGS iterations per round")
    coarse_scale: float = param(4.0, "robust scale of the first round, as a multiple of robust_scale (halved every "
                                     "round down to robust_scale)",
                                effect="larger: catches faster limbs, but stray points pull more in the first round")
    init_yaws: int = param(8, "facing directions tried on the first frame and after the person is lost")
    lost_residual: float = param(0.05, "median point distance above which a frame is fitted again from several "
                                       "starts", unit="m")
    recover_distance: float = param(0.06, "a point farther than this from the visible body is unexplained", unit="m")
    recover_fraction: float = param(0.02, "fraction of unexplained points above which the frame is fitted again "
                                          "from other starts: the previous pose, legs, arms (0: never)",
                                    effect="smaller: more searches (about 5 s per frame on a CPU)")


@dataclass
class RefineConfig:
    """Whole-sequence refinement (stage 2): natural motion."""
    enabled: bool = param(True, "refine the whole sequence after the frame-by-frame fit")
    window: int = param(40, "frames optimised together")
    overlap: int = param(10, "frames shared with the previous window (held, for continuity)")
    rounds: int = param(2, "correspondence rounds per window")
    iterations: int = param(40, "L-BFGS iterations per round")
    smoothness: str = param("jerk", "what the smoothness term penalises: the jerk (change of acceleration) or "
                                    "the acceleration of the joints", choices=("jerk", "acceleration"),
                            effect="acceleration also pulls a jump's flight (free fall, 9.81 m/s2) flatter")
    smoothness_weight: float = param(0.5, "weight of the smoothness term",
                                     effect="larger: smoother motion, sharp events (landings) softened")
    hidden_smoothness: float = param(10.0, "the smoothness weight of a joint whose body part is less than 30 % "
                                           "visible is multiplied by up to 1 + this (fully hidden)")
    smoothness_scale: float = param(200.0, "Charbonnier scale of the smoothness term: below it smoothed like a "
                                           "spring, above it penalised linearly (m/s3 for jerk, m/s2 for "
                                           "acceleration)")
    contact_height: float = param(0.035, "a foot whose lowest vertex is below this is on the floor if also slow",
                                  unit="m")
    contact_speed: float = param(0.35, "... and its ankle moves slower than this", unit="m/s")
    skating_weight: float = param(2.0, "weight of the no-sliding of feet on the floor")
    rolling_shutter: bool = param(True, "compare every point with the body at the time it was measured")


@dataclass
class TrackPipelineConfig:
    tracking: TrackConfig = section(TrackConfig)
    refine: RefineConfig = section(RefineConfig)


# Starting poses of an arm for the search of recover_limbs: (elevation, swing,
# twist, elbow) [deg] as in bodyscan.dynamic.motions.Angles (elevation 0:
# hanging, 90: horizontal at the side; swing: forward; twist 90: palm forward).
ARM_STARTS = (
    (40, 0, 90, 10),        # A-pose
    (10, 0, 90, 15),        # hanging
    (10, 0, 90, 90),        # hanging, elbow bent
    (90, 0, 90, 10),        # horizontal at the side
    (15, 90, 0, 10),        # horizontal in front
    (15, 160, 0, 10),       # up, in front
    (165, 0, 90, 10),       # up, at the side
    (15, -40, 0, 15),       # back
)


# Joints whose rotation about the long axis (y in the T-pose frames: the spine,
# the thighs) trades with a turn of the pelvis (indices into the 21 body joints).
TWIST_JOINTS = [skeleton.JOINT[name] - 1 for name in ("spine1", "spine2", "spine3", "left_hip", "right_hip")]

# Starting poses of both legs for the same search: (hip flexion, knee flexion,
# ankle dorsiflexion) [deg]: standing, half squat, deep squat (a landing).
LEG_STARTS = ((0, 3, 0), (35, 55, 15), (65, 95, 25))


def _leg_rotations(hip, knee, ankle) -> dict:
    """Hip, knee and ankle rotations of both legs: {index in the (21, 3) body pose: axis-angle}."""
    from bodyscan.dynamic.motions import Angles, angles_to_body
    angles = Angles(1)
    for side in ("left", "right"):
        for name, value in (("hip", hip), ("knee", knee), ("ankle", ankle)):
            getattr(angles, f"{side}_{name}")[:] = value
    body = angles_to_body(angles)[0]
    return {skeleton.JOINT[f"{side}_{name}"] - 1: body[skeleton.JOINT[f"{side}_{name}"] - 1]
            for side in ("left", "right") for name in ("hip", "knee", "ankle")}


def _arm_rotations(side: str, elevation, swing, twist, elbow) -> dict:
    """Shoulder and elbow rotations of one arm: {index in the (21, 3) body pose: axis-angle}."""
    from bodyscan.dynamic.motions import Angles, angles_to_body
    angles = Angles(1)
    for name, value in (("elevation", elevation), ("swing", swing), ("twist", twist), ("elbow", elbow)):
        getattr(angles, f"{side}_{name}")[:] = value
    body = angles_to_body(angles)[0]
    return {index: body[index] for index in (skeleton.JOINT[f"{side}_shoulder"] - 1,
                                               skeleton.JOINT[f"{side}_elbow"] - 1)}


# ----------------------------------------------------------------------------
# Observations of one frame
# ----------------------------------------------------------------------------

class FrameData:
    """One segmented frame on the device: the person's points and the silhouette crop."""

    def __init__(self, person, geometry, device, dtype):
        import torch
        from bodyscan.dynamic.fitting import distance_map
        self.index, self.time = person.index, person.time
        self.points_np = person.points
        self.points = torch.as_tensor(person.points, dtype=dtype, device=device)
        self.point_times = person.point_times
        self.crop_origin = person.crop_origin
        self.mask = person.crop_mask
        self.crop_range = person.crop_range
        self.crop_background = person.crop_background
        self.crop_times = person.crop_times                     # None for segmentations of version 1.2.0 beta
        h, w = self.mask.shape
        self.reference_column = person.crop_origin[1] + w / 2.0
        # one pixel around the person is uncertain: the edge filter of the segmentation removed mixed
        # pixels there, so the sensor did not clearly see through it
        from bodyscan.dynamic.segment import _dilate
        self.uncertain = _dilate(self.mask > 0, 1)
        silhouette = distance_map(self.uncertain, geometry.vertical_step, geometry.column_step)
        self.distance = torch.as_tensor(silhouette, dtype=dtype, device=device)
        self.center = person.center


@dataclass
class FrameMatch:
    """Correspondences of one frame for one optimisation round."""
    point_index: object
    face_ids: object
    bary: object
    silhouette: object            # vertex indices in free space
    silhouette_pixels: tuple      # their pixels (rows, columns) for the projection
    front: object                 # vertex indices in front of the measured surface
    front_pixels: tuple
    front_range: object           # measured ranges at those vertices
    visible: np.ndarray           # boolean per vertex
    visible_count: int
    silhouette_offsets: np.ndarray = None     # time of their pixel minus the frame time [s] (float64)
    front_offsets: np.ndarray = None
    distance: np.ndarray = None               # every point's distance to the visible body [m]


# ----------------------------------------------------------------------------
# Tracker
# ----------------------------------------------------------------------------

class Tracker:
    def __init__(self, avatar, model, segmented, config: TrackPipelineConfig | None = None):
        import torch
        from bodyscan.dynamic.motions import foot_vertices
        self.config = config or TrackPipelineConfig()
        c = self.config.tracking
        self.model = model
        self.avatar = avatar
        self.segmented = segmented
        self.geometry = segmented.geometry
        self.device, self.dtype = model.device, model.dtype
        self.shaped = avatar.shaped(model, c.level)
        self.faces_np = self.shaped.faces_np
        self.faces = self.shaped.faces
        self.tables = self.geometry.torch_tables(self.device, self.dtype)
        self.sensor = segmented.sensor_position
        self.feet = torch.as_tensor(foot_vertices(self.shaped), device=self.device)
        self.feet_np = foot_vertices(self.shaped)
        self.hands = model.rotations(count=1)[0, skeleton.LEFT_HAND.start:]          # relaxed hands, fixed
        labels = skeleton.part_labels(self.shaped.weights.detach().cpu().numpy())
        self.part_labels = labels
        self.part_count = np.bincount(labels, minlength=len(skeleton.PART_NAMES))
        lower, upper = skeleton.limit_arrays()
        self.start_body = np.clip(avatar.scan_body, lower[skeleton.BODY], upper[skeleton.BODY])
        segmentation = segmented.summary.get("config", {})
        self.foreground_margin = float(segmentation.get("bg_threshold", 0.08)) + 0.02
        self.background_range = segmented.background_range                # (H, W) empty room [m], NaN: unknown

    # -- posing ---------------------------------------------------------------------------------
    def rotations(self, body):
        """(F, 55, 3, 3) from body pose (F, 21, 3) (torch), hands relaxed."""
        import torch
        from bodyscan.body.rotations import axis_angle_to_matrix
        f = body.shape[0]
        head = torch.zeros(f, 1, 3, dtype=self.dtype, device=self.device)
        tail = torch.zeros(f, 3, 3, dtype=self.dtype, device=self.device)
        main = axis_angle_to_matrix(torch.cat([head, body, tail], dim=1))
        return torch.cat([main, self.hands.expand(f, -1, -1, -1)], dim=1)

    def pose(self, omega, reference, position, body, subset=None):
        """World vertices and joints of F frames: root rotation exp(omega) . reference."""
        from bodyscan.body.rotations import axis_angle_to_matrix
        root = axis_angle_to_matrix(omega) @ reference
        posed = self.shaped.pose(self.rotations(body), root, position, subset=subset)
        return posed.vertices, posed.joints

    # -- correspondences ------------------------------------------------------------------------
    def match(self, vertices: np.ndarray, normals: np.ndarray, frame: FrameData,
              max_distance: float | None = None) -> FrameMatch:
        import torch
        from bodyscan.dynamic.fitting import closest_points, visible_faces, visible_vertices
        c = self.config.tracking
        g = self.geometry
        visible = visible_vertices(vertices, normals, g, self.sensor, c.visibility_tolerance)
        faces = visible_faces(self.faces_np, visible)
        face_ids, bary, distance = closest_points(vertices, self.faces_np, frame.points_np, faces)
        valid = np.flatnonzero(distance < (c.max_correspondence if max_distance is None else max_distance))
        index = np.flatnonzero(visible)
        rows, cols, ranges = g.project(vertices[index])
        unwrapped = g.column_distance_unwrapped(cols, frame.reference_column)
        r0, c0 = frame.crop_origin
        h, w = frame.mask.shape
        rr = np.round(rows - r0).astype(int)
        cc = np.round(unwrapped - c0).astype(int)
        in_view = (rows > -0.5) & (rows < g.height - 0.5)
        in_crop = in_view & (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rr_c, cc_c = np.clip(rr, 0, h - 1), np.clip(cc, 0, w - 1)
        pixel_rows = np.clip(np.round(rows).astype(int), 0, g.height - 1)
        pixel_cols = np.mod(np.round(cols).astype(int), g.width)
        measured = np.where(in_crop, frame.crop_range[rr_c, cc_c], np.nan)
        # outside the crop only the empty room is known: no person was found there
        background = np.where(in_crop, frame.crop_background[rr_c, cc_c],
                              self.background_range[pixel_rows, pixel_cols])
        person = in_crop & frame.uncertain[rr_c, cc_c]
        margin = max(c.free_space_margin, self.foreground_margin)
        with np.errstate(invalid="ignore"):
            # free space: the sensor saw (or, outside the crop, would have seen) beyond the vertex, and the
            # vertex would have been segmented as the person had it been there (closer than the empty
            # scene by the segmentation margin); a vertex behind furniture is not free
            nearest = np.fmin(measured, background)
            beyond = np.isfinite(nearest) & (ranges < nearest - margin)
            free = in_view & ~person & beyond
            front = person & np.isfinite(measured) & (ranges < measured - margin)
        offsets = np.zeros(len(index))
        if frame.crop_times is not None:
            offsets = np.where(in_crop, frame.crop_times[rr_c, cc_c] - frame.time, 0.0)

        def tensor(values, dtype=None):
            return torch.as_tensor(values, device=self.device, dtype=dtype)

        return FrameMatch(tensor(valid), tensor(face_ids[valid]), tensor(bary[valid], self.dtype), tensor(index[free]),
                          (tensor(pixel_rows[free]), tensor(pixel_cols[free])), tensor(index[front]),
                          (tensor(pixel_rows[front]), tensor(pixel_cols[front])), tensor(measured[front], self.dtype),
                          visible, max(len(index), 1), offsets[free], offsets[front], distance)

    # -- energy of one frame --------------------------------------------------------------------
    def frame_energy(self, vertices, frame: FrameData, match: FrameMatch, shifted_points=None,
                     shifted_silhouette=None, shifted_front=None, scale: float | None = None):
        """Data, silhouette, front and floor terms of one frame (vertices (V, 3) torch).
        shifted_points: optional (M, 3) model points already moved to the point times;
        shifted_silhouette, shifted_front: the vertices of those terms moved to the times of their pixels;
        scale: robust scale of the data term (default robust_scale; the normalisation stays that of
        robust_scale, so a larger scale widens the reach without changing the weight of small residuals)."""
        import torch
        from bodyscan.dynamic.fitting import face_normals, geman_mcclure, sample_map, surface_points
        c = self.config.tracking
        s2 = c.robust_scale ** 2
        data_scale = c.robust_scale if scale is None else scale
        energy = vertices.new_zeros(())
        if len(match.point_index):
            points = frame.points[match.point_index]
            model_points = (surface_points(vertices, self.faces, match.face_ids, match.bary)
                            if shifted_points is None else shifted_points)
            normals = face_normals(vertices, self.faces, match.face_ids)
            diff = points - model_points
            plane = (diff * normals).sum(-1)
            energy = energy + (geman_mcclure(plane, data_scale).mean()
                               + c.point_to_point * geman_mcclure(diff.norm(dim=-1), data_scale).mean()) / s2
        g = self.geometry
        r0, c0 = frame.crop_origin
        h, w = frame.mask.shape
        if len(match.silhouette):
            moved = vertices[match.silhouette] if shifted_silhouette is None else shifted_silhouette
            rows, cols, ranges = g.project_torch(moved, self.tables, frame.reference_column, match.silhouette_pixels)
            rr, cc = rows - r0, cols - c0
            inside = sample_map(frame.distance, rr, cc)
            out_r = torch.relu(-rr) + torch.relu(rr - (h - 1))
            out_c = torch.relu(-cc) + torch.relu(cc - (w - 1))
            angle = inside + torch.sqrt((out_r * g.vertical_step) ** 2 + (out_c * g.column_step) ** 2 + 1e-12)
            lateral = angle * ranges
            scale = c.silhouette_scale
            energy = energy + c.silhouette_weight * geman_mcclure(lateral, scale).sum() / scale ** 2 / match.visible_count
        if len(match.front):
            moved = vertices[match.front] if shifted_front is None else shifted_front
            _, _, ranges = g.project_torch(moved, self.tables, frame.reference_column, match.front_pixels)
            excess = torch.relu(match.front_range - ranges - c.free_space_margin)
            energy = energy + c.silhouette_weight * geman_mcclure(excess, c.robust_scale).sum() / s2 / match.visible_count
        feet_z = vertices[self.feet, 2]
        energy = energy + c.floor_weight * (torch.relu(-feet_z) ** 2).mean() / s2
        return energy

    # -- stage 1 ----------------------------------------------------------------------------------
    def fit_frame(self, frame: FrameData, reference, position, body, prediction, rounds=None, iterations=None,
                  free_body=True, coarse=True, joints=None):
        """Fit one frame from a start (reference rotation (3, 3), position (3,), body (21, 3) numpy).
        prediction: (reference, position, body) the continuity term pulls towards, or None.
        coarse: start the rounds at the coarse robust scale; joints: indices (into the 21 body
        joints) of the only joints that move (the root stays), for searches of one limb.
        Returns (rotation matrix, position, body, energy, match)."""
        import torch
        from bodyscan.body.model import vertex_normals
        from bodyscan.dynamic.fitting import limit_penalty
        c = self.config.tracking
        rounds = c.rounds if rounds is None else rounds
        iterations = c.iterations if iterations is None else iterations
        dev, dt = self.device, self.dtype
        reference_t = torch.as_tensor(reference, dtype=dt, device=dev).reshape(1, 3, 3)
        omega = torch.zeros(1, 3, dtype=dt, device=dev, requires_grad=True)
        pos = torch.as_tensor(position, dtype=dt, device=dev).reshape(1, 3).clone().requires_grad_(True)
        b = torch.as_tensor(body, dtype=dt, device=dev).reshape(1, 21, 3).clone().requires_grad_(free_body)
        if prediction is not None:
            from bodyscan.body.rotations import matrix_to_axis_angle_np
            pred_ref, pred_pos, pred_body = prediction
            pred_omega = torch.as_tensor(matrix_to_axis_angle_np(pred_ref @ reference.T), dtype=dt, device=dev)
            pred_pos = torch.as_tensor(pred_pos, dtype=dt, device=dev)
            pred_body = torch.as_tensor(pred_body, dtype=dt, device=dev).reshape(1, 21, 3)
        tensors = [omega, pos] + ([b] if free_body else [])
        if joints is not None:
            mask = torch.zeros(1, 21, 1, dtype=dt, device=dev)
            mask[0, list(joints)] = 1.0
            b.register_hook(lambda grad: grad * mask)
            tensors = [b]
        energy_value, match = None, None
        for number in range(rounds):
            # coarse to fine: the robust scale halves every round down to robust_scale
            scale = (c.robust_scale * max(c.coarse_scale / 2.0 ** number, 1.0) if coarse and number < rounds - 1
                     else c.robust_scale)
            with torch.no_grad():
                vertices, _ = self.pose(omega, reference_t, pos, b)
                normals = vertex_normals(vertices, self.faces)
            match = self.match(vertices[0].cpu().numpy().astype(np.float64),
                               normals[0].cpu().numpy().astype(np.float64), frame,
                               max(c.max_correspondence, 3.0 * scale))
            optimizer = torch.optim.LBFGS(tensors, lr=1.0, max_iter=iterations, history_size=10,
                                          line_search_fn="strong_wolfe")
            seen = self.joint_seen(match.visible)[1:]                  # the 21 body joints
            continuity = torch.as_tensor(1.0 + c.hidden_continuity * self.hidden_factor(seen), dtype=dt,
                                         device=dev).reshape(1, 21, 1)

            def total():
                vertices, _ = self.pose(omega, reference_t, pos, b)
                energy = self.frame_energy(vertices[0], frame, match, scale=scale)
                energy = energy + c.limit_weight * limit_penalty(b)
                energy = energy + c.twist_weight * (b[:, TWIST_JOINTS, 1] ** 2).sum()
                if prediction is not None:
                    energy = energy + c.pose_continuity * (((b - pred_body) ** 2) * continuity).sum()
                    energy = energy + c.root_continuity * (((omega[0] - pred_omega) ** 2).sum()
                                                           + ((pos[0] - pred_pos) ** 2).sum() / 0.05 ** 2)
                return energy

            def closure():
                optimizer.zero_grad()
                energy = total()
                energy.backward()
                return energy

            optimizer.step(closure)
            with torch.no_grad():
                energy_value = float(total().item())
        from bodyscan.body.rotations import axis_angle_to_matrix
        with torch.no_grad():
            rotation = (axis_angle_to_matrix(omega) @ reference_t)[0].cpu().numpy().astype(np.float64)
        return (rotation, pos.detach()[0].cpu().numpy().astype(np.float64),
                b.detach()[0].cpu().numpy().astype(np.float64), energy_value, match)

    def residual(self, rotation, position, body, frame: FrameData) -> float:
        """Median distance [m] from the frame's points to the visible body surface."""
        return float(np.median(self.explain(rotation, position, body, frame)["distance"]))

    def explain(self, rotation, position, body, frame: FrameData) -> dict:
        """How well a pose explains a frame: the distance of every point to the visible body, the
        fraction of points farther than recover_distance ('unexplained'), and a score (lower is
        better): the mean point distance, each capped at 0.15 m, plus 0.15 m times the fraction of
        the visible vertices that lie where the sensor saw through."""
        import torch
        from bodyscan.body.model import vertex_normals
        with torch.no_grad():
            vertices, _ = self.pose(torch.zeros(1, 3, dtype=self.dtype, device=self.device),
                                    torch.as_tensor(rotation, dtype=self.dtype, device=self.device)[None],
                                    torch.as_tensor(position, dtype=self.dtype, device=self.device)[None],
                                    torch.as_tensor(body, dtype=self.dtype, device=self.device)[None])
            normals = vertex_normals(vertices, self.faces)
        match = self.match(vertices[0].cpu().numpy().astype(np.float64), normals[0].cpu().numpy().astype(np.float64),
                           frame)
        distance = match.distance
        far = distance > self.config.tracking.recover_distance
        return {"distance": distance, "unexplained": float(np.mean(far)), "unexplained_count": int(far.sum()),
                "score": float(np.mean(np.minimum(distance, 0.15)) + 0.15 * len(match.silhouette) / match.visible_count),
                "visible": match.visible}

    def joint_seen(self, visible: np.ndarray) -> np.ndarray:
        """(22,) fraction of the body part of each main joint (pelvis to wrists) that the sensor sees."""
        seen = np.zeros(len(skeleton.MAIN_JOINTS))
        for k, joint in enumerate(skeleton.MAIN_JOINTS):
            members = self.part_labels == skeleton.PART_NAMES.index(skeleton.PART_OF_JOINT[joint])
            seen[k] = visible[members].mean() if members.any() else 1.0
        return seen

    @staticmethod
    def hidden_factor(seen: np.ndarray) -> np.ndarray:
        """0 for a body part seen at 30 % or more, rising linearly to 1 for a part not seen at all."""
        return np.clip(1.0 - np.asarray(seen) / 0.3, 0.0, 1.0)

    def _arm_seen(self, side: str, visible: np.ndarray) -> float:
        """Fraction of the vertices of one arm (upper arm, forearm, hand) that the sensor sees."""
        parts = [skeleton.PART_NAMES.index(f"{side}_{name}") for name in ("upper_arm", "forearm", "hand")]
        members = np.isin(self.part_labels, parts)
        return float(visible[members].mean()) if members.any() else 0.0

    def recover_limbs(self, frame: FrameData, result, prediction, previous=None):
        """Fit the frame again from other starts: the previous frame's pose (without the
        prediction's velocity, which overshoots when a limb stops or turns), both legs from every
        pose of LEG_STARTS (a landing: knees forward, not feet forward), then each arm from every
        pose of ARM_STARTS. Each start first moves only its limb, at the fine scale (the rest of the
        fit stays), and is kept when it lowers the score of explain() by 2 mm; an arm only when the
        sensor sees that arm in the result (at least 20 % of it): no point can justify moving an
        arm the sensor does not see. The pose kept is then fitted once more as a whole."""
        best = result
        state = self.explain(best[0], best[1], best[2], frame)
        improved = False

        def attempt(trial, side=None):
            nonlocal best, state, improved
            trial_state = self.explain(trial[0], trial[1], trial[2], frame)
            if trial_state["score"] < state["score"] - 0.002 and (
                    side is None or self._arm_seen(side, trial_state["visible"]) >= 0.2):
                best, state, improved = trial, trial_state, True

        if previous is not None:
            attempt(self.fit_frame(frame, previous[0], previous[1], previous[2], prediction, coarse=False))
        legs = [skeleton.JOINT[f"{side}_{name}"] - 1 for side in ("left", "right") for name in ("hip", "knee", "ankle")]
        for start in LEG_STARTS:
            body = best[2].copy()
            for index, value in _leg_rotations(*start).items():
                body[index] = value
            attempt(self.fit_frame(frame, best[0], best[1], body, prediction, rounds=2, iterations=10, coarse=False,
                                   joints=legs))
        for side in ("left", "right"):
            arm = [skeleton.JOINT[f"{side}_{name}"] - 1 for name in ("collar", "shoulder", "elbow", "wrist")]
            for start in ARM_STARTS:
                body = best[2].copy()
                for index, value in _arm_rotations(side, *start).items():
                    body[index] = value
                attempt(self.fit_frame(frame, best[0], best[1], body, prediction, rounds=2, iterations=10,
                                       coarse=False, joints=arm), side)
        if improved:
            best = self.fit_frame(frame, best[0], best[1], best[2], prediction, rounds=2, coarse=False)
        return best

    def initialise(self, frame: FrameData, body=None, yaws=None, position=None):
        """Several facing directions from the start pose; keeps the best fit."""
        from bodyscan.body.rotations import rotation_z_np
        c = self.config.tracking
        body = self.start_body if body is None else body
        if position is None:
            center = np.median(frame.points_np[:, :2], axis=0)
            away = center - self.sensor[:2]
            center = center + 0.12 * away / max(np.linalg.norm(away), 1e-6)
            position = np.array([center[0], center[1], self.avatar.standing_height])
        yaws = np.linspace(0.0, 2 * np.pi, max(c.init_yaws, 1), endpoint=False) if yaws is None else yaws
        best = None
        for yaw in yaws:
            result = self.fit_frame(frame, rotation_z_np(yaw), position, body, None, rounds=2, iterations=10,
                                    free_body=False)
            if best is None or result[3] < best[3]:
                best = result
        rotation, position, body, _, _ = best
        return self.fit_frame(frame, rotation, position, body, None)

    def track(self, start: int = 0, stop: int | None = None) -> dict:
        """Stage 1 over the segmented frames; returns arrays per frame."""
        from bodyscan.body.rotations import matrix_to_axis_angle_np
        c = self.config.tracking
        count = len(self.segmented)
        stop = count if stop is None else min(stop, count)
        rotations, positions, bodies, times, indices, energies, residuals = [], [], [], [], [], [], []
        slots, restarts = [], []
        searched = 0
        prediction = None
        progress = Progress("tracked frames", stop - start)
        for k in range(start, stop):
            person = self.segmented.load(k)
            frame = FrameData(person, self.geometry, self.device, self.dtype)
            restarted = False
            if not rotations:
                result = self.initialise(frame)
                restarted = True
                explained = self.explain(result[0], result[1], result[2], frame)
            else:
                # after a restart the last two poses do not give a velocity
                last = slice(-1, None) if restarts[-1] else slice(None)
                prediction = self.predict(rotations[last], positions[last], bodies[last], times[last], frame.time)
                result = self.fit_frame(frame, prediction[0], prediction[1], prediction[2], prediction)
                explained = self.explain(result[0], result[1], result[2], frame)
                residual = float(np.median(explained["distance"]))
                if residual > c.lost_residual:
                    yaw = np.arctan2(rotations[-1][1, 0], rotations[-1][0, 0])
                    retry = self.initialise(frame, bodies[-1], yaw + np.radians([0, -45, 45, -90, 90, 180]),
                                            positions[-1])
                    retried = self.explain(retry[0], retry[1], retry[2], frame)
                    # compared on the points: the retry's energy has no continuity terms
                    if np.median(retried["distance"]) < residual:
                        result, restarted, explained = retry, True, retried
            if 0 < c.recover_fraction < explained["unexplained"] and explained["unexplained_count"] >= 10:
                result = self.recover_limbs(frame, result, None if restarted else prediction,
                                           None if restarted else (rotations[-1], positions[-1], bodies[-1]))
                explained = self.explain(result[0], result[1], result[2], frame)
                searched += 1
            rotation, position, body, energy, _ = result
            rotations.append(rotation)
            positions.append(position)
            bodies.append(body)
            times.append(frame.time)
            indices.append(person.index)
            slots.append(k)
            restarts.append(restarted)
            energies.append(energy)
            residuals.append(float(np.median(explained["distance"])))
            progress.maybe(k - start + 1, 25)
        root = np.array([matrix_to_axis_angle_np(r) for r in rotations])
        if searched:
            info(f"  limbs searched again in {searched} frames (points left unexplained by the fit)")
        return {"times": np.array(times), "frame_index": np.array(indices), "slot": np.array(slots, dtype=np.int64),
                "restart": np.array(restarts, dtype=bool), "searches": searched, "root_rotation": root,
                "root_matrix": np.array(rotations), "root_position": np.array(positions), "body": np.array(bodies),
                "energy": np.array(energies), "residual": np.array(residuals)}

    @staticmethod
    def predict(rotations, positions, bodies, times, t):
        """Constant-velocity prediction of the next frame (rotation, position, body)."""
        from bodyscan.body.rotations import axis_angle_to_matrix_np, matrix_to_axis_angle_np
        if len(rotations) < 2 or times[-1] <= times[-2]:
            return rotations[-1], positions[-1], bodies[-1]
        ratio = np.clip((t - times[-1]) / (times[-1] - times[-2]), 0.0, 3.0)
        delta = matrix_to_axis_angle_np(rotations[-1] @ rotations[-2].T)
        rotation = axis_angle_to_matrix_np(delta * ratio) @ rotations[-1]
        position = positions[-1] + (positions[-1] - positions[-2]) * ratio
        body = bodies[-1] + 0.5 * (bodies[-1] - bodies[-2]) * ratio
        return rotation, position, body

    # -- stage 2 ----------------------------------------------------------------------------------
    def contacts(self, result: dict) -> np.ndarray:
        """(F, 2) feet on the floor: lowest foot vertex low and ankle slow."""
        import torch
        r = self.config.refine
        frames = len(result["times"])
        if frames == 0:
            return np.zeros((0, 2), dtype=bool)
        with torch.no_grad():
            zero = torch.zeros(frames, 3, dtype=self.dtype, device=self.device)
            vertices, joints = self.pose(zero, torch.as_tensor(result["root_matrix"], dtype=self.dtype,
                                                               device=self.device),
                                         torch.as_tensor(result["root_position"], dtype=self.dtype, device=self.device),
                                         torch.as_tensor(result["body"], dtype=self.dtype, device=self.device),
                                         subset=self.feet)
        vertices = vertices.cpu().numpy()
        joints = joints.cpu().numpy()
        labels = self.part_labels[self.feet_np]
        times = result["times"]
        contact = np.zeros((frames, 2), dtype=bool)
        for side, name in enumerate(("left", "right")):
            mine = labels == skeleton.PART_NAMES.index(f"{name}_foot")
            low = vertices[:, mine, 2].min(axis=1)
            ankle = joints[:, skeleton.JOINT[f"{name}_ankle"], :2]
            speed = np.zeros(frames)
            if frames > 1:
                step = np.linalg.norm(np.diff(ankle, axis=0), axis=1) / np.maximum(np.diff(times), 1e-6)
                speed[1:] = step
                speed[:-1] = np.minimum(speed[:-1], step) if frames > 2 else step
                speed[0] = step[0]
            contact[:, side] = (low < r.contact_height) & (speed < r.contact_speed)
        # remove single-frame flickers
        for side in range(2):
            flags = contact[:, side].copy()
            for k in range(1, frames - 1):
                if flags[k - 1] == flags[k + 1] != flags[k]:
                    contact[k, side] = flags[k - 1]
        return contact

    def refine(self, result: dict, contacts: np.ndarray) -> dict:
        """Stage 2: overlapping windows over the whole sequence."""
        import torch
        from bodyscan.body.model import vertex_normals
        from bodyscan.body.rotations import axis_angle_to_matrix, matrix_to_axis_angle_np
        from bodyscan.dynamic.fitting import charbonnier, geman_mcclure, limit_penalty, surface_points
        r = self.config.refine
        c = self.config.tracking
        dev, dt = self.device, self.dtype
        frames = len(result["times"])
        times = result["times"]
        reference = result["root_matrix"].copy()
        position = result["root_position"].copy()
        body = result["body"].copy()
        data = {}
        main = list(skeleton.MAIN_JOINTS)
        feet_joints = {0: [skeleton.JOINT["left_ankle"], skeleton.JOINT["left_foot"]],
                       1: [skeleton.JOINT["right_ankle"], skeleton.JOINT["right_foot"]]}
        step = max(r.window - r.overlap, 1)
        starts = list(range(0, max(frames - r.overlap, 1), step))
        progress = Progress("refined windows", len(starts))
        for number, first in enumerate(starts):
            last = min(first + r.window, frames)
            held = r.overlap if first > 0 else 0
            span = np.arange(first, last)
            count = len(span)
            if count < 3:
                continue
            for k in span:
                if k not in data:
                    data[k] = FrameData(self.segmented.load(int(result["slot"][k])), self.geometry, dev, dt)
            ref_t = torch.as_tensor(reference[span], dtype=dt, device=dev)
            omega = torch.zeros(count, 3, dtype=dt, device=dev)
            pos = torch.as_tensor(position[span], dtype=dt, device=dev)
            b = torch.as_tensor(body[span], dtype=dt, device=dev)
            omega_f = omega[held:].clone().requires_grad_(True)
            pos_f = pos[held:].clone().requires_grad_(True)
            b_f = b[held:].clone().requires_grad_(True)
            # relative times: sensor clocks run to 1e9 s, beyond the resolution of float32 (GPU)
            t_span = torch.as_tensor(times[span] - times[span[0]], dtype=dt, device=dev)
            window_contacts = contacts[span]

            def assemble():
                return (torch.cat([omega[:held], omega_f]), torch.cat([pos[:held], pos_f]), torch.cat([b[:held], b_f]))

            for number in range(r.rounds):
                # coarse to fine as in stage 1, starting from half the coarse scale (stage 1 is close)
                scale = (c.robust_scale * max(c.coarse_scale / 2.0 ** (number + 1), 1.0) if number < r.rounds - 1
                         else c.robust_scale)
                with torch.no_grad():
                    o, p, bb = assemble()
                    vertices, _ = self.pose(o, ref_t, p, bb)
                    normals = vertex_normals(vertices, self.faces)
                matches = [self.match(vertices[i].cpu().numpy().astype(np.float64),
                                      normals[i].cpu().numpy().astype(np.float64), data[k],
                                      max(c.max_correspondence, 3.0 * scale))
                           for i, k in enumerate(span)]
                seen = np.stack([self.joint_seen(m.visible) for m in matches])          # (count, 22)
                hidden_weight = 1.0 + r.hidden_smoothness * self.hidden_factor(seen)
                shutter = []
                for i, k in enumerate(span):
                    frame, match = data[k], matches[i]
                    before, after = max(i - 1, 0), min(i + 1, count - 1)
                    span_t = float(times[span[after]] - times[span[before]])
                    if not r.rolling_shutter or span_t <= 0:
                        shutter.append(None)
                        continue

                    def factor(offsets):              # time offsets [s] (float64) as fractions of span_t
                        return torch.as_tensor(np.asarray(offsets, dtype=np.float64) / span_t, dtype=dt, device=dev)

                    point_offsets = frame.point_times[match.point_index.cpu().numpy()] - times[k]
                    shutter.append({"before": before, "after": after, "points": factor(point_offsets),
                                    "silhouette": factor(match.silhouette_offsets),
                                    "front": factor(match.front_offsets)})
                tensors = [omega_f, pos_f, b_f]
                optimizer = torch.optim.LBFGS(tensors, lr=1.0, max_iter=r.iterations, history_size=10,
                                              line_search_fn="strong_wolfe")

                def total():
                    o, p, bb = assemble()
                    vertices, joints = self.pose(o, ref_t, p, bb)
                    energy = vertices.new_zeros(())
                    for i, k in enumerate(span):
                        match = matches[i]
                        moved = {}
                        if shutter[i] is not None:
                            # the body at the time of each point and pixel: moved along its motion
                            # between the neighbouring frames
                            s = shutter[i]
                            step = vertices[s["after"]] - vertices[s["before"]]
                            if len(match.point_index):
                                moved["shifted_points"] = (
                                    surface_points(vertices[i], self.faces, match.face_ids, match.bary)
                                    + s["points"][:, None] * surface_points(step, self.faces, match.face_ids,
                                                                            match.bary))
                            if len(match.silhouette):
                                moved["shifted_silhouette"] = (vertices[i][match.silhouette]
                                                               + s["silhouette"][:, None] * step[match.silhouette])
                            if len(match.front):
                                moved["shifted_front"] = (vertices[i][match.front]
                                                          + s["front"][:, None] * step[match.front])
                        energy = energy + self.frame_energy(vertices[i], data[k], match, scale=scale, **moved)
                    energy = energy / count + c.limit_weight * limit_penalty(bb) / count
                    energy = energy + c.twist_weight * (bb[:, TWIST_JOINTS, 1] ** 2).sum() / count
                    # smoothness of the joint motion (non-uniform frame times): accelerations at the inner
                    # frames, jerk between them; normalised by the scale (about |x| / scale when large)
                    j = joints[:, main]
                    dt1 = (t_span[1:] - t_span[:-1]).clamp_min(1e-4)
                    velocity = (j[1:] - j[:-1]) / dt1[:, None, None]
                    accel = 2.0 * (velocity[1:] - velocity[:-1]) / (t_span[2:] - t_span[:-2]).clamp_min(1e-4)[:, None, None]
                    if r.smoothness == "jerk" and count > 3:
                        rough = (accel[1:] - accel[:-1]) / dt1[1:-1, None, None]
                        weight = 0.5 * (hidden_weight[1:-2] + hidden_weight[2:-1])
                    else:
                        rough = accel
                        weight = hidden_weight[1:-1]
                    weight = torch.as_tensor(weight, dtype=dt, device=dev)
                    energy = energy + r.smoothness_weight * (weight * charbonnier(rough.norm(dim=-1),
                                                                                  r.smoothness_scale)).mean() \
                        / r.smoothness_scale
                    # feet that stand on the floor do not slide
                    for side, ids in feet_joints.items():
                        both = torch.as_tensor(window_contacts[1:, side] & window_contacts[:-1, side], device=dev)
                        if bool(both.any()):
                            slide = (joints[1:, ids, :2] - joints[:-1, ids, :2]).norm(dim=-1) / dt1[:, None]
                            energy = energy + r.skating_weight * geman_mcclure(slide[both], 0.1).mean() / 0.01
                    return energy

                def closure():
                    optimizer.zero_grad()
                    energy = total()
                    energy.backward()
                    return energy

                optimizer.step(closure)
            with torch.no_grad():
                o, p, bb = assemble()
                final = (axis_angle_to_matrix(o) @ ref_t).cpu().numpy().astype(np.float64)
            reference[span] = final
            position[span] = p.detach().cpu().numpy()
            body[span] = bb.detach().cpu().numpy()
            for k in list(data):
                if k < first + step:
                    del data[k]
            progress.maybe(number + 1, 1)
        refined = dict(result)
        refined["root_matrix"] = reference
        refined["root_rotation"] = np.array([matrix_to_axis_angle_np(m) for m in reference])
        refined["root_position"] = position
        refined["body"] = body
        return refined

    # -- quality ----------------------------------------------------------------------------------
    def assess(self, result: dict) -> dict:
        """Per frame: point residual, silhouette violations, observed fraction per body part,
        floor penetration, joint-limit excess; and the joints."""
        import torch
        from bodyscan.body.model import vertex_normals
        from bodyscan.dynamic.fitting import closest_points, visible_faces
        frames = len(result["times"])
        parts = len(skeleton.PART_NAMES)
        observed = np.zeros((frames, parts))
        residual_median = np.zeros(frames)
        residual_p90 = np.zeros(frames)
        unexplained = np.zeros(frames)
        violation = np.zeros(frames)
        floor = np.zeros(frames)
        joints_out = np.zeros((frames, skeleton.NUM_JOINTS, 3), dtype=np.float32)
        lower, upper = skeleton.limit_arrays()
        excess = np.degrees(np.maximum(lower[skeleton.BODY] - result["body"], 0)
                            + np.maximum(result["body"] - upper[skeleton.BODY], 0)).max(axis=(1, 2))
        batch = 32
        for first in range(0, frames, batch):
            span = np.arange(first, min(first + batch, frames))
            with torch.no_grad():
                zero = torch.zeros(len(span), 3, dtype=self.dtype, device=self.device)
                vertices, joints = self.pose(zero, torch.as_tensor(result["root_matrix"][span], dtype=self.dtype,
                                                                   device=self.device),
                                             torch.as_tensor(result["root_position"][span], dtype=self.dtype,
                                                             device=self.device),
                                             torch.as_tensor(result["body"][span], dtype=self.dtype, device=self.device))
                normals = vertex_normals(vertices, self.faces)
            joints_out[span] = joints.cpu().numpy()
            for i, k in enumerate(span):
                frame = FrameData(self.segmented.load(int(result["slot"][k])), self.geometry, self.device, self.dtype)
                v = vertices[i].cpu().numpy().astype(np.float64)
                match = self.match(v, normals[i].cpu().numpy().astype(np.float64), frame)
                _, _, distance = closest_points(v, self.faces_np, frame.points_np,
                                                visible_faces(self.faces_np, match.visible))
                residual_median[k] = np.median(distance)
                residual_p90[k] = np.percentile(distance, 90)
                unexplained[k] = np.mean(distance > 0.05)
                violation[k] = len(match.silhouette) / match.visible_count
                floor[k] = v[self.feet_np, 2].min()
                # observed: visible vertices with a point of the person within 2 robust scales
                visible = np.flatnonzero(match.visible)
                if len(visible):
                    from bodyscan.dynamic.avatar import _nearest
                    _, near = _nearest(frame.points_np, v[visible])
                    seen = visible[near < 2 * self.config.tracking.robust_scale]
                    observed[k] = np.bincount(self.part_labels[seen], minlength=parts) / np.maximum(self.part_count, 1)
        observed[:, self.part_count == 0] = np.nan                     # parts without vertices in this body
        return {"joints": joints_out, "residual_median": residual_median, "residual_p90": residual_p90,
                "unexplained": unexplained, "silhouette_violation": violation, "floor": floor,
                "limit_excess_deg": excess, "observed": observed}


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

def flag_frames(quality: dict, contacts: np.ndarray, times: np.ndarray) -> dict:
    """Frames that need a look: points far from the body (most of them, or more
    than 5 % of them: a limb in the wrong place), body in free space, feet
    below the floor or sliding while standing, joint limits exceeded, sudden
    accelerations."""
    joints = quality["joints"][:, list(skeleton.MAIN_JOINTS)].astype(np.float64)
    frames = len(times)
    accel = np.zeros(frames)
    if frames > 2:
        dt1 = np.maximum(np.diff(times), 1e-4)
        velocity = np.diff(joints, axis=0) / dt1[:, None, None]
        a = 2 * np.diff(velocity, axis=0) / np.maximum(times[2:] - times[:-2], 1e-4)[:, None, None]
        accel[1:-1] = np.linalg.norm(a, axis=2).max(axis=1)
    sliding = np.zeros(frames)
    for side, name in enumerate(("left", "right")):
        ankle = quality["joints"][:, skeleton.JOINT[f"{name}_ankle"], :2]
        if frames > 1:
            speed = np.linalg.norm(np.diff(ankle, axis=0), axis=1) / np.maximum(np.diff(times), 1e-4)
            both = contacts[1:, side] & contacts[:-1, side]
            sliding[1:] = np.maximum(sliding[1:], np.where(both, speed, 0.0))
    checks = {
        "residual": quality["residual_median"] > 0.03,
        "unexplained": quality["unexplained"] > 0.05,
        "free_space": quality["silhouette_violation"] > 0.05,
        "below_floor": quality["floor"] < -0.02,
        "sliding": sliding > 0.3,
        "joint_limits": quality["limit_excess_deg"] > 10.0,
        "acceleration": accel > 120.0,
    }
    flags = np.zeros(frames, dtype=np.int64)
    for bit, (name, mask) in enumerate(checks.items()):
        flags |= mask.astype(np.int64) << bit
    return {"flags": flags, "flag_names": list(checks), "acceleration_max": accel, "sliding": sliding,
            "counts": {name: int(mask.sum()) for name, mask in checks.items()}}


def save_motion(path, result: dict, quality: dict, contacts: np.ndarray, flags: dict, avatar_path, avatar,
                config: TrackPipelineConfig, extra: dict | None = None) -> dict:
    observed = quality["observed"]
    summary = {
        "frames": int(len(result["times"])),
        "duration_s": float(result["times"][-1] - result["times"][0]) if len(result["times"]) > 1 else 0.0,
        "residual_median_mm": float(1000 * np.median(quality["residual_median"])),
        "residual_p90_mm": float(1000 * np.median(quality["residual_p90"])),
        "observed_fraction_per_part": {name: float(np.nanmean(observed[:, k])) for k, name in
                                       enumerate(skeleton.PART_NAMES) if np.isfinite(observed[:, k]).any()},
        "flagged_frames": flags["counts"],
        "contact_fraction": {"left": float(contacts[:, 0].mean()), "right": float(contacts[:, 1].mean())},
        **(extra or {}),
    }
    np.savez_compressed(path, times=result["times"], root_rotation=result["root_rotation"],
                        root_position=result["root_position"], body=result["body"],
                        frame_index=result["frame_index"], joints=quality["joints"], contacts=contacts,
                        observed=observed.astype(np.float32), residual_median=quality["residual_median"],
                        residual_p90=quality["residual_p90"], unexplained=quality["unexplained"],
                        silhouette_violation=quality["silhouette_violation"],
                        floor=quality["floor"], limit_excess_deg=quality["limit_excess_deg"], flags=flags["flags"],
                        flag_names=np.array(flags["flag_names"]), avatar=str(avatar_path), model=avatar.model_path,
                        model_sha1=avatar.model_sha1, level=np.int64(config.tracking.level),
                        config=json.dumps({"tracking": asdict(config.tracking), "refine": asdict(config.refine)}),
                        summary=json.dumps(summary))
    Path(str(path).replace(".npz", "") + ".json").write_text(json.dumps(summary, indent=2))
    return summary


def summary_text(summary: dict) -> str:
    lines = [f"tracked {summary['frames']} frames over {summary['duration_s']:.1f} s; points to body: median "
             f"{summary['residual_median_mm']:.1f} mm, p90 {summary['residual_p90_mm']:.1f} mm (median of frames)"]
    observed = summary["observed_fraction_per_part"]
    weakest = sorted(observed.items(), key=lambda item: item[1])[:4]
    lines.append("  least observed parts (fraction of the part seen, mean over frames): "
                 + ", ".join(f"{name} {value:.0%}" for name, value in weakest))
    flagged = {k: v for k, v in summary["flagged_frames"].items() if v}
    lines.append("  frames to review: " + (", ".join(f"{k} {v}" for k, v in flagged.items()) if flagged else "none"))
    return "\n".join(lines)


def run_tracking(segment_folder, avatar_path, out, config: TrackPipelineConfig | None = None, model_path=None,
                 start: int = 0, stop: int | None = None) -> dict:
    """Stage 1, contacts, stage 2, quality; writes <out>.npz and <out>.json."""
    import torch
    from bodyscan.dynamic.avatar import Avatar
    from bodyscan.dynamic.segment import SegmentedRecording
    config = config or TrackPipelineConfig()
    device = config.tracking.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device.startswith("cuda") else torch.float64
    avatar = Avatar.load(avatar_path)
    model = avatar.model(model_path, device, dtype)
    segmented = SegmentedRecording(segment_folder)
    tracker = Tracker(avatar, model, segmented, config)
    info(f"tracking {len(segmented)} frames on {device}")
    result = tracker.track(start, stop)
    contacts = tracker.contacts(result)
    if config.refine.enabled and len(result["times"]) >= 3:
        info("refining the whole sequence")
        result = tracker.refine(result, contacts)
        contacts = tracker.contacts(result)
    quality = tracker.assess(result)
    flags = flag_frames(quality, contacts, result["times"])
    out = Path(out)
    if out.suffix != ".npz":
        out = out.with_suffix(".npz")
    summary = save_motion(out, result, quality, contacts, flags, avatar_path, avatar, config,
                          {"limb_searches": int(result.get("searches", 0)),
                           "restarts": int(np.sum(result.get("restart", [])))})
    info(summary_text(summary))
    if any(flags["counts"].values()):
        warning("some frames are flagged: look at them with 'bodyscan review-motion'")
    return summary
