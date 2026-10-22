"""What the viewer shows, independent of the user interface (testable without a browser).

A catalogue lists the tracked takes found in a results folder (one subfolder
per take with motion.npz, as written by the cluster runs), and pairs each with
its segment-motion folder and, if present, its evaluate-take folder. Takes are
grouped by motion type: the take name without its repetition number
(s02_walk03_2_radial and s02_walk03_radial are both 's02_walk03_radial').

TakeData gives, frame by frame: the posed body (posed on demand and cached),
the person's points and their distance to the body, the joints, the limbs the
tracker guessed (it did not see them), the tracker flags, and the per-vertex
values of the evaluation (this frame's error, or the take's median, p95,
violation rate, coverage).
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VERTEX_MODES = ("plain", "guessed limbs", "error this frame", "median error", "p95 error", "violation rate",
                "coverage")
# scale of each vertex mode: (key in the evaluation arrays, vmin, vmax, unit text)
MODE_SCALES = {"error this frame": (None, 0.0, 0.05, "m"), "median error": ("median", 0.0, 0.05, "m"),
               "p95 error": ("p95", 0.0, 0.08, "m"), "violation rate": ("violation_rate", 0.0, 0.5, ""),
               "coverage": ("coverage", 0.0, 1.0, "")}


def motion_type(take_name: str) -> str:
    """The take name without its repetition number."""
    return "_".join(token for token in take_name.split("_") if not re.fullmatch(r"[2-9]", token))


@dataclass
class TakeEntry:
    name: str
    motion: Path
    segments: Path | None
    evaluation: Path | None

    @property
    def kind(self) -> str:
        return motion_type(self.name)


def catalogue(results_root, segments_root=None, evaluations_root=None) -> dict[str, list[TakeEntry]]:
    """{motion type: [takes]} of the tracked takes under results_root (subfolders with motion.npz, or
    *.npz motion files directly in it)."""
    results_root = Path(results_root)
    entries = []
    candidates = sorted(p for p in results_root.iterdir() if p.is_dir() and (p / "motion.npz").exists())
    for folder in candidates:
        name = folder.name
        segments = Path(segments_root) / f"{name}_seg" if segments_root else None
        evaluation = Path(evaluations_root) / name if evaluations_root else None
        entries.append(TakeEntry(name, folder / "motion.npz",
                                 segments if segments is not None and segments.exists() else None,
                                 evaluation if evaluation is not None and (evaluation / "metrics.npz").exists()
                                 else None))
    groups: dict[str, list[TakeEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.kind, []).append(entry)
    return groups


class TakeData:
    """One tracked take, frame by frame."""

    def __init__(self, motion_path, segment_folder=None, avatar_path=None, model_path=None, evaluation_folder=None,
                 level: int = 0, cache_frames: int = 64):
        import torch
        from bodyscan.body import skeleton
        from bodyscan.dynamic.avatar import Avatar
        from bodyscan.dynamic.motions import PoseSequence
        self.motion_path = Path(motion_path)
        self.data = np.load(self.motion_path, allow_pickle=False)
        self.motion = PoseSequence.load(self.motion_path)
        files = self.data.files
        self.frame_index = self.data["frame_index"] if "frame_index" in files else np.arange(len(self.motion))
        avatar_path = avatar_path or (str(self.data["avatar"]) if "avatar" in files else None)
        if not avatar_path:
            raise SystemExit("the avatar is needed to draw the body (--avatar)")
        avatar = Avatar.load(avatar_path)
        self.model = avatar.model(model_path, "cpu", torch.float32)
        self.shaped = avatar.shaped(self.model, level)
        self.faces = np.asarray(self.shaped.faces_np, dtype=np.int64)
        weights = self.shaped.weights.detach().cpu().numpy()
        self.labels = skeleton.part_labels(weights)
        self.part_names = skeleton.PART_NAMES
        self.joints_all = self.data["joints"] if "joints" in files else None
        self.flag_names = [str(n) for n in self.data["flag_names"]] if "flag_names" in files else []
        self.flag_bits = self.data["flags"] if "flags" in files else np.zeros(len(self.motion), dtype=np.int64)
        self._guess_masks(skeleton)
        self.segmented = None
        self.by_index = {}
        if segment_folder is not None and Path(segment_folder).exists():
            from bodyscan.dynamic.segment import SegmentedRecording
            self.segmented = SegmentedRecording(segment_folder)
            self.by_index = {int(p.stem.split("_")[-1]): k for k, p in enumerate(self.segmented.paths)}
        self.evaluation = None
        self.evaluation_summary = None
        if evaluation_folder is not None and (Path(evaluation_folder) / "metrics.npz").exists():
            self.evaluation = np.load(Path(evaluation_folder) / "metrics.npz", allow_pickle=False)
            summary = Path(evaluation_folder) / "metrics.json"
            self.evaluation_summary = json.loads(summary.read_text()) if summary.exists() else None
            slots = self.evaluation["frame_index"]
            self.evaluation_slot = {int(index): s for s, index in enumerate(slots)}
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_frames = cache_frames

    def _guess_masks(self, skeleton) -> None:
        self.guessed = self.data["guessed"] if "guessed" in self.data.files else None
        self.limb_masks = []
        if self.guessed is None or self.guessed.size == 0:
            return
        for limb in [str(n) for n in self.data["guessed_limbs"]]:
            side, kind = limb.split("_")
            parts = ("upper_arm", "forearm", "hand") if kind == "arm" else ("thigh", "shank", "foot")
            self.limb_masks.append(np.isin(self.labels, [skeleton.PART_NAMES.index(f"{side}_{p}") for p in parts]))

    def __len__(self) -> int:
        return len(self.motion)

    @property
    def sensor_position(self) -> np.ndarray | None:
        return None if self.segmented is None else self.segmented.sensor_position

    def time(self, k: int) -> float:
        return float(self.motion.times[k] - self.motion.times[0])

    def vertices(self, k: int) -> np.ndarray:
        """(V, 3) float32 body of tracked frame k (posed once, then cached)."""
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        posed, _ = self.motion.subset(slice(k, k + 1)).posed(self.shaped)
        vertices = posed[0].astype(np.float32)
        self._cache[k] = vertices
        if len(self._cache) > self._cache_frames:
            self._cache.popitem(last=False)
        return vertices

    def joints(self, k: int) -> np.ndarray | None:
        return None if self.joints_all is None else np.asarray(self.joints_all[k], dtype=np.float32)

    def points(self, k: int) -> np.ndarray:
        """(N, 3) the person's points of the segmented frame of tracked frame k (empty without one)."""
        if self.segmented is None:
            return np.zeros((0, 3))
        slot = self.by_index.get(int(self.frame_index[k]))
        return np.zeros((0, 3)) if slot is None else self.segmented.load(slot).points

    def point_distances(self, k: int, points: np.ndarray | None = None) -> np.ndarray:
        from bodyscan.dynamic.fitting import closest_points
        points = self.points(k) if points is None else points
        if len(points) == 0:
            return np.zeros(0)
        return closest_points(self.vertices(k), self.faces, points)[2]

    def guessed_mask(self, k: int) -> np.ndarray:
        mask = np.zeros(len(self.labels), dtype=bool)
        if self.guessed is not None and self.guessed.size:
            for limb, members in enumerate(self.limb_masks):
                if self.guessed[k, limb] > 0.5:
                    mask |= members
        return mask

    def flags(self, k: int) -> list[str]:
        bits = int(self.flag_bits[k]) if len(self.flag_bits) else 0
        return [name for b, name in enumerate(self.flag_names) if bits >> b & 1]

    def flagged_frames(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.flag_bits) != 0)

    def vertex_values(self, k: int, mode: str) -> np.ndarray | None:
        """(V,) values of a vertex mode at tracked frame k (NaN: no value), or None without an evaluation."""
        if self.evaluation is None or mode not in MODE_SCALES:
            return None
        key = MODE_SCALES[mode][0]
        if key is None:
            slot = self.evaluation_slot.get(int(self.frame_index[k]))
            if slot is None or "frame_error_mm" not in self.evaluation.files:
                return np.full(len(self.labels), np.nan)
            return self.evaluation["frame_error_mm"][slot].astype(np.float64) / 1000.0
        values = np.asarray(self.evaluation[key], dtype=np.float64)
        if key != "coverage":
            values = np.where(np.asarray(self.evaluation["coverage"]) >= 0.05, values, np.nan)
        return values

    def frame_text(self, k: int) -> str:
        """Markdown lines describing tracked frame k."""
        flags = self.flags(k)
        lines = [f"**Frame {int(self.frame_index[k])}**, t = {self.time(k):.2f} s ({k + 1} of {len(self)} tracked)",
                 f"Flags: {', '.join(flags) if flags else 'none'}"]
        if self.guessed is not None and self.guessed.size:
            guessed = [str(n).replace("_", " ") for n, g in zip(self.data["guessed_limbs"], self.guessed[k]) if g > 0.5]
            lines.append(f"Guessed (not seen): {', '.join(guessed) if guessed else 'nothing'}")
        if self.evaluation is not None:
            slot = self.evaluation_slot.get(int(self.frame_index[k]))
            if slot is not None:
                counts = self.evaluation["class_counts"][slot]
                ranged = max(int(counts[:3].sum()), 1)
                lines.append(f"Pixels: fitted {counts[0] / ranged:.0%}, violation {counts[1] / ranged:.0%}, "
                             f"unexplained {counts[2] / ranged:.0%}")
        return "\n\n".join(lines)

    def take_text(self) -> str:
        """Markdown summary of the take's evaluation (or a hint to run evaluate-take)."""
        s = self.evaluation_summary
        if not s:
            return "No evaluation for this take: run `bodyscan evaluate-take` to get errors and heatmaps."
        shares = s.get("shares_of_ranged_pixels", {})
        points = s.get("points", {})
        lines = [f"**Take**: {s.get('frames_evaluated', 0)} frames evaluated",
                 f"Pixels: fitted {shares.get('fitted', float('nan')):.1%}, violation "
                 f"{shares.get('violation', float('nan')):.1%}, unexplained {shares.get('unexplained', float('nan')):.1%}",
                 f"Points to body: median {points.get('distance_median_mm') or float('nan'):.1f} mm, outliers "
                 f"{points.get('outlier', float('nan')):.1%}",
                 f"Body surface observed: {s.get('body_coverage', float('nan')):.0%}"]
        parts = s.get("parts", {})
        worst = sorted(((row.get("p95_mm", -1.0), name) for name, row in parts.items() if "p95_mm" in row),
                       reverse=True)[:4]
        if worst:
            lines.append("Worst parts (p95): " + ", ".join(f"{name} {value:.0f} mm" for value, name in worst))
        return "\n\n".join(lines)

    def series(self) -> dict:
        """Per-frame curves for plotting: time, violation and unexplained shares (evaluation), and the
        median point-to-body distance of the tracker (motion.npz)."""
        out = {"time": self.motion.times - self.motion.times[0]}
        if "residual_median" in self.data.files:
            out["residual_median_mm"] = 1000.0 * np.asarray(self.data["residual_median"], dtype=np.float64)
        if self.evaluation is not None:
            counts = self.evaluation["class_counts"].astype(np.float64)
            ranged = np.maximum(counts[:, :3].sum(axis=1), 1.0)
            slots = [self.evaluation_slot.get(int(i)) for i in self.frame_index]
            violation = np.full(len(self), np.nan)
            unexplained = np.full(len(self), np.nan)
            for k, s in enumerate(slots):
                if s is not None:
                    violation[k] = counts[s, 1] / ranged[s]
                    unexplained[k] = counts[s, 2] / ranged[s]
            out["violation_share"] = violation
            out["unexplained_share"] = unexplained
        return out
