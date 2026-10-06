"""Diagnostic figures (matplotlib is optional: without it the figures are skipped)."""

from __future__ import annotations

import numpy as np


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def angle_plot(path, sample_times, sample_angles, times, used_angles, source, measured=None, profile_only=None,
               alternatives=None, usable=None) -> bool:
    """Top: the angles used (red), the chained angles of neighbouring samples
    (black dots; a few per cent short) and the stepper profile alone (gray,
    dashed). Bottom: everything minus the angles used: chained angles,
    stepper profile, and the other solutions. Curves that stay near zero
    agree with the angles used; a curve that drifts away by tens of degrees
    is a solution that was rejected. 'usable' (frames with the person)
    restricts the curves, so frames after the person stepped off do not
    stretch the time axis."""
    plt = _pyplot()
    if plt is None:
        return False
    times = np.asarray(times)
    keep = np.ones(len(times), dtype=bool) if usable is None else np.asarray(usable, dtype=bool)
    order = np.argsort(times[keep])
    t = times[keep][order]

    def curve(values):
        return np.asarray(values)[keep][order]

    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    used = curve(used_angles)
    used_at_samples = np.interp(sample_times, t, used)
    axes[0].plot(t, np.degrees(used), lw=1.0, color="tab:red", label=f"used ({source})")
    axes[0].plot(sample_times, np.degrees(sample_angles), ".", ms=3, color="black",
                 label="chained (neighbouring samples)")
    axes[1].plot(sample_times, np.degrees(sample_angles - used_at_samples), ".", ms=3, color="black")
    if profile_only is not None:
        profile = curve(profile_only)
        axes[0].plot(t, np.degrees(profile), "--", lw=0.8, color="tab:gray", label="stepper profile alone")
        axes[1].plot(t, np.degrees(profile - used), "--", lw=0.8, color="tab:gray")
    colors = iter(["tab:orange", "tab:green", "tab:purple"])
    for label, angles in (alternatives or {}).items():
        axes[1].plot(t, np.degrees(curve(angles) - used), lw=1.2, color=next(colors), label=label)
    if alternatives:
        axes[1].legend(fontsize=8)
    if measured is not None:
        view_times, view_angles = measured
        axes[0].plot(view_times, np.degrees(view_angles), ".", ms=3, color="tab:blue", label="measured per view")
        axes[1].plot(view_times, np.degrees(view_angles - np.interp(view_times, t, used)), ".", ms=3,
                     color="tab:blue")
    axes[0].set_ylabel("platform angle [deg]")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="tab:red", lw=0.8)
    axes[1].set_ylabel("minus the angles used [deg]")
    axes[1].set_xlabel("sensor time [s]")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return True


def top_view(path, points, colors) -> bool:
    """Horizontal slices at torso and knee height, coloured by view. A correct
    fusion is one closed ring per slice; separate rings mean misplaced views."""
    plt = _pyplot()
    if plt is None:
        return False
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axis, (low, high) in zip(axes, [(1.0, 1.15), (0.3, 0.45)]):
        keep = (points[:, 2] > low) & (points[:, 2] < high)
        axis.scatter(points[keep, 0], points[keep, 1], s=1, c=colors[keep] if len(colors) else None)
        axis.set_aspect("equal")
        axis.set_title(f"top view, height {low:.2f} to {high:.2f} m (colour = view)")
    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)
    return True


def motion_plot(path, times, scores, threshold, segments, used) -> bool:
    plt = _pyplot()
    if plt is None:
        return False
    figure, axis = plt.subplots(figsize=(11, 3.2))
    for n, (start, stop) in enumerate(segments):
        color = "tab:green" if n in used else "tab:gray"
        axis.axvspan(times[start], times[stop - 1], color=color, alpha=0.25, lw=0)
    axis.plot(times, scores, color="black", lw=0.8)
    axis.axhline(threshold, color="tab:red", lw=0.8, ls="--")
    axis.set_xlabel("time since capture start [s]")
    axis.set_ylabel("changed person pixels")
    axis.set_title("motion score; green = still segments used as keyframes, red = threshold")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return True
