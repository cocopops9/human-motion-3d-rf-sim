"""Geometry of the range image: from a point in the world to its pixel, and pixel times.

A spinning LiDAR image has one row per beam (fixed elevation) and one column
per azimuth step. RangeImageGeometry is built from the per-pixel lookup
table of the recording (xyz = range * direction + offset, sensor frame) and
the floor frame, and answers:

    project(world points)    fractional (row, column), the range along the
                             pixel's ray, numpy; the torch version is
                             differentiable with respect to the points
    pixel_times(timestamps)  the time of every pixel from the column timestamps

Columns may turn either way (an Ouster image runs clockwise, the synthetic
sensor counter-clockwise); the direction is read from the lookup table.
Beam origins off the sensor axis (the offset, about 1.5 to 3 cm on Ouster
sensors) are handled by one refinement step: project, look up the offset of
that pixel, project again from it.
"""

from __future__ import annotations

import numpy as np


class RangeImageGeometry:
    def __init__(self, direction: np.ndarray, offset: np.ndarray, world_from_sensor: np.ndarray,
                 pixel_shift: np.ndarray | None = None):
        self.direction = np.asarray(direction, dtype=np.float64)
        self.offset = np.asarray(offset, dtype=np.float64)
        self.height, self.width = self.direction.shape[:2]
        self.world_from_sensor = np.asarray(world_from_sensor, dtype=np.float64)
        self.rotation = self.world_from_sensor[:3, :3]
        self.translation = self.world_from_sensor[:3, 3]
        self.pixel_shift = (np.zeros(self.height, dtype=np.int64) if pixel_shift is None
                            else np.asarray(pixel_shift, dtype=np.int64))
        elevation = np.arcsin(np.clip(self.direction[..., 2], -1.0, 1.0))
        self.row_elevation = np.median(elevation, axis=1)                 # (H,), decreasing with the row
        if self.row_elevation[0] < self.row_elevation[-1]:
            raise ValueError("the rows of the range image must go from the top beam to the bottom beam")
        azimuth = np.unwrap(np.arctan2(self.direction[..., 1], self.direction[..., 0]), axis=1)
        steps = np.diff(azimuth, axis=1)
        self.sense = float(np.sign(np.median(steps)))                     # +1 counter-clockwise, -1 clockwise
        self.column_step = 2.0 * np.pi / self.width                       # radians per column
        columns = np.arange(self.width)
        # per row: azimuth of column 0, from a least-squares fit with the known slope
        self.row_azimuth0 = np.median(azimuth - self.sense * self.column_step * columns, axis=1)
        self.vertical_step = float(np.median(np.abs(np.diff(self.row_elevation))))

    # -- numpy --------------------------------------------------------------------------------
    def to_sensor(self, world: np.ndarray) -> np.ndarray:
        return (np.asarray(world, dtype=np.float64) - self.translation) @ self.rotation

    def _angles_to_pixel(self, elevation, azimuth):
        """Fractional row (linear between beams, extrapolated beyond the first and
        last beam: a point outside the field of view gets a row < 0 or > H - 1)
        and fractional column (0 .. W)."""
        elev = self.row_elevation
        index = np.clip(np.searchsorted(-elev, -np.asarray(elevation)) - 1, 0, self.height - 2)
        row = index + (elevation - elev[index]) / (elev[index + 1] - elev[index])
        r0 = np.clip(np.round(row).astype(int), 0, self.height - 1)
        azimuth0 = self.row_azimuth0[r0]
        column = np.mod((azimuth - azimuth0) * self.sense / self.column_step, self.width)
        return row, column

    def project(self, world: np.ndarray):
        """(N, 3) world points -> fractional rows, columns (0 .. W, wrapping) and
        the range of each point from the beam origin of its pixel."""
        q = self.to_sensor(world)
        row, column = self._angles_to_pixel(np.arcsin(np.clip(q[:, 2] / np.maximum(np.linalg.norm(q, axis=1), 1e-9),
                                                               -1, 1)), np.arctan2(q[:, 1], q[:, 0]))
        r = np.clip(np.round(row).astype(int), 0, self.height - 1)
        c = np.mod(np.round(column).astype(int), self.width)
        local = q - self.offset[r, c]
        distance = np.maximum(np.linalg.norm(local, axis=1), 1e-9)
        row, column = self._angles_to_pixel(np.arcsin(np.clip(local[:, 2] / distance, -1, 1)),
                                            np.arctan2(local[:, 1], local[:, 0]))
        return row, column, distance

    def pixel_times(self, timestamps: np.ndarray) -> np.ndarray:
        """(H, W) time of every pixel [s] from the (W,) column timestamps of a
        frame (measurement order): pixel (r, c) of the destaggered image was
        measured at column (c - shift_r) mod W."""
        columns = np.mod(np.arange(self.width)[None, :] - self.pixel_shift[:, None], self.width)
        return np.asarray(timestamps, dtype=np.float64)[columns]

    def column_distance_unwrapped(self, columns: np.ndarray, reference: float) -> np.ndarray:
        """Columns unwrapped to be within half a turn of 'reference'."""
        return reference + np.mod(columns - reference + self.width / 2, self.width) - self.width / 2

    # -- torch ---------------------------------------------------------------------------------
    def torch_tables(self, device, dtype):
        import torch
        return {"rotation": torch.as_tensor(self.rotation, dtype=dtype, device=device),
                "translation": torch.as_tensor(self.translation, dtype=dtype, device=device),
                "elevation": torch.as_tensor(self.row_elevation, dtype=dtype, device=device),
                "azimuth0": torch.as_tensor(self.row_azimuth0, dtype=dtype, device=device),
                "offset": torch.as_tensor(self.offset, dtype=dtype, device=device)}

    def pixels_of(self, world: np.ndarray):
        """Nearest pixel (row, column), clipped to the image, of (N, 3) world points (numpy)."""
        row, column, _ = self.project(world)
        return (np.clip(np.round(row).astype(int), 0, self.height - 1),
                np.mod(np.round(column).astype(int), self.width))

    def project_torch(self, world, tables, reference_column: float, pixels=None):
        """Differentiable projection of (N, 3) world points: rows and columns
        (unwrapped around reference_column) and ranges. The pixel used for the
        beam-origin offset and for the row's azimuth origin ('pixels': (rows,
        columns) tensors, computed once per optimisation round with pixels_of)
        is held fixed; the angles keep their gradient."""
        import torch
        q = (world - tables["translation"]) @ tables["rotation"]
        if pixels is None:
            with torch.no_grad():
                r_np, c_np = self.pixels_of(world.detach().cpu().numpy().astype(np.float64))
            r = torch.as_tensor(r_np, device=world.device)
            c = torch.as_tensor(c_np, device=world.device)
        else:
            r, c = pixels
        local = q - tables["offset"][r, c]
        distance = torch.linalg.norm(local, dim=1).clamp_min(1e-9)
        elevation = torch.asin((local[:, 2] / distance).clamp(-1.0, 1.0))
        azimuth = torch.atan2(local[:, 1], local[:, 0])
        # rows: piecewise linear in the elevation between the two neighbouring beams
        elev = tables["elevation"]
        r0 = torch.clamp(torch.searchsorted(-elev, -elevation.detach()) - 1, 0, self.height - 2)
        e0, e1 = elev[r0], elev[r0 + 1]
        rows = r0.to(world.dtype) + (elevation - e0) / (e1 - e0)
        a0 = tables["azimuth0"][r.clamp(0, self.height - 1)]
        raw = (azimuth - a0) * self.sense / self.column_step
        # unwrap around the reference column
        columns = reference_column + torch.remainder(raw - reference_column + self.width / 2, self.width) - self.width / 2
        return rows, columns, distance
