from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# In DICOM's patient coordinates (LPS) the heart's apex points left, anterior and inferior.
_APEX_DIRECTION = np.array([1.0, -1.0, -1.0]) / np.sqrt(3.0)


@dataclass
class VolumeGeometry:
    """DICOM-derived spatial geometry for one image volume.

    Coordinate convention follows DICOM:
      row_dir — direction of increasing column index (IOP[0:3])
      col_dir — direction of increasing row index   (IOP[3:6])
      normal  — cross(row_dir, col_dir), points in the slice-stack direction

    pixel_spacing — (dr, dc) in mm: dr is the spacing between rows,
                    dc is the spacing between columns.
    ipp_list      — one IPP (3D world point) per slice, in display order.
    """

    row_dir: np.ndarray
    col_dir: np.ndarray
    normal: np.ndarray
    ipp_list: list[np.ndarray]
    pixel_spacing: tuple[float, float]
    n_rows: int
    n_cols: int

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dicom_slices(cls, slice_metas: list) -> VolumeGeometry | None:
        """Build from an ordered list of pydicom datasets (one per slice).

        Returns None if the required DICOM tags are missing or malformed.
        The caller is responsible for passing slices in the desired display order.
        """
        if not slice_metas:
            return None
        try:
            first = slice_metas[0]
            iop = np.array(first.ImageOrientationPatient, dtype=float)
            row_dir = iop[:3]
            col_dir = iop[3:]
            normal = np.cross(row_dir, col_dir)
            mag = np.linalg.norm(normal)
            if mag < 1e-9:
                return None
            normal /= mag

            ps = first.PixelSpacing
            pixel_spacing = (float(ps[0]), float(ps[1]))
            n_rows = int(first.Rows)
            n_cols = int(first.Columns)

            ipp_list = [
                np.array(m.ImagePositionPatient, dtype=float) for m in slice_metas
            ]
            return cls(row_dir, col_dir, normal, ipp_list, pixel_spacing, n_rows, n_cols)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Core transforms
    # ------------------------------------------------------------------

    def pixel_to_world(self, row: float, col: float, slice_idx: int) -> np.ndarray:
        """Return the 3D world point for pixel (row, col) in slice slice_idx."""
        ipp = self.ipp_list[int(slice_idx)]
        dr, dc = self.pixel_spacing
        return ipp + row * dr * self.col_dir + col * dc * self.row_dir

    def world_to_pixel(self, world_pt: np.ndarray) -> tuple[float, float, int] | None:
        """Return (row, col, slice_idx) for the closest voxel to world_pt.

        Returns None if ipp_list is empty.
        """
        if not self.ipp_list:
            return None
        projections = [float(np.dot(world_pt - ipp, self.normal)) for ipp in self.ipp_list]
        slice_idx = int(np.argmin(np.abs(projections)))
        delta = world_pt - self.ipp_list[slice_idx]
        dr, dc = self.pixel_spacing
        row = float(np.dot(delta, self.col_dir)) / dr
        col = float(np.dot(delta, self.row_dir)) / dc
        return (row, col, slice_idx)

    def slice_position(self, slice_idx: int) -> float:
        """Scalar position of slice_idx along the stack normal (mm)."""
        return float(np.dot(self.ipp_list[int(slice_idx)], self.normal))

    def apex_to_base(self) -> list[int]:
        """Slice indices of a short-axis stack, from the apex to the base."""
        toward_apex = [float(np.dot(ipp, _APEX_DIRECTION)) for ipp in self.ipp_list]
        return sorted(range(len(self.ipp_list)), key=lambda i: -toward_apex[i])

    # ------------------------------------------------------------------
    # Multi-panel utilities
    # ------------------------------------------------------------------

    def nearest_slice(self, other: VolumeGeometry, z_idx: int) -> int | None:
        """The slice of other that is closest to the centre of slice z_idx."""
        centre = self.pixel_to_world((self.n_rows - 1) / 2, (self.n_cols - 1) / 2, z_idx)
        hit = other.world_to_pixel(centre)
        return hit[2] if hit is not None else None

    def is_parallel_to(self, other: VolumeGeometry, tol: float = 0.05) -> bool:
        """True if the two volumes have nearly the same slice orientation."""
        return abs(float(np.dot(self.normal, other.normal))) > 1.0 - tol

    def intersection_line_in_image(
        self, other: VolumeGeometry, z_idx: int
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Pixel endpoints of the line where self's slice z_idx intersects other.

        Returns two (row, col) points in other's pixel space, or None if the
        planes are parallel or the intersection misses the image entirely.
        Used to draw scout lines across panels.
        """
        p_a = self.ipp_list[int(z_idx)]
        n_a = self.normal
        p_b = other.ipp_list[0]
        dr_b, dc_b = other.pixel_spacing

        d0 = float(np.dot(p_b - p_a, n_a))
        d_r = dr_b * float(np.dot(other.col_dir, n_a))
        d_c = dc_b * float(np.dot(other.row_dir, n_a))

        if abs(d_r) < 1e-9 and abs(d_c) < 1e-9:
            return None  # planes are parallel

        H = float(other.n_rows - 1)
        W = float(other.n_cols - 1)
        pts: list[tuple[float, float]] = []

        def _add(r: float, c: float) -> None:
            if 0.0 <= r <= H and 0.0 <= c <= W:
                pts.append((r, c))

        if abs(d_r) > 1e-9:
            for r_val in (0.0, H):
                if abs(d_c) > 1e-9:
                    _add(r_val, (-d0 - d_r * r_val) / d_c)

        if abs(d_c) > 1e-9:
            for c_val in (0.0, W):
                if abs(d_r) > 1e-9:
                    _add((-d0 - d_c * c_val) / d_r, c_val)

        if len(pts) < 2:
            return None
        return pts[0], pts[-1]
