"""
LAx conditioning preprocessor for context-aware SAx segmentation.

For each SAx (slice, frame) position, computes a 1D label vector of length
`n_samples` sampled from a 4CH segmentation along the geometric intersection
of the SAx plane with the 4CH plane.

Entry point:
    la_vectors = compute_la_vectors(
        sax_dicom_dir, lax_dicom_dir, lax_model_path,
        n_samples=256, device='cuda'
    )
    # Returns: torch.Tensor, shape (Z, T, n_samples), dtype=torch.long
    # If 4CH data is missing or geometry fails -> returns None (safe fallback)

Depends on `qcardia_data.pipeline.la_sa_intersection.geometry_utils`, which
(as of writing) only exists on qcardia-data-dev's `dim-2D+Z` branch — that
package must be installed from that branch for LA-conditioned models to work.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pydicom
import torch
from natsort import natsorted

try:
    from qcardia_data.pipeline.la_sa_intersection.geometry_utils import (
        calculate_intersection_line,
        create_intersection_mask,
        find_line_segment_bounds,
        get_slice_plane_parameters,
    )

    _GEOM_AVAILABLE = True
except ImportError:
    _GEOM_AVAILABLE = False
    print(
        "[la_conditioning] WARNING: qcardia_data geometry utils not available "
        "(requires qcardia-data-dev on the dim-2D+Z branch) — LA conditioning "
        "will return None (model runs without LA vectors)."
    )


def compute_la_vectors(
    sax_dicom_dir: Path,
    lax_dicom_dir: Path,
    lax_model_path: Path,
    n_samples: int = 256,
    device: str = "cpu",
) -> Optional[torch.Tensor]:
    """
    Compute per-(slice, frame) LA conditioning vectors from 4CH DICOM data.

    Steps:
      1. Build DICOM affine matrices for the SAx stack and the 4CH plane.
      2. Segment the 4CH DICOM with a plain (non-conditioned) UNet2d.
      3. For each SAx slice, compute the geometric intersection line on the
         4CH image and build a binary mask.
      4. Sample `n_samples` class labels along that mask for every frame.

    Args:
        sax_dicom_dir: Folder containing the SAx DICOM files (self.folder).
        lax_dicom_dir: Folder containing the 4CH DICOM files.
        lax_model_path: Model directory for the plain UNet2d used to segment
            the 4CH view.
        n_samples: Length of each LA vector (must equal the wLA model's
            `la_vector_dim`).
        device: Inference device for the 4CH segmentation.

    Returns:
        Tensor of shape (Z, T, n_samples) with integer class labels, or None
        if the computation cannot be completed.
    """
    if not _GEOM_AVAILABLE:
        return None

    sax_dicom_dir = Path(sax_dicom_dir)
    lax_dicom_dir = Path(lax_dicom_dir)
    lax_model_path = Path(lax_model_path)

    if not lax_dicom_dir.exists():
        print(f"[la_conditioning] 4CH folder not found: {lax_dicom_dir} — skipping")
        return None

    try:
        sax_affine, sax_shape, _ = _build_volume_affine(sax_dicom_dir)
        lax_affine, lax_shape, lax_frames = _build_volume_affine(lax_dicom_dir)
        Z = sax_shape[2]

        lax_seg = _segment_lax(lax_dicom_dir, lax_model_path, device)
        # lax_seg from CineSeries has shape (Z_lax, T, H, W), Z_lax == 1 for 4CH.
        if lax_seg.ndim == 4:
            lax_seg = lax_seg[0, :, :, :]  # (T, H, W)
        if lax_seg.shape[0] == 1 and lax_frames > 1:
            lax_seg = np.repeat(lax_seg, lax_frames, axis=0)

        H_lax, W_lax = lax_shape[0], lax_shape[1]
        intersection_masks = _compute_intersection_masks(
            sax_affine, sax_shape, lax_affine, (H_lax, W_lax, lax_shape[2])
        )

        la_vectors = _sample_vectors(lax_seg, intersection_masks, n_samples)
        return la_vectors

    except Exception as exc:
        print(
            f"[la_conditioning] ERROR — returning None "
            f"(model will run without LA vectors): {exc}"
        )
        return None


# ── DICOM geometry ────────────────────────────────────────────────────────────


def _build_volume_affine(
    dicom_dir: Path,
) -> Tuple[np.ndarray, Tuple[int, int, int], int]:
    """
    Build a NIfTI-convention 4x4 affine from DICOM headers.

    Returns:
        affine: (4, 4) float64, maps voxel index (row, col, slice) to LPS mm.
        shape: (H, W, Z) spatial dimensions.
        n_frames: number of temporal frames.
    """
    files = natsorted(
        [f for f in dicom_dir.iterdir() if f.is_file() and not f.stem.startswith(".")]
    )
    datasets = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            if "PixelData" in ds:
                datasets.append(ds)
        except Exception:
            continue

    if not datasets:
        raise FileNotFoundError(f"No valid DICOMs in {dicom_dir}")

    ds0 = datasets[0]
    row_cosines = np.array([float(x) for x in ds0.ImageOrientationPatient[:3]])
    col_cosines = np.array([float(x) for x in ds0.ImageOrientationPatient[3:]])
    position = np.array([float(x) for x in ds0.ImagePositionPatient])
    dr = float(ds0.PixelSpacing[0])
    dc = float(ds0.PixelSpacing[1])
    H, W = int(ds0.Rows), int(ds0.Columns)

    normal = np.cross(row_cosines, col_cosines)
    normal_norm = normal / np.linalg.norm(normal)

    try:
        dz = float(ds0.SliceThickness)
        if dz <= 0:
            dz = 8.0
    except Exception:
        dz = 8.0

    affine = np.eye(4, dtype=float)
    affine[:3, 0] = col_cosines * dr
    affine[:3, 1] = row_cosines * dc
    affine[:3, 2] = normal_norm * dz
    affine[:3, 3] = position

    true_positions = [
        float(
            np.dot(
                [float(x) for x in ds.ImagePositionPatient],
                np.cross(
                    [float(x) for x in ds.ImageOrientationPatient[:3]],
                    [float(x) for x in ds.ImageOrientationPatient[3:]],
                ),
            )
        )
        for ds in datasets
    ]
    n_slices = len(set(round(p, 2) for p in true_positions))
    n_frames = max(1, len(datasets) // n_slices)

    return affine, (H, W, n_slices), n_frames


def _segment_lax(lax_dicom_dir: Path, model_path: Path, device: str) -> np.ndarray:
    """Segment the 4CH DICOM series with the given plain UNet2d model."""
    from qcardia.series import CineSeries  # lazy import — avoids circular dep

    series = CineSeries(lax_dicom_dir)
    seg = series.predict_segmentation(model_path)
    return seg.astype(np.int32)


def _compute_intersection_masks(
    sax_affine: np.ndarray,
    sax_shape: Tuple[int, int, int],
    lax_affine: np.ndarray,
    lax_shape: Tuple[int, int, int],
    line_width: int = 3,
) -> np.ndarray:
    """
    Compute a binary intersection mask on the LAx image for each SAx slice.

    Returns:
        masks: (Z_sax, H_lax, W_lax) uint8 — 1 = on intersection line.
    """
    H_lax, W_lax, _ = lax_shape
    Z_sax = sax_shape[2]

    la_params = {
        "center": lax_affine[:3, 3],
        "normal": lax_affine[:3, 2] / np.linalg.norm(lax_affine[:3, 2]),
    }
    sax_voxel_dims = np.array(
        [
            np.linalg.norm(sax_affine[:3, 0]),
            np.linalg.norm(sax_affine[:3, 1]),
            np.linalg.norm(sax_affine[:3, 2]),
        ]
    )

    masks = np.zeros((Z_sax, H_lax, W_lax), dtype=np.uint8)
    for z in range(Z_sax):
        sa_params = get_slice_plane_parameters(sax_affine, z, sax_shape)
        line_params = calculate_intersection_line(sa_params, la_params)
        start_world, end_world = find_line_segment_bounds(
            line_params, sax_affine, sax_shape, sax_voxel_dims, z
        )
        if start_world is None or end_world is None:
            continue
        masks[z] = create_intersection_mask(
            (H_lax, W_lax),
            start_world,
            end_world,
            line_width=line_width,
            coordinate_type="world",
            affine=lax_affine,
        )

    return masks


def _sample_vectors(
    lax_seg: np.ndarray, intersection_masks: np.ndarray, n_samples: int
) -> torch.Tensor:
    """
    Sample label vectors from the LAx segmentation along each intersection mask.

    Args:
        lax_seg: (T, H, W) integer class labels.
        intersection_masks: (Z, H, W) binary uint8.

    Returns:
        (Z, T, n_samples) long tensor.
    """
    Z = intersection_masks.shape[0]
    T = lax_seg.shape[0]
    vectors = torch.zeros(Z, T, n_samples, dtype=torch.long)

    for z in range(Z):
        points = np.argwhere(intersection_masks[z] > 0)  # (N, 2) [row, col]
        if len(points) == 0:
            continue
        points_ordered = _sort_line_points(points)
        for t in range(T):
            labels = _sample_labels_at_points(lax_seg[t], points_ordered, n_samples)
            vectors[z, t] = torch.from_numpy(labels)

    return vectors


def _sort_line_points(points: np.ndarray) -> np.ndarray:
    """Sort intersection pixels along the line direction via PCA."""
    if len(points) <= 1:
        return points
    center = points.mean(axis=0)
    centered = points - center
    cov = centered.T @ centered
    _, vecs = np.linalg.eigh(cov)
    direction = vecs[:, -1]
    proj = centered @ direction
    return points[np.argsort(proj)]


def _sample_labels_at_points(
    seg_hw: np.ndarray, ordered_points: np.ndarray, n_samples: int
) -> np.ndarray:
    """Gather labels at `ordered_points`, then uniformly resample to `n_samples`."""
    rows = np.clip(ordered_points[:, 0], 0, seg_hw.shape[0] - 1)
    cols = np.clip(ordered_points[:, 1], 0, seg_hw.shape[1] - 1)
    raw = seg_hw[rows, cols].astype(np.int64)

    n = len(raw)
    idx = np.round(np.linspace(0, n - 1, n_samples)).astype(int)
    return raw[idx]
