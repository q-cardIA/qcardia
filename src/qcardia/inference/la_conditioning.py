"""
LAx conditioning preprocessor for context-aware SAx segmentation.

For each SAx (slice, frame) position, computes a 1D label vector of length
`n_samples` (default 256) sampled from a 4CH segmentation along the geometric
intersection of the SAx plane with the 4CH plane.

Entry point:
    la_vectors = compute_la_vectors(
        sax_dicom_dir, lax_dicom_dir, lax_model_path,
        n_samples=256, device='cuda'
    )
    # Returns: torch.Tensor  shape (Z, T, n_samples), dtype=torch.long
    # Raises if the 4CH data or its geometry cannot produce vectors: a model
    # trained with LA conditioning silently skips FiLM when handed None, which
    # would look like a successful run of a different model.
"""

import numpy as np
import torch
import pydicom
from pathlib import Path
from natsort import natsorted
from typing import Tuple


from qcardia_data.pipeline.la_sa_intersection.geometry_utils import (
    calculate_intersection_line,
    find_line_segment_bounds,
    create_intersection_mask,
    get_slice_plane_parameters,
)


# ── public entry point ───────────────────────────────────────────────────────

def compute_la_vectors(
    sax_dicom_dir: Path,
    lax_dicom_dir: Path,
    lax_model_path: Path,
    n_samples: int = 256,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute per-(slice, frame) LA conditioning vectors from 4CH DICOM data.

    Steps:
      1. Build DICOM affine matrices for SAx stack and 4CH plane.
      2. Run the plain nnUNet on the 4CH DICOM to get a LAx segmentation.
      3. For each SAx slice, compute the geometric intersection line on the
         4CH image and create a binary mask.
      4. Sample `n_samples` class labels along that mask for every frame.

    Args:
        sax_dicom_dir:  Folder containing SAx DICOM files.
        lax_dicom_dir:  Folder containing 4CH DICOM files.
        lax_model_path: WandB-style model directory for the plain (non-wLA)
                        nnUNet used to segment the 4CH view.
        n_samples:      Length of each LA vector (must equal la_vector_dim in
                        the wLA model config, typically 256).
        device:         Inference device for LAx segmentation ('cuda'/'cpu').

    Returns:
        Tensor of shape (Z, T, n_samples) with integer class labels.

    Raises:
        FileNotFoundError / ValueError: if the 4CH data or its geometry cannot
            produce vectors.
    """
    sax_dicom_dir = Path(sax_dicom_dir)
    lax_dicom_dir = Path(lax_dicom_dir)
    lax_model_path = Path(lax_model_path)

    if not lax_dicom_dir.exists():
        raise FileNotFoundError(f"4CH folder not found: {lax_dicom_dir}")

    print(f"[la_conditioning] Computing LA vectors")
    print(f"  SAx dir : {sax_dicom_dir}")
    print(f"  4CH dir : {lax_dicom_dir}")
    print(f"  Model   : {lax_model_path}")
    print(f"  Samples : {n_samples}")

    # Resolve the actual DICOM folders (CINE_4CH/ may contain a subdirectory).
    # sax_dicom_dir is already the resolved DICOM folder (set from CineSeries.folder).
    # lax_dicom_dir is the CINE_4CH parent — needs get_data_directory().
    from qcardia.utils import get_data_directory
    resolved_lax = get_data_directory(lax_dicom_dir)
    if resolved_lax is None:
        raise FileNotFoundError(f"No DICOM files found under {lax_dicom_dir}")
    if resolved_lax != lax_dicom_dir:
        print(f"  4CH resolved: {resolved_lax}")
    lax_dicom_dir = resolved_lax

    # Step 1 — build affines
    sax_affine, sax_shape, sax_frames = _build_volume_affine(sax_dicom_dir)
    lax_affine, lax_shape, lax_frames = _build_volume_affine(lax_dicom_dir)
    Z, T = sax_shape[2], sax_frames
    print(f"  SAx geometry: shape={sax_shape}, frames={sax_frames}")
    print(f"  4CH geometry: shape={lax_shape}, frames={lax_frames}")

    # Step 2 — segment 4CH
    lax_seg = _segment_lax(lax_dicom_dir, lax_model_path, device)
    # lax_seg from CineSeries has shape (Z, T, H, W) where Z=1 for 4CH.
    # Squeeze dim-0 (the single spatial slice) to get (T, H, W).
    if lax_seg.ndim == 4:
        lax_seg = lax_seg[0, :, :, :]   # (T, H, W)
    # Fallback: if still only one frame, broadcast to match SAx frame count
    if lax_seg.shape[0] == 1 and T > 1:
        lax_seg = np.repeat(lax_seg, T, axis=0)
    print(f"  4CH seg shape: {lax_seg.shape}  labels: {np.unique(lax_seg).tolist()}")

    # Step 3 — intersection masks
    H_lax, W_lax = lax_shape[0], lax_shape[1]
    intersection_masks = _compute_intersection_masks(
        sax_affine, sax_shape, lax_affine, (H_lax, W_lax, lax_shape[2])
    )
    n_valid = int((intersection_masks.sum(axis=(1, 2)) > 0).sum())
    print(f"  Intersection masks: {Z} slices, {n_valid} with valid intersection")

    # Step 4 — sample vectors
    la_vectors = _sample_vectors(lax_seg, intersection_masks, n_samples)
    print(f"  LA vectors tensor: {la_vectors.shape}  "
          f"non-zero: {(la_vectors != 0).float().mean():.1%}")

    return la_vectors


# ── DICOM geometry ────────────────────────────────────────────────────────────

def _build_volume_affine(
    dicom_dir: Path,
) -> Tuple[np.ndarray, Tuple[int, int, int], int]:
    """
    Build a NIfTI-convention 4×4 affine from DICOM headers.

    The affine maps voxel index (i=row, j=col, k=slice) to LPS world mm.

    Returns:
        affine: (4, 4) float64
        shape:  (H, W, Z)  spatial dimensions
        n_frames: number of temporal frames
    """
    files = natsorted([
        f for f in dicom_dir.iterdir()
        if f.is_file() and not f.stem.startswith(".")
    ])
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
    F = np.array([float(x) for x in ds0.ImageOrientationPatient[:3]])  # row cosines
    U = np.array([float(x) for x in ds0.ImageOrientationPatient[3:]])  # col cosines
    P = np.array([float(x) for x in ds0.ImagePositionPatient])
    dr = float(ds0.PixelSpacing[0])  # row spacing (mm)
    dc = float(ds0.PixelSpacing[1])  # col spacing (mm)
    H, W = int(ds0.Rows), int(ds0.Columns)

    # Slice normal direction
    normal = np.cross(F, U)
    normal_norm = normal / np.linalg.norm(normal)

    # Slice spacing, same tag preference as BaseSeries._get_pixel_spacing
    dz = getattr(ds0, "SpacingBetweenSlices", None) or getattr(ds0, "SliceThickness", None)
    dz = float(dz) if dz is not None else 0.0
    if dz <= 0:
        raise ValueError(
            f"{dicom_dir} has no usable SpacingBetweenSlices or SliceThickness; "
            f"cannot build a physical affine for LA conditioning."
        )

    # Build affine (NIfTI: col-0 = d/drow, col-1 = d/dcol, col-2 = d/dslice)
    affine = np.eye(4, dtype=float)
    affine[:3, 0] = U * dr          # moving along image rows
    affine[:3, 1] = F * dc          # moving along image cols
    affine[:3, 2] = normal_norm * dz  # moving between slices
    affine[:3, 3] = P               # world position of voxel (0, 0, 0)

    # Count unique slices
    true_pos = [
        float(np.dot([float(x) for x in ds.ImagePositionPatient],
                     np.cross([float(x) for x in ds.ImageOrientationPatient[:3]],
                              [float(x) for x in ds.ImageOrientationPatient[3:]])))
        for ds in datasets
    ]
    n_slices = len(set(round(p, 2) for p in true_pos))
    n_frames = max(1, len(datasets) // n_slices)

    return affine, (H, W, n_slices), n_frames


# ── LAx segmentation ─────────────────────────────────────────────────────────

def _segment_lax(
    lax_dicom_dir: Path,
    model_path: Path,
    device: str = "cuda",
) -> np.ndarray:
    """
    Run segmentation on the 4CH DICOM series using the given model.

    Returns integer label array of shape (T, H, W).
    Lazy-imports CineSeries to avoid circular dependency.
    """
    from qcardia.series import CineSeries  # lazy import — avoids circular dep
    from qcardia.utils import get_data_directory

    # Resolve actual DICOM directory (4CH may have a subdirectory)
    actual_dir = get_data_directory(lax_dicom_dir)
    if actual_dir is None:
        raise FileNotFoundError(f"No DICOM data found under {lax_dicom_dir}")

    print(f"  [la_conditioning] Segmenting 4CH from: {actual_dir}")
    series = CineSeries(actual_dir)
    seg = series.predict_segmentation(model_path)
    # seg: (Z_lax, T, H, W)  — for 4CH, Z_lax == 1
    return seg.astype(np.int32)


# ── intersection mask computation ────────────────────────────────────────────

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
        masks: (Z_sax, H_lax, W_lax) uint8  — 1 = on intersection line
    """
    H_lax, W_lax, _ = lax_shape
    Z_sax = sax_shape[2]

    la_params = {
        "center": lax_affine[:3, 3],
        "normal": lax_affine[:3, 2] / np.linalg.norm(lax_affine[:3, 2]),
    }

    # Pixel spacings from affine diagonal magnitudes
    sax_voxel_dims = np.array([
        np.linalg.norm(sax_affine[:3, 0]),
        np.linalg.norm(sax_affine[:3, 1]),
        np.linalg.norm(sax_affine[:3, 2]),
    ])

    masks = np.zeros((Z_sax, H_lax, W_lax), dtype=np.uint8)

    for z in range(Z_sax):
        sa_params = get_slice_plane_parameters(sax_affine, z, sax_shape)
        line_params = calculate_intersection_line(sa_params, la_params)
        start_world, end_world = find_line_segment_bounds(
            line_params, sax_affine, sax_shape, sax_voxel_dims, z
        )
        if start_world is None or end_world is None:
            continue

        # Create a 2D mask on the LAx image in world coordinates
        mask_2d = create_intersection_mask(
            (H_lax, W_lax), start_world, end_world,
            line_width=line_width,
            coordinate_type="world",
            affine=lax_affine,
        )
        masks[z] = mask_2d

    return masks


# ── label vector sampling ────────────────────────────────────────────────────

def _sample_vectors(
    lax_seg: np.ndarray,
    intersection_masks: np.ndarray,
    n_samples: int,
) -> torch.Tensor:
    """
    Sample label vectors from the LAx segmentation along each intersection mask.

    Args:
        lax_seg:             (T, H, W) integer class labels
        intersection_masks:  (Z, H, W) binary uint8
        n_samples:           length of each output vector

    Returns:
        (Z, T, n_samples) long tensor
    """
    Z = intersection_masks.shape[0]
    T = lax_seg.shape[0]
    vectors = torch.zeros(Z, T, n_samples, dtype=torch.long)

    for z in range(Z):
        mask = intersection_masks[z]
        points = np.argwhere(mask > 0)          # (N, 2) [row, col]
        if len(points) == 0:
            continue

        # Sort points along the intersection line direction
        points_ordered = _sort_line_points(points)

        for t in range(T):
            seg_frame = lax_seg[t]              # (H, W)
            labels = _sample_labels_at_points(seg_frame, points_ordered, n_samples)
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
    seg_hw: np.ndarray,
    ordered_points: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """
    Gather class labels at `ordered_points` from `seg_hw`, then
    uniformly resample to exactly `n_samples` values.
    """
    rows = np.clip(ordered_points[:, 0], 0, seg_hw.shape[0] - 1)
    cols = np.clip(ordered_points[:, 1], 0, seg_hw.shape[1] - 1)
    raw = seg_hw[rows, cols].astype(np.int64)

    n = len(raw)
    idx = np.round(np.linspace(0, n - 1, n_samples)).astype(int)
    return raw[idx]
