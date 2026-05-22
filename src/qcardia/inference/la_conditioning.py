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
    # If 4CH data is missing or geometry fails -> returns None (safe fallback)

Debug visualisation:
    save_LAx_conditioning(la_vectors, intersection_masks, lax_seg, out_dir)
"""

import sys
import json
import numpy as np
import torch
import pydicom
from pathlib import Path
from natsort import natsorted
from typing import Optional, Tuple


# ── geometry utilities from qcardia-data-dev ────────────────────────────────
_DATA_DEV = Path(__file__).parents[5] / "qcardia-data-dev" / "src"
if _DATA_DEV.exists() and str(_DATA_DEV) not in sys.path:
    sys.path.insert(0, str(_DATA_DEV))

try:
    from qcardia_data.pipeline.la_sa_intersection.geometry_utils import (
        calculate_intersection_line,
        find_line_segment_bounds,
        create_intersection_mask,
        get_slice_plane_parameters,
    )
    _GEOM_AVAILABLE = True
except ImportError:
    _GEOM_AVAILABLE = False
    print("[la_conditioning] WARNING: qcardia_data geometry utils not available — "
          "LA conditioning will return None (model runs without LA vectors).")


# ── public entry point ───────────────────────────────────────────────────────

def compute_la_vectors(
    sax_dicom_dir: Path,
    lax_dicom_dir: Path,
    lax_model_path: Path,
    n_samples: int = 256,
    device: str = "cuda",
    debug_out_dir: Optional[Path] = None,
) -> Optional[torch.Tensor]:
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
        debug_out_dir:  If given, save debug images/JSON here.

    Returns:
        Tensor of shape (Z, T, n_samples) with integer class labels, or None
        if the computation cannot be completed (missing files, geometry failure).
    """
    if not _GEOM_AVAILABLE:
        return None

    sax_dicom_dir = Path(sax_dicom_dir)
    lax_dicom_dir = Path(lax_dicom_dir)
    lax_model_path = Path(lax_model_path)

    if not lax_dicom_dir.exists():
        print(f"[la_conditioning] 4CH folder not found: {lax_dicom_dir} — skipping LA vectors")
        return None

    print(f"[la_conditioning] Computing LA vectors")
    print(f"  SAx dir : {sax_dicom_dir}")
    print(f"  4CH dir : {lax_dicom_dir}")
    print(f"  Model   : {lax_model_path}")
    print(f"  Samples : {n_samples}")

    try:
        # Resolve the actual DICOM folders (CINE_4CH/ may contain a subdirectory).
        # sax_dicom_dir is already the resolved DICOM folder (set from CineSeries.folder).
        # lax_dicom_dir is the CINE_4CH parent — needs get_data_directory().
        from qcardia import pipeline_utils as _pu
        resolved_lax = _pu.get_data_directory(lax_dicom_dir)
        if resolved_lax is None:
            print(f"[la_conditioning] WARNING: no DICOM files found under {lax_dicom_dir}")
            return None
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

        # Attach intermediate products so callers can access them for debug saves
        la_vectors._debug_intersection_masks = intersection_masks
        la_vectors._debug_lax_seg = lax_seg
        la_vectors._debug_lax_pixel_spacing_mm = (
            float(np.linalg.norm(lax_affine[:3, 0])),
            float(np.linalg.norm(lax_affine[:3, 1])),
        )

        # Optional immediate debug output
        if debug_out_dir is not None:
            save_LAx_conditioning(la_vectors, intersection_masks, lax_seg, Path(debug_out_dir))

        return la_vectors

    except Exception as exc:
        import traceback
        print(f"[la_conditioning] ERROR — returning None (model will run without LA vectors): {exc}")
        traceback.print_exc()
        return None


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

    # Slice thickness
    dz = float(getattr(ds0, "SliceThickness", None) or 8.0)
    try:
        dz = float(ds0.SliceThickness)
        if dz <= 0:
            dz = 8.0
    except Exception:
        dz = 8.0

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
    from qcardia import pipeline_utils

    # Resolve actual DICOM directory (4CH may have a subdirectory)
    actual_dir = pipeline_utils.get_data_directory(lax_dicom_dir)
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


# ── debug / visualisation ────────────────────────────────────────────────────

# Colour palette for the 4 cardiac classes (BG, LV, MYO, RV)
_CLASS_COLORS = np.array([
    [0.15, 0.15, 0.15],   # 0  BG    — dark grey
    [0.92, 0.15, 0.15],   # 1  LV    — red
    [0.15, 0.80, 0.15],   # 2  MYO   — green
    [0.15, 0.35, 0.95],   # 3  RV    — blue
])
_CLASS_NAMES = {0: "BG", 1: "LV", 2: "MYO", 3: "RV"}
_CLASS_MPLCOLORS = ["#262626", "#eb2626", "#26cc26", "#2659f2"]


def save_la_debug(
    la_vectors: torch.Tensor,
    intersection_masks: np.ndarray,
    lax_seg: np.ndarray,
    out_dir: Path,
) -> None:
    """
    Save three diagnostic figures + a JSON summary for verifying LA conditioning.

    Files written to `out_dir/`:
      la_vectors_summary.json      — per-slice statistics
      01_lax_segmentation.png      — 4CH segmentation across selected frames,
                                     with intersection lines overlaid
      02_volume_curves.png         — LV / MYO / RV volume-over-frame from 4CH seg
      03_la_vectors_per_slice.png  — extracted label vectors as a heatmap grid
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Z, T, N = la_vectors.shape
    vecs_np = la_vectors.numpy()           # (Z, T, N)

    # Recover pixel spacing for volume calculation (mm²/pixel)
    px_dr, px_dc = getattr(la_vectors, "_debug_lax_pixel_spacing_mm",
                           (1.0, 1.0))
    pixel_area_cm2 = (px_dr * px_dc) / 100.0   # mm² → cm² (2-D area, not volume)

    # ── 1. JSON summary ──────────────────────────────────────────────────────
    summary = {}
    for z in range(Z):
        n_px = int((intersection_masks[z] > 0).sum())
        frame_stats = {}
        for t in range(T):
            v = vecs_np[z, t]
            unique, counts = np.unique(v, return_counts=True)
            frame_stats[str(t)] = {
                _CLASS_NAMES.get(int(u), str(u)): int(c)
                for u, c in zip(unique, counts)
            }
        dominant_labels = [
            _CLASS_NAMES.get(int(vecs_np[z, t].argmax()), "?")
            for t in range(T)
        ]
        summary[f"slice_{z:02d}"] = {
            "intersection_pixels_on_4ch": n_px,
            "has_intersection": n_px > 0,
            "frame_label_counts": frame_stats,
        }
    json_path = out_dir / "la_vectors_summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  [LAx_conditioning] {json_path.name}")

    try:
        # ── 2. 4CH segmentation + intersection lines ──────────────────────────
        n_show = min(6, T)
        show_frames = np.round(np.linspace(0, T - 1, n_show)).astype(int)

        # Each column: one frame; top row = seg+mask overlay, bottom = seg alone
        fig, axes = plt.subplots(2, n_show, figsize=(3.2 * n_show, 5.5))
        if n_show == 1:
            axes = axes[:, np.newaxis]

        combined_mask = (intersection_masks.sum(axis=0) > 0).astype(np.uint8)

        for col_i, t in enumerate(show_frames):
            seg = lax_seg[t] if lax_seg.ndim == 3 else lax_seg[t, 0]
            seg_clipped = np.clip(seg, 0, 3).astype(int)
            rgb = _CLASS_COLORS[seg_clipped]

            # Top: segmentation coloured with intersection lines
            ax_top = axes[0, col_i]
            ax_top.imshow(rgb, interpolation="nearest")
            # Intersection lines as white overlay
            line_rgba = np.zeros((*seg.shape, 4))
            line_rgba[combined_mask > 0] = [1, 1, 1, 0.75]
            ax_top.imshow(line_rgba, interpolation="nearest")
            ax_top.set_title(f"frame {t}", fontsize=8)
            ax_top.axis("off")

            # Bottom: segmentation only
            ax_bot = axes[1, col_i]
            ax_bot.imshow(rgb, interpolation="nearest")
            ax_bot.axis("off")
            if col_i == 0:
                ax_top.set_ylabel("+ intersection\nlines", fontsize=7)
                ax_bot.set_ylabel("seg only", fontsize=7)

        legend_patches = [
            Patch(color=_CLASS_MPLCOLORS[c], label=_CLASS_NAMES[c])
            for c in range(4)
        ]
        legend_patches.append(Patch(color="white", label="SAx intersects"))
        fig.legend(handles=legend_patches, loc="lower center", ncol=5,
                   fontsize=8, frameon=False)
        plt.suptitle("4CH LAx segmentation used for LA conditioning", fontsize=10)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        seg_path = out_dir / "01_lax_segmentation.png"
        plt.savefig(seg_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  [LAx_conditioning] {seg_path.name}")

        # ── 3. Volume-over-frame curves from 4CH segmentation ─────────────────
        # We only have area (2-D), not true 3-D volume — label this correctly.
        areas = {c: np.zeros(T) for c in [1, 2, 3]}   # LV, MYO, RV
        for t in range(T):
            seg = lax_seg[t] if lax_seg.ndim == 3 else lax_seg[t, 0]
            for c in [1, 2, 3]:
                areas[c][t] = float((seg == c).sum()) * pixel_area_cm2

        fig, ax = plt.subplots(figsize=(7, 3.5))
        frames = np.arange(T)
        for c, label, color in [(1, "LV", _CLASS_MPLCOLORS[1]),
                                  (2, "MYO", _CLASS_MPLCOLORS[2]),
                                  (3, "RV", _CLASS_MPLCOLORS[3])]:
            ax.plot(frames, areas[c], color=color, linewidth=2, label=label,
                    marker=".", markersize=4)

        # Mark ED (frame 0) and ES (min LV area)
        es_frame = int(np.argmin(areas[1]))
        ax.axvline(0,         color="gray",  linewidth=1, linestyle="--", alpha=0.6,
                   label=f"ED (frame 0)")
        ax.axvline(es_frame,  color="black", linewidth=1, linestyle=":",  alpha=0.8,
                   label=f"ES (~frame {es_frame})")

        # Annotate EF if LV ED/ES areas are nonzero
        lv_ed = areas[1][0]
        lv_es = areas[1][es_frame]
        if lv_ed > 0:
            ef_approx = (lv_ed - lv_es) / lv_ed * 100
            ax.set_title(f"4CH segmentation area over cardiac frames  "
                         f"(approx. LV EF ≈ {ef_approx:.0f}%)", fontsize=9)
        else:
            ax.set_title("4CH segmentation area over cardiac frames", fontsize=9)

        ax.set_xlabel("Frame")
        ax.set_ylabel("Segmented area (cm²)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        vol_path = out_dir / "02_volume_curves.png"
        plt.savefig(vol_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  [LAx_conditioning] {vol_path.name}")

        # ── 4. LA vectors per-slice heatmap ───────────────────────────────────
        # Show label values as heatmap: rows = slices, cols = sample index.
        # One panel per frame (up to 4), plus a per-frame LV fraction bar.
        n_panels = min(4, T)
        panel_frames = np.round(np.linspace(0, T - 1, n_panels)).astype(int)

        fig = plt.figure(figsize=(5 * n_panels, max(3, Z * 0.35 + 2)))
        gs_outer = gridspec.GridSpec(1, n_panels, figure=fig, wspace=0.35)

        cmap = matplotlib.colors.ListedColormap(_CLASS_MPLCOLORS)
        norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)

        for p_i, t in enumerate(panel_frames):
            gs_inner = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs_outer[p_i],
                height_ratios=[Z * 0.9, 1], hspace=0.08
            )
            ax_hm = fig.add_subplot(gs_inner[0])
            ax_bar = fig.add_subplot(gs_inner[1])

            data = vecs_np[:, t, :]   # (Z, N)
            im = ax_hm.imshow(data, aspect="auto", cmap=cmap, norm=norm,
                              interpolation="nearest", origin="upper")
            ax_hm.set_title(f"frame {t}", fontsize=9)
            ax_hm.set_ylabel("SAx slice", fontsize=8)
            ax_hm.set_yticks(range(Z))
            ax_hm.set_yticklabels([str(z) for z in range(Z)], fontsize=6)
            ax_hm.set_xticks([])
            ax_hm.set_xlabel("Sample along intersection →", fontsize=7)

            # Bar: fraction of each class across all slices for this frame
            flat = data.ravel()
            fracs = [(flat == c).mean() for c in range(4)]
            left = 0.0
            for c in range(4):
                ax_bar.barh(0, fracs[c], left=left, color=_CLASS_MPLCOLORS[c],
                            height=0.8, label=_CLASS_NAMES[c])
                if fracs[c] > 0.05:
                    ax_bar.text(left + fracs[c] / 2, 0, f"{fracs[c]:.0%}",
                                ha="center", va="center", fontsize=7, color="white",
                                fontweight="bold")
                left += fracs[c]
            ax_bar.set_xlim(0, 1)
            ax_bar.set_yticks([])
            ax_bar.set_xticks([])

        # Shared colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
        cb = fig.colorbar(im, cax=cbar_ax, ticks=[0, 1, 2, 3])
        cb.ax.set_yticklabels(["BG", "LV", "MYO", "RV"], fontsize=8)
        cb.set_label("Label", fontsize=8)

        plt.suptitle("LA conditioning vectors per SAx slice (sampled along 4CH intersection)",
                     fontsize=10, y=1.01)
        vec_path = out_dir / "03_la_vectors_per_slice.png"
        plt.savefig(vec_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  [LAx_conditioning] {vec_path.name}")

    except Exception as exc:
        import traceback
        print(f"  [LAx_conditioning] WARNING: figure generation failed: {exc}")
        traceback.print_exc()
