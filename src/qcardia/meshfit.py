"""
Biventricular mesh fitting of qcardia cine segmentations with cmrpro.

cmrpro (https://gitlab.ethz.ch/ibt-cmr-public/cmrpro) fits a biventricular
shape model to segmentations it reads from a folder of VTI files:

    <seg_folder>/cine-sax_VTI/frame_XX.vti   (labels 1 LV, 2 Myo, 3 RV)
    <seg_folder>/cine-4ch_VTI/frame_XX.vti   (labels 1 LV, 2 Myo, 3 RV, 4 LA, 5 RA)
    <seg_folder>/cine-*_VTI/Img2World_matrix4x4.npy

Each VTI holds an `intensity` and a `segmentation` point array, and the .npy
affine maps voxel index (row, col, slice) to patient (LPS) coordinates in mm.
SAX and 4CH frames are paired by file name, so both views are written with
the same number of frames.

The qcardia 4CH model segments BG/LV/Myo/RV only. cmrpro needs LA (label 4)
for the mitral valve, which it takes as the part of the LV blood pool contour
that touches LA. The myocardium surrounds the LV blood pool except at the
mitral opening, so the background pixels that touch the LV blood pool are
written as label 4 (see `add_mitral_opening`). This band is not an LA
segmentation. Without RA, cmrpro estimates the tricuspid valve from the RV
contour, uses it for alignment only, and leaves out the TV fitting loss.

Usage:
    sax.predict_segmentation(...)
    lax_4ch.predict_segmentation(...)
    export_for_cmrpro(seg_folder, sax, lax_4ch)
    fit_mesh(seg_folder, mesh_folder, cmrpro_config_path)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import yaml
from scipy.ndimage import binary_dilation

from qcardia.series import CineSeries

LV_LABEL = 1
CMRPRO_LA_LABEL = 4


def series_affine(series: CineSeries) -> np.ndarray:
    """
    Build the voxel (row, col, slice) to LPS world affine of a series.

    The slice step is taken from the first and last slice positions, in the
    slice order of `series.slice_data`, so it follows the order of the arrays.

    Args:
        series (CineSeries): A loaded series.

    Returns:
        np.ndarray: The (4, 4) affine.
    """
    first = series.slice_data["slice01"]["meta_data"][0]
    orientation = np.array(first.ImageOrientationPatient, dtype=float)
    row_cosines, col_cosines = orientation[:3], orientation[3:]
    row_spacing, col_spacing = (float(x) for x in first.PixelSpacing)

    affine = np.eye(4)
    affine[:3, 0] = col_cosines * row_spacing  # moving down the rows
    affine[:3, 1] = row_cosines * col_spacing  # moving along the columns
    affine[:3, 3] = np.array(first.ImagePositionPatient, dtype=float)

    n_slices = series.number_of_slices
    if n_slices > 1:
        positions = np.array(
            [
                series.slice_data[f"slice{i+1:02}"]["meta_data"][0].ImagePositionPatient
                for i in range(n_slices)
            ],
            dtype=float,
        )
        steps = np.diff(positions, axis=0)
        if np.ptp(np.linalg.norm(steps, axis=1)) > 0.1:
            print(
                f"    Warning: uneven slice spacing in {series.folder}; the mesh "
                f"fit assumes a constant slice step."
            )
        affine[:3, 2] = (positions[-1] - positions[0]) / (n_slices - 1)
    else:
        normal = np.cross(row_cosines, col_cosines)
        affine[:3, 2] = normal * float(series._get_pixel_spacing()[2])

    return affine


def add_mitral_opening(segmentation: np.ndarray) -> np.ndarray:
    """
    Label the background pixels that touch the LV blood pool as LA.

    Args:
        segmentation (np.ndarray): BG/LV/Myo/RV labels, shape (..., H, W).

    Returns:
        np.ndarray: A copy with the mitral opening band set to label 4.
    """
    out = segmentation.copy()
    lv = segmentation == LV_LABEL
    # Dilate in-plane only, with 4-connectivity, so that a myocardium corner
    # does not let the band reach through to the LV.
    structure = np.zeros((1,) * (segmentation.ndim - 2) + (3, 3), dtype=bool)
    structure[..., 1, :] = True
    structure[..., :, 1] = True
    band = binary_dilation(lv, structure=structure) & (segmentation == 0)
    out[band] = CMRPRO_LA_LABEL
    return out


def resample_frames(array: np.ndarray, n_frames: int) -> np.ndarray:
    """
    Resample the frame axis to `n_frames` by nearest cardiac phase.

    Both series are assumed to cover one full cardiac cycle (retrospective
    gating), so frame t maps to the same fraction of the cycle.

    Args:
        array (np.ndarray): Shape (Z, T, H, W).
        n_frames (int): The number of output frames.

    Returns:
        np.ndarray: Shape (Z, n_frames, H, W).
    """
    n_source = array.shape[1]
    if n_source == n_frames:
        return array
    idx = np.round(np.arange(n_frames) * n_source / n_frames).astype(int) % n_source
    return array[:, idx]


def write_vti_frames(
    out_dir: Path, image: np.ndarray, segmentation: np.ndarray, affine: np.ndarray
) -> None:
    """
    Write one VTI per frame, and the affine, in the layout cmrpro reads.

    Args:
        out_dir (Path): The cine-<view>_VTI folder.
        image (np.ndarray): Intensities, shape (Z, T, H, W).
        segmentation (np.ndarray): Labels, shape (Z, T, H, W).
        affine (np.ndarray): The (4, 4) voxel (row, col, slice) to world affine.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.vti"):
        old.unlink()
    np.save(out_dir / "Img2World_matrix4x4.npy", affine)

    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    n_slices, n_frames, rows, columns = segmentation.shape
    for t in range(n_frames):
        grid = pv.ImageData(
            dimensions=(rows, columns, n_slices),
            spacing=spacing,
            origin=affine[:3, 3],
            direction_matrix=affine[:3, :3] / spacing,
        )
        # (Z, H, W) -> (H, W, Z), flattened in the Fortran order cmrpro reshapes with.
        grid.point_data["intensity"] = (
            image[:, t].transpose(1, 2, 0).astype(np.float32).ravel(order="F")
        )
        grid.point_data["segmentation"] = (
            segmentation[:, t].transpose(1, 2, 0).astype(np.uint8).ravel(order="F")
        )
        grid.save(out_dir / f"frame_{t:02d}.vti")


def export_for_cmrpro(
    seg_folder: Path, sax_series: CineSeries, lax_4ch_series: CineSeries
) -> None:
    """
    Write segmented SAX and 4CH cine series as cmrpro input.

    The 4CH series is resampled to the SAX number of frames, and its mitral
    opening is labelled as LA.

    Args:
        seg_folder (Path): The cmrpro segmentation folder.
        sax_series (CineSeries): SAX series after `predict_segmentation`.
        lax_4ch_series (CineSeries): 4CH series after `predict_segmentation`.
    """
    sax_image = sax_series._get_array()
    sax_seg = sax_series._segmentation_prediction
    lax_image = lax_4ch_series._get_array()
    lax_seg = lax_4ch_series._segmentation_prediction
    for series, image in ((sax_series, sax_image), (lax_4ch_series, lax_image)):
        # _get_array drops slices with too few frames, which would break the
        # slice order the affine is built from.
        if image.shape[0] != series.number_of_slices:
            raise ValueError(
                f"{series.folder}: {series.number_of_slices} slices loaded but "
                f"{image.shape[0]} kept; cannot build a consistent affine."
            )

    n_frames = sax_seg.shape[1]
    lax_image = resample_frames(lax_image, n_frames)
    lax_seg = add_mitral_opening(resample_frames(lax_seg, n_frames))

    seg_folder = Path(seg_folder)
    write_vti_frames(
        seg_folder / "cine-sax_VTI", sax_image, sax_seg, series_affine(sax_series)
    )
    write_vti_frames(
        seg_folder / "cine-4ch_VTI", lax_image, lax_seg, series_affine(lax_4ch_series)
    )


def fit_mesh(
    seg_folder: Path,
    mesh_folder: Path,
    config_path: Path,
    n_iterations: int = 300,
    save_plots: bool = False,
) -> None:
    """
    Fit the cmrpro biventricular mesh to every frame in `seg_folder`.

    cmrpro must be importable (it has no package metadata, so add its source
    checkout to sys.path). Meshes are saved to <mesh_folder>/meshes/frame_XX.vtk
    in patient (LPS) coordinates.

    Args:
        seg_folder (Path): The folder written by `export_for_cmrpro`.
        mesh_folder (Path): The output folder.
        config_path (Path): A cmrpro config.yaml, for its `mesh_model` and
            `mesh_fitting_params` sections.
        n_iterations (int): Optimisation iterations per frame.
        save_plots (bool): Save segmentation/mesh overlay images per frame.
    """
    from cmrpro.meshfitter import MeshModel

    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    mesh_model = MeshModel(str(seg_folder), str(mesh_folder), **config["mesh_model"])
    mesh_model.set_regularization_weights(config["mesh_fitting_params"])
    mesh_model.fit(use_warp=True, n_iterations=n_iterations, save_plots=save_plots)
