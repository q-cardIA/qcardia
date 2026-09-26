"""Sequence and plane classification for raw cardiac MRI series directories.

Classifies each raw-sequence subdirectory of a patient folder (e.g. "REST",
"DBscar_SA") by MRI sequence type (e.g. CINE, DBLGE) and imaging plane (e.g.
SAX, 4CH), using the CardisortClassifier model. Label indices and their
ordering come from the training label CSV's column headers
("2021-10-25OptimisedSerDescMatchingMRIUpdated.csv" in cardisort-v2), split on
the first underscore into sequence/plane, in order of first appearance. The
"NoGroup" column is not a class: cardisort-v2 labels those series -1
(unlabelled), and they are excluded from the loss during training.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pydicom
import skimage.transform
import torch
import torch.nn as nn
import yaml
from natsort import natsorted
from torch.nn import functional as F

import qcardia.utils as utils
from qcardia_models.models.networks.encoder_mlp import EncoderMLP2d

SEQUENCE_NAMES = {
    0: "MOLLI+", 1: "MOLLI-", 2: "SHMOLLI-", 3: "SHMOLLI+", 4: "CINE",
    5: "HASTE", 6: "PC", 7: "T2MAPPING", 8: "T2starMAPPING", 9: "EGE",
    10: "TIscout", 11: "DBLGE", 12: "WBLGE", 13: "B0map", 14: "TestPERF",
    15: "PERF", 16: "Scouts", 17: "T1TSE", 18: "MRA", 19: "SFFPMRA",
    20: "SSFPMRA", 21: "BOLUSTRACK",
    -1: "unlabelled",
}

PLANE_NAMES = {
    0: "SAX", 1: "2CH", 2: "3CH", 3: "4CH", 4: "LVOT2", 5: "RVOT",
    6: "RVOT2", 7: "AV", 8: "AX", 9: "ARCH", 10: "AORTA", 11: "MPA",
    12: "DA", 13: "PULV", 14: "MV", 15: "MP", 16: "SAG", 17: "COR",
    -1: "unlabelled",
}


# Frame positions, as fractions of the valid frames of a series, and the
# intensity above which a frame counts as an unexpected reconstruction. Both
# come from select_valid_image_slices in cardisort-v2's qcardia_data fork,
# which prepares the training images.
FRAME_LOCATIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
MAX_VALID_FRAME_INTENSITY = 7000.0


class CardisortClassifier(nn.Module):
    """Shared encoder backbone with separate sequence and plane classification heads."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.backbone = EncoderMLP2d(
            nr_input_channels=config["model"]["nr_input_channels"],
            encoder_channels_list=config["model"]["encoder_channels"],
            mlp_channels_list=config["model"]["mlp_channels"],
        )
        feature_size = config["model"]["mlp_channels"][-1]
        # The heads cover every label index the training CSV can produce, which
        # is why the config calls them maximums: the indices are non-contiguous,
        # so some outputs belong to classes absent from the training data.
        self.seq_head = nn.Linear(
            feature_size, config["model"]["max_nr_sequence_classes"]
        )
        self.plane_head = nn.Linear(
            feature_size, config["model"]["max_nr_plane_classes"]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.seq_head(features), self.plane_head(features)


def load_cardisort_model(wandb_run_path: Path) -> tuple[CardisortClassifier, dict]:
    """Loads a trained CardisortClassifier and its config from a WandB run directory."""
    config_path = wandb_run_path / "files" / "config.yaml"
    with config_path.open() as f:
        raw_config = yaml.safe_load(f) or {}
    config = {
        k: v["value"] for k, v in raw_config.items() if isinstance(v, dict) and "value" in v
    }

    n_channels = config["model"]["nr_input_channels"]
    if n_channels != len(FRAME_LOCATIONS):
        raise ValueError(
            f"Model expects {n_channels} input channels, but the frames are"
            f" picked at {len(FRAME_LOCATIONS)} positions (FRAME_LOCATIONS)"
        )

    model = CardisortClassifier(config)
    model_weights_path = wandb_run_path / "files" / "best_model.pt"
    model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
    return model, config


def get_sequence_dirs(patient: Path) -> list[Path]:
    """All raw-sequence subdirectories for a patient, excluding derived
    outputs (e.g. "*_segmentation") and hidden files (e.g. ".DS_Store")."""
    return natsorted(
        [
            f
            for f in patient.iterdir()
            if f.is_dir()
            and not f.name.startswith(".")
            and not f.name.endswith("_segmentation")
        ]
    )


# SOP Class UIDs for Secondary Capture Image Storage and its multi-frame
# variants: scanner-generated workspace screenshots, not acquired images.
# They carry no PixelSpacing and aren't a real sequence to classify.
SECONDARY_CAPTURE_SOP_CLASS_PREFIX = "1.2.840.10008.5.1.4.1.1.7"


def load_series_datasets(sequence_dir: Path) -> list[pydicom.Dataset]:
    """All DICOM datasets in a sequence directory, ordered by InstanceNumber."""
    files = [
        f
        for f in sequence_dir.rglob("*")
        if f.is_file() and not f.name.startswith(".")
    ]
    datasets = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
        except Exception:
            continue
        if str(getattr(ds, "SOPClassUID", "")).startswith(SECONDARY_CAPTURE_SOP_CLASS_PREFIX):
            continue
        datasets.append(ds)

    def instance_number(ds: pydicom.Dataset) -> int:
        try:
            return int(getattr(ds, "InstanceNumber", 0))
        except (TypeError, ValueError):
            return 0

    datasets.sort(key=instance_number)
    return datasets


def get_harmonized_pixel_arrays(datasets: list[pydicom.Dataset]) -> list[np.ndarray]:
    """Pixel arrays for all datasets, dropping non-2D frames (e.g. RGB
    thumbnails) and resizing any that don't match the series' dominant (most
    common) shape. Mirrors the reformat step in cardisort-v2's qcardia_data
    fork, which reconciles minor matrix-size differences between reconstruction
    types (e.g. magnitude vs. PSIR, or an interleaved scout) within a single
    series before frame selection."""
    pixel_arrays = []
    for ds in datasets:
        try:
            pixel_array = ds.pixel_array
        except Exception:
            # Corrupt file or unsupported transfer syntax — skip, don't fail the series.
            continue
        if pixel_array.ndim == 2:
            pixel_arrays.append(pixel_array)
    if not pixel_arrays:
        raise ValueError("No readable pixel data in series")

    dominant_shape = Counter(a.shape for a in pixel_arrays).most_common(1)[0][0]
    return [
        a
        if a.shape == dominant_shape
        else skimage.transform.resize(
            a, dominant_shape, order=1, preserve_range=True, anti_aliasing=True
        ).astype(a.dtype)
        for a in pixel_arrays
    ]


def select_valid_frames(pixel_arrays: list[np.ndarray]) -> list[np.ndarray]:
    """The frames the classifier reads, one per fraction in FRAME_LOCATIONS,
    taken from the frames that pass the validity check. Mirrors
    select_valid_image_slices in cardisort-v2's qcardia_data fork: a frame with
    negative or very large values comes from an unexpected reconstruction (e.g.
    a PSIR series exported alongside its magnitude images), and is left out. The
    same frame can be picked more than once for a short series."""
    valid_frames = [
        frame
        for frame in pixel_arrays
        if frame.min() >= 0.0 and frame.max() <= MAX_VALID_FRAME_INTENSITY
    ]
    if not valid_frames:
        raise ValueError("No valid frames in series")

    valid_idxs = np.arange(len(valid_frames))
    frame_idxs = [
        int(np.argmin(np.abs(valid_idxs - location * (len(valid_frames) - 1))))
        for location in FRAME_LOCATIONS
    ]
    return [valid_frames[idx] for idx in frame_idxs]


def build_cardisort_input(
    datasets: list[pydicom.Dataset],
    n_channels: int,
    target_pixdim: tuple[float, float],
    target_size: tuple[int, int],
    grid_sample_mode: str,
) -> torch.Tensor:
    """Picks n_channels representative frames across the series, resamples them
    (like RandResample2Dd) to target_size at target_pixdim, and standardizes
    each frame to zero mean and unit variance (like StandardizeIntensityd with
    reference_level "channel"), matching the CardisortClassifier's training-time
    preprocessing.

    The training images are NIfTI files written with an identity affine, so the
    pixel spacing of the source series plays no part in the resampling: only the
    matrix size does. The DICOM PixelSpacing is therefore ignored here as well.
    """
    pixel_arrays = get_harmonized_pixel_arrays(datasets)
    channels = np.stack(
        [frame.astype(np.float32) for frame in select_valid_frames(pixel_arrays)],
        axis=0,
    )
    # (n_channels, 1, H, W): grid_sample batches over channels via the batch dim
    channels_tensor = torch.tensor(channels).unsqueeze(1)

    source_size = torch.tensor(channels_tensor.shape[-2:], dtype=torch.float32)
    real_target_size = torch.tensor(target_pixdim, dtype=torch.float32) * torch.tensor(
        target_size, dtype=torch.float32
    )
    dimension_scale_factor = real_target_size / source_size

    scale_t = utils.t_2d_scale(dimension_scale_factor)
    grid = F.affine_grid(
        theta=torch.repeat_interleave(scale_t[:-1, :].unsqueeze(0), n_channels, dim=0),
        size=(n_channels, 1, target_size[0], target_size[1]),
        align_corners=False,
    )
    resampled = F.grid_sample(
        channels_tensor,
        grid,
        align_corners=False,
        mode=grid_sample_mode,
        padding_mode="zeros",
    ).squeeze(1)  # (n_channels, target_h, target_w)

    mean = resampled.mean(dim=(1, 2), keepdim=True)
    std = resampled.std(dim=(1, 2), keepdim=True)
    standardized = (resampled - mean) / std
    return standardized.unsqueeze(0)  # (1, n_channels, target_h, target_w)


def print_class_probs(head_name: str, probs: torch.Tensor, class_names: dict) -> None:
    """Prints every class's predicted probability for a classification head,
    sorted highest first."""
    ranked_idxs = torch.argsort(probs, descending=True).tolist()
    print(f"  {head_name} probabilities:")
    for idx in ranked_idxs:
        name = class_names.get(idx, f"class_{idx}")
        print(f"    {name:>16}: {probs[idx]:.4f}")


def classify_sequence_dir(
    sequence_dir: Path,
    model: nn.Module,
    n_channels: int,
    target_pixdim: tuple[float, float],
    target_size: tuple[int, int],
    grid_sample_mode: str,
    verbose: bool = False,
) -> tuple[str, str] | None:
    """Predicts the (sequence, plane) label pair for a sequence directory.
    Returns None (instead of raising) if the series can't be classified, so
    one malformed series doesn't abort a batch run over many patients."""
    try:
        datasets = load_series_datasets(sequence_dir)
        if not datasets:
            return None

        input_tensor = build_cardisort_input(
            datasets, n_channels, target_pixdim, target_size, grid_sample_mode
        )
        model.eval()
        with torch.no_grad():
            seq_logits, plane_logits = model(input_tensor)
    except Exception as e:
        if verbose:
            print(f"  Skipping {sequence_dir.name}: {e}")
        return None

    seq_probs = torch.softmax(seq_logits, dim=1).squeeze(0)
    plane_probs = torch.softmax(plane_logits, dim=1).squeeze(0)

    if verbose:
        print_class_probs("sequence", seq_probs, SEQUENCE_NAMES)
        print_class_probs("plane", plane_probs, PLANE_NAMES)

    seq_idx = int(torch.argmax(seq_probs))
    plane_idx = int(torch.argmax(plane_probs))
    return (
        SEQUENCE_NAMES.get(seq_idx, f"seq_class_{seq_idx}"),
        PLANE_NAMES.get(plane_idx, f"plane_class_{plane_idx}"),
    )
