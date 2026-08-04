"""Sequence and plane classification for raw cardiac MRI series directories.

Classifies each raw-sequence subdirectory of a patient folder (e.g. "REST",
"DBscar_SA") by MRI sequence type (e.g. CINE, DBLGE) and imaging plane (e.g.
SAX, 4CH), using the CardisortClassifier model. Label indices and their
ordering come from the training label CSV's column headers
("2021-10-25OptimisedSerDescMatchingMRIUpdated.csv" in cardisort-v2), split on
the first underscore into sequence/plane, in order of first appearance.
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
    20: "SSFPMRA", 21: "BOLUSTRACK", 22: "NoGroup",
    -1: "unlabelled",
}

PLANE_NAMES = {
    0: "SAX", 1: "2CH", 2: "3CH", 3: "4CH", 4: "LVOT2", 5: "RVOT",
    6: "RVOT2", 7: "AV", 8: "AX", 9: "ARCH", 10: "AORTA", 11: "MPA",
    12: "DA", 13: "PULV", 14: "MV", 15: "MP", 16: "SAG", 17: "COR",
    -1: "unlabelled",
}


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
        self.seq_head = nn.Linear(feature_size, config["model"]["n_sequence_classes"])
        self.plane_head = nn.Linear(feature_size, config["model"]["n_plane_classes"])

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
    """Pixel arrays for all datasets, resizing any that don't match the series'
    dominant (most common) shape. Mirrors the reformat step in cardisort-v2's
    qcardia_data fork, which reconciles minor matrix-size differences between
    reconstruction types (e.g. magnitude vs. PSIR, or an interleaved scout)
    within a single series before frame selection."""
    pixel_arrays = []
    for ds in datasets:
        try:
            pixel_arrays.append(ds.pixel_array)
        except Exception:
            # Corrupt file or unsupported transfer syntax — skip, don't fail the series.
            continue
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


def build_cardisort_input(
    datasets: list[pydicom.Dataset],
    n_channels: int,
    target_pixdim: tuple[float, float],
    target_size: tuple[int, int],
    grid_sample_mode: str,
) -> torch.Tensor:
    """Picks n_channels representative frames evenly spaced across the series,
    resamples them (pixel-spacing aware, like RandResample2Dd) to target_size at
    target_pixdim, and min-max normalizes the stack to [0, 1] (like
    NormalizeIntensityd), matching the CardisortClassifier's training-time
    preprocessing."""
    pixel_arrays = get_harmonized_pixel_arrays(datasets)
    frame_idxs = np.linspace(0, len(pixel_arrays) - 1, n_channels).round().astype(int)
    channels = np.stack(
        [pixel_arrays[idx].astype(np.float32) for idx in frame_idxs], axis=0
    )
    # (n_channels, 1, H, W): grid_sample batches over channels via the batch dim
    channels_tensor = torch.tensor(channels).unsqueeze(1)

    pixel_spacing = datasets[frame_idxs[0]].PixelSpacing
    source_size = torch.tensor(channels_tensor.shape[-2:], dtype=torch.float32)
    real_source_size = (
        torch.tensor([float(pixel_spacing[0]), float(pixel_spacing[1])]) * source_size
    )
    real_target_size = torch.tensor(target_pixdim, dtype=torch.float32) * torch.tensor(
        target_size, dtype=torch.float32
    )
    dimension_scale_factor = real_target_size / real_source_size

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

    resampled = (resampled - resampled.min()) / (resampled.max() - resampled.min() + 1e-6)
    return resampled.unsqueeze(0)  # (1, n_channels, target_h, target_w)


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
