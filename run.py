from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pydicom
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.animation import FuncAnimation, PillowWriter
from monai.networks.blocks import Warp
from natsort import natsorted
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import find_contours
from skimage.transform import warp
import yaml

import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb

import utils
import qcardia.utils as qcardia_utils
from qcardia.series import CineSeries

from qcardia_models.models.networks.encoder_mlp import EncoderMLP2d
from qcardia_models.utils import seed_everything

seed_everything(42)

MOTION_WANDB_RUN_PATH = Path.cwd() / "wandb" / "motion-model"
WANDB_RUN_PATH = Path.cwd() / "wandb" / "cine-seg"
PATH_TO_DATASET = Path.cwd() / "data"

patient_list = natsorted([f for f in PATH_TO_DATASET.iterdir() if f.is_dir()])

warp_layer = Warp()

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
            datasets.append(pydicom.dcmread(f))
        except Exception:
            continue
    datasets.sort(key=lambda ds: int(getattr(ds, "InstanceNumber", 0)))
    return datasets


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
    frame_idxs = np.linspace(0, len(datasets) - 1, n_channels).round().astype(int)
    channels = np.stack(
        [datasets[idx].pixel_array.astype(np.float32) for idx in frame_idxs], axis=0
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

    scale_t = qcardia_utils.t_2d_scale(dimension_scale_factor)
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
    """Predicts the (sequence, plane) label pair for a sequence directory."""
    datasets = load_series_datasets(sequence_dir)
    if not datasets:
        return None

    input_tensor = build_cardisort_input(
        datasets, n_channels, target_pixdim, target_size, grid_sample_mode
    )
    model.eval()
    with torch.no_grad():
        seq_logits, plane_logits = model(input_tensor)

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


config_path = Path("wandb") / "cardisort" / "files" / "config.yaml"
raw_config = yaml.load(config_path.open(), Loader=yaml.FullLoader)
config = {k: v["value"] for k, v in raw_config.items() if isinstance(v, dict) and "value" in v}

cardisort_model = CardisortClassifier(config)
model_weights_path = Path("wandb") / "cardisort" / "files" / "best_model.pt"
cardisort_model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))



for patient in patient_list[:]:
    sequence_dirs = get_sequence_dirs(patient)

    sequence_classifications = {}
    for sequence_dir in sequence_dirs:
        prediction = classify_sequence_dir(
            sequence_dir,
            cardisort_model,
            config["model"]["nr_input_channels"],
            tuple(config["data"]["target_pixdim"]),
            tuple(config["data"]["target_size"]),
            config["data"]["image_grid_sample_mode"],
            verbose=False,
        )
        if prediction is None:
            continue
        sequence_classifications[sequence_dir] = prediction

    cine_dirs = [
        sequence_dir
        for sequence_dir, (sequence_name, _) in sequence_classifications.items()
        if sequence_name == "CINE"
    ]
    print(cine_dirs)


