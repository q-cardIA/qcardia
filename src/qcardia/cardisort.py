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
    raw_config = yaml.load(config_path.open(), Loader=yaml.FullLoader)
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


def _first_readable_dataset(sequence_dir: Path) -> pydicom.Dataset | None:
    for f in sorted(sequence_dir.iterdir()):
        if not f.is_file() or f.name.startswith("."):
            continue
        try:
            return pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
    return None


# Maximum acquisition_time_seconds gap, within a (FrameOfReferenceUID,
# ProtocolName) key, for two directories to be treated as reconstruction
# variants of the same acquisition rather than separate acquisitions (e.g.
# a genuine repeat) sharing that key. Real Siemens data: true reconstruction
# variants (magnitude/PSIR pairs, a perfusion acquisition's AIF/MOCO/LR/HR
# variants) were all within ~3s of each other; a real repeated "Aortic Flow"
# acquisition under the same protocol was ~58s later - this sits with a wide
# margin on both sides of that gap.
GROUP_TIME_WINDOW_SECONDS = 15


def _cluster_by_acquisition_time(members: list[tuple[Path, float | None]]) -> list[list[Path]]:
    """Splits directories sharing a (FrameOfReferenceUID, ProtocolName) key
    into clusters of near-simultaneous acquisition_time_seconds. A member
    with no timestamp is conservatively kept on its own, since proximity
    can't be confirmed."""
    with_time = sorted((m for m in members if m[1] is not None), key=lambda m: m[1])
    without_time = [m[0] for m in members if m[1] is None]

    clusters = []
    current: list[tuple[Path, float]] = []
    for d, t in with_time:
        if current and t - current[-1][1] > GROUP_TIME_WINDOW_SECONDS:
            clusters.append([m[0] for m in current])
            current = []
        current.append((d, t))
    if current:
        clusters.append([m[0] for m in current])

    clusters.extend([d] for d in without_time)
    return clusters


def group_reconstruction_variants(sequence_dirs: list[Path]) -> list[list[Path]]:
    """Groups sequence directories that are different reconstructions of the
    same underlying acquisition (e.g. a magnitude/PSIR pair, or a
    perfusion acquisition's AIF/AIF_MOCO/LR/HR_MOCO/AIF_SEG/MAP_PERF
    variants, each exported by some scanners - seen: Siemens - as separate
    directories) rather than genuinely different acquisitions needing
    disambiguation.

    Candidate directories are those whose first instance shares the same
    FrameOfReferenceUID AND ProtocolName - FrameOfReferenceUID alone isn't
    enough (other real acquisitions in the same exam without patient
    repositioning can share it too); ProtocolName is populated from the
    scanner's exam-card protocol selection and, unlike SeriesDescription,
    doesn't vary per reconstruction (no MAG/PSIR/LR/HR_MOCO-style suffix).
    Within that, _cluster_by_acquisition_time splits out any genuine repeat
    acquisition sharing the same key (see GROUP_TIME_WINDOW_SECONDS).

    Directories that can't be keyed this way (unreadable, or missing either
    tag - e.g. Philips data, which doesn't split reconstructions into
    separate directories in the first place) are returned as their own
    single-item group. Preserves the input order of first appearance.
    """
    keyed = []
    for d in sequence_dirs:
        ds = _first_readable_dataset(d)
        for_uid = getattr(ds, "FrameOfReferenceUID", None) if ds is not None else None
        protocol = getattr(ds, "ProtocolName", None) if ds is not None else None
        acq_time = get_acquisition_time_seconds(ds) if ds is not None else None
        key = (str(for_uid), str(protocol)) if for_uid and protocol else None
        keyed.append((d, key, acq_time))

    groups = []
    grouped = set()
    for i, (d, key, _) in enumerate(keyed):
        if d in grouped:
            continue
        if key is None:
            groups.append([d])
            grouped.add(d)
            continue
        same_key = [(d2, t2) for d2, k2, t2 in keyed[i:] if k2 == key and d2 not in grouped]
        for cluster in _cluster_by_acquisition_time(same_key):
            groups.append(cluster)
            grouped.update(cluster)
    return groups


def _count_files(sequence_dir: Path) -> int:
    return sum(1 for f in sequence_dir.iterdir() if f.is_file() and not f.name.startswith("."))


def pick_processing_representative(group: list[Path]) -> Path:
    """Picks which directory within an already-resolved reconstruction-
    variant group (see group_reconstruction_variants) to actually use
    downstream (e.g. to build a CineSeries/LGESeries from), as opposed to
    which one to classify (classify_sequence_group pools all of them).

    Prefers a non-magnitude reconstruction (mr_reconstruction_flag not "m")
    when exactly one member has it - e.g. PSIR over magnitude for LGE,
    matching what LGESeries._extract_psir looks for - else falls back to
    the member with the most files, which skips tiny derived products (a
    single-frame parametric map or segmentation overlay) in favor of the
    real image stack.
    """
    if len(group) == 1:
        return group[0]

    flags = {}
    for d in group:
        ds = _first_readable_dataset(d)
        image_type = list(getattr(ds, "ImageType", [])) if ds is not None else None
        flags[d] = mr_reconstruction_flag(image_type)
    non_magnitude = [d for d, flag in flags.items() if flag is not None and flag != "m"]
    if len(non_magnitude) == 1:
        return non_magnitude[0]

    return max(group, key=_count_files)


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
    datasets.sort(key=lambda ds: int(getattr(ds, "InstanceNumber", 0)))
    return datasets


def load_group_datasets(group: list[Path]) -> list[pydicom.Dataset]:
    """All DICOM datasets across every directory in a reconstruction-variant
    group (see group_reconstruction_variants), concatenated in group order -
    each directory's own datasets are already ordered by InstanceNumber via
    load_series_datasets. A tiny derived product (e.g. a single-frame
    parametric map) contributes proportionally few frames to the pooled
    list, so it doesn't need special-casing here."""
    datasets = []
    for sequence_dir in group:
        datasets.extend(load_series_datasets(sequence_dir))
    return datasets


def get_frame_pixel_spacing(ds: pydicom.Dataset, frame_index: int = 0) -> list:
    """PixelSpacing for a given frame. Single-frame instances carry it as a
    top-level attribute; multi-frame ("Enhanced") instances instead carry it
    per-frame inside PerFrameFunctionalGroupsSequence, or once for the whole
    series inside SharedFunctionalGroupsSequence."""
    if "PixelSpacing" in ds:
        return ds.PixelSpacing
    shared_groups = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if shared_groups:
        pixel_measures = getattr(shared_groups[0], "PixelMeasuresSequence", None)
        if pixel_measures:
            return pixel_measures[0].PixelSpacing
    per_frame_groups = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if per_frame_groups:
        pixel_measures = getattr(per_frame_groups[frame_index], "PixelMeasuresSequence", None)
        if pixel_measures:
            return pixel_measures[0].PixelSpacing
    raise AttributeError(f"No PixelSpacing found for {ds!r} frame {frame_index}")


def mr_reconstruction_flag(image_type: list | None) -> str | None:
    """The DICOM-standard MR reconstruction-type flag (ImageType's 3rd
    value: M=magnitude, P=phase, R=real, I=imaginary), lowercased to its
    first letter. Mirrors the check LGESeries._extract_psir (series.py)
    does per-instance within one series; used both to disambiguate across
    separate magnitude/PSIR directories (disambiguate.py, for scanners that
    export them as separate series) and to pick which reconstruction to use
    within an already-resolved reconstruction-variant group
    (pick_processing_representative)."""
    if not image_type or len(image_type) < 3:
        return None
    return str(image_type[2])[:1].lower()


def get_acquisition_time_seconds(ds: pydicom.Dataset) -> float | None:
    """Time-of-day (seconds since midnight) a series was acquired, used to
    order series within one exam and to tell reconstruction variants of one
    acquisition (near-identical timestamps) apart from a genuine repeat
    acquisition (seconds to minutes apart) sharing the same
    FrameOfReferenceUID/ProtocolName - see group_reconstruction_variants.

    Tries, in order: AcquisitionDateTime (a single DT field some enhanced/
    multi-frame instances carry at the top level even when the plain TM
    fields below are per-frame or absent - see get_frame_pixel_spacing for
    the same enhanced-instance problem), then AcquisitionTime, SeriesTime,
    ContentTime. Returns None, rather than guessing, if none are present at
    the top level - e.g. an enhanced instance with only per-frame timing in
    PerFrameFunctionalGroupsSequence, which isn't handled here since no real
    example has been seen yet."""
    # (tag, index where the time-of-day portion starts): AcquisitionDateTime
    # is DT (YYYYMMDDHHMMSS...), the others are TM (HHMMSS...).
    for tag, time_start in (
        ("AcquisitionDateTime", 8),
        ("AcquisitionTime", 0),
        ("SeriesTime", 0),
        ("ContentTime", 0),
    ):
        value = getattr(ds, tag, None)
        if not value:
            continue
        value = str(value).strip()
        for sign in ("+", "-"):  # strip a DT field's optional UTC offset suffix
            offset_idx = value.find(sign, time_start)
            if offset_idx != -1:
                value = value[:offset_idx]
        time_part = value[time_start:]
        try:
            hours, minutes = int(time_part[0:2]), int(time_part[2:4])
            seconds = float(time_part[4:]) if time_part[4:] else 0.0
            return hours * 3600 + minutes * 60 + seconds
        except (ValueError, IndexError):
            continue
    return None


def get_harmonized_pixel_arrays(
    datasets: list[pydicom.Dataset],
) -> tuple[list[np.ndarray], list[list]]:
    """2D pixel arrays and their PixelSpacing for every frame across all
    datasets, resizing any that don't match the series' dominant (most common)
    shape. Mirrors the reformat step in cardisort-v2's qcardia_data fork, which
    reconciles minor matrix-size differences between reconstruction types (e.g.
    magnitude vs. PSIR, or an interleaved scout) within a single series before
    frame selection. Multi-frame ("Enhanced") instances are flattened into
    their individual 2D frames so they mix in with single-frame instances."""
    pixel_arrays = []
    pixel_spacings = []
    for ds in datasets:
        array = ds.pixel_array
        if array.ndim == 3:
            for frame_index, frame in enumerate(array):
                pixel_arrays.append(frame)
                pixel_spacings.append(get_frame_pixel_spacing(ds, frame_index))
        else:
            pixel_arrays.append(array)
            pixel_spacings.append(get_frame_pixel_spacing(ds))

    dominant_shape = Counter(a.shape for a in pixel_arrays).most_common(1)[0][0]
    pixel_arrays = [
        a
        if a.shape == dominant_shape
        else skimage.transform.resize(
            a, dominant_shape, order=1, preserve_range=True, anti_aliasing=True
        ).astype(a.dtype)
        for a in pixel_arrays
    ]
    return pixel_arrays, pixel_spacings


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
    pixel_arrays, pixel_spacings = get_harmonized_pixel_arrays(datasets)
    frame_idxs = np.linspace(0, len(pixel_arrays) - 1, n_channels).round().astype(int)
    channels = np.stack(
        [pixel_arrays[idx].astype(np.float32) for idx in frame_idxs], axis=0
    )
    # (n_channels, 1, H, W): grid_sample batches over channels via the batch dim
    channels_tensor = torch.tensor(channels).unsqueeze(1)

    pixel_spacing = pixel_spacings[frame_idxs[0]]
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


def classify_datasets(
    datasets: list[pydicom.Dataset],
    model: nn.Module,
    n_channels: int,
    target_pixdim: tuple[float, float],
    target_size: tuple[int, int],
    grid_sample_mode: str,
    verbose: bool = False,
) -> tuple[str, str] | None:
    """Predicts the (sequence, plane) label pair from a list of DICOM
    datasets - either one sequence directory's own (classify_sequence_dir)
    or a reconstruction-variant group's pooled datasets
    (classify_sequence_group)."""
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


def classify_sequence_dir(
    sequence_dir: Path,
    model: nn.Module,
    n_channels: int,
    target_pixdim: tuple[float, float],
    target_size: tuple[int, int],
    grid_sample_mode: str,
    verbose: bool = False,
) -> tuple[str, str] | None:
    """Predicts the (sequence, plane) label pair for a single sequence
    directory."""
    return classify_datasets(
        load_series_datasets(sequence_dir),
        model, n_channels, target_pixdim, target_size, grid_sample_mode, verbose,
    )


def classify_sequence_group(
    group: list[Path],
    model: nn.Module,
    n_channels: int,
    target_pixdim: tuple[float, float],
    target_size: tuple[int, int],
    grid_sample_mode: str,
    verbose: bool = False,
) -> tuple[str, str] | None:
    """Predicts the (sequence, plane) label pair for a reconstruction-variant
    group as a whole (see group_reconstruction_variants): classification
    frames are drawn from across every member's pooled datasets, since
    they're physically one acquisition rather than distinct candidates."""
    return classify_datasets(
        load_group_datasets(group),
        model, n_channels, target_pixdim, target_size, grid_sample_mode, verbose,
    )
