from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import yaml
from qcardia.cardisort import (
    classify_sequence_dir,
    get_sequence_dirs,
    load_cardisort_model,
    load_series_datasets,
)
from qcardia.disambiguate import disambiguate_duplicates, summarize_sequence_dir
from qcardia.series import BaseSeries, CineSeries, LGESeries

from utils.app_config import DISAMBIGUATION_CACHE_DIR
from view.volume_geometry import VolumeGeometry

CARDISORT_SEQUENCE_TO_PREFIX = {"CINE": "CINE", "DBLGE": "LGE", "WBLGE": "LGE"}
CARDISORT_SUPPORTED_PLANES = {"SAX", "2CH", "3CH", "4CH"}


@dataclass
class SequenceDescriptor:
    path: Path
    sequence_name: str | None
    plane_name: str | None
    key: str | None
    is_primary: bool = False


@dataclass
class PatientLoadResult:
    loaded: dict[str, dict]
    entries: list[SequenceDescriptor]


def read_disk_thumbnail(seq_dir: Path, prefer_first_file: bool) -> np.ndarray | None:
    """Representative 2D uint8 slice read directly from disk, for series never loaded as a volume."""
    dcm_files = sorted(
        f for f in seq_dir.rglob("*") if f.is_file() and not f.name.startswith(".")
    )
    if not dcm_files:
        return None
    pick = dcm_files[0] if prefer_first_file else dcm_files[len(dcm_files) // 2]
    try:
        ds = pydicom.dcmread(str(pick))
        arr = ds.pixel_array.astype(np.float32)
    except Exception:
        return None

    if arr.ndim == 3:  # multi-frame DICOM: take the middle frame
        arr = arr[arr.shape[0] // 2]

    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + intercept  # window tags reference rescaled values, so this must come first

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr  # MONOCHROME1: stored values are inverted

    try:
        wc_raw, ww_raw = getattr(ds, "WindowCenter", None), getattr(ds, "WindowWidth", None)
        if wc_raw is None or ww_raw is None:
            raise ValueError
        wc = float(wc_raw[0]) if hasattr(wc_raw, "__len__") else float(wc_raw)
        ww = float(ww_raw[0]) if hasattr(ww_raw, "__len__") else float(ww_raw)
        if ww <= 0:
            raise ValueError
        lo = wc - ww / 2.0
        arr = np.clip(arr, lo, lo + ww)
        arr = (arr - lo) / ww * 255.0
    except Exception:
        p_lo, p_hi = np.percentile(arr, 2), np.percentile(arr, 98)
        if p_hi > p_lo:
            arr = np.clip(arr, p_lo, p_hi)
            arr = (arr - p_lo) / (p_hi - p_lo) * 255.0
        else:
            arr = np.zeros_like(arr)

    return arr.astype(np.uint8)


class DICOMModel:
    def __init__(self):
        self.series: dict[str, BaseSeries] = {}
        self.folder_path = None
        self.temporal_volumes = []
        self.temporal_segmentations = []
        self.num_time_frames = 0
        self.slice_positions = []
        self._cardisort_model = None
        self._cardisort_config = None
        self.use_cardisort_cache = True  # welcome-screen toggles, see WelcomeWidget / main.py
        self.use_disambiguation_cache = True
        self._load_segmentation_model()

    def _load_segmentation_model(self):
        wandb_path = Path.cwd() / "wandb"
        # CINE_4CH doubles as the LAx model for CINE_2CH/CINE_3CH until chamber-specific models exist.
        self._seg_wandb_paths = {
            "CINE_SAX": wandb_path / "CINE_SAX",
            "CINE_2CH": wandb_path / "CINE_4CH",
            "CINE_3CH": wandb_path / "CINE_4CH",
            "CINE_4CH": wandb_path / "CINE_4CH",
        }
        self.lge_wandb_run_path = wandb_path / "LGE_SAX"

        # Only feeds a best-effort shape heuristic in predict_segmentation; falls back to {}.
        self.config = {}
        cine_sax_path = self._seg_wandb_paths["CINE_SAX"]
        for config_path in (
            cine_sax_path / "config-copy.yaml",
            cine_sax_path / "config.yaml",
            cine_sax_path / "files" / "config-copy.yaml",
            cine_sax_path / "files" / "config.yaml",
        ):
            if config_path.exists():
                try:
                    self.config = yaml.load(Path.open(config_path), Loader=yaml.FullLoader)
                except Exception:
                    logging.warning(f"Failed to parse {config_path}")
                break
        else:
            logging.warning(f"No config file found under {cine_sax_path}")

    def load_cine_series(self, folder_path, series_key: str):
        self.series[series_key] = CineSeries(Path(folder_path), batch_size=50)
        if series_key == "CINE_SAX":
            self.folder_path = Path(folder_path)
        return self._process_temporal_volumes(series_key)

    def load_lge_series(self, folder_path, series_key: str):
        self.series[series_key] = LGESeries(Path(folder_path))
        return self._build_lge_volume(series_key)

    def _build_lge_volume(self, series_key: str):
        lge = self.series.get(series_key)
        if lge is None:
            raise ValueError("No LGE data loaded")

        slice_entries = []
        for idx in range(lge.number_of_slices):
            key = f"slice{idx+1:02}"
            slice_info = lge.slice_data.get(key)
            if not slice_info:
                continue
            try:
                image = self._extract_lge_slice_image(slice_info)
            except Exception:
                continue
            position = float(slice_info.get("slice_position", idx))
            slice_entries.append((position, image))

        if not slice_entries:
            raise ValueError("LGE stack contains no readable slices")

        slice_entries.sort(key=lambda item: item[0])
        slice_positions = [float(pos) for pos, _ in slice_entries]
        volume = np.stack([entry[1] for entry in slice_entries], axis=0)
        normalized = self._normalize_lge_volume(volume)
        return normalized, slice_positions

    @staticmethod
    def _extract_lge_slice_image(slice_info):
        image = slice_info.get("psir_array")
        if image is None:
            pixel_array = slice_info.get("pixel_array")
            if isinstance(pixel_array, list):
                if not pixel_array:
                    raise ValueError("Empty pixel array for slice")
                image = pixel_array[0]
            else:
                image = pixel_array
        return np.asarray(image, dtype=np.float32)

    @staticmethod
    def _normalize_lge_volume(volume):
        if volume.size == 0:
            raise ValueError("Empty LGE volume")

        lower = np.percentile(volume, 1)
        upper = np.percentile(volume, 99)
        if np.isclose(upper, lower):
            upper = lower + 1.0
        clipped = np.clip(volume, lower, upper)
        min_val = clipped.min()
        max_val = clipped.max()
        denom = max(max_val - min_val, 1e-6)
        scaled = (clipped - min_val) / denom
        return (scaled * 255.0).astype(np.uint8)

    def predict_lge_segmentation(self):
        if "LGE_SAX" not in self.series:
            raise ValueError("No LGE data loaded")

        segmentation = self.series["LGE_SAX"].predict_segmentation(self.lge_wandb_run_path)
        seg_arr = np.asarray(segmentation)

        if seg_arr.ndim == 4:
            seg_arr = seg_arr[:, 0, :, :]
        seg_arr = np.squeeze(seg_arr)

        if seg_arr.ndim != 3:
            raise ValueError(
                f"Unexpected LGE segmentation shape {seg_arr.shape}; expected (Z,H,W)"
            )

        # qcardia sorts slices descending by position; _build_lge_volume sorts ascending,
        # so reverse here to align segmentation indexing with the display volume.
        return seg_arr[::-1].astype(np.uint8)

    def apply_window(self, image, center, width):
        min_val = center - width / 2
        max_val = center + width / 2
        windowed = np.clip(image, min_val, max_val)
        windowed = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return windowed

    def _process_temporal_volumes(self, series_key: str):
        cine = self.series.get(series_key)
        if cine is None:
            raise ValueError("No DICOM data loaded")

        center = float(cine.slice_data["slice01"]["meta_data"][0].WindowCenter)
        width = float(cine.slice_data["slice01"]["meta_data"][0].WindowWidth)

        first_key = sorted(cine.slice_data.keys())[0]
        num_time_frames = len(cine.slice_data[first_key]["pixel_array"])

        slice_entries = []
        for idx, key in enumerate(sorted(cine.slice_data.keys())):
            slice_meta = cine.slice_data[key]["meta_data"][0]
            position = self._extract_slice_position(slice_meta, idx)
            slice_entries.append((position, cine.slice_data[key]["pixel_array"]))

        slice_entries.sort(key=lambda item: item[0])
        slice_positions = [float(entry[0]) for entry in slice_entries]

        temporal_volumes = []
        for time_idx in range(num_time_frames):
            slices = [entry[1][time_idx] for entry in slice_entries]
            image_stack = np.stack(slices, axis=0)
            temporal_volumes.append(self.apply_window(image_stack, center, width))

        if series_key == "CINE_SAX":
            self.num_time_frames = num_time_frames
            self.slice_positions = slice_positions
            self.temporal_volumes = temporal_volumes

        volume_4d = np.stack(temporal_volumes, axis=0)
        return volume_4d, num_time_frames

    @staticmethod
    def _extract_slice_position(meta, fallback_index):
        try:
            ipp = np.array(meta.ImagePositionPatient, dtype=float)
            iop = np.array(meta.ImageOrientationPatient, dtype=float)
            row = iop[:3]
            col = iop[3:]
            normal = np.cross(row, col)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm
                return float(np.dot(ipp, normal))
        except Exception:
            pass

        for attr in ("SliceLocation", "InstanceNumber"):
            if hasattr(meta, attr):
                try:
                    return float(getattr(meta, attr))
                except Exception:
                    continue

        return float(fallback_index)

    def predict_segmentation(self, series_key: str = "CINE_SAX"):
        cine = self.series.get(series_key)
        if cine is None:
            raise ValueError(f"No {series_key} DICOM data loaded")

        wandb_run_path = self._seg_wandb_paths[series_key]
        cine_segmentation = cine.predict_segmentation(wandb_run_path)
        arr = np.asarray(cine_segmentation)
        dims = arr.shape

        nslices = len(sorted(cine.slice_data.keys()))
        tframes = self.num_time_frames if self.num_time_frames else None

        scale_axis = None
        try:
            n_scales = int(self.config["unet"]["nr_output_scales"])
            if n_scales in dims:
                scale_axis = dims.index(n_scales)
        except Exception:
            scale_axis = None

        def extract_ZHW_at_time(time_idx: int) -> np.ndarray:
            x = arr
            if scale_axis is not None and 0 <= scale_axis < x.ndim:
                sel = 1 if dims[scale_axis] > 1 else 0
                x = np.take(x, sel, axis=scale_axis)

            d = x.shape
            time_axis = None
            slice_axis = None

            if tframes is not None and tframes in d:
                time_axis = d.index(tframes)
            if nslices in d:
                candidates = [i for i, s in enumerate(d) if s == nslices]
                if len(candidates) == 1:
                    slice_axis = candidates[0]
                else:
                    slice_axis = next(
                        (i for i in candidates if i != time_axis), candidates[0]
                    )

            if time_axis is None and d[0] == tframes:
                time_axis = 0
            if slice_axis is None:
                if d[0] == nslices:
                    slice_axis = 0
                elif d[1] == nslices:
                    slice_axis = 1

            if time_axis is not None and 0 <= time_idx < d[time_axis]:
                x = np.take(x, time_idx, axis=time_axis)

            d = x.shape
            if slice_axis is not None and slice_axis < x.ndim:
                if slice_axis != 0:
                    axes = list(range(x.ndim))
                    axes[0], axes[slice_axis] = axes[slice_axis], axes[0]
                    x = np.transpose(x, axes)

            if x.ndim == 3:
                result = x
            elif x.ndim == 2:
                result = x[np.newaxis, :, :]
            else:
                result = x.reshape(nslices, -1, x.shape[-1])

            # qcardia sorts slices descending by position; _process_temporal_volumes
            # sorts ascending, so reverse here to align segmentation indexing with
            # the display volume (see predict_lge_segmentation for the same fix).
            return result[::-1]

        self.temporal_segmentations = []
        if tframes and tframes > 1:
            for t in range(tframes):
                try:
                    vol = extract_ZHW_at_time(t)
                    self.temporal_segmentations.append(vol)
                except Exception as e:
                    logging.warning(f"Failed to extract segmentation for time {t}: {e}")

        if not self.temporal_segmentations:
            try:
                seg_volume = extract_ZHW_at_time(0)
                if self.num_time_frames and self.num_time_frames > 1:
                    self.temporal_segmentations = [seg_volume] * self.num_time_frames
                else:
                    self.temporal_segmentations = [seg_volume]
            except Exception as e:
                logging.error(f"Failed to extract any segmentation: {e}")
                self.temporal_segmentations = []

        return self.temporal_segmentations

    def get_geometry(self, series_key: str) -> VolumeGeometry | None:
        series = self.series.get(series_key)
        if series is None:
            return None
        try:
            slice_entries = []
            for idx, key in enumerate(sorted(series.slice_data.keys())):
                meta = series.slice_data[key]["meta_data"][0]
                position = self._extract_slice_position(meta, idx)
                slice_entries.append((position, meta))
            slice_entries.sort(key=lambda x: x[0])
            return VolumeGeometry.from_dicom_slices([m for _, m in slice_entries])
        except Exception:
            return None

    def save_segmentation(self, output_path=None):
        cine = self.series.get("CINE_SAX")
        if cine is None:
            raise ValueError("No DICOM data loaded")

        if not hasattr(cine, "_segmentation_prediction"):
            raise ValueError("No segmentation to save")

        if output_path is None:
            output_path = self.folder_path.parent / "results" / self.folder_path.name

        cine.save_predictions(Path(output_path))
        return output_path

    def _get_cardisort_model(self):
        if self._cardisort_model is None:
            cardisort_wandb_path = Path.cwd() / "wandb" / "cardisort"
            self._cardisort_model, self._cardisort_config = load_cardisort_model(
                cardisort_wandb_path
            )
        return self._cardisort_model, self._cardisort_config

    def _describe_patient_folder(self, folder_path) -> list[SequenceDescriptor]:
        folder_path = Path(folder_path)

        if self.use_cardisort_cache:
            cardisorted_target = self._resolve_cardisorted_target(folder_path)
            if cardisorted_target is not None:
                return self._describe_cardisorted_folder(cardisorted_target)

        model, config = self._get_cardisort_model()
        sequence_dirs = get_sequence_dirs(folder_path)

        descriptors = []
        for seq_dir in sequence_dirs:
            prediction = classify_sequence_dir(
                seq_dir,
                model,
                config["model"]["nr_input_channels"],
                tuple(config["data"]["target_pixdim"]),
                tuple(config["data"]["target_size"]),
                config["data"]["image_grid_sample_mode"],
            )
            if prediction is None:
                descriptors.append(SequenceDescriptor(seq_dir, None, None, None))
                continue

            sequence_name, plane_name = prediction
            prefix = CARDISORT_SEQUENCE_TO_PREFIX.get(sequence_name)
            key = (
                f"{prefix}_{plane_name}"
                if prefix is not None and plane_name in CARDISORT_SUPPORTED_PLANES
                else None
            )
            descriptors.append(SequenceDescriptor(seq_dir, sequence_name, plane_name, key))

        return descriptors

    def _resolve_cardisorted_target(self, folder_path: Path) -> Path | None:
        """Dev/runtime speedup: if folder_path already IS cardisort's own output
        layout, or a 'CardiSorted_<name>' sibling of it already exists on disk,
        use that directly instead of re-running cardisort classification (slow
        — one model inference per sequence dir — and redundant if it's already
        been done once for this subject). Gated by self.use_cardisort_cache."""
        if self._is_cardisorted_layout(folder_path):
            return folder_path
        sibling = folder_path.parent / f"CardiSorted_{folder_path.name}"
        if self._is_cardisorted_layout(sibling):
            return sibling
        return None

    @staticmethod
    def _is_cardisorted_layout(folder_path: Path) -> bool:
        """True if folder_path has top-level subfolders already named in
        cardisort's own '{SEQUENCE}_{PLANE}' output convention (e.g. CINE_SAX,
        WBLGE_4CH) — the unambiguous signature of a CardiSorted_* folder."""
        if not folder_path.is_dir():
            return False
        return any(
            d.is_dir() and d.name.split("_", 1)[0] in CARDISORT_SEQUENCE_TO_PREFIX
            for d in folder_path.iterdir()
        )

    @staticmethod
    def _describe_cardisorted_folder(folder_path: Path) -> list[SequenceDescriptor]:
        """Reconstructs SequenceDescriptors straight from an already-cardisorted
        folder's structure — no model inference. Candidate disambiguation among
        same-key duplicates (e.g. 'scar_4CH' vs 'DB scar_4CH') still runs
        normally afterward via _group_and_resolve, unchanged."""
        descriptors = []
        for top_dir in sorted(d for d in folder_path.iterdir() if d.is_dir()):
            sequence_name, _, plane_name = top_dir.name.partition("_")
            prefix = CARDISORT_SEQUENCE_TO_PREFIX.get(sequence_name)
            key = (
                f"{prefix}_{plane_name}"
                if prefix is not None and plane_name in CARDISORT_SUPPORTED_PLANES
                else None
            )
            for leaf_dir in sorted(d for d in top_dir.iterdir() if d.is_dir()):
                descriptors.append(SequenceDescriptor(leaf_dir, sequence_name, plane_name, key))
        return descriptors

    def _group_and_resolve(self, descriptors: list[SequenceDescriptor], folder_path) -> dict[str, Path]:
        dirs_by_key: dict[str, list[SequenceDescriptor]] = defaultdict(list)
        for d in descriptors:
            if d.key is not None:
                dirs_by_key[d.key].append(d)

        classified: dict[str, Path] = {}
        for key, group in dirs_by_key.items():
            primary = self._resolve_primary_dir(key, [d.path for d in group], folder_path)
            classified[key] = primary
            for d in group:
                d.is_primary = d.path == primary

        return classified

    def classify_patient_folder(self, folder_path) -> dict[str, Path]:
        return self._group_and_resolve(self._describe_patient_folder(folder_path), folder_path)

    # -- duplicate-series disambiguation cache, gated by self.use_disambiguation_cache --

    def _cache_subject_key(self, folder_path) -> str:
        """Identifies a subject for cache-file naming: the resolved
        CardiSorted folder's name if one exists (stable regardless of
        whether callers pass the raw QLGE folder or its CardiSorted_
        sibling), else the folder passed in directly."""
        target = self._resolve_cardisorted_target(Path(folder_path))
        return (target or Path(folder_path)).name

    @staticmethod
    def _cache_path(subject_key: str) -> Path:
        return DISAMBIGUATION_CACHE_DIR / f"{subject_key}.json"

    def _read_cached_primary(self, subject_key: str, key: str, signature: list[str]) -> str | None:
        path = self._cache_path(subject_key)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            return None
        entry = cache.get(key)
        if entry is None or entry.get("candidates") != signature:
            return None  # stale (candidate dirs changed) or never cached for this key
        return entry.get("primary")

    def _write_cached_primary(self, subject_key: str, key: str, signature: list[str], primary_name: str):
        path = self._cache_path(subject_key)
        try:
            DISAMBIGUATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache = {}
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    cache = json.load(f)
            cache[key] = {"candidates": signature, "primary": primary_name}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to write disambiguation cache for {subject_key}/{key}: {e}")

    def _resolve_primary_dir(self, key: str, dirs: list[Path], folder_path) -> Path:
        if len(dirs) == 1:
            return dirs[0]

        subject_key = self._cache_subject_key(folder_path)
        signature = sorted(d.name for d in dirs)

        if self.use_disambiguation_cache:
            cached_name = self._read_cached_primary(subject_key, key, signature)
            if cached_name is not None:
                cached = next((d for d in dirs if d.name == cached_name), None)
                if cached is not None:
                    return cached
                logging.warning(f"Disambiguation cache for {subject_key}/{key} pointed at a missing dir, recomputing")

        prefix, _, plane_name = key.partition("_")
        try:
            candidates = [
                summarize_sequence_dir(d, load_series_datasets(d)) for d in dirs
            ]
            result = disambiguate_duplicates((prefix, plane_name), candidates)
            primary = next(d for d in dirs if d.name == result["primary_series"])
        except Exception as e:
            logging.warning(
                f"Disambiguation failed for {key} ({[d.name for d in dirs]}): {e}; "
                "falling back to largest series"
            )
            primary = max(dirs, key=lambda d: sum(1 for f in d.rglob("*") if f.is_file()))

        if self.use_disambiguation_cache:
            self._write_cached_primary(subject_key, key, signature, primary.name)
        return primary

    def load_series_for_key(self, seq_dir: Path, key: str) -> dict:
        if key.startswith("CINE_"):
            volume, num_frames = self.load_cine_series(seq_dir, key)
            info = {"volume": volume, "is_temporal": True, "num_frames": num_frames}
        else:
            volume, _slice_positions = self.load_lge_series(seq_dir, key)
            info = {"volume": volume, "is_temporal": False, "num_frames": None}
        info["geometry"] = self.get_geometry(key)
        return info

    def load_patient(self, folder_path) -> PatientLoadResult:
        descriptors = self._describe_patient_folder(folder_path)
        classified = self._group_and_resolve(descriptors, folder_path)

        loaded: dict[str, dict] = {}
        for key, seq_dir in classified.items():
            try:
                loaded[key] = self.load_series_for_key(seq_dir, key)
            except Exception as e:
                logging.warning(f"Failed to load {key} from {seq_dir}: {e}")

        return PatientLoadResult(loaded=loaded, entries=descriptors)
