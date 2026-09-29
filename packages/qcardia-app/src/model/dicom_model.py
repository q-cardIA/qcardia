from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
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

from model.subject_labels import (
    SeriesLabel,
    SubjectLabels,
    load_labels,
    save_labels,
    update_primary,
)
from utils.windowing import dicom_window, percentile_window, window_to_uint8
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

    window = dicom_window(ds) or percentile_window(arr, 2, 98)
    return window_to_uint8(arr, *window) if window else np.zeros(arr.shape, np.uint8)


def _model_name(run: Path) -> str:
    """'project/name' from a wandb run's config, else the run folder name."""
    try:
        with open(run / "files" / "config-copy.yaml") as f:
            exp = yaml.load(f, Loader=yaml.FullLoader)["experiment"]
        return f"{exp['project']}/{exp['name']}"
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return run.name


class DICOMModel:
    def __init__(self):
        self.series: dict[str, BaseSeries] = {}
        self.folder_path = None
        self.num_time_frames = 0
        self.slice_positions = []
        self._cardisort_model = None
        self._cardisort_config = None
        self.use_cardisort_cache = True  # welcome-screen toggles, see WelcomeWidget / main.py
        self.use_segmentation_cache = True
        self.subject_folder: Path | None = None
        self._series_dirs: dict[str, Path] = {}
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
        self.cine_modalities = tuple(self._seg_wandb_paths)
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
        # 10, not 50: same output, ~1.5 GB less memory for CINE_SAX, no slower.
        self.series[series_key] = CineSeries(Path(folder_path), batch_size=10)
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
        return volume, slice_positions

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

    def _process_temporal_volumes(self, series_key: str):
        cine = self.series.get(series_key)
        if cine is None:
            raise ValueError("No DICOM data loaded")

        first_key = sorted(cine.slice_data.keys())[0]
        num_time_frames = len(cine.slice_data[first_key]["pixel_array"])

        slice_entries = []
        for idx, key in enumerate(sorted(cine.slice_data.keys())):
            slice_meta = cine.slice_data[key]["meta_data"][0]
            position = self._extract_slice_position(slice_meta, idx)
            slice_entries.append((position, cine.slice_data[key]["pixel_array"]))

        slice_entries.sort(key=lambda item: item[0])
        slice_positions = [float(entry[0]) for entry in slice_entries]

        volume_4d = np.stack(
            [np.stack([entry[1][t] for entry in slice_entries], axis=0) for t in range(num_time_frames)]
        )
        if series_key == "CINE_SAX":
            self.num_time_frames = num_time_frames
            self.slice_positions = slice_positions
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

        results = []
        if tframes and tframes > 1:
            for t in range(tframes):
                try:
                    results.append(extract_ZHW_at_time(t))
                except Exception as e:
                    logging.warning(f"Failed to extract segmentation for time {t}: {e}")

        if not results:
            try:
                seg_volume = extract_ZHW_at_time(0)
                results = [seg_volume] * self.num_time_frames if self.num_time_frames and self.num_time_frames > 1 else [seg_volume]
            except Exception as e:
                logging.error(f"Failed to extract any segmentation: {e}")
                results = []

        return results

    def segment(self, series_key: str) -> np.ndarray:
        """From the subject's .segmentation_cache if made from the same series and weights, else from the model."""
        run = self.lge_wandb_run_path if series_key == "LGE_SAX" else self._seg_wandb_paths[series_key]
        weights = run / "files" / "last_model.pt"
        source = {
            "series": self._series_dirs[series_key].name,
            "weights": weights.relative_to(run.parent.parent).as_posix(),
            "weights_modified": datetime.fromtimestamp(weights.stat().st_mtime).isoformat(timespec="seconds"),
        }
        folder = self.subject_folder / ".segmentation_cache" / series_key
        if self.use_segmentation_cache and (folder / "info.json").exists():
            info = json.loads((folder / "info.json").read_text())
            if all(info.get(k) == v for k, v in source.items()):
                with np.load(folder / "segmentation.npz") as cached:
                    return cached["seg"]

        if series_key == "LGE_SAX":
            seg = self.predict_lge_segmentation()
        else:
            frames = self.predict_segmentation(series_key)
            if not frames:
                raise ValueError(f"The {series_key} segmentation model produced no usable output; see the log.")
            seg = np.stack(frames).astype(np.uint8)
        if self.use_segmentation_cache:
            info = {"model": _model_name(run), **source,
                    "created": datetime.now().isoformat(timespec="seconds"), "manual_edits": []}
            try:
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / "segmentation.npz", seg=seg)
                (folder / "info.json").write_text(json.dumps(info, indent=2))
            except OSError as e:
                logging.warning(f"Could not cache the {series_key} segmentation: {e}")
        return seg

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
            cardisort_wandb_path = Path.cwd() / "wandb" / "new_cardisort"
            self._cardisort_model, self._cardisort_config = load_cardisort_model(
                cardisort_wandb_path
            )
        return self._cardisort_model, self._cardisort_config

    def _describe_patient_folder(self, folder_path) -> tuple[list[SequenceDescriptor], bool]:
        folder_path = Path(folder_path)
        if self.use_cardisort_cache:
            cached = load_labels(folder_path)
            if cached is not None:
                descriptors = [
                    SequenceDescriptor(folder_path / s.dir, s.sequence_name, s.plane_name, s.key)
                    for s in cached.series
                ]
                return descriptors, True
        return self._classify_live(folder_path), False

    def _classify_live(self, folder_path: Path) -> list[SequenceDescriptor]:
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

    def _group_and_resolve(
        self, descriptors: list[SequenceDescriptor], folder_path, ignore_cache: bool = False
    ) -> tuple[dict[str, Path], dict[str, dict]]:
        dirs_by_key: dict[str, list[SequenceDescriptor]] = defaultdict(list)
        for d in descriptors:
            if d.key is not None:
                dirs_by_key[d.key].append(d)

        use_cache = self.use_cardisort_cache and not ignore_cache
        cached = load_labels(folder_path) if use_cache else None
        primary_by_key = cached.primary_by_key if cached is not None else {}

        classified: dict[str, Path] = {}
        resolved: dict[str, dict] = {}
        for key, group in dirs_by_key.items():
            dirs = [d.path for d in group]
            primary, confident = self._resolve_primary_dir(key, dirs, primary_by_key)
            classified[key] = primary
            for d in group:
                d.is_primary = d.path == primary
            # A fallback guess must not be cached, or it can never be revisited.
            if len(dirs) > 1 and confident:
                resolved[key] = {"candidates": sorted(p.name for p in dirs), "primary": primary.name}

        return classified, resolved

    def classify_patient_folder(self, folder_path) -> dict[str, Path]:
        descriptors, _ = self._describe_patient_folder(folder_path)
        return self._group_and_resolve(descriptors, folder_path)[0]

    def label_patient_folder(self, folder_path) -> SubjectLabels:
        """Live classification + disambiguation for a whole subject, ignoring any cached sidecar; used by the preprocessing script."""
        folder_path = Path(folder_path)
        descriptors = self._classify_live(folder_path)
        _, primary_by_key = self._group_and_resolve(descriptors, folder_path, ignore_cache=True)
        return self._build_labels(folder_path, descriptors, primary_by_key)

    def _build_labels(
        self, folder_path: Path, descriptors: list[SequenceDescriptor], primary_by_key: dict[str, dict]
    ) -> SubjectLabels:
        series = [
            SeriesLabel(d.path.relative_to(folder_path).as_posix(), d.sequence_name, d.plane_name, d.key)
            for d in descriptors
        ]
        return SubjectLabels(series=series, primary_by_key=primary_by_key)

    def _cache_labels(
        self, folder_path: Path, descriptors: list[SequenceDescriptor], primary_by_key: dict[str, dict]
    ) -> None:
        try:
            save_labels(folder_path, self._build_labels(folder_path, descriptors, primary_by_key))
        except Exception as e:
            logging.warning(f"Could not cache cardisort labels for {folder_path.name}: {e}")

    def update_cached_label(self, folder_path, key: str, candidate_names: list[str], primary_name: str):
        """Keeps a manual reclassification (reassign_series) from being reverted by a stale sidecar entry."""
        update_primary(Path(folder_path), key, candidate_names, primary_name)

    def _resolve_primary_dir(
        self, key: str, dirs: list[Path], primary_by_key: dict
    ) -> tuple[Path, bool]:
        """Second element is False when the primary is a fallback guess rather than a real decision."""
        if len(dirs) == 1:
            return dirs[0], True

        signature = sorted(d.name for d in dirs)
        entry = primary_by_key.get(key)
        if entry is not None and entry.get("candidates") == signature:
            cached = next((d for d in dirs if d.name == entry.get("primary")), None)
            if cached is not None:
                return cached, True
            logging.warning(f"Cached primary for {key} points at a missing dir, recomputing")

        prefix, _, plane_name = key.partition("_")
        try:
            candidates = [summarize_sequence_dir(d, load_series_datasets(d)) for d in dirs]
            result = disambiguate_duplicates((prefix, plane_name), candidates)
            return next(d for d in dirs if d.name == result["primary_series"]), True
        except Exception as e:
            logging.warning(
                f"Disambiguation failed for {key} ({[d.name for d in dirs]}): {e}; "
                "falling back to largest series"
            )
            return max(dirs, key=lambda d: sum(1 for f in d.rglob("*") if f.is_file())), False

    def load_series_for_key(self, seq_dir: Path, key: str) -> dict:
        self._series_dirs[key] = Path(seq_dir)
        if key.startswith("CINE_"):
            volume, num_frames = self.load_cine_series(seq_dir, key)
            info = {"volume": volume, "is_temporal": True, "num_frames": num_frames}
        else:
            volume, _slice_positions = self.load_lge_series(seq_dir, key)
            info = {"volume": volume, "is_temporal": False, "num_frames": None}
        info["geometry"] = self.get_geometry(key)
        info["window"] = self._default_window(key, info["volume"])
        return info

    def _default_window(self, key: str, volume: np.ndarray) -> tuple[float, float]:
        """(center, width): the DICOM window for CINE if valid, else the 1st-99th percentile range."""
        window = None
        if key.startswith("CINE_"):
            window = dicom_window(self.series[key].slice_data["slice01"]["meta_data"][0])
        sample = volume[::4] if volume.ndim == 4 else volume
        return window or percentile_window(sample, 1, 99) or (float(volume.min()) + 0.5, 1.0)

    def clear_patient(self):
        """Forget the loaded patient, so a new session cannot save or show its series."""
        self.series = {}
        self._series_dirs = {}
        self.subject_folder = None
        self.folder_path = None
        self.num_time_frames = 0
        self.slice_positions = []

    def load_patient(self, folder_path) -> PatientLoadResult:
        folder_path = Path(folder_path)
        self.subject_folder = folder_path
        descriptors, from_cache = self._describe_patient_folder(folder_path)
        classified, primary_by_key = self._group_and_resolve(descriptors, folder_path)

        if not from_cache and self.use_cardisort_cache:
            self._cache_labels(folder_path, descriptors, primary_by_key)

        loaded: dict[str, dict] = {}
        for key, seq_dir in classified.items():
            try:
                loaded[key] = self.load_series_for_key(seq_dir, key)
            except Exception as e:
                logging.warning(f"Failed to load {key} from {seq_dir}: {e}")

        return PatientLoadResult(loaded=loaded, entries=descriptors)
