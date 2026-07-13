from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from qcardia.cardisort import (
    classify_sequence_dir,
    get_sequence_dirs,
    load_cardisort_model,
    load_series_datasets,
)
from qcardia.disambiguate import disambiguate_duplicates, summarize_sequence_dir
from qcardia.series import BaseSeries, CineSeries, LGESeries
from qcardia_models.models import UNet2d

from view.volume_geometry import VolumeGeometry

CARDISORT_SEQUENCE_TO_PREFIX = {"CINE": "CINE", "DBLGE": "LGE", "WBLGE": "LGE"}
CARDISORT_SUPPORTED_PLANES = {"SAX", "2CH", "3CH", "4CH"}


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
        self._load_segmentation_model()

    def _load_segmentation_model(self):
        wandb_path = Path.cwd() / "wandb"
        config_path = wandb_path / "files" / "config-copy.yaml"

        self.config = yaml.load(Path.open(config_path), Loader=yaml.FullLoader)
        self.wandb_run_path = wandb_path

        self.seg_model = UNet2d(
            nr_input_channels=self.config["unet"]["nr_image_channels"],
            channels_list=self.config["unet"]["channels_list"],
            nr_output_classes=self.config["unet"]["nr_output_classes"],
            nr_output_scales=self.config["unet"]["nr_output_scales"],
        ).to("cpu")

        model_weights = torch.load(wandb_path / "files" / "last_model.pt")
        self.seg_model.load_state_dict(model_weights)

    def load_cine_series(self, folder_path, series_key: str):
        self.series[series_key] = CineSeries(Path(folder_path), batch_size=200)
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

        segmentation = self.series["LGE_SAX"].predict_segmentation(self.wandb_run_path)
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

    def predict_segmentation(self):
        cine = self.series.get("CINE_SAX")
        if cine is None:
            raise ValueError("No DICOM data loaded")

        cine_segmentation = cine.predict_segmentation(self.wandb_run_path)
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
                return x
            elif x.ndim == 2:
                return x[np.newaxis, :, :]
            else:
                return x.reshape(nslices, -1, x.shape[-1])

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

    def classify_patient_folder(self, folder_path) -> dict[str, Path]:
        model, config = self._get_cardisort_model()
        sequence_dirs = get_sequence_dirs(Path(folder_path))

        dirs_by_key: dict[str, list[Path]] = defaultdict(list)
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
                continue

            sequence_name, plane_name = prediction
            prefix = CARDISORT_SEQUENCE_TO_PREFIX.get(sequence_name)
            if prefix is None or plane_name not in CARDISORT_SUPPORTED_PLANES:
                continue

            key = f"{prefix}_{plane_name}"
            dirs_by_key[key].append(seq_dir)

        classified: dict[str, Path] = {}
        for key, dirs in dirs_by_key.items():
            classified[key] = self._resolve_primary_dir(key, dirs)

        return classified

    def _resolve_primary_dir(self, key: str, dirs: list[Path]) -> Path:
        if len(dirs) == 1:
            return dirs[0]

        prefix, _, plane_name = key.partition("_")
        try:
            candidates = [
                summarize_sequence_dir(d, load_series_datasets(d)) for d in dirs
            ]
            result = disambiguate_duplicates((prefix, plane_name), candidates)
            return next(d for d in dirs if d.name == result["primary_series"])
        except Exception as e:
            logging.warning(
                f"Disambiguation failed for {key} ({[d.name for d in dirs]}): {e}; "
                "falling back to largest series"
            )
            return max(dirs, key=lambda d: sum(1 for f in d.rglob("*") if f.is_file()))

    def load_patient(self, folder_path) -> dict[str, dict]:
        classified = self.classify_patient_folder(folder_path)

        loaded: dict[str, dict] = {}
        for key, seq_dir in classified.items():
            try:
                if key.startswith("CINE_"):
                    volume, num_frames = self.load_cine_series(seq_dir, key)
                    loaded[key] = {"volume": volume, "is_temporal": True, "num_frames": num_frames}
                else:
                    volume, _slice_positions = self.load_lge_series(seq_dir, key)
                    loaded[key] = {"volume": volume, "is_temporal": False, "num_frames": None}
                loaded[key]["geometry"] = self.get_geometry(key)
            except Exception as e:
                logging.warning(f"Failed to load {key} from {seq_dir}: {e}")

        return loaded
