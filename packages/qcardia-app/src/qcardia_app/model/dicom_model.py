from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom
import yaml
from qcardia.analysis import (
    CINE_LABELS,
    bounding_box,
    ed_es,
    myocardial_coverage,
    segmented_slices,
    volume_curve,
)
from qcardia.cardisort import (
    classify_sequence_dir,
    get_sequence_dirs,
    group_sequence_dirs,
    load_cardisort_model,
    load_series_datasets,
)
from qcardia.disambiguate import (
    FLOW_SEQUENCE_NAME,
    PERFUSION_SEQUENCE_NAMES,
    disambiguate_duplicates,
    select_flow_series,
    select_lge_series,
    select_perfusion_series,
    summarize_sequence_dir,
)
from qcardia.series import BaseSeries, CineSeries, FlowSeries, LGESeries, PerfusionSeries

from qcardia_app.model.perfusion_regions import matching_frames, perfusion_events, project_labels, region_curves
from qcardia_app.model.subject_labels import (
    SeriesLabel,
    SubjectLabels,
    load_labels,
    save_labels,
    update_primary,
)
from qcardia_app.utils.hanging_protocols import FLOW_KEYS, FLOW_MAGNITUDE_SUFFIX, PERF_REST_KEY, PERF_STRESS_KEY
from qcardia_app.utils.windowing import dicom_window, percentile_window, window_to_uint8
from qcardia_app.view.volume_geometry import VolumeGeometry

CARDISORT_SEQUENCE_TO_PREFIX = {"CINE": "CINE", "DBLGE": "LGE", "WBLGE": "LGE"}
CARDISORT_SUPPORTED_PLANES = {"SAX", "2CH", "3CH", "4CH"}
# Series that are read first load first; the perfusion runs are the largest.
_LOAD_ORDER = ("CINE_SAX", "CINE", "LGE_SAX", "LGE", "PC", "PERF_STRESS_SAX", "PERF")
PERFUSION_RUN_KEYS = {"stress": PERF_STRESS_KEY, "rest": PERF_REST_KEY}
PERFUSION_CACHE_KEY = "PERF"
FLOW_CACHE_KEY = "PC"


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


def _load_rank(key: str) -> tuple:
    rank = next((i for i, k in enumerate(_LOAD_ORDER) if key == k or key.startswith(k + "_")), len(_LOAD_ORDER))
    return rank, key


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
        self._perfusion_dirs: dict[str, list[Path]] = {}
        self._flow_dirs: dict[str, list[Path]] = {}
        self._lge_dirs: dict[Path, list[Path]] = {}  # an LGE folder -> the folders of its acquisition
        self.analysis: dict[str, dict] = {}  # per segmented series, see _analyse
        self._perfusion: dict[str, tuple] = {}  # run -> (volume, CINE_SAX segmentation, result), see perfusion_analysis
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

    def load_perfusion_series(self, folders: list[Path], series_key: str):
        self.series[series_key] = PerfusionSeries(folders)
        return self._process_temporal_volumes(series_key)

    def load_lge_series(self, folders: list[Path], series_key: str):
        self.series[series_key] = LGESeries([Path(f) for f in folders])
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

    @property
    def segmentable(self) -> tuple[str, ...]:
        """The series a segmentation model exists for."""
        return (*self.cine_modalities, "LGE_SAX")

    def segment(self, series_key: str, cached_only: bool = False) -> np.ndarray | None:
        """From the subject's .segmentation_cache if made from the same series and weights, else from the model
        (with cached_only, None). Without the model weights, a cache of the same series is used as it is,
        so a dataset with segmentations can be viewed without the models. Measured (see _analyse) the first time."""
        run = self.lge_wandb_run_path if series_key == "LGE_SAX" else self._seg_wandb_paths[series_key]
        weights = run / "files" / "last_model.pt"
        source = {"series": self._series_dirs[series_key].name}
        if weights.is_file():
            source["weights"] = weights.relative_to(run.parent.parent).as_posix()
            source["weights_modified"] = datetime.fromtimestamp(weights.stat().st_mtime).isoformat(timespec="seconds")
        else:
            logging.info(f"No {series_key} model weights at {weights}: only a cached segmentation can be shown.")
        folder = self.subject_folder / ".segmentation_cache" / series_key
        if self.use_segmentation_cache and (folder / "info.json").exists():
            info = json.loads((folder / "info.json").read_text())
            if all(info.get(k) == v for k, v in source.items()):
                with np.load(folder / "segmentation.npz") as cached:
                    seg = cached["seg"]
                if series_key not in self.analysis:
                    self.analysis[series_key] = self._analyse(series_key, seg)
                return seg
        if cached_only:
            return None
        if not weights.is_file():
            raise FileNotFoundError(f"No {series_key} segmentation model is installed and this series has no "
                                    f"cached segmentation (expected the weights at {weights}).")

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
        self.analysis[series_key] = self._analyse(series_key, seg)
        return seg

    def _analyse(self, series_key: str, seg: np.ndarray) -> dict:
        """A summary for navigation: the region of interest and, for CINE_SAX, the LV volume per
        frame, ED/ES and the LV slices per frame. Indices follow the display order."""
        seg = seg if seg.ndim == 4 else seg[np.newaxis]
        result = {"box": bounding_box(seg)}
        geom = self.get_geometry(series_key)
        if geom is None:
            return result
        positions = [geom.slice_position(i) for i in range(len(geom.ipp_list))]
        spacing = float(np.median(np.abs(np.diff(positions)))) if len(positions) > 1 else 0.0
        voxel_mm3 = geom.pixel_spacing[0] * geom.pixel_spacing[1] * spacing
        if series_key == "CINE_SAX":
            curve = volume_curve(seg, voxel_mm3)
            ed, es = ed_es(curve)
            rv = volume_curve(seg, voxel_mm3, CINE_LABELS["rv"])
            result.update(volumes=curve.round(1).tolist(), ed=ed, es=es, apex_to_base=geom.apex_to_base(),
                          rv_volumes=rv.round(1).tolist(),
                          lv_slices=[segmented_slices(seg, t) for t in range(len(seg))],
                          lv_myo_rv_slices=[sorted(set.intersection(*(set(segmented_slices(seg, t, label))
                                                                      for label in CINE_LABELS.values())))
                                            for t in range(len(seg))],
                          lv_coverage=[[round(c, 2) for c in myocardial_coverage(seg, t)] for t in range(len(seg))])
        elif series_key == "LGE_SAX":
            # LGE class 1 = myocardium without scar, class 2 = scar, so the myocardium is both.
            scar = float(volume_curve(seg, voxel_mm3, 2)[0])
            myo = float(volume_curve(seg, voxel_mm3, 1)[0]) + scar
            result.update(myo_ml=round(myo, 1), scar_ml=round(scar, 1), scar_slices=segmented_slices(seg, 0, 2))
        return result

    def perfusion_analysis(self, run: str, volume: np.ndarray | None, cine_seg: np.ndarray | None) -> dict | None:
        """Where to navigate in a perfusion run: the frames (0-based) of contrast arrival and peak in the LV blood
        pool and the myocardium (see perfusion_regions), from the CINE_SAX segmentation mapped onto the perfusion
        slices through patient coordinates, without registration. Measured once per run and segmentation."""
        if volume is None or cine_seg is None:
            return None
        cached = self._perfusion.get(run)
        if cached and cached[0] is volume and cached[1] is cine_seg:
            return cached[2]
        cine, perf = self.get_geometry("CINE_SAX"), self.get_geometry(run)
        if cine is None or perf is None:
            return None
        try:
            frames = matching_frames(self._trigger_times("CINE_SAX")[0], [t[0] for t in self._trigger_times(run)])
            result = {"events": perfusion_events(region_curves(volume, project_labels(cine_seg, cine, frames, perf)))}
        except Exception as e:  # e.g. no trigger times: the run just has no events
            logging.warning(f"Could not measure the {run} frames from the CINE_SAX segmentation: {e}")
            result = {"events": {}}
        self._perfusion[run] = (volume, cine_seg, result)
        return result

    def _trigger_times(self, series_key: str) -> list[list[float]]:
        """Per slice in display order (by slice position, as the volume), the trigger time of each frame in ms."""
        series = self.series[series_key]
        metas = [series.slice_data[k]["meta_data"] for k in sorted(series.slice_data)]
        ordered = sorted(((self._extract_slice_position(m[0], i), m) for i, m in enumerate(metas)), key=lambda e: e[0])
        return [[float(meta.TriggerTime) for meta in m] for _, m in ordered]

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

        # Siemens splits an acquisition into series (e.g. a perfusion run's slices and AIF): classified together.
        prediction_by_dir = {}
        for group in group_sequence_dirs(sequence_dirs):
            prediction = classify_sequence_dir(
                group,
                model,
                config["model"]["nr_input_channels"],
                tuple(config["data"]["target_pixdim"]),
                tuple(config["data"]["target_size"]),
                config["data"]["image_grid_sample_mode"],
            )
            prediction_by_dir.update(dict.fromkeys(group, prediction))

        descriptors = []
        for seq_dir in sequence_dirs:
            prediction = prediction_by_dir[seq_dir]
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
        use_cache = self.use_cardisort_cache and not ignore_cache
        cached = load_labels(folder_path) if use_cache else None
        primary_by_key = cached.primary_by_key if cached is not None else {}

        classified: dict[str, Path] = {}
        resolved: dict[str, dict] = {}
        self._lge_dirs = {}
        perfusion_entry = self._assign_perfusion_keys(descriptors, primary_by_key)
        if perfusion_entry is not None:
            resolved[PERFUSION_CACHE_KEY] = perfusion_entry
        flow_entry = self._assign_flow_keys(descriptors, primary_by_key)
        if flow_entry is not None:
            resolved[FLOW_CACHE_KEY] = flow_entry

        dirs_by_key: dict[str, list[SequenceDescriptor]] = defaultdict(list)
        for d in descriptors:
            if d.key is not None:
                dirs_by_key[d.key].append(d)
        for key, group in dirs_by_key.items():
            dirs = [d.path for d in group]
            if key in self._perfusion_dirs:
                # Siemens stores each slice of a run as its own series: all of them make up the run.
                classified[key] = dirs[0]
                for d in group:
                    d.is_primary = True
                continue
            if key in self._flow_dirs:  # the series of the acquisition load together, as one
                classified[key] = dirs[0]
                for d in group:
                    d.is_primary = d.path == dirs[0]
                continue
            if key.startswith("LGE_"):  # a Siemens LGE is a magnitude and a PSIR series: one candidate
                acquisitions = select_lge_series(dirs)
                self._lge_dirs.update({p: a for a in acquisitions for p in a})
                dirs = [a[0] for a in acquisitions]
            primary, confident = self._resolve_primary_dir(key, dirs, primary_by_key)
            classified[key] = primary
            primary_dirs = self._lge_dirs.get(primary, [primary])
            for d in group:
                d.is_primary = d.path in primary_dirs
            # A fallback guess must not be cached, or it can never be revisited.
            if len(dirs) > 1 and confident:
                resolved[key] = {"candidates": sorted(p.name for p in dirs), "primary": primary.name}

        return classified, resolved

    def _assign_perfusion_keys(self, descriptors: list[SequenceDescriptor], primary_by_key: dict) -> dict | None:
        """Gives the dirs of the stress and rest runs their keys; test runs and unmatched dirs get none.
        The keys are cached with the series, so a cached grouping holds while its candidates match."""
        perfusion = [d for d in descriptors if d.sequence_name in PERFUSION_SEQUENCE_NAMES]
        entry = {"candidates": sorted(d.path.name for d in perfusion)} if perfusion else None
        if entry is not None and primary_by_key.get(PERFUSION_CACHE_KEY) != entry:
            try:
                runs = select_perfusion_series([d.path for d in perfusion])
            except Exception as e:
                logging.warning(f"Could not group the perfusion series into stress and rest runs: {e}")
                runs, entry = {}, None  # a failure must not be cached, or it can never be revisited
            key_by_dir = {p: PERFUSION_RUN_KEYS[role] for role, dirs in runs.items() for p in dirs}
            for d in perfusion:
                d.key = key_by_dir.get(d.path)

        self._perfusion_dirs = defaultdict(list)
        for d in perfusion:
            if d.key is not None:
                self._perfusion_dirs[d.key].append(d.path)
        return entry

    def _assign_flow_keys(self, descriptors: list[SequenceDescriptor], primary_by_key: dict) -> dict | None:
        """Gives the series of each vessel's flow acquisition its key. As for perfusion, the keys
        are cached with the series and hold while the subject's series are the same."""
        has_flow = any(d.sequence_name == FLOW_SEQUENCE_NAME for d in descriptors)
        entry = {"candidates": sorted(d.path.name for d in descriptors)} if has_flow else None
        if primary_by_key.get(FLOW_CACHE_KEY) != entry:
            for d in descriptors:
                if d.key is not None and d.key.startswith("PC_"):
                    d.key = None
            try:
                selected = select_flow_series({d.path: (d.sequence_name, d.plane_name) for d in descriptors})
            except Exception as e:
                logging.warning(f"Could not find the flow acquisitions: {e}")
                selected, entry = {}, None  # a failure must not be cached, or it can never be revisited
            key_by_dir = {p: FLOW_KEYS[vessel] for vessel, dirs in selected.items() for p in dirs}
            for d in descriptors:
                if d.path in key_by_dir:
                    d.key = key_by_dir[d.path]

        self._flow_dirs = defaultdict(list)
        for d in descriptors:
            if d.key is not None and d.key.startswith("PC_"):
                self._flow_dirs[d.key].append(d.path)
        return entry

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
        elif key.startswith("PERF_"):
            volume, num_frames = self.load_perfusion_series(self._perfusion_dirs.get(key, [seq_dir]), key)
            info = {"volume": volume, "is_temporal": True, "num_frames": num_frames}
        elif key.startswith("PC_") and key.endswith(FLOW_MAGNITUDE_SUFFIX):  # loads after its velocity
            self.series[key] = self.series[key.removesuffix(FLOW_MAGNITUDE_SUFFIX)]
            volume, num_frames = self._process_temporal_volumes(key)
            info = {"volume": volume, "is_temporal": True, "num_frames": num_frames}
        elif key.startswith("PC_"):
            flow = self.series[key] = FlowSeries(self._flow_dirs.get(key, [seq_dir]))
            volume = flow.get_velocity_array()[:, np.newaxis]  # (T, 1, H, W), as a cine
            info = {"volume": volume, "is_temporal": True, "num_frames": volume.shape[0]}
        else:
            volume, _slice_positions = self.load_lge_series(self._lge_dirs.get(Path(seq_dir), [seq_dir]), key)
            info = {"volume": volume, "is_temporal": False, "num_frames": None}
        info["geometry"] = self.get_geometry(key)
        info["window"] = self._default_window(key, info["volume"])
        if key in self.segmentable:
            self.analysis.pop(key, None)  # measured from the series this one replaces, e.g. on a reassign
            try:  # in the viewer and measured at once; a damaged cache must not stop the series from loading
                info["segmentation"] = self.segment(key, cached_only=True)
            except Exception as e:
                logging.warning(f"Could not read the cached {key} segmentation: {e}")
        return info

    def _default_window(self, key: str, volume: np.ndarray) -> tuple[float, float]:
        """(center, width): the DICOM window for CINE if valid, -VENC..+VENC for a flow velocity,
        else the 1st-99th percentile range."""
        if key.startswith("PC_") and not key.endswith(FLOW_MAGNITUDE_SUFFIX):
            return 0.0, 2 * self.series[key].venc
        window = None
        if key.startswith("CINE_"):
            window = dicom_window(self.series[key].slice_data["slice01"]["meta_data"][0])
        sample = volume[::4] if volume.ndim == 4 else volume
        return window or percentile_window(sample, 1, 99) or (float(volume.min()) + 0.5, 1.0)

    def clear_patient(self):
        """Forget the loaded patient, so a new session cannot save or show its series."""
        self.series = {}
        self._series_dirs = {}
        self._perfusion_dirs = {}
        self._flow_dirs = {}
        self._lge_dirs = {}
        self.analysis = {}
        self._perfusion = {}  # run -> (volume, CINE_SAX segmentation, result), see perfusion_analysis
        self.subject_folder = None
        self.folder_path = None
        self.num_time_frames = 0
        self.slice_positions = []

    def load_patient(self, folder_path, report=None) -> PatientLoadResult:
        """report, if given, gets ("classified", result) once the series are known, then
        ("series", key, info) as each loads and ("thumbnail", path, image) for the series that do not."""
        folder_path = Path(folder_path)
        self.clear_patient()  # else the last patient's series stay, e.g. in what the panel state reports as loaded
        self.subject_folder = folder_path
        descriptors, from_cache = self._describe_patient_folder(folder_path)
        classified, primary_by_key = self._group_and_resolve(descriptors, folder_path)

        if self.use_cardisort_cache:
            cached = load_labels(folder_path) if from_cache else None
            if cached is None:
                self._cache_labels(folder_path, descriptors, primary_by_key)
            elif any(cached.primary_by_key.get(k) != v for k, v in primary_by_key.items()):
                self._cache_labels(folder_path, descriptors, {**cached.primary_by_key, **primary_by_key})

        # A flow acquisition shows as two series: its velocity, and its magnitude.
        magnitudes = [replace(d, key=d.key + FLOW_MAGNITUDE_SUFFIX) for d in descriptors
                      if d.is_primary and d.key in self._flow_dirs]
        classified.update({d.key: d.path for d in magnitudes})
        descriptors = descriptors + magnitudes

        if report:
            report(("classified", PatientLoadResult(loaded={}, entries=descriptors)))
        loaded: dict[str, dict] = {}
        for key in sorted(classified, key=_load_rank):
            try:
                loaded[key] = self.load_series_for_key(classified[key], key)
            except Exception as e:
                logging.warning(f"Failed to load {key} from {classified[key]}: {e}")
                continue
            if report:
                report(("series", key, loaded[key]))
        if report:
            for d in descriptors:
                if not (d.is_primary and d.key in loaded):
                    image = read_disk_thumbnail(d.path, d.sequence_name in {"DBLGE", "WBLGE"})
                    report(("thumbnail", d.path, image))

        return PatientLoadResult(loaded=loaded, entries=descriptors)
