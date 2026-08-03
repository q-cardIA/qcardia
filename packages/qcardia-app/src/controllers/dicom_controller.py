import logging
from pathlib import Path

import numpy as np

from model.background_worker import CallableWorker
from model.dicom_model import read_disk_thumbnail
from utils.hanging_protocols import PRESETS, build_perfusion_preset
from view.series_browser_widget import SeriesCardInfo


class DICOMController:
    def __init__(self, view, model):
        self.view = view
        self.model = model
        self.current_time_index = 0
        self._last_result = None
        self._load_worker = None
        self._seg_worker = None
        self._lge_seg_worker = None

        self.view.load_patient_button_signal.connect(self.load_patient)
        self.view.seg_button_signal.connect(self.run_segmentation)
        self.view.lge_seg_button_signal.connect(self.run_lge_segmentation)
        self.view.save_button_signal.connect(self.save_segmentation)
        self.view.time_frame_changed_signal.connect(self.set_time_frame)
        self.view.preset_requested_signal.connect(self.apply_preset)
        self.view.reclassify_requested_signal.connect(self.reassign_series)

    @property
    def _widget(self):
        return self.view.display_area_widget

    def load_patient(self, folder_path, on_complete=None):
        """Classify + load a patient folder in the background (cardisort +
        series loading can take tens of seconds). `on_complete()` is called
        once loading has finished (success or failure) — used by the chat
        assistant to defer any tool calls that depend on this subject being
        loaded until it's actually ready."""
        if self._load_worker is not None and self._load_worker.isRunning():
            if on_complete:
                on_complete()
            return

        self.view.show_loading(f"Loading {Path(folder_path).name}…")
        self._load_worker = CallableWorker(self.model.load_patient, folder_path)
        self._load_worker.finished_ok.connect(
            lambda result: self._on_patient_loaded(result, on_complete)
        )
        self._load_worker.error.connect(lambda exc: self._on_patient_load_error(exc, on_complete))
        self._load_worker.start()

    def _on_patient_load_error(self, exc, on_complete=None):
        self.view.hide_loading()
        self._load_worker = None
        logging.error(f"Failed to load patient: {exc}")
        self.view.show_error("Load Error", f"Failed to load patient folder:\n{exc}")
        if on_complete:
            on_complete()

    def _on_patient_loaded(self, result, on_complete=None):
        self.view.hide_loading()
        self._load_worker = None

        if not result.entries:
            self.view.show_warning(
                "No Series Found", "No sequence directories found in this folder."
            )
            if on_complete:
                on_complete()
            return

        max_frames = 0
        for key, info in result.loaded.items():
            self._widget.register_series(
                key, info["volume"], is_temporal=info["is_temporal"], geometry=info["geometry"]
            )
            if info["is_temporal"]:
                max_frames = max(max_frames, info["num_frames"])

        self._last_result = result
        self.view.series_browser_widget.populate(self._build_browser_cards(result))

        if max_frames:
            self.current_time_index = 0
            self.view.update_time_slider_range(max_frames)

        if not result.loaded:
            self.view.show_warning(
                "No Loadable Series",
                "Cardisort did not recognize any loadable series in this folder; "
                "showing all detected sequences in the browser.",
            )

        if on_complete:
            on_complete()

    def reassign_series(self, card_info: SeriesCardInfo, new_key: str):
        if self._last_result is None:
            return
        try:
            info = self.model.load_series_for_key(card_info.path, new_key)
        except Exception as e:
            logging.error(f"Reassign failed: {e}")
            self.view.show_error(
                "Reassign Failed", f"Could not load this series as {new_key}:\n{e}"
            )
            return

        self._widget.register_series(
            new_key, info["volume"], is_temporal=info["is_temporal"], geometry=info["geometry"]
        )
        self._widget.refresh_series(new_key)

        for entry in self._last_result.entries:
            if entry.path == card_info.path:
                entry.key, entry.is_primary = new_key, True
            elif entry.key == new_key:
                entry.is_primary = False

        self._last_result.loaded[new_key] = info
        self.view.series_browser_widget.populate(self._build_browser_cards(self._last_result))

    def _build_browser_cards(self, result) -> list[SeriesCardInfo]:
        cards = []
        for entry in result.entries:
            is_lge_like = entry.sequence_name in {"DBLGE", "WBLGE"}
            loaded_here = entry.key is not None and entry.is_primary and entry.key in result.loaded

            thumbnail = (
                self._widget.get_thumbnail(entry.key)
                if loaded_here
                else read_disk_thumbnail(entry.path, is_lge_like)
            )

            if entry.key is not None:
                group = entry.key.split("_", 1)[0]
                label = entry.key if entry.is_primary else entry.path.name
            elif entry.sequence_name is not None:
                group = entry.sequence_name
                label = f"{entry.sequence_name}_{entry.plane_name}"
            else:
                group = "Unclassified"
                label = entry.path.name

            cards.append(
                SeriesCardInfo(
                    label=label,
                    group=group,
                    thumbnail=thumbnail,
                    draggable=loaded_here,
                    path=entry.path,
                    series_key=entry.key if loaded_here else None,
                )
            )
        return cards

    def run_segmentation(self):
        if self._seg_worker is not None and self._seg_worker.isRunning():
            return
        self.view.show_loading("Running segmentation…")
        self._seg_worker = CallableWorker(self.model.predict_segmentation)
        self._seg_worker.finished_ok.connect(self._on_segmentation_done)
        self._seg_worker.error.connect(self._on_segmentation_error)
        self._seg_worker.start()

    def _on_segmentation_done(self, temporal_segmentations):
        self.view.hide_loading()
        self._seg_worker = None
        if temporal_segmentations:
            seg_4d = np.stack(temporal_segmentations, axis=0)
            self._widget.set_segmentation("CINE_SAX", seg_4d)
        else:
            logging.error(
                "CINE segmentation produced no usable output — see earlier "
                "'Failed to extract segmentation for time N' warnings for the cause."
            )
            self.view.show_warning(
                "Segmentation Empty",
                "The CINE segmentation model ran but produced no usable output for "
                "this subject. Check the terminal/log output for details.",
            )

    def _on_segmentation_error(self, exc):
        self.view.hide_loading()
        self._seg_worker = None
        logging.error(f"Segmentation failed: {exc}")
        self.view.show_error("Segmentation Error", f"Failed to run segmentation:\n{exc}")

    def run_lge_segmentation(self):
        if self._lge_seg_worker is not None and self._lge_seg_worker.isRunning():
            return
        self.view.show_loading("Running LGE segmentation…")
        self._lge_seg_worker = CallableWorker(self.model.predict_lge_segmentation)
        self._lge_seg_worker.finished_ok.connect(self._on_lge_segmentation_done)
        self._lge_seg_worker.error.connect(self._on_lge_segmentation_error)
        self._lge_seg_worker.start()

    def _on_lge_segmentation_done(self, seg_volume):
        self.view.hide_loading()
        self._lge_seg_worker = None
        self._widget.set_segmentation("LGE_SAX", seg_volume)

    def _on_lge_segmentation_error(self, exc):
        self.view.hide_loading()
        self._lge_seg_worker = None
        if isinstance(exc, ValueError):
            self.view.show_warning("No LGE Volume", str(exc))
        else:
            logging.error(f"LGE segmentation failed: {exc}")
            self.view.show_error("LGE Segmentation Error", f"Failed to run LGE segmentation:\n{exc}")

    def apply_preset(self, name: str):
        if name == "perfusion":
            avail = set(self.model.series.keys())
            preset = build_perfusion_preset(
                has_stress="PERF_STRESS_SAX" in avail,
                has_rest="PERF_REST_SAX" in avail,
            )
        else:
            preset = PRESETS.get(name)

        if preset is None:
            logging.warning(f"apply_preset: unknown preset {name!r}")
            return

        rows, cols = preset.grid
        self._widget.set_grid_size(rows, cols)
        for panel in preset.panels:
            if self._widget.is_series_registered(panel.key):
                self._widget.set_panel_series(panel.idx, panel.key, axis=panel.axis)
            else:
                self._widget.set_panel_empty(panel.idx)

    def set_time_frame(self, time_index):
        self.current_time_index = time_index
        self._widget.set_time_index(time_index)

    def save_segmentation(self):
        try:
            output_path = self.model.save_segmentation()
            self.view.show_info("Saved", f"Segmentation saved to:\n{output_path}")
        except Exception as e:
            logging.error(f"Save failed: {e}")
            self.view.show_error("Save Error", f"Failed to save segmentation:\n{str(e)}")
