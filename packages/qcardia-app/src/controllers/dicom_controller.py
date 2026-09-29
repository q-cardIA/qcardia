import logging
from pathlib import Path

from model.background_worker import CallableWorker
from model.dicom_model import read_disk_thumbnail
from utils import study_log
from utils.hanging_protocols import PERF_REST_KEY, PERF_STRESS_KEY, PRESETS, build_perfusion_preset
from view.series_browser_widget import SeriesCardInfo


class DICOMController:
    def __init__(self, view, model):
        self.view = view
        self.model = model
        self.current_time_index = 0
        self._last_result = None
        self._current_subject_folder = None
        self._load_worker = None
        # Runs one at a time, first entry is running: several at once doubled peak memory (#11).
        self._seg_queue: list[str] = []
        self._seg_worker: CallableWorker | None = None
        self._seg_spans: dict[str, study_log.Span] = {}

        self.view.load_patient_button_signal.connect(self.load_patient)
        self.view.seg_button_signal.connect(self.run_segmentation)
        self.view.segmentable_series = (*self.model.cine_modalities, "LGE_SAX")
        self.view.save_button_signal.connect(self.save_segmentation)
        self.view.time_frame_changed_signal.connect(self.set_time_frame)
        self.view.preset_requested_signal.connect(self.apply_preset)
        self.view.reclassify_requested_signal.connect(self.reassign_series)

    def is_busy(self) -> bool:
        return self._load_worker is not None or self._seg_worker is not None or bool(self._seg_queue)

    def reset_session(self):
        """Back at the start screen: nothing of the last patient may reach the next session."""
        self._last_result = None
        self._current_subject_folder = None
        self._seg_spans.clear()
        self.model.clear_patient()

    @property
    def _widget(self):
        return self.view.display_area_widget

    def load_patient(self, folder_path, on_complete=None):
        """Classify + load a patient folder in the background. `on_complete(success)`
        fires once loading has actually finished (or been rejected as busy)."""
        if self._load_worker is not None and self._load_worker.isRunning():
            if on_complete:
                on_complete(False)
            return

        self._current_subject_folder = Path(folder_path)
        self.view.show_loading(f"Loading {Path(folder_path).name}…")
        self._load_worker = CallableWorker(self.model.load_patient, folder_path)
        span = study_log.Span("load_subject", subject=Path(folder_path).name)
        self._load_worker.finished_ok.connect(
            lambda result: self._on_patient_loaded(result, on_complete)
        )
        self._load_worker.error.connect(lambda exc: self._on_patient_load_error(exc, on_complete))
        self._load_worker.finished_ok.connect(lambda result: span.end(series=sorted(result.loaded)))
        self._load_worker.error.connect(lambda exc: span.end(error=str(exc)))
        # `finished`, not `finished_ok`/`error`: those fire before the thread
        # actually stops, so clearing the reference there races teardown.
        self._load_worker.finished.connect(self._clear_load_worker)
        self._load_worker.start()

    def _clear_load_worker(self):
        self._load_worker = None

    def _on_patient_load_error(self, exc, on_complete=None):
        self.view.hide_loading()
        logging.error(f"Failed to load patient: {exc}")
        self.view.show_error("Load Error", f"Failed to load patient folder:\n{exc}")
        if on_complete:
            on_complete(False)

    def _on_patient_loaded(self, result, on_complete=None):
        self.view.hide_loading()

        if not result.entries:
            self.view.show_warning(
                "No Series Found", "No sequence directories found in this folder."
            )
            if on_complete:
                on_complete(False)
            return

        max_frames = 0
        self._widget.clear_series()
        for key, info in result.loaded.items():
            self._register(key, info)
            if info["is_temporal"]:
                max_frames = max(max_frames, info["num_frames"])

        self._widget.refresh_panels()
        self._last_result = result
        self.view.series_browser_widget.populate(self._build_browser_cards(result))
        self.view.sidebar.show_patient(self._current_subject_folder.name, len(result.loaded))

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
            on_complete(True)

    def reassign_series(self, card_info: SeriesCardInfo, new_key: str):
        if self._last_result is None:
            return
        try:
            info = self.model.load_series_for_key(card_info.path, new_key)
        except Exception as e:
            study_log.log("reassign_series", series_folder=card_info.path.name, key=new_key, error=str(e))
            logging.error(f"Reassign failed: {e}")
            self.view.show_error(
                "Reassign Failed", f"Could not load this series as {new_key}:\n{e}"
            )
            return

        study_log.log("reassign_series", series_folder=card_info.path.name, key=new_key)
        self._register(new_key, info)
        self._widget.refresh_panels(new_key)

        for entry in self._last_result.entries:
            if entry.path == card_info.path:
                entry.key, entry.is_primary = new_key, True
            elif entry.key == new_key:
                entry.is_primary = False

        if self._current_subject_folder is not None:
            candidate_names = [e.path.name for e in self._last_result.entries if e.key == new_key]
            self.model.update_cached_label(
                self._current_subject_folder, new_key, candidate_names, card_info.path.name
            )

        self._last_result.loaded[new_key] = info
        self.view.series_browser_widget.populate(self._build_browser_cards(self._last_result))
        self.view.sidebar.show_patient(self._current_subject_folder.name, len(self._last_result.loaded))

    def _register(self, key: str, info: dict):
        self._widget.register_series(key, info["volume"], is_temporal=info["is_temporal"],
                                     window=info["window"], geometry=info["geometry"])

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

    def run_segmentation(self, series_key: str = "CINE_SAX") -> bool:
        """Queues segmentation for series_key. Returns False if it is already queued or running."""
        if series_key in self._seg_queue:
            return False
        self._seg_queue.append(series_key)
        self._seg_spans[series_key] = study_log.Span("segmentation", series=series_key, ahead=len(self._seg_queue) - 1)
        self._widget.show_segmentation_loading(series_key, "Queued…")
        if len(self._seg_queue) == 1:
            self._start_segmentation()
        return True

    def _start_segmentation(self):
        key = self._seg_queue[0]
        self._widget.show_segmentation_loading(key, "Segmenting…")
        self._seg_worker = CallableWorker(self.model.segment, key)
        span = self._seg_spans.pop(key)
        span.mark("running")
        self._seg_worker.finished_ok.connect(lambda _: span.end())
        self._seg_worker.error.connect(lambda exc: span.end(error=str(exc)))
        self._seg_worker.finished_ok.connect(lambda result: self._on_segmentation_done(key, result))
        self._seg_worker.error.connect(lambda exc: self._on_segmentation_error(key, exc))
        self._seg_worker.finished.connect(self._on_segmentation_finished)
        self._seg_worker.start()

    def _on_segmentation_finished(self):
        self._seg_worker = None
        self._seg_queue.pop(0)
        if self._seg_queue:
            self._start_segmentation()

    def _on_segmentation_done(self, series_key: str, result):
        self._widget.hide_segmentation_loading(series_key)
        self._widget.set_segmentation(series_key, result)

    def _on_segmentation_error(self, series_key: str, exc):
        self._widget.hide_segmentation_loading(series_key)
        if isinstance(exc, ValueError):
            self.view.show_warning("No Segmentation Data", str(exc))
        else:
            logging.error(f"{series_key} segmentation failed: {exc}")
            self.view.show_error(
                "Segmentation Error", f"Failed to run {series_key} segmentation:\n{exc}"
            )

    def run_lge_segmentation(self) -> bool:
        return self.run_segmentation("LGE_SAX")

    def apply_preset(self, name: str) -> tuple[int, int] | None:
        """Returns (panels_filled, panels_total), or None for an unknown preset name."""
        if name == "perfusion":
            avail = set(self.model.series.keys())
            preset = build_perfusion_preset(
                has_stress=PERF_STRESS_KEY in avail,
                has_rest=PERF_REST_KEY in avail,
            )
        else:
            preset = PRESETS.get(name)

        if preset is None:
            logging.warning(f"apply_preset: unknown preset {name!r}")
            return None

        total = len(preset.panels)
        n_slots = sum(p.slice_slot is not None for p in preset.panels)
        slices = [self._preset_slice(p, n_slots) for p in preset.panels]
        filled = sum(z is not None for z in slices)
        if filled == 0:
            # Leave the current grid/panels untouched — a total no-op should
            # actually be a no-op, not silently wipe what was displayed.
            return 0, total

        rows, cols = preset.grid
        self._widget.set_grid_size(rows, cols)
        for panel, z in zip(preset.panels, slices):
            if z is not None:
                self._widget.set_panel_series(panel.idx, panel.key, axis=panel.axis, slice_idx=z)
            else:
                self._widget.set_panel_empty(panel.idx)
        study_log.log("layout", preset=name, panels=self._widget.panel_assignments())
        return filled, total

    def _preset_slice(self, panel, n_slots: int) -> int | None:
        """Slice to show in a preset panel, or None to leave it empty."""
        if not self._widget.is_series_registered(panel.key):
            return None
        n = self._widget.n_slices(panel.key, panel.axis)
        if panel.slice_slot is None:
            return n // 2
        if n >= n_slots:  # spread over the stack, steps of at least one slice
            return round(panel.slice_slot * (n - 1) / max(n_slots - 1, 1))
        return panel.slice_slot if panel.slice_slot < n else None

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
