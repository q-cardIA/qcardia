import logging
from collections import Counter
from pathlib import Path

from qcardia_app.model.background_worker import CallableWorker
from qcardia_app.model.dicom_model import read_disk_thumbnail
from qcardia_app.utils import study_log
from qcardia_app.utils.hanging_protocols import PRESETS, PanelAssignment, available_presets
from qcardia_app.utils.windowing import null_myocardium_window
from qcardia_app.view.central_display_area_widget import CINE_TIME_AXIS
from qcardia_app.view.series_browser_widget import SeriesCardInfo


class DICOMController:
    def __init__(self, view, model):
        self.view = view
        self.model = model
        self.current_time_index = 0
        self._last_result = None
        self._current_subject_folder = None
        self._load_worker = None
        self._cine_frames = 0
        self._loading_keys: set[str] = set()  # series of the patient that is loading, not loaded yet
        self._load_total = 0  # series the load started with, for its progress
        self._on_ready = None  # load_patient's callback, until it has been called
        # Panels that show their series once it has loaded: panel -> (assignment, its preset's slots).
        self._waiting: dict[int, tuple[PanelAssignment, Counter]] = {}
        # Runs one at a time, first entry is running: several at once doubled peak memory (#11).
        self._seg_queue: list[str] = []
        self._seg_worker: CallableWorker | None = None
        self._seg_spans: dict[str, study_log.Span] = {}

        self.view.load_patient_button_signal.connect(self.load_patient)
        self.view.seg_button_signal.connect(self.run_segmentation)
        self.view.segmentable_series = self.model.segmentable
        self.view.save_button_signal.connect(self.save_segmentation)
        self.view.time_frame_changed_signal.connect(self.set_time_frame)
        self.view.preset_requested_signal.connect(self.apply_preset)
        self.view.reclassify_requested_signal.connect(self.reassign_series)
        self.view.zoom_requested.connect(self.zoom_all)
        self.view.contours_requested.connect(self.show_contours_everywhere)
        self.view.window_requested.connect(self.window_everywhere)
        self.view.phase_requested.connect(self.go_to_phase)
        self.view.panels_moved.connect(self._move_waiting)

    def is_busy(self) -> bool:
        return self._load_worker is not None or self._seg_worker is not None or bool(self._seg_queue)

    @property
    def loading_series(self) -> set[str]:
        """Series of the current patient that are still loading."""
        return set(self._loading_keys)

    def panel_assignments(self) -> list[dict]:
        """The widget's panels, where a panel that waits for its series has that series, with loading."""
        panels = self._widget.panel_assignments()
        for p in panels:
            if p["series_key"] is None and p["idx"] in self._waiting:
                panel = self._waiting[p["idx"]][0]
                p.update(series_key=panel.key, axis=panel.axis, loading=True)
        return panels

    def reset_session(self):
        """Back at the start screen: nothing of the last patient may reach the next session."""
        self._last_result = None
        self._current_subject_folder = None
        self._seg_spans.clear()
        self.model.clear_patient()

    @property
    def _widget(self):
        return self.view.display_area_widget

    def load_patient(self, folder_path, on_ready=None):
        """Classify + load a patient folder in the background. `on_ready(success)` fires once
        the patient can be used: its series are known, though they may still load. It also
        fires when the load fails or is rejected as busy."""
        if self._load_worker is not None and self._load_worker.isRunning():
            if on_ready:
                on_ready(False)
            return

        self._current_subject_folder = Path(folder_path)
        self._on_ready = on_ready
        self.view.show_loading(f"Loading {Path(folder_path).name}…")
        self.view.results_widget.show_analysis({})  # the previous patient's, also when this load fails
        self._load_worker = CallableWorker(self.model.load_patient, folder_path, report=True)
        self._load_worker.progress.connect(self._on_load_progress)
        span = study_log.Span("load_subject", subject=Path(folder_path).name)
        self._load_worker.finished_ok.connect(self._on_patient_loaded)
        self._load_worker.error.connect(self._on_patient_load_error)
        self._load_worker.finished_ok.connect(lambda result: span.end(series=sorted(result.loaded)))
        self._load_worker.error.connect(lambda exc: span.end(error=str(exc)))
        # `finished`, not `finished_ok`/`error`: those fire before the thread
        # actually stops, so clearing the reference there races teardown.
        self._load_worker.finished.connect(self._clear_load_worker)
        self._load_worker.start()

    def _clear_load_worker(self):
        self._load_worker = None

    def _ready(self, success: bool):
        """Calls load_patient's on_ready, once."""
        on_ready, self._on_ready = self._on_ready, None
        if on_ready:
            on_ready(success)

    def _end_loading(self):
        self._loading_keys.clear()
        self._fill_waiting_panels()
        self._start_next_segmentation()
        self.view.series_browser_widget.stop_loading()

    def _on_patient_load_error(self, exc):
        self.view.hide_loading()
        self._end_loading()
        logging.error(f"Failed to load patient: {exc}")
        self.view.show_error("Load Error", f"Failed to load patient folder:\n{exc}")
        self._ready(False)

    def _on_load_progress(self, event):
        kind, *args = event
        if kind == "classified":
            self._show_patient_outline(*args)
        elif kind == "series":
            self._add_loaded_series(*args)
        elif kind == "thumbnail":
            self.view.series_browser_widget.update_card(*args)

    def _show_patient_outline(self, result):
        """The series are known: show their cards and hide the loading screen, while they load."""
        if not result.entries:
            return  # reported once the load has finished
        self.view.hide_loading()
        self._last_result = result
        self._loading_keys = {e.key for e in result.entries if e.key is not None and e.is_primary}
        self._load_total = len(self._loading_keys)
        self.view.sidebar.set_available(available_presets(self._loading_keys))  # a preset's panels wait for its series
        # A new patient starts on an empty 2x2.
        self._waiting = {}
        self._widget.clear_series()
        self._widget.refresh_panels()  # every panel's series is gone: all empty
        self._widget.set_grid_size(2, 2)
        self._show_results()
        self.view.series_browser_widget.populate(self._build_browser_cards(result, loading=True))
        self.view.series_browser_widget.set_load_progress(0, self._load_total)
        self.view.sidebar.show_patient(self._current_subject_folder, 0)
        self.current_time_index = self._cine_frames = 0
        self.view.update_time_slider_range(0)
        self._ready(True)

    def _add_loaded_series(self, key: str, info: dict):
        self._register(key, info)
        self.view.schedule_screen_log()  # a waiting panel may show it now
        self._last_result.loaded[key] = info
        self._loading_keys.discard(key)
        self._fill_waiting_panels()
        if key in self._seg_queue:
            self._widget.show_segmentation_loading(key, "Queued…")
        self._start_next_segmentation()
        self.view.series_browser_widget.set_load_progress(self._load_total - len(self._loading_keys), self._load_total)
        path = next(e.path for e in self._last_result.entries if e.key == key and e.is_primary)
        self.view.series_browser_widget.update_card(path, self._widget.get_thumbnail(key), key)
        self.view.sidebar.show_patient(self._current_subject_folder, len(self._last_result.loaded))
        cine_frames = self._widget.n_frames(CINE_TIME_AXIS)
        if cine_frames != self._cine_frames:  # else a later cine would reset the frame shown
            self._cine_frames = cine_frames
            self.view.update_time_slider_range(cine_frames)

    def _on_patient_loaded(self, result):
        self.view.hide_loading()

        if not result.entries:
            self.view.show_warning(
                "No Series Found", "No sequence directories found in this folder."
            )
            self._ready(False)
            return

        self._end_loading()

        if not result.loaded:
            self.view.show_warning(
                "No Loadable Series",
                "Cardisort did not recognize any loadable series in this folder; "
                "showing all detected sequences in the browser.",
            )
        self._ready(True)

    def reassign_series(self, card_info: SeriesCardInfo, new_key: str):
        if self._last_result is None:
            return
        if self._load_worker is not None:  # the load thread still adds series to the model
            self.view.show_info("Please wait", "Reassign a series once the patient has loaded.")
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
        self.view.sidebar.set_available(available_presets(set(self.model.series) | self._loading_keys))
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
        self.view.sidebar.show_patient(self._current_subject_folder, len(self._last_result.loaded))

    def _register(self, key: str, info: dict):
        self._widget.register_series(key, info["volume"], is_temporal=info["is_temporal"],
                                     window=info["window"], geometry=info["geometry"])
        # Its size, to score "mid level" or "last frame" from the log.
        study_log.log("series_loaded", series=key, slices=self._widget.n_slices(key),
                      frames=info["volume"].shape[0] if info["is_temporal"] else 1)
        if info.get("segmentation") is not None:  # cached: ready to show, its contours still off
            self._widget.set_segmentation(key, info["segmentation"])
        if key in ("CINE_SAX", "LGE_SAX"):  # a cached segmentation was analysed while it loaded
            self._show_results()

    def _build_browser_cards(self, result, loading: bool = False) -> list[SeriesCardInfo]:
        """With loading, the series still load: no thumbnails yet, they come with the series."""
        cards = []
        carded_keys = set()
        for entry in result.entries:
            if entry.key is not None and entry.is_primary:
                if entry.key in carded_keys:  # a Siemens perfusion run has a primary folder per slice: one card
                    continue
                carded_keys.add(entry.key)
            is_lge_like = entry.sequence_name in {"DBLGE", "WBLGE"}
            loaded_here = entry.key is not None and entry.is_primary and entry.key in result.loaded

            if loading:
                thumbnail = None
            elif loaded_here:
                thumbnail = self._widget.get_thumbnail(entry.key)
            else:
                thumbnail = read_disk_thumbnail(entry.path, is_lge_like)

            if entry.key is not None and entry.is_primary:
                group = entry.key.split("_", 1)[0]
                label = entry.key
            elif entry.key is not None:  # a candidate the disambiguation did not pick: never loaded
                group = "Unclassified"
                label = entry.path.name
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
                    loading=loading and entry.key is not None and entry.is_primary,
                )
            )
        return cards

    def run_segmentation(self, series_key: str = "CINE_SAX") -> bool:
        """Queues segmentation for series_key; a series that is still loading is segmented
        once it has loaded. Returns False if it is already queued or running."""
        if series_key in self._seg_queue:
            return False
        if series_key not in self._loading_keys:  # a valid cache shows at once, for a next step in the same request
            try:
                seg = self.model.segment(series_key, cached_only=True)
            except Exception as e:
                logging.warning(f"Could not read the cached {series_key} segmentation: {e}")
                seg = None
            if seg is not None:
                self._on_segmentation_done(series_key, seg)
                return True
        self._seg_queue.append(series_key)
        self._seg_spans[series_key] = study_log.Span("segmentation", series=series_key, ahead=len(self._seg_queue) - 1)
        self._widget.show_segmentation_loading(series_key, "Queued…")
        self._start_next_segmentation()
        return True

    def segmentation_for(self, series_key: str):
        """series_key's segmentation: the one shown, else a valid cache. None if there is none."""
        seg = self._widget.segmentation(series_key)
        if seg is None and series_key in self.model.analysis:
            seg = self.model.segment(series_key, cached_only=True)
        return seg

    def zoom_all(self, mode: str):
        done = self._widget.zoom_panels(list(range(self._widget.n_panels)), mode, 100.0, self.model.analysis)
        if mode == "roi" and not done:
            self.view.show_info("Zoom to Heart", "No panel shows a segmented series or one of its plane. "
                                "Segment it first (Tools > Predict Labels).")

    def show_contours_everywhere(self, shown: bool):
        """With shown, series without a segmentation are segmented (or read from the cache) first."""
        for panel in self._widget.panel_assignments():
            key = panel["series_key"]
            if key is None:
                continue
            self._widget.set_panel_contours(panel["idx"], shown)
            if shown and not self._widget.has_segmentation(key) and key in self.model.segmentable:
                self.run_segmentation(key)

    def window_everywhere(self, mode: str):
        if mode == "reset":
            for key in self.model.series:
                if self._widget.is_series_registered(key):
                    self._widget.set_window(key, self._widget.window(key, default=True))
            return
        seg = self.segmentation_for("LGE_SAX") if self._widget.is_series_registered("LGE_SAX") else None
        window = null_myocardium_window(self._widget.volume("LGE_SAX"), seg) if seg is not None else None
        if window is None:
            self.view.show_info("Null Myocardium", "This needs a loaded LGE SAX with its segmentation "
                                "(Tools > Predict Labels).")
            return
        self._widget.set_window("LGE_SAX", window)

    def go_to_phase(self, phase: str):
        sax = self.model.analysis.get("CINE_SAX", {})
        if phase.lower() not in sax:
            self.view.show_info("Go to " + phase, "ED and ES come from the CINE SAX segmentation. "
                                "Segment it first (Tools > Predict Labels).")
            return
        self._widget.release_frames(CINE_TIME_AXIS)
        self._widget.set_time_index(sax[phase.lower()], CINE_TIME_AXIS)

    def is_segmenting(self, series_key: str) -> bool:
        """Queued or running in the background."""
        return series_key in self._seg_queue

    def _start_next_segmentation(self):
        if self._seg_worker is None and self._seg_queue and self._seg_queue[0] not in self._loading_keys:
            self._start_segmentation()

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
        self._start_next_segmentation()

    def _on_segmentation_done(self, series_key: str, result):
        self._widget.hide_segmentation_loading(series_key)
        self._widget.set_segmentation(series_key, result)
        self._show_results()
        self.view.schedule_screen_log()

    def _show_results(self):
        self.view.results_widget.show_analysis(self.model.analysis)

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
        preset = PRESETS.get(name)
        if preset is None:
            logging.warning(f"apply_preset: unknown preset {name!r}")
            return None
        present = lambda k: self._widget.is_series_registered(k) or k in self._loading_keys  # noqa: E731
        preset = preset.for_loaded(present)

        slots = Counter(p.key for p in preset.panels if p.slice_slot is not None)
        slices = [None if self._must_wait(p) else self._preset_slice(p, slots) for p in preset.panels]
        filled = sum(z is not None or self._must_wait(p) for p, z in zip(preset.panels, slices))
        total = filled + sum(not present(p.key) for p in preset.panels)  # a too-short SAX stack misses nothing
        if filled == 0:
            # Leave the current grid/panels untouched — a total no-op should
            # actually be a no-op, not silently wipe what was displayed.
            return 0, total

        rows, cols = preset.grid
        self._waiting = {}
        self._widget.set_grid_size(rows, cols)
        for panel, z in zip(preset.panels, slices):
            if z is not None:
                self._widget.set_panel_series(panel.idx, panel.key, axis=panel.axis, slice_idx=z)
            elif self._must_wait(panel):
                self._waiting[panel.idx] = (panel, slots)
                self._widget.set_panel_waiting(panel.idx, panel.key)
            else:
                self._widget.set_panel_empty(panel.idx)
        study_log.log("layout", preset=name, panels=self._widget.state()["panels"])  # 1-based, as the state events
        return filled, total

    def show_series_when_loaded(self, panel_idx: int, series_key: str, axis: str = "transverse"):
        """For a series that is still loading: the panel shows it once it has loaded."""
        self._waiting[panel_idx] = (PanelAssignment(panel_idx, series_key, axis), Counter())
        self._widget.set_panel_waiting(panel_idx, series_key)

    def _move_waiting(self, moves: dict):
        """The layout changed: a waiting panel moves with its cell, or stops waiting when its cell is gone."""
        self._waiting = {moves[i]: waiting for i, waiting in self._waiting.items() if i in moves}
        for idx, (panel, _) in self._waiting.items():
            self._widget.set_panel_waiting(idx, panel.key)

    def _must_wait(self, panel: PanelAssignment) -> bool:
        """Its series, or the series its slice is matched to, is still loading."""
        return panel.key in self._loading_keys or any(k in self._loading_keys for k in panel.match_keys)

    def _fill_waiting_panels(self):
        for idx, (panel, slots) in list(self._waiting.items()):
            if self._must_wait(panel):
                continue
            del self._waiting[idx]
            if not self._widget.is_panel_empty(idx):  # the user put something else there meanwhile
                continue
            z = self._preset_slice(panel, slots)
            if z is None:
                self._widget.set_panel_empty(idx)
            else:
                self._widget.set_panel_series(idx, panel.key, axis=panel.axis, slice_idx=z)

    def _preset_slice(self, panel, slots: Counter) -> int | None:
        """Slice to show in a preset panel, or None to leave it empty. Slots go from apex to base."""
        if not self._widget.is_series_registered(panel.key):
            return None
        n = self._widget.n_slices(panel.key, panel.axis)
        if panel.slice_slot is None:
            return n // 2
        geom = self.model.get_geometry(panel.key) if panel.axis == "transverse" else None
        if geom is not None and len(geom.ipp_list) != n:
            geom = None
        match_key = next((k for k in panel.match_keys if self._widget.is_series_registered(k)), None)
        if match_key is not None and geom is not None:
            ref_z = self._preset_slice(PanelAssignment(panel.idx, match_key, panel.axis, panel.slice_slot), slots)
            ref_geom = self.model.get_geometry(match_key)
            if ref_z is not None and ref_geom is not None:
                return ref_geom.nearest_slice(geom, ref_z)
        n_slots = slots[panel.key]
        if n >= n_slots:  # spread over the stack, steps of at least one slice
            i = round(panel.slice_slot * (n - 1) / max(n_slots - 1, 1))
        elif panel.slice_slot < n:
            i = panel.slice_slot
        else:
            return None
        return geom.apex_to_base()[i] if geom is not None else i

    def set_time_frame(self, time_index):
        self._widget.set_time_index(time_index)
        self.current_time_index = self._widget.current_time_index

    def save_segmentation(self):
        try:
            output_path = self.model.save_segmentation()
            self.view.show_info("Saved", f"Segmentation saved to:\n{output_path}")
        except Exception as e:
            logging.error(f"Save failed: {e}")
            self.view.show_error("Save Error", f"Failed to save segmentation:\n{str(e)}")
