"""ChatController — Qt orchestration around the LLM interaction layer in `llm/`.

The worker class comes from utils.llm_devices, so a device speaking another wire
format changes nothing here.
"""
from __future__ import annotations

import logging
import re
import time
from contextlib import suppress

import numpy as np
from PySide6.QtCore import Qt, QTimer

from llm.context import base_mid_apex, build_state_block, grid_text
from llm.prompt import build_messages
from llm.tools import ToolContext, build_tools
from model.background_worker import CallableWorker
from model.ollama_client import warm_model
from utils import llm_devices, study_log
from utils.app_config import data_dir, list_subjects
from utils.windowing import WINDOW_STEPS, null_myocardium_window, step_window
from utils.layouts import CINE_SPEEDS, MAX_GRID_COLS, MAX_GRID_ROWS, MAX_HALF
from view.central_display_area_widget import CINE_TIME_AXIS

_MAX_HISTORY_TURNS = 10


def _drop_planned(remaining: list, after: str, keep: str | None = None) -> str:
    """Drops the calls in remaining (except keep) that were planned before after happened; the text for
    the tool result, so the model plans them again with the new state."""
    dropped = [name for name, _ in remaining if name != keep]
    remaining[:] = [call for call in remaining if call[0] == keep]
    if not dropped:
        return ""
    return f" Not done (planned before {after}): {', '.join(dropped)}. Plan them again with the new state block."


def _panel_list(arguments: dict, n_panels: int) -> list[int] | str:
    """The 'panels' argument (all panels if left out), or an error text."""
    panels = [int(i) for i in arguments.get("panels") or range(n_panels)]
    if any(not 0 <= i < n_panels for i in panels):
        return f"panel out of range (grid has {n_panels} panels)."
    return panels


def _panels_text(panels) -> str:
    """'panel 1' or 'panels 0, 2'."""
    panels = list(panels)
    return f"panel{'s' if len(panels) > 1 else ''} {', '.join(map(str, panels))}"


class ChatController:
    def __init__(self, view, dicom_controller, model):
        self.view = view
        self.dicom_controller = dicom_controller
        self.model = model
        self._history: list[dict] = []
        self._worker = None
        # ChatWidget builds and selects its model combo before this controller
        # exists (see MainWindow.build_analysis_interface), so read its actual
        # selection here rather than a default: the two must not silently disagree.
        self._llm_model = self.view.chat_widget.current_tag() or llm_devices.default_model()
        self._think = self.view.chat_widget.think_enabled()
        self._available_subjects_cache: list[str] | None = None
        self._warmup_worker: CallableWorker | None = None  # see _warm_model_in_background
        self._model_load_worker: CallableWorker | None = None  # see _preload_selected_model
        self._live_workers: list = []  # anything whose thread must outlive its variable

        self.view.chat_widget.message_submitted.connect(self.send_message)
        self.view.chat_widget.model_changed.connect(self._on_model_changed)
        self.view.chat_widget.device_changed.connect(self._on_device_changed)
        self.view.chat_widget.think_changed.connect(self._on_think_changed)

    def is_busy(self) -> bool:
        """A thread of this controller or its chat panel is still running; dropping it would abort."""
        chat = self.view.chat_widget
        threads = [self._worker, self._warmup_worker, self._model_load_worker, *self._live_workers,
                   getattr(chat, "_import_worker", None), getattr(chat, "_device_switch_worker", None),
                   getattr(chat, "_mic_worker", None), getattr(chat, "_transcription_worker", None)]
        return any(t is not None and t.isRunning() for t in threads)

    def _on_model_changed(self, model_name: str):
        self._llm_model = model_name
        self._preload_selected_model()

    def _on_think_changed(self, enabled: bool):
        self._think = enabled

    def _on_device_changed(self, device):
        """Detach, not stop, a request still running on the device we just left.

        It is blocked in a socket read, so it can only be disconnected (a late reply
        must not land in a moved-on conversation, nor null out the next worker) and
        held until it finishes (dropping it mid-run aborts the process).
        """
        worker = self._worker
        if worker is None:
            # A tool sequence with no worker in flight is still mid-run, and re-enabling
            # input would let the next send race its follow-up request.
            return
        self._worker = None
        for signal in (worker.token_received, worker.thinking_received,
                       worker.tool_calls_ready,
                       worker.finished_ok, worker.error, worker.finished):
            with suppress(RuntimeError, TypeError):
                signal.disconnect()  # no args = drop every slot on that signal
        self._hold(worker)
        self._chat_span.end(error="abandoned on device switch")
        self.view.chat_widget.finalize_assistant_message()  # close the dangling bubble
        self.view.chat_widget.append_system_message(
            f"Abandoned the request still running on the previous device. "
            f"Ask again to run it on {device.label}."
        )
        self.view.chat_widget.set_enabled(True)

    def _hold(self, worker):
        """Keep a worker alive until its thread really ends; dropping it mid-run aborts."""
        self._live_workers.append(worker)
        worker.finished.connect(lambda w=worker: w in self._live_workers and self._live_workers.remove(w))

    def _preload_selected_model(self):
        """Load the newly picked model now, so the next send doesn't absorb the cold load."""
        if not llm_devices.supports_warmup():
            return
        self.view.chat_widget.set_model_loading(True)
        worker = CallableWorker(warm_model, self._llm_model)
        # Overwrites a possibly still-running warm-up, so _hold keeps that one referenced.
        self._model_load_worker = worker
        self._hold(worker)
        worker.finished.connect(lambda w=worker: self._on_model_loaded(w))
        worker.start()

    def _on_model_loaded(self, worker):
        # A newer selection can start first; only the most recent clears the indicator.
        if worker is not self._model_load_worker:
            return
        self._model_load_worker = None
        self.view.chat_widget.set_model_loading(False)

    def _available_subjects(self) -> list[str]:
        """Subject folders rarely change mid-session — cache the disk scan."""
        if self._available_subjects_cache is None:
            self._available_subjects_cache = list_subjects()
        return self._available_subjects_cache

    def _resolve_subject(self, subject_id: str) -> str | None:
        """Exact folder name, else the single one matching case-insensitively or by number ('3' -> 'QLGE03_')."""
        names = self._available_subjects()
        if subject_id in names:
            return subject_id
        digits = re.findall(r"\d+", subject_id)
        matches = [n for n in names if n.lower() == subject_id.lower()
                   or (len(digits) == 1 and int(digits[0]) in map(int, re.findall(r"\d+", n)))]
        return matches[0] if len(matches) == 1 else None

    def _context_snapshot(self) -> str:
        widget = self.view.display_area_widget
        time_frame = None
        if self.model.num_time_frames:
            time_frame = (widget.current_time_index, self.model.num_time_frames)
        loading = self.dicom_controller.loading_series
        return build_state_block(
            subject=self.model.subject_folder.name if self.model.subject_folder else "none",
            loaded_series=sorted(set(self.model.series) - loading),
            loading_series=sorted(loading),
            time_frame=time_frame,
            grid=widget.grid_size,
            panels=self.dicom_controller.panel_assignments(),
            analysis=self.model.analysis,
            segmentable=self.model.segmentable,
            level_matches=self._level_matches(),
            playing=self.view.is_playing(),
            split=widget.split,
            speed=self.view.cine_speed,
        )

    def _level_matches(self) -> dict[str, list[int]]:
        """For each other loaded short-axis stack, its slices nearest the CINE_SAX base, mid and apex
        slices at the current frame (0-based), so the model can show those levels in it too."""
        sax = self.model.analysis.get("CINE_SAX", {})
        widget = self.view.display_area_widget
        levels = base_mid_apex(sax, widget.current_time_index) if "lv_slices" in sax else None
        if levels is None:
            return {}
        matches = {}
        for key in sorted(set(self.model.series) - self.dicom_controller.loading_series):
            if key != "CINE_SAX" and key.endswith("_SAX") and widget.is_series_registered(key):
                slices = [widget.matching_slice("CINE_SAX", z, key) for z in levels]
                if None not in slices:
                    matches[key] = slices
        return matches

    def _messages_for_request(self) -> list[dict]:
        # Attached to the latest user turn, not the system prompt, so the
        # system prompt itself stays stable/cacheable.
        history = list(self._history)
        if history and history[-1]["role"] == "user":
            history[-1] = {**history[-1], "content": f"{self._context_snapshot()}\n\n{history[-1]['content']}"}
        elif history and history[-1]["role"] == "tool":
            # A follow-up after tool calls: the screen and the subject may have changed since the user turn.
            history[-1] = {**history[-1], "content": f"{history[-1]['content']}\n\nNow:\n{self._context_snapshot()}"}
        return build_messages(history)

    def _build_tools(self) -> list[dict]:
        ctx = ToolContext.from_app(self.model, self.view, self.dicom_controller.loading_series)
        return build_tools(ctx)

    def send_message(self, user_text: str):
        if self._worker is not None and self._worker.isRunning():
            return  # one turn at a time for this bare version

        self._history.append({"role": "user", "content": user_text})
        self._trim_history()
        self._chat_span = study_log.Span("chat", text=user_text, model=self._llm_model)

        self.view.chat_widget.set_enabled(False)
        self.view.chat_widget.begin_assistant_message()
        self._start_worker(self._messages_for_request())

    def _trim_history(self):
        """Keep only the last _MAX_HISTORY_TURNS user turns, cutting only at
        a user-message boundary — a raw entry-count slice can land between an
        assistant tool_calls message and its paired tool result, sending
        Ollama a malformed, orphaned half of that pair."""
        user_indices = [i for i, m in enumerate(self._history) if m["role"] == "user"]
        if len(user_indices) > _MAX_HISTORY_TURNS:
            cutoff = user_indices[-_MAX_HISTORY_TURNS]
            self._history = self._history[cutoff:]

    def _start_worker(self, messages: list[dict]):
        worker_cls = llm_devices.worker_class()
        self._worker = worker_cls(messages, tools=self._build_tools(), model=self._llm_model,
                                  think=self._think)
        self._worker.token_received.connect(self.view.chat_widget.append_assistant_chunk)
        self._worker.thinking_received.connect(self.view.chat_widget.append_thinking_chunk)
        self._worker.token_received.connect(lambda _: self._chat_span.mark("first_token"), Qt.SingleShotConnection)
        self._worker.tool_calls_ready.connect(self._on_tool_calls)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        # `finished`, not `finished_ok`/`error`: those fire before the thread
        # actually stops, so clearing the reference there races teardown.
        self._worker.finished.connect(self._clear_worker)
        self._worker.start()

    def _clear_worker(self):
        self._worker = None

    def _on_finished(self):
        text = self.view.chat_widget.finalize_assistant_message()
        if text:
            self._history.append({"role": "assistant", "content": text})
        self._chat_span.end(reply=text)
        self.view.chat_widget.set_enabled(True)

    def _on_error(self, message: str):
        self._chat_span.end(error=message)
        self.view.chat_widget.finalize_assistant_message()
        self.view.chat_widget.append_system_message(f"Error: {message}")
        self.view.chat_widget.set_enabled(True)

    def _on_tool_calls(self, calls: list):
        # load_subject loads in the background; remaining calls wait until its series are known.
        self._warm_model_in_background()
        self._execute_tool_sequence(list(calls))

    def _warm_model_in_background(self):
        """Gets Ollama resident with the right num_ctx while the tool call(s)
        below are still running, so the follow-up chat request doesn't also
        pay a cold-load penalty."""
        if not llm_devices.supports_warmup():
            return
        if self._warmup_worker is not None and self._warmup_worker.isRunning():
            return
        worker = CallableWorker(warm_model, self._llm_model)
        self._warmup_worker = worker
        worker.finished.connect(lambda: setattr(self, "_warmup_worker", None))
        worker.start()

    def _execute_tool_sequence(self, remaining: list):
        if not remaining:
            self.view.chat_widget.begin_assistant_message()
            self._start_worker(self._messages_for_request())
            return

        name, arguments = remaining.pop(0)
        bubble = self.view.chat_widget.append_tool_call(name, arguments)

        with study_log.acting("assistant"):
            if name == "load_subject":
                self._execute_load_subject(arguments, remaining, bubble)
                return  # continues once the patient's series are known; they may still load
            result = self._execute_tool(name, arguments)
            modality = str(arguments.get("modality", "")).upper()
            if name == "run_segmentation" and self.dicom_controller.is_segmenting(modality):
                bubble.set_result(result)
                self._poll_until(lambda: not self.dicom_controller.is_segmenting(modality), 600,
                                 lambda: self._finish_segmentation(modality, arguments, remaining, bubble))
                return  # continues once the segmentation has finished
        bubble.set_result(result)
        self._append_tool_history(name, arguments, result)
        self._execute_tool_sequence(remaining)

    def _append_tool_history(self, name: str, arguments: dict, result: str):
        study_log.log("tool_call", name=name, arguments=arguments, result=result)
        self._history.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
        })
        self._history.append({"role": "tool", "content": result})

    def _execute_load_subject(self, arguments: dict, remaining: list, bubble):
        subject_id = str(arguments.get("subject_id", ""))
        name = self._resolve_subject(subject_id)
        if name is None:
            result = f"No single subject folder matches '{subject_id}'. Call list_subjects for the exact names."
            self._finish_load_subject(result, arguments, remaining, bubble)
            return
        path = data_dir() / name

        def _on_ready(success: bool):
            result = (
                f"Loaded subject {name} from {path}."
                if success
                else f"Failed to load subject {name} (still busy loading a previous subject, or load failed)."
            )
            self._finish_load_subject(result, arguments, remaining, bubble)

        self.dicom_controller.load_patient(str(path), on_ready=_on_ready)

    def _finish_load_subject(self, result: str, arguments: dict, remaining: list, bubble):
        bubble.set_result(result)
        loaded = result.startswith("Loaded")
        if loaded:
            result += _drop_planned(remaining, "this subject loaded")
        self._append_tool_history("load_subject", arguments, result)
        # The cached segmentations (ED/ES, LV slices) are read as the CINE and LGE series load.
        segmentable = set(self.model.segmentable)
        self._poll_until(lambda: not (self.dicom_controller.loading_series & segmentable), 60,
                         lambda: self._execute_tool_sequence(remaining))

    def _poll_until(self, done, timeout_s: float, then, started: float | None = None):
        """then(), once done() is true or after timeout_s seconds."""
        started = time.monotonic() if started is None else started
        if done() or time.monotonic() - started > timeout_s:
            then()
            return
        QTimer.singleShot(200, lambda: self._poll_until(done, timeout_s, then, started))

    def _finish_segmentation(self, modality: str, arguments: dict, remaining: list, bubble):
        widget = self.view.display_area_widget
        if widget.has_segmentation(modality):
            panels = [p["idx"] for p in widget.panel_assignments() if p["series_key"] == modality]
            result = f"{modality} segmented; its contours show in {_panels_text(panels) if panels else 'no panel yet'}."
        else:
            result = f"Not done: the {modality} segmentation failed or is still running (see the app)."
        result += _drop_planned(remaining, "the segmentation finished", keep="run_segmentation")
        bubble.set_result(result)
        self._append_tool_history("run_segmentation", arguments, result)
        self._execute_tool_sequence(remaining)

    def _layout_result(self, widget) -> str:
        assignments = widget.panel_assignments()
        study_log.log("layout", grid=list(widget.grid_size), split=widget.split, panels=assignments)
        empty = [p["idx"] for p in assignments if p["series_key"] is None]
        return (f"Layout now {grid_text(*widget.grid_size, widget.split)}; empty: "
                f"{_panels_text(empty) if empty else 'none'}.")

    def _set_window(self, series_key: str, mode: str, steps: int, panels: list[int] | None = None) -> str:
        widget = self.view.display_area_widget
        if not widget.is_series_registered(series_key):
            return f"Series '{series_key}' is not loaded."
        if panels is not None:
            shown = widget.panel_assignments()
            wrong = [i for i in panels if not 0 <= i < len(shown) or shown[i]["series_key"] != series_key]
            if wrong:
                return f"Not done: panels {', '.join(map(str, wrong))} do not show {series_key}."
        if mode == "reset":
            window = widget.window(series_key, default=True)
        elif mode == "null_myocardium":
            if series_key != "LGE_SAX":
                return ("null_myocardium works on LGE_SAX only; for other series use darker, brighter, "
                        "more_contrast or less_contrast.")
            seg = self.dicom_controller.segmentation_for(series_key)
            if seg is None:
                return "Not done: null_myocardium needs an LGE_SAX segmentation. Ask the user whether to run it."
            window = null_myocardium_window(widget.volume(series_key), seg)
            if window is None:
                return "The LGE_SAX segmentation has no healthy myocardium to set the window from."
        elif mode in WINDOW_STEPS:
            steps = max(1, min(5, steps))
            if panels is not None:  # each panel steps from the window it shows now
                for i in panels:
                    widget.set_window(series_key, step_window(widget.window(series_key, panel=i), mode, steps), [i])
                return f"{series_key} window ({mode}) set in {_panels_text(panels)} only, as their own window."
            window = step_window(widget.window(series_key), mode, steps)
        else:
            return f"Unknown mode '{mode}'."
        widget.set_window(series_key, window, panels)
        center, width = widget.window(series_key, panel=panels[0] if panels else None)
        where = f" in {_panels_text(panels)} only, as their own window" if panels else " in every panel of it"
        result = f"{series_key} window set ({mode}){where}: level {center:.0f}, width {width:.0f}"
        if mode == "null_myocardium":
            healthy = widget.volume(series_key)[seg == 1]
            grey = 100 * (float(np.median(healthy)) - (center - width / 2)) / width
            result += f"; healthy myocardium at about {max(grey, 0):.0f}% grey, so dark with its noise visible"
        return result + ". The user can fine-tune it with darker, brighter or more contrast."

    def _execute_tool(self, name: str, arguments: dict) -> str:
        try:
            if name == "list_subjects":
                names = self._available_subjects()
                return f"{len(names)} subjects in {data_dir()}: {', '.join(names)}" if names else f"No subject folders in {data_dir()}."

            if name == "run_segmentation":
                modality = str(arguments.get("modality", "")).upper()
                if modality not in self.model.segmentable:
                    return (
                        f"Unknown modality '{modality}' — expected "
                        f"{', '.join(repr(m) for m in self.model.cine_modalities)} or 'LGE_SAX'."
                    )
                loading = modality in self.dicom_controller.loading_series
                if modality not in self.model.series and not loading:
                    return f"'{modality}' is not loaded for this subject."
                widget = self.view.display_area_widget
                panels = widget.show_segmentation(modality)
                where = _panels_text(panels) if panels else f"any panel that shows {modality}"
                if widget.has_segmentation(modality):
                    return f"{modality} was already segmented; its contours now show in {where}."
                if not self.dicom_controller.run_segmentation(modality):
                    return f"Not done yet: {modality} segmentation is already running or queued."
                if widget.has_segmentation(modality):
                    return f"{modality} segmented (from the cache); its contours show in {where}."
                return f"Segmenting {modality} now; the app waits until it has finished."

            if name == "apply_preset":
                preset_name = str(arguments["preset"])
                result = self.dicom_controller.apply_preset(preset_name)
                if result is None:
                    return f"Unknown preset '{preset_name}'."
                filled, total = result
                if filled == 0:
                    return f"Could not apply '{preset_name}': none of its required series are loaded for this subject."
                if filled < total:
                    return (
                        f"Applied '{preset_name}' preset ({filled}/{total} panels filled; "
                        f"the rest need series not loaded for this subject)."
                    )
                return f"Applied '{preset_name}' preset."

            if name == "display_series":
                widget = self.view.display_area_widget
                panel_idx = int(arguments["panel_idx"])
                series_key = str(arguments["series_key"])
                axis = str(arguments.get("axis", "transverse"))
                if not (0 <= panel_idx < widget.n_panels):
                    return f"panel_idx out of range (grid has {widget.n_panels} panels)."
                if series_key in self.dicom_controller.loading_series:
                    self.dicom_controller.show_series_when_loaded(panel_idx, series_key, axis)
                    return f"{series_key} is still loading; panel {panel_idx} shows it ({axis}) once loaded."
                if not widget.is_series_registered(series_key):
                    return f"Series '{series_key}' is not loaded."
                n = widget.n_slices(series_key, axis)
                z = None
                if arguments.get("match_panel") is not None:
                    other = int(arguments["match_panel"])
                    if not 0 <= other < widget.n_panels:
                        return f"match_panel out of range (grid has {widget.n_panels} panels)."
                    source = widget.panel_assignments()[other]
                    if source["series_key"] is None or "slice" not in source:
                        return f"Panel {other} shows no stack to match."
                    z = widget.matching_slice(source["series_key"], source["slice"], series_key)
                    if z is None or axis != "transverse":
                        return f"{series_key} is not parallel to {source['series_key']}, so no slice matches."
                elif arguments.get("slice") is not None:
                    if not 1 <= int(arguments["slice"]) <= n:
                        return f"slice out of range (1-{n} for {series_key})."
                    z = int(arguments["slice"]) - 1
                widget.set_panel_series(panel_idx, series_key, axis=axis, slice_idx=z)
                at = f" at slice {widget.panel_assignments()[panel_idx]['slice'] + 1}/{n}" if n > 1 else ""
                return f"Displaying {series_key} ({axis}) in panel {panel_idx}{at}."

            if name == "set_time_frame":
                widget = self.view.display_area_widget
                total = widget.n_frames(CINE_TIME_AXIS)
                frame = int(arguments["frame"])
                if not (1 <= frame <= total):
                    return f"frame out of range (1-{total})."
                if widget.current_time_index == frame - 1:
                    return f"Already at frame {frame}."
                widget.set_time_index(frame - 1, CINE_TIME_AXIS)
                return f"Showing cine frame {frame}/{total}."

            if name == "set_window":
                panels = _panel_list(arguments, self.view.display_area_widget.n_panels) if arguments.get("panels") else None
                if isinstance(panels, str):
                    return panels
                return self._set_window(str(arguments.get("series_key", "")), str(arguments.get("mode", "")),
                                        int(arguments.get("steps") or 1), panels)

            if name == "set_contours":
                widget = self.view.display_area_widget
                shown = bool(arguments["shown"])
                panels = _panel_list(arguments, widget.n_panels)
                if isinstance(panels, str):
                    return panels
                assignments = widget.panel_assignments()
                done, missing = [], []
                for i in panels:
                    key = assignments[i]["series_key"]
                    if key is None:
                        continue
                    if shown and not widget.has_segmentation(key):
                        missing.append(i)
                        continue
                    widget.set_panel_contours(i, shown)
                    done.append(i)
                result = (f"Contours {'shown' if shown else 'hidden'} in {_panels_text(done)}."
                          if done else "No contours changed.")
                if missing:
                    result += (f" Not done for {_panels_text(missing)}: their series has no "
                               "segmentation. Ask the user whether to run it.")
                return result

            if name == "set_playback":
                widget = self.view.display_area_widget
                if not widget.n_frames(CINE_TIME_AXIS):
                    return "No cine is loaded to play."
                playing = bool(arguments["playing"])
                if arguments.get("speed") is not None:
                    speed = float(arguments["speed"])
                    if speed not in CINE_SPEEDS:
                        return f"speed must be one of {', '.join(f'{s:g}' for s in CINE_SPEEDS)}."
                    self.view.set_cine_speed(speed)
                widget.drive_cine_time()
                if self.view.is_playing() != playing:
                    self.view.toggle_cine()
                if playing and not self.view.is_playing():
                    return "Not done: the cines have a single frame, so there is nothing to play."
                if playing:
                    return f"Playing the cines at {self.view.cine_speed:g}x (heartbeats per second)."
                return f"Cines stopped at frame {widget.current_time_index + 1}."

            if name == "set_grid":
                widget = self.view.display_area_widget
                rows, cols = int(arguments["rows"]), int(arguments["cols"])
                if not (1 <= rows <= MAX_GRID_ROWS and 1 <= cols <= MAX_GRID_COLS):
                    return f"Grid out of range: 1-{MAX_GRID_ROWS} rows and 1-{MAX_GRID_COLS} columns."
                widget.set_grid_size(rows, cols, (int(arguments.get("shift_rows") or 0),
                                                  int(arguments.get("shift_cols") or 0)))
                return self._layout_result(widget)

            if name == "set_split_grid":
                widget = self.view.display_area_widget
                split = tuple(int(arguments[k]) for k in ("left_rows", "left_cols", "right_rows", "right_cols"))
                if not all(1 <= n <= MAX_HALF for n in split):
                    return f"Each half has 1-{MAX_HALF} rows and 1-{MAX_HALF} columns."
                widget.set_split_layout(split)
                return self._layout_result(widget)

            if name == "zoom":
                widget = self.view.display_area_widget
                mode = str(arguments["mode"])
                if mode == "percent" and arguments.get("percent") is None:
                    return "mode 'percent' needs a percent value."
                panels = _panel_list(arguments, widget.n_panels)
                if isinstance(panels, str):
                    return panels
                done = widget.zoom_panels(panels, mode, float(arguments.get("percent") or 100), self.model.analysis)
                assignments = widget.panel_assignments()
                skipped = [i for i in panels if i not in done and assignments[i]["series_key"]]
                result = f"Zoomed {_panels_text(done) if done else 'no panel'} ({mode})."
                if skipped:
                    result += f" Not done for {_panels_text(skipped)}: no segmentation of their plane to zoom to."
                return result

            if name == "save_segmentation":
                self.dicom_controller.save_segmentation()
                return "Requested save (check the app for the result or any errors)."

            return f"Unknown tool: {name}"
        except Exception as e:
            logging.error(f"Tool {name} failed: {e}")
            return f"Tool {name} failed: {e}"
