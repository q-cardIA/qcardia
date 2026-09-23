"""ChatController — Qt orchestration around the LLM interaction layer in `llm/`."""
from __future__ import annotations

import logging
import re

from llm.context import build_state_block
from llm.prompt import build_messages
from llm.tools import ToolContext, build_tools
from model.background_worker import CallableWorker
from model.llm_client import OllamaWorker, warm_model
from utils.app_config import OLLAMA_MODEL, data_dir, list_subjects

_MAX_HISTORY_TURNS = 10


class ChatController:
    def __init__(self, view, dicom_controller, model):
        self.view = view
        self.dicom_controller = dicom_controller
        self.model = model
        self._history: list[dict] = []
        self._worker = None
        # Read the widget's actual model selection rather than assume OLLAMA_MODEL.
        self._llm_model = self.view.chat_widget.current_tag() or OLLAMA_MODEL
        self._available_subjects_cache: list[str] | None = None
        self._warmup_worker: CallableWorker | None = None  # see _warm_model_in_background

        self.view.chat_widget.message_submitted.connect(self.send_message)
        self.view.chat_widget.model_changed.connect(self._on_model_changed)

    def _on_model_changed(self, model_name: str):
        self._llm_model = model_name

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
            time_frame = (self.dicom_controller.current_time_index, self.model.num_time_frames)
        return build_state_block(
            subject=self.model.folder_path.name if self.model.folder_path else "none",
            loaded_series=sorted(self.model.series),
            time_frame=time_frame,
            grid=widget.grid_size,
            panels=widget.panel_assignments(),
        )

    def _messages_for_request(self) -> list[dict]:
        # Attached to the latest user turn, not the system prompt, so the
        # system prompt itself stays stable/cacheable.
        history = list(self._history)
        if history and history[-1]["role"] == "user":
            history[-1] = {**history[-1], "content": f"{self._context_snapshot()}\n\n{history[-1]['content']}"}
        return build_messages(history)

    def _build_tools(self) -> list[dict]:
        ctx = ToolContext.from_app(self.model, self.view)
        return build_tools(ctx)

    def send_message(self, user_text: str):
        if self._worker is not None and self._worker.isRunning():
            return  # one turn at a time for this bare version

        self._history.append({"role": "user", "content": user_text})
        self._trim_history()

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
        self._worker = OllamaWorker(messages, tools=self._build_tools(), model=self._llm_model)
        self._worker.token_received.connect(self.view.chat_widget.append_assistant_chunk)
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
        self.view.chat_widget.set_enabled(True)

    def _on_error(self, message: str):
        self.view.chat_widget.finalize_assistant_message()
        self.view.chat_widget.append_system_message(f"Error: {message}")
        self.view.chat_widget.set_enabled(True)

    def _on_tool_calls(self, calls: list):
        # load_subject loads in the background; remaining calls wait for it.
        self._warm_model_in_background()
        self._execute_tool_sequence(list(calls))

    def _warm_model_in_background(self):
        """Gets Ollama resident with the right num_ctx while the tool call(s)
        below are still running, so the follow-up chat request doesn't also
        pay a cold-load penalty."""
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

        if name == "load_subject":
            self._execute_load_subject(arguments, remaining, bubble)
            return  # continues once the background load actually finishes

        result = self._execute_tool(name, arguments)
        bubble.set_result(result)
        self._append_tool_history(name, arguments, result)
        self._execute_tool_sequence(remaining)

    def _append_tool_history(self, name: str, arguments: dict, result: str):
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

        def _on_complete(success: bool):
            result = (
                f"Loaded subject {name} from {path}."
                if success
                else f"Failed to load subject {name} (still busy loading a previous subject, or load failed)."
            )
            self._finish_load_subject(result, arguments, remaining, bubble)

        self.dicom_controller.load_patient(str(path), on_complete=_on_complete)

    def _finish_load_subject(self, result: str, arguments: dict, remaining: list, bubble):
        bubble.set_result(result)
        self._append_tool_history("load_subject", arguments, result)
        self._execute_tool_sequence(remaining)

    def _execute_tool(self, name: str, arguments: dict) -> str:
        try:
            if name == "list_subjects":
                names = self._available_subjects()
                return f"{len(names)} subjects in {data_dir()}: {', '.join(names)}" if names else f"No subject folders in {data_dir()}."

            if name == "run_segmentation":
                modality = str(arguments.get("modality", "")).upper()
                if modality not in (*self.model.cine_modalities, "LGE_SAX"):
                    return (
                        f"Unknown modality '{modality}' — expected "
                        f"{', '.join(repr(m) for m in self.model.cine_modalities)} or 'LGE_SAX'."
                    )
                if modality not in self.model.series:
                    return f"'{modality}' is not loaded for this subject."
                if not self.dicom_controller.run_segmentation(modality):
                    return f"A segmentation is already running — try requesting {modality} again once it finishes."
                return f"Requested {modality} segmentation (check the app for the result or any errors)."

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
                if not widget.is_series_registered(series_key):
                    return f"Series '{series_key}' is not loaded."
                widget.set_panel_series(panel_idx, series_key, axis=axis)
                return f"Displaying {series_key} ({axis}) in panel {panel_idx}."

            if name == "set_time_frame":
                time_index = int(arguments["time_index"])
                max_val = self.view.time_slider.maximum()
                if not (0 <= time_index <= max_val):
                    return f"time_index out of range (0-{max_val})."
                if self.view.time_slider.value() == time_index:
                    return f"Already at time frame {time_index}."
                self.view.time_slider.setValue(time_index)  # triggers the normal slider signal chain
                return f"Jumped to time frame {time_index}."

            if name == "save_segmentation":
                self.dicom_controller.save_segmentation()
                return "Requested save (check the app for the result or any errors)."

            return f"Unknown tool: {name}"
        except Exception as e:
            logging.error(f"Tool {name} failed: {e}")
            return f"Tool {name} failed: {e}"
