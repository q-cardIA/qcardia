"""ChatController: wires a minimal chat panel to whichever LLM server is selected.

No subprocess, no separate conda env: this talks to Ollama exactly like
qcardia.disambiguate does (see model/ollama_client.py), just streaming and with
native tool-calling wired to a handful of existing DICOMController actions.
The worker class comes from utils.llm_devices, so a device speaking another wire
format changes nothing below.
"""
from __future__ import annotations

import copy
import logging
from contextlib import suppress

from model.background_worker import CallableWorker
from model.ollama_client import warm_model
from utils import llm_devices
from utils.app_config import discover_subject_numbers, subject_folder_path

# Plane abbreviations used throughout the app's series keys (e.g. CINE_SAX).
_PLANE_ABBREVIATIONS = {
    "SAX": "Short Axis",
    "2CH": "2-Chamber",
    "3CH": "3-Chamber",
    "4CH": "4-Chamber",
}

# Ported from cardiagent's prompt (see prompt_builder.py/tool_registry.py)
# where these were the highest-value rules: closed series-key grounding,
# question-vs-command triggering, never-clamp-invalid-args, and the
# already-done state check. Left out: the agent loop, dynamic per-state
# example generation, set_grid, and the WBLGE/DBLGE glossary — none of
# which apply to this app's tool surface.
SYSTEM_PROMPT = (
    "You are an assistant embedded in a cardiac MRI DICOM viewer. You help "
    "navigate the viewer and answer questions about the loaded study, and you "
    "call tools to drive the viewer.\n"
    "\n"
    "SERIES KEYS — only use series_key values listed EXACTLY under 'Loaded "
    "series' in the state block. If the user asks for a series that is not "
    "listed, call no tools and say it is unavailable for this subject. Never "
    "invent, approximate, or substitute a key — a similar-sounding series is a "
    "different acquisition and is WRONG (e.g. LGE is not perfusion; T2* is not "
    "T1 mapping).\n"
    "\n"
    "QUESTIONS — if the user is asking rather than instructing ('what', "
    "'which', 'how', 'why', 'is this', 'explain', or ends with '?'), answer in "
    "text and call NO tools. You can answer questions about the viewer and the "
    "loaded data itself: which series/subjects/tools are available, what an "
    "abbreviation means, what is currently displayed. You are not a diagnostic "
    "tool: for clinical or interpretive questions (e.g. whether a measurement "
    "is abnormal, what a finding means) or any other out-of-scope request "
    "(report writing, diagnosis, clinical judgment), reply in one sentence "
    "that you help navigate the viewer, not interpret the data. Either way, a "
    "question never triggers a tool call by itself: 'what shows LV function?' "
    "is a question, 'show LV function' is a command.\n"
    "\n"
    "INVALID ARGUMENTS — subject_number not in the available list, panel_idx "
    "outside the current grid, time_index out of range: reply in text only and "
    "do NOT emit the tool call. Never clamp an out-of-range value to a valid "
    "one.\n"
    "\n"
    "ALREADY DONE — check the state block before acting. If the requested "
    "series is already in the requested panel, or the requested layout is "
    "already applied, call no tools and say so.\n"
    "\n"
    "MULTI-STEP — a request can need more than one action (e.g. 'load subject "
    "5, apply the functional preset, and segment the cine stack'). Call the "
    "first tool now. Every time a tool result comes back, check the request "
    "again: if a step is still pending, call its tool in that same reply — "
    "do not stop to narrate or ask before doing it. Repeat until every step "
    "is done, then reply with one short summary line and no tool call. "
    "Calling several tools in one reply is fine when you already know you'll "
    "need all of them, but continuing one tool at a time after each result "
    "is just as correct — never leave a step for the user to ask about with "
    "a follow-up like 'continue'.\n"
    "\n"
    "VAGUE — if intent is unclear, ask one short clarifying question and call "
    "no tools. When in doubt, call no tools.\n"
    "\n"
    "When you do call a tool, keep any accompanying text to one short sentence "
    "or leave it empty.\n"
    "\n"
    "PANEL ADDRESSING — panel_idx = row * cols + col, both 0-indexed, "
    "left-to-right then top-to-bottom. For a 2x4 grid: top-right = row 0, "
    "col 3 -> 0*4+3 = panel 3; bottom-left = row 1, col 0 -> 1*4+0 = panel 4.\n"
    "PLANE ABBREVIATIONS: "
    + ", ".join(f"{k}={v}" for k, v in _PLANE_ABBREVIATIONS.items())
    + ". These name the acquisition (part of series_key), not the reslicing axis."
)

# Two static hand-written exchanges (not cardiagent's dynamic per-state
# generator): the first shows a plain navigational question answered in text
# with no tool call (the assistant is a viewer copilot, not a diagnostic
# tool); the second shows a THREE-step command resolved by continuing to call
# the next tool after each tool result comes back, one at a time, instead of
# narrating an intention and stopping — the exact failure mode this teaches
# against is a model doing step 1, then replying "I'll do the rest next" and
# waiting for the user to say "continue". Never persisted into real history.
_FEW_SHOT_EXAMPLE = [
    {"role": "user", "content": "What image data do we have for this subject?"},
    {
        "role": "assistant",
        "content": (
            "This subject has CINE_SAX, CINE_2CH, CINE_4CH, and LGE_SAX loaded — "
            "see the 'Loaded series' list for the current subject."
        ),
    },
    {
        "role": "user",
        "content": "Load subject 3, apply the functional preset, and segment the cine stack",
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "load_subject", "arguments": {"subject_number": 3}}}],
    },
    {"role": "tool", "content": "Loaded subject 3 from .../QLGE03_."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "apply_preset", "arguments": {"preset": "functional"}}}],
    },
    {"role": "tool", "content": "Applied 'functional' preset."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "run_segmentation", "arguments": {"modality": "CINE_SAX"}}}
        ],
    },
    {"role": "tool", "content": "Requested CINE_SAX segmentation."},
    {
        "role": "assistant",
        "content": (
            "Loaded subject 3, applied the functional preset, and started CINE_SAX "
            "segmentation."
        ),
    },
]

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "load_subject",
            "description": (
                "Load a patient study by subject number. This replaces the current "
                "patient and clears all panels. Only use numbers listed as "
                "available in the state block."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_number": {
                        "type": "integer",
                        "description": "Subject number as an integer, e.g. 5 for QLGE05_.",
                    }
                },
                "required": ["subject_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_segmentation",
            "description": (
                "Run AI segmentation on a loaded stack. modality is the exact "
                "series_key of the stack to segment, e.g. CINE_SAX or LGE_SAX — "
                "same series_key values used by display_series."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "modality": {
                        "type": "string",
                        "enum": ["CINE_SAX", "LGE_SAX"],
                        "description": (
                            "Exact series_key of the stack to segment: CINE_SAX or "
                            "LGE_SAX."
                        ),
                    },
                },
                "required": ["modality"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_preset",
            "description": (
                "Apply a clinical hanging protocol: sets the grid AND fills the "
                "panels, replacing the current layout. Prefer this over several "
                "display_series calls. To also show a custom panel, call "
                "apply_preset first, then display_series. Check the state block "
                "that the preset's series are loaded first; if not, call no tools "
                "and say the data is unavailable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "preset": {
                        "type": "string",
                        "enum": ["functional", "viability", "perfusion"],
                        "description": (
                            "functional: ventricular/LV/RV function, ejection "
                            "fraction, volumes, wall motion (CINE only). "
                            "viability: scar, late gadolinium enhancement, "
                            "infarct, fibrosis, transmurality (needs LGE). "
                            "perfusion: perfusion, ischaemia, CAD, stress testing "
                            "(needs PERF_STRESS_SAX or PERF_REST_SAX)."
                        ),
                    }
                },
                "required": ["preset"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "display_series",
            "description": (
                "Show one loaded series in one panel. series_key must appear "
                "exactly in 'Loaded series' in the state block."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "panel_idx": {
                        "type": "integer",
                        "description": "0-indexed panel in the current grid: row * cols + col.",
                    },
                    "series_key": {
                        "type": "string",
                        "description": (
                            "Exact key from 'Loaded series', e.g. CINE_SAX or "
                            "LGE_4CH. Must match a listed key character for "
                            "character — never invent one."
                        ),
                    },
                    "axis": {
                        "type": "string",
                        "enum": ["transverse", "coronal", "sagittal"],
                        "description": (
                            "Reslicing direction, almost always 'transverse'. This "
                            "is NOT the cardiac plane — the cardiac plane is part "
                            "of series_key. Only use coronal/sagittal if the user "
                            "explicitly asks to reslice."
                        ),
                    },
                },
                "required": ["panel_idx", "series_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_time_frame",
            "description": "Jump all cine panels to a specific time frame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_index": {
                        "type": "integer",
                        "description": (
                            "0-based frame index, range 0 to (total frames - 1). "
                            "The state block shows 'Time frame: current / total' "
                            "using 1-based numbering, so subtract 1."
                        ),
                    }
                },
                "required": ["time_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_segmentation",
            "description": (
                "Save the current segmentation to disk. Only call after a "
                "segmentation has been run."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_MAX_HISTORY_TURNS = 10


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
        self._available_subjects_cache: list[int] | None = None
        self._warmup_worker: CallableWorker | None = None  # see _warm_model_in_background
        self._model_load_worker: CallableWorker | None = None  # see _preload_selected_model
        self._live_workers: list = []  # anything whose thread must outlive its variable

        self.view.chat_widget.message_submitted.connect(self.send_message)
        self.view.chat_widget.model_changed.connect(self._on_model_changed)
        self.view.chat_widget.device_changed.connect(self._on_device_changed)

    def _on_model_changed(self, model_name: str):
        self._llm_model = model_name
        self._preload_selected_model()

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
        for signal in (worker.token_received, worker.tool_calls_ready,
                       worker.finished_ok, worker.error, worker.finished):
            with suppress(RuntimeError, TypeError):
                signal.disconnect()  # no args = drop every slot on that signal
        self._hold(worker)
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

    def _available_subjects(self) -> list[int]:
        """Subject folders rarely change mid-session — cache the disk scan
        instead of re-running it on every chat turn."""
        if self._available_subjects_cache is None:
            self._available_subjects_cache = discover_subject_numbers()
        return self._available_subjects_cache

    def _context_snapshot(self) -> str:
        """Labelled, multi-line block (not a single comma-joined sentence) so
        the system prompt's rules can reference specific fields ('the state
        block', 'Loaded series') — mirrors cardiagent's STATE/SERIES layout."""
        loaded = sorted(self.model.series.keys())
        subject = self.model.folder_path.name if self.model.folder_path else "none"

        available_subjects = self._available_subjects()
        available_label = _format_ranges(available_subjects) if available_subjects else "none found"

        if self.model.num_time_frames:
            current = self.dicom_controller.current_time_index
            total = self.model.num_time_frames
            time_line = (
                f"Time frame: {current + 1} / {total} (1-based display; "
                f"set_time_frame uses 0-{total - 1})"
            )
        else:
            time_line = "Time frame: n/a"

        widget = self.view.display_area_widget
        rows, cols = widget.grid_size
        panel_lines = []
        for panel in widget.panel_assignments():
            if panel["series_key"] is None:
                panel_lines.append(f"  {panel['idx']}: empty")
            else:
                seg_note = "  seg=True" if panel["has_segmentation"] else ""
                panel_lines.append(
                    f"  {panel['idx']}: {panel['series_key']} / {panel['axis']}{seg_note}"
                )

        return (
            "STATE:\n"
            f"Subject: {subject}  |  Available: {available_label}\n"
            f"Loaded series: {', '.join(loaded) if loaded else 'none'}\n"
            f"{time_line}\n"
            f"Grid: {rows}x{cols}\n"
            "Panels:\n" + ("\n".join(panel_lines) if panel_lines else "  none")
        )

    def _messages_for_request(self) -> list[dict]:
        # The state snapshot is attached to the latest user turn rather than
        # the system prompt: this keeps it close to the actual question in
        # the token stream (better attention for small local models) and
        # keeps the system prompt itself stable/cacheable across requests.
        history = list(self._history)
        if history and history[-1]["role"] == "user":
            last = history[-1]
            history[-1] = {
                "role": "user",
                "content": f"{self._context_snapshot()}\n\n{last['content']}",
            }
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *_FEW_SHOT_EXAMPLE,
            *history,
        ]

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
        a user-message boundary. A raw entry-count slice can land between an
        assistant tool_calls message and its paired tool-result message,
        sending Ollama a malformed, orphaned half of that pair."""
        user_indices = [i for i, m in enumerate(self._history) if m["role"] == "user"]
        if len(user_indices) > _MAX_HISTORY_TURNS:
            cutoff = user_indices[-_MAX_HISTORY_TURNS]
            self._history = self._history[cutoff:]

    def _build_tools(self) -> list[dict]:
        """Only offer tools whose preconditions are currently met, so the model
        isn't tempted to call e.g. run_segmentation(modality='lge') with no LGE
        series loaded — mirrors cardiagent's context-filtered tool list."""
        series = self.model.series
        tools = []
        for tool in _TOOLS:
            name = tool["function"]["name"]

            if name == "run_segmentation":
                modalities = [
                    key for key in (*self.model.cine_modalities, "LGE_SAX") if key in series
                ]
                if not modalities:
                    continue
                tool = copy.deepcopy(tool)
                tool["function"]["parameters"]["properties"]["modality"]["enum"] = modalities
                tools.append(tool)
                continue

            preconditions = {
                "display_series": bool(series),
                "set_time_frame": self.model.num_time_frames > 0,
                "save_segmentation": hasattr(series.get("CINE_SAX"), "_segmentation_prediction"),
            }
            if preconditions.get(name, True):
                tools.append(tool)
        return tools

    def _start_worker(self, messages: list[dict]):
        worker_cls = llm_devices.worker_class()
        self._worker = worker_cls(messages, tools=self._build_tools(), model=self._llm_model)
        self._worker.token_received.connect(self.view.chat_widget.append_assistant_chunk)
        self._worker.tool_calls_ready.connect(self._on_tool_calls)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        # Clear the reference on `finished`, not `finished_ok`/`error` — those fire
        # from inside run() before the thread has actually stopped, so dropping the
        # last reference there races the thread teardown and aborts the process.
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
        # Execute every tool call from this turn, in order, before asking the
        # model to continue — so multi-tool requests ("load subject 5 and
        # apply the functional preset") actually run both actions.
        # load_subject loads in the background (see DICOMController.load_patient),
        # so any remaining calls in this batch must wait for it to actually
        # finish rather than firing immediately against not-yet-loaded series.
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

        if name == "load_subject":
            self._execute_load_subject(arguments, remaining, bubble)
            return  # continues once the background load actually finishes

        result = self._execute_tool(name, arguments)
        bubble.set_result(result)
        self._append_tool_history(name, arguments, result)
        self._execute_tool_sequence(remaining)

    def _append_tool_history(self, name: str, arguments: dict, result: str):
        self._history.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
            }
        )
        self._history.append({"role": "tool", "content": result})

    def _execute_load_subject(self, arguments: dict, remaining: list, bubble):
        try:
            subject_number = int(arguments["subject_number"])
            path = subject_folder_path(subject_number)
        except Exception as e:
            result = f"load_subject failed: {e}"
            bubble.set_result(result)
            self._append_tool_history("load_subject", arguments, result)
            self._execute_tool_sequence(remaining)
            return

        if not path.exists():
            result = f"Subject folder not found: {path}"
            bubble.set_result(result)
            self._append_tool_history("load_subject", arguments, result)
            self._execute_tool_sequence(remaining)
            return

        def _on_complete(success: bool):
            result = (
                f"Loaded subject {subject_number} from {path}."
                if success
                else f"Failed to load subject {subject_number} (still busy loading a previous subject, or load failed)."
            )
            bubble.set_result(result)
            self._append_tool_history("load_subject", arguments, result)
            self._execute_tool_sequence(remaining)

        self.dicom_controller.load_patient(str(path), on_complete=_on_complete)

    def _execute_tool(self, name: str, arguments: dict) -> str:
        try:
            if name == "run_segmentation":
                modality = str(arguments.get("modality", "")).upper()
                if modality not in (*self.model.cine_modalities, "LGE_SAX"):
                    return (
                        f"Unknown modality '{modality}' — expected "
                        f"{', '.join(repr(m) for m in self.model.cine_modalities)} or 'LGE_SAX'."
                    )
                if not self.dicom_controller.run_segmentation(modality):
                    return f"A segmentation is already running — try requesting {modality} again once it finishes."
                return f"Requested {modality} segmentation (check the app for the result or any errors)."

            if name == "apply_preset":
                preset_name = str(arguments["preset"])
                self.dicom_controller.apply_preset(preset_name)
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


def _format_ranges(numbers: list[int]) -> str:
    """[1,2,3,7,8,9] -> '1-3, 7-9' — cheaper and easier for a small model to
    check membership against than a raw 60+ element list."""
    if not numbers:
        return ""
    numbers = sorted(numbers)
    ranges = []
    start = prev = numbers[0]
    for n in numbers[1:]:
        if n == prev + 1:
            prev = n
            continue
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = n
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ", ".join(ranges)
