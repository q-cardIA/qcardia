"""Builds a real ChatController via object.__new__() (skips __init__, no Qt
touched) and runs one Ollama tool-calling turn synchronously via
OllamaWorker.run() — real, unmodified prompt/tool-building code, no mock."""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllers.chat_controller import ChatController  # noqa: E402
from utils import llm_devices  # noqa: E402

from fakes import FakeDicomController  # noqa: E402
from test_cases import TestCase  # noqa: E402


def build_controller(test_case: TestCase, llm_model: str | None = None) -> ChatController:
    cc = object.__new__(ChatController)  # skip __init__, no Qt widgets touched
    # Copies: running the calls changes the fakes, and each run must start from the case's state.
    cc.view, cc.model = copy.deepcopy(test_case.view), copy.deepcopy(test_case.model)
    cc.dicom_controller = FakeDicomController(cc.model, cc.view)
    cc._history = []
    cc._worker = None
    cc._llm_model = llm_model or llm_devices.default_model()  # as the app's model picker starts
    cc._available_subjects_cache = list(test_case.available_subjects)
    cc._warmup_worker = None
    return cc


class RunResult:
    def __init__(self, tool_calls: list[dict], assistant_text: str, elapsed_s: float, error: str | None = None):
        self.tool_calls = tool_calls
        self.assistant_text = assistant_text
        self.elapsed_s = elapsed_s
        self.error = error


def run_one(cc: ChatController, user_text: str, think: bool | None = None, timeout: float = 120.0,
            history: list[dict] = (), max_rounds: int = 6) -> RunResult:
    """Runs one user turn as the app does: each reply's tool calls run on the fakes and their results
    go back to the model, until it calls no tool (at most max_rounds replies). A load_subject is
    answered without loading. Captures all tool calls; the fakes then show the screen they led to."""
    cc._history = [*history, {"role": "user", "content": user_text}]
    tools = cc._build_tools()

    # think=None means "use OllamaWorker's own default", not a literal False.
    kwargs = {"think": think} if think is not None else {}

    calls: list[dict] = []
    tokens: list[str] = []
    errors: list[str] = []

    start = time.monotonic()
    for _ in range(max_rounds):
        worker = llm_devices.worker_class()(cc._messages_for_request(), tools=tools, model=cc._llm_model, **kwargs)
        new: list[dict] = []
        tokens.clear()
        worker.tool_calls_ready.connect(lambda c: new.extend({"name": n, "arguments": a} for n, a in c))
        worker.token_received.connect(tokens.append)
        worker.error.connect(errors.append)
        worker.run()  # synchronous — see module docstring
        if errors or not new:
            calls += new
            break
        done, loaded = run_calls(cc, new)
        calls += done
        if loaded:
            break
        tools = cc._build_tools()
    elapsed = time.monotonic() - start

    return RunResult(
        tool_calls=calls,
        assistant_text="".join(tokens),
        elapsed_s=elapsed,
        error=errors[0] if errors else None,
    )


def run_calls(cc: ChatController, calls: list[dict]) -> tuple[list[dict], bool]:
    """Runs calls in order as the app does, into the history; returns the calls that ran and whether a
    subject loaded. A load ends the reply there: the app plans the rest again with the new subject,
    which the fakes cannot load."""
    for i, c in enumerate(calls):
        if c["name"] == "load_subject":
            name = cc._resolve_subject(str(c["arguments"].get("subject_id", "")))
            result = (f"Loaded subject {name}." if name else f"No single subject folder matches "
                      f"'{c['arguments'].get('subject_id')}'. Call list_subjects for the exact names.")
            cc._append_tool_history(c["name"], c["arguments"], result)
            if name:
                cc.loaded_subjects = [*getattr(cc, "loaded_subjects", []), name]
                return calls[:i + 1], True
            continue
        cc._append_tool_history(c["name"], c["arguments"], cc._execute_tool(c["name"], c["arguments"]))
    return list(calls), False


def screen_after(cc: ChatController, calls: list[dict] = ()) -> dict:
    """Runs calls (see run_calls) and returns the fake viewer's state. After run_one the calls have run
    already: pass none."""
    run_calls(cc, list(calls))
    return {**cc.view.display_area_widget.state(), "segmentations": cc.dicom_controller.segmentations,
            "playing": cc.view.playing, "speed": cc.view.cine_speed,
            "loaded": getattr(cc, "loaded_subjects", [])}


def render_dry_run(test_case: TestCase) -> dict:
    """No Ollama call — just builds messages/tools, for --dry-run fixture checks."""
    cc = build_controller(test_case)
    cc._history = [{"role": "user", "content": test_case.prompt}]
    messages = cc._messages_for_request()
    tools = cc._build_tools()
    return {"messages": messages, "tools": tools}
