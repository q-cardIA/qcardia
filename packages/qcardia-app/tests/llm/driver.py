"""Builds a real ChatController via object.__new__() (skips __init__, no Qt
touched) and runs one Ollama tool-calling turn synchronously via
OllamaWorker.run() — real, unmodified prompt/tool-building code, no mock."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllers.chat_controller import ChatController  # noqa: E402
from model.llm_client import OllamaWorker  # noqa: E402
from utils.app_config import OLLAMA_MODEL  # noqa: E402

from fakes import FakeDicomController  # noqa: E402
from test_cases import TestCase  # noqa: E402


def build_controller(test_case: TestCase, llm_model: str = OLLAMA_MODEL) -> ChatController:
    cc = object.__new__(ChatController)  # skip __init__, no Qt widgets touched
    cc.view = test_case.view
    cc.dicom_controller = FakeDicomController(test_case.model, test_case.view)
    cc.model = test_case.model
    cc._history = []
    cc._worker = None
    cc._llm_model = llm_model
    cc._available_subjects_cache = list(test_case.available_subjects)
    cc._warmup_worker = None
    return cc


class RunResult:
    def __init__(self, tool_calls: list[dict], assistant_text: str, elapsed_s: float, error: str | None = None):
        self.tool_calls = tool_calls
        self.assistant_text = assistant_text
        self.elapsed_s = elapsed_s
        self.error = error


def run_one(cc: ChatController, user_text: str, think: bool | None = None, timeout: float = 120.0) -> RunResult:
    """Runs one user turn and captures the model's tool_calls. Ignores
    multi-step continuation, except past a list_subjects lookup."""
    cc._history = [{"role": "user", "content": user_text}]
    tools = cc._build_tools()

    # think=None means "use OllamaWorker's own default", not a literal False.
    kwargs = {"think": think} if think is not None else {}

    calls: list[dict] = []
    tokens: list[str] = []
    errors: list[str] = []

    start = time.monotonic()
    for _ in range(2):
        worker = OllamaWorker(cc._messages_for_request(), tools=tools, model=cc._llm_model, **kwargs)
        new: list[dict] = []
        tokens.clear()
        worker.tool_calls_ready.connect(lambda c: new.extend({"name": n, "arguments": a} for n, a in c))
        worker.token_received.connect(tokens.append)
        worker.error.connect(errors.append)
        worker.run()  # synchronous — see module docstring
        calls += new
        if errors or not new or any(c["name"] != "list_subjects" for c in new):
            break
        for c in new:  # answer the lookup and ask again, as the app does
            cc._append_tool_history(c["name"], c["arguments"], cc._execute_tool(c["name"], c["arguments"]))
    elapsed = time.monotonic() - start

    return RunResult(
        tool_calls=calls,
        assistant_text="".join(tokens),
        elapsed_s=elapsed,
        error=errors[0] if errors else None,
    )


def render_dry_run(test_case: TestCase) -> dict:
    """No Ollama call — just builds messages/tools, for --dry-run fixture checks."""
    cc = build_controller(test_case)
    cc._history = [{"role": "user", "content": test_case.prompt}]
    messages = cc._messages_for_request()
    tools = cc._build_tools()
    return {"messages": messages, "tools": tools}
