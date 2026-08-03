"""Thin Ollama HTTP client — streams chat responses and native tool calls.

No subprocess, no separate conda env: Ollama runs as its own local service
(see utils/app_config.py), so this is a plain requests.post from the app's
main process/env — the same pattern already used by
qcardia.disambiguate.disambiguate_duplicates, just streaming and with tools.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests
from PySide6.QtCore import QThread, Signal

from utils.app_config import OLLAMA_MODEL, OLLAMA_NUM_CTX, OLLAMA_URL, import_gguf_model


class OllamaWorker(QThread):
    """Runs one /api/chat request in the background and streams the result.

    Emits `token_received` for each streamed content chunk, `tool_calls_ready`
    once with the full list of calls if the model decided to call tools
    (Ollama sends tool calls whole, not token-by-token, and a single turn can
    contain more than one call — e.g. "load subject 5 and apply the
    functional preset" — so all of them must be emitted together rather than
    only the first), `finished_ok` when the response is complete, and `error`
    on failure.
    """

    token_received = Signal(str)
    tool_calls_ready = Signal(list)  # list of (tool_name, arguments) tuples
    finished_ok = Signal()
    error = Signal(str)

    def __init__(self, messages: list[dict], tools: list[dict] | None = None, model: str = OLLAMA_MODEL):
        super().__init__()
        self._messages = messages
        self._tools = tools
        self._model = model

    def run(self):
        payload = {
            "model": self._model,
            "messages": self._messages,
            "stream": True,
            "think": False,
            "options": {"num_ctx": OLLAMA_NUM_CTX, "temperature": 0.2},
        }
        if self._tools:
            payload["tools"] = self._tools

        try:
            response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
            response.raise_for_status()
        except requests.RequestException as e:
            self.error.emit(f"Could not reach Ollama at {OLLAMA_URL}: {e}")
            return

        try:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                message = chunk.get("message", {})
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    calls = [
                        (tc["function"]["name"], tc["function"].get("arguments", {}))
                        for tc in tool_calls
                    ]
                    self.tool_calls_ready.emit(calls)
                    return

                content = message.get("content")
                if content:
                    self.token_received.emit(content)

                if chunk.get("done"):
                    break
        except requests.RequestException as e:
            self.error.emit(f"Ollama stream interrupted: {e}")
            return

        self.finished_ok.emit()


class ModelImportWorker(QThread):
    """Registers a local GGUF file as an Ollama model in the background
    (`ollama create` can take a little while on first import)."""

    finished_ok = Signal(str)  # resulting model tag
    error = Signal(str)

    def __init__(self, gguf_path: Path):
        super().__init__()
        self._gguf_path = gguf_path

    def run(self):
        try:
            tag = import_gguf_model(self._gguf_path)
        except Exception as e:
            self.error.emit(str(e))
            return
        self.finished_ok.emit(tag)
