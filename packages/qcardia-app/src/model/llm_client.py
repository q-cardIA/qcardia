"""Thin Ollama HTTP client — streams chat responses and native tool calls.

No subprocess, no separate conda env: Ollama runs as its own local service
(see utils/app_config.py), so this is a plain requests.post from the app's
main process/env — the same pattern already used by
qcardia.disambiguate.disambiguate_duplicates, just streaming and with tools.
"""
from __future__ import annotations

import json
import logging

import requests
from PySide6.QtCore import QThread, Signal

from utils.app_config import OLLAMA_GENERATE_URL, OLLAMA_MODEL, OLLAMA_NUM_CTX, OLLAMA_URL

KEEP_ALIVE = "30m"  # how long Ollama keeps the model resident after a request (default is 5 min)


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

    def __init__(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str = OLLAMA_MODEL,
        think: bool = False,
    ):
        super().__init__()
        self._messages = messages
        self._tools = tools
        self._model = model
        self._think = think

    def run(self):
        payload = {
            "model": self._model,
            "messages": self._messages,
            "stream": True,
            "think": self._think,
            "keep_alive": KEEP_ALIVE,
            "options": {"num_ctx": OLLAMA_NUM_CTX, "temperature": 0.2},
        }
        if self._tools:
            payload["tools"] = self._tools

        try:
            # 10s to connect, 120s between chunks once streaming.
            response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=(10, 120))
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
                    logging.info(f"[{self._model}] tool_calls: {calls}")
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

        logging.info(f"[{self._model}] finished with no tool_calls (text-only reply)")
        self.finished_ok.emit()


def warm_model(model: str = OLLAMA_MODEL):
    """Blocking call that gets the model resident with num_ctx matching
    chat's, without generating text. Meant to run off the main thread,
    fired in parallel with a slow tool so chat's follow-up call doesn't
    also pay a cold-load penalty. Mismatched num_ctx forces Ollama to
    reload the model (~12-14s), so matching it here is what matters."""
    try:
        requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": model,
                "prompt": "",
                "keep_alive": KEEP_ALIVE,
                "options": {"num_ctx": OLLAMA_NUM_CTX},
            },
            timeout=(10, 60),
        )
    except requests.RequestException:
        pass  # best-effort warm-up; the real chat call will surface any real connectivity error
