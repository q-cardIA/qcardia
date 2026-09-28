"""Thin Ollama HTTP client: streams chat responses and native tool calls.

No subprocess and no separate env, just a requests.post from the app's own process,
the same pattern qcardia.disambiguate already uses. See openai_client for the other
wire format; utils.llm_devices picks between them.
"""
from __future__ import annotations

import json
import logging

import requests
from model.background_worker import ChatWorker
from utils.app_config import OLLAMA_GENERATE_URL, OLLAMA_MODEL, OLLAMA_NUM_CTX, OLLAMA_URL

KEEP_ALIVE = "30m"  # how long Ollama keeps the model resident after a request (default is 5 min)


class OllamaWorker(ChatWorker):
    """Streams one /api/chat request. Ollama sends tool calls whole, and a turn can hold
    several ("load subject 5 and apply the functional preset"), so all are emitted at once."""

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
            # Ollama refuses think:true without the capability and says why in the body, not the status.
            detail = ""
            if getattr(e, "response", None) is not None:
                try:
                    detail = e.response.json().get("error", "")
                except ValueError:
                    detail = ""
            if detail:
                self.error.emit(detail[:1].upper() + detail[1:] + ".")
            else:
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

                thinking = message.get("thinking")
                if thinking:
                    self.thinking_received.emit(thinking)

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
    """Make the model resident with chat's num_ctx, without generating text.

    Blocking, so run it off the main thread. A mismatched num_ctx forces Ollama to reload
    the model (~12-14s), which is the whole point of matching it here.
    """
    try:
        requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": model,
                "prompt": "",
                "keep_alive": KEEP_ALIVE,
                "options": {"num_ctx": OLLAMA_NUM_CTX},
            },
            # Generous: backs the picker's loading bar, and a big remote model takes minutes.
            timeout=(10, 300),
        )
    except requests.RequestException:
        pass  # best-effort warm-up; the real chat call will surface any real connectivity error
