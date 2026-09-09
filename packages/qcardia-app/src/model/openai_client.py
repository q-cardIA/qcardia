"""Thin OpenAI-compatible chat client: the second wire format the assistant can speak.

Kept apart from ollama_client rather than branching inside it, since the two
share nothing but the Qt signals and only one is ever held at a time. Nothing here is
specific to any one gateway: any /v1 endpoint with a bearer key works. utils.llm_devices
picks the class and rebinds the two constants below, so both are read at call time.
"""
from __future__ import annotations

import json

import requests

from model.background_worker import ChatWorker

CHAT_URL: str = ""  # rebound by utils.llm_devices.apply(); empty until a device is selected
API_KEY: str = ""


class OpenAIWorker(ChatWorker):
    """Streams one /chat/completions request."""

    def run(self):
        payload = {
            "model": self._model,
            "messages": self._messages,
            "stream": True,
            "max_tokens": 4096,
            "temperature": 0.2,
            # Hosted models usually reason first, which looks like a hang and costs budget.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self._tools:
            payload["tools"] = self._tools

        try:
            response = requests.post(
                CHAT_URL, json=payload, headers={"Authorization": f"Bearer {API_KEY}"},
                stream=True, timeout=180,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            self.error.emit(f"Could not reach {CHAT_URL}: {e}")
            return

        # Arguments stream in fragments, so collect per index: one turn can request several tools.
        partial: dict[int, dict] = {}
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if isinstance(chunk.get("error"), dict) and not chunk.get("choices"):
                    self.error.emit(chunk["error"].get("message", "stream error"))
                    return

                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    for call in delta.get("tool_calls") or []:
                        slot = partial.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                        fn = call.get("function") or {}
                        slot["name"] = fn.get("name") or slot["name"]
                        slot["arguments"] += fn.get("arguments") or ""
                    if delta.get("content"):
                        self.token_received.emit(delta["content"])
        except requests.Timeout:
            self.error.emit(
                f"{self._model} did not start answering in time. It is most likely still "
                "loading onto a GPU, so try a model shown as running."
            )
            return
        except requests.RequestException as e:
            self.error.emit(f"Stream interrupted: {e}")
            return

        if not partial:
            self.finished_ok.emit()
            return

        calls = []
        for index in sorted(partial):
            try:
                # A truncated argument string shouldn't kill the turn; the tool layer validates anyway.
                arguments = json.loads(partial[index]["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append((partial[index]["name"], arguments))
        self.tool_calls_ready.emit(calls)
