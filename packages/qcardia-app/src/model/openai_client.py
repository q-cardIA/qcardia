"""Thin OpenAI-compatible chat client: the second wire format the assistant can speak.

Kept apart from ollama_client rather than branching inside it, since the two share
nothing but the Qt signals and only one is ever held at a time. utils.llm_devices picks
the class and rebinds the two constants below, so both are read at call time.

Any /v1 endpoint with a bearer key works: anything non-standard a particular gateway
needs goes in that device's `extra_body`, not in the payload built here.
"""
from __future__ import annotations

import json

import requests

from model.background_worker import ChatWorker

CHAT_URL: str = ""  # rebound by utils.llm_devices.apply(); empty until a device is selected
API_KEY: str = ""
EXTRA_BODY: dict = {}  # the device's non-standard request fields, if it needs any

MAX_TOKENS = 4096  # reply cap; quoted back to the user when a reply hits it

_COLD_START = (
    "{model} did not start answering in time. It is most likely still loading onto a "
    "GPU, so try a model shown as running."
)


def _to_openai(messages: list[dict]) -> list[dict]:
    """Ollama-shaped history to OpenAI's, which the rest of the app does not speak.

    ChatController builds tool history Ollama's way: no call id, and arguments as a dict.
    OpenAI needs an id on every call, `type: "function"`, arguments as a JSON *string*,
    and a matching tool_call_id on each result. Ids are synthesised per request, which is
    fine because they only have to be consistent within the payload being sent.
    """
    out: list[dict] = []
    pending: list[str] = []
    for message in messages:
        calls = message.get("tool_calls")
        if calls:
            ids = [f"call_{len(out)}_{i}" for i in range(len(calls))]
            out.append({
                "role": message.get("role", "assistant"),
                "content": message.get("content") or "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call["function"]["name"],
                            "arguments": json.dumps(call["function"].get("arguments", {})),
                        },
                    }
                    for call_id, call in zip(ids, calls)
                ],
            })
            pending = ids
        elif message.get("role") == "tool":
            # Results come back in call order, so pair them off against the ids above.
            call_id = pending.pop(0) if pending else "call_0_0"
            out.append({"role": "tool", "tool_call_id": call_id, "content": message.get("content", "")})
        else:
            out.append(message)
    return out


def _unreachable(exc: requests.RequestException, model: str) -> str:
    """Name the real cause: 401 is the key and 404 the model, not the network."""
    response = getattr(exc, "response", None)
    if response is None:
        return f"Could not reach the gateway — is the VPN it needs connected? ({exc})"
    if response.status_code in (401, 403):
        return (f"The gateway rejected the API key (HTTP {response.status_code}). Check the "
                f"variable named by this device's api_key_env.")
    if response.status_code == 404:
        return f"'{model}' is not available on this gateway — pick another in the model list."
    return f"The gateway answered HTTP {response.status_code}."


class OpenAIWorker(ChatWorker):
    """Streams one /chat/completions request."""

    def run(self):
        payload = {
            "model": self._model,
            "messages": _to_openai(self._messages),
            "stream": True,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.2,
        }
        payload.update(EXTRA_BODY)
        if self._tools:
            payload["tools"] = self._tools

        try:
            response = requests.post(
                CHAT_URL, json=payload, headers={"Authorization": f"Bearer {API_KEY}"},
                stream=True, timeout=180,
            )
            response.raise_for_status()
        except requests.Timeout:
            # Headers don't arrive until generation starts, so a cold start times out here.
            self.error.emit(_COLD_START.format(model=self._model))
            return
        except requests.RequestException as e:
            self.error.emit(_unreachable(e, self._model))
            return

        # SSE carries no charset, so requests leaves encoding unset and would hand back
        # raw bytes; without this the decode below is skipped and every line is bytes.
        response.encoding = response.encoding or "utf-8"

        # Arguments stream in fragments, so collect per index: one turn can request several tools.
        partial: dict[int, dict] = {}
        finish = ""  # last finish_reason seen; "length" means the reply was cut off
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
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta") or {}
                    for call in delta.get("tool_calls") or []:
                        slot = partial.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                        fn = call.get("function") or {}
                        slot["name"] = fn.get("name") or slot["name"]
                        slot["arguments"] += fn.get("arguments") or ""
                    if delta.get("content"):
                        self.token_received.emit(delta["content"])
        except requests.Timeout:
            self.error.emit(_COLD_START.format(model=self._model))
            return
        except requests.RequestException as e:
            self.error.emit(f"Stream interrupted: {e}")
            return

        # Truncated arguments parse to {} and would run the tool with none, so refuse instead.
        if finish == "length":
            if partial:
                self.error.emit(
                    f"{self._model} hit the {MAX_TOKENS}-token reply limit while requesting a "
                    f"tool, so nothing was run. Try asking for one step at a time."
                )
                return
            self.token_received.emit(f" […truncated at the {MAX_TOKENS}-token limit]")

        if not partial:
            self.finished_ok.emit()
            return

        calls = []
        for index in sorted(partial):
            try:
                arguments = json.loads(partial[index]["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}  # unreachable for a complete reply; guarded above otherwise
            calls.append((partial[index]["name"], arguments))
        self.tool_calls_ready.emit(calls)
