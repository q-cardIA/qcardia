"""Which machine serves the LLMs, and which wire format it speaks.

Switching rebinds the module-level URL constants that every caller reads at request
time; CLAUDE.md explains why that is safe. Devices come from a gitignored config file
(DEVICES_PATH), so no addresses or credentials are committed: see the README and
config/llm_devices.example.json. With no config file the app runs against the local
Ollama exactly as it did before this module existed.
"""
from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field, fields
from pathlib import Path

import requests

FAST_PROBE_TIMEOUT: float = 1.0     # startup check of the local server
DEVICE_PROBE_TIMEOUT: float = 5.0   # user picked a device and is waiting on it
TUNNEL_READY_TIMEOUT: float = 8.0   # ssh -L starting to accept connections

_PROTOCOLS = ("ollama", "openai")

# Gitignored, so your own file sits next to the example and still cannot be committed.
_DEFAULT_DEVICES_PATH = Path(__file__).parents[2] / "config" / "llm_devices.json"
DEVICES_PATH: Path = Path(os.environ.get("QCARDIA_LLM_DEVICES", "") or _DEFAULT_DEVICES_PATH)


@dataclass(frozen=True)
class Device:
    """One machine that can serve the LLMs. See config/llm_devices.example.json."""

    id: str
    label: str
    mode: str = "local"            # transport: local | tunnel (ssh -L) | http
    protocol: str = "ollama"       # wire format: ollama | openai (a /v1 gateway)
    host: str = "127.0.0.1"
    port: int = 11434
    url: str = ""                  # full base URL; a gateway needs a scheme and path
    api_key_env: str = ""          # env var holding the bearer token, never the key
    model: str = ""                # default tag; a gateway shares none with Ollama
    ssh_alias: str | None = None   # ~/.ssh/config entry, so no credentials live here
    local_port: int = 11435        # tunnel's local end; 11434 is this machine's Ollama
    patient_data_ok: bool = False  # may DICOM headers go here? enforced in apply()
    note: str = ""

    @property
    def base_url(self) -> str:
        if self.url:
            return self.url.rstrip("/")
        if self.mode == "tunnel":
            # Not always local_port: see _resolve_local_port().
            return f"http://127.0.0.1:{_active_ports.get(self.id, self.local_port)}"
        return f"http://{self.host}:{self.port}"

    @property
    def is_local(self) -> bool:
        return self.mode == "local"


LOCAL_DEVICE = Device(id="local", label="This computer", patient_data_ok=True)
_LOCAL_URL = LOCAL_DEVICE.base_url  # what non-cleared paths stay pinned to


@dataclass
class ProbeResult:
    ok: bool
    detail: str = ""
    models: list[dict] = field(default_factory=list)


def load_devices() -> list[Device]:
    """Devices from DEVICES_PATH, local first. A missing or bad file is not an error."""
    devices = [LOCAL_DEVICE]
    try:
        raw = json.loads(DEVICES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return devices

    known = {f.name for f in fields(Device)}
    for entry in raw if isinstance(raw, list) else raw.get("devices", []):
        try:
            # A local entry is already covered above; an unknown protocol is unusable.
            if entry.get("mode", "local") == "local" or entry.get("protocol", "ollama") not in _PROTOCOLS:
                continue
            kwargs = {k: v for k, v in entry.items() if k in known}
            kwargs.update({k: int(kwargs[k]) for k in ("port", "local_port") if k in kwargs})
            devices.append(Device(**kwargs))
        except (AttributeError, TypeError, ValueError):
            continue  # skip one bad entry rather than losing the whole file
    return devices


# --- SSH tunnels -----------------------------------------------------------

_tunnels: dict[str, subprocess.Popen] = {}
_active_ports: dict[str, int] = {}  # port each tunnel actually bound


def _port_open(port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _serves_ollama(port: int, timeout: float = 1.0) -> bool:
    """An open port is not proof of our tunnel: VS Code's Remote-SSH helper squats on 11435."""
    try:
        response = requests.get(f"http://127.0.0.1:{port}/api/tags", timeout=timeout)
        return response.ok and "models" in response.json()
    except (requests.RequestException, ValueError):
        return False


def _resolve_local_port(device: Device) -> int:
    """The configured port, reused if a live tunnel already holds it, else a free one."""
    if not _port_open(device.local_port) or _serves_ollama(device.local_port):
        return device.local_port
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ensure_tunnel(device: Device) -> None:
    """Start `ssh -N -L` if not up, via an ~/.ssh/config alias so this app holds no key."""
    if device.mode != "tunnel":
        return
    if not device.ssh_alias:
        raise RuntimeError(f"Device '{device.label}' is mode 'tunnel' but has no ssh_alias.")

    existing = _tunnels.get(device.id)
    if existing is not None and existing.poll() is None and _port_open(_active_ports[device.id]):
        return
    close_tunnel(device.id)

    local_port = _resolve_local_port(device)
    if _serves_ollama(local_port):
        _active_ports[device.id] = local_port  # tunnel left by an earlier run, adopt it
        return

    proc = subprocess.Popen(
        ["ssh", "-N",
         "-o", "BatchMode=yes",
         "-o", "ExitOnForwardFailure=yes",
         "-o", "ServerAliveInterval=30",
         "-L", f"{local_port}:localhost:{device.port}",
         device.ssh_alias],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    _tunnels[device.id] = proc
    _active_ports[device.id] = local_port

    deadline = time.monotonic() + TUNNEL_READY_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            err = (proc.stderr.read() or b"").decode(errors="replace").strip()
            close_tunnel(device.id)
            raise RuntimeError(err or f"ssh tunnel to {device.ssh_alias} exited immediately.")
        if _port_open(local_port):
            return
        time.sleep(0.15)

    close_tunnel(device.id)
    raise RuntimeError(f"ssh tunnel to {device.ssh_alias} did not open within {TUNNEL_READY_TIMEOUT:.0f}s.")


def close_tunnel(device_id: str) -> None:
    proc = _tunnels.pop(device_id, None)
    _active_ports.pop(device_id, None)
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        with suppress(Exception):
            proc.kill()


def shutdown() -> None:
    """Close every tunnel this process opened, so quitting leaves no orphaned ssh."""
    for device_id in list(_tunnels):
        close_tunnel(device_id)


atexit.register(shutdown)


# --- Probing ---------------------------------------------------------------

def _api_key(device: Device) -> str:
    """Read at call time from the env var the config names, so no key is stored here."""
    return os.environ.get(device.api_key_env, "").strip() if device.api_key_env else ""


def _auth(device: Device) -> dict:
    key = _api_key(device)
    return {"Authorization": f"Bearer {key}"} if key else {}


def _fetch_models(device: Device, timeout: float) -> list[dict]:
    """The device's models in the picker's shape. Raises requests.RequestException if unreachable."""
    base = device.base_url
    if device.protocol != "openai":
        response = requests.get(f"{base}/api/tags", timeout=timeout)
        response.raise_for_status()
        from utils.app_config import format_ollama_models

        return format_ollama_models(response.json().get("models", []))

    response = requests.get(f"{base}/models", headers=_auth(device), timeout=timeout)
    response.raise_for_status()
    entries = [m for m in response.json().get("data", []) if m.get("accessible", True)]
    # GPU-resident models first: the rest pay a cold start that can run into minutes.
    entries.sort(key=lambda m: (not m.get("running"), m.get("id", "")))
    out = []
    for m in entries:
        tag, status = m.get("id", ""), m.get("status_label", "")
        out.append({
            "tag": tag,
            "label": f"{tag} ({status})" if status else tag,
            "local": False,
            "running": bool(m.get("running")),
        })
    return out


def _readable_error(device: Device, exc: Exception) -> str:
    """Turn a requests exception into something the user can act on."""
    if isinstance(exc, requests.Timeout):
        return "Timed out. Reachable, but did not answer in time."
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        code = exc.response.status_code
        if code in (401, 403):
            return f"Rejected the API key (HTTP {code}). Check {device.api_key_env}."
        return f"Answered with HTTP {code}."
    if isinstance(exc, requests.ConnectionError):
        if device.protocol == "openai":
            return "Could not reach the gateway. Is the VPN it needs connected?"
        if device.is_local:
            return "No Ollama server on this computer (is `ollama serve` running?)."
        return "Could not connect. Is Ollama running on that device?"
    return str(exc)


def _no_models_detail(device: Device) -> str:
    if device.protocol == "openai":
        return "Reachable, but this key has access to no models."
    return "Reachable, but no models are installed (run `ollama pull`)."


def probe(device: Device, timeout: float = DEVICE_PROBE_TIMEOUT) -> ProbeResult:
    """Check the device answers, opening its tunnel if needed. Blocking, so run it off the GUI thread."""
    try:
        ensure_tunnel(device)
    except RuntimeError as e:
        return ProbeResult(False, str(e))

    if device.protocol == "openai" and not _api_key(device):
        return ProbeResult(False, f"No API key. Set the {device.api_key_env or 'api_key_env'} environment variable.")

    try:
        models = _fetch_models(device, timeout)
    except (requests.RequestException, ValueError) as e:
        return ProbeResult(False, _readable_error(device, e))

    if not models:
        return ProbeResult(True, _no_models_detail(device), [])
    detail = f"{len(models)} model{'s' if len(models) != 1 else ''} available"
    if device.protocol == "openai" and not any(m["running"] for m in models):
        # A gateway lists everything even when swapped out, hiding a long cold start.
        detail += ", but none is loaded on a GPU, so the first reply will be slow"
    return ProbeResult(True, detail, models)


# --- The active device -----------------------------------------------------

_current: Device = LOCAL_DEVICE
_available: bool = False
_models: list[dict] = []


def current() -> Device:
    return _current


def chat_available() -> bool:
    """Whether the selected device answered its last probe."""
    return _available


def apply(device: Device) -> None:
    """Point the app's LLM URLs at `device` by rebinding what each caller reads at request time."""
    global _current
    base = device.base_url
    # Only chat can follow a non-Ollama device; the rest is Ollama-shaped, so it stays local.
    ollama = base if device.protocol == "ollama" else _LOCAL_URL

    from model import ollama_client
    from utils import app_config

    # Both modules, not just app_config: ollama_client imported these names by value.
    for module in (app_config, ollama_client):
        module.OLLAMA_URL = f"{ollama}/api/chat"
        module.OLLAMA_GENERATE_URL = f"{ollama}/api/generate"
    app_config.OLLAMA_TAGS_URL = f"{ollama}/api/tags"

    if device.protocol == "openai":
        from model import openai_client

        openai_client.CHAT_URL = f"{base}/chat/completions"
        openai_client.API_KEY = _api_key(device)

    # Disambiguation sends DICOM headers, so it follows the device only when cleared for them.
    try:
        from qcardia import disambiguate
    except ImportError:
        pass  # qcardia not importable here; chat still switches fine
    else:
        disambiguate.OLLAMA_URL = f"{ollama if device.patient_data_ok else _LOCAL_URL}/api/chat"

    _current = device


def activate(device: Device, timeout: float = DEVICE_PROBE_TIMEOUT) -> ProbeResult:
    """Probe and, on success, make active. A failure leaves the previous device in place."""
    global _available, _models
    result = probe(device, timeout=timeout)
    if result.ok:
        apply(device)
        _available, _models = True, result.models
    return result


def worker_class():
    """The QThread class speaking the active protocol. Both take the same args and signals."""
    if _current.protocol == "openai":
        from model.openai_client import OpenAIWorker

        return OpenAIWorker
    from model.ollama_client import OllamaWorker

    return OllamaWorker


def list_models(refresh: bool = False) -> list[dict]:
    """The active device's models, cached by its last probe. Refresh only after changing them."""
    global _models
    if refresh:
        _models = probe(_current).models
    return _models


def default_model() -> str:
    """Tag to fall back on when the picker has no selection of its own."""
    from utils.app_config import OLLAMA_MODEL

    return _current.model or OLLAMA_MODEL


def supports_warmup() -> bool:
    """Ollama-only: a gateway schedules its own models and has no keep_alive to set."""
    return _current.protocol == "ollama"
