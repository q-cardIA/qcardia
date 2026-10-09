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
    token_url: str = ""            # set instead of api_key_env when the key is a short-lived token
    client_id_env: str = ""        # env vars holding the client id and secret traded at token_url
    client_secret_env: str = ""
    model: str = ""                # default tag; a gateway shares none with Ollama
    extra_body: dict = field(default_factory=dict)  # non-standard request fields, e.g. vLLM's
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


def _local_model() -> str:
    """This computer's default chat model: the "model" of an entry with "mode": "local" in DEVICES_PATH."""
    with suppress(OSError, ValueError, AttributeError):
        devices = json.loads(DEVICES_PATH.read_text(encoding="utf-8")).get("devices", [])
        return next((d.get("model", "") for d in devices if d.get("mode") == "local"), "")
    return ""


LOCAL_DEVICE = Device(id="local", label="This computer", patient_data_ok=True, model=_local_model())
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
            # A gateway is remote by definition, so its mode can be left out of the config.
            mode = entry.get("mode") or ("http" if entry.get("url") else "local")
            # A local entry is already covered above; an unknown protocol is unusable.
            if mode == "local" or entry.get("protocol", "ollama") not in _PROTOCOLS:
                continue
            kwargs = {k: v for k, v in entry.items() if k in known} | {"mode": mode}
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


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _resolve_local_port(device: Device) -> int:
    """The configured port, reused if this device's own tunnel already holds it."""
    # local_port defaults to 11435 for every device, so two remote machines configured
    # from the example share it; without this check the second would adopt the first's
    # tunnel and quietly send its chat (and DICOM headers) to the wrong machine.
    if device.local_port in {p for i, p in _active_ports.items() if i != device.id}:
        return _free_port()
    if not _port_open(device.local_port) or _serves_ollama(device.local_port):
        return device.local_port
    return _free_port()


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
    if local_port == device.local_port and _serves_ollama(local_port):
        _active_ports[device.id] = local_port  # this device's tunnel from an earlier run
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
            proc.wait(timeout=3)  # reap it, so POSIX doesn't keep a zombie
    with suppress(Exception):
        proc.stderr.close()


def shutdown() -> None:
    """Close every tunnel this process opened, so quitting leaves no orphaned ssh."""
    for device_id in list(_tunnels):
        close_tunnel(device_id)


atexit.register(shutdown)


# --- Probing ---------------------------------------------------------------

TOKEN_REFRESH_MARGIN: float = 120.0  # renew a token this long before it expires
_tokens: dict[str, tuple[str, float]] = {}  # device id -> (token, time.monotonic() it expires)


def _env(name: str) -> str:
    """An env var, falling back on Windows to the user's saved ones.

    A process copies its env from its parent at start, so a variable set after the
    terminal or IDE opened stays invisible here until that parent restarts as well.
    """
    if not name:
        return ""
    value = os.environ.get(name, "").strip()
    if value or sys.platform != "win32":
        return value
    import winreg

    with suppress(OSError):
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return str(winreg.QueryValueEx(key, name)[0]).strip()
    return ""


def _key_source(device: Device) -> str:
    """The env vars to name when the key is missing or rejected."""
    if device.token_url:
        return f"{device.client_id_env or 'client_id_env'} and {device.client_secret_env or 'client_secret_env'}"
    return device.api_key_env or "api_key_env"


def _fetch_token(device: Device) -> str:
    """Trade the client id and secret for a bearer token (Run:ai's client-credentials grant).

    Raises requests.RequestException if the token server refuses or cannot be reached,
    and ValueError if it answers without a token.
    """
    client_id, secret = _env(device.client_id_env), _env(device.client_secret_env)
    if not (client_id and secret):
        return ""
    response = requests.post(
        device.token_url,
        json={"grantType": "client_credentials", "clientId": client_id, "clientSecret": secret},
        timeout=DEVICE_PROBE_TIMEOUT,
    )
    response.raise_for_status()
    body = response.json()
    token = body.get("accessToken") if isinstance(body, dict) else None
    if not token:
        raise ValueError(f"The token server at {device.token_url} answered without an accessToken.")
    _tokens[device.id] = (token, time.monotonic() + float(body.get("expiresIn") or 1800))
    return token


def _api_key(device: Device, refresh: bool = False) -> str:
    """The bearer key, read at call time so no key is stored here.

    A token device gets a fresh token when the cached one is near expiry, or when
    `refresh` says the server just rejected it. Can raise like _fetch_token.
    """
    if not device.token_url:
        return _env(device.api_key_env)
    cached = _tokens.get(device.id)
    if refresh or cached is None or cached[1] - time.monotonic() < TOKEN_REFRESH_MARGIN:
        return _fetch_token(device)
    return cached[0]


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
    # A gateway marks each model; a single-model endpoint has no "running" and is always loaded.
    entries.sort(key=lambda m: (not m.get("running", True), m.get("id", "")))
    out = []
    for m in entries:
        tag, status = m.get("id", ""), m.get("status_label", "")
        out.append({
            "tag": tag,
            "label": f"{tag} ({status})" if status else tag,
            "local": False,
            "running": bool(m.get("running", True)),
        })
    return out


def _readable_error(device: Device, exc: Exception) -> str:
    """Turn a requests exception into something the user can act on."""
    if isinstance(exc, requests.Timeout):
        return "Timed out. Reachable, but did not answer in time."
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        code = exc.response.status_code
        if code in (401, 403):
            return f"Rejected the API key (HTTP {code}). Check {_key_source(device)}."
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

    if device.protocol == "openai":
        try:
            key = _api_key(device, refresh=True)  # a token device: prove the credentials still work
        except requests.ConnectionError:
            return ProbeResult(False, f"Could not reach the token server at {device.token_url}.")
        except (requests.RequestException, ValueError) as e:
            return ProbeResult(False, _readable_error(device, e))
        if not key:
            return ProbeResult(False, f"No API key. Set the {_key_source(device)} environment variable(s).")

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
    # Chat sends the state block: series names, slice numbers, and LV volumes and ED/ES frames from
    # the segmentation. These are derived numbers with no DICOM headers or pixels, so any chat
    # device may get them (decided 2026-10-06).
    ollama = base if device.protocol == "ollama" else _LOCAL_URL

    from model import ollama_client
    from utils import app_config

    # Both modules, not just app_config: ollama_client imported these names by value.
    for module in (app_config, ollama_client):
        module.OLLAMA_URL = f"{ollama}/api/chat"
        module.OLLAMA_GENERATE_URL = f"{ollama}/api/generate"
    app_config.OLLAMA_TAGS_URL = f"{ollama}/api/tags"
    app_config.OLLAMA_SHOW_URL = f"{ollama}/api/show"

    if device.protocol == "openai":
        from model import openai_client

        openai_client.CHAT_URL = f"{base}/chat/completions"
        openai_client.get_api_key = lambda refresh=False: _api_key(device, refresh)
        openai_client.EXTRA_BODY = dict(device.extra_body)

    # Disambiguation sends DICOM headers, so it follows the device only when cleared for
    # them and that device actually has the fixed model qcardia asks for; otherwise every
    # subject load with a duplicate-series group would fail with model-not-found.
    try:
        from qcardia import disambiguate
    except ImportError:
        pass  # qcardia not importable here; chat still switches fine
    else:
        follows = device.patient_data_ok and (
            device.is_local or any(m["tag"] == disambiguate.OLLAMA_MODEL for m in _models)
        )
        disambiguate.OLLAMA_URL = f"{ollama if follows else _LOCAL_URL}/api/chat"

    _current = device


def activate(device: Device, timeout: float = DEVICE_PROBE_TIMEOUT) -> ProbeResult:
    """Probe and, on success, make active. A failure leaves the previous device in place."""
    global _available, _models
    result = probe(device, timeout=timeout)
    if result.ok:
        _models = result.models  # before apply(), which checks them
        apply(device)
        _available = True
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


def supports_thinking(tag: str) -> bool:
    """Whether this model can be asked to reason: Ollama answers via /api/show, a gateway always can."""
    if not tag:
        return False
    if _current.protocol != "ollama":
        return True
    try:
        from utils.app_config import OLLAMA_SHOW_URL

        response = requests.post(OLLAMA_SHOW_URL, json={"model": tag}, timeout=2.0)
        response.raise_for_status()
        return "thinking" in (response.json().get("capabilities") or [])
    except (requests.RequestException, ValueError):
        return False  # unknown: better a disabled button than a request Ollama will reject


def supports_warmup() -> bool:
    """Ollama-only: a gateway schedules its own models and has no keep_alive to set."""
    return _current.protocol == "ollama"
