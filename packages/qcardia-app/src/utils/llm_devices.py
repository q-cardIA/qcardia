"""Which machine serves the LLMs — local Ollama, or a remote one over SSH.

The app talks to Ollama through module-level URL constants in three places
(`utils.app_config`, `model.llm_client`, and `qcardia.disambiguate` in the
pinned qcardia dependency). All three read those constants *inside* the
function that makes the request, so reassigning the module attribute
redirects every later call. That is what lets this module repoint the whole
app at another machine without forking qcardia, and what makes switching
work mid-session rather than only at startup.

Device definitions live OUTSIDE the repo (see DEVICES_PATH) so no host
addresses or credentials are committed. The preferred remote transport is an
SSH tunnel: the address and key come from the user's ~/.ssh/config, so there
is nothing to store here, no firewall change, and Ollama on the remote host
can stay bound to localhost.
"""
from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

# Probing a machine that is off/unreachable costs the full timeout, so the
# startup default is short (the local server answers in ms) and the explicit
# "validate this device" path passes a longer budget.
FAST_PROBE_TIMEOUT: float = 1.0
DEVICE_PROBE_TIMEOUT: float = 5.0

# Seconds to wait for `ssh -L` to start accepting connections on the local
# side. Separate from the probe budget above: the tunnel has to exist before
# there is anything to probe.
TUNNEL_READY_TIMEOUT: float = 8.0


def _default_devices_path() -> Path:
    """Per-user config location, outside the repo and outside the data dir."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "qcardia" / "llm_devices.json"


DEVICES_PATH: Path = Path(
    os.environ.get("QCARDIA_LLM_DEVICES", "") or _default_devices_path()
)


@dataclass(frozen=True)
class Device:
    """One machine that can serve the LLMs.

    `mode` is one of:
      local   — Ollama on this machine (no tunnel, no network hop)
      tunnel  — Ollama on a remote host, reached through `ssh -L` using
                `ssh_alias` from the user's ~/.ssh/config
      http    — Ollama on a remote host reachable directly (requires the
                remote server to be bound beyond localhost)

    `patient_data_ok` gates the disambiguation path specifically. That path
    sends DICOM header content, so a device cleared only for general chat
    must never receive it — see `apply()`.
    """

    id: str
    label: str
    mode: str = "local"
    host: str = "127.0.0.1"
    port: int = 11434
    ssh_alias: str | None = None
    # Local end of the tunnel. Deliberately not 11434 by default: that is
    # usually taken by this machine's own Ollama, and binding over it would
    # make "local" and "remote" indistinguishable.
    local_port: int = 11435
    patient_data_ok: bool = True
    note: str = ""

    @property
    def base_url(self) -> str:
        if self.mode == "tunnel":
            return f"http://127.0.0.1:{self.local_port}"
        return f"http://{self.host}:{self.port}"

    @property
    def is_local(self) -> bool:
        return self.mode == "local"


LOCAL_DEVICE = Device(id="local", label="This computer", mode="local")

# Where the local Ollama lives, kept so disambiguation can be pinned back to
# it when the selected chat device is not cleared for patient data.
_LOCAL_BASE_URL = LOCAL_DEVICE.base_url


@dataclass
class ProbeResult:
    ok: bool
    detail: str = ""
    models: list[dict] = field(default_factory=list)


def load_devices() -> list[Device]:
    """Devices from DEVICES_PATH, always with the local one first.

    A missing or malformed file is not an error: the app still works with
    local Ollama alone, which is the pre-existing behaviour.
    """
    devices = [LOCAL_DEVICE]
    try:
        raw = json.loads(DEVICES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return devices

    entries = raw if isinstance(raw, list) else raw.get("devices", [])
    for entry in entries:
        try:
            if entry.get("mode", "local") == "local":
                continue  # the built-in local entry above already covers it
            devices.append(
                Device(
                    id=str(entry["id"]),
                    label=str(entry.get("label", entry["id"])),
                    mode=str(entry.get("mode", "http")),
                    host=str(entry.get("host", "127.0.0.1")),
                    port=int(entry.get("port", 11434)),
                    ssh_alias=entry.get("ssh_alias"),
                    local_port=int(entry.get("local_port", 11435)),
                    # Default False: a device has to be explicitly declared
                    # safe before DICOM headers are allowed to reach it.
                    patient_data_ok=bool(entry.get("patient_data_ok", False)),
                    note=str(entry.get("note", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue  # skip a bad entry rather than losing the whole file
    return devices


# ---------------------------------------------------------------------------
# SSH tunnels
# ---------------------------------------------------------------------------

_tunnels: dict[str, subprocess.Popen] = {}

# Port a device's tunnel actually bound, which is not always its configured
# `local_port` — see _resolve_local_port().
_active_ports: dict[str, int] = {}


def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _serves_ollama(port: int, timeout: float = 1.0) -> bool:
    """Whether an Ollama server answers on this local port.

    An open port is not enough to identify one of our tunnels: unrelated
    local services squat on high ports (VS Code's Remote-SSH helper is one),
    and treating those as a live tunnel sends every request into a black hole
    that only surfaces as an unexplained timeout.
    """
    try:
        response = requests.get(f"http://127.0.0.1:{port}/api/tags", timeout=timeout)
        return response.ok and "models" in response.json()
    except (requests.RequestException, ValueError):
        return False


def _free_port() -> int:
    """An unused local port, for when the configured one is taken."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _resolve_local_port(device: Device) -> int:
    """Pick the local port for this device's tunnel.

    Prefers the configured port, reuses it if a working Ollama is already
    there (a tunnel left by an earlier run), and otherwise steps aside to a
    free port rather than fighting whatever else holds it.
    """
    preferred = device.local_port
    if not _port_open(preferred):
        return preferred
    if _serves_ollama(preferred):
        return preferred
    return _free_port()


def local_port_for(device: Device) -> int:
    return _active_ports.get(device.id, device.local_port)


def base_url_for(device: Device) -> str:
    """Base URL for a device, accounting for the port its tunnel really bound."""
    if device.mode == "tunnel":
        return f"http://127.0.0.1:{local_port_for(device)}"
    return device.base_url


def ensure_tunnel(device: Device) -> None:
    """Start `ssh -N -L local_port:localhost:port <alias>` if not already up.

    Uses the alias rather than an address so the key, port and current host
    all come from ~/.ssh/config — this app never handles credentials.
    """
    if device.mode != "tunnel":
        return
    if not device.ssh_alias:
        raise RuntimeError(f"Device '{device.label}' is mode 'tunnel' but has no ssh_alias.")

    existing = _tunnels.get(device.id)
    if existing is not None and existing.poll() is None and _port_open(local_port_for(device)):
        return  # already running and accepting connections
    close_tunnel(device.id)

    local_port = _resolve_local_port(device)
    if _serves_ollama(local_port):
        # A working Ollama is already exposed here — almost certainly a tunnel
        # left by an earlier run. Adopt it rather than stacking another on top.
        _active_ports[device.id] = local_port
        return

    cmd = [
        "ssh",
        "-N",
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-L", f"{local_port}:localhost:{device.port}",
        device.ssh_alias,
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
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
    raise RuntimeError(
        f"ssh tunnel to {device.ssh_alias} did not open within {TUNNEL_READY_TIMEOUT:.0f}s."
    )


def close_tunnel(device_id: str) -> None:
    proc = _tunnels.pop(device_id, None)
    _active_ports.pop(device_id, None)
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def shutdown() -> None:
    """Close every tunnel this process opened. Registered with atexit so a
    normal quit doesn't leave orphaned ssh processes behind."""
    for device_id in list(_tunnels):
        close_tunnel(device_id)


atexit.register(shutdown)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def probe(device: Device, timeout: float = DEVICE_PROBE_TIMEOUT) -> ProbeResult:
    """Check the device actually serves Ollama, opening a tunnel if needed.

    Blocking, and can take the full timeout — callers on the GUI thread must
    run this through CallableWorker.
    """
    try:
        ensure_tunnel(device)
    except RuntimeError as e:
        return ProbeResult(False, str(e))

    try:
        response = requests.get(f"{base_url_for(device)}/api/tags", timeout=timeout)
        response.raise_for_status()
        models = response.json().get("models", [])
    except requests.RequestException as e:
        return ProbeResult(False, _readable_error(device, e))

    if not models:
        return ProbeResult(True, "Reachable, but no models are pulled on this device.", [])
    plural = "s" if len(models) != 1 else ""
    return ProbeResult(True, f"{len(models)} model{plural} available", models)


def _readable_error(device: Device, exc: Exception) -> str:
    """Turn a requests exception into something a user can act on."""
    if isinstance(exc, requests.Timeout):
        return "Timed out — reachable, but did not answer in time."
    if isinstance(exc, requests.ConnectionError):
        if device.mode == "local":
            return "No Ollama server on this computer (is `ollama serve` running?)."
        return "Could not connect — is Ollama running on that device?"
    return str(exc)


# ---------------------------------------------------------------------------
# Applying a selection
# ---------------------------------------------------------------------------

_current: Device = LOCAL_DEVICE
_available: bool = False


def current() -> Device:
    return _current


def chat_available() -> bool:
    """Whether the *selected* device answered its last probe.

    Replaces the old import-time `CHAT_AVAILABLE` constant, which probed
    localhost before the user had chosen anything and blocked startup while
    it did.
    """
    return _available


def set_available(value: bool) -> None:
    global _available
    _available = value


def apply(device: Device) -> None:
    """Point the app's Ollama URLs at `device`.

    Rebinds the module attributes each caller reads at request time. Note
    `model.llm_client` imported its URLs by value, so it must be patched on
    its own module — patching only `utils.app_config` would silently leave
    chat pointing at localhost while everything else moved.
    """
    global _current
    base = base_url_for(device)

    from utils import app_config

    app_config.OLLAMA_URL = f"{base}/api/chat"
    app_config.OLLAMA_GENERATE_URL = f"{base}/api/generate"
    app_config.OLLAMA_TAGS_URL = f"{base}/api/tags"

    from model import llm_client

    llm_client.OLLAMA_URL = f"{base}/api/chat"
    llm_client.OLLAMA_GENERATE_URL = f"{base}/api/generate"

    # Disambiguation sends DICOM header content, so it follows the selected
    # device only when that device is cleared for patient data; otherwise it
    # stays pinned to the local server regardless of where chat points.
    try:
        from qcardia import disambiguate
    except ImportError:
        pass  # qcardia not importable here; chat still switches fine
    else:
        disambiguate_base = base if device.patient_data_ok else _LOCAL_BASE_URL
        disambiguate.OLLAMA_URL = f"{disambiguate_base}/api/chat"

    _current = device


def activate(device: Device, timeout: float = DEVICE_PROBE_TIMEOUT) -> ProbeResult:
    """Probe `device` and, if it answers, make it the active one.

    A failed probe leaves the previous device in place, so a dead remote host
    never silently becomes the thing the app thinks it is talking to.
    Blocking — run it off the GUI thread.
    """
    result = probe(device, timeout=timeout)
    if result.ok:
        apply(device)
        set_available(True)
    return result
