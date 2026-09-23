"""Application path/config constants — new users start here.

Layout assumed by the defaults:
    <WORKSPACE>/
    ├── qcardia-app-dev/    ← this repo
    └── data/               ← subject folders (any names; changeable on the welcome screen)

If your layout differs, update the paths below.
"""
import re
import subprocess
import tempfile
from pathlib import Path

import requests
from PySide6.QtCore import QSettings

# Root of the workspace that holds all repos and data.
WORKSPACE_DIR: Path = Path(__file__).parent.parent.parent.parent

# Root of this repo (for bundled assets like the window icon).
APP_DIR: Path = Path(__file__).parent.parent.parent
ICON_PATH: Path = APP_DIR / "assets" / "images" / "q-cardIA.png"

# Holds one folder per subject, any names; changed on the welcome screen.
DEFAULT_DATA_DIR: Path = WORKSPACE_DIR / "data"
_SETTINGS = QSettings("q-cardIA", "qcardia-app")


def data_dir() -> Path:
    return Path(_SETTINGS.value("data_dir", str(DEFAULT_DATA_DIR)))


def set_data_dir(path: str):
    _SETTINGS.setValue("data_dir", path)


def disambiguation_cache_dir() -> Path:
    return data_dir() / "disambiguation_cache"


def list_subjects() -> list[str]:
    folder = data_dir()
    if not folder.is_dir():
        return []
    skip = disambiguation_cache_dir().name
    return sorted(e.name for e in folder.iterdir() if e.is_dir() and e.name[0] != "." and e.name != skip)


# ---------------------------------------------------------------------------
# Ollama (local LLM chat assistant)
# ---------------------------------------------------------------------------

# Local Ollama server — 127.0.0.1, not "localhost" (can resolve to an
# unreachable IPv6 address and hang).
OLLAMA_URL: str = "http://127.0.0.1:11434/api/chat"
OLLAMA_GENERATE_URL: str = "http://127.0.0.1:11434/api/generate"
OLLAMA_TAGS_URL: str = "http://127.0.0.1:11434/api/tags"
OLLAMA_MODEL: str = "qwen3.5:latest"

# Too low silently truncates the prompt (dropping the system prompt/tools) with no error.
OLLAMA_NUM_CTX: int = 16384


def ollama_available() -> bool:
    """True when an Ollama server is reachable at OLLAMA_TAGS_URL."""
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=0.5)
        return response.ok
    except requests.RequestException:
        return False


def _format_size(num_bytes: int) -> str:
    gb = num_bytes / (1024 ** 3)
    return f"{gb:.1f}GB"


def list_ollama_models() -> list[dict]:
    """Models already pulled into the local Ollama server, with a friendly
    display label (parameter size + file size instead of the raw tag, which
    is often just a meaningless ':latest')."""
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=0.5)
        response.raise_for_status()
        models = response.json().get("models", [])
    except requests.RequestException:
        return []

    out = []
    for m in models:
        name = m["name"]
        base = name.split(":")[0]
        param_size = m.get("details", {}).get("parameter_size", "")
        size_label = _format_size(m["size"]) if m.get("size") else ""
        bits = [b for b in (param_size, size_label) if b]
        label = f"{base} ({', '.join(bits)})" if bits else base
        out.append({"tag": name, "label": label, "local": False})
    return sorted(out, key=lambda d: d["label"])


# ---------------------------------------------------------------------------
# Local GGUF weights (models/llm/) — imported into Ollama on first use via `ollama create`.
# ---------------------------------------------------------------------------

LLM_WEIGHTS_DIR: Path = WORKSPACE_DIR / "models" / "llm"


def discover_local_gguf_files() -> list[Path]:
    """GGUF weight files under LLM_WEIGHTS_DIR, not yet necessarily imported."""
    if not LLM_WEIGHTS_DIR.is_dir():
        return []
    return sorted(LLM_WEIGHTS_DIR.glob("*.gguf"))


def gguf_model_tag(gguf_path: Path) -> str:
    """Deterministic Ollama model tag derived from a GGUF filename."""
    stem = re.sub(r"[^a-z0-9._-]", "-", gguf_path.stem.lower())
    return f"local/{stem}"


def import_gguf_model(gguf_path: Path) -> str:
    """Register a local GGUF file as an Ollama model (idempotent) and return
    its tag. Ollama links the existing file rather than copying/converting
    it, but this still shells out to `ollama create`, so callers should run
    it off the GUI thread."""
    tag = gguf_model_tag(gguf_path)
    existing = {m["tag"] for m in list_ollama_models()}
    if tag in existing:
        return tag

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".Modelfile", delete=False, encoding="utf-8"
    ) as f:
        f.write(f"FROM {gguf_path}\n")
        modelfile_path = f.name

    try:
        subprocess.run(
            ["ollama", "create", tag, "-f", modelfile_path],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ollama create failed for {gguf_path.name}: {e.stderr}") from e
    finally:
        Path(modelfile_path).unlink(missing_ok=True)

    return tag


CHAT_AVAILABLE: bool = ollama_available()
