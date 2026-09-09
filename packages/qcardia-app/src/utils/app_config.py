"""Application path/config constants — new users start here.

Layout assumed by the defaults:
    <WORKSPACE>/
    ├── qcardia-app-dev/    ← this repo
    └── data/
        └── new_data/       ← subject folders (QLGE01_/, QLGE02_/, ...)

If your layout differs, update the paths below.
"""
import re
import subprocess
import tempfile
from pathlib import Path

import requests

# Root of the workspace that holds all repos and data.
WORKSPACE_DIR: Path = Path(__file__).parent.parent.parent.parent

# Root of this repo (for bundled assets like the window icon).
APP_DIR: Path = Path(__file__).parent.parent.parent
ICON_PATH: Path = APP_DIR / "assets" / "images" / "q-cardIA.png"

# Base directory that holds subject folders (e.g. QLGE01_/, QLGE02_/, ...).
# Used by the AI assistant's load_subject tool.
SUBJECT_BASE_DIR: Path = WORKSPACE_DIR / "data" / "new_data"

# Cached duplicate-series disambiguation results (model/dicom_model.py).
# Lives alongside the subject data, not inside the app repo, since it's
# derived from that data.
DISAMBIGUATION_CACHE_DIR: Path = SUBJECT_BASE_DIR / "disambiguation_cache"


def subject_folder_path(subject_number: int) -> Path:
    """Return the subject folder path for a given integer subject number."""
    return SUBJECT_BASE_DIR / f"QLGE{subject_number:02d}_"


def discover_subject_numbers() -> list[int]:
    """Scan SUBJECT_BASE_DIR for QLGE{NN}_ folders and return the sorted subject numbers.

    Lets the chat assistant ground load_subject calls against subjects that
    actually exist, instead of guessing/hallucinating a number.
    """
    if not SUBJECT_BASE_DIR.is_dir():
        return []
    numbers = []
    for entry in SUBJECT_BASE_DIR.iterdir():
        if entry.is_dir() and entry.name.startswith("QLGE") and entry.name.endswith("_"):
            digits = entry.name[len("QLGE"):-1]
            if digits.isdigit():
                numbers.append(int(digits))
    return sorted(numbers)


# ---------------------------------------------------------------------------
# Ollama (local LLM chat assistant)
# ---------------------------------------------------------------------------

# utils.llm_devices rebinds these at runtime, so every function below reads them at call time.
OLLAMA_URL: str = "http://localhost:11434/api/chat"
OLLAMA_GENERATE_URL: str = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL: str = "http://localhost:11434/api/tags"
OLLAMA_MODEL: str = "qwen3:8b"

# Explicit context window — Ollama silently truncates from the front of the
# prompt if this is left too low, which would otherwise drop the system
# prompt/tool definitions with no error.
OLLAMA_NUM_CTX: int = 16384


# 3s rather than 0.5s, which was localhost-sized and called a live remote server dead.
OLLAMA_HTTP_TIMEOUT: float = 3.0


def ollama_available(timeout: float | None = None) -> bool:
    """True when an Ollama server is reachable at OLLAMA_TAGS_URL."""
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=timeout or OLLAMA_HTTP_TIMEOUT)
        return response.ok
    except requests.RequestException:
        return False


def _format_size(num_bytes: int) -> str:
    gb = num_bytes / (1024 ** 3)
    return f"{gb:.1f}GB"


def format_ollama_models(models: list[dict]) -> list[dict]:
    """Raw /api/tags entries to the picker's shape, labelled better than a bare ':latest'."""
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


def list_ollama_models(timeout: float | None = None) -> list[dict]:
    """Models already pulled into the currently selected Ollama server."""
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=timeout or OLLAMA_HTTP_TIMEOUT)
        response.raise_for_status()
        models = response.json().get("models", [])
    except (requests.RequestException, ValueError):
        return []  # ValueError: a rebound URL can land on something serving HTML
    return format_ollama_models(models)


# ---------------------------------------------------------------------------
# Local GGUF weights (models/llm/) — not pulled into Ollama by default.
# Ollama only knows about models in its own store, so these are surfaced
# separately and imported into Ollama on first use via `ollama create`.
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
    it off the GUI thread. Local device only: it drives this machine's ollama CLI."""
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
            timeout=600,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ollama create failed for {gguf_path.name}: {e.stderr}") from e
    finally:
        Path(modelfile_path).unlink(missing_ok=True)

    return tag


# Chat availability moved to utils.llm_devices.chat_available(); probing here blocked startup.
