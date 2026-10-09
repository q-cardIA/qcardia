"""Application path/config constants — new users start here.

Layout assumed by the defaults:
    <WORKSPACE>/
    ├── qcardia-app-dev/    ← this repo
    └── data/               ← subject folders (any names; changeable on the welcome screen)

If your layout differs, update the paths below.
"""
import csv
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
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
SETTINGS = QSettings("q-cardIA", "qcardia-app")

COMMUNITY_URL = "https://github.com/q-cardIA"
REPO_URL = "https://github.com/q-cardIA/qcardia"
WIKI_URL = f"{REPO_URL}/wiki"
ISSUES_URL = f"{REPO_URL}/issues"


def patient_source() -> Path:
    """What the welcome screen chose: a patient list (.csv) or a folder with one subfolder per subject."""
    return Path(SETTINGS.value("data_dir", str(DEFAULT_DATA_DIR)))


def patient_list() -> Path:
    """The chosen patient list, else the chosen folder's patients.csv (which may not exist)."""
    source = patient_source()
    return source if source.suffix.lower() == ".csv" else source / PATIENTS_CSV_NAME


def data_dir() -> Path:
    """The folder the subjects' paths start from: the patient list's folder, or the chosen folder."""
    return patient_list().parent


def set_data_dir(path: str):
    """A folder or a patient list (.csv)."""
    SETTINGS.setValue("data_dir", path)


def list_subjects() -> list[str]:
    """With a patient list, its subjects whose folder exists; with a folder, its subfolders."""
    if patient_source() == patient_list():
        return [s for s in _patient_db() if (data_dir() / s).is_dir()]
    folder = data_dir()
    if not folder.is_dir():
        return []
    return sorted(e.name for e in folder.iterdir() if e.is_dir() and e.name[0] != ".")


# A patient list: per subject folder a made-up name and hospital ID (src/tools/make_patient_db.py). A
# chosen folder uses the patients.csv in it, if any, for the names.
PATIENTS_CSV_NAME = "patients.csv"


@dataclass(frozen=True)
class Patient:
    folder: str
    name: str = ""  # empty for a folder the patient database does not list
    hospital_id: str = ""


def _patient_db() -> dict[str, Patient]:
    """The patient list's patients by subject (its path as listed, from data_dir()); empty without a list."""
    try:
        with open(patient_list(), newline="", encoding="utf-8-sig") as f:  # also as Excel saves it
            return {r["folder"]: Patient(r["folder"], r["name"], r["hospital_id"]) for r in csv.DictReader(f)}
    except (OSError, KeyError):
        return {}


def patients(folders: list[str]) -> list[Patient]:
    """The patient of each subject folder; a folder the database does not list keeps its folder name."""
    db = _patient_db()
    return [db.get(f, Patient(f)) for f in folders]


def patient_label(subject: str | Path) -> str:
    """A subject's patient name and ID, by its path or its entry in the patient list; else its folder name."""
    same = lambda s: os.path.normcase(os.path.abspath(data_dir() / s))  # noqa: E731 (no disk access)
    p = next((p for s, p in _patient_db().items() if same(s) == same(subject)), None)
    return f"{p.name} (ID {p.hospital_id})" if p else Path(subject).name


def patient_name(key) -> str:
    """The name for a hospital ID or a folder in the patient database, else key itself."""
    k = str(key or "").replace(" ", "").lower()
    return next((p.name for p in _patient_db().values() if k in (p.hospital_id.lower(), p.folder.lower())), str(key))


# ---------------------------------------------------------------------------
# Ollama (local LLM chat assistant)
# ---------------------------------------------------------------------------

# utils.llm_devices rebinds these at runtime, so every function below reads them at call time.
# 127.0.0.1, not "localhost": that can resolve to an unreachable IPv6 address and hang.
OLLAMA_URL: str = "http://127.0.0.1:11434/api/chat"
OLLAMA_GENERATE_URL: str = "http://127.0.0.1:11434/api/generate"
OLLAMA_TAGS_URL: str = "http://127.0.0.1:11434/api/tags"
OLLAMA_SHOW_URL: str = "http://127.0.0.1:11434/api/show"
OLLAMA_MODEL: str = "qwen3.5:latest"

# Too low silently truncates the prompt (dropping the system prompt/tools) with no error.
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
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ollama create failed for {gguf_path.name}: {e.stderr}") from e
    finally:
        Path(modelfile_path).unlink(missing_ok=True)

    return tag


# Chat availability moved to utils.llm_devices.chat_available(); probing here blocked startup.
