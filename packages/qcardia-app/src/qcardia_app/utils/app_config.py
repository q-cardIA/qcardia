"""Application path/config constants — new users start here.

Layout assumed by the defaults:
    <WORKSPACE>/
    ├── <repo>/             ← this repository; the app is packages/qcardia-app
    └── data/             ← subject folders (any names; changeable on the welcome screen)

If your layout differs, update the paths below.
"""
import csv
import os
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QSettings

# The package itself; bundled assets like the window icon live in it.
PACKAGE_DIR: Path = Path(__file__).resolve().parents[1]
ASSETS_DIR: Path = PACKAGE_DIR / "assets"
ICON_PATH: Path = ASSETS_DIR / "images" / "q-cardIA.png"

# packages/qcardia-app in the repository (logs, tests). Assumes an editable install.
APP_DIR: Path = PACKAGE_DIR.parents[1]
# Root of the workspace that holds the repository and the data, next to each other.
WORKSPACE_DIR: Path = APP_DIR.parents[2]

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


# A patient list: per subject folder a made-up name and hospital ID (columns hospital_id,name,folder). A
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


def patient_of(subject: str | Path) -> Patient | None:
    """A subject's patient, by its path or its entry in the patient list; None if the list does not name it."""
    same = lambda s: os.path.normcase(os.path.abspath(data_dir() / s))  # noqa: E731 (no disk access)
    return next((p for s, p in _patient_db().items() if same(s) == same(subject)), None)


def patient_label(subject: str | Path) -> str:
    """A subject's patient name and ID; else its folder name."""
    p = patient_of(subject)
    return f"{p.name} (ID {p.hospital_id})" if p else Path(subject).name


def patient_name(key) -> str:
    """The name for a hospital ID or a folder in the patient database, else key itself."""
    k = str(key or "").replace(" ", "").lower()
    return next((p.name for p in _patient_db().values() if k in (p.hospital_id.lower(), p.folder.lower())), str(key))
