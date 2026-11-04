"""Reads and writes the .cardisort.json sidecar: cardisort classification + disambiguation, cached beside each subject folder."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SIDECAR_NAME = ".cardisort.json"
# Older labels are made again. 2: Siemens series of one protocol are classified together.
# 3: the duplicate disambiguation prefers dark-blood LGE.
LABELS_VERSION = 3


@dataclass
class SeriesLabel:
    dir: str
    sequence_name: str | None
    plane_name: str | None
    key: str | None


@dataclass
class SubjectLabels:
    series: list[SeriesLabel] = field(default_factory=list)
    primary_by_key: dict[str, dict] = field(default_factory=dict)


def sidecar_path(subject_dir: Path) -> Path:
    return Path(subject_dir) / SIDECAR_NAME


def load_labels(subject_dir: Path) -> SubjectLabels | None:
    """None if the subject has no sidecar yet, it can't be parsed, or it is from an older version."""
    path = sidecar_path(subject_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != LABELS_VERSION:
            return None
        series = [SeriesLabel(**s) for s in data.get("series", [])]
        return SubjectLabels(series=series, primary_by_key=data.get("primary_by_key", {}))
    except Exception:
        return None


def save_labels(subject_dir: Path, labels: SubjectLabels) -> None:
    data = {"version": LABELS_VERSION, "series": [asdict(s) for s in labels.series],
            "primary_by_key": labels.primary_by_key}
    sidecar_path(subject_dir).write_text(json.dumps(data, indent=2), encoding="utf-8")


def update_primary(subject_dir: Path, key: str, candidates: list[str], primary: str) -> bool:
    """Updates one disambiguation entry in place; no-op if the subject has no sidecar yet."""
    labels = load_labels(subject_dir)
    if labels is None:
        return False
    labels.primary_by_key[key] = {"candidates": sorted(candidates), "primary": primary}
    save_labels(subject_dir, labels)
    return True
