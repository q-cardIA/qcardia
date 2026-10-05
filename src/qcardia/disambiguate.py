"""LLM-based disambiguation of sequence directories that share a
(sequence, plane) classification.

CardisortClassifier predicts a coarse (sequence, plane) label per directory
(e.g. ("CINE", "SAX")), but a patient folder can contain several directories
with the same label that aren't interchangeable: a low-res planning cine
alongside the real diagnostic SAX stack, or repeated scout/localizer
attempts. This module asks a local LLM (via Ollama) to pick the primary
series in each such group from DICOM metadata, so downstream steps can use
the intended series instead of an arbitrary one.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:8b"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_series": {
            "type": "string",
            "description": "Name of the sequence directory that is the primary "
            "series for this class.",
        },
        "assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "series": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": ["primary", "planning", "repeat", "other"],
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["series", "role", "reasoning"],
            },
        },
    },
    "required": ["primary_series", "assessments"],
}

PROMPT_TEMPLATE = """You are organizing cardiac MRI DICOM series for a research pipeline.

A deep-learning classifier has labeled all of the following series as the same \
sequence/plane class: {sequence_name}/{plane_name}.

In a patient folder, several series often collapse into the same class even \
though only one of them is the intended diagnostic series for that class — \
the others are typically low-res planning acquisitions, repeated/failed \
localizer attempts, or otherwise not meant to be used downstream.

Given the metadata below for each candidate series, identify which ONE series \
is the primary series for this class, and briefly classify the role of each \
of the others (planning, repeat, or other).

Candidates:
{candidates_json}

Respond with the primary series name and a role + one-sentence reasoning for \
every candidate, including the primary one."""


def summarize_sequence_dir(sequence_dir: Path, datasets: list[pydicom.Dataset]) -> dict:
    """DICOM metadata relevant to picking the primary series in a duplicate group."""
    ds0 = datasets[0]
    return {
        "series": sequence_dir.name,
        "series_description": str(getattr(ds0, "SeriesDescription", "")),
        "series_number": getattr(ds0, "SeriesNumber", None),
        "n_instances": len(datasets),
        "rows": getattr(ds0, "Rows", None),
        "columns": getattr(ds0, "Columns", None),
        "slice_thickness": getattr(ds0, "SliceThickness", None),
    }


def disambiguate_duplicates(
    class_label: tuple[str, str], candidates: list[dict]
) -> dict:
    """Asks a local Qwen model (via Ollama) to pick the primary series among
    directories that share a (sequence, plane) classification.

    Returns the parsed JSON response: {"primary_series": str, "assessments": [...]}.
    Raises RuntimeError if Ollama is unreachable or returns an unparsable response.
    """
    sequence_name, plane_name = class_label
    prompt = PROMPT_TEMPLATE.format(
        sequence_name=sequence_name,
        plane_name=plane_name,
        candidates_json=json.dumps(candidates, indent=2),
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "format": RESPONSE_SCHEMA,
                "stream": False,
                "think": False,
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL} (is `ollama serve` running?)"
        ) from e

    content = response.json()["message"]["content"]
    result = json.loads(content)

    candidate_names = {c["series"] for c in candidates}
    if result["primary_series"] not in candidate_names:
        raise RuntimeError(
            f"Model picked unknown series {result['primary_series']!r}, "
            f"expected one of {candidate_names}"
        )
    return result


# Cardisort labels that a perfusion series (a stress, rest or test run, or one
# slice or the AIF of a run that Siemens stores as separate series) can get.
PERFUSION_SEQUENCE_NAMES = {"PERF", "TestPERF"}
# A run with fewer dynamics than this fraction of the longest run is a test run.
TEST_RUN_DYNAMICS_FRACTION = 0.5


def _dicom_time_to_seconds(dicom_time) -> float:
    dicom_time = str(dicom_time)
    return 3600 * int(dicom_time[0:2]) + 60 * int(dicom_time[2:4]) + float(dicom_time[4:])


def _summarize_perfusion_dir(sequence_dir: Path) -> dict | None:
    """Header information used to group perfusion series into runs, or None if
    the directory has no dynamic images that a series class can load."""
    images_per_slice = Counter()
    acquisition_times = []
    for f in sequence_dir.iterdir():
        if not f.is_file() or f.name.startswith(".") or "dicomdir" in f.name.lower():
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
        # Enhanced (multi-frame) images have no per-file position, and
        # BaseSeries cannot load them.
        if "ImagePositionPatient" not in ds or "AcquisitionTime" not in ds:
            continue
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
        position = float(
            np.dot(ds.ImagePositionPatient, np.cross(orientation[:3], orientation[3:]))
        )
        # The images of one perfusion slice share a position and a trigger
        # time. The Philips AIF slice can share its position with the basal
        # slice, but not its trigger time.
        images_per_slice[(round(position, 1), getattr(ds, "TriggerTime", None))] += 1
        acquisition_times.append(_dicom_time_to_seconds(ds.AcquisitionTime))

    # Cine frames, or single anatomy images, give one image per group.
    if not images_per_slice or max(images_per_slice.values()) == 1:
        return None
    return {
        "dir": sequence_dir,
        "n_dynamics": max(images_per_slice.values()),
        "start_time": min(acquisition_times),
        "end_time": max(acquisition_times),
    }


def select_perfusion_series(perfusion_dirs: list[Path]) -> dict:
    """Finds the stress and rest perfusion runs among the directories that
    cardisort labelled as perfusion.

    Directories whose acquisition times overlap belong to the same run (Siemens
    stores each slice and the AIF of a run as a separate series). Test runs are
    much shorter than the others, and stress is acquired before rest.

    Returns:
        {"stress": [Path, ...], "rest": [Path, ...]}, the directories of each
        run, or {} if there are not exactly two runs after the test runs are
        removed.
    """
    summaries = [_summarize_perfusion_dir(d) for d in perfusion_dirs]
    summaries = sorted(
        (s for s in summaries if s is not None), key=lambda s: s["start_time"]
    )

    runs = []
    for s in summaries:
        if runs and s["start_time"] <= runs[-1]["end_time"]:
            runs[-1]["dirs"].append(s["dir"])
            runs[-1]["end_time"] = max(runs[-1]["end_time"], s["end_time"])
            runs[-1]["n_dynamics"] = max(runs[-1]["n_dynamics"], s["n_dynamics"])
        else:
            runs.append({**s, "dirs": [s["dir"]]})
    if not runs:
        return {}

    longest = max(run["n_dynamics"] for run in runs)
    runs = [r for r in runs if r["n_dynamics"] >= TEST_RUN_DYNAMICS_FRACTION * longest]
    if len(runs) != 2:
        print(
            f"    Warning: expected a stress and a rest perfusion run, found "
            f"{len(runs)}: {[[d.name for d in r['dirs']] for r in runs]}; "
            f"perfusion skipped."
        )
        return {}

    return {role: run["dirs"] for role, run in zip(("stress", "rest"), runs)}


FLOW_SEQUENCE_NAME = "PC"
FLOW_VESSELS = {"AORTA": "aorta", "MPA": "pulmonary"}  # cardisort plane -> vessel
# Siemens starts the series of one acquisition (reference, magnitude, phase) milliseconds apart.
FLOW_SAME_ACQUISITION_SECONDS = 2.0


def is_phase_image(ds) -> bool:
    """The velocity image of a flow acquisition: P is the third (Siemens) or
    fourth (Philips) value of ImageType."""
    return "P" in [str(v).upper() for v in ds.get("ImageType", [])][2:4]


def _has_phase_images(dirs: list[Path]) -> bool:
    for d in dirs:
        for f in d.iterdir():
            if not f.is_file() or f.name.startswith("."):
                continue
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=["ImageType"])
            except Exception:
                continue
            if is_phase_image(ds):
                return True
    return False


def _summarize_flow_dir(sequence_dir: Path) -> dict | None:
    """The slice geometry and acquisition time of a series, from its first
    image, or None if it has no image with a position."""
    for f in sorted(sequence_dir.iterdir()):
        if not f.is_file() or f.name.startswith(".") or "dicomdir" in f.name.lower():
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
        if "ImagePositionPatient" not in ds or "AcquisitionTime" not in ds:
            continue
        return {
            "geometry": (
                tuple(round(float(v), 1) for v in ds.ImagePositionPatient),
                tuple(round(float(v), 3) for v in ds.ImageOrientationPatient),
            ),
            "time": _dicom_time_to_seconds(ds.AcquisitionTime),
        }
    return None


def _same_acquisition(a: dict, b: dict) -> bool:
    return a["geometry"] == b["geometry"] and abs(a["time"] - b["time"]) <= FLOW_SAME_ACQUISITION_SECONDS


def select_flow_series(labels: dict[Path, tuple[str | None, str | None]]) -> dict:
    """Finds the phase-contrast flow acquisition of each vessel.

    Siemens stores one acquisition as three series (reference, magnitude and
    phase), and cardisort can label only some of them PC. So each PC series
    starts an acquisition, and every series with the same slice and about the
    same acquisition time joins it, whatever its label. The vessel is the
    plane most of its PC series are labelled with. Acquisitions without phase
    images (e.g. a valve cine labelled PC) are skipped, and when a vessel was
    acquired more than once, the latest acquisition is kept.

    Args:
        labels: cardisort's (sequence_name, plane_name) for each series
            directory of the subject.

    Returns:
        {"aorta": [Path, ...], "pulmonary": [Path, ...]}, the directories of
        each vessel's acquisition, for the vessels that were found.
    """
    if not any(sequence == FLOW_SEQUENCE_NAME for sequence, _ in labels.values()):
        return {}
    summaries = {d: _summarize_flow_dir(d) for d in labels}
    summaries = {d: s for d, s in summaries.items() if s is not None}

    acquisitions = []
    for d, (sequence, plane) in labels.items():
        if sequence != FLOW_SEQUENCE_NAME or d not in summaries:
            continue
        acq = next((a for a in acquisitions if _same_acquisition(a, summaries[d])), None)
        if acq is None:
            acq = {**summaries[d], "votes": Counter()}
            acquisitions.append(acq)
        if plane in FLOW_VESSELS:
            acq["votes"][FLOW_VESSELS[plane]] += 1

    selected = {}
    for acq in sorted(acquisitions, key=lambda a: a["time"]):
        dirs = sorted(d for d, s in summaries.items() if _same_acquisition(acq, s))
        if acq["votes"] and _has_phase_images(dirs):
            selected[acq["votes"].most_common(1)[0][0]] = dirs  # a later acquisition replaces an earlier one
    return selected
