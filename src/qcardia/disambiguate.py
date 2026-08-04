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
from pathlib import Path

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
