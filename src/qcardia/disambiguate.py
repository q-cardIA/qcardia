"""LLM-based disambiguation of sequence directories that share a
(sequence, plane) classification.

CardisortClassifier predicts a coarse (sequence, plane) label per directory
(e.g. ("CINE", "SAX")), but a patient folder can contain several directories
with the same label that aren't interchangeable. For most classes this is a
low-res planning cine vs. the real diagnostic stack, or a repeated/failed
localizer attempt, so the LLM picks a single primary series and labels the
rest planning/repeat/other.

Some classes need a different role vocabulary: perfusion at SAX covers both
rest and stress acquisitions, which are clinically distinct and should both
be identified, not have one discarded as a "repeat" of the other.
ROLE_SCHEMAS lets a (sequence_name, plane_name) class swap in its own role
list and prompt guidance for this. primary_series is still always requested
(see disambiguate_duplicates) — the app currently only keeps one series per
class downstream, so until it can keep more than one (e.g. both REST_PERF
and STRESS_PERF), primary_series is what actually gets used; that's a
separate, later change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pydicom
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:8b"

DEFAULT_ROLES = ["primary", "planning", "repeat", "other"]


@dataclass
class RoleSchema:
    roles: list[str] = field(default_factory=lambda: list(DEFAULT_ROLES))
    prompt_hint: str = ""


ROLE_SCHEMAS: dict[tuple[str, str], RoleSchema] = {
    ("PERF", "SAX"): RoleSchema(
        roles=["rest", "stress", "planning", "repeat", "other"],
        prompt_hint=(
            "This class covers both rest and stress perfusion acquisitions. "
            "Rest and stress are clinically distinct and should both be "
            'identified by role, not collapsed into one primary/repeat pair. '
            "Rest vs stress is often deducible from the series description: "
            'if one candidate\'s description clearly says "stress", the other '
            'duplicate candidate is likely "rest", and vice versa.'
        ),
    ),
    ("TestPERF", "SAX"): RoleSchema(
        roles=["rest", "stress", "planning", "repeat", "other"],
        prompt_hint=(
            "This class covers both rest and stress perfusion acquisitions. "
            "Rest and stress are clinically distinct and should both be "
            'identified by role, not collapsed into one primary/repeat pair. '
            "Rest vs stress is often deducible from the series description: "
            'if one candidate\'s description clearly says "stress", the other '
            'duplicate candidate is likely "rest", and vice versa.'
        ),
    ),
}


def _response_schema(roles: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "primary_series": {
                "type": "string",
                "description": "Name of the single sequence directory to treat as "
                "primary for this class for now — even for a class with a richer "
                "role vocabulary (e.g. rest/stress), the app can currently only "
                "keep one series downstream per class.",
            },
            "assessments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "series": {"type": "string"},
                        "role": {"type": "string", "enum": roles},
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
{class_hint}
Given the metadata below for each candidate series, classify the role of \
each candidate using ONLY these roles: {roles}. Also identify which ONE \
series should be treated as the primary series for this class.

Candidates:
{candidates_json}

Respond with the primary series name and a role + one-sentence reasoning for \
every candidate."""


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
        "acquisition_date": getattr(ds0, "AcquisitionDate", None) or getattr(ds0, "SeriesDate", None),
        "acquisition_time": getattr(ds0, "AcquisitionTime", None) or getattr(ds0, "SeriesTime", None),
    }


def disambiguate_duplicates(
    class_label: tuple[str, str], candidates: list[dict]
) -> dict:
    """Asks a local Qwen model (via Ollama) to assign a role to every
    candidate directory that shares a (sequence, plane) classification, and
    pick one as primary_series. The role vocabulary is ROLE_SCHEMAS.get(
    class_label, default primary/planning/repeat/other).

    Returns the parsed JSON response: {"primary_series": str, "assessments": [...]}.
    Raises RuntimeError if Ollama is unreachable or returns an unparsable response.
    """
    sequence_name, plane_name = class_label
    schema = ROLE_SCHEMAS.get(class_label, RoleSchema())
    prompt = PROMPT_TEMPLATE.format(
        sequence_name=sequence_name,
        plane_name=plane_name,
        class_hint=f"\n{schema.prompt_hint}\n" if schema.prompt_hint else "",
        roles=", ".join(schema.roles),
        candidates_json=json.dumps(candidates, indent=2),
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "format": _response_schema(schema.roles),
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
