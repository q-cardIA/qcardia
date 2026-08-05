"""LLM confirmation of cardisort's own classification when it's unsure.

CardisortClassifier's softmax output degrades into a near coin-flip between
its top-2 candidates on hard cases. classify_sequence_dir exposes this via
ClassificationResult.is_sequence_uncertain / is_plane_uncertain — True when
the top-1/top-2 probability margin for that head is below
qcardia.cardisort.CONFIDENCE_MARGIN_THRESHOLD.

When either head is uncertain, refine_classification asks the LLM to pick
between cardisort's own top-3 candidates (with their probabilities) using
DICOM metadata cardisort itself never sees, e.g. SeriesDescription. It can
only choose among what cardisort already considered plausible, never invent
a new label. This is a distinct question from qcardia.disambiguate's — "is
this classification right?" vs. "which of these same-class candidates is
primary?" — kept in its own module so each can be reasoned about and
prompt-tuned independently.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import requests

from qcardia.cardisort import ClassificationResult

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:8b"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sequence_name": {"type": "string"},
        "plane_name": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["sequence_name", "plane_name", "reasoning"],
}

PROMPT_TEMPLATE = """A deep-learning classifier (cardisort) analyzed a cardiac MRI series and was \
unsure of its sequence/plane classification — its top two candidates for at \
least one of the two labels were close in confidence.

Series metadata:
{metadata_json}

Cardisort's top candidate sequence types, most likely first (name: probability): \
{sequence_top3}
Cardisort's top candidate imaging planes, most likely first (name: probability): \
{plane_top3}

Using the metadata, pick the best (sequence_name, plane_name) pair. You may \
confirm cardisort's top-1 guess for either label, or choose a different \
candidate from the lists above if the metadata clearly supports it. Only \
choose sequence_name/plane_name values from the candidate lists above — do \
not invent a new label."""


def refine_classification(
    sequence_dir: Path, metadata: dict, result: ClassificationResult
) -> ClassificationResult:
    """No-ops (returns result unchanged) unless cardisort was uncertain about
    at least one head. Otherwise asks the LLM to pick among cardisort's top-3
    candidates per head, giving it the probabilities as context, and returns
    result with sequence_name/plane_name updated to the LLM's pick (top-3
    lists are left as-is — those come from cardisort, not the LLM).

    Never raises: if Ollama is unreachable or picks a label outside
    cardisort's candidates, logs a warning and returns cardisort's own top-1
    unchanged, so one bad/unreachable call doesn't block a batch run.
    """
    if not (result.is_sequence_uncertain or result.is_plane_uncertain):
        return result

    prompt = PROMPT_TEMPLATE.format(
        metadata_json=json.dumps(metadata, indent=2),
        sequence_top3=result.sequence_top3,
        plane_top3=result.plane_top3,
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
        picked = json.loads(response.json()["message"]["content"])

        sequence_names = {name for name, _ in result.sequence_top3}
        plane_names = {name for name, _ in result.plane_top3}
        if picked["sequence_name"] not in sequence_names or picked["plane_name"] not in plane_names:
            raise ValueError(
                f"Model picked ({picked['sequence_name']!r}, {picked['plane_name']!r}) "
                f"outside cardisort's candidates"
            )
    except (requests.RequestException, ValueError, KeyError) as e:
        logging.warning(
            f"Uncertain-classification LLM step failed for {sequence_dir.name}: {e}; "
            "keeping cardisort's top-1"
        )
        return result

    return replace(
        result,
        sequence_name=picked["sequence_name"],
        plane_name=picked["plane_name"],
        escalation_reasoning=picked["reasoning"],
    )
