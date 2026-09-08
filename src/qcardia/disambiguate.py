"""LLM-based disambiguation of sequence directories that share a
(sequence, plane) classification.

CardisortClassifier predicts a (sequence, plane) label per directory
(e.g. ("CINE", "SAX")), but a patient folder can contain several directories
with the same label that aren't interchangeable: a planning/pilot
acquisition alongside the real diagnostic stack, a test scan
alongside the real perfusion acquisition, or a stress/rest pair that both
end up with the same PERF/SAX label. This module picks the primary series in
each such group.

try_deterministic_primary resolves what it can from metadata the scanner
generates itself (instance count, acquisition time) rather than from
series_description: that field is typed by hand, inconsistent across sites,
and tied to whatever language/convention a given site uses. Everything else
goes to a local LLM (via Ollama), which is passed the description as context
but isn't given a hard-coded rule to apply to it.

A pixel-intensity-based signal for telling dark-blood LGE apart from a
non-suppressed counterpart was tried and deliberately left out: it required
a per-vendor calibration tag (RealWorldValueMappingSequence) that isn't
universal, making it a vendor-specific rule in general-purpose clothing
rather than an actual cross-vendor heuristic - not worth the added
complexity for a rule that would only ever fire on some scanners' data.

Note: magnitude-vs-PSIR reconstruction pairs (e.g. Siemens exporting them as
separate directories) are resolved upstream of this module, before
classification even runs - see group_reconstruction_variants and
pick_processing_representative in cardisort.py - so they never reach here as
separate candidates in the first place.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pydicom
import requests

from qcardia.cardisort import get_acquisition_time_seconds, get_frame_pixel_spacing

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
the others are typically low-effort planning/pilot acquisitions, repeated/failed \
attempts, or otherwise not meant to be used downstream.

The "series_description" field below is typed by hand at scan time and is \
NOT a reliable rule by itself: it can be inconsistent, abbreviated, in a \
different language than you'd expect, or even misleading (e.g. an unrelated \
anatomy-reference scan whose description happens to contain the word "REST"). \
Treat it as weak supporting context, not as the primary signal — prefer \
metadata the scanner generates itself, like n_instances, series_number, and \
acquisition_time_seconds, when it's informative. Note that a low-effort \
planning/pilot acquisition is often (not always) acquired earlier in the exam \
than the corresponding diagnostic series.

Given the metadata below for each candidate series, identify which ONE series \
is the primary series for this class, and briefly classify the role of each \
of the others (planning, repeat, or other).

Candidates:
{candidates_json}

Respond with the primary series name and a role + one-sentence reasoning for \
every candidate, including the primary one."""

# A candidate needs fewer than this fraction of the group's largest
# n_instances to be flagged as a low-instance-count outlier. A planning/
# pilot/test acquisition covers far fewer slices and/or cardiac phases than
# the real diagnostic series, so this catches it regardless of what its
# description says.
LOW_INSTANCE_COUNT_RATIO = 1 / 3


def summarize_sequence_dir(sequence_dir: Path, datasets: list[pydicom.Dataset]) -> dict:
    """DICOM metadata relevant to picking the primary series in a duplicate
    group. series_description is included as LLM context only; nothing in
    this module builds a rule on it."""
    ds0 = datasets[0]
    try:
        pixel_spacing_mm = [float(v) for v in get_frame_pixel_spacing(ds0)]
    except AttributeError:
        pixel_spacing_mm = None
    return {
        "series": sequence_dir.name,
        "series_description": str(getattr(ds0, "SeriesDescription", "")),
        "series_number": getattr(ds0, "SeriesNumber", None),
        "n_instances": len(datasets),
        "rows": getattr(ds0, "Rows", None),
        "columns": getattr(ds0, "Columns", None),
        "slice_thickness": getattr(ds0, "SliceThickness", None),
        "pixel_spacing_mm": pixel_spacing_mm,
        "acquisition_time_seconds": get_acquisition_time_seconds(ds0),
        "image_type": list(ds0.ImageType) if "ImageType" in ds0 else None,
    }


def flag_low_instance_count(candidates: list[dict]) -> dict[str, list[str]]:
    """Flags candidates with a dramatically lower n_instances than the
    largest candidate in the group. See LOW_INSTANCE_COUNT_RATIO."""
    counts = {c["series"]: c.get("n_instances") for c in candidates}
    valid = {name: n for name, n in counts.items() if isinstance(n, int) and n > 0}
    if len(valid) < 2:
        return {c["series"]: [] for c in candidates}

    max_count = max(valid.values())
    flags = {}
    for c in candidates:
        name = c["series"]
        count = counts.get(name)
        if (
            isinstance(count, int)
            and count > 0
            and count < LOW_INSTANCE_COUNT_RATIO * max_count
        ):
            flags[name] = [
                f"only {count} instances vs. {max_count} for the largest "
                "candidate in this group"
            ]
        else:
            flags[name] = []
    return flags


def _resolve_perf_stress_rest(remaining: list[dict]) -> dict | None:
    """After instance-count filtering leaves exactly two PERF/SAX candidates,
    picks the earlier-acquired one as stress (primary) - stress is acquired
    before rest in this protocol. Uses acquisition_time_seconds directly
    rather than series_number, which is only a proxy for acquisition order.

    Returns {"primary": name, "primary_reasoning": str, "other_reasoning":
    {name: str}}, or None if either candidate's acquisition time is missing,
    or the two are equal - an identical acquisition_time_seconds is a sign
    the pair is an original and a reprocessed re-export of the same
    acquisition.
    """
    if len(remaining) != 2:
        return None
    times = [c.get("acquisition_time_seconds") for c in remaining]
    if any(t is None for t in times) or times[0] == times[1]:
        return None
    stress, rest = sorted(remaining, key=lambda c: c["acquisition_time_seconds"])
    return {
        "primary": stress["series"],
        "primary_reasoning": "Earlier-acquired of the two remaining PERF/SAX "
        "candidates after instance-count filtering; stress is acquired before rest.",
        "other_reasoning": {
            rest["series"]: "Later-acquired of the two remaining PERF/SAX "
            "candidates after instance-count filtering, i.e. the rest acquisition."
        },
    }


def try_deterministic_primary(class_label: tuple[str, str], candidates: list[dict]) -> dict | None:
    """Resolves a duplicate group without the LLM, using only
    scanner-generated metadata (never series_description - see the module
    docstring):

    1. flag_low_instance_count - if exactly one candidate isn't a
       low-instance-count outlier, it's the primary.
    2. If step 1 leaves multiple candidates, a sequence-specific resolver may
       settle it: _resolve_perf_stress_rest for PERF/SAX (earlier-acquired
       is stress, exactly two remaining only).

    Returns a result shaped like disambiguate_duplicates' ({"primary_series":
    str, "assessments": [...]}), or None if nothing resolves it (i.e. the
    group needs the LLM) - e.g. an LGE/SAX group with two comparably-sized
    candidates, where no scanner-generated field distinguishes them (see
    LGE_HEURISTIC_HINT).
    """
    sequence_name, plane_name = class_label
    flags = flag_low_instance_count(candidates)
    remaining = [c for c in candidates if not flags[c["series"]]]
    excluded = [c for c in candidates if flags[c["series"]]]

    resolution = None
    if len(remaining) == 1:
        primary_name = remaining[0]["series"]
    else:
        if sequence_name == "PERF" and plane_name == "SAX":
            resolution = _resolve_perf_stress_rest(remaining)
        if resolution is None:
            return None
        primary_name = resolution["primary"]

    assessments = []
    for c in candidates:
        name = c["series"]
        if name == primary_name:
            reasoning = (
                resolution["primary_reasoning"]
                if resolution
                else "No dramatically lower instance count relative to its peers."
            )
            assessments.append({"series": name, "role": "primary", "reasoning": reasoning})
        elif any(name == e["series"] for e in excluded):
            assessments.append(
                {
                    "series": name,
                    "role": "planning",
                    "reasoning": "Flagged as non-primary: " + "; ".join(flags[name]) + ".",
                }
            )
        else:
            assessments.append(
                {"series": name, "role": "other", "reasoning": resolution["other_reasoning"][name]}
            )
    return {"primary_series": primary_name, "assessments": assessments}


LGE_HEURISTIC_HINT = """

For LGE specifically: reconstruction variants of one acquisition \
(e.g. a magnitude/PSIR pair) are already merged before classification even \
runs (see group_reconstruction_variants in cardisort.py), and low-instance- \
count outliers are already excluded - so if you're seeing multiple \
comparably-sized SAX candidates here (there may be more than two - e.g. one \
dark-blood acquisition alongside two separate white-blood ones), they're \
genuinely different acquisitions, NOT repeated attempts of each other - \
dark-blood and white-blood are different acquisition techniques by design, \
so acquisition-time ordering between them means nothing here and should be \
ignored; this is not a "prefer the later one" situation. We want to take \
the dark-blood one as the primary series regardless of acquisition order, \
Typically the series description will give a hint, e.g. containing \
"DB", "dark blood", or "black blood" for the dark-blood acquisition, but \
don't treat that as a hard rule: it can be inconsistent, abbreviated, in a \
different language than you'd expect, or even misleading. In particular, \
"BB" is a confirmed trap in real data - it reads like "bright blood" but has \
been observed labeling a dark-blood-style series, so don't assume it means \
non-dark. Use the series description as weak supporting context, not as the \
primary signal.

Separately: ONLY if you think there is a repeat, i.e. two candidates have the \
exact same series description or there is a clear hint in the description (e.g. containing "repeat" or "retest") \
if two candidates are the SAME blood-suppression type (both \
dark-blood, or both non-dark) with matching/near-identical descriptions and \
a real time gap between them (not the near-simultaneous timing of a \
magnitude/PSIR reconstruction pair, already merged before you see it), \
that's a genuine repeat of the same protocol - the description may hint at \
this (e.g. containing "repeat" or "retest") - only if its a repeat situation, \
prefer the LATER one, since a \
technologist redoes a scan because the earlier attempt had a quality problem \
(motion, a bad breath-hold), not the other way round."""

PERF_SAX_HEURISTIC_HINT = """

For perfusion SAX specifically: there are two main types of acquisitions: stress and rest. \
The primary series is a stress acquisition, \
not a rest acquisition. This is usually already resolved deterministically \
from acquisition_time_seconds (stress is acquired first) when exactly two \
candidates remain before the LLM is consulted at all; you're only seeing \
this because that didn't resolve it - e.g. a missing timestamp, or more than \
two candidates remaining. Prefer acquisition_time_seconds and series_number \
over the description generally to tell stress and rest apart.

Separately: ONLY if you think there is a repeat, i.e. two candidates have the \
exact same series description or there is a clear hint in the description (e.g. containing "repeat" or "retest") \
if two candidates are the SAME type (both stress-like, or both \
rest-like) with matching/near-identical descriptions and a real time gap \
between them (not the near-simultaneous timing of a reconstruction-variant \
pair, already merged before you see it), that's a genuine repeat of the same \
protocol - the description may hint at this (e.g. containing "repeat" or \
"retest") - prefer the LATER one, since a technologist redoes a scan because \
the earlier attempt had a quality problem (motion, arrhythmia), not the \
other way round. """

# Per-sequence heuristic hints appended to the LLM prompt, keyed by exact
# sequence name. PERF and LGE are handled separately in _build_heuristic_hint
# (PERF's hint also depends on plane; LGE is matched by substring rather than
# exact name - see there for why) - populate further entries (PC, CINE, ...)
# from real observed candidates rather than guessing.
HEURISTIC_HINTS: dict[str, str] = {}


def _build_heuristic_hint(
    sequence_name: str, plane_name: str, candidates: list[dict]
) -> str:
    """Prior knowledge appended to the LLM prompt: a per-sequence hint (if
    any) plus a note about any candidates flag_low_instance_count already
    flagged (the same signal try_deterministic_primary uses)."""
    hint = ""
    if sequence_name == "PERF" and plane_name == "SAX":
        hint += PERF_SAX_HEURISTIC_HINT
    elif "LGE" in sequence_name:
        # Matched by substring, not an exact "DBLGE"/"WBLGE" key: a future
        # classifier is expected to merge these into one class under a name
        # that isn't decided yet, and this hint's content doesn't depend on
        # which of the old class names it was - only that it's an LGE
        # duplicate group - so matching stays correct across that rename as
        # long as "LGE" remains part of the class name.
        hint += LGE_HEURISTIC_HINT
    else:
        hint += HEURISTIC_HINTS.get(sequence_name, "")

    flags = flag_low_instance_count(candidates)
    flagged = {name: reasons for name, reasons in flags.items() if reasons}
    if flagged:
        flagged_lines = "\n".join(
            f"- {name}: {'; '.join(reasons)}" for name, reasons in flagged.items()
        )
        hint += (
            "\n\nAn automated check flagged the following candidates as likely "
            f"non-primary (a dramatically lower instance count than its peers "
            f"in this group):\n{flagged_lines}\nWeigh this alongside the rest "
            "of the metadata."
        )
    return hint


def disambiguate_duplicates(
    class_label: tuple[str, str], candidates: list[dict]
) -> dict:
    """Picks the primary series among directories that share a (sequence,
    plane) classification.

    Tries try_deterministic_primary first - resolved purely from
    scanner-generated metadata, no LLM call needed. Otherwise asks a local
    Qwen model (via Ollama), with the prompt extended by
    _build_heuristic_hint: a per-sequence prior (e.g. LGE_HEURISTIC_HINT,
    PERF_SAX_HEURISTIC_HINT) plus a note on any instance-count-flagged
    candidates. The LLM still makes the final call in both cases.

    Retries once (after a short backoff) on a transient Ollama request
    failure or a malformed/incomplete response.

    Returns the parsed JSON response: {"primary_series": str, "assessments": [...]}.
    Raises RuntimeError if Ollama is unreachable or keeps returning an
    unusable response after retrying.
    """
    deterministic_result = try_deterministic_primary(class_label, candidates)
    if deterministic_result is not None:
        return deterministic_result

    sequence_name, plane_name = class_label
    prompt = PROMPT_TEMPLATE.format(
        sequence_name=sequence_name,
        plane_name=plane_name,
        candidates_json=json.dumps(candidates, indent=2),
    )
    prompt += _build_heuristic_hint(sequence_name, plane_name, candidates)

    candidate_names = {c["series"] for c in candidates}

    last_error: Exception | None = None
    for attempt in range(2):
        if attempt > 0:
            time.sleep(1)

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": RESPONSE_SCHEMA,
                    "stream": False,
                    "think": True,
                    "options": {"temperature": 0},
                },
                timeout=600,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            last_error = e
            continue

        content = response.json()["message"]["content"]
        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            last_error = e
            continue

        if result.get("primary_series") not in candidate_names:
            last_error = RuntimeError(
                f"Model picked unknown series {result.get('primary_series')!r}, "
                f"expected one of {candidate_names}"
            )
            continue

        assessed_names = {a["series"] for a in result.get("assessments", [])}
        if assessed_names != candidate_names:
            last_error = RuntimeError(
                f"Model's assessments {assessed_names} didn't cover all "
                f"candidates {candidate_names}"
            )
            continue

        return result

    if isinstance(last_error, requests.RequestException):
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL} (is `ollama serve` running?)"
        ) from last_error
    raise RuntimeError(
        f"Ollama kept returning an unusable response: {last_error}"
    ) from last_error
