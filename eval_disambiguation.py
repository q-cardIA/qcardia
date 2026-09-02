"""Evaluation harness for series-disambiguation.

Classifies every sequence directory across the dataset folders below,
groups directories that collapse onto the same (sequence, plane) label
restricted to TARGET_SEQUENCES, and for each duplicate group compares
disambiguate.try_deterministic_primary's pick against the full LLM path
in disambiguate.disambiguate_duplicates.

Classification results are cached to CACHE_PATH (keyed by directory path
plus a cheap signature of its file listing) since running the classifier
fresh over these folders takes ~15-20 minutes on CPU. Re-running this
script after a prompt/heuristic change in disambiguate.py re-uses the
cache and only re-does the (fast) disambiguation step.

There is no hand-labeled ground truth here (a deliberate choice - see
plans/starry-skipping-kahn.md): the "agreement rate" below measures how
often the LLM path agrees with the deterministic heuristic layer, not
verified accuracy. Disagreements are printed in full so they can be
eyeballed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from natsort import natsorted

from qcardia.cardisort import (
    classify_sequence_dir,
    get_sequence_dirs,
    load_cardisort_model,
    load_series_datasets,
)
from qcardia.disambiguate import (
    disambiguate_duplicates,
    summarize_sequence_dir,
    try_deterministic_primary,
)

CARDISORT_WANDB_RUN_PATH = Path.cwd() / "wandb" / "cardisort"
DATASET_DIRS = [Path.cwd() / "job-test", Path.cwd() / "data"]
CACHE_PATH = Path.cwd() / ".cardisort_cache.json"

TARGET_SEQUENCES = {"CINE", "DBLGE", "WBLGE", "PERF", "PC"}


def _dir_signature(sequence_dir: Path) -> str:
    """Cheap signature to invalidate the cache if a directory's contents
    change: file count plus the newest mtime among its files."""
    files = [f for f in sequence_dir.rglob("*") if f.is_file()]
    if not files:
        return "empty"
    newest_mtime = max(f.stat().st_mtime for f in files)
    return f"{len(files)}:{newest_mtime}"


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def classify_with_cache(
    sequence_dir: Path, cache: dict, model, config
) -> tuple[str, str] | None:
    key = str(sequence_dir)
    signature = _dir_signature(sequence_dir)
    cached = cache.get(key)
    if cached is not None and cached.get("signature") == signature:
        prediction = cached["prediction"]
        return tuple(prediction) if prediction is not None else None

    try:
        prediction = classify_sequence_dir(
            sequence_dir,
            model,
            config["model"]["nr_input_channels"],
            tuple(config["data"]["target_pixdim"]),
            tuple(config["data"]["target_size"]),
            config["data"]["image_grid_sample_mode"],
            verbose=False,
        )
    except Exception as e:
        print(f"  ! failed to classify {sequence_dir}: {e!r}")
        prediction = None

    cache[key] = {
        "signature": signature,
        "prediction": list(prediction) if prediction is not None else None,
    }
    return prediction


def main() -> None:
    cardisort_model, cardisort_config = load_cardisort_model(CARDISORT_WANDB_RUN_PATH)
    cache = load_cache()

    # (sequence_name, plane_name) -> list of candidate summary dicts, one per patient group
    groups = []

    for dataset_dir in DATASET_DIRS:
        if not dataset_dir.exists():
            continue
        patient_list = natsorted([f for f in dataset_dir.iterdir() if f.is_dir()])
        for patient in patient_list:
            sequence_dirs = get_sequence_dirs(patient)

            dirs_by_class = defaultdict(list)
            for sequence_dir in sequence_dirs:
                prediction = classify_with_cache(
                    sequence_dir, cache, cardisort_model, cardisort_config
                )
                if prediction is None:
                    continue
                sequence_name, _ = prediction
                if sequence_name not in TARGET_SEQUENCES:
                    continue
                dirs_by_class[prediction].append(sequence_dir)

            for class_label, dirs in dirs_by_class.items():
                if len(dirs) > 1:
                    groups.append((patient, class_label, dirs))

        save_cache(cache)

    print(f"\n{len(groups)} duplicate group(s) found across {TARGET_SEQUENCES}\n")

    counts = defaultdict(int)
    deterministic_counts = defaultdict(int)
    agree_counts = defaultdict(int)
    compared_counts = defaultdict(int)

    for patient, class_label, dirs in groups:
        sequence_name, plane_name = class_label
        counts[sequence_name] += 1

        candidates = [summarize_sequence_dir(d, load_series_datasets(d)) for d in dirs]

        deterministic_result = try_deterministic_primary(class_label, candidates)
        if deterministic_result is not None:
            deterministic_counts[sequence_name] += 1

        try:
            llm_result = disambiguate_duplicates(class_label, candidates)
        except Exception as e:
            print(f"  ! LLM disambiguation failed for {patient.name}/{class_label}: {e!r}")
            llm_result = None

        if deterministic_result is not None and llm_result is not None:
            compared_counts[sequence_name] += 1
            agree = deterministic_result["primary_series"] == llm_result["primary_series"]
            if agree:
                agree_counts[sequence_name] += 1
            else:
                print(
                    f"DISAGREEMENT {patient.name} {class_label}: "
                    f"heuristic={deterministic_result['primary_series']!r} "
                    f"llm={llm_result['primary_series']!r}"
                )
                for c in candidates:
                    print(f"    {c}")
                for a in llm_result["assessments"]:
                    print(f"    llm: {a['series']} -> {a['role']} ({a['reasoning']})")

    print("\nSummary by sequence:")
    for sequence_name in sorted(counts):
        n = counts[sequence_name]
        n_det = deterministic_counts[sequence_name]
        n_cmp = compared_counts[sequence_name]
        n_agree = agree_counts[sequence_name]
        agree_str = f"{n_agree}/{n_cmp} agree" if n_cmp else "n/a"
        print(
            f"  {sequence_name:10} groups={n:3}  resolved_deterministically={n_det:3}  {agree_str}"
        )


if __name__ == "__main__":
    main()
