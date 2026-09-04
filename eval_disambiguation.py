"""Evaluation harness for series-disambiguation.

Classifies every reconstruction-variant group (see
cardisort.group_reconstruction_variants) across the dataset folders below,
groups those that collapse onto the same (sequence, plane) label restricted
to TARGET_SEQUENCES, and for each duplicate group compares
disambiguate.try_deterministic_primary's pick against the full LLM path in
disambiguate.disambiguate_duplicates. This mirrors run.py's pipeline
(classify per acquisition group, not per raw directory) so a magnitude/PSIR
pair exported as separate directories (seen: Siemens) is resolved the same
way here as it is in production - merged before classification, not left as
two candidates for this module to reconcile.

Classification results are cached to CACHE_PATH (keyed by the group's
directory paths, invalidated by a cheap signature of their file listings)
since running the classifier fresh over these folders takes ~15-20 minutes
on CPU. Re-running this script after a prompt/heuristic change in
disambiguate.py re-uses the cache and only re-does the (fast) disambiguation
step.

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
    classify_sequence_group,
    get_sequence_dirs,
    group_reconstruction_variants,
    load_cardisort_model,
    load_series_datasets,
    pick_processing_representative,
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


def _group_key(group: list[Path]) -> str:
    return "|".join(str(d) for d in sorted(group))


def _group_signature(group: list[Path]) -> str:
    """Cheap signature to invalidate the cache if any member's contents
    change: total file count plus the newest mtime across the whole group."""
    files = [f for d in group for f in d.rglob("*") if f.is_file()]
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


def classify_group_with_cache(
    group: list[Path], cache: dict, model, config
) -> tuple[str, str] | None:
    key = _group_key(group)
    signature = _group_signature(group)
    cached = cache.get(key)
    if cached is not None and cached.get("signature") == signature:
        prediction = cached["prediction"]
        return tuple(prediction) if prediction is not None else None

    try:
        prediction = classify_sequence_group(
            group,
            model,
            config["model"]["nr_input_channels"],
            tuple(config["data"]["target_pixdim"]),
            tuple(config["data"]["target_size"]),
            config["data"]["image_grid_sample_mode"],
            verbose=False,
        )
    except Exception as e:
        print(f"  ! failed to classify {[d.name for d in group]}: {e!r}")
        prediction = None

    cache[key] = {
        "signature": signature,
        "prediction": list(prediction) if prediction is not None else None,
    }
    return prediction


def main() -> None:
    cardisort_model, cardisort_config = load_cardisort_model(CARDISORT_WANDB_RUN_PATH)
    cache = load_cache()

    # (patient, class_label, [group, group, ...]) - one entry per duplicate
    # (sequence, plane) class with more than one acquisition group.
    duplicate_groups = []

    for dataset_dir in DATASET_DIRS:
        if not dataset_dir.exists():
            continue
        patient_list = natsorted([f for f in dataset_dir.iterdir() if f.is_dir()])
        for patient in patient_list:
            sequence_dirs = get_sequence_dirs(patient)
            acquisition_groups = group_reconstruction_variants(sequence_dirs)

            groups_by_class = defaultdict(list)
            for group in acquisition_groups:
                prediction = classify_group_with_cache(
                    group, cache, cardisort_model, cardisort_config
                )
                if prediction is None:
                    continue
                sequence_name, _ = prediction
                if sequence_name not in TARGET_SEQUENCES:
                    continue
                groups_by_class[prediction].append(group)

            for class_label, class_groups in groups_by_class.items():
                if len(class_groups) > 1:
                    duplicate_groups.append((patient, class_label, class_groups))

        save_cache(cache)

    print(f"\n{len(duplicate_groups)} duplicate group(s) found across {TARGET_SEQUENCES}\n")

    counts = defaultdict(int)
    deterministic_counts = defaultdict(int)
    agree_counts = defaultdict(int)
    compared_counts = defaultdict(int)

    for patient, class_label, class_groups in duplicate_groups:
        sequence_name, plane_name = class_label
        counts[sequence_name] += 1

        representatives = [pick_processing_representative(g) for g in class_groups]
        candidates = [
            summarize_sequence_dir(d, load_series_datasets(d)) for d in representatives
        ]

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
