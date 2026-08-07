"""Warms the duplicate-series disambiguation cache (model/dicom_model.py,
utils/app_config.DISAMBIGUATION_CACHE_DIR) for every subject on disk.

Loading a subject calls an LLM (qcardia.disambiguate.disambiguate_duplicates)
once per duplicate-candidate group, and that call is the dominant cost of
total load time. The result only depends on which candidate directory names
exist for that subject, so it's cached to disk after the first computation
-- this script just does that "first computation" for every subject up
front, so a live demo only pays the cheap part (cardisort classification if
not already CardiSorted, plus DICOM volume reading) instead of also paying
for LLM calls on stage.

Run once before a demo, or after adding/reclassifying subject data (a
subject whose candidate dirs haven't changed since its cache was written is
skipped -- see main()'s "already cached" count -- so re-running after adding
one new subject is cheap, not a full re-run).

Usage (from qcardia-app-dev), Ollama must be running:
    uv run src/precompute_classification_cache.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    from model.dicom_model import DICOMModel
    from utils.app_config import discover_subject_numbers, ollama_available, subject_folder_path

    logging.basicConfig(level=logging.WARNING)

    if not ollama_available():
        print("Ollama not reachable -- disambiguation needs it. Start Ollama and rerun.")
        return

    numbers = discover_subject_numbers()
    if not numbers:
        print("No subjects found under SUBJECT_BASE_DIR.")
        return

    print(f"Warming classification cache for {len(numbers)} subjects...")
    model = DICOMModel()

    for n in numbers:
        folder = subject_folder_path(n)
        subject_id = f"QLGE{n:02d}"
        t0 = time.time()
        try:
            classified = model.classify_patient_folder(folder)
        except Exception as e:
            print(f"  {subject_id}: FAILED ({e})")
            continue
        elapsed = time.time() - t0
        # A cache hit for every key resolves in well under a second; a cold
        # subject with any duplicate group takes several seconds per LLM
        # call (plus cardisort inference time if not already CardiSorted) --
        # this threshold is just for a readable progress log, not a
        # correctness check (the cache read/write logic is what's authoritative).
        status = "already cached" if elapsed < 1.0 else f"computed in {elapsed:.1f}s"
        print(f"  {subject_id}: {len(classified)} series keys, {status}")

    print("Done.")


if __name__ == "__main__":
    main()
