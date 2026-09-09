#!/usr/bin/env python
"""Runs cardisort classification + disambiguation for every subject under SUBJECT_BASE_DIR and writes .cardisort.json.
Safe to re-run: skips subjects that already have a sidecar, unless --force.

Usage:
    uv run src/utils/preprocess_subject_labels.py            # every discoverable subject
    uv run src/utils/preprocess_subject_labels.py 3 5 12      # specific subjects
    uv run src/utils/preprocess_subject_labels.py --force     # regenerate existing sidecars too
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.dicom_model import DICOMModel  # noqa: E402
from model.subject_labels import save_labels, sidecar_path  # noqa: E402
from utils.app_config import discover_subject_numbers, subject_folder_path  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subjects", nargs="*", type=int, help="Subject numbers to process (default: all discoverable)")
    p.add_argument("--force", action="store_true", help="Regenerate sidecars that already exist")
    args = p.parse_args()

    numbers = args.subjects or discover_subject_numbers()
    model = DICOMModel()

    for n in numbers:
        folder = subject_folder_path(n)
        if not folder.is_dir():
            print(f"QLGE{n:02d}_: folder not found, skipping")
            continue
        if sidecar_path(folder).exists() and not args.force:
            print(f"QLGE{n:02d}_: already cached, skipping")
            continue
        try:
            labels = model.label_patient_folder(folder)
        except Exception as e:
            print(f"QLGE{n:02d}_: FAILED ({type(e).__name__}: {e})")
            continue
        save_labels(folder, labels)
        print(f"QLGE{n:02d}_: wrote {sidecar_path(folder).name} "
              f"({len(labels.series)} series, {len(labels.primary_by_key)} disambiguated)")


if __name__ == "__main__":
    main()
