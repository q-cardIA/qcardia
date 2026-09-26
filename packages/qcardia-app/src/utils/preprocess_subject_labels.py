#!/usr/bin/env python
"""Runs cardisort classification + disambiguation for every subject folder in the data folder and writes .cardisort.json.
Safe to re-run: skips subjects that already have a sidecar, unless --force.

Usage:
    uv run src/utils/preprocess_subject_labels.py                      # every subject in the data folder chosen on the welcome screen
    uv run src/utils/preprocess_subject_labels.py sub-01 sub-02        # specific subject folders
    uv run src/utils/preprocess_subject_labels.py --data-dir D:/study  # another data folder
    uv run src/utils/preprocess_subject_labels.py --force              # regenerate existing sidecars too
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.dicom_model import DICOMModel
from model.subject_labels import save_labels, sidecar_path
from utils.app_config import data_dir


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subjects", nargs="*", help="Subject folder names (default: all folders in the data folder)")
    p.add_argument("--data-dir", type=Path, default=data_dir(), help="Folder with one subfolder per subject (default: the data folder chosen on the welcome screen, now %(default)s)")
    p.add_argument("--force", action="store_true", help="Regenerate sidecars that already exist")
    args = p.parse_args()

    names = args.subjects or sorted(e.name for e in args.data_dir.iterdir() if e.is_dir() and e.name[0] != ".")
    model = DICOMModel()

    for name in names:
        folder = args.data_dir / name
        if not folder.is_dir():
            print(f"{name}: folder not found, skipping")
            continue
        if sidecar_path(folder).exists() and not args.force:
            print(f"{name}: already cached, skipping")
            continue
        try:
            labels = model.label_patient_folder(folder)
        except Exception as e:
            print(f"{name}: FAILED ({type(e).__name__}: {e})")
            continue
        if not labels.series:
            print(f"{name}: no DICOM series found, skipping")
            continue
        save_labels(folder, labels)
        print(f"{name}: wrote {sidecar_path(folder).name} "
              f"({len(labels.series)} series, {len(labels.primary_by_key)} disambiguated)")


if __name__ == "__main__":
    main()
