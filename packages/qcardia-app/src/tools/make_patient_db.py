"""Writes a patient list: a made-up name and hospital ID for QLGE subject folders, so the app and its
assistant work with patients as in a hospital ("load Adam James") instead of numbered subjects. Open it
on the welcome screen (Patient list); a data folder's own patients.csv also names its subfolders.

The names and IDs come from a fixed seed over all of the data folder's QLGE folders, so a patient has the
same name in every list. The paths are relative to the list's folder: zip the list with the subject
folders and it still works. A study session lists only its own patients:

    uv run src/tools/make_patient_db.py                      # data folder/patients.csv, every subject
    uv run src/tools/make_patient_db.py --out ../session1.csv --subjects QLGE01_ QLGE05_ QLGE12_
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.app_config import PATIENTS_CSV_NAME, data_dir  # noqa: E402

FIRST_NAMES = [
    "Adam", "Alice", "Amelia", "Arthur", "Benjamin", "Charlotte", "Daniel", "Edward", "Eleanor", "Emily",
    "Ethan", "Florence", "George", "Grace", "Hannah", "Harry", "Henry", "Isabel", "Jack", "James", "Jessica",
    "Joseph", "Katherine", "Laura", "Leo", "Lucy", "Margaret", "Matthew", "Michael", "Olivia", "Oscar",
    "Peter", "Rachel", "Rebecca", "Richard", "Robert", "Samuel", "Sarah", "Sophie", "Thomas", "Victoria",
    "William",
]
LAST_NAMES = [
    "James", "Smith", "Jones", "Taylor", "Brown", "Williams", "Wilson", "Johnson", "Davies", "Robinson",
    "Wright", "Thompson", "Evans", "Walker", "White", "Roberts", "Green", "Hall", "Wood", "Jackson", "Clarke",
    "Harris", "Lewis", "Cooper", "King", "Baker", "Turner", "Hill", "Morris", "Ward", "Moore", "Parker",
    "Bennett", "Carter", "Hughes", "Edwards", "Collins", "Stewart", "Morgan", "Bell", "Murphy", "Fisher",
]
ID_LETTERS = "".join(c for c in string.ascii_uppercase if c not in "IO")  # never read as 1 or 0


def hospital_id(rng: random.Random) -> str:
    """Ten characters: two digits, two letters, then a digit and a letter in turn, e.g. 02HD8H7Y9K."""
    return "".join(rng.choice(string.digits if c == "9" else ID_LETTERS) for c in "99AA9A9A9A")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", type=Path, default=data_dir(), help="The data folder (default: %(default)s)")
    p.add_argument("--out", type=Path, help=f"The list to write (default: {PATIENTS_CSV_NAME} in the data folder)")
    p.add_argument("--subjects", nargs="+", help="Only these subject folders (default: every QLGE folder)")
    args = p.parse_args()

    subjects = sorted(d.name for d in args.folder.iterdir() if d.is_dir() and re.fullmatch(r"QLGE\d+_", d.name))
    if not subjects:  # e.g. the folder of an open session list: give the data folder
        sys.exit(f"No QLGE subject folders in {args.folder}")
    rng = random.Random(34)  # issue #34's seed: the same names every run
    names = rng.sample([f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES if f != l], len(subjects))
    ids = []
    while len(ids) < len(subjects):
        new = hospital_id(rng)
        ids += [new] if new not in ids else []
    unknown = set(args.subjects or ()) - set(subjects)
    if unknown:
        sys.exit(f"Not QLGE folders in {args.folder}: {', '.join(sorted(unknown))}")

    out = (args.out or args.folder / PATIENTS_CSV_NAME).resolve()
    rows = [(i, n, Path(os.path.relpath(args.folder.resolve() / s, out.parent)).as_posix())
            for i, n, s in zip(ids, names, subjects) if not args.subjects or s in args.subjects]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hospital_id", "name", "folder"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} patients to {out}")


if __name__ == "__main__":
    main()
