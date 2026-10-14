"""Visual check that the cine segmentation fits the perfusion images without registration.

For every subject with a cached CINE_SAX segmentation and a perfusion run, the cine labels are mapped onto the
perfusion slices (model/perfusion_regions.py) and drawn on the frame of peak LV enhancement, next to the region
curves with the navigation events. Writes one PNG per subject, overlay_check.csv (fill in the `fit` column:
good / shifted / bad) and index.html to page through the images, all in <data dir>/perfusion_overlay_check/.

Run from the repo root: .venv/Scripts/python.exe src/tools/perfusion_overlay_check.py [QLGE12_ ...]
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from skimage import measure  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.dicom_model import DICOMModel  # noqa: E402
from model.perfusion_regions import (  # noqa: E402
    matching_frames, perfusion_events, project_labels, region_curves)
from utils.app_config import data_dir, patient_label  # noqa: E402

RUNS = ("PERF_STRESS_SAX", "PERF_REST_SAX", "PERF_SAX")
EVENT_STYLE = {"lv_arrival": ("r", ":"), "lv_peak": ("r", "-"), "myo_arrival": ("g", ":"), "myo_peak": ("g", "-")}


def check_subject(model: DICOMModel, folder: Path, out: Path) -> list[dict]:
    model.clear_patient()
    model.subject_folder = folder
    classified = model.classify_patient_folder(folder)
    runs = [k for k in RUNS if k in classified]
    if "CINE_SAX" not in classified or not runs:
        return []
    cine = model.load_series_for_key(classified["CINE_SAX"], "CINE_SAX")
    if cine.get("segmentation") is None or cine["geometry"] is None:
        return [{"subject": folder.name, "run": "", "note": "no cached CINE_SAX segmentation"}]
    cine_trigger = model._trigger_times("CINE_SAX")[0]

    rows = []
    fig, axes = plt.subplots(len(runs), 4, figsize=(18, 4.6 * len(runs)), squeeze=False)
    for r, key in enumerate(runs):
        perf = model.load_series_for_key(classified[key], key)
        perf_trigger = [t[0] for t in model._trigger_times(key)]
        frames = matching_frames(cine_trigger, perf_trigger)
        labels = project_labels(cine["segmentation"], cine["geometry"], frames, perf["geometry"])
        curves = region_curves(perf["volume"], labels)
        events = perfusion_events(curves)

        ax = axes[r, 0]
        for z in range(labels.shape[0]):
            ax.plot(curves["lv"][:, z], color="r", alpha=min(1.0, 0.4 + 0.2 * z), lw=1)
            ax.plot(curves["myo"][:, z], color="g", alpha=min(1.0, 0.4 + 0.2 * z), lw=1)
        for name, frame in events.items():
            color, style = EVENT_STYLE[name]
            ax.axvline(frame, color=color, ls=style, lw=1)
        ax.set_title(f"{key}: LV (red), myocardium (green)\n" + ", ".join(f"{k} {v + 1}" for k, v in events.items()),
                     fontsize=9)
        peak = events.get("lv_peak", 0)
        for z in range(min(3, labels.shape[0])):
            a = axes[r, z + 1]
            a.imshow(perf["volume"][peak, z], cmap="gray")
            for label, color in ((2, "lime"), (1, "red")):
                for contour in measure.find_contours((labels[z] == label).astype(float), 0.5):
                    a.plot(contour[:, 1], contour[:, 0], color, lw=0.8)
            a.set_title(f"slice {z + 1}, frame {peak + 1} (LV peak), cine frame {frames[z] + 1}", fontsize=9)
            a.axis("off")
        rows.append({"subject": folder.name, "run": key, "frames": perf["volume"].shape[0],
                     "cine_frames": " ".join(str(f + 1) for f in frames),
                     "lv_px": " ".join(str(int(n)) for n in (labels == 1).sum(axis=(1, 2))),
                     **{k: v + 1 for k, v in events.items()}})
    fig.suptitle(f"{folder.name} {patient_label(folder.name)}", fontsize=11)
    plt.tight_layout()
    fig.savefig(out / f"{folder.name}.png", dpi=60)
    plt.close(fig)
    return rows


def main(subjects: list[str]):
    logging.basicConfig(level=logging.WARNING)
    root = data_dir()
    out = root / "perfusion_overlay_check"
    out.mkdir(exist_ok=True)
    folders = [root / s for s in subjects] or sorted(p for p in root.glob("QLGE*_") if p.is_dir())
    rows, model = [], DICOMModel()
    for folder in folders:
        try:
            found = check_subject(model, folder, out)
        except Exception as e:  # one damaged subject must not stop the list
            found = [{"subject": folder.name, "run": "", "note": f"error: {e!r}"[:200]}]
        rows += found
        print(folder.name, "; ".join(f"{r.get('run')} {r.get('lv_peak', r.get('note', ''))}" for r in found), flush=True)

    fields = ["subject", "run", "frames", "cine_frames", "lv_px", "lv_arrival", "lv_peak", "myo_arrival", "myo_peak",
              "note", "fit"]
    with open(out / "overlay_check.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    images = sorted(p.name for p in out.glob("QLGE*_.png"))
    (out / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Perfusion overlay check</title>"
        "<style>body{font-family:sans-serif;background:#111;color:#ddd}img{max-width:100%}h2{margin-top:2em}</style>"
        "<p>Cine segmentation (red LV, green myocardium) on the perfusion frame of peak LV enhancement. "
        "Mark each subject in overlay_check.csv (fit: good / shifted / bad).</p>"
        + "".join(f"<h2 id='{n[:-4]}'>{n[:-4]}</h2><img src='{n}' loading='lazy'>" for n in images),
        encoding="utf-8")
    print(f"{len(images)} images in {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
