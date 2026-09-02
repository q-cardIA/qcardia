from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from natsort import natsorted

from qcardia.cardisort import (
    classify_sequence_dir,
    get_sequence_dirs,
    load_cardisort_model,
    load_series_datasets,
)
from qcardia.disambiguate import disambiguate_duplicates, summarize_sequence_dir
from qcardia.series import CineSeries, LGESeries

CARDISORT_WANDB_RUN_PATH = Path.cwd() / "wandb" / "cardisort"
LGE_SEG_WANDB_RUN_PATH = Path.cwd() / "wandb" / "lge-seg"
CINE_SEG_WANDB_RUN_PATH = Path.cwd() / "wandb" / "cine-seg"
PATH_TO_DATASET = Path.cwd() / "data" 

patient_list = natsorted([f for f in PATH_TO_DATASET.iterdir() if f.is_dir()])

cardisort_model, cardisort_config = load_cardisort_model(CARDISORT_WANDB_RUN_PATH)

for patient in patient_list[:]:
    sequence_dirs = get_sequence_dirs(patient)

    sequence_classifications = {}
    for sequence_dir in sequence_dirs:
        try:
            prediction = classify_sequence_dir(
                sequence_dir,
                cardisort_model,
                cardisort_config["model"]["nr_input_channels"],
                tuple(cardisort_config["data"]["target_pixdim"]),
                tuple(cardisort_config["data"]["target_size"]),
                cardisort_config["data"]["image_grid_sample_mode"],
                verbose=False,
            )
        except Exception as e:
            print(f"  ! failed to classify {sequence_dir}: {e!r}")
            continue
        if prediction is None:
            continue
        sequence_classifications[sequence_dir] = prediction

    # Multiple directories can land on the same (sequence, plane) class (e.g. a
    # low-res planning cine alongside the real diagnostic SAX stack). For each
    # such group, ask a local LLM to pick the primary series from DICOM
    # metadata; the rest are dropped from downstream processing.
    dirs_by_class = defaultdict(list)
    for sequence_dir, class_label in sequence_classifications.items():
        dirs_by_class[class_label].append(sequence_dir)

    primary_dirs = {}
    for class_label, dirs in dirs_by_class.items():
        if len(dirs) == 1:
            primary_dirs[dirs[0]] = class_label
            continue

        candidates = [
            summarize_sequence_dir(d, load_series_datasets(d)) for d in dirs
        ]
        result = disambiguate_duplicates(class_label, candidates)
        # for assessment in result["assessments"]:
        #     print(
        #         f"  {class_label}: {assessment['series']} -> "
        #         f"{assessment['role']} ({assessment['reasoning']})"
        #     )
        primary_dir = next(d for d in dirs if d.name == result["primary_series"])
        primary_dirs[primary_dir] = class_label

    cine_dirs = [
        sequence_dir
        for sequence_dir, (sequence_name, plane_name) in primary_dirs.items()
        if sequence_name == "CINE" and plane_name == "SAX"
    ]
    cine_dir = cine_dirs[0]
    cine_seq = CineSeries(cine_dir)
    cine_segmentation = cine_seq.predict_segmentation(CINE_SEG_WANDB_RUN_PATH)
    cine_seq.save_predictions(Path(f"{cine_dir}_segmentation"))

    # db_lge_sax_dirs = [
    #     sequence_dir
    #     for sequence_dir, (sequence_name, plane_name) in primary_dirs.items()
    #     if sequence_name == "DBLGE" and plane_name == "SAX"
    # ]

    # perf_sax_dirs = [
    #     sequence_dir
    #     for sequence_dir, (sequence_name, plane_name) in primary_dirs.items()
    #     if sequence_name == "PERF" and plane_name == "SAX"
    # ]
    # for perf_dir in perf_sax_dirs:
    #     print(f"{perf_dir}")


    # for lge_dir in db_lge_sax_dirs:
    #     try:
    #         lge_seq = LGESeries(lge_dir)
    #         lge_segmentation = lge_seq.predict_segmentation(
    #             LGE_SEG_WANDB_RUN_PATH
    #         )
    #         lge_seq.save_predictions(Path(f"{lge_dir}_segmentation"))
    #         print(f"{lge_dir}")
    #     except Exception as e:
    #         print(f"  ! failed to segment {lge_dir}: {e!r}")
    #         continue

    mid_slice = cine_seq.number_of_slices // 2 +2
    time_phase = 7  # 5th time phase, 0-indexed

    image = np.asarray(
        cine_seq.slice_data[f"slice{mid_slice + 1:02}"]["pixel_array"][time_phase]
    )
    mask = cine_segmentation[mid_slice, time_phase]

    cmap = ListedColormap(["none", "red", "green", "blue"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    ax.imshow(mask, cmap=cmap, alpha=0.45, vmin=0, vmax=3)
    ax.set_title(
        f"slice {mid_slice + 1}/{cine_seq.number_of_slices}, "
        f"time phase {time_phase + 1}/{cine_seq.number_of_temporal_positions}"
    )
    ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color=c, lw=6, label=l)
        for c, l in zip(["red", "green", "blue"], ["LV", "Myo", "RV"])
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.6)
    fig.savefig(Path(f"{cine_dir}_mid_slice_overlay.png"), dpi=150, bbox_inches="tight")
    plt.show()
