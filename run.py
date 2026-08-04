from collections import defaultdict
from pathlib import Path

from natsort import natsorted

from qcardia.cardisort import (
    classify_sequence_dir,
    get_sequence_dirs,
    load_cardisort_model,
    load_series_datasets,
)
from qcardia.disambiguate import disambiguate_duplicates, summarize_sequence_dir
from qcardia.series import LGESeries

CARDISORT_WANDB_RUN_PATH = Path.cwd() / "wandb" / "cardisort"
LGE_SEG_WANDB_RUN_PATH = Path.cwd() / "wandb" / "lge-seg"
PATH_TO_DATASET = Path.cwd() / "data copy"

patient_list = natsorted([f for f in PATH_TO_DATASET.iterdir() if f.is_dir()])

cardisort_model, cardisort_config = load_cardisort_model(CARDISORT_WANDB_RUN_PATH)

for patient in patient_list[:]:
    sequence_dirs = get_sequence_dirs(patient)

    sequence_classifications = {}
    for sequence_dir in sequence_dirs:
        prediction = classify_sequence_dir(
            sequence_dir,
            cardisort_model,
            cardisort_config["model"]["nr_input_channels"],
            tuple(cardisort_config["data"]["target_pixdim"]),
            tuple(cardisort_config["data"]["target_size"]),
            cardisort_config["data"]["image_grid_sample_mode"],
            verbose=False,
        )
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
        for sequence_dir, (sequence_name, _) in primary_dirs.items()
        if sequence_name == "CINE"
    ]

    db_lge_sax_dirs = [
        sequence_dir
        for sequence_dir, (sequence_name, plane_name) in primary_dirs.items()
        if sequence_name == "DBLGE" and plane_name == "SAX"
    ]
    for lge_dir in db_lge_sax_dirs:
        lge_seq = LGESeries(lge_dir)
        lge_segmentation = lge_seq.predict_segmentation(
            LGE_SEG_WANDB_RUN_PATH
        )
        lge_seq.save_predictions(Path(f"{lge_dir}_segmentation"))
        print(f"  {lge_dir} -> {lge_segmentation.shape}")


