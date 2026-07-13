from pathlib import Path

from natsort import natsorted

from qcardia.cardisort import classify_sequence_dir, get_sequence_dirs, load_cardisort_model

CARDISORT_WANDB_RUN_PATH = Path.cwd() / "wandb" / "cardisort"
PATH_TO_DATASET = Path.cwd() / "data"

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

    cine_dirs = [
        sequence_dir
        for sequence_dir, (sequence_name, _) in sequence_classifications.items()
        if sequence_name == "CINE"
    ]

