from pathlib import Path

import numpy as np
import torch
import yaml
from natsort import natsorted

from src.qcardia.data.sequence import BaseSequence

WANDB_RUN_PATH = Path.cwd() / "wandb"
PATH_TO_DATASET = Path.cwd() / "data"

patient_list = natsorted([f for f in PATH_TO_DATASET.iterdir() if f.is_dir()])

patient = patient_list[0]
cine_dir = Path(list(patient.glob("*[sS][aA]*[sS][tT][aA][cC]*"))[0])
cine_seq = BaseSequence(cine_dir)
cine_segmentation = cine_seq.run_model(WANDB_RUN_PATH)

# quick check
from matplotlib import pyplot as plt

plt.imshow(
    cine_seq.slice_data["slice09"]["pixel_array"][10] / 300 + cine_segmentation[9, 10],
)
plt.show()
