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

# quick check, look here to improve- https://github.com/moralesq/DeepStrain/blob/main/utils/visualizer.py
from matplotlib import pyplot as plt
from skimage import measure

contour_lv = measure.find_contours(cine_segmentation[8, 10] == 1)[0]
contour_myo = measure.find_contours(cine_segmentation[8, 10] == 2)[0]
contour_rv = measure.find_contours(cine_segmentation[8, 10] == 3)[0]

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(cine_seq.slice_data["slice09"]["pixel_array"][10] / 300, cmap="gray")
ax.plot(contour_rv[:, 1], contour_rv[:, 0], linewidth=2.5, color="tab:blue")
ax.plot(contour_lv[:, 1], contour_lv[:, 0], linewidth=2.5, color="tab:green")
ax.plot(contour_myo[:, 1], contour_myo[:, 0], linewidth=2.5, color="tab:orange")
plt.axis("off")
plt.show()
