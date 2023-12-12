from pathlib import Path

import numpy as np
from natsort import natsorted

from src.qcardia.data.sequence import BaseSequence

PATH_TO_DATASET = Path.cwd() / "data"
patient_list = natsorted([f for f in PATH_TO_DATASET.iterdir() if f.is_dir()])
patient = patient_list[0]
cine_dir = Path(list(patient.glob("*[sS][aA]*[sS][tT][aA][cC]*"))[0])

cine_seq = BaseSequence(cine_dir)

print(cine_seq.number_of_slices)
