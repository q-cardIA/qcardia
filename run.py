from pathlib import Path

import numpy as np
import torch
import yaml
from natsort import natsorted

from src.qcardia.series import CineSeries

WANDB_RUN_PATH = Path.cwd() / "wandb"
PATH_TO_DATASET = Path.cwd() / "data"

patient_list = natsorted([f for f in PATH_TO_DATASET.iterdir() if f.is_dir()])

patient = patient_list[0]
cine_dir = Path(list(patient.glob("*[sS][aA]*[sS][tT][aA][cC]*"))[0])
cine_seq = CineSeries(cine_dir)
cine_segmentation = cine_seq.predict_segmentation(WANDB_RUN_PATH)
# number of slices is first?

cine_rvs = cine_seq.get_rv_insertion_points()

# lge_dir = Path(list(patient.glob("*[sS][cC][aA][rR]*"))[0])
# lge_seq = BaseSequence(lge_dir)

# test_myo1 = np.load(PATH_TO_DATASET / "QLGE71_DBLGE_SAX_3_myo.npy")
# test_scar1 = np.load(PATH_TO_DATASET / "QLGE71_DBLGE_SAX_3_scar.npy")
# test_myo2 = np.load(PATH_TO_DATASET / "QLGE71_DBLGE_SAX_6_myo.npy")
# test_scar2 = np.load(PATH_TO_DATASET / "QLGE71_DBLGE_SAX_6_scar.npy")
# test_myo3 = np.load(PATH_TO_DATASET / "QLGE71_DBLGE_SAX_9_myo.npy")
# test_scar3 = np.load(PATH_TO_DATASET / "QLGE71_DBLGE_SAX_9_scar.npy")


# quick check
from matplotlib import pyplot as plt
from skimage import measure

# contour_epi = measure.find_contours(test_myo1)[0]
# contour_endo = measure.find_contours(test_myo1)[1]
# contour_scar = measure.find_contours(test_scar2)[0]
# plt.imshow(test_myo1, cmap="gray")
# plt.show()
contour_lv = measure.find_contours(cine_segmentation[8, 10] == 1)[0]
contour_myo = measure.find_contours(cine_segmentation[8, 10] == 2)[0]
contour_rv = measure.find_contours(cine_segmentation[8, 10] == 3)[0]

import imgaug.augmenters as iaa


def get_offset(seg, RV):
    RVinsertionx = RV[0]
    RVinsertiony = RV[1]
    [xs, ys] = np.where(seg > 0)
    centx = np.mean(xs)
    centy = np.mean(ys)

    spoke1m = (centy - RVinsertiony) / (centx - RVinsertionx)

    return np.arctan(spoke1m)


def rotate_to_rv(image, rv, interp_order=3):
    angle = np.pi - get_offset(image > 0, rv)
    rotate_im = iaa.Affine(rotate=np.rad2deg(angle), order=interp_order)
    rotated_image = rotate_im.augment_image(image)

    return rotated_image


# plt.subplot(1, 2, 1)
# plt.imshow(lge_seq.slice_data["slice06"]["pixel_array"][1] / 300, cmap="gray")
# plt.subplot(1, 2, 2)
# plt.imshow(
#     rotate_to_rv(lge_seq.slice_data["slice06"]["pixel_array"][1] / 300, [210, 130]),
#     cmap="gray",
# )
# plt.show()

# print(lge_seq.slice_data["slice06"]["pixel_array"][1].shape)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(cine_seq.slice_data["slice06"]["pixel_array"][1] / 300, cmap="gray")
# ax.imshow(lge_seq.slice_data["slice06"]["pixel_array"][1] / 300, cmap="gray")

# ax.plot(contour_rv[:, 1], contour_rv[:, 0], linewidth=2.5, color="tab:blue")
# ax.plot(contour_lv[:, 1], contour_lv[:, 0], linewidth=2.5, color="tab:green")
ax.plot(contour_myo[:, 1], contour_myo[:, 0], linewidth=2.5, color="tab:orange")
# ax.plot(contour_epi[:, 1], contour_epi[:, 0], linewidth=2.5, color="tab:orange")
# ax.plot(contour_endo[:, 1], contour_endo[:, 0], linewidth=2.5, color="tab:orange")
# ax.plot(contour_scar[:, 1], contour_scar[:, 0], linewidth=2.5, color="tab:red")
plt.axis("off")
plt.show()
