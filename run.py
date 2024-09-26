from pathlib import Path

import numpy as np
import torch
import yaml
from natsort import natsorted
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label

from src.qcardia.series import LGESeries

WANDB_RUN_PATH_CENTER = Path.cwd() / "wandb" / "lge-center"
WANDB_RUN_PATH_SEG = Path.cwd() / "wandb" / "lge-seg"

PATH_TO_DATASET = Path.cwd() / "data"

patient_list = natsorted([f for f in PATH_TO_DATASET.iterdir() if f.is_dir()])

patient = patient_list[0]
lge_dir = Path(list(patient.glob("*[dD][bB]*[sS][cC][aA][rR]*[sS][aA]"))[0])
# lge_dir = Path(list(patient.glob("*[lL][gG][eE]*"))[0])
lge_seq = LGESeries(lge_dir)


lge_center = lge_seq.predict_segmentation(WANDB_RUN_PATH_CENTER)
lge_segmentation = lge_seq.predict_segmentation(WANDB_RUN_PATH_SEG, lge_center[5])


# # number of slices is first?

# # cine_rvs = cine_seq.get_rv_insertion_points()

# # lge_dir = Path(list(patient.glob("*[sS][cC][aA][rR]*"))[0])
# # lge_seq = BaseSequence(lge_dir)

# # test_myo1 = np.load(PATH_TO_DATASET / "QLGE71_DBLGE_SAX_3_myo.npy")
# # test_scar1 = np.load(PATH_TO_DATASET / "QLGE71_DBLGE_SAX_3_scar.npy")
# # test_myo2 = np.load(PATH_TO_DATASET / "QLGE71_DBLGE_SAX_6_myo.npy")
# # test_scar2 = np.load(PATH_TO_DATASET / "QLGE71_DBLGE_SAX_6_scar.npy")
# # test_myo3 = np.load(PATH_TO_DATASET / "QLGE71_DBLGE_SAX_9_myo.npy")
# # test_scar3 = np.load(PATH_TO_DATASET / "QLGE71_DBLGE_SAX_9_scar.npy")


# # quick check
from matplotlib import pyplot as plt
from skimage import measure

# # contour_epi = measure.find_contours(test_myo1)[0]
# # contour_endo = measure.find_contours(test_myo1)[1]
# # # contour_scar = measure.find_contours(test_scar2)[0]
# plt.imshow(cine_segmentation[5], cmap="gray")
# plt.show()

pred_R = (cine_segmentation[5] == 1) + (cine_segmentation[5] == 2)
pred_R = binary_fill_holes(pred_R)
distance_R = distance_transform_edt(pred_R)
distance_R[distance_R != 1] = 0

pred_L = (cine_segmentation[5] == 2) + (cine_segmentation[5] == 3)
pred_L = binary_fill_holes(pred_L)
distance_L = distance_transform_edt(pred_L)
distance_L[distance_L != 1] = 0

# Find the intersection of the two lines to find the center of the label
list_L = np.where(distance_L == 1)
list_R = np.where(distance_R == 1)

b_L = 1
b_R = -1
a_L = np.mean(list_L[0]) - b_L * np.mean(list_L[1])
a_R = np.mean(list_R[0]) - b_R * np.mean(list_R[1])


inter_R = int((a_L - a_R) / (b_R - b_L))
inter_C = int(a_L + b_L * inter_R)

print(inter_R, inter_C)

plt.imshow(
    lge_seq.slice_data["slice06"]["pixel_array"][0] / 300 + cine_segmentation[5],
    cmap="gray",
)
plt.plot(inter_R, inter_C, "bo")

plt.show()

# contour_lv = measure.find_contours(cine_segmentation[5] == 1)[0]
# contour_myo = measure.find_contours(cine_segmentation[5] == 2)[0]
# contour_rv = measure.find_contours(cine_segmentation[5] == 3)[0]


# # import imgaug.augmenters as iaa


# # def get_offset(seg, RV):
# #     RVinsertionx = RV[0]
# #     RVinsertiony = RV[1]
# #     [xs, ys] = np.where(seg > 0)
# #     centx = np.mean(xs)
# #     centy = np.mean(ys)

# #     spoke1m = (centy - RVinsertiony) / (centx - RVinsertionx)

# #     return np.arctan(spoke1m)


# # def rotate_to_rv(image, rv, interp_order=3):
# #     angle = np.pi - get_offset(image > 0, rv)
# #     rotate_im = iaa.Affine(rotate=np.rad2deg(angle), order=interp_order)
# #     rotated_image = rotate_im.augment_image(image)

# #     return rotated_image


# # plt.subplot(1, 2, 1)
# # plt.imshow(lge_seq.slice_data["slice06"]["pixel_array"][1] / 300, cmap="gray")
# # plt.subplot(1, 2, 2)
# # plt.imshow(
# #     rotate_to_rv(lge_seq.slice_data["slice06"]["pixel_array"][1] / 300, [210, 130]),
# #     cmap="gray",
# # )
# # plt.show()

# # print(lge_seq.slice_data["slice06"]["pixel_array"][1].shape)

# # Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(lge_seq.slice_data["slice06"]["psir_array"] / 300, cmap="gray")
# # ax.imshow(lge_seq.slice_data["slice06"]["pixel_array"][1] / 300, cmap="gray")

ax.plot(contour_rv[:, 1], contour_rv[:, 0], linewidth=2.5, color="tab:blue")
ax.plot(contour_lv[:, 1], contour_lv[:, 0], linewidth=2.5, color="tab:green")
ax.plot(contour_myo[:, 1], contour_myo[:, 0], linewidth=2.5, color="tab:orange")
# # ax.plot(contour_epi[:, 1], contour_epi[:, 0], linewidth=2.5, color="tab:orange")
# # ax.plot(contour_endo[:, 1], contour_endo[:, 0], linewidth=2.5, color="tab:orange")
# # ax.plot(contour_scar[:, 1], contour_scar[:, 0], linewidth=2.5, color="tab:red")
plt.axis("off")
plt.show()
