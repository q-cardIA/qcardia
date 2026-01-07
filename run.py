from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.animation import FuncAnimation, PillowWriter
from monai.networks.blocks import Warp
from natsort import natsorted
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import find_contours
from skimage.transform import warp

import utils
from qcardia.series import CineSeries

MOTION_WANDB_RUN_PATH = Path.cwd() / "wandb" / "motion-model"
WANDB_RUN_PATH = Path.cwd() / "wandb" / "cine-seg"
PATH_TO_DATASET = Path.cwd() / "data"

patient_list = natsorted([f for f in PATH_TO_DATASET.iterdir() if f.is_dir()])

warp_layer = Warp()

for patient in patient_list[:1]:
    try:
        print(patient)
        cine_dir = Path(list(patient.glob("*[sS][aA]*[sS][tT][aA][cC]*"))[0])
        # cine_dir = Path(list(patient.glob("*[cC][iI][nN][eE]*"))[0])
        cine_seq = CineSeries(cine_dir, batch_size=200)
        cine_segmentation = cine_seq.predict_segmentation(WANDB_RUN_PATH)
        cine_seq.save_predictions(Path(f"{cine_dir}_segmentation"))
        lv_vol_curve = cine_seq.compute_volume_curve()
        ef = cine_seq.compute_ejection_fraction(lv_vol_curve)
    except:
        pass
        # motion = cine_seq.motion_track(MOTION_WANDB_RUN_PATH)
        # motion = motion.reshape(
        #     3,
        #     cine_seq.number_of_temporal_positions - 1,
        #     2,
        #     *motion.shape[-2:],
        # )

        # motion = motion[1, ...]
        # input_images = np.asarray(cine_seq.slice_data["slice06"]["pixel_array"])
        # myo = cine_segmentation[cine_seq.mid_slice_num - 1, :] == 2

        # the_pts = utils.get_polar_points(
        #     myo[0, ...].astype(float),
        #     cine_seq.get_lv_center_points()[1][0],
        #     cine_seq.get_rv_insertion_points(),
        #     num_spokes=10,
        # )
from scipy.ndimage import binary_fill_holes

from src.qcardia.series import LGESeries

WANDB_RUN_PATH_CENTER = Path.cwd() / "wandb" / "lge-center"
WANDB_RUN_PATH_SEG = Path.cwd() / "wandb" / "lge-seg"

PATH_TO_DATASET = Path.cwd() / "data"
# PATH_TO_DATASET = Path.cwd() / "vida-data-combined"

patient_list = natsorted([f for f in PATH_TO_DATASET.iterdir() if f.is_dir()])

for patient in patient_list[:1]:
    print(patient)
    lge_dir = Path(list(patient.glob("*[dD][bB]*[sS][cC][aA][rR]*[sS][aA]"))[0])
    # lge_dir = Path(list(patient.glob("*[lL][gG][eE]*"))[0])
    lge_seq = LGESeries(lge_dir)

    lge_center = lge_seq.predict_segmentation(WANDB_RUN_PATH_CENTER)
    lge_segmentation = lge_seq.predict_segmentation(WANDB_RUN_PATH_SEG, lge_center[5])

    lge_seq.save_predictions(Path(f"{lge_dir}_segmentation"))

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # colors = utils.get_colors(10)

        # def animate(i):

        #     if i == 0:
        #         im = ax.imshow(input_images[0, ...], cmap="gray")
        #         for idx, pt in enumerate(the_pts):
        #             ax.plot(
        #                 pt[0][0], pt[0][1], "o", color=matplotlib.colors.to_hex(colors[idx])
        #             )
        #         ax.set_axis_off()
        #         return [im]

        #     else:
        #         the_motion = np.transpose(motion[i, ...], (1, 2, 0))
        #         nx, ny = the_motion.shape[1], the_motion.shape[0]
        #         X = np.arange(nx)
        #         Y = np.arange(ny)
        #         the_motion_x = RegularGridInterpolator((X, Y), the_motion[..., 0])
        #         the_motion_y = RegularGridInterpolator((X, Y), the_motion[..., 1])

        #         ax.clear()
        #         im = ax.imshow(input_images[i + 1, ...], cmap="gray")
        #         for idx, pt in enumerate(the_pts):

        #             deformed_pt_y = pt[0][0] - the_motion_y((pt[0][1], pt[0][0]))
        #             deformed_pt_x = pt[0][1] - the_motion_x((pt[0][1], pt[0][0]))
        #             ax.plot(
        #                 deformed_pt_y,
        #                 deformed_pt_x,
        #                 "o",
        #                 color=matplotlib.colors.to_hex(colors[idx]),
        #             )

        #         ax.set_axis_off()
        #         return [im]

        #     # fig.canvas.draw()
        #     # plt.pause(0.2)

        # fig.tight_layout()
        # anim = FuncAnimation(
        #     fig,
        #     animate,
        #     frames=input_images.shape[0] - 1,
        #     interval=50,
        #     blit=True,
        # )
        # # Save as GIF
        # writer = PillowWriter(fps=30)
        # anim.save(f"{patient.name}.gif", writer=writer)

        # # Close the figure to free memory
        # plt.close()
# if __name__ == "__main__":
see_num = 7
# # number of slices is first?

# # cine_rvs = cine_seq.get_rv_insertion_points()

# # lge_dir = Path(list(patient.glob("*[sS][cC][aA][rR]*"))[0])
# # lge_seq = BaseSequence(lge_dir)
# # quick check
from matplotlib import pyplot as plt
from skimage import measure

# # contour_epi = measure.find_contours(test_myo1)[0]
# # contour_endo = measure.find_contours(test_myo1)[1]
# # # contour_scar = measure.find_contours(test_scar2)[0]
# plt.imshow(cine_segmentation[5], cmap="gray")
# plt.show()

myo = (lge_segmentation[see_num] == 2) + (lge_segmentation[see_num] == 1)


lv = binary_fill_holes(myo) * 1 - myo

from skimage.measure import label


def getLargestCC(segmentation):
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC


lv = getLargestCC(lv)

contour_lv = measure.find_contours(lv == 1)[0]
contour_myo = measure.find_contours(myo == 1)[0]
contour_scar = measure.find_contours(lge_segmentation[see_num] == 2)[0]
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
ax.imshow(lge_seq.slice_data[f"slice0{see_num+1}"]["psir_array"] / 300, cmap="gray")
# # ax.imshow(lge_seq.slice_data["slice06"]["pixel_array"][1] / 300, cmap="gray")

ax.plot(contour_lv[:, 1], contour_lv[:, 0], linewidth=2.5, color="tab:blue")
ax.plot(contour_myo[:, 1], contour_myo[:, 0], linewidth=2.5, color="tab:green")
ax.plot(contour_scar[:, 1], contour_scar[:, 0], linewidth=2.5, color="tab:orange")
# # ax.plot(contour_epi[:, 1], contour_epi[:, 0], linewidth=2.5, color="tab:orange")
# # ax.plot(contour_endo[:, 1], contour_endo[:, 0], linewidth=2.5, color="tab:orange")
# # ax.plot(contour_scar[:, 1], contour_scar[:, 0], linewidth=2.5, color="tab:red")
plt.axis("off")
plt.show()
