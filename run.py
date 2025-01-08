from pathlib import Path

import numpy as np
import torch
import yaml
from monai.networks.blocks import Warp
from natsort import natsorted

from qcardia.series import CineSeries

MOTION_WANDB_RUN_PATH = Path.cwd() / "wandb" / "motion-model"
WANDB_RUN_PATH = Path.cwd() / "wandb" / "cine-seg"
PATH_TO_DATASET = Path.cwd() / "data"

patient_list = natsorted([f for f in PATH_TO_DATASET.iterdir() if f.is_dir()])

warp_layer = Warp()
# warp_layer = Warp(mode="nearest")

from skimage.measure import find_contours


def get_polar_points(myo, lv, rv, num_spokes=60):

    cy = lv[0]
    cx = lv[1]

    # Get image dimensions
    height, width = myo.shape

    # Calculate angles for spokes (in radians)
    angles = np.linspace(0, 2 * np.pi, num_spokes, endpoint=False)
    # Maximum radius to check (diagonal of image)
    max_radius = np.sqrt(width**2 + height**2)

    # Store intersection points for each spoke
    all_intersections = []
    for angle in angles:
        # Calculate unit vector for this angle
        dx = np.cos(angle)
        dy = np.sin(angle)

        # Generate points along the spoke
        num_points = int(max_radius * 4)
        radii = np.linspace(0, max_radius, num_points)
        spoke_points = np.array([(cx + r * dx, cy + r * dy) for r in radii])

        # Find intersections with the donut
        intersections = []
        for i in range(len(spoke_points) - 1):
            x1, y1 = spoke_points[i]
            x2, y2 = spoke_points[i + 1]

            # Check if we're crossing the boundary
            if (
                0 <= int(y1) < height
                and 0 <= int(x1) < width
                and 0 <= int(y2) < height
                and 0 <= int(x2) < width
            ):
                val1 = myo[int(y1), int(x1)]
                val2 = myo[int(y2), int(x2)]
                if val1 != val2:  # We found an intersection
                    # Use the midpoint as the intersection point
                    intersections.append([(x1 + x2) / 2, (y1 + y2) / 2])

        all_intersections.append(intersections)

    return all_intersections


for patient in patient_list[2:]:
    cine_dir = Path(list(patient.glob("*[sS][aA]*[sS][tT][aA][cC]*"))[0])
    cine_seq = CineSeries(cine_dir, batch_size=200)
    cine_segmentation = cine_seq.predict_segmentation(WANDB_RUN_PATH)
    cine_seq.save_predictions(Path(f"{cine_dir}_segmentation"))
    lv_vol_curve = cine_seq.compute_volume_curve()
    ef = cine_seq.compute_ejection_fraction(lv_vol_curve)

    motion = cine_seq.motion_track(MOTION_WANDB_RUN_PATH)

    # input_images = np.stack(
    #     (
    #         cine_seq.slice_data["slice04"]["pixel_array"][0],
    #         cine_seq.slice_data["slice06"]["pixel_array"][0],
    #         cine_seq.slice_data["slice10"]["pixel_array"][0],
    #     ),
    #     axis=0,
    # )

    input_images = np.asarray(cine_seq.slice_data["slice06"]["pixel_array"])

    motion = motion.reshape(
        3,
        cine_seq.number_of_temporal_positions - 1,
        2,
        *motion.shape[-2:],
    )

    motion = motion[1, ...]
    from skimage.transform import warp

    # myo = cine_segmentation[[4, 6, 10], 0] == 2
    myo = cine_segmentation[6, :] == 2
    # moved_myo1 = warp_layer(torch.Tensor(myo[:, None, ...]), torch.Tensor(motion))
    # moved_myo = warp(myo[1, ...], np.transpose(motion[1, ...], (1, 2, 0)), order=0)

    moved_images = warp_layer(
        torch.Tensor(input_images[1:, None, ...]), torch.Tensor(motion)
    )

    from scipy.interpolate import RegularGridInterpolator

    the_pts = get_polar_points(
        myo[0, ...].astype(float),
        cine_seq.get_lv_center_points()[1][0],
        cine_seq.get_rv_insertion_points(),
        num_spokes=10,
    )
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = plt.cm.jet(np.linspace(0, 1, len(the_pts)))

    im = ax.imshow(input_images[0, ...], cmap="gray")
    for idx, pt in enumerate(the_pts):
        ax.plot(pt[0][0], pt[0][1], "o", color=colors[idx])
    # for i in range(motion.shape[0]):
    ax.set_axis_off()

    def animate(i):
        the_motion = np.transpose(motion[i, ...], (1, 2, 0))
        nx, ny = the_motion.shape[1], the_motion.shape[0]
        X = np.arange(nx)
        Y = np.arange(ny)
        the_motion_x = RegularGridInterpolator((X, Y), the_motion[..., 0])
        the_motion_y = RegularGridInterpolator((X, Y), the_motion[..., 1])

        ax.clear()
        im = ax.imshow(input_images[i + 1, ...], cmap="gray")
        for idx, pt in enumerate(the_pts):

            deformed_pt_y = pt[0][0] - the_motion_y((pt[0][1], pt[0][0]))
            deformed_pt_x = pt[0][1] - the_motion_x((pt[0][1], pt[0][0]))
            ax.plot(deformed_pt_y, deformed_pt_x, "o", color=colors[idx])

        ax.set_axis_off()
        return [im]

        # fig.canvas.draw()
        # plt.pause(0.2)

    # plt.close()

    anim = FuncAnimation(
        fig,
        animate,
        frames=input_images.shape[0] - 1,
        interval=50,
        blit=True,
    )
    # Save as GIF
    writer = PillowWriter(fps=30)
    anim.save("test.gif", writer=writer)

    # Close the figure to free memory
    plt.close()
    # plt.imshow(moved_myo + moved_myo1[1, 0, ...].numpy())
    # plt.show()

    # the_motion = np.transpose(motion[i, ...], (1, 2, 0))
    # nx, ny = the_motion.shape[1], the_motion.shape[0]
    # # X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    # X = np.arange(nx)
    # Y = np.arange(ny)
    # the_motion_x = RegularGridInterpolator(
    #     (X, Y), the_motion[..., 0]
    # )  # , method="linear", bounds_error=False, fill_value=None
    # # )
    # the_motion_y = RegularGridInterpolator(
    #     (X, Y), the_motion[..., 1]
    # )  # , method="linear", bounds_error=False, fill_value=None
    # # )

    # # #### plot actual images
    # # row_coords, col_coords = np.meshgrid(
    # #     np.arange(ny), np.arange(nx), indexing="ij"
    # )
    # # moved_myo = warp(
    # #     myo[5, ...].astype(float),
    # #     np.array([row_coords + motion[1, 0], col_coords + motion[1, 1]]),
    # #     mode="constant",
    # #     preserve_range=False,
    # #     order=0,
    # # )

    # plt.subplot(221)
    # plt.imshow(myo[1, ...])
    # plt.subplot(222)
    # plt.imshow(cine_segmentation[6, cine_seq.es_time, ...])
    # plt.subplot(223)
    # # plt.imshow(moved_myo[1, 0, ...])
    # plt.imshow(moved_myo)
    # plt.subplot(224)
    # # plt.imshow(myo[1, ...] + moved_myo[1, 0, ...].numpy())
    # plt.imshow(myo[1, ...].astype(float) + moved_myo.astype(float))
    # plt.show()

    # n_step = 10
    # grid_x, grid_y = np.meshgrid(
    #     np.arange(0, motion.shape[-2], n_step), np.arange(0, motion.shape[-1], n_step)
    # )

    # plt.quiver(
    #     grid_x,
    #     grid_y,
    #     motion[1, 0, ::n_step, ::n_step],
    #     motion[1, 1, ::n_step, ::n_step],
    #     units="dots",
    #     angles="xy",
    #     scale_units="xy",
    #     lw=3,
    # )
    # plt.show()

    # from skimage import measure

    # endo = measure.find_contours(myo[1, ...])[1]

    # endo_pt = endo[100]

    # deformed_endo_pt_y = endo_pt[1] - the_motion_y((endo_pt[0], endo_pt[1]))
    # deformed_endo_pt_x = endo_pt[0] - the_motion_x((endo_pt[0], endo_pt[1]))

    # # plt.imshow(myo[1, ...])
    # # plt.plot(endo_pt[1], endo_pt[0], "ro")
    # # plt.plot(deformed_endo_pt_y, deformed_endo_pt_x, "bo")
    # # plt.show()

    # plt.subplot(121)
    # plt.imshow(input_images[1, ...], cmap="gray")
    # plt.plot(endo_pt[1], endo_pt[0], "ro")
    # plt.subplot(122)
    # plt.imshow(
    #     cine_seq.slice_data["slice06"]["pixel_array"][cine_seq.es_time], cmap="gray"
    # )
    # plt.plot(deformed_endo_pt_y, deformed_endo_pt_x, "bo")
    # plt.show()

    # # plt.imshow(myo[1, ...].astype(float) + moved_myo.astype(float))
    # # plt.plot(endo_pt[1], endo_pt[0], "ro")
    # # plt.plot(deformed_endo_pt_y, deformed_endo_pt_x, "bo")
    # # plt.show()

    # plt.subplot(221)
    # plt.imshow(input_images[1, ...])
    # plt.subplot(222)
    # plt.imshow(cine_seq.slice_data["slice06"]["pixel_array"][cine_seq.es_time])
    # plt.subplot(223)
    # plt.imshow(moved_images[1, 0, ...])
    # plt.subplot(224)
    # plt.imshow(input_images[1, ...] + moved_images[1, 0, ...].numpy())
    # plt.show()

    # plt.plot(lv_vol_curve)
    # plt.show()

    # print(patient, ef)

# number of slices is first?

# cine_rvs = cine_seq.get_rv_insertion_points()

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
contour_lv = measure.find_contours(cine_segmentation[5, 1] == 1)[0]
contour_myo = measure.find_contours(cine_segmentation[5, 1] == 2)[0]
contour_rv = measure.find_contours(cine_segmentation[5, 1] == 3)[0]

lv = cine_segmentation == 1
myo = cine_segmentation == 2
rv = cine_segmentation == 3

lv_pix_vol = np.sum(lv, axis=(0, 2, 3))


plt.plot(lv_pix_vol)
plt.show()


def ejection_fraction(volume_ed, volume_es):
    stroke_volume = volume_ed - volume_es
    ejection_frac = (stroke_volume / volume_ed) * 100
    return ejection_frac


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

ax.plot(contour_rv[:, 1], contour_rv[:, 0], linewidth=2.5, color="tab:blue")
ax.plot(contour_lv[:, 1], contour_lv[:, 0], linewidth=2.5, color="tab:green")
ax.plot(contour_myo[:, 1], contour_myo[:, 0], linewidth=2.5, color="tab:orange")
# ax.plot(contour_epi[:, 1], contour_epi[:, 0], linewidth=2.5, color="tab:orange")
# ax.plot(contour_endo[:, 1], contour_endo[:, 0], linewidth=2.5, color="tab:orange")
# ax.plot(contour_scar[:, 1], contour_scar[:, 0], linewidth=2.5, color="tab:red")
plt.axis("off")
plt.show()
