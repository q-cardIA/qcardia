from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.animation import FuncAnimation, PillowWriter
from monai.networks.blocks import Warp
from natsort import natsorted
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.measure import find_contours
from skimage.transform import warp

import utils
from qcardia.series import CineSeries

MOTION_WANDB_RUN_PATH = Path.cwd() / "wandb" / "motion-model"
WANDB_RUN_PATH = Path.cwd() / "wandb" / "cine-seg"
PATH_TO_DATASET = Path.cwd() / "data"

patient_list = natsorted([f for f in PATH_TO_DATASET.iterdir() if f.is_dir()])

warp_layer = Warp(mode="bilinear", padding_mode="border")
DDF_SMOOTHING_SIGMA = 1.0

def warp_landmarks(flow_hw, points_xy, warp_module):
    """Warp sparse landmarks by splatting each point with bilinear weights."""
    if flow_hw.shape[0] != 2:
        flow_hw = np.moveaxis(flow_hw, -1, 0)

    _, H, W = flow_hw.shape
    num_pts = points_xy.shape[0]
    pts_img = np.zeros((num_pts, 1, H, W), dtype=np.float32)
    

    for idx, (x_coord, y_coord) in enumerate(points_xy):
        x0 = np.clip(int(np.floor(x_coord)), 0, W - 1)
        y0 = np.clip(int(np.floor(y_coord)), 0, H - 1)
        x1 = min(x0 + 1, W - 1)
        y1 = min(y0 + 1, H - 1)

        wx = float(x_coord - x0)
        wy = float(y_coord - y0)

        pts_img[idx, 0, y0, x0] += (1.0 - wx) * (1.0 - wy)
        pts_img[idx, 0, y0, x1] += wx * (1.0 - wy)
        pts_img[idx, 0, y1, x0] += (1.0 - wx) * wy
        pts_img[idx, 0, y1, x1] += wx * wy

    # plt.imshow(np.sum(pts_img[:, 0, ...], axis=0), cmap="gray")
    # for x_val, y_val in points_xy:
    #     plt.plot(x_val, y_val, "x")
    # plt.sh
    
    pts_tensor = torch.from_numpy(pts_img)
    flow_tensor = torch.from_numpy(flow_hw[None]).float().repeat(num_pts, 1, 1, 1)

    warped = warp_module(
        pts_tensor,
        flow_tensor,
    ).numpy()

    y_indices, x_indices = np.indices((H, W), dtype=np.float32)
    coords = []
    for idx, channel in enumerate(warped):
        mask = channel[0]
        total = float(mask.sum())
        if total <= 1e-6:
            coords.append((float(points_xy[idx][0]), float(points_xy[idx][1])))
            continue
        x_center = float((mask * x_indices).sum() / total)
        y_center = float((mask * y_indices).sum() / total)
        coords.append((x_center, y_center))

    return np.array(coords, dtype=np.float32)


def warp_landmarks_direct(flow_hw, points_xy):
    """Warp sparse landmarks by directly sampling the DDF at each point.

    flow_hw: (2, H, W) backward DDF — flow_hw[0] is dy, flow_hw[1] is dx.
    points_xy: (N, 2) landmark coordinates in (x, y) / fixed-frame space.
    Returns: (N, 2) warped coordinates p_moved = p + DDF[p].
    """
    if flow_hw.shape[0] != 2:
        flow_hw = np.moveaxis(flow_hw, -1, 0)

    _, H, W = flow_hw.shape
    coords = []

    for x_coord, y_coord in points_xy:
        x0 = np.clip(int(np.floor(x_coord)), 0, W - 2)
        y0 = np.clip(int(np.floor(y_coord)), 0, H - 2)
        x1 = x0 + 1
        y1 = y0 + 1

        wx = float(x_coord - x0)
        wy = float(y_coord - y0)

        def interp(ch):
            return (
                (1 - wx) * (1 - wy) * flow_hw[ch, y0, x0]
                + wx       * (1 - wy) * flow_hw[ch, y0, x1]
                + (1 - wx) * wy       * flow_hw[ch, y1, x0]
                + wx       * wy       * flow_hw[ch, y1, x1]
            )

        dx = interp(1)
        dy = interp(0)

        coords.append((x_coord - dx, y_coord - dy))

    return np.array(coords, dtype=np.float32)


def smooth_ddf(ddf_hw, sigma=DDF_SMOOTHING_SIGMA):
    """Apply spatial Gaussian smoothing to dense displacement fields."""
    if sigma <= 0:
        return ddf_hw
    sigmas = [0.0] * ddf_hw.ndim
    sigmas[-2:] = [sigma, sigma]
    smoothed = gaussian_filter(ddf_hw, sigma=tuple(sigmas))
    return smoothed.astype(ddf_hw.dtype, copy=False)

for patient in patient_list:
    # try:
    print(patient)
    cine_dir = Path(list(patient.glob("*[sS][aA]*[sS][tT][aA][cC]*"))[0])
    # cine_dir = Path(list(patient.glob("*[cC][iI][nN][eE]*"))[0])
    cine_seq = CineSeries(cine_dir, batch_size=200)
    cine_segmentation = cine_seq.predict_segmentation(WANDB_RUN_PATH)
    cine_seq.save_predictions(Path(f"{cine_dir}_segmentation"))
    lv_vol_curve = cine_seq.compute_volume_curve()
    ef = cine_seq.compute_ejection_fraction(lv_vol_curve)

    motion = cine_seq.motion_track(MOTION_WANDB_RUN_PATH)
    motion = motion.reshape(
        3,
        cine_seq.number_of_temporal_positions - 1,
        2,
        *motion.shape[-2:],
    )

    motion = motion[1, ...]
    # motion = smooth_ddf(motion)
    input_images = np.asarray(cine_seq.slice_data["slice06"]["pixel_array"])
    myo = cine_segmentation[cine_seq.mid_slice_num - 1, :] == 2

    the_pts = utils.get_polar_points(
        myo[0, ...].astype(float),
        cine_seq.get_lv_center_points()[1][0],
        cine_seq.get_rv_insertion_points(),
        num_spokes=10,
    )
    
    landmark_xy = np.array(
        [[float(pt[0][0]), float(pt[0][1])] for pt in the_pts], dtype=np.float32
    )
    
    # def warp_point(flow_hw, point_xy):
    #     """flow_hw: (2,H,W) backward flow in pixels, point_xy: (x,y) in pixels."""
    #     H, W = flow_hw.shape[1:]
    #     # build a 1x1 grid at the query point, normalized for grid_sample
    #     grid = torch.tensor([[[[
    #         2.0 * point_xy[0] / (W - 1) - 1.0,
    #         2.0 * point_xy[1] / (H - 1) - 1.0,
    #     ]]]], dtype=flow_hw.dtype)
    #     disp = F.grid_sample(
    #         flow_hw.unsqueeze(0),  # B=1
    #         grid,
    #         mode="bilinear",
    #         padding_mode="border",
    #         align_corners=True,
    #     )[0, :, 0, 0]
    #     return point_xy[0] - disp[0].item(), point_xy[1] - disp[1].item()    
    
    # def solve_deformed_points(flow_slice, points, num_iters=5):
    #     """Iteratively invert the backward flow so points align with Warp output."""
    #     flow_hw = np.transpose(flow_slice, (1, 2, 0))
    #     grid_y = np.arange(flow_hw.shape[0])
    #     grid_x = np.arange(flow_hw.shape[1])
    #     disp_x = RegularGridInterpolator(
    #         (grid_y, grid_x),
    #         flow_hw[..., 0],
    #         bounds_error=False,
    #         fill_value=0.0,
    #     )
    #     disp_y = RegularGridInterpolator(
    #         (grid_y, grid_x),
    #         flow_hw[..., 1],
    #         bounds_error=False,
    #         fill_value=0.0,
    #     )
    #     deformed = []
    #     for spoke in points:
    #         src_x = float(spoke[0][0])
    #         src_y = float(spoke[0][1])
    #         tgt_x, tgt_y = src_x, src_y
    #         for _ in range(num_iters):
    #             delta_x = float(disp_x((tgt_y, tgt_x)))
    #             delta_y = float(disp_y((tgt_y, tgt_x)))
    #             tgt_x = src_x - delta_x
    #             tgt_y = src_y - delta_y
    #             deformed.append((tgt_x, tgt_y))
    #         return deformed

    # deformed_pts = solve_deformed_points(motion[10, ...], the_pts)
    # for x_val, y_val in deformed_pts:
    #     plt.plot(x_val, y_val, "x")
    # plt.show() 

    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = utils.get_colors(10)

    def animate(i):

        ax.clear()
        if i == 0:
            frame_pts = landmark_xy
            im = ax.imshow(input_images[0, ...]/np.amax(input_images[0, ...])+myo[0], cmap="gray")
        else:
            frame_pts = warp_landmarks_direct(motion[i-1, ...], landmark_xy)
            im = ax.imshow(input_images[i, ...], cmap="gray")

        for idx, (pt_x, pt_y) in enumerate(frame_pts):
            ax.plot(
                pt_x,
                pt_y,
                "o",
                color=matplotlib.colors.to_hex(colors[idx]),
                markersize=3,
                markeredgewidth=0.0,
            )

        ax.set_axis_off()
        return [im]

    # animate(10)

    fig.tight_layout()
    anim = FuncAnimation(
        fig,
        animate,
        frames=input_images.shape[0],
        interval=50,
        blit=True,
    )
    # Save as GIF
    writer = PillowWriter(fps=30)
    anim.save(f"{patient.name}.gif", writer=writer)

    # # Close the figure to free memory
    plt.close()
        
    # except:
    #     pass
        
print(motion[1111111])

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
