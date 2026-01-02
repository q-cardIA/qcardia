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

for patient in patient_list:
    try:
        # cine_dir = Path(list(patient.glob("*[sS][aA]*[sS][tT][aA][cC]*"))[0])
        cine_dir = Path(list(patient.glob("*[cC][iI][nN][eE]*"))[0])
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