from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from monai.networks.blocks import Warp
from natsort import natsorted

import utils
from qcardia.series import CineSeries

MOTION_WANDB_RUN_PATH = Path.cwd() / "wandb" / "motion-model"
WANDB_RUN_PATH = Path.cwd() / "wandb" / "cine-seg"
PATH_TO_DATASET = Path.cwd() / "data"

patient_list = natsorted([f for f in PATH_TO_DATASET.iterdir() if f.is_dir()])

warp_layer = Warp(mode="bilinear", padding_mode="border")

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
    mid_slice_key = f"slice{cine_seq.mid_slice_num:02d}"
    input_images = np.asarray(cine_seq.slice_data[mid_slice_key]["pixel_array"])
    myo = cine_segmentation[cine_seq.mid_slice_num - 1, :] == 2
    rv = cine_segmentation[cine_seq.mid_slice_num - 1, :] == 3

    num_landmarks = 10
    landmark_xy = cine_seq.get_lv_endo_landmarks(t=0, n=num_landmarks)
    landmark_xy_rv = cine_seq.get_rv_freewall_landmarks(t=0, n=num_landmarks)

    seg_ed = torch.from_numpy(myo[0].astype(np.float32)).unsqueeze(0).unsqueeze(0)
    seg_ed_rv = torch.from_numpy(rv[0].astype(np.float32)).unsqueeze(0).unsqueeze(0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = utils.get_colors(num_landmarks)

    def animate(i):
        ax.clear()
        if i == 0:
            frame_pts = landmark_xy
            frame_pts_rv = landmark_xy_rv
            warped_myo = myo[0]
            warped_rv = rv[0]
        else:
            flow = torch.from_numpy(motion[i - 1, ...]).unsqueeze(0).float()
            frame_pts = cine_seq.warp_landmarks(motion[i - 1, ...], landmark_xy)
            frame_pts_rv = cine_seq.warp_landmarks(motion[i - 1, ...], landmark_xy_rv)
            warped_myo = warp_layer(seg_ed, flow).squeeze().numpy() > 0.5
            warped_rv = warp_layer(seg_ed_rv, flow).squeeze().numpy() > 0.5

        overlay = input_images[i] / np.amax(input_images[i]) + warped_myo + warped_rv
        im = ax.imshow(overlay, cmap="gray")
        for idx, (pt_x, pt_y) in enumerate(frame_pts):
            ax.plot(
                pt_x,
                pt_y,
                "o",
                color=matplotlib.colors.to_hex(colors[idx]),
                markersize=3,
                markeredgewidth=0.0,
            )
        for idx, (pt_x, pt_y) in enumerate(frame_pts_rv):
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
