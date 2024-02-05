from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from natsort import natsorted
from qcardia_models.models import UNet2d
from torch.nn import functional as F

from src.qcardia.data.sequence import BaseSequence

WANDB_RUN_PATH = Path.cwd() / "wandb"
PATH_TO_DATASET = Path.cwd() / "data"
patient_list = natsorted([f for f in PATH_TO_DATASET.iterdir() if f.is_dir()])
patient = patient_list[0]
cine_dir = Path(list(patient.glob("*[sS][aA]*[sS][tT][aA][cC]*"))[0])

cine_seq = BaseSequence(cine_dir)

cine = cine_seq.get_array()

cine_img_summed = np.sum(cine, axis=(0, 1))
borderless_idxs_0 = np.nonzero(np.any(cine_img_summed, axis=1))[0]
borderless_idxs_1 = np.nonzero(np.any(cine_img_summed, axis=0))[0]

start_idx_0, stop_idx_0 = borderless_idxs_0[0], borderless_idxs_0[-1] + 1
start_idx_1, stop_idx_1 = borderless_idxs_1[0], borderless_idxs_1[-1] + 1

cine_borderless = cine[..., start_idx_0:stop_idx_0, start_idx_1:stop_idx_1]

reshaped_cine_borderless = cine_borderless.reshape(
    -1,
    1,
    cine_borderless.shape[-2],
    cine_borderless.shape[-1],
)
reshaped_cine_borderless -= np.amin(reshaped_cine_borderless)

config_path = WANDB_RUN_PATH / "files" / "config-copy.yaml"
config = yaml.load(Path.open(config_path), Loader=yaml.FullLoader)

cine_torch_tensor = torch.tensor(reshaped_cine_borderless.astype(np.float32))
cine_pixdim = torch.tensor(
    [
        float(cine_seq.slice_data["slice01"]["meta_data"][0].PixelSpacing[0]),
        float(cine_seq.slice_data["slice01"]["meta_data"][0].PixelSpacing[1]),
        float(cine_seq.slice_data["slice01"]["meta_data"][0].SliceThickness),
    ],
    dtype=torch.float32,
)
nr_slices = cine_torch_tensor.shape[0]
target_pixdim = torch.tensor(config["data"]["target_pixdim"])
target_size = torch.tensor(config["data"]["target_size"])
grid_sample_modes = [config["data"]["image_grid_sample_mode"]]


def T_2D_scale(scales):
    T_scale = torch.tensor(
        [
            [scales[1], 0, 0],
            [0, scales[0], 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    return T_scale


source_shape = torch.tensor(cine_torch_tensor.shape[2:], dtype=torch.float32)
real_source_size = cine_pixdim[:2] * source_shape
real_target_size = target_pixdim * target_size
dimension_scale_factor = real_target_size / real_source_size
T = T_2D_scale(dimension_scale_factor)

grid_size = [nr_slices, 1, target_size[0], target_size[1]]


print(cine_torch_tensor.shape)
print(grid_size)

grid = F.affine_grid(
    theta=torch.repeat_interleave(T[:-1, :].unsqueeze(0), nr_slices, dim=0),
    size=grid_size,
    align_corners=False,
)
cine_torch_tensor = F.grid_sample(
    cine_torch_tensor,
    grid,
    align_corners=False,
    mode=grid_sample_modes[0],
    padding_mode="zeros",
)


cine_torch_tensor -= np.mean(reshaped_cine_borderless)
cine_torch_tensor /= np.std(reshaped_cine_borderless)


model = UNet2d(
    nr_input_channels=config["unet"]["nr_image_channels"],
    channels_list=config["unet"]["channels_list"],
    nr_output_classes=config["unet"]["nr_output_classes"],
    nr_output_scales=config["unet"]["nr_output_scales"],
).to("cpu")

model_weights = torch.load(WANDB_RUN_PATH / "files" / "last_model.pt")
model.load_state_dict(model_weights)
model.eval()
model_output = torch.zeros(
    nr_slices, config["unet"]["nr_output_classes"], target_size[0], target_size[1]
)
batch_size = 50
with torch.no_grad():
    for i in range(0, nr_slices // batch_size + 1):
        model_output[i * batch_size : (i + 1) * batch_size] = model(
            cine_torch_tensor[i * batch_size : (i + 1) * batch_size]
        )[0]

dimension_rescale_factor = 1 / dimension_scale_factor
T_inv = T_2D_scale(dimension_rescale_factor)

inv_grid_size = [
    nr_slices,
    1,
    int(source_shape[0]),
    int(source_shape[1]),
]

grid = F.affine_grid(
    theta=torch.repeat_interleave(T_inv[:-1, :].unsqueeze(0), nr_slices, dim=0),
    size=inv_grid_size,
    align_corners=False,
)
resample_model_output = F.grid_sample(
    torch.argmax(model_output, dim=1, keepdim=True).float(),
    grid,
    align_corners=False,
    mode="nearest",
    padding_mode="zeros",
)

reshaped_cine_segmentation = np.zeros(
    (nr_slices, cine.shape[-2], cine.shape[-1]), dtype=np.uint8
)

reshaped_cine_segmentation[..., start_idx_0:stop_idx_0, start_idx_1:stop_idx_1] = (
    resample_model_output.squeeze().numpy().astype(np.uint8)
)

cine_segmentation = reshaped_cine_segmentation.reshape(cine.shape)

# quick check
from matplotlib import pyplot as plt

plt.imshow(
    cine[6, 6] / 1000 + cine_segmentation[6, 6],
)
plt.show()
