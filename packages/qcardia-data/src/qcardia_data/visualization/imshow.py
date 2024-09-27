import matplotlib.pyplot as plt
import numpy as np
import torch

from qcardia_data.visualization.utils import histogram_equalization_np, overlay_img


def plot_data_dict(
    data_dict: dict,
    image_key: str,
    label_key: str,
    overlay_alpha=0.5,
    histogram_equalization=True,
    figwidth: int = 12,
    figlimit: int = 4,
):
    batch_size = data_dict[image_key].size(0)
    nr_images = min(batch_size, figlimit) if figlimit > 0 else batch_size

    meta_key = f"{image_key}_meta_dict"
    min_intensity = data_dict[meta_key]["min_intensity"][:nr_images]
    max_intensity = data_dict[meta_key]["max_intensity"][:nr_images]
    file_ids = data_dict["meta_dict"]["file_id"][:nr_images]

    image = data_dict[image_key][:nr_images, ...].squeeze(1)
    label = data_dict[label_key][:nr_images, ...]
    nr_classes = label.size(1)
    label = torch.argmax(label, dim=1, keepdim=False)

    for i in range(nr_images):
        if histogram_equalization:
            img = torch.tensor(histogram_equalization_np(np.array(image[i])))
        else:
            vmin = min_intensity[i].item()
            vmax = max_intensity[i].item()
            img = (image[i] - vmin) / (vmax - vmin)
            img = torch.clamp(img, 0.0, 1.0)
        overlayed_img = overlay_img(img, label[i], overlay_alpha)

        print(file_ids[i])
        _, axs = plt.subplots(1, 3, figsize=(figwidth, figwidth / 3))
        settings = {"interpolation": "none"}
        axs[0].imshow(img, cmap="gray", vmin=0, vmax=1, **settings)
        axs[1].imshow(label[i], cmap="gray", vmin=0, vmax=nr_classes, **settings)
        axs[2].imshow(overlayed_img, vmin=0, vmax=1, **settings)
        for ax in axs:
            ax.set_axis_off()
        plt.tight_layout()
        plt.show()
