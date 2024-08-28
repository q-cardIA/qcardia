import matplotlib.pyplot as plt
import numpy as np
import torch


def overlay_img(image: torch.Tensor, label: torch.Tensor, overlay_alpha: float):
    if len(image.shape) == 2:
        image = image.unsqueeze(-1)
        image = torch.cat([image, image, image], dim=-1)

    colors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.8, 0.0, 1.0],
            [1.0, 0.5, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    nr_classes = int(torch.max(label).item()) + 1
    if nr_classes > len(colors) + 1:
        raise NotImplementedError(
            f"Not enough colors implemented for {nr_classes} classes"
            + f"(only {len(colors)} implemented)"
        )
    colored_gt = torch.zeros_like(image)
    for class_nr in range(1, nr_classes):
        class_label = (torch.round(label).long() == class_nr).float().unsqueeze(-1)
        class_label = torch.cat([class_label, class_label, class_label], dim=-1)
        colored_gt += class_label * colors[class_nr - 1][None, None, ...]
    colored_gt = torch.clamp(colored_gt, 0.0, 1.0)
    image_filter = 1.0 - torch.max(colored_gt, dim=-1, keepdim=True)[0] * overlay_alpha
    image = image * image_filter + colored_gt * overlay_alpha
    return torch.clamp(image, 0.0, 1.0)


def histogram_equalization_np(img, nr_bins=256):
    img_flat = img.flatten()
    imhist, bins = np.histogram(img_flat, nr_bins)
    cdf = imhist.cumsum()
    cdf = cdf - np.min(cdf)
    cdf = cdf / np.max(cdf)
    img_equalized = np.interp(img_flat, bins[:-1], cdf)
    return img_equalized.reshape(img.shape)
