"""Window/level: map image values to uint8 display values."""
from __future__ import annotations

import numpy as np
from qcardia.analysis import bounding_box


def dicom_window(ds) -> tuple[float, float] | None:
    """(center, width) of a DICOM header's first window, or None if missing or invalid."""
    try:
        center, width = (float(v[0]) if hasattr(v, "__len__") else float(v)
                         for v in (ds.WindowCenter, ds.WindowWidth))
    except (AttributeError, TypeError, ValueError, IndexError):
        return None
    return (center, width) if width > 0 else None


def percentile_window(arr: np.ndarray, low: float, high: float) -> tuple[float, float] | None:
    """(center, width) spanning the low-high percentile range, or None if that range is empty."""
    lo, hi = np.percentile(arr, [low, high])
    return ((lo + hi) / 2, hi - lo) if hi > lo else None


def window_to_uint8(arr: np.ndarray, center: float, width: float) -> np.ndarray:
    out = arr.astype(np.float32)
    out -= center - width / 2
    out *= 255.0 / width
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


# Relative steps of set_window: the level moves by a fraction of the width, the width scales.
WINDOW_STEPS = {"darker": (0.1, 1.0), "brighter": (-0.1, 1.0), "more_contrast": (0.0, 0.8),
                "less_contrast": (0.0, 1.25)}


def null_myocardium_window(volume: np.ndarray, seg: np.ndarray, low: float = 5, high: float = 99,
                           margin: float = 0.3) -> tuple[float, float] | None:
    """(center, width) for LGE after the SCMR 2020 post-processing recommendations (Schulz-Menger et
    al.): nulled myocardium dark but not one intensity, so its noise stays visible, and LGE not
    saturated. The window runs from the low percentile of the healthy myocardium (seg == 1) to the
    high percentile of the heart: each slice's segmentation box plus margin of its size on each side.
    None if the segmentation does not fit the volume or has no healthy myocardium."""
    if volume.shape != seg.shape or not (seg == 1).any():
        return None
    heart = []
    for image, labels in zip(volume, seg):
        box = bounding_box(labels)
        if box is not None:
            r0, c0, r1, c1 = box
            dr, dc = int(margin * (r1 - r0 + 1)), int(margin * (c1 - c0 + 1))
            heart.append(image[max(0, r0 - dr):r1 + dr + 1, max(0, c0 - dc):c1 + dc + 1].ravel())
    lo, hi = float(np.percentile(volume[seg == 1], low)), float(np.percentile(np.concatenate(heart), high))
    return ((lo + hi) / 2, hi - lo) if hi > lo else None


def step_window(window: tuple[float, float], mode: str, steps: int = 1) -> tuple[float, float]:
    """window moved steps times by a WINDOW_STEPS mode."""
    center, width = window
    shift, scale = WINDOW_STEPS[mode]
    for _ in range(steps):
        center, width = center + shift * width, width * scale
    return center, width
