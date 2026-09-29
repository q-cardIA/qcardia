"""Window/level: map image values to uint8 display values."""
from __future__ import annotations

import numpy as np


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
