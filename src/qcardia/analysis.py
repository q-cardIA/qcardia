"""Measurements from a segmentation array, whether just predicted or read back from a cache.

A cine segmentation is (T, Z, H, W) with 1 = LV blood pool, 2 = myocardium, 3 = RV.
"""
from __future__ import annotations

import numpy as np

CINE_LABELS = {"lv": 1, "myo": 2, "rv": 3}


def volume_curve(seg: np.ndarray, voxel_mm3: float, label: int = CINE_LABELS["lv"]) -> np.ndarray:
    """Volume of label in ml for each frame of a (T, Z, H, W) segmentation."""
    return (seg == label).sum(axis=(1, 2, 3)) * voxel_mm3 / 1000


def ed_es(curve: np.ndarray) -> tuple[int, int]:
    """End-diastole and end-systole frames: the largest and the smallest LV volume."""
    return int(np.argmax(curve)), int(np.argmin(curve))


def segmented_slices(seg: np.ndarray, frame: int, label: int = CINE_LABELS["lv"]) -> list[int]:
    """The slices of a (T, Z, H, W) segmentation that contain label in that frame."""
    return np.flatnonzero((seg[frame] == label).any(axis=(1, 2))).tolist()


def bounding_box(seg: np.ndarray) -> tuple[int, int, int, int] | None:
    """(row0, col0, row1, col1), inclusive, around every labelled pixel in any frame and slice."""
    in_plane = seg.reshape(-1, *seg.shape[-2:]).any(axis=0)
    rows, cols = np.flatnonzero(in_plane.any(axis=1)), np.flatnonzero(in_plane.any(axis=0))
    if not rows.size:
        return None
    return int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])


def myocardial_coverage(seg: np.ndarray, frame: int, rays: int = 72, reach: int = 3) -> list[float]:
    """Per slice of a (T, Z, H, W) segmentation, how much of the LV blood pool myocardium surrounds:
    the fraction of rays from the blood-pool centre that meet myocardium within reach pixels of where
    they leave the blood pool; 0 for a slice without LV. The SCMR 2020 post-processing recommendations
    (Schulz-Menger et al.) take the basal slice as one with at least 50% of the blood pool surrounded."""
    lv, myo = CINE_LABELS["lv"], CINE_LABELS["myo"]
    angles = np.linspace(0, 2 * np.pi, rays, endpoint=False)
    coverage = []
    for sl in seg[frame]:
        rows, cols = np.nonzero(sl == lv)
        if not rows.size:
            coverage.append(0.0)
            continue
        h, w = sl.shape
        r0, c0 = rows.mean(), cols.mean()
        # Far enough to leave the blood pool and look reach pixels beyond it.
        radius = np.arange(int(np.hypot(rows - r0, cols - c0).max()) + reach + 2)
        r = np.rint(r0 + np.outer(np.sin(angles), radius)).astype(int)
        c = np.rint(c0 + np.outer(np.cos(angles), radius)).astype(int)
        inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        labels = np.where(inside, sl[r.clip(0, h - 1), c.clip(0, w - 1)], 0)
        leave = np.argmax(labels != lv, axis=1)
        ahead = (leave[:, None] + np.arange(reach + 1)).clip(0, len(radius) - 1)
        coverage.append(float((np.take_along_axis(labels, ahead, axis=1) == myo).any(axis=1).mean()))
    return coverage
