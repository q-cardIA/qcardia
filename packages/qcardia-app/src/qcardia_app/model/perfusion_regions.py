"""Perfusion regions from the cine segmentation: the CINE_SAX labels mapped onto the perfusion slices through
patient coordinates (no registration), and the mean intensity of each region per frame, for navigation such as
"the frame of peak LV enhancement". Pure numpy, no Qt."""
from __future__ import annotations

import numpy as np
from qcardia.analysis import CINE_LABELS
from scipy.ndimage import binary_erosion

from qcardia_app.view.volume_geometry import VolumeGeometry


def matching_frames(cine_trigger_ms: list[float], perf_trigger_ms: list[float]) -> list[int]:
    """Per perfusion slice, the cine frame nearest its trigger time: the same phase of the heartbeat."""
    cine = np.asarray(cine_trigger_ms, dtype=float)
    return [int(np.argmin(np.abs(cine - t))) for t in perf_trigger_ms]


def project_labels(seg: np.ndarray, cine: VolumeGeometry, frames: list[int], perf: VolumeGeometry) -> np.ndarray:
    """The labels of seg (T, Z, H, W, CINE_SAX in display order) on each perfusion slice (Z, H, W): each
    perfusion pixel takes the label of the nearest cine voxel at that slice's frame (see matching_frames), or 0
    when it lies more than half a cine slice gap from every cine slice."""
    ipp = np.stack(cine.ipp_list)  # (Zc, 3)
    depth = ipp @ cine.normal
    gap = float(np.median(np.abs(np.diff(depth)))) if len(depth) > 1 else 10.0
    rows, cols = np.mgrid[0:perf.n_rows, 0:perf.n_cols]
    out = np.zeros((len(perf.ipp_list), perf.n_rows, perf.n_cols), dtype=np.uint8)
    for z, frame in enumerate(frames):
        world = perf.pixel_to_world(rows[..., None], cols[..., None], z)  # (H, W, 3)
        d = world @ cine.normal  # depth of each pixel along the cine stack
        zc = np.abs(d[..., None] - depth).argmin(-1)
        near = np.abs(d - depth[zc]) <= gap / 2
        delta = world - ipp[zc]
        r = np.rint(np.einsum("hwk,k->hw", delta, cine.col_dir) / cine.pixel_spacing[0]).astype(int)
        c = np.rint(np.einsum("hwk,k->hw", delta, cine.row_dir) / cine.pixel_spacing[1]).astype(int)
        inside = near & (r >= 0) & (r < cine.n_rows) & (c >= 0) & (c < cine.n_cols)
        out[z][inside] = seg[frame, zc[inside], r[inside], c[inside]]
    return out


def first_dynamic(curve: np.ndarray) -> int:
    """The first frame after the proton-density frames at the start of a run, which are several times brighter
    than the baseline."""
    head = curve[: min(len(curve), 15)]
    baseline = np.nanmedian(head)
    start = 0
    while start < len(head) - 1 and head[start] > 1.5 * baseline:
        start += 1
    return start


def perfusion_events(curves: dict[str, np.ndarray]) -> dict[str, int]:
    """Frames (0-based) to navigate to in one run, from the region curves averaged over the slices: the peak
    of the LV blood pool and of the myocardium, and the arrival of the contrast in each: the first frame of the
    rise to the main peak that is above baseline + 20% of the rise. Searching back from the main peak skips a
    dilute pre-bolus."""
    events = {}
    lv = np.nanmean(curves["lv"], axis=1)
    myo = np.convolve(np.nanmean(curves["myo"], axis=1), np.ones(3) / 3, mode="same")  # noisier: smoothed
    start = first_dynamic(lv)
    for name, curve in (("lv", lv), ("myo", myo)):
        if np.all(np.isnan(curve[start:])):
            continue
        after = events.get("lv_peak", start)  # the myocardium enhances after the blood pool
        peak = after + int(np.nanargmax(curve[after:]))
        baseline = float(np.nanmedian(curve[start:start + 5]))
        level = baseline + 0.2 * (curve[peak] - baseline)
        first = events.get("lv_arrival", start)  # contrast reaches the myocardium after the blood pool
        arrival = peak
        while arrival > first and curve[arrival - 1] > level:
            arrival -= 1
        events[f"{name}_peak"], events[f"{name}_arrival"] = peak, arrival
    return events


def region_curves(volume: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Mean intensity per frame and slice (T, Z) in the LV blood pool and the myocardium; NaN where a slice
    has no pixel of the region. The regions lose one pixel at their edges, so that a small shift between the
    cine and the perfusion (no registration) does not mix blood into the myocardium."""
    curves = {}
    for name in ("lv", "myo"):
        mask = np.stack([binary_erosion(z == CINE_LABELS[name]) for z in labels])  # (Z, H, W)
        n = mask.sum(axis=(1, 2))
        total = np.stack([volume[:, z][:, mask[z]].sum(1, dtype=np.float64) for z in range(len(mask))], 1)
        curves[name] = np.where(n > 0, total / np.maximum(n, 1), np.nan)
    return curves
