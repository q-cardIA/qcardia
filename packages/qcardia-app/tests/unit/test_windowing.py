import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from qcardia_app.utils.windowing import null_myocardium_window, step_window  # noqa: E402


def _lge_phantom():
    """Two slices: a myocardial ring (1 = healthy, about 20 +- 5) with a scar patch (2, 250) around a
    bright blood pool (200), on a dim background (60)."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:96, :96]
    disc = (yy - 48) ** 2 + (xx - 48) ** 2
    seg = np.zeros((2, 96, 96), np.uint8)
    seg[:, (disc <= 20 ** 2) & (disc > 12 ** 2)] = 1
    seg[0, (disc <= 20 ** 2) & (disc > 12 ** 2) & (xx > 58)] = 2
    volume = np.full(seg.shape, 60.0)
    volume[:, disc <= 12 ** 2] = 200
    volume[seg == 1] = rng.normal(20, 5, (seg == 1).sum())
    volume[seg == 2] = 250
    return volume, seg


def test_null_myocardium_window():
    volume, seg = _lge_phantom()
    center, width = null_myocardium_window(volume, seg)
    bottom, top = center - width / 2, center + width / 2
    grey = (np.median(volume[seg == 1]) - bottom) / width
    assert 0 < grey < 0.1  # healthy myocardium dark, but above black: its noise stays visible
    assert (volume[seg == 1] > bottom).mean() > 0.9
    assert top >= 250  # scar not saturated


def test_null_myocardium_window_needs_a_fitting_segmentation():
    volume, seg = _lge_phantom()
    assert null_myocardium_window(volume, seg[:1]) is None
    assert null_myocardium_window(volume, np.zeros_like(seg)) is None


def test_step_window():
    assert step_window((100, 200), "darker", 2) == (140, 200)
    assert step_window((100, 200), "brighter") == (80, 200)
    assert step_window((100, 200), "more_contrast") == (100, 160)
