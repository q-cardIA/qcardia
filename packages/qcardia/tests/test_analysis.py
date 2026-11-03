import numpy as np

from qcardia.analysis import bounding_box, ed_es, myocardial_coverage, segmented_slices, volume_curve


def _phantom():
    """10 frames, 6 slices: an LV disc whose radius peaks in frame 2 and is smallest in frame 7,
    present in slices 1-4 except slice 4 in the frames after frame 5, with an RV blob beside it."""
    radii = [8, 9, 10, 9, 8, 7, 6, 5, 6, 7]
    yy, xx = np.mgrid[:64, :64]
    seg = np.zeros((10, 6, 64, 64), np.uint8)
    for t, r in enumerate(radii):
        for z in range(1, 5 if t <= 5 else 4):
            seg[t, z][(yy - 30) ** 2 + (xx - 30) ** 2 <= r * r] = 1
        seg[t, 2, 28:33, 45:50] = 3
    return seg


def test_volume_curve_and_phases():
    seg = _phantom()
    curve = volume_curve(seg, voxel_mm3=1000.0)  # 1 ml per voxel
    assert curve[2] == 4 * (seg[2, 1] == 1).sum()
    assert ed_es(curve) == (2, 7)


def test_segmented_slices():
    seg = _phantom()
    assert segmented_slices(seg, 0) == [1, 2, 3, 4]
    assert segmented_slices(seg, 7) == [1, 2, 3]
    assert segmented_slices(seg, 7, label=3) == [2]


def test_bounding_box():
    assert bounding_box(_phantom()) == (20, 20, 40, 49)
    assert bounding_box(np.zeros((2, 3, 8, 8), np.uint8)) is None


def test_myocardial_coverage():
    """A full myocardial ring, a half ring, and a slice without LV."""
    yy, xx = np.mgrid[:64, :64]
    disc = (yy - 32) ** 2 + (xx - 32) ** 2
    seg = np.zeros((1, 3, 64, 64), np.uint8)
    for z in (0, 1):
        seg[0, z][disc <= 15 ** 2] = 2
        seg[0, z][disc <= 10 ** 2] = 1
    seg[0, 1][(disc <= 15 ** 2) & (disc > 10 ** 2) & (xx < 32)] = 0  # myocardium on one side only
    full, half, empty = myocardial_coverage(seg, 0)
    assert full == 1.0
    assert 0.4 <= half <= 0.6
    assert empty == 0.0
