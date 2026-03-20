import colorsys
import random
from typing import List, Tuple

import numpy as np


# from: https://github.com/riponazad/echotracker/blob/main/utils/viz_utils.py
# Generate random colormaps for visualizing different points.
def get_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    """Gets colormap for points."""
    colors = []
    for i in np.arange(0.0, 360.0, 360.0 / num_colors):
        hue = i / 360.0
        lightness = (50 + np.random.rand() * 10) / 100.0
        saturation = (90 + np.random.rand() * 10) / 100.0
        color = colorsys.hls_to_rgb(hue, lightness, saturation)
        # colors.append((int(color[0]), int(color[1] * 255), int(color[2] * 255)))
        colors.append((color[0], color[1], color[2]))
    random.shuffle(colors)
    return colors


def _sample_mask(myo: np.ndarray, x: float, y: float, height: int, width: int) -> int:
    """Sample nearest pixel from floating point coordinates."""
    ix = int(round(x))
    iy = int(round(y))
    if 0 <= iy < height and 0 <= ix < width:
        return int(myo[iy, ix] > 0)
    return 0


def _refine_transition(
    myo: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
    start_val: int,
    height: int,
    width: int,
    iterations: int = 8,
) -> List[float]:
    """Binary search between two samples that straddle the boundary."""
    ax, ay = start
    bx, by = end
    for _ in range(iterations):
        mx = (ax + bx) * 0.5
        my = (ay + by) * 0.5
        mid_val = _sample_mask(myo, mx, my, height, width)
        if mid_val == start_val:
            ax, ay = mx, my
        else:
            bx, by = mx, my
    return [(ax + bx) * 0.5, (ay + by) * 0.5]


def _trace_spoke_intersections(
    myo: np.ndarray,
    cx: float,
    cy: float,
    angle: float,
    max_radius: float,
    height: int,
    width: int,
    step: float = 0.25,
    max_intersections: int = 2,
):
    dx = np.cos(angle)
    dy = np.sin(angle)
    radii = np.arange(step, max_radius + step, step)
    prev_point = (cx, cy)
    prev_val = _sample_mask(myo, cx, cy, height, width)
    intersections = []
    for r in radii:
        x = cx + r * dx
        y = cy + r * dy
        curr_val = _sample_mask(myo, x, y, height, width)
        if curr_val != prev_val:
            intersections.append(
                _refine_transition(myo, prev_point, (x, y), prev_val, height, width)
            )
            prev_val = curr_val
            if len(intersections) >= max_intersections:
                break
        prev_point = (x, y)
    return intersections


def get_polar_points(myo, lv, rv, num_spokes=60):

    cy = lv[0]
    cx = lv[1]

    # Get image dimensions
    height, width = myo.shape

    # Calculate angles for spokes (in radians)
    angles = np.linspace(0, 2 * np.pi, num_spokes, endpoint=False)
    # Maximum radius to check (diagonal of image)
    max_radius = np.sqrt(width**2 + height**2)

    # Store intersection points for each spoke
    all_intersections = []
    for angle in angles:
        intersections = _trace_spoke_intersections(
            myo,
            cx,
            cy,
            angle,
            max_radius,
            height,
            width,
            step=0.25,
            max_intersections=2,
        )
        all_intersections.append(intersections)

    return all_intersections


def get_rv_polar_points(myo, lv, rv_pt, num_spokes=60):

    cy = lv[0]
    cx = lv[1]

    # Get image dimensions
    height, width = myo.shape

    # Calculate angles for spokes (in radians)
    angles = np.linspace(0, 2 * np.pi, num_spokes, endpoint=False)
    # Maximum radius to check (diagonal of image)
    max_radius = np.sqrt(width**2 + height**2)

    # Store intersection points for each spoke
    all_intersections = []
    for angle in angles:
        intersections = _trace_spoke_intersections(
            myo,
            cx,
            cy,
            angle,
            max_radius,
            height,
            width,
            step=0.125,
            max_intersections=4,
        )
        all_intersections.append(intersections)

    return all_intersections
