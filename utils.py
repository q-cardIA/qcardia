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
        # Calculate unit vector for this angle
        dx = np.cos(angle)
        dy = np.sin(angle)

        # Generate points along the spoke
        num_points = int(max_radius * 4)
        radii = np.linspace(0, max_radius, num_points)
        spoke_points = np.array([(cx + r * dx, cy + r * dy) for r in radii])

        # Find intersections with the donut
        intersections = []
        for i in range(len(spoke_points) - 1):
            x1, y1 = spoke_points[i]
            x2, y2 = spoke_points[i + 1]

            # Check if we're crossing the boundary
            if (
                0 <= int(y1) < height
                and 0 <= int(x1) < width
                and 0 <= int(y2) < height
                and 0 <= int(x2) < width
            ):
                val1 = myo[int(y1), int(x1)]
                val2 = myo[int(y2), int(x2)]
                if val1 != val2:  # We found an intersection
                    # Use the midpoint as the intersection point
                    intersections.append([(x1 + x2) / 2, (y1 + y2) / 2])

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
        # Calculate unit vector for this angle
        dx = np.cos(angle)
        dy = np.sin(angle)

        # Generate points along the spoke
        num_points = int(max_radius * 4)
        radii = np.linspace(0, max_radius, num_points)
        spoke_points = np.array([(cx + r * dx, cy + r * dy) for r in radii])

        # Find intersections with the donut
        intersections = []
        for i in range(len(spoke_points) - 1):
            x1, y1 = spoke_points[i]
            x2, y2 = spoke_points[i + 1]

            # Check if we're crossing the boundary
            if (
                0 <= int(y1) < height
                and 0 <= int(x1) < width
                and 0 <= int(y2) < height
                and 0 <= int(x2) < width
            ):
                val1 = myo[int(y1), int(x1)]
                val2 = myo[int(y2), int(x2)]
                if val1 != val2:  # We found an intersection
                    # Use the midpoint as the intersection point
                    intersections.append([(x1 + x2) / 2, (y1 + y2) / 2])

        all_intersections.append(intersections)

    return all_intersections
