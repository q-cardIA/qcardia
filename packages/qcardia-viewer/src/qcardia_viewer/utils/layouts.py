"""Panel layouts as grid cells, without Qt, so code without a window can name and check them."""
from __future__ import annotations

from math import lcm

MAX_GRID_ROWS, MAX_GRID_COLS = 4, 8  # the largest grid that can be requested
MAX_HALF = 4  # rows and columns of each half of a split layout
CINE_SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)  # playback speeds, heartbeats per second


def layout_cells(rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    """The panels of a rows x cols grid as (row, col, rows, cols), in reading order."""
    return [(r, c, 1, 1) for r in range(rows) for c in range(cols)]


def split_layout(split: tuple[int, int, int, int]) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    """(grid rows, grid columns, cells) for the window split into two equal halves, a left_rows x
    left_cols grid and a right_rows x right_cols grid; a 1x1 half is one large panel. The cells are
    numbered through the left half first, then the right half, each in reading order."""
    left_rows, left_cols, right_rows, right_cols = split
    rows, half = lcm(left_rows, right_rows), lcm(left_cols, right_cols)

    def cells(r: int, c: int, col0: int) -> list[tuple[int, int, int, int]]:
        h, w = rows // r, half // c
        return [(i * h, col0 + j * w, h, w) for i in range(r) for j in range(c)]

    return rows, 2 * half, cells(left_rows, left_cols, 0) + cells(right_rows, right_cols, half)


def split_text(split: tuple[int, int, int, int]) -> str:
    """'left 3x3 (panels 1-9), right one large panel (10)', numbered from 1 as the user sees them."""
    left_rows, left_cols, right_rows, right_cols = split
    n_left, n_right = left_rows * left_cols, right_rows * right_cols

    def half(rows: int, cols: int, first: int, n: int) -> str:
        if n == 1:
            return f"one large panel ({first})"
        return f"{rows}x{cols} (panels {first}-{first + n - 1})"

    return (f"left {half(left_rows, left_cols, 1, n_left)}, "
            f"right {half(right_rows, right_cols, n_left + 1, n_right)}")


def split_position(split: tuple[int, int, int, int], panel: int) -> str:
    """Where panel (0-based) is in a split layout, e.g. 'left, row 2, col 1' or 'right, large'; rows and columns from 1."""
    left_rows, left_cols, right_rows, right_cols = split
    side, rows, cols = "left", left_rows, left_cols
    if panel >= left_rows * left_cols:
        side, rows, cols, panel = "right", right_rows, right_cols, panel - left_rows * left_cols
    return f"{side}, large" if rows * cols == 1 else f"{side}, row {panel // cols + 1}, col {panel % cols + 1}"
