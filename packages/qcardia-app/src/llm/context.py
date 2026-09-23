"""Builds the per-turn STATE block attached to the user's message. Pure
string formatting, no Qt."""
from __future__ import annotations


def build_state_block(
    *,
    subject: str,
    loaded_series: list[str],
    time_frame: tuple[int, int] | None,
    grid: tuple[int, int],
    panels: list[dict],
) -> str:
    if time_frame:
        current, total = time_frame
        time_line = f"Time frame: {current + 1} / {total} (1-based display; set_time_frame uses 0-{total - 1})"
    else:
        time_line = "Time frame: n/a"

    rows, cols = grid
    panel_lines = []
    for panel in panels:
        if panel["series_key"] is None:
            panel_lines.append(f"  {panel['idx']}: empty")
        else:
            seg_note = "  seg=True" if panel["has_segmentation"] else ""
            panel_lines.append(f"  {panel['idx']}: {panel['series_key']} / {panel['axis']}{seg_note}")

    return (
        "STATE:\n"
        f"Subject: {subject}\n"
        f"Loaded series: {', '.join(loaded_series) if loaded_series else 'none'}\n"
        f"{time_line}\n"
        f"Grid: {rows}x{cols}\n"
        "Panels:\n" + ("\n".join(panel_lines) if panel_lines else "  none")
    )
