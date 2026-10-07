"""Builds the per-turn STATE block attached to the user's message. Pure
string formatting, no Qt."""
from __future__ import annotations

from utils.layouts import split_position, split_text


def build_state_block(
    *,
    subject: str,
    loaded_series: list[str],
    time_frame: tuple[int, int] | None,
    grid: tuple[int, int],
    panels: list[dict],
    loading_series: list[str] = (),
    analysis: dict | None = None,
    segmentable: tuple[str, ...] = (),
    playing: bool = False,
    speed: float = 1.0,
    level_matches: dict[str, list[int]] | None = None,
    split: tuple[int, int, int, int] | None = None,
) -> str:
    if time_frame:
        current, total = time_frame
        time_line = f"Time frame: {current + 1} / {total}" + (f"  (cines playing at {speed:g}x)" if playing else "")
    else:
        time_line = "Time frame: n/a"

    rows, cols = grid
    panel_lines = []
    for panel in panels:
        where = split_position(split, panel["idx"]) if split else "row %d, col %d" % divmod(panel["idx"], cols)
        line = f"  {panel['idx']} ({where}): "
        if panel["series_key"] is None:
            panel_lines.append(line + "empty")
            continue
        line += f"{panel['series_key']} / {panel['axis']}"
        if "slice" in panel:
            line += f"  slice {panel['slice'] + 1}/{panel['n_slices']}"
        if panel["has_segmentation"]:
            line += "  seg shown"
        elif panel.get("segmented"):
            line += "  seg hidden"
        if panel.get("own_window"):
            line += "  own window"
        if panel.get("loading"):
            line += "  (loading)"
        panel_lines.append(line)

    loading_line = ""
    if loading_series:
        loading_line = (f"Still loading: {', '.join(loading_series)} (use them like loaded series: "
                        "panels and segmentation wait until they have loaded)\n")
    return (
        "STATE:\n"
        f"Subject: {subject}\n"
        f"Loaded series: {', '.join(loaded_series) if loaded_series else 'none'}\n"
        f"{loading_line}"
        f"{time_line}\n"
        f"Grid: {grid_text(rows, cols, split)}\n"
        "Panels:\n" + ("\n".join(panel_lines) if panel_lines else "  none")
        + _measurements(analysis or {}, [k for k in segmentable if k in loaded_series], time_frame, level_matches or {},
                        {pl["series_key"] for pl in panels if pl["series_key"] and pl["has_segmentation"]})
    )


BASAL_COVERAGE = 0.5  # Schulz-Menger et al. 2020: the basal slice has >= 50% of the blood pool surrounded


def base_to_apex(sax: dict, frame: int) -> list[int]:
    """The CINE_SAX slices with LV in frame, from the basal slice to the apex. The basal slice is the most
    basal one whose blood pool myocardium surrounds for at least BASAL_COVERAGE (see
    qcardia.analysis.myocardial_coverage); without coverage, the most basal slice with LV. Empty for a
    frame CINE_SAX does not have (the cine frame can come from a longer long-axis cine)."""
    if not 0 <= frame < len(sax["lv_slices"]):
        return []
    with_lv = set(sax["lv_slices"][frame])
    order = [z for z in reversed(sax["apex_to_base"]) if z in with_lv]
    coverage = sax.get("lv_coverage")
    if coverage:
        start = next((i for i, z in enumerate(order) if coverage[frame][z] >= BASAL_COVERAGE), 0)
        order = order[start:]
    return order


def grid_text(rows: int, cols: int, split: tuple[int, int, int, int] | None = None) -> str:
    if split:
        return f"split in two halves: {split_text(split)}, i.e. set_split_grid{split}"
    return f"{rows}x{cols}"


def base_mid_apex(sax: dict, frame: int) -> tuple[int, int, int] | None:
    """The CINE_SAX base, mid and apex slices in frame (0-based), by base_to_apex."""
    order = base_to_apex(sax, frame)
    return (order[0], order[len(order) // 2], order[-1]) if order else None


def _measurements(analysis: dict, segmentable: list[str], time_frame, level_matches: dict,
                  shown: set[str] = frozenset()) -> str:
    """What the segmentations say, or which navigation needs a segmentation first."""
    lines = []
    sax = analysis.get("CINE_SAX")
    if sax and "ed" in sax:
        order = sax["apex_to_base"]
        n = len(order)
        if order == sorted(order):
            lines.append(f"  CINE_SAX slices run from apex (slice 1) to base (slice {n}).")
        elif order == sorted(order, reverse=True):
            lines.append(f"  CINE_SAX slices run from base (slice 1) to apex (slice {n}).")
        ed, es, volumes = sax["ed"], sax["es"], sax["volumes"]
        lines.append(f"  CINE_SAX ED = frame {ed + 1} (largest LV volume, {volumes[ed]} ml), "
                     f"ES = frame {es + 1} (smallest LV volume, {volumes[es]} ml).")
        at = [(f"ED (frame {ed + 1})", ed), (f"ES (frame {es + 1})", es)]
        current = time_frame[0] if time_frame else 0
        if current not in (ed, es):
            at.append((f"the current frame {current + 1}", current))
        lines.append("  CINE_SAX slices with LV from base to apex (the base is the most basal slice with at least "
                     "half of the blood pool surrounded by myocardium): " + "; ".join(
                         f"at {name}: {', '.join(str(z + 1) for z in base_to_apex(sax, t)) or 'none'}"
                         for name, t in at) + ".")
        levels = base_mid_apex(sax, current)
        if levels:
            lines.append(f"  At the current frame {current + 1}: CINE_SAX base = slice {levels[0] + 1}, mid = slice "
                         f"{levels[1] + 1}, apex = slice {levels[2] + 1}.")
        for key, (base, mid, apex) in level_matches.items():
            lines.append(f"  {key} slices nearest to these: base = slice {base + 1}, mid = slice {mid + 1}, "
                         f"apex = slice {apex + 1}.")
    segmented = [k for k in segmentable if analysis.get(k, {}).get("box")]
    if segmented:
        lines.append(f"  Segmentation available (cached or run) for: {', '.join(segmented)}. ED/ES, the LV "
                     "slices and zoom roi use it now: no run_segmentation needed for those.")
        hidden = [k for k in segmented if k not in shown]
        if hidden:
            lines.append(f"  Contours drawn in no panel: {', '.join(hidden)}. Draw them only when the user asks to "
                         "segment or to show the segmentation: run_segmentation.")
    missing = [k for k in segmentable if k not in analysis]
    if missing:
        lines.append(f"  Not segmented: {', '.join(missing)}. ED/ES, the LV slices and zoom roi based "
                     "on these need run_segmentation first: ask the user before you run it.")
    return "\nSegmentation:\n" + "\n".join(lines) if lines else ""

