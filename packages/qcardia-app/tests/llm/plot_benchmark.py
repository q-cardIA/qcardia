#!/usr/bin/env python
"""Plots for a benchmark.py run: an extended spider chart (pass rate per
category x model) and a per-model latency boxplot, styled to match
cardiagent's old tests/generate_figure.py output (same palette, same
two-pass quadrant-aligned spider label placement, same filled boxplot) —
reproduced directly against this app's own test_cases.py/results, not
vendored code.

Usage:
    python tests/llm/plot_benchmark.py                            # latest run dir
    python tests/llm/plot_benchmark.py results/benchmark/20260908_170000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from test_cases import TEST_CASES

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "benchmark"
SAVE_DPI = 150

# Seaborn "muted" palette, same colors cardiagent's figures used.
PALETTE = ["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    "font.size": 9,
    "axes.linewidth": 0.8,
})


def latest_run_dir() -> Path:
    runs = sorted((d for d in RESULTS_DIR.iterdir() if d.is_dir()), reverse=True) if RESULTS_DIR.exists() else []
    if not runs:
        raise SystemExit(f"No run directories under {RESULTS_DIR} — run benchmark.py first.")
    return runs[0]


def load_run(run_dir: Path) -> dict[str, list[dict]]:
    """model -> its result records, from each model's <slug>.json file."""
    by_model = {}
    for f in sorted(run_dir.glob("*.json")):
        if f.name == "summary.json":
            continue
        records = json.loads(f.read_text())
        if records:
            by_model[records[0]["model"]] = records
    return by_model


def _category_examples() -> dict[str, str]:
    """One representative prompt per category, straight from test_cases.py —
    no separate catalog of axis labels/examples to keep in sync by hand."""
    examples = {}
    for tc in TEST_CASES:
        examples.setdefault(tc.category, tc.prompt)
    return examples


def _pass_rates_by_category(records: list[dict], categories: list[str]) -> list[float]:
    by_cat = defaultdict(lambda: [0, 0])
    for r in records:
        by_cat[r["category"]][1] += 1
        by_cat[r["category"]][0] += int(r["passed"])
    return [100 * by_cat[c][0] / by_cat[c][1] if by_cat[c][1] else 0.0 for c in categories]


def _model_size_key(name: str) -> tuple[float, str]:
    """Sorts by parameter count parsed out of the tag (e.g. 'qwen3:14b' -> 14,
    'gpt-oss:20b' -> 20); unrecognized tags sort last, alphabetically."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name, re.IGNORECASE)
    return (float(m.group(1)) if m else float("inf"), name)


def _sorted_models(by_model: dict[str, list[dict]]) -> list[str]:
    return sorted(by_model, key=_model_size_key)


def _axis_alignment(angle_rad: float) -> tuple[str, str]:
    """(ha, va) for a label placed at polar (angle, r), N=0 clockwise: labels
    on the right half get ha='left' so their text runs further right (away
    from the plot), and vice versa — same for top/bottom via va."""
    x_comp = np.sin(angle_rad)
    y_comp = np.cos(angle_rad)
    ha = "center" if abs(x_comp) < 0.30 else ("left" if x_comp > 0 else "right")
    va = "center" if abs(y_comp) < 0.30 else ("bottom" if y_comp > 0 else "top")
    return ha, va


def plot_spider(by_model: dict[str, list[dict]], out_path: Path):
    categories = sorted({r["category"] for records in by_model.values() for r in records})
    examples = _category_examples()
    models = _sorted_models(by_model)
    n = len(categories)

    # Offset by half a step so no axis lands exactly at North — keeps the
    # top gridline label clear and matches the reference figure's layout.
    half_step = np.pi / n
    angles = [(half_step + i * 2 * np.pi / n) % (2 * np.pi) for i in range(n)]
    angles_closed = angles + angles[:1]

    size = max(11, 8 + 0.55 * n)
    fig, ax = plt.subplots(figsize=(size, size * 0.85), subplot_kw=dict(polar=True))
    fig.set_dpi(SAVE_DPI)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_facecolor("#FAFAFA")
    ax.spines["polar"].set_color("#CCCCCC")
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=10, color="#888888")
    ax.set_rlabel_position(0)  # sits in the axis-free gap at North
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, linestyle="--")
    ax.xaxis.grid(True, color="#CCCCCC", linewidth=0.6, linestyle="-")
    ax.set_xticks(angles)
    ax.set_xticklabels([""] * n)

    for mi, model in enumerate(models):
        values = _pass_rates_by_category(by_model[model], categories)
        values_closed = values + values[:1]
        color = PALETTE[mi % len(PALETTE)]
        ax.plot(angles_closed, values_closed, linewidth=2.0, color=color, zorder=4)
        ax.fill(angles_closed, values_closed, color=color, alpha=0.10, zorder=3)
        ax.scatter(angles, values, s=50, color=color, zorder=5, linewidths=1.0, edgecolors="white")

    # Pass 1 — category name labels, quadrant-aligned so text runs away from
    # the plot rather than always centering (which collides between axes).
    name_texts = []
    for angle, cat in zip(angles, categories):
        label = cat.replace("_", " ").title()
        ha, va = _axis_alignment(angle)
        t = ax.text(angle, 108, label, ha=ha, va=va, fontsize=17, fontweight="bold",
                    color="#222222", linespacing=1.25, clip_on=False)
        name_texts.append((t, ha, cat))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w_px = fig.get_figwidth() * SAVE_DPI
    fig_h_px = fig.get_figheight() * SAVE_DPI

    # Pass 2 — example prompts, placed from the actual rendered bbox of each
    # name label (not a guessed radius), so they never collide with it.
    example_artists = []
    for t, ha, cat in name_texts:
        example = examples.get(cat)
        if not example:
            continue
        bbox = t.get_window_extent(renderer)
        ex_y_px = bbox.y0 - 6
        ex_x_px = bbox.x0 if ha == "left" else (bbox.x1 if ha == "right" else (bbox.x0 + bbox.x1) / 2)
        wrapped = "\n".join(textwrap.wrap(f'"{example}"', width=26))
        et = fig.text(ex_x_px / fig_w_px, ex_y_px / fig_h_px, wrapped, ha=ha, va="top",
                      fontsize=12, color="#999999", fontstyle="italic", linespacing=1.2)
        example_artists.append(et)

    # Pass 3 — legend, placed below the lowest example text actually drawn,
    # so it can never overlap the labels regardless of how much room they need.
    fig.canvas.draw()
    renderer2 = fig.canvas.get_renderer()
    min_y_px = min((et.get_window_extent(renderer2).y0 for et in example_artists), default=fig_h_px * 0.1)
    legend_y_frac = max(0.02, (min_y_px - 20) / fig_h_px)

    handles = [mpatches.Patch(color=PALETTE[mi % len(PALETTE)], label=m) for mi, m in enumerate(models)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, legend_y_frac),
               ncol=min(len(models), 4), fontsize=11, framealpha=0.95, edgecolor="#cccccc", frameon=True)

    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_latency(by_model: dict[str, list[dict]], out_path: Path) -> bool:
    models = _sorted_models(by_model)
    data = [[r["elapsed_s"] for r in by_model[m] if r.get("elapsed_s") is not None] for m in models]
    if not any(data):
        return False

    n = len(models)
    step = 0.65
    positions = [1 + i * step for i in range(n)]
    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(2.2, 0.5 + n * step + 0.5), 4.2))
    bp = ax.boxplot(
        data, positions=positions, widths=step, patch_artist=True,
        showfliers=False,  # outliers otherwise blow out the y-range and shrink the boxes
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(color="#555555", linewidth=1.0),
        capprops=dict(color="#555555", linewidth=1.0),
        zorder=3,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    ax.set_xticks(positions)
    ax.set_xticklabels(models, fontsize=8.5, rotation=35, ha="right")
    ax.set_xlim(positions[0] - step, positions[-1] + step)
    ax.set_ylabel("Response time (s)", fontsize=8.5)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)
    return True


def generate(run_dir: Path) -> list[Path]:
    by_model = load_run(run_dir)
    if not by_model:
        raise SystemExit(f"No per-model result files found in {run_dir}")

    written = [run_dir / "pass_rate.png"]
    plot_spider(by_model, written[0])
    latency_path = run_dir / "latency_boxplot.png"
    if plot_latency(by_model, latency_path):
        written.append(latency_path)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?", help="Run directory (default: latest under results/benchmark/)")
    args = p.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    for path in generate(run_dir):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
