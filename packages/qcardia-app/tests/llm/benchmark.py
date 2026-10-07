#!/usr/bin/env python
"""Sweeps the core eval (see test_cases.py/run_eval.py) across several
models and prints one summary table — which model available on this
machine currently does best on this app's tool-calling.

Each model runs in its own subprocess, supervised by the parent: a network
call to Ollama can sit stalled indefinitely with an established connection
and zero bytes flowing (observed live — a `requests` stream timeout doesn't
reliably bound this), so the parent watches for real progress instead of
trusting the child to time itself out. If a model produces no new completed
case within --stall-timeout, the parent kills it and moves on, keeping
whatever partial results it already flushed.

Each run gets its own directory under results/benchmark/<timestamp>/,
containing every model's log/progress/result files, a summary.txt +
summary.json, and (via plot_benchmark.py, unless --no-plot) pass_rate.png
+ latency_boxplot.png — one self-contained folder per comparison.

Usage:
    python tests/llm/benchmark.py                          # every model pulled locally, 1 run
    python tests/llm/benchmark.py --models qwen3:8b,gpt-oss:20b --runs 3
    python tests/llm/benchmark.py --models qwen3.6:latest --stall-timeout 300
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from driver import build_controller, run_one, screen_after  # noqa: E402
from test_cases import TEST_CASES  # noqa: E402
from utils.app_config import OLLAMA_MODEL, list_ollama_models  # noqa: E402
from validator import validate  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "benchmark"
DEFAULT_STALL_TIMEOUT_S = 600  # 10 minutes with no new completed case = abort this model


def _new_run_dir() -> Path:
    return RESULTS_DIR / time.strftime("%Y%m%d_%H%M%S")


def _default_models() -> list[str]:
    """No --models given: use whatever's actually pulled on this machine
    (not a hardcoded list, so this runs unmodified on any machine) — some
    may not support tool-calling and will just fail/score 0, that's fine
    for a manual sweep. Falls back to OLLAMA_MODEL if Ollama isn't reachable."""
    tags = [m["tag"] for m in list_ollama_models()]
    return tags or [OLLAMA_MODEL]


def _slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def _progress_path(model: str, run_dir: Path) -> Path:
    return run_dir / f"{_slug(model)}.progress.jsonl"


def _run_one_model(model: str, cases: list, runs: int, run_dir: Path) -> list[dict]:
    """Runs every case for one model, appending each result to its progress
    file as soon as it completes (not just at the end) — this is what lets
    a supervisor detect a stall instead of only a total-time cutoff."""
    progress_path = _progress_path(model, run_dir)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    records = []

    with open(progress_path, "w") as f:
        for tc in cases:
            for run_idx in range(runs):
                cc = build_controller(tc, llm_model=model)
                try:
                    result = run_one(cc, tc.prompt, history=tc.history)
                    screen = screen_after(cc, result.tool_calls) if tc.match_mode == "state" else None
                    passed = not result.error and validate(tc, result.tool_calls, screen).passed
                    elapsed = result.elapsed_s
                except Exception as e:
                    passed, elapsed = False, None
                    print(f"  [ERROR] {tc.id} run {run_idx + 1}/{runs}: {type(e).__name__}: {e}")

                record = {
                    "model": model, "test_id": tc.id, "category": tc.category,
                    "run": run_idx, "passed": passed, "elapsed_s": elapsed,
                }
                records.append(record)
                f.write(json.dumps(record) + "\n")
                f.flush()

    n_pass = sum(r["passed"] for r in records)
    print(f"  {model}: {n_pass}/{len(records)} ({100 * n_pass / len(records):.1f}%)")
    return records


def _read_progress(model: str, run_dir: Path) -> list[dict]:
    path = _progress_path(model, run_dir)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_model_supervised(model: str, runs: int, stall_timeout: int, run_dir: Path) -> tuple[list[dict], bool]:
    """Runs one model in a child process (`python benchmark.py --models
    <model> ...`) and kills it if `_progress_path(model, run_dir)` stops
    growing for `stall_timeout` seconds. Returns (records, completed) —
    completed=False means the child was killed mid-run; records holds
    whatever it flushed before that."""
    progress_path = _progress_path(model, run_dir)
    progress_path.unlink(missing_ok=True)

    argv = [sys.executable, str(Path(__file__).resolve()), "--models", model, "--runs", str(runs),
            "--run-dir", str(run_dir), "--_bare"]
    log_path = run_dir / f"{_slug(model)}.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)

    last_size, last_change = -1, time.monotonic()
    while proc.poll() is None:
        time.sleep(5)
        size = progress_path.stat().st_size if progress_path.exists() else 0
        if size != last_size:
            last_size, last_change = size, time.monotonic()
        elif time.monotonic() - last_change > stall_timeout:
            print(f"  [STALLED] {model}: no progress for {stall_timeout}s — aborting, keeping partial results")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            return _read_progress(model, run_dir), False

    records = _read_progress(model, run_dir)
    n_pass = sum(r["passed"] for r in records)
    if records:
        print(f"  {model}: {n_pass}/{len(records)} ({100 * n_pass / len(records):.1f}%)")
    return records, True


def _categorize(records: list[dict]) -> dict:
    by_category = defaultdict(lambda: [0, 0])
    for r in records:
        by_category[r["category"]][1] += 1
        by_category[r["category"]][0] += int(r["passed"])
    return dict(by_category)


def _format_summary(all_by_category: dict[str, dict]) -> str:
    categories = sorted({c for cats in all_by_category.values() for c in cats})
    models = list(all_by_category)

    header = f"  {'category':20s}" + "".join(f"{m:>16s}" for m in models)
    lines = ["=" * len(header), header, "-" * len(header)]
    for cat in categories:
        row = f"  {cat:20s}"
        for m in models:
            p, t = all_by_category[m].get(cat, (0, 0))
            row += f"{f'{p}/{t}':>16s}" if t else f"{'n/a':>16s}"
        lines.append(row)
    lines.append("-" * len(header))
    total_row = f"  {'OVERALL':20s}"
    for m in models:
        p = sum(v[0] for v in all_by_category[m].values())
        t = sum(v[1] for v in all_by_category[m].values())
        total_row += f"{f'{p}/{t} ({100 * p / t:.0f}%)':>16s}" if t else f"{'n/a':>16s}"
    lines.append(total_row)
    lines.append("=" * len(header))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", help="Comma-separated Ollama tags (default: this machine's current lineup)")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--stall-timeout", type=int, default=DEFAULT_STALL_TIMEOUT_S,
                    help=f"Abort a model if it produces no new completed case for this many seconds (default {DEFAULT_STALL_TIMEOUT_S})")
    p.add_argument("--_bare", action="store_true", help=argparse.SUPPRESS)  # internal: run in-process, no supervision
    p.add_argument("--run-dir", help=argparse.SUPPRESS)  # internal: supervised child's run directory
    p.add_argument("--no-plot", action="store_true", help="Skip generating pass_rate.png/latency_boxplot.png")
    args = p.parse_args()

    models = args.models.split(",") if args.models else _default_models()
    cases = TEST_CASES

    if args._bare:
        # Invoked by run_model_supervised() as the supervised child — just do
        # the work, no subprocess/supervision recursion.
        run_dir = Path(args.run_dir)
        for model in models:
            _run_one_model(model, cases, args.runs, run_dir)
        return

    run_dir = _new_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Benchmarking {len(models)} model(s) over {len(cases)} case(s) x {args.runs} run(s), "
          f"stall timeout {args.stall_timeout}s. Writing to {run_dir}\n")

    all_records: list[dict] = []
    all_by_category: dict[str, dict] = {}
    incomplete: list[str] = []

    for model in models:
        print(f"=== {model} ===")
        records, completed = run_model_supervised(model, args.runs, args.stall_timeout, run_dir)
        if not completed:
            incomplete.append(model)
        all_records.extend(records)
        all_by_category[model] = _categorize(records)
        (run_dir / f"{_slug(model)}.json").write_text(json.dumps(records, indent=2))

    summary = _format_summary(all_by_category)
    print("\n" + summary)
    if incomplete:
        incomplete_note = f"\nIncomplete (stalled, partial results only): {', '.join(incomplete)}"
        print(incomplete_note)
        summary += "\n" + incomplete_note

    (run_dir / "summary.txt").write_text(summary + "\n")
    (run_dir / "summary.json").write_text(json.dumps(all_records, indent=2))
    print(f"\nWrote {run_dir}/summary.txt and summary.json")

    if not args.no_plot:
        try:
            from plot_benchmark import generate as generate_plots
            for path in generate_plots(run_dir):
                print(f"Wrote {path}")
        except Exception as e:
            print(f"\n[plots skipped] {type(e).__name__}: {e} — run "
                  f"`python tests/llm/plot_benchmark.py {run_dir}` manually once fixed.")


if __name__ == "__main__":
    main()
