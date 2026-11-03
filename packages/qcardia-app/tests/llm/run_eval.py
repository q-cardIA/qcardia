#!/usr/bin/env python
"""Eval runner for qcardia-app-dev's chat-assistant tool-calling.

Requires a local Ollama server reachable with the model pulled (default: the
app's, utils/llm_devices.default_model()) — exercises the real prompt/tools
against a live model, not a unit test, not meant to run in CI. See README.md.

Usage:
    python tests/llm/run_eval.py --dry-run
    python tests/llm/run_eval.py --category preset
    python tests/llm/run_eval.py --ids preset_neg_001,preset_neg_002
    python tests/llm/run_eval.py --runs 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from driver import build_controller, render_dry_run, run_one, screen_after  # noqa: E402
from test_cases import TEST_CASES  # noqa: E402
from validator import validate  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _select_cases(args):
    cases = TEST_CASES
    if args.category:
        wanted = set(args.category.split(","))
        cases = [c for c in cases if c.category in wanted]
    if args.ids:
        wanted_ids = set(args.ids.split(","))
        cases = [c for c in cases if c.id in wanted_ids]
    return cases


def _dry_run(cases):
    ok, failed = 0, 0
    for tc in cases:
        try:
            rendered = render_dry_run(tc)
            n_tools = len(rendered["tools"])
            print(f"  [ok] {tc.id:24s} ({tc.category}) — {n_tools} tools offered, "
                  f"{len(rendered['messages'])} messages")
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {tc.id:24s} — {type(e).__name__}: {e}")
            failed += 1
    print(f"\ndry-run: {ok} ok, {failed} failed (fixture/render errors only, no Ollama call made)")
    return 0 if failed == 0 else 1


def _check_result_text(tc, cc, run_result) -> tuple[bool, str]:
    """For cases with result_contains set, execute the matching tool call for
    real and check its result string; others skip this entirely."""
    if not tc.result_contains:
        return True, ""
    wanted_name = tc.result_contains["tool_name"]
    substr = tc.result_contains["contains"]
    match = next((c for c in run_result.tool_calls if c["name"] == wanted_name), None)
    if match is None:
        return False, f"expected a '{wanted_name}' call to check its result text, none was made"
    result_text = cc._execute_tool(match["name"], match["arguments"])
    if substr not in result_text:
        return False, f"result text {result_text!r} does not contain {substr!r}"
    return True, ""


def _run_live(cases, runs: int, think: bool | None, model: str | None = None):
    results = []
    by_category = defaultdict(lambda: {"pass": 0, "fail": 0})

    for tc in cases:
        for run_idx in range(runs):
            cc = build_controller(tc, **({"llm_model": model} if model else {}))
            before = screen_after(build_controller(tc))
            try:
                run_result = run_one(cc, tc.prompt, think=think, history=tc.history)
            except Exception as e:
                print(f"  [ERROR] {tc.id} run {run_idx + 1}/{runs}: {type(e).__name__}: {e}")
                by_category[tc.category]["fail"] += 1
                results.append({"id": tc.id, "category": tc.category, "run": run_idx, "passed": False,
                                 "error": str(e)})
                continue

            if run_result.error:
                print(f"  [ERROR] {tc.id} run {run_idx + 1}/{runs}: {run_result.error}")
                by_category[tc.category]["fail"] += 1
                results.append({"id": tc.id, "category": tc.category, "run": run_idx, "passed": False,
                                 "error": run_result.error})
                continue

            screen = screen_after(cc)
            vres = validate(tc, run_result.tool_calls, screen, before)
            passed, reason = vres.passed, vres.failure_summary
            if passed:
                passed, reason = _check_result_text(tc, cc, run_result)

            status = "PASS" if passed else "FAIL"
            by_category[tc.category]["pass" if passed else "fail"] += 1
            print(f"  [{status}] {tc.id:24s} ({tc.category}) run {run_idx + 1}/{runs} "
                  f"— {run_result.elapsed_s:.1f}s" + (f" — {reason}" if reason else ""))

            results.append({
                "id": tc.id, "category": tc.category, "run": run_idx, "passed": passed,
                "prompt": tc.prompt, "expected": tc.expected_tool_calls, "actual": run_result.tool_calls,
                "elapsed_s": run_result.elapsed_s, "reason": reason,
            })

    return results, by_category


def _print_report(by_category: dict):
    print("\n" + "=" * 60)
    total_pass = total_fail = 0
    for cat in sorted(by_category):
        p, f = by_category[cat]["pass"], by_category[cat]["fail"]
        total_pass += p
        total_fail += f
        n = p + f
        rate = (p / n * 100) if n else 0.0
        print(f"  {cat:20s} {p:3d}/{n:<3d} ({rate:5.1f}%)")
    n_total = total_pass + total_fail
    overall = (total_pass / n_total * 100) if n_total else 0.0
    print("-" * 60)
    print(f"  {'OVERALL':20s} {total_pass:3d}/{n_total:<3d} ({overall:5.1f}%)")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", help="Comma-separated category filter, e.g. preset,load")
    p.add_argument("--ids", help="Comma-separated test id filter")
    p.add_argument("--runs", type=int, default=1, help="Repeat each case N times (default 1)")
    p.add_argument("--dry-run", action="store_true", help="Render prompts/tools only, no Ollama call")
    p.add_argument("--device", help="A device id from config/llm_devices.json, e.g. spike (default: this computer)")
    p.add_argument("--model", help="The model on that device (default: the app's default)")
    p.add_argument("--out", help="Results file name in results/ (default: a timestamp)")
    think_group = p.add_mutually_exclusive_group()
    think_group.add_argument("--think", action="store_true", dest="think", default=None)
    think_group.add_argument("--no-think", action="store_false", dest="think")
    args = p.parse_args()

    cases = _select_cases(args)
    if not cases:
        print("No test cases matched the given --category/--ids filter.")
        return 1
    print(f"Selected {len(cases)} test case(s).\n")

    if args.dry_run:
        return _dry_run(cases)

    if args.device:
        from utils import llm_devices
        device = next((d for d in llm_devices.load_devices() if d.id == args.device), None)
        probe = llm_devices.activate(device, timeout=30) if device else None
        if probe is None or not probe.ok:
            print(f"Device {args.device!r} is not available: {getattr(probe, 'detail', 'unknown id')}")
            return 1
    results, by_category = _run_live(cases, runs=args.runs, think=args.think, model=args.model)
    _print_report(by_category)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / (args.out or f"results_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out_path.write_text(json.dumps({"version": 1, "results": results}, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
