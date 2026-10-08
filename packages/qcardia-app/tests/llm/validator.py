"""Checks actual tool calls against a TestCase's expected calls/match_mode.
Pure Python, no I/O."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from test_cases import TestCase


@dataclass
class CallMatch:
    """Outcome of matching one expected call against actual output."""
    expected: dict
    matched: bool
    actual_matched: Optional[dict] = None
    reason: str = ""


@dataclass
class ValidationResult:
    test_id: str
    passed: bool
    match_mode: str
    expected: List[dict]
    actual: List[dict]
    call_matches: List[CallMatch]
    failure_summary: str = ""


def _args_match(expected_args: dict, actual_args: dict) -> tuple[bool, str]:
    """Every key in expected_args must match in actual_args; extra actual keys are ignored."""
    for key, exp_val in expected_args.items():
        if key not in actual_args:
            return False, f"argument '{key}' missing in actual"
        act_val = actual_args[key]
        if exp_val != act_val:
            return False, f"argument '{key}': expected {exp_val!r}, got {act_val!r}"
    return True, ""


def _calls_match(expected: dict, actual: dict) -> tuple[bool, str]:
    """Return (match, reason) for one expected vs one actual tool call."""
    if expected.get("name") != actual.get("name"):
        return False, f"name mismatch: expected '{expected.get('name')}', got '{actual.get('name')}'"
    return _args_match(expected.get("arguments", {}), actual.get("arguments", {}))


def _find_and_remove(expected_call: dict, pool: list) -> tuple[bool, Optional[dict], str]:
    """Find the first matching call in pool, remove it, return (found, matched_call, reason)."""
    for i, candidate in enumerate(pool):
        ok, reason = _calls_match(expected_call, candidate)
        if ok:
            pool.pop(i)
            return True, candidate, ""
    if pool:
        _, reason = _calls_match(expected_call, pool[0])
    else:
        reason = "no remaining actual calls"
    return False, None, reason


def validate(test_case: TestCase, actual_tool_calls: list, screen: dict | None = None,
             before: dict | None = None) -> ValidationResult:
    """actual_tool_calls: raw list of {"name": ..., "arguments": ...} dicts. screen: the viewer state
    after those calls (driver.screen_after), needed by match_mode "state". With before, the state at
    the start, match_mode "none" passes when nothing changed: a tool that refused is no harm."""
    expected = test_case.expected_tool_calls
    mode = test_case.match_mode

    actual = [c for c in actual_tool_calls if isinstance(c, dict)]
    # A lookup is a free step toward the answer, scored only by a case that expects it.
    if not any(e.get("name") == "list_subjects" for e in expected):
        actual = [c for c in actual if c.get("name") != "list_subjects"]

    if mode == "none" and before is not None and screen is not None:
        changed = sorted(k for k in screen if screen[k] != before.get(k))
        problem = f"changed: {', '.join(changed)}" if changed else ""
        return ValidationResult(test_case.id, not changed, mode, expected, actual, [], problem)
    if mode == "state":
        problem = test_case.expected_state(screen)
        return ValidationResult(test_case.id, problem is None, mode, expected, actual, [], problem or "")
    if mode == "exact":
        return _validate_exact(test_case.id, mode, expected, actual)
    elif mode == "any_order":
        if len(actual) != len(expected):
            summary = f"length mismatch: expected {len(expected)}, got {len(actual)}"
            matches = [CallMatch(expected=e, matched=False, reason=summary) for e in expected]
            return ValidationResult(test_case.id, False, mode, expected, actual, matches, summary)
        return _match_each(test_case.id, mode, expected, actual, "some expected calls not found in actual")
    elif mode == "subset":
        return _match_each(test_case.id, mode, expected, actual, "some expected calls missing from actual")
    elif mode == "none":
        return _validate_none(test_case.id, mode, expected, actual)
    else:
        raise ValueError(f"Unknown match_mode: {mode!r}")


def _validate_exact(test_id, mode, expected, actual) -> ValidationResult:
    matches = []
    if len(actual) != len(expected):
        summary = f"length mismatch: expected {len(expected)}, got {len(actual)}"
        for e in expected:
            matches.append(CallMatch(expected=e, matched=False, reason=summary))
        return ValidationResult(test_id, False, mode, expected, actual, matches, summary)

    all_ok = True
    for e, a in zip(expected, actual):
        ok, reason = _calls_match(e, a)
        matches.append(CallMatch(expected=e, matched=ok, actual_matched=a if ok else None, reason=reason))
        if not ok:
            all_ok = False

    summary = "" if all_ok else f"call mismatch at position(s): {[i for i, m in enumerate(matches) if not m.matched]}"
    return ValidationResult(test_id, all_ok, mode, expected, actual, matches, summary)


def _match_each(test_id, mode, expected, actual, fail_summary) -> ValidationResult:
    """Shared by any_order (pre-checked equal length) and subset (extras OK):
    find-and-remove each expected call from a copy of actual, in order."""
    pool = list(actual)
    matches = []
    all_ok = True
    for e in expected:
        found, matched_call, reason = _find_and_remove(e, pool)
        matches.append(CallMatch(expected=e, matched=found, actual_matched=matched_call, reason=reason))
        all_ok = all_ok and found
    return ValidationResult(test_id, all_ok, mode, expected, actual, matches, "" if all_ok else fail_summary)


def _validate_none(test_id, mode, expected, actual) -> ValidationResult:
    if actual:
        summary = f"expected no tool calls, got {len(actual)}"
        return ValidationResult(test_id, False, mode, expected, actual, [], summary)
    return ValidationResult(test_id, True, mode, expected, actual, [], "")
