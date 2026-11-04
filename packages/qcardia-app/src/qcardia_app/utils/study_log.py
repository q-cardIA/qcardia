"""User-study log (issue #26): one JSON line per event, in <repo>/logs/<start time>_<participant>_<condition>/.

Every record has the wall-clock time, "ms" since the session started, the participant, the condition and
the event name, plus "by" for actions not taken by the user (see acting). log() does nothing until
start_session() is called. Anything with a duration is a Span, logged as "<name>_start" and "<name>_end".
"""
from __future__ import annotations

import json
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from qcardia_app.utils.app_config import APP_DIR

LOG_DIR = APP_DIR / "logs"

_lock = threading.Lock()
_file = None
_t0 = 0.0
_session: dict = {}
_actor: str | None = None
_task = 0
_in_task = False
_screen: dict = {}  # the last state logged, see log_state


def start_session(participant: str, condition: str = "", **fields) -> Path:
    global _file, _t0, _session, _task, _in_task, _screen
    _session = {"participant": participant.strip() or "anonymous", "condition": condition.strip()}
    stem = "_".join(re.sub(r"[^\w.-]+", "-", v) for v in _session.values() if v)
    folder = LOG_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{stem}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{folder.name}.jsonl"
    _file = open(path, "a", encoding="utf-8", buffering=1)
    _t0, _task, _in_task, _screen = time.monotonic(), 0, False, {}
    log("session_start", **fields)
    return path


def end_session() -> Path | None:
    """Closes the log and returns its path, or None if no session was running."""
    global _file
    if _file is None:
        return None
    if _in_task:
        toggle_task()
    log("session_end")
    _file.close()
    path, _file = Path(_file.name), None
    return path


def log(event: str, **fields):
    if _file is None:
        return
    record = {"time": datetime.now().isoformat(timespec="milliseconds"), "ms": round((time.monotonic() - _t0) * 1000),
              **_session, "event": event, **({"by": _actor} if _actor else {}), **fields}
    with _lock:
        _file.write(json.dumps(record, default=str) + "\n")


def log_state(state: dict):
    """Logs the screen as "state": whole the first time, then only the fields that changed. Of a dict
    field only its changed entries while none was removed, and of "panels" only the changed panels by
    number while the layout keeps its panel count."""
    global _screen
    if _file is None:
        return
    changed = {}
    for key, new in state.items():
        old = _screen.get(key)
        if new == old:
            continue
        if isinstance(new, dict) and isinstance(old, dict) and old.keys() <= new.keys():
            new = {k: v for k, v in new.items() if old.get(k) != v}
        elif key == "panels" and old is not None and len(new) == len(old):
            new = {i: p for i, (p, q) in enumerate(zip(new, old)) if p != q}
        changed[key] = new
    _screen = state
    if changed:
        log("state", **changed)


@contextmanager
def acting(actor: str):
    """Marks everything logged inside the block as done by actor, e.g. a plugin, instead of the user."""
    global _actor
    previous, _actor = _actor, actor
    try:
        yield
    finally:
        _actor = previous


def toggle_task():
    global _task, _in_task
    if not _in_task:
        _task += 1
    log("task_end" if _in_task else "task_start", task=_task)
    _in_task = not _in_task


class Span:
    """Something with a duration whose start and end are logged in different places."""

    def __init__(self, name: str, **fields):
        self.name, self._t, self._logged = name, time.monotonic(), set()
        self.fields = {**({"by": _actor} if _actor else {}), **fields}
        log(f"{name}_start", **self.fields)

    def mark(self, step: str, **result):
        """Logs '<name>_<step>' with the time since the start, once per step."""
        if step in self._logged:
            return
        self._logged.add(step)
        log(f"{self.name}_{step}", duration_ms=round((time.monotonic() - self._t) * 1000), **self.fields, **result)

    def end(self, **result):
        self.mark("end", **result)
