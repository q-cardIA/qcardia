#!/usr/bin/env python
"""Review tool for qcardia cardisort classification + LLM disambiguation.

One page, one section per subject, five collapsible subsections walking the
full pipeline for that subject:

  1. raw       — every series found, unlabeled, with its DICOM metadata
  2. cardisort — same table + cardisort's predicted (sequence, plane), sortable
  3. classify  — per series: cardisort's top-3 candidates per head with
                 probabilities, and the LLM's confirm/correct decision + why
                 (or "not activated" if cardisort was confident enough that
                 qcardia.refine_classification never called it)
  4. disambig  — per duplicate-class group: every candidate's LLM-assigned
                 role (primary/repeat/other, or e.g. rest/stress for classes
                 with a custom role vocabulary) + reasoning
  5. resolved  — the final key -> series mapping that would actually load
                 into the app

Stages:
  catalog          (slow, once)     cardisort over raw subject dirs -> catalog.json
  run --name X     (fast, iterate)  replay disambiguation over the catalog -> runs/X.json
  serve            build the page live + serve it (image preview included)

Typical workflow:
    uv run python bench.py catalog --subjects 1-15
    uv run python bench.py run --name baseline
    uv run python bench.py serve
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("bench")

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]  # benchmarks/disambiguation -> qcardia -> Documents
DEFAULT_DATA_ROOT = WORKSPACE / "data" / "new_data"
DEFAULT_WEIGHTS = WORKSPACE / "models" / "weights" / "cardisort"

CATALOG_PATH = HERE / "catalog.json"
RUNS_DIR = HERE / "runs"

# Human review notes — corrections to cardisort's class and free-text
# comments on the LLM steps. Kept in its own file, never touched by
# `catalog`/`run`, so re-cataloguing or trying a different disambiguation
# run never loses annotations. Corrections (section 2) are keyed by
# "subject||series_dir" and apply regardless of which run you're looking
# at — they're about the raw data, not about one run's LLM output.
# Per-run comments (sections 3/4) live under runs.<run_name> instead, since
# they're about that run's specific reasoning — switching --run shows a
# different, run-scoped set of comments rather than mixing them together.
ANNOTATIONS_PATH = HERE / "annotations.json"


def _load_annotations() -> dict:
    if not ANNOTATIONS_PATH.exists():
        return {"corrections": {}, "runs": {}}
    data = json.loads(ANNOTATIONS_PATH.read_text())
    data.setdefault("corrections", {})
    data.setdefault("runs", {})
    return data


def _save_annotations_merge(incoming: dict) -> dict:
    """Merges incoming (from one page's Save click, which only covers the
    corrections + the single run currently being viewed) into the full file
    on disk, so saving while viewing run A never drops run B's comments."""
    current = _load_annotations()
    current["corrections"].update(incoming.get("corrections", {}))
    for run_name, run_data in incoming.get("runs", {}).items():
        current["runs"].setdefault(run_name, {})
        for scope, rows in run_data.items():
            current["runs"][run_name].setdefault(scope, {}).update(rows)
    ANNOTATIONS_PATH.write_text(json.dumps(current, indent=2))
    return current


# ---------------------------------------------------------------------------
# subject discovery
# ---------------------------------------------------------------------------

def _subject_sort_key(p: Path):
    m = re.search(r"(\d+)", p.name)
    return int(m.group(1)) if m else 0


def find_subjects(data_root: Path, subjects_arg: str) -> list[Path]:
    """Raw subject folders (QLGE<NN>_), excluding CardiSorted_* outputs."""
    raw = [
        d for d in data_root.iterdir()
        if d.is_dir() and not d.name.startswith("CardiSorted_") and re.match(r"QLGE\d+_?$", d.name)
    ]
    raw.sort(key=_subject_sort_key)
    if subjects_arg.strip().lower() == "all":
        return raw

    wanted: set[int] = set()
    for part in subjects_arg.replace(",", " ").split():
        if "-" in part:
            lo, hi = part.split("-")
            wanted.update(range(int(lo), int(hi) + 1))
        else:
            wanted.add(int(part))
    return [d for d in raw if _subject_sort_key(d) in wanted]


# ---------------------------------------------------------------------------
# Stage: catalog — cardisort over raw dirs (cached)
# ---------------------------------------------------------------------------

def cmd_catalog(args):
    from qcardia.cardisort import (
        classify_sequence_dir,
        get_sequence_dirs,
        load_cardisort_model,
        load_series_datasets,
    )
    from qcardia.disambiguate import summarize_sequence_dir

    data_root = Path(args.data_root)
    subjects = find_subjects(data_root, args.subjects)
    if not subjects:
        sys.exit(f"No subjects matched {args.subjects!r} in {data_root}")

    log.info("Loading cardisort model from %s", args.weights)
    model, config = load_cardisort_model(Path(args.weights))
    nch = config["model"]["nr_input_channels"]
    pd = tuple(config["data"]["target_pixdim"])
    ts = tuple(config["data"]["target_size"])
    gm = config["data"]["image_grid_sample_mode"]

    catalog: dict = {
        "meta": {
            "created": datetime.now().isoformat(timespec="seconds"),
            "weights": str(args.weights),
            "data_root": str(data_root),
            "n_subjects": len(subjects),
        },
        "subjects": {},
    }

    existing = {}
    if CATALOG_PATH.exists() and args.merge:
        existing = json.loads(CATALOG_PATH.read_text()).get("subjects", {})
        catalog["subjects"].update(existing)

    for si, subj in enumerate(subjects, 1):
        if args.merge and subj.name in existing:
            log.info("[%d/%d] %s — already catalogued, skipping", si, len(subjects), subj.name)
            continue
        seq_dirs = get_sequence_dirs(subj)
        log.info("[%d/%d] %s — %d series", si, len(subjects), subj.name, len(seq_dirs))
        series_records = []
        t0 = time.time()
        for d in seq_dirs:
            datasets = load_series_datasets(d)
            if not datasets:
                series_records.append(
                    {"series_dir": d.name, "pred_seq": None, "pred_plane": None,
                     "seq_top3": None, "plane_top3": None, "summary": None}
                )
                continue
            summary = summarize_sequence_dir(d, datasets)
            try:
                pred = classify_sequence_dir(d, model, nch, pd, ts, gm, verbose=False)
            except Exception as exc:
                log.warning("   %s classify failed: %s", d.name, exc)
                pred = None
            if pred is None:
                pred_seq = pred_plane = seq_top3 = plane_top3 = None
            else:
                pred_seq, pred_plane = pred.sequence_name, pred.plane_name
                seq_top3, plane_top3 = pred.sequence_top3, pred.plane_top3
            series_records.append(
                {
                    "series_dir": d.name,
                    "pred_seq": pred_seq,
                    "pred_plane": pred_plane,
                    # top-3 (name, prob) per head — lets `run` replay the
                    # confidence-escalation LLM step without recomputing cardisort.
                    "seq_top3": seq_top3,
                    "plane_top3": plane_top3,
                    "summary": _jsonable(summary),
                }
            )
            log.info("   %-28s -> %s/%s", d.name, pred_seq, pred_plane)
        catalog["subjects"][subj.name] = series_records
        log.info("   done in %.0fs", time.time() - t0)

    CATALOG_PATH.write_text(json.dumps(catalog, indent=2))
    log.info("Wrote catalog: %s (%d subjects)", CATALOG_PATH, len(catalog["subjects"]))


def _jsonable(summary: dict) -> dict:
    """summarize_sequence_dir may return pydicom value types; coerce to plain JSON."""
    out = {}
    for k, v in summary.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


# ---------------------------------------------------------------------------
# Stage: confidence-escalation replay, from cached catalog data only — no
# CNN/DICOM reads. (Rest/stress and other class-specific role handling is
# NOT done here — it happens inside disambiguate_duplicates itself via
# qcardia.disambiguate.ROLE_SCHEMAS, once groups are formed below.)
# ---------------------------------------------------------------------------

def _refine_record_classification(record: dict) -> tuple[str, str, dict]:
    """Replays qcardia.refine_classification.refine_classification against
    one cached catalog record. Returns (seq, plane, note) where note
    describes whether/why the LLM escalation step fired, for the review
    UI's step-by-step view."""
    from qcardia.cardisort import ClassificationResult
    from qcardia.refine_classification import refine_classification

    note = {"escalated": False, "escalation_from": None, "escalation_to": None,
             "escalation_reasoning": None}

    seq_top3 = [tuple(x) for x in (record.get("seq_top3") or [])]
    plane_top3 = [tuple(x) for x in (record.get("plane_top3") or [])]
    result = ClassificationResult(
        sequence_name=record["pred_seq"],
        plane_name=record["pred_plane"],
        sequence_top3=seq_top3,
        plane_top3=plane_top3,
    )
    activated = result.is_sequence_uncertain or result.is_plane_uncertain
    note["activated"] = activated

    if activated:
        refined = refine_classification(
            Path(record["series_dir"]), record["summary"] or {}, result
        )
        if refined.escalation_reasoning is not None:
            note["escalated"] = True
            note["escalation_from"] = (result.sequence_name, result.plane_name)
            note["escalation_to"] = (refined.sequence_name, refined.plane_name)
            note["escalation_reasoning"] = refined.escalation_reasoning
        result = refined

    return result.sequence_name, result.plane_name, note


def _refine_and_group(records: list[dict]) -> tuple[dict[tuple, list[dict]], dict[str, dict]]:
    """Groups records by their confidence-escalation-refined (seq, plane).
    Also returns per-series refinement notes keyed by series_dir."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    notes: dict[str, dict] = {}
    for r in records:
        if r["pred_seq"] is None or r["pred_plane"] is None:
            continue
        seq, plane, note = _refine_record_classification(r)
        notes[r["series_dir"]] = note
        groups[(seq, plane)].append(r)
    return groups, notes


# ---------------------------------------------------------------------------
# Stage: run disambiguation over the cached catalog
# ---------------------------------------------------------------------------

def cmd_run(args):
    import qcardia.disambiguate as qd
    import qcardia.refine_classification as qr

    qd.OLLAMA_URL = qr.OLLAMA_URL = f"{args.ollama_url}/api/chat"
    qd.OLLAMA_MODEL = qr.OLLAMA_MODEL = args.model
    from qcardia.disambiguate import disambiguate_duplicates

    if not CATALOG_PATH.exists():
        sys.exit("No catalog.json — run `catalog` first.")
    catalog = json.loads(CATALOG_PATH.read_text())

    run = {
        "meta": {
            "name": args.name,
            "created": datetime.now().isoformat(timespec="seconds"),
            "model": args.model,
            "disambiguate_version": _disambiguate_version(qd),
        },
        "decisions": {},  # subject -> [ {group_key, seq, plane, members, chosen, method, assessments} ]
        "classification_notes": {},  # subject -> {series_dir: note} — confidence-escalation trail
    }

    n_llm = n_single = n_err = 0
    for subject, records in catalog["subjects"].items():
        groups, notes = _refine_and_group(records)
        run["classification_notes"][subject] = notes
        decisions = []
        for (seq, plane), members in groups.items():
            member_dirs = [m["series_dir"] for m in members]
            group_key = f"{seq}_{plane}"
            if len(members) == 1:
                decisions.append(
                    {"group_key": group_key, "seq": seq, "plane": plane, "members": member_dirs,
                     "chosen": member_dirs[0], "method": "single", "assessments": []}
                )
                n_single += 1
                continue
            candidates = [m["summary"] for m in members if m["summary"]]
            try:
                result = disambiguate_duplicates((seq, plane), candidates)
                chosen = result["primary_series"]
                method = "llm"
                assessments = result.get("assessments", [])
                n_llm += 1
                log.info("%s %-14s (%d) -> %s", subject, group_key, len(members), chosen)
            except Exception as exc:
                chosen = max(members, key=lambda r: (r["summary"] or {}).get("n_instances", 0))["series_dir"]
                method = "fallback_largest"
                assessments = [{"error": str(exc)}]
                n_err += 1
                log.warning("%s %-14s LLM failed (%s) -> fallback %s", subject, group_key, exc, chosen)
            decisions.append(
                {"group_key": group_key, "seq": seq, "plane": plane, "members": member_dirs,
                 "chosen": chosen, "method": method, "assessments": assessments}
            )
        run["decisions"][subject] = decisions

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{args.name}.json"
    out.write_text(json.dumps(run, indent=2))
    log.info("Wrote run: %s  (%d LLM groups, %d singletons, %d fallbacks)", out, n_llm, n_single, n_err)


def _disambiguate_version(qd) -> dict:
    """Fingerprint the disambiguation logic so runs are attributable to changes."""
    info = {}
    try:
        f = Path(qd.__file__)
        info["file_sha1"] = hashlib.sha1(f.read_bytes()).hexdigest()[:12]
        info["git_commit"] = subprocess.check_output(
            ["git", "-C", str(f.parent), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        pass
    return info


def _latest_run() -> Path | None:
    if not RUNS_DIR.exists():
        return None
    runs = sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


# ---------------------------------------------------------------------------
# review page — 5 sections per subject
# ---------------------------------------------------------------------------

def _esc(s) -> str:
    return (str(s) if s is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


RAW_COLS = [
    ("series_dir", "series"), ("series_description", "description"),
    ("series_number", "#"), ("n_instances", "instances"),
    ("rows", "rows"), ("columns", "cols"), ("slice_thickness", "slice thk"),
    ("acquisition_date", "acq. date"), ("acquisition_time", "acq. time"),
]


def _series_cell_attrs(subject: str, series_dir: str) -> str:
    return f'data-subject="{_esc(subject)}" data-series="{_esc(series_dir)}"'


def _comment_input(scope: str, row_key: str, value: str) -> str:
    return (
        f'<input type="text" class="ann-input" data-ann-scope="{_esc(scope)}" '
        f'data-ann-row="{_esc(row_key)}" data-ann-field="comment" '
        f'value="{_esc(value)}" placeholder="comment...">'
    )


def _select_input(scope: str, row_key: str, field: str, options: list[str], value: str) -> str:
    opts = ['<option value="">—</option>']
    for o in options:
        sel = " selected" if o == value else ""
        opts.append(f'<option value="{_esc(o)}"{sel}>{_esc(o)}</option>')
    return (
        f'<select class="ann-input" data-ann-scope="{_esc(scope)}" data-ann-row="{_esc(row_key)}" '
        f'data-ann-field="{_esc(field)}">{"".join(opts)}</select>'
    )


def _table(
    rows: list[dict], cols: list[tuple[str, str]], sortable: bool = True,
    subject: str | None = None, series_col: str | None = None,
) -> str:
    """series_col, if given (must be a key in cols), makes that column's
    cells clickable to open the image preview (needs `subject` too)."""
    cls_attr = ' class="sortable"' if sortable else ""
    thead = "".join(
        f'<th{cls_attr} data-k="{k}">{_esc(label)}</th>' for k, label in cols
    )
    body = []
    for r in rows:
        cells = []
        for k, _ in cols:
            if k == series_col and subject is not None:
                cells.append(
                    f'<td class="mono series-cell" {_series_cell_attrs(subject, r.get(k, ""))}>'
                    f'{_esc(r.get(k, ""))}</td>'
                )
            else:
                cells.append(f'<td data-k="{k}">{_esc(r.get(k, ""))}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    body_html = "".join(body)
    return f'<table class="dtbl"><thead><tr>{thead}</tr></thead><tbody>{body_html}</tbody></table>'


def _raw_rows(records: list[dict]) -> list[dict]:
    rows = []
    for r in records:
        s = r["summary"] or {}
        row = {"series_dir": r["series_dir"], **s}
        rows.append(row)
    return rows


def _section_raw(records: list[dict], subject: str) -> str:
    return _table(_raw_rows(records), RAW_COLS, subject=subject, series_col="series_dir")


def _section_cardisort(
    records: list[dict], subject: str, corrections: dict,
    seq_options: list[str], plane_options: list[str],
) -> str:
    cols = [("pred_seq", "pred. seq"), ("pred_plane", "pred. plane")] + RAW_COLS
    thead = "".join(f'<th class="sortable" data-k="{k}">{_esc(label)}</th>' for k, label in cols)
    thead += "<th>correct seq</th><th>correct plane</th><th>comment</th>"

    rows_html = []
    for r, raw in zip(records, _raw_rows(records)):
        merged = {"pred_seq": r["pred_seq"], "pred_plane": r["pred_plane"], **raw}
        row_key = f"{subject}||{r['series_dir']}"
        ann = corrections.get(row_key, {})
        cells = []
        for k, _ in cols:
            if k == "series_dir":
                cells.append(
                    f'<td class="mono series-cell" {_series_cell_attrs(subject, merged.get(k, ""))}>'
                    f'{_esc(merged.get(k, ""))}</td>'
                )
            else:
                cells.append(f'<td data-k="{k}">{_esc(merged.get(k, ""))}</td>')
        cells.append(f'<td>{_select_input("corrections", row_key, "corrected_seq", seq_options, ann.get("corrected_seq", ""))}</td>')
        cells.append(f'<td>{_select_input("corrections", row_key, "corrected_plane", plane_options, ann.get("corrected_plane", ""))}</td>')
        cells.append(f'<td>{_comment_input("corrections", row_key, ann.get("comment", ""))}</td>')
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    body_html = "".join(rows_html)
    return f'<table class="dtbl"><thead><tr>{thead}</tr></thead><tbody>{body_html}</tbody></table>'


def _fmt_top3(top3: list | None) -> str:
    if not top3:
        return "—"
    return "<br>".join(f"{_esc(name)} ({prob:.2f})" for name, prob in top3)


def _section_classify(
    records: list[dict], notes: dict[str, dict], subject: str, classify_comments: dict,
) -> str:
    parts = [
        '<table class="dtbl fixed"><colgroup>'
        '<col style="width:9%"><col style="width:13%">'
        '<col style="width:20%"><col style="width:20%">'
        '<col style="width:22%"><col style="width:16%">'
        '</colgroup><thead><tr><th>series</th><th>description</th>'
        '<th>top-3 sequence</th><th>top-3 plane</th><th>LLM step</th><th>comment</th></tr></thead><tbody>'
    ]
    for r in records:
        if r["pred_seq"] is None:
            continue
        note = notes.get(r["series_dir"], {})
        if not note.get("activated"):
            llm_cell = '<span class="mut">not activated — cardisort confident (margin ≥ threshold)</span>'
        elif note.get("escalated"):
            frm = note.get("escalation_from")
            to = note.get("escalation_to")
            frm_s = f"{frm[0]}/{frm[1]}" if frm else "?"
            to_s = f"{to[0]}/{to[1]}" if to else "?"
            verdict = "confirmed" if frm == to else "corrected"
            llm_cell = (
                f'<b>{verdict}</b> — was {_esc(frm_s)}, {"kept" if verdict=="confirmed" else "changed to"} '
                f'{_esc(to_s)}<br>'
                f'<span class="reason">{_esc(note.get("escalation_reasoning"))}</span>'
            )
        else:
            llm_cell = '<span class="bad">escalation attempted but failed — kept cardisort top-1 (see logs)</span>'
        description = (r.get("summary") or {}).get("series_description", "")
        row_key = f"{subject}||{r['series_dir']}"
        comment_val = classify_comments.get(row_key, {}).get("comment", "")
        parts.append(
            f'<tr><td class="mono series-cell" {_series_cell_attrs(subject, r["series_dir"])}>'
            f'{_esc(r["series_dir"])}</td>'
            f"<td>{_esc(description)}</td>"
            f"<td>{_fmt_top3(r.get('seq_top3'))}</td>"
            f"<td>{_fmt_top3(r.get('plane_top3'))}</td>"
            f"<td>{llm_cell}</td>"
            f"<td>{_comment_input('classify_comments', row_key, comment_val)}</td></tr>"
        )
    parts.append("</tbody></table>")
    return "".join(parts)


def _section_disambig(
    decisions: list[dict], records_by_dir: dict[str, dict], subject: str, disambig_comments: dict,
) -> str:
    parts = []
    for d in decisions:
        if len(d["members"]) < 2:
            continue
        reason_by = {a.get("series"): a for a in d.get("assessments", [])}
        rows = []
        for m in d["members"]:
            rec = records_by_dir.get(m, {})
            s = (rec.get("summary") or {}) if rec else {}
            a = reason_by.get(m, {})
            chosen_mark = " ★" if m == d["chosen"] else ""
            row_cls = ' class="chosen"' if m == d["chosen"] else ""
            row_key = f"{subject}||{d['group_key']}||{m}"
            comment_val = disambig_comments.get(row_key, {}).get("comment", "")
            rows.append(
                f"<tr{row_cls}>"
                f'<td class="mono series-cell" {_series_cell_attrs(subject, m)}>{_esc(m)}{chosen_mark}</td>'
                f"<td>{_esc(s.get('series_description', ''))}</td>"
                f"<td><b>{_esc(a.get('role', ''))}</b></td>"
                f'<td class="reason">{_esc(a.get("reasoning", ""))}</td>'
                f"<td>{_comment_input('disambig_comments', row_key, comment_val)}</td></tr>"
            )
        parts.append(
            f'<div class="grp"><div class="ghead"><span class="gk">{_esc(d["group_key"])}</span>'
            f'<span class="badge">{len(d["members"])} candidates</span>'
            f'<span class="pick">method: {_esc(d["method"])}</span></div>'
            f'<table class="dtbl"><thead><tr><th>series</th><th>description</th>'
            f'<th>role</th><th>reasoning</th><th>comment</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        )
    return "".join(parts) or '<p class="mut">No duplicate-class groups for this subject.</p>'


def _section_resolved(decisions: list[dict], subject: str) -> str:
    from qcardia.cardisort import series_key

    rows = []
    for d in decisions:
        key = series_key(d["seq"], d["plane"])
        if key is None:
            continue
        rows.append((key, d["seq"], d["plane"], d["chosen"]))
    rows.sort()
    body = "".join(
        f'<tr><td class="mono"><b>{_esc(k)}</b></td><td>{_esc(seq)}/{_esc(plane)}</td>'
        f'<td class="mono series-cell" {_series_cell_attrs(subject, chosen)}>{_esc(chosen)}</td></tr>'
        for k, seq, plane, chosen in rows
    )
    if not body:
        return '<p class="mut">Nothing resolved to an app key for this subject.</p>'
    return (
        '<table class="dtbl"><thead><tr><th>app key</th><th>cardisort class</th>'
        f'<th>series that loads</th></tr></thead><tbody>{body}</tbody></table>'
    )


def _subject_section(
    subject: str, records: list[dict], run: dict | None, annotations: dict,
    seq_options: list[str], plane_options: list[str],
) -> str:
    decisions = run["decisions"].get(subject, []) if run else []
    notes = run["classification_notes"].get(subject, {}) if run else {}
    records_by_dir = {r["series_dir"]: r for r in records}
    n_classified = sum(1 for r in records if r["pred_seq"] is not None)
    corrections = annotations.get("corrections", {})
    run_ann = annotations.get("runs", {}).get(run["meta"]["name"], {}) if run else {}
    classify_comments = run_ann.get("classify_comments", {})
    disambig_comments = run_ann.get("disambig_comments", {})

    def details(id_, label, body, open_=False):
        return (
            f'<details{" open" if open_ else ""} id="{id_}"><summary>{label}</summary>'
            f'<div class="secbody">{body}</div></details>'
        )

    no_run_msg = '<p class="mut">No run loaded — run `bench.py run --name X` first.</p>'
    model = run["meta"].get("model") if run else None
    model_s = f" (LLM: {_esc(model)})" if model else ""
    body = "".join([
        details(f"{subject}-raw", f"1. Raw / unlabeled ({len(records)} series)",
                _section_raw(records, subject)),
        details(f"{subject}-cardisort", f"2. Cardisort predictions ({n_classified} classified)",
                _section_cardisort(records, subject, corrections, seq_options, plane_options)),
        details(f"{subject}-classify", f"3. Per-series classification confirmation{model_s}",
                _section_classify(records, notes, subject, classify_comments) if run else no_run_msg),
        details(f"{subject}-disambig", f"4. Disambiguation (LLM role assignment){model_s}",
                _section_disambig(decisions, records_by_dir, subject, disambig_comments) if run else no_run_msg),
        details(f"{subject}-resolved", "5. Resolved app input",
                _section_resolved(decisions, subject) if run else no_run_msg),
    ])
    return f'<section class="subj-block"><h2 class="subj">{_esc(subject)}</h2>{body}</section>'


def render_page(catalog: dict, run: dict | None, annotations: dict | None = None) -> str:
    from qcardia.cardisort import PLANE_NAMES, SEQUENCE_NAMES

    annotations = annotations or {"corrections": {}, "runs": {}}
    seq_options = sorted(n for n in set(SEQUENCE_NAMES.values()) if n != "unlabelled")
    plane_options = sorted(n for n in set(PLANE_NAMES.values()) if n != "unlabelled")

    subjects_html = "".join(
        _subject_section(subject, records, run, annotations, seq_options, plane_options)
        for subject, records in catalog["subjects"].items()
    )
    run_label = run["meta"]["name"] if run else "none"
    nav = "".join(
        f'<a href="#{_esc(s)}">{_esc(s)}</a>' for s in catalog["subjects"]
    )
    return (
        _PAGE.replace("__RUN__", _esc(run_label))
        .replace("__RUN_NAME_JS__", json.dumps(run["meta"]["name"]) if run else "null")
        .replace("__NAV__", nav)
        .replace("__BODY__", subjects_html)
    )


_PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>cardisort + disambiguation review</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e2e2e2;--card:#fafafa;
--good:#2A9D3C;--bad:#E45756;--accent:#4C78A8;--warn:#b8860b}
@media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e8;--mut:#9a9a9a;
--line:#333;--card:#1e1e1e;--good:#4ec06a;--bad:#ff6b6b;--accent:#6ea3d8;--warn:#d9a740}}
*{box-sizing:border-box}
body{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:var(--bg);color:var(--fg)}
header{position:sticky;top:0;z-index:10;background:var(--bg);border-bottom:1px solid var(--line);
padding:.7rem 1.1rem;display:flex;align-items:center;gap:1.1rem;flex-wrap:wrap}
h1{font-size:1.05rem;margin:0}
.stat{font-size:.82rem;color:var(--mut)}
nav{font-size:.78rem;display:flex;gap:.5rem;flex-wrap:wrap}
nav a{color:var(--accent);text-decoration:none}
main{max-width:1250px;margin:0 auto;padding:1rem 1.1rem 4rem}
.subj-block{margin:1.6rem 0}
h2.subj{font-size:1.2rem;border-bottom:2px solid var(--line);padding-bottom:.3rem;scroll-margin-top:4rem}
details{border:1px solid var(--line);border-radius:8px;margin:.5rem 0;background:var(--card);overflow:hidden}
summary{cursor:pointer;padding:.5rem .8rem;font-weight:600;font-size:.92rem}
.secbody{padding:.3rem .8rem .8rem}
table.dtbl{width:100%;border-collapse:collapse;font-size:.82rem}
table.dtbl.fixed{table-layout:fixed}
table.dtbl td,table.dtbl th{padding:.32rem .5rem;border-top:1px solid var(--line);text-align:left;vertical-align:top;
overflow-wrap:break-word}
table.dtbl th{font-weight:500;color:var(--mut);font-size:.74rem}
th.sortable{cursor:pointer;user-select:none}
th.sortable:hover{color:var(--fg)}
th.sortable::after{content:" ⇅";opacity:.4;font-size:.7rem}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:.78rem}
.series-cell{text-decoration:underline dotted;text-underline-offset:2px}
.mut{color:var(--mut)}
.ann-input{font:inherit;font-size:.78rem;width:100%;padding:.15rem .3rem;background:var(--bg);
color:var(--fg);border:1px solid var(--line);border-radius:4px}
button#saveBtn{font:inherit;padding:.4rem .8rem;border:1px solid var(--accent);background:var(--accent);
color:#fff;border-radius:6px;cursor:pointer}
#saveStatus{font-size:.8rem;color:var(--mut)}
.bad{color:var(--bad)}
.reason{color:var(--mut);font-style:italic}
tr.chosen td{background:color-mix(in srgb,var(--accent) 10%,transparent)}
.grp{border:1px solid var(--line);border-radius:6px;margin:.4rem 0;overflow:hidden}
.ghead{display:flex;align-items:center;gap:.6rem;padding:.4rem .7rem;font-size:.85rem;
background:color-mix(in srgb,var(--accent) 8%,transparent)}
.gk{font-weight:600}
.badge{font-size:.72rem;padding:.05rem .45rem;border-radius:10px;border:1px solid var(--line)}
.pick{font-size:.72rem;color:var(--accent);margin-left:auto}
button.viewbtn{font:inherit;padding:.1rem .5rem;font-size:.72rem;border-radius:4px;border:1px solid var(--accent);
background:transparent;color:var(--accent);cursor:pointer}
#vmask{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;align-items:center;justify-content:center;z-index:50}
#vbox{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:1rem;max-width:90vw}
#vbox h3{margin:.1rem 0 .5rem;font-size:.95rem}
#vimg{background:#000;image-rendering:pixelated;width:min(70vw,520px);height:auto;display:block;margin:0 auto;border-radius:4px}
.vrow{display:flex;align-items:center;gap:.5rem;margin-top:.5rem;font-size:.8rem;color:var(--mut)}
.vrow input[type=range]{flex:1}
button.ghost{font:inherit;padding:.4rem .8rem;border:1px solid var(--accent);background:transparent;
color:var(--accent);border-radius:6px;cursor:pointer;margin-top:.6rem}
</style></head><body>
<header>
  <h1>cardisort + disambiguation review</h1>
  <span class="stat">run: <b>__RUN__</b></span>
  <button id="saveBtn">Save annotations</button>
  <span id="saveStatus"></span>
  <nav>__NAV__</nav>
</header>
<main>__BODY__</main>

<div id="vmask"><div id="vbox">
  <h3 id="vtitle"></h3>
  <img id="vimg" alt="">
  <div class="vrow"><span>slice</span><input type="range" id="vslice" min="0" max="0" value="0"><span id="vslabel">1/1</span></div>
  <div class="vrow"><span>frame</span><input type="range" id="vframe" min="0" max="0" value="0"><span id="vflabel">1/1</span></div>
  <button class="ghost" id="vclose">close</button>
</div></div>

<script>
const SERVED = location.protocol.startsWith("http");
const RUN_NAME = __RUN_NAME_JS__;

// generic click-to-sort for any table.dtbl
document.querySelectorAll('table.dtbl').forEach(tbl=>{
  tbl.querySelectorAll('th.sortable').forEach((th,i)=>{
    th.addEventListener('click',()=>{
      const tbody=tbl.querySelector('tbody');
      const rows=[...tbody.querySelectorAll('tr')];
      const asc = th.dataset.dir !== 'asc';
      tbl.querySelectorAll('th').forEach(h=>delete h.dataset.dir);
      th.dataset.dir = asc ? 'asc':'desc';
      const idx=[...th.parentNode.children].indexOf(th);
      rows.sort((a,b)=>{
        const av=a.children[idx].textContent.trim(), bv=b.children[idx].textContent.trim();
        const an=parseFloat(av), bn=parseFloat(bv);
        let cmp;
        if(!isNaN(an)&&!isNaN(bn)) cmp = an-bn; else cmp = av.localeCompare(bv);
        return asc?cmp:-cmp;
      });
      rows.forEach(r=>tbody.appendChild(r));
    });
  });
});

// per-series DICOM image preview (needs `serve`, not file://) — every
// .series-cell td carries data-subject/data-series, set server-side.
function attachViewers(){
  document.querySelectorAll('[data-subject][data-series]').forEach(el=>{
    el.style.cursor='pointer'; el.title='click to preview';
    el.addEventListener('click',()=>openViewer(el.dataset.subject, el.dataset.series));
  });
}

const V={subject:null,series:null,nslices:0,frames:{},slice:0,frame:0};
async function openViewer(subject,series){
  if(!SERVED){alert("Image preview needs the local server. Run:  uv run python bench.py serve");return;}
  V.subject=subject;V.series=series;V.frames={};V.slice=0;V.frame=0;
  document.getElementById('vtitle').textContent=subject+" / "+series;
  document.getElementById('vmask').style.display='flex';
  document.getElementById('vimg').src="";
  document.getElementById('vflabel').textContent="";
  try{
    const metaResp=await fetch(`/api/series/${encodeURIComponent(subject)}/${encodeURIComponent(series)}`);
    if(!metaResp.ok) throw new Error(`series lookup failed (HTTP ${metaResp.status})`);
    const meta=await metaResp.json();
    V.nslices=meta.n_slices||0;
    if(V.nslices===0){
      document.getElementById('vtitle').textContent=subject+" / "+series+" — no readable DICOM frames found";
      return;
    }
    const ss=document.getElementById('vslice');ss.max=Math.max(0,V.nslices-1);ss.value=0;
    await loadSlice(0);
  }catch(e){document.getElementById('vtitle').textContent="failed to load: "+e;}
}
async function loadSlice(si){
  if(!V.frames[si]){
    const r=await fetch(`/api/frames/${encodeURIComponent(V.subject)}/${encodeURIComponent(V.series)}/${si}`);
    if(!r.ok) throw new Error(`frame fetch failed (HTTP ${r.status})`);
    const j=await r.json();
    V.frames[si]=j.frames||[];
  }
  V.slice=si;V.frame=0;
  const fr=document.getElementById('vframe');fr.max=Math.max(0,V.frames[si].length-1);fr.value=0;
  document.getElementById('vslabel').textContent=(si+1)+"/"+V.nslices;
  showFrame();
}
function showFrame(){
  const f=V.frames[V.slice];
  if(!f||!f.length){document.getElementById('vimg').src="";document.getElementById('vflabel').textContent="no frames";return;}
  document.getElementById('vimg').src="data:image/png;base64,"+f[V.frame];
  document.getElementById('vflabel').textContent=(V.frame+1)+"/"+f.length;
}
document.getElementById('vslice').oninput=e=>loadSlice(+e.target.value);
document.getElementById('vframe').oninput=e=>{V.frame=+e.target.value;showFrame();};
document.getElementById('vclose').onclick=()=>document.getElementById('vmask').style.display='none';
document.getElementById('vmask').onclick=e=>{if(e.target.id==='vmask')e.target.style.display='none';};
attachViewers();

/* ---- annotations: corrections (section 2) + per-run comments (3/4) ----
   Collected straight from the DOM's current values on Save, not from a
   separately-tracked JS state object — the server merges the payload into
   annotations.json rather than overwriting, so saving while viewing one
   run never touches another run's comments. */
function collectAnnotations(){
  const out={corrections:{}, runs:{}};
  if(RUN_NAME) out.runs[RUN_NAME]={classify_comments:{}, disambig_comments:{}};
  document.querySelectorAll('[data-ann-scope]').forEach(el=>{
    const scope=el.dataset.annScope, row=el.dataset.annRow, field=el.dataset.annField;
    const root = scope==='corrections' ? out.corrections
      : (out.runs[RUN_NAME]=out.runs[RUN_NAME]||{}, out.runs[RUN_NAME][scope]=out.runs[RUN_NAME][scope]||{}, out.runs[RUN_NAME][scope]);
    root[row]=root[row]||{};
    root[row][field]=el.value;
  });
  return out;
}
document.getElementById('saveBtn').onclick=async()=>{
  const status=document.getElementById('saveStatus');
  status.textContent='saving…';
  try{
    const r=await fetch('/api/save-annotations',{method:'POST',body:JSON.stringify(collectAnnotations())});
    if(!r.ok) throw new Error('HTTP '+r.status);
    status.textContent='saved ✓';
  }catch(e){status.textContent='save failed: '+e;}
};
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Stage: serve — builds the page live + image preview API
# ---------------------------------------------------------------------------

def cmd_serve(args):
    import http.server

    if not CATALOG_PATH.exists():
        sys.exit("No catalog.json — run `catalog` first.")

    run_path = Path(args.run) if args.run and Path(args.run).exists() else \
        (RUNS_DIR / f"{args.run}.json" if args.run else _latest_run())

    data_root = Path(args.data_root)
    series_path_cache: dict[str, dict[str, Path]] = {}
    grid_cache: dict[tuple, list] = {}

    def series_paths(subject: str) -> dict[str, Path]:
        if subject not in series_path_cache:
            from qcardia.cardisort import get_sequence_dirs
            subj_dir = data_root / subject
            try:
                series_path_cache[subject] = {d.name: d for d in get_sequence_dirs(subj_dir)}
            except Exception:
                series_path_cache[subject] = {}
        return series_path_cache[subject]

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            # the page and catalog/run data change between requests as you
            # iterate — never let the browser serve a stale cached copy.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                catalog = json.loads(CATALOG_PATH.read_text())
                run = json.loads(run_path.read_text()) if run_path and run_path.exists() else None
                annotations = _load_annotations()
                page = render_page(catalog, run, annotations)
                return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            if p.startswith("/api/series/"):
                return self._series_meta(p)
            if p.startswith("/api/frames/"):
                return self._frames(p)
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            p = self.path.split("?")[0]
            if p == "/api/save-annotations":
                n = int(self.headers.get("Content-Length", 0))
                incoming = json.loads(self.rfile.read(n) or b"{}")
                _save_annotations_merge(incoming)
                log.info("Saved annotations -> %s", ANNOTATIONS_PATH.name)
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})

        def _grid(self, subject, series_dir):
            key = (subject, series_dir)
            if key not in grid_cache:
                path = series_paths(subject).get(series_dir)
                grid_cache[key] = _series_grid(path) if path else []
            return grid_cache[key]

        def _series_meta(self, p):
            subject, series_dir = p[len("/api/series/"):].split("/", 1)
            grid = self._grid(_uq(subject), _uq(series_dir))
            return self._send(200, {"n_slices": len(grid), "n_frames": [len(s) for s in grid]})

        def _frames(self, p):
            subject, series_dir, sl = p[len("/api/frames/"):].split("/", 2)
            grid = self._grid(_uq(subject), _uq(series_dir))
            si = int(sl)
            if si >= len(grid):
                return self._send(404, {"error": "slice out of range"})
            frames = [_png_b64(fr) for fr in grid[si]]
            return self._send(200, {"frames": frames})

    port = args.port
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    log.info("Serving on http://localhost:%d  (run: %s)", port, run_path.stem if run_path else "none")
    log.info("Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopped.")


def _uq(s: str) -> str:
    from urllib.parse import unquote
    return unquote(s)


def _series_grid(series_path: Path) -> list:
    """(slice, frame) grid of 2-D arrays. Uses qcardia BaseSeries; falls back to raw sort."""
    try:
        from qcardia.series import BaseSeries
        bs = BaseSeries(series_path)
        return [bs.slice_data[k]["pixel_array"] for k in sorted(bs.slice_data)]
    except Exception as exc:
        log.warning("BaseSeries failed for %s (%s) — raw fallback", series_path.name, exc)
        try:
            import pydicom
            dss = []
            for f in sorted(series_path.iterdir()):
                if f.is_file():
                    try:
                        ds = pydicom.dcmread(str(f))
                        if "PixelData" in ds:
                            dss.append(ds)
                    except Exception:
                        pass
            dss.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))
            return [[d.pixel_array] for d in dss]
        except Exception:
            return []


def _png_b64(arr) -> str:
    """Window/level a 2-D array to 8-bit grayscale PNG, capped ~256px, base64."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = np.asarray(arr).astype(np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    step = max(1, int(max(a.shape) / 256))
    a = a[::step, ::step]
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
    buf = io.BytesIO()
    plt.imsave(buf, a, cmap="gray", format="png")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("catalog", help="run cardisort over subjects (cached)")
    c.add_argument("--subjects", default="all", help="e.g. '1-15' or '1,3,7' or 'all'")
    c.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    c.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    c.add_argument("--merge", action="store_true", help="keep already-catalogued subjects")
    c.set_defaults(func=cmd_catalog)

    r = sub.add_parser("run", help="run disambiguation over the catalog")
    r.add_argument("--name", required=True, help="run name, e.g. 'baseline' or 'tuned-v1'")
    r.add_argument("--model", default="qwen3:8b")
    r.add_argument("--ollama-url", default="http://localhost:11434")
    r.set_defaults(func=cmd_run)

    sv = sub.add_parser("serve", help="build + serve the review page, with image preview")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    sv.add_argument("--run", default=None, help="run name (default: most recently created run)")
    sv.set_defaults(func=cmd_serve)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
