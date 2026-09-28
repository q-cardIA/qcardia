"""Time per state from study logs (written by src/utils/study_log.py).

    uv run tests/study/study_report.py logs/<session>/<session>.jsonl   # into that session's folder
    uv run tests/study/study_report.py logs/*/*.jsonl                  # all sessions together, into logs/report/

The app runs the first form by itself when a logged session ends. Each moment of a session, or of each task
when the experimenter marked tasks with Ctrl+Shift+T, gets the first state in PRIORITY that is active then;
time in no state is "observing".
"""
from __future__ import annotations

import argparse
import glob
import os
import traceback
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SPAN_STATES = {"away": "away", "loading": "load_subject", "speaking": "voice", "transcribing": "transcription",
               "waiting": "chat"}
PRIORITY = [*SPAN_STATES, "typing", "scrolling", "image"]
COLORS = {"typing": "#2a78d6", "speaking": "#eb6834", "transcribing": "#1baf7a", "waiting": "#eda100",
          "scrolling": "#e87ba4", "image": "#008300", "loading": "#4a3aa7", "observing": "#c3c2b7", "away": "#e1e0d9"}
LABELS = {"loading": "loading subject", "waiting": "waiting for assistant", "image": "image interaction"}
LATENCY = {"chat_first_token": "first token", "chat_end": "full reply", "transcription_end": "transcription",
           "load_subject_end": "subject load", "segmentation_end": "segmentation"}
REPEATS = {"load_subject_start": ("subject", "subject reloaded"),
           "reassign_series": ("series_folder", "series reassigned again"),
           "segmentation_start": ("series", "segmentation requested again")}
COUNTED = ("chat_start", "voice_start", "tool_call", "button", "wheel", "key", "mouse")
IMAGE_TARGET = "panel_"
TYPING_GAP, SCROLL_GAP, SCROLL_MIN = 5.0, 1.0, 0.25
INK, INK_2, MUTED, GRID, SURFACE, RETRY = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": "#c3c2b7", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.titlelocation": "left", "axes.titlesize": 10, "axes.titlecolor": INK, "axes.labelcolor": INK_2,
    "figure.constrained_layout.use": True, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def load(paths) -> pd.DataFrame:
    df = pd.concat([pd.read_json(p, lines=True, dtype={"participant": str, "condition": str}).assign(session=Path(p).stem)
                    for p in paths], ignore_index=True)
    for column in ("condition", "target", "text", "error", "duration_ms"):
        if column not in df:
            df[column] = np.nan
    df["condition"] = df["condition"].fillna("")
    df["t"] = df["ms"] / 1000
    df["is_error"] = df["error"].notna()
    return df


def _pairs(ev, start, end, t_end):
    out, opened = [], None
    for t, e in ev.loc[ev["event"].isin([start, end]), ["t", "event"]].itertuples(index=False):
        if e == start and opened is None:
            opened = t
        elif e == end and opened is not None:
            out.append((opened, t))
            opened = None
    return out + ([(opened, t_end)] if opened is not None else [])


def _bursts(times, gap, min_len=0.0):
    out = []
    for t in sorted(times):
        if out and t - out[-1][1] <= gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [(a, max(b, a + min_len)) for a, b in out]


def intervals(ev) -> dict[str, list]:
    end = ev["t"].max()
    spans = {state: _pairs(ev, f"{name}_start", f"{name}_end", end) for state, name in SPAN_STATES.items()}
    sent = ev.loc[ev["event"] == "chat_start", "t"].to_numpy()
    spans["typing"] = [(a, next((s for s in sent if b <= s <= b + TYPING_GAP), b))
                       for a, b in _bursts(ev.loc[ev["event"] == "keystroke", "t"], TYPING_GAP)]
    spans["scrolling"] = _bursts(ev.loc[ev["event"].isin(["wheel", "slider"]), "t"], SCROLL_GAP, SCROLL_MIN)
    drags = ev[(ev["event"] == "mouse") & ev["target"].astype(str).str.startswith(IMAGE_TARGET)]
    spans["image"] = list(zip(drags["t"] - drags["duration_ms"] / 1000, drags["t"]))
    return spans


def _covered(spans, points):
    if not spans:
        return np.zeros(len(points), bool)
    spans = sorted(spans)
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    starts, ends = np.array(merged).T
    i = np.searchsorted(starts, points, side="right") - 1
    return (i >= 0) & (points < ends[np.clip(i, 0, None)])


def windows(ev):
    """(task, start, end): one per marked task, else the whole session."""
    starts, ends = ev[ev["event"] == "task_start"], ev[ev["event"] == "task_end"]
    if starts.empty:
        return [("", ev["t"].min(), ev["t"].max())]
    return [(int(r.task), r.t, ends.loc[ends["task"] == r.task, "t"].min() if (ends["task"] == r.task).any()
             else ev["t"].max()) for r in starts.itertuples()]


def episodes(spans, wins) -> pd.DataFrame:
    rows = []
    for task, t0, t1 in wins:
        cuts = np.unique(np.clip([t0, t1, *(x for s in spans.values() for ab in s for x in ab)], t0, t1))
        mids = (cuts[:-1] + cuts[1:]) / 2
        state = np.full(len(mids), "observing", dtype=object)
        for name in reversed(PRIORITY):
            state[_covered(spans[name], mids)] = name
        for s, a, b in zip(state, cuts[:-1], cuts[1:]):
            if rows and rows[-1]["task"] == task and rows[-1]["state"] == s:
                rows[-1]["end"] = b
            else:
                rows.append({"task": task, "state": s, "start": a, "end": b})
    df = pd.DataFrame(rows, columns=["task", "state", "start", "end"])
    df["duration"] = df["end"] - df["start"]
    return df


def _similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.8


def retries(ev) -> pd.DataFrame:
    """Actions repeated within one task: a similar message within 2 minutes, a message within a minute of an
    error, a voice recording discarded for a new one, or the same subject, reassignment or segmentation again."""
    def fresh():
        return [], set(), -np.inf, False

    rows, (sent, seen, last_error, pending_voice) = [], fresh()
    for r in ev.itertuples():
        e = r.event
        if e == "task_start":
            sent, seen, last_error, pending_voice = fresh()
        elif r.is_error:
            last_error = r.t
        elif e == "chat_start":
            if any(r.t - t < 120 and _similar(str(r.text), old) for t, old in sent):
                rows.append((r.t, "message repeated"))
            elif r.t - last_error < 60:
                rows.append((r.t, "message after error"))
            sent.append((r.t, str(r.text)))
            pending_voice = False
        elif e == "transcription_end":
            pending_voice = True
        elif e == "voice_start" and pending_voice:
            rows.append((r.t, "voice re-recorded"))
            pending_voice = False
        elif e in REPEATS:
            field, label = REPEATS[e]
            key = (e, getattr(r, field, None))
            if key in seen and getattr(r, "by", None) != "assistant":
                rows.append((r.t, label))
            seen.add(key)
    return pd.DataFrame(rows, columns=["t", "kind"])


def latencies(ev) -> pd.DataFrame:
    done = ev[ev["event"].isin(LATENCY) & ~ev["is_error"]]
    return pd.DataFrame({"metric": done["event"].map(LATENCY), "seconds": done["duration_ms"] / 1000})


def analyse(df):
    eps, ret, lat, summary = [], [], [], []
    for (session, participant, condition), ev in df.groupby(["session", "participant", "condition"], sort=False):
        ev = ev.sort_values("t", kind="stable")
        info = {"session": session, "participant": participant, "condition": condition}
        wins = windows(ev)
        e, r = episodes(intervals(ev), wins).assign(**info), retries(ev).assign(**info)
        eps.append(e)
        ret.append(r)
        lat.append(latencies(ev).assign(**info))
        for task, t0, t1 in wins:
            inside = ev[ev["t"].between(t0, t1)]
            seconds = e[e["task"] == task].groupby("state")["duration"].sum()
            counts = inside["event"].value_counts()
            summary.append({**info, "task": task, "total_s": round(t1 - t0, 1),
                            **{f"{s}_s": round(seconds.get(s, 0.0), 1) for s in COLORS},
                            **{f"{s}_pct": round(100 * seconds.get(s, 0.0) / max(t1 - t0, 1e-9), 1) for s in COLORS},
                            **{f"n_{k}": int(counts.get(k, 0)) for k in COUNTED},
                            "errors": int(inside["is_error"].sum()), "retries": int(r["t"].between(t0, t1).sum())})

    def cat(frames):
        return pd.concat([f for f in frames if not f.empty] or frames[:1], ignore_index=True)

    return cat(eps), cat(ret), cat(lat), pd.DataFrame(summary)


def _title(participant, condition, task=""):
    return " · ".join(str(v) for v in (participant, condition, f"task {task}" if task != "" else "") if v)


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def _groups(df):
    key = "condition" if (df["condition"] != "").any() else "participant"
    return key, list(dict.fromkeys(df[key]))


def plot_time_per_state(summary, path):
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(summary) + 1.6))
    ax.yaxis.grid(False)
    ax.xaxis.grid(True)
    left = np.zeros(len(summary))
    for s in COLORS:
        width = summary[f"{s}_s"].to_numpy()
        bars = ax.barh(range(len(summary)), width, left=left, height=0.6, color=COLORS[s], edgecolor=SURFACE,
                       linewidth=1.5, hatch="//" if s == "away" else None, label=LABELS.get(s, s))
        for bar, w, total in zip(bars, width, summary["total_s"]):
            if w >= 0.12 * total:
                ax.text(bar.get_x() + w / 2, bar.get_y() + bar.get_height() / 2, f"{w:.0f} s", ha="center",
                        va="center", fontsize=8, color=INK)
        left += width
    ax.set_yticks(range(len(summary)), [_title(r.participant, r.condition, r.task) for r in summary.itertuples()])
    ax.invert_yaxis()
    ax.set_xlabel("seconds")
    ax.set_title("Time per state", fontsize=11)
    fig.legend(ncol=5, fontsize=8, frameon=False, loc="outside lower left")
    _save(fig, path)


def _boxes(df, facet, value, facets, path, color=lambda f: INK_2, title=str):
    if not facets:
        return
    key, groups = _groups(df)
    cols = min(4, len(facets))
    rows = -(-len(facets) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.7 * rows), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, f in zip(axes.flat, facets):
        c = color(f)
        data = [df.loc[(df[facet] == f) & (df[key] == g), value].to_numpy() for g in groups]
        ax.boxplot(data, widths=0.5, showfliers=False, medianprops={"color": INK, "linewidth": 1.5},
                   boxprops={"color": c, "linewidth": 1.5}, whiskerprops={"color": c}, capprops={"color": c})
        for i, d in enumerate(data, start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=10, color=c, alpha=0.6, linewidths=0)
        ax.set_xticks(range(1, len(groups) + 1), groups)
        ax.set_title(title(f))
        ax.set_ylabel("seconds", fontsize=8)
    for ax in list(axes.flat)[len(facets):]:
        ax.set_visible(False)
    _save(fig, path)


def plot_timeline(eps, ret, path):
    sessions = list(dict.fromkeys(eps["session"]))
    if not sessions:
        return
    lanes = [s for s in COLORS if s != "observing"]
    fig, axes = plt.subplots(len(sessions), 1, figsize=(10, 1.9 * len(sessions) + 0.4), squeeze=False, sharex=True)
    for ax, session in zip(axes[:, 0], sessions):
        ax.yaxis.grid(False)
        ax.xaxis.grid(True)
        e = eps[eps["session"] == session]
        for task, span in e[e["task"] != ""].groupby("task"):
            ax.axvspan(span["start"].min(), span["end"].max(), color=GRID, alpha=0.35, linewidth=0)
            ax.text(span["start"].min(), -1.5, f" task {task}", color=MUTED, fontsize=8, va="top")
        for y, s in enumerate(lanes):
            part = e[e["state"] == s]
            ax.broken_barh(list(zip(part["start"], part["duration"])), (y - 0.35, 0.7), color=COLORS[s],
                           hatch="//" if s == "away" else None)
        r = ret[ret["session"] == session]
        ax.scatter(r["t"], np.full(len(r), -1), marker="x", s=30, color=RETRY)
        ax.set_yticks(range(-1, len(lanes)), ["retry"] + [LABELS.get(s, s) for s in lanes], fontsize=8)
        ax.set_ylim(len(lanes) - 0.5, -1.6)
        ax.set_title(_title(e["participant"].iloc[0], e["condition"].iloc[0]))
    axes[-1, 0].set_xlabel("seconds since session start (observing: gaps)")
    _save(fig, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", help="Log files; wildcards such as logs/*/*.jsonl also work in PowerShell")
    p.add_argument("--out", type=Path, help="Output folder (default: the log's folder, or report/ for several logs)")
    args = p.parse_args()
    logs = [Path(f) for pattern in args.logs for f in sorted(glob.glob(str(pattern))) or [pattern]]
    parents = [log.resolve().parent for log in logs]
    out = args.out or (parents[0] if len(parents) == 1 else Path(os.path.commonpath(parents)) / "report")
    out.mkdir(parents=True, exist_ok=True)
    try:
        report(logs, out)
    except Exception:
        (out / "report_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def report(logs, out):
    eps, ret, lat, summary = analyse(load(logs))
    for name, frame in {"episodes": eps, "retries": ret, "latency": lat, "summary": summary}.items():
        frame.round(3).to_csv(out / f"{name}.csv", index=False)
    plot_time_per_state(summary, out / "time_per_state.png")
    states = [s for s in COLORS if s != "away" and s in set(eps["state"])]
    _boxes(eps, "state", "duration", states, out / "episode_durations.png", COLORS.get, lambda s: LABELS.get(s, s))
    plot_timeline(eps, ret, out / "timeline.png")
    _boxes(lat, "metric", "seconds", list(dict.fromkeys(lat["metric"])), out / "latency.png")
    print(summary[["participant", "condition", "task", "total_s", *(f"{s}_pct" for s in COLORS), "retries"]]
          .to_string(index=False))
    print(f"\nWrote CSVs and figures to {out}")


if __name__ == "__main__":
    main()
