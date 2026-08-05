# Findings — LLM disambiguation (Track 2)

Direction for the disambiguation tuning (prompt / model size / metadata). Qualitative
conclusions from reviewing the LLM's primary-series picks on the benchmark cohort;
back each with a number from `score-disambig` once
`groundtruth/disambig_labels.csv` is annotated. Score the LLM on the
**cardisort-correct subset** so cardisort's errors (see `FINDINGS_cardisort.md`)
aren't blamed on the LLM.

**Mechanism reminder:** one Ollama call per duplicate `(sequence, plane)` group
(`qcardia/src/qcardia/disambiguate.py`), returning a single `primary_series` plus a
role + reasoning per candidate. It can only ever pick **one** primary per group.

---

## 1. Within a correctly-formed group, the LLM picks the primary well

When cardisort groups duplicates correctly, disambiguation generally selects the
intended diagnostic series (e.g. the real SAX stack over a low-res planning cine).
This is the mechanism working as designed. Quantify it as the **cardisort-correct
subset** agreement in `score-disambig`.

## 2. One-primary-per-group discards the non-primary LGE views

The structural limit: when cardisort collapses 2CH / 3CH / 4CH / WBLGE LGE into one
`DBLGE_SAX` group (`FINDINGS_cardisort.md` finding 1), disambiguation picks one
primary and throws the rest away as duplicates — so those views are lost. **A
prompt tweak cannot recover them**; the fix is upstream in cardisort, OR a
post-classification split that separates the group by plane/sequence *before*
disambiguation runs.

## 3. Stress vs rest perfusion is not distinguished

Perfusion is grouped only as `PERF_SAX` — no rest/stress separation. Both collapse
into one group and disambiguation picks a single primary, losing a clinically
meaningful distinction.

**This needs a different task shape**, not a prompt tweak: instead of "choose one
primary," the LLM should operate *inside* `PERF_SAX` and **label** each member as
rest / stress. That likely needs:
- a dedicated schema returning a role (rest / stress) per series,
- richer metadata: acquisition/trigger time, HR, contrast-bolus timing, series
  description keywords.

## 4. Tuning levers to try (in rough priority)

Iterate in `qcardia/src/qcardia/disambiguate.py`; re-`run --name X` picks up edits
via the editable install; `compare` shows before/after.

1. **Richer metadata** in `summarize_sequence_dir` — currently only description,
   series number, instance count, rows/cols, slice thickness. Add acquisition time,
   TR/TE, contrast flags, etc. (cheapest, highest-leverage first).
2. **Prompt rules/hints** in `PROMPT_TEMPLATE` — encode heuristics (e.g. "the
   diagnostic stack usually has the most instances / largest matrix").
3. **Few-shot examples** of tricky groups.
4. **Model size** — compare `qwen3:8b` against a larger local model on the same
   `catalog` (`run --model ...` then `compare`) to separate model-capability from
   prompt quality.

## 5. Limitations to quantify

- **agreement, cardisort-correct subset** — the headline "does the LLM work" number.
- **scattered_by_cardisort** — groups where the true primary was scattered elsewhere
  by cardisort; reported separately (not the LLM's fault).
- **fallbacks** — groups where the LLM call failed and we defaulted to largest.
- **per-sequence agreement** — which sequence types the LLM handles worst.

---

## Status / next steps

- [ ] Annotate `groundtruth/disambig_labels.csv` via the review page + `serve`.
- [ ] `score-disambig --run baseline` → real baseline numbers (all + clean subset).
- [ ] Prototype rest/stress split inside `PERF_SAX` (new task shape — finding 3).
- [ ] Then tune (findings 4) and use `compare` to prove improvements.
