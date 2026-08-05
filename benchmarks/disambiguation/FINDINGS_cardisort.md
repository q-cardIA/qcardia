# Findings — Cardisort classification (Track 1)

Direction for the cardisort model author. Qualitative conclusions from reviewing
predictions on the benchmark cohort (QLGE01–15, 597 series); back each with a
number from `score-cardisort` once `groundtruth/cardisort_labels.csv` is annotated.

**Model design reminder:** cardisort predicts **two independent labels** per series
— a **sequence** (23 classes: CINE, DBLGE, WBLGE, PERF, …) and a **plane** (18
classes: SAX, 2CH, 3CH, 4CH, …), from `SEQUENCE_NAMES` / `PLANE_NAMES` in
`qcardia/src/qcardia/cardisort.py`. A "category" like *WB-LGE 4-chamber* is the
combination `WBLGE` × `4CH`, not a single class.

---

## 1. Cardisort collapses almost all LGE into `DBLGE_SAX`

Across the 15-subject cohort, LGE acquisitions are predicted overwhelmingly as
**DBLGE / SAX**, with only a handful landing in other LGE combinations:

```
147  DBLGE/SAX      <- dominates
 10  WBLGE/SAX
  1  DBLGE/4CH
  1  DBLGE/2CH
  1  WBLGE/4CH
```

Effectively cardisort treats "LGE" as one bucket (dark-blood, short-axis) rather
than resolving the distinct LGE categories in the protocol (WBLGE vs DBLGE ×
SAX / 2CH / 3CH / 4CH).

**To confirm quantitatively (`score-cardisort` → confusion + "category collapse"
table):** is each missing category *absent from the protocol*, or **mislabeled**?
The key question: is a WB-LGE 4-chamber getting the plane right but the sequence
wrong (predicted `CINE_4CH`), so it never reaches an LGE group at all? Flag those
in the review page (set `true_seq = WBLGE`); they surface as red rows and as
confusion pairs in the report.

## 2. The collapse is what starves the downstream LGE categories

Because LGE is funnelled into one `DBLGE_SAX` group, the 2CH / 3CH / 4CH / WBLGE
LGE views never appear as their own classes downstream — they sit inside that
collapsed group and are discarded as non-primary duplicates by disambiguation
(see `FINDINGS_disambig.md`, which cannot fix this — it's upstream here).

**Direction:** cardisort needs to separate the LGE sub-categories (sequence
*and* plane) before grouping. This is a cardisort-model problem, not a prompt
problem.

## 3. Limitations to quantify (not yet numbers)

- **per-class error rate + common confusions** — which (seq, plane) get swapped,
  ranked by frequency (`score-cardisort` confusion table). Watch for LGE→CINE and
  WBLGE→DBLGE.
- **planes vs sequences** — is the plane head more reliable than the sequence head
  (or vice-versa)? Splitting accuracy by head tells the author which to prioritise.
- **unclassified series** — series cardisort returns nothing for (listed per
  subject on the review page); how many, and what are they?

---

## Status / next steps

- [ ] Annotate `groundtruth/cardisort_labels.csv` via the review page + `serve`.
- [ ] `score-cardisort` → attach real numbers + confusion table to findings 1–3.
- [ ] Hand the report to the model author with the collapse table as the headline.
