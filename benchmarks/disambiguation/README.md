# Cardisort + disambiguation review

Interactive, human-in-the-loop review of qcardia's full series-classification
pipeline: cardisort's raw predictions, the LLM confidence-escalation step
that double-checks cardisort when it's unsure, and the LLM disambiguation
step that assigns a role (primary/repeat/other, or a class-specific
vocabulary like rest/stress) to duplicate series. It's a **qualitative
review tool**, not a scoring harness — the goal is to see exactly what each
stage saw and why it decided what it decided, correct/comment on it, and
iterate on prompts and thresholds. Runs headless against the qcardia package
directly (no app, no Qt), so it measures exactly what's in
`qcardia/src/qcardia/{cardisort,refine_classification,disambiguate}.py`.

Because cardisort itself is still mid-training, treat this as a starting
point rather than a strict pass/fail benchmark — the pipeline shape and
review workflow are what matter now; tightening accuracy comes once
cardisort is further along.

## Workflow

```bash
cd benchmarks/disambiguation

# 1. classify (slow, ~1-2 min/subject). Caches to catalog.json.
uv run python bench.py catalog --subjects 1-15

# 2. run disambiguation over the cache (fast; only step that needs Ollama)
uv run python bench.py run --name baseline

# 3. build + serve the review page (needs `serve` for image preview + saving)
uv run python bench.py serve
#    open http://localhost:8765
```

`catalog --merge` keeps already-catalogued subjects when adding more (skips
re-running the CNN on them). `run --name X` is cheap to repeat under a new
name after tuning `disambiguate.py`/`refine_classification.py` — it replays
against the cached catalog, no CNN or DICOM re-reads. `serve --run X` picks
which run to view (defaults to the most recently created one).

## The review page

One page, one collapsible section per subject, five subsections walking the
full pipeline for that subject:

1. **raw** — every series found, unlabeled, with its DICOM metadata
   (description, dimensions, instance count, acquisition date/time).
2. **cardisort** — the same table + cardisort's predicted `(sequence,
   plane)`, sortable by any column (click a header).
3. **classify** — per series, cardisort's top-3 candidates per head with
   probabilities, and the LLM confidence-escalation verdict: "not
   activated" if cardisort's top-1/top-2 margin was wide enough
   (`qcardia.cardisort.CONFIDENCE_MARGIN_THRESHOLD`), or "confirmed"/
   "corrected" with the LLM's reasoning if it fired.
4. **disambig** — per duplicate-class group, every candidate's LLM-assigned
   role + reasoning. Most classes use primary/planning/repeat/other; some
   (e.g. `PERF_SAX`) use a custom role vocabulary via
   `qcardia.disambiguate.ROLE_SCHEMAS` — rest/stress instead of a single
   primary, since both should be identifiable rather than one being
   discarded as a "repeat" of the other.
5. **resolved** — the final `app key -> series` mapping that would actually
   load into the viewer app.

Click any series name to open a DICOM image preview (slice + frame
sliders) — needs `serve`, since it renders frames from disk on demand.

## Correcting and commenting

- **Section 2** has two dropdown columns (correct sequence / correct plane,
  from cardisort's full label vocabulary) plus a free-text comment column.
- **Sections 3 and 4** each have a free-text comment column.
- **Save** (top of page) writes everything to `annotations.json`.

Annotations are kept in their own file, separate from `catalog.json` and
`runs/*.json`, so re-cataloguing or trying a different disambiguation run
never loses them:

- **Section 2 corrections** are keyed by `subject||series_dir` only —
  they're about the raw data, not about any one run's LLM output, so they
  apply no matter which `--run` you're viewing.
- **Section 3/4 comments** are scoped per run name (`runs.<name>.*` inside
  `annotations.json`), since a comment on "this reasoning is wrong" is
  about that run's specific LLM output. Switching `--run` shows that run's
  own comments, not another run's. Saving while viewing one run merges into
  the file rather than overwriting it, so it never drops another run's data.

## Notes

- Uses the same Ollama model as the app (`qwen3:8b` default; `run --model`
  and `--ollama-url` to change).
- Cardisort weights default to `../../models/weights/cardisort`; override
  with `catalog --weights`.
- Every run records the model + a fingerprint of `disambiguate.py` (file
  hash + git commit), so a run is attributable to a specific code version.
- `catalog.json`, `runs/`, and `annotations.json` are experiment
  artifacts — commit them alongside code changes for reproducibility.
