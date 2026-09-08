# Agent tool-calling eval

Measures how accurately the chat assistant's tool-calling (`src/llm/`'s `SYSTEM_PROMPT` + `TOOL_SPECS` + `build_tools()`, orchestrated by `chat_controller.py`) picks the right tool call for a given user instruction — built to track down and fix issue #12 ("agent performance ... worse than the LLM-assistant branch"), particularly `apply_preset` accuracy.

## Prerequisites

- A local Ollama server running and reachable (`utils.app_config.ollama_available()`), with the model pulled (default: see `utils/app_config.OLLAMA_MODEL`).
- Run from the repo root with the app's own `uv` environment, e.g.:
  ```
  uv run python tests/llm/run_eval.py --category preset
  ```

## How it works

`driver.py` builds a real `ChatController` via `object.__new__()` (no Qt/QApplication touched) against fake `model`/`view`/`dicom_controller` stand-ins (`fakes.py`), then calls the controller's **real, unmodified** `_messages_for_request()`/`_build_tools()` and runs one `OllamaWorker` request synchronously — so it automatically tracks every prompt/precondition change, no separate copy to keep in sync.

Each test case (`test_cases.py`) specifies a fixture (`model`+`view`), a user prompt, and the tool call(s) the model is expected to emit, plus a `match_mode` (`exact` / `any_order` / `subset` / `none`) that `validator.py` checks against.

## What's in the suite, and what's parked

`test_cases.py` is one flat suite in the app's own vocabulary — every expected call names a real tool and real arguments (`load_subject(subject_number=)`, `display_series(panel_idx=, series_key=, axis=)`, `apply_preset(preset=)`, `run_segmentation(modality=)`). Categories: `load`, `preset`, `display_explicit`, `display_natural`, `display_multi`, `display_positional`, `display_state`, `segmentation`, `refusal`, `compound`.

There is no skip mechanism: **everything in `TEST_CASES` runs and counts.** Cases for capabilities the app doesn't have yet are written out but **commented out**, each under a header naming what has to land first — grid sizing (no `set_grid` tool), the extended-functional and flow presets (no `functional_complex`/`flow`/`flow_aorta`/`flow_mpa` in `hanging_protocols.PRESETS`, no `PC_AORTA`/`PC_MPA` keys), per-panel slice state (`display_series` takes no slice argument), the `run_analysis`/`zoom_to_roi`/playback tools, and the T1-map/EGE/black-blood series keys cardisort doesn't emit. Uncomment a block when its feature ships.

### Reading the `compound` category

`driver.run_one()` runs a **single turn**, so a compound request can only be scored on what one reply contains. Two of the cases (`compound_003`, `compound_004`) expect the model to finish a three-step request in one reply. That is a legitimate expectation — the `display_multi_*` cases make the same assumption — but `SYSTEM_PROMPT`'s MULTI-STEP rule explicitly also blesses the other path: *"continuing one tool at a time after each result is just as correct."* A model that chains correctly across follow-up turns scores 0 on these two and is not broken.

So treat those two as a *one-reply completion* measurement, useful for comparing models, not as a correctness gate. Scoring the chained path properly needs a multi-turn driver — feed each `_execute_tool()` result back and re-prompt until the model stops calling tools. `_execute_tool` already works against the fakes for `apply_preset`, `display_series` and (with a small `FakeDicomController` addition) `run_segmentation`; `load_subject` is the one that would need a fixture-declared "catalog after load", since the real path does an async DICOM load.

Note there is only ever one LGE key per plane: cardisort's `WBLGE` and `DBLGE` both map to the `LGE_` prefix (`model/dicom_model.py`'s `CARDISORT_SEQUENCE_TO_PREFIX`), so the suite has no white-blood/dark-blood cases — that distinction doesn't exist downstream of classification.

## Usage

```bash
# Sanity-check all fixtures render without exceptions — no Ollama call
python tests/llm/run_eval.py --dry-run

# Run one category
python tests/llm/run_eval.py --category preset

# Run specific test ids
python tests/llm/run_eval.py --ids preset_neg_001,preset_neg_002

# Repeat each case 3x
python tests/llm/run_eval.py --category preset --runs 3

# think:False vs think:True ablation
python tests/llm/run_eval.py --category preset --think
python tests/llm/run_eval.py --category preset --no-think
```

Results are written to `tests/llm/results/results_<timestamp>.json` and a per-category + overall pass-rate table is printed to stdout.

## Not covered

This is a **manual/local eval tool, not wired into CI** — it needs a live Ollama server with a real model pulled to produce meaningful numbers, so CI coverage would need a separate, much smaller structural check (fixtures render without exceptions — `--dry-run` already does this) rather than running the full suite against a live model on every push.
