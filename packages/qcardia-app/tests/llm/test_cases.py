"""Eval test cases for qcardia-app-dev's ChatController tool-calling.

One suite, one vocabulary: every case is written against the app's real
tools and argument names — `list_subjects()`, `load_subject(subject_id=)`,
`display_series(panel_idx=, series_key=, axis=)`, `apply_preset(preset=)`,
`run_segmentation(modality=)`. See fakes.py for the fixture classes and
src/llm/ for the real prompt/tool specs this harness exercises unmodified.

Capabilities the app doesn't have yet — grid sizing, flow presets, per-panel
slice state, analysis/zoom/playback tools, and the T1/EGE/anatomy series keys
cardisort doesn't produce — have their cases written out but **commented
out**, each under a header saying what has to land before they're
uncommented. Nothing is skipped at runtime: if a case is
in TEST_CASES it gets run and it counts towards the pass rate.

match_mode:
    "exact"      — same calls in same order, no extras
    "any_order"  — same calls, any order, no extras
    "subset"     — every expected call appears somewhere in actual (extras OK)
    "none"       — actual tool_calls must be empty
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from fakes import DEFAULT_CINE_MODALITIES, FakeDisplayAreaWidget, FakeModel, FakeView


@dataclass
class TestCase:
    id: str
    category: str
    description: str
    model: FakeModel
    view: FakeView
    prompt: str
    expected_tool_calls: List[dict]
    match_mode: str = "exact"
    available_subjects: List[str] = field(default_factory=lambda: [f"QLGE{n:02d}_" for n in range(1, 22)])
    notes: str = ""
    # For one expected call (matched by name), assert _execute_tool()'s result
    # text contains this substring — for regression cases where the result
    # text itself is what's being checked (e.g. partial-fill reporting).
    result_contains: dict | None = None  # {"tool_name": str, "contains": str}


# ---------------------------------------------------------------------------
# Series catalogs
#
# A catalog is what the subject has *loaded and registered*, independent of
# what any panel is currently showing. Note there is one LGE key per plane:
# cardisort's WBLGE and DBLGE both map to the LGE_ prefix (see
# model/dicom_model.py's CARDISORT_SEQUENCE_TO_PREFIX), so the app has no
# white-blood/dark-blood distinction to test.
# ---------------------------------------------------------------------------

CINE_ONLY = {"CINE_SAX": True, "CINE_2CH": True, "CINE_3CH": True, "CINE_4CH": True}
CINE_LGE = {**CINE_ONLY, "LGE_SAX": True, "LGE_2CH": True, "LGE_3CH": True, "LGE_4CH": True}
FULL = {**CINE_LGE, "PERF_SAX": True}
FULL_STRESS = {**FULL, "PERF_STRESS_SAX": True}


# --- Fixture helpers --------------------------------------------------------

def _model(series: dict, subject: str | None = "QLGE01_") -> FakeModel:
    return FakeModel(
        series=series,
        folder_path=Path(subject) if subject else None,
        cine_modalities=DEFAULT_CINE_MODALITIES,
    )


def _view(grid=(2, 2), panels=None, registered_series: set[str] | None = None) -> FakeView:
    return FakeView(FakeDisplayAreaWidget(grid_size=grid, panels=panels or [], registered_series=registered_series))


def _loaded(catalog: dict = FULL, grid=(2, 2)) -> tuple[FakeModel, FakeView]:
    """Subject loaded, nothing displayed yet — everything in `catalog` counts
    as both loaded (model.series) and registered (is_series_registered),
    which is the app's state right after a load."""
    return _model(catalog), _view(grid=grid, registered_series=set(catalog))


def _displaying(panels: list[dict], grid=(2, 2), catalog: dict = FULL) -> tuple[FakeModel, FakeView]:
    """As _loaded, plus panels already on screen.
    panels: [{"idx", "series_key", "has_segmentation"?}]."""
    return _model(catalog), _view(grid=grid, panels=panels, registered_series=set(catalog))


def _no_subject(grid=(2, 2)) -> tuple[FakeModel, FakeView]:
    return _model({}, subject=None), _view(grid=grid, registered_series=set())


# --- Tool-call helpers — argument keys verified against src/llm/tools.py -----

def load(subject_id: str) -> dict:
    return {"name": "load_subject", "arguments": {"subject_id": subject_id}}


def list_subjects() -> dict:
    return {"name": "list_subjects", "arguments": {}}


def preset(name: str) -> dict:
    return {"name": "apply_preset", "arguments": {"preset": name}}


def display(panel_idx: int, series_key: str, axis: str | None = None) -> dict:
    """axis omitted by default — the app already defaults it to "transverse"
    server-side, so don't require the model to echo that. Pass it explicitly
    only when specifically testing reslicing."""
    args = {"panel_idx": panel_idx, "series_key": series_key}
    if axis is not None:
        args["axis"] = axis
    return {"name": "display_series", "arguments": args}


def display_any_panel(series_key: str, axis: str | None = None) -> dict:
    """Omits panel_idx so the validator accepts any panel index."""
    args = {"series_key": series_key}
    if axis is not None:
        args["axis"] = axis
    return {"name": "display_series", "arguments": args}


def segment(modality: str) -> dict:
    return {"name": "run_segmentation", "arguments": {"modality": modality}}


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES: List[TestCase] = []


def _add(**kwargs):
    TEST_CASES.append(TestCase(**kwargs))


# === LOAD ====================================================================

_m, _v = _no_subject()
_add(
    id="load_001", category="load", description="Load by number, no subject loaded",
    model=_m, view=_v, prompt="Load subject 3",
    expected_tool_calls=[load("QLGE03_")], match_mode="exact",
)
_m, _v = _no_subject()
_add(
    id="load_002", category="load", description="Load via 'open patient' synonym",
    model=_m, view=_v, prompt="Open patient 7",
    expected_tool_calls=[load("QLGE07_")], match_mode="exact",
)
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_2CH"}])
_add(
    id="load_003", category="load", description="Switch subject while one is already loaded",
    model=_m, view=_v, prompt="Switch to subject 5",
    expected_tool_calls=[load("QLGE05_")], match_mode="exact",
)
_m, _v = _no_subject()
_add(
    id="load_neg_001", category="load", description="Subject not in the data folder",
    model=_m, view=_v, prompt="Load subject 999",
    expected_tool_calls=[], match_mode="none",
    notes="999 is out of range — INVALID ARGUMENTS rule: no tool call, no clamping.",
)
_m, _v = _no_subject()
_add(
    id="load_neg_002", category="load", description="Subject zero — below valid range",
    model=_m, view=_v, prompt="Load subject 0",
    expected_tool_calls=[], match_mode="none",
)
_m, _v = _no_subject()
_add(
    id="load_004", category="load", description="Other dataset naming: number to folder name",
    model=_m, view=_v, prompt="Open patient 12",
    expected_tool_calls=[load("patient_012")], match_mode="exact",
    available_subjects=[f"patient_{n:03d}" for n in range(1, 41)],
)
_m, _v = _no_subject()
_add(
    id="load_list_001", category="load", description="Which subjects exist — a lookup, not a load",
    model=_m, view=_v, prompt="Which subjects can I open?",
    expected_tool_calls=[list_subjects()], match_mode="exact",
)


# === PRESET — apply_preset ===================================================

# --- Positive: functional (always available — CINE is enough) ---
for i, prompt in enumerate([
    "Assess LV function",
    "Show me the cardiac function",
    "I want to evaluate the left ventricle",
], start=1):
    _m, _v = _loaded(FULL, grid=(2, 2))
    _add(
        id=f"preset_func_{i:03d}", category="preset", description="LV function trigger phrase",
        model=_m, view=_v, prompt=prompt,
        expected_tool_calls=[preset("functional")], match_mode="subset",
    )

# --- Positive: viability (needs LGE) ---
for i, prompt in enumerate([
    "Set up a viability assessment",
    "Show the viability sequences",
    "I'm assessing for myocardial infarction",
], start=1):
    _m, _v = _loaded(FULL, grid=(2, 4))
    _add(
        id=f"preset_viab_{i:03d}", category="preset", description="Viability trigger phrase",
        model=_m, view=_v, prompt=prompt,
        expected_tool_calls=[preset("viability")], match_mode="subset",
    )

# --- Positive: perfusion (needs PERF_STRESS_SAX / PERF_REST_SAX / PERF_SAX) ---
for i, prompt in enumerate([
    "Show the perfusion imaging",
    "Perfusion assessment",
    "Stress perfusion protocol",
], start=1):
    _m, _v = _loaded(FULL_STRESS, grid=(2, 3))
    _add(
        id=f"preset_perf_{i:03d}", category="preset", description="Perfusion trigger phrase",
        model=_m, view=_v, prompt=prompt,
        expected_tool_calls=[preset("perfusion")], match_mode="subset",
    )

# --- Partial fill: preset offered, but not every panel can be filled ---
_m, _v = _loaded({"CINE_4CH": True}, grid=(2, 2))
_add(
    id="preset_partial_001", category="preset",
    description="Functional requested — only CINE_4CH loaded (3CH/2CH/SAX missing)",
    model=_m, view=_v, prompt="Assess LV function",
    expected_tool_calls=[preset("functional")], match_mode="subset",
    result_contains={"tool_name": "apply_preset", "contains": "1/4 panels filled"},
    notes="Partial-fill regression: some (not all) required series loaded, result text must say so.",
)

# --- True negative: the preset's data isn't loaded at all ---
_m, _v = _loaded(CINE_ONLY)
_add(
    id="preset_neg_001", category="preset", description="Viability requested — CINE-only subject (no LGE)",
    model=_m, view=_v, prompt="Set up a viability assessment",
    expected_tool_calls=[], match_mode="none",
    notes="The silent-no-op bug case from issue #12.",
)
_m, _v = _loaded(CINE_ONLY)
_add(
    id="preset_neg_002", category="preset", description="Perfusion requested — CINE-only subject",
    model=_m, view=_v, prompt="Stress perfusion protocol",
    expected_tool_calls=[], match_mode="none",
    notes="No perfusion or LGE data at all.",
)
_m, _v = _loaded(CINE_LGE, grid=(2, 3))
_add(
    id="preset_neg_003", category="preset", description="Perfusion requested — CINE+LGE loaded, no perfusion data",
    model=_m, view=_v, prompt="Show the perfusion imaging",
    expected_tool_calls=[], match_mode="none",
    notes="Harder than preset_neg_002: plenty is loaded, just not the perfusion series.",
)

# --- Extended functional preset ---------------------------------------------

for i, prompt in enumerate([
    "Show LV function per slice",
    "Extended functional view please",
    "I want to see all SAX slices",
], start=1):
    _m, _v = _loaded()
    _add(
        id=f"preset_funcx_{i:03d}", category="preset", description="Extended functional preset",
        model=_m, view=_v, prompt=prompt,
        expected_tool_calls=[preset("functional_complex")], match_mode="subset",
    )

# --- Not implemented yet: flow presets --------------------------------------
# Needs PC_AORTA / PC_MPA series keys out of cardisort plus flow / flow_aorta
# / flow_mpa preset entries. Both the positives and the "no PC data" negatives
# below are meaningless until then, since apply_preset's enum can't offer a
# preset that doesn't exist.
#
# for i, prompt in enumerate([
#     "Show me the flow sequences",
#     "Phase contrast imaging please",
#     "Show cardiac output estimation",
# ], start=1):
#     _m, _v = _loaded()
#     _add(
#         id=f"preset_flow_{i:03d}", category="preset", description="Flow preset",
#         model=_m, view=_v, prompt=prompt,
#         expected_tool_calls=[preset("flow")], match_mode="subset",
#     )
# for i, prompt in enumerate([
#     "Show aortic flow",
#     "Display the aortic phase contrast",
#     "Aortic regurgitation assessment",
# ], start=1):
#     _m, _v = _loaded()
#     _add(
#         id=f"preset_faorta_{i:03d}", category="preset", description="Aortic flow preset",
#         model=_m, view=_v, prompt=prompt,
#         expected_tool_calls=[preset("flow_aorta")], match_mode="subset",
#     )
# for i, prompt in enumerate([
#     "Show pulmonary flow",
#     "MPA phase contrast",
#     "Pulmonary regurgitation assessment",
# ], start=1):
#     _m, _v = _loaded()
#     _add(
#         id=f"preset_fmpa_{i:03d}", category="preset", description="MPA flow preset",
#         model=_m, view=_v, prompt=prompt,
#         expected_tool_calls=[preset("flow_mpa")], match_mode="subset",
#     )
# _m, _v = _loaded(CINE_LGE)
# _add(
#     id="preset_neg_flow_001", category="preset", description="Flow — no PC sequences loaded",
#     model=_m, view=_v, prompt="Show me the flow sequences",
#     expected_tool_calls=[], match_mode="none",
# )
# _m, _v = _loaded(CINE_LGE)
# _add(
#     id="preset_neg_flow_002", category="preset", description="Aortic flow — no PC_AORTA loaded",
#     model=_m, view=_v, prompt="Aortic flow assessment",
#     expected_tool_calls=[], match_mode="none",
# )


# === DISPLAY_EXPLICIT — series named by its exact key =========================

_m, _v = _loaded()
_add(
    id="display_ex_001", category="display_explicit", description="CINE_SAX by exact key",
    model=_m, view=_v, prompt="Show CINE_SAX in panel 2",
    expected_tool_calls=[display(2, "CINE_SAX")], match_mode="subset",
)
_m, _v = _loaded()
_add(
    id="display_ex_002", category="display_explicit", description="LGE_SAX by exact key",
    model=_m, view=_v, prompt="Display LGE_SAX in panel 0",
    expected_tool_calls=[display(0, "LGE_SAX")], match_mode="subset",
)
_m, _v = _loaded()
_add(
    id="display_ex_003", category="display_explicit", description="PERF_SAX by exact key",
    model=_m, view=_v, prompt="Put PERF_SAX in panel 1",
    expected_tool_calls=[display(1, "PERF_SAX")], match_mode="subset",
)
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_4CH"}])
_add(
    id="display_ex_004", category="display_explicit", description="Fill panel 1 without touching filled panel 0",
    model=_m, view=_v, prompt="Show CINE_SAX in panel 1",
    expected_tool_calls=[display(1, "CINE_SAX")], match_mode="subset",
    notes="Must not re-display panel 0 or call load_subject.",
)
_m, _v = _loaded()
_add(
    id="display_ex_neg_001", category="display_explicit", description="Key that doesn't exist in any catalog",
    model=_m, view=_v, prompt="Show CINE_5CH in panel 0",
    expected_tool_calls=[], match_mode="none",
)
_m, _v = _loaded()
_add(
    id="display_ex_neg_002", category="display_explicit", description="Real sequence, not acquired for this patient",
    model=_m, view=_v, prompt="Display T2STAR_SAX in panel 0",
    expected_tool_calls=[], match_mode="none",
)
_m, _v = _loaded(grid=(2, 2))
_add(
    id="display_ex_neg_003", category="display_explicit", description="Panel index out of range for the grid",
    model=_m, view=_v, prompt="Show LGE_SAX in panel 9",
    expected_tool_calls=[], match_mode="none",
)


# === DISPLAY_NATURAL — series described in clinical language ==================

_m, _v = _loaded()
_add(
    id="display_nat_001", category="display_natural", description="'Short-axis cine' -> CINE_SAX",
    model=_m, view=_v, prompt="Show the short-axis cine in panel 1",
    expected_tool_calls=[display(1, "CINE_SAX")], match_mode="subset",
)
_m, _v = _loaded()
_add(
    id="display_nat_002", category="display_natural", description="'4-chamber cine view' -> CINE_4CH",
    model=_m, view=_v, prompt="Display the 4-chamber cine view in panel 0",
    expected_tool_calls=[display(0, "CINE_4CH")], match_mode="subset",
)
_m, _v = _loaded()
_add(
    id="display_nat_003", category="display_natural", description="'Late gadolinium enhancement SAX' -> LGE_SAX",
    model=_m, view=_v, prompt="Show the late gadolinium enhancement short axis in panel 0",
    expected_tool_calls=[display(0, "LGE_SAX")], match_mode="subset",
)
_m, _v = _loaded(FULL_STRESS)
_add(
    id="display_nat_004", category="display_natural", description="'Stress perfusion images' -> PERF_STRESS_SAX",
    model=_m, view=_v, prompt="Show the stress perfusion images in panel 1",
    expected_tool_calls=[display(1, "PERF_STRESS_SAX")], match_mode="subset",
    notes="Must pick the stress series over the plain PERF_SAX also in the catalog.",
)
_m, _v = _loaded()
_add(
    id="display_nat_neg_001", category="display_natural", description="'T2 star mapping' — real sequence, not in catalog",
    model=_m, view=_v, prompt="Show T2 star mapping in panel 0",
    expected_tool_calls=[], match_mode="none",
)
_m, _v = _loaded()
_add(
    id="display_nat_neg_002", category="display_natural", description="'Myocardial oxygenation map' — no such thing",
    model=_m, view=_v, prompt="Display the myocardial oxygenation map in panel 1",
    expected_tool_calls=[], match_mode="none",
)

# --- Not implemented yet: series keys cardisort doesn't produce -------------
# CARDISORT_SEQUENCE_TO_PREFIX only maps CINE/DBLGE/WBLGE, so T1 mapping,
# black-blood anatomy and EGE never get a series_key and can't be displayed.
# Uncomment (here and in DISPLAY_MULTI below) once they're classified.
#
# _m, _v = _loaded({**FULL, "T1MAP_POST_SAX": True})
# _add(
#     id="display_ex_t1_001", category="display_explicit", description="T1 map by exact key",
#     model=_m, view=_v, prompt="Display T1MAP_POST_SAX in panel 2",
#     expected_tool_calls=[display(2, "T1MAP_POST_SAX")], match_mode="subset",
# )
# _m, _v = _loaded({**FULL, "T1MAP_POST_SAX": True})
# _add(
#     id="display_nat_t1_001", category="display_natural", description="'T1 map'",
#     model=_m, view=_v, prompt="Display the T1 map in panel 2",
#     expected_tool_calls=[display_any_panel("T1MAP_POST_SAX")], match_mode="subset",
# )
# _m, _v = _loaded({**FULL, "HASTE_SAX": True})
# _add(
#     id="display_nat_anat_001", category="display_natural", description="'Black blood anatomy'",
#     model=_m, view=_v, prompt="Show the black blood anatomy scan in panel 0",
#     expected_tool_calls=[display(0, "HASTE_SAX")], match_mode="subset",
# )


# === DISPLAY_MULTI — one instruction, several panels ==========================

_m, _v = _loaded()
_add(
    id="display_multi_001", category="display_multi", description="Two explicit series in two specific panels",
    model=_m, view=_v, prompt="Show CINE_2CH in panel 0 and LGE_SAX in panel 1",
    expected_tool_calls=[display(0, "CINE_2CH"), display(1, "LGE_SAX")], match_mode="any_order",
)
_m, _v = _loaded(grid=(2, 2))
_add(
    id="display_multi_002", category="display_multi", description="All 4 CINE planes across 4 panels",
    model=_m, view=_v, prompt="Display all CINE planes in different panels",
    expected_tool_calls=[display_any_panel("CINE_SAX"), display_any_panel("CINE_2CH"),
                         display_any_panel("CINE_3CH"), display_any_panel("CINE_4CH")],
    match_mode="subset",
)
_m, _v = _loaded(grid=(2, 4))
_add(
    id="display_multi_003", category="display_multi", description="'All LGE views' should reach for the preset",
    model=_m, view=_v, prompt="Show all LGE views",
    expected_tool_calls=[preset("viability")], match_mode="subset",
    notes="Four display_series calls would also fill the screen, but the preset is the intended route.",
)
_m, _v = _loaded()
_add(
    id="display_multi_004", category="display_multi", description="All long-axis cine planes (3 LAX)",
    model=_m, view=_v, prompt="Show all long-axis cine views",
    expected_tool_calls=[display_any_panel("CINE_2CH"), display_any_panel("CINE_3CH"), display_any_panel("CINE_4CH")],
    match_mode="subset",
    notes="CINE_SAX must NOT appear.",
)
_m, _v = _displaying(
    [{"idx": 0, "series_key": "CINE_2CH"}, {"idx": 1, "series_key": "CINE_3CH"},
     {"idx": 2, "series_key": "CINE_4CH"}, {"idx": 3, "series_key": "CINE_SAX"}],
)
_add(
    id="display_multi_neg_001", category="display_multi", description="All CINE already shown — nothing to do",
    model=_m, view=_v, prompt="Display all CINE planes in different panels",
    expected_tool_calls=[], match_mode="none",
)
_m, _v = _loaded()
_add(
    id="display_multi_neg_002", category="display_multi", description="'All T2 star planes' — not in catalog",
    model=_m, view=_v, prompt="Show all T2 star planes",
    expected_tool_calls=[], match_mode="none",
)
_m, _v = _loaded()
_add(
    id="display_multi_neg_003", category="display_multi", description="Two nonexistent keys — must refuse both",
    model=_m, view=_v, prompt="Display CINE_5CH and CINE_6CH in panels 0 and 1",
    expected_tool_calls=[], match_mode="none",
)

# --- Not implemented yet: EGE across planes (see the series-key note above) ---
#
# _m, _v = _loaded({**FULL, "EGE_SAX": True, "EGE_2CH": True, "EGE_4CH": True})
# _add(
#     id="display_multi_ege_001", category="display_multi", description="Early gadolinium enhancement, all planes",
#     model=_m, view=_v, prompt="Display early gadolinium enhancement in all available planes",
#     expected_tool_calls=[display_any_panel("EGE_SAX"), display_any_panel("EGE_2CH"),
#                          display_any_panel("EGE_4CH")],
#     match_mode="subset",
# )


# === DISPLAY_POSITIONAL — panel_idx = row*cols + col regression guard =========

for i, (prompt, grid, idx, key) in enumerate([
    ("Show CINE_SAX in the bottom right panel", (2, 4), 7, "CINE_SAX"),
    ("Show CINE_2CH in the top right panel", (2, 4), 3, "CINE_2CH"),
    ("Show LGE_SAX in the bottom left panel", (2, 4), 4, "LGE_SAX"),
    ("Show CINE_SAX in the rightmost panel", (1, 3), 2, "CINE_SAX"),
    ("Show CINE_3CH in the middle panel", (1, 3), 1, "CINE_3CH"),
    ("Display CINE_4CH in the top left panel", (2, 3), 0, "CINE_4CH"),
    ("Show LGE_SAX in the bottom right panel", (2, 3), 5, "LGE_SAX"),
], start=1):
    _m, _v = _loaded(grid=grid)
    _add(
        id=f"display_pos_{i:03d}", category="display_positional",
        description=f"Positional panel addressing, {grid[0]}x{grid[1]} grid",
        model=_m, view=_v, prompt=prompt,
        expected_tool_calls=[display(idx, key)], match_mode="subset",
    )

for i, (prompt, grid) in enumerate([
    ("Show CINE_SAX in panel 8", (2, 4)),
    ("Show CINE_SAX in the 5th row", (2, 2)),
], start=1):
    _m, _v = _loaded(grid=grid)
    _add(
        id=f"display_pos_neg_{i:03d}", category="display_positional",
        description="Out-of-range panel/row",
        model=_m, view=_v, prompt=prompt,
        expected_tool_calls=[], match_mode="none",
    )

# --- Not implemented yet: grid sizing ---------------------------------------
# There is no set_grid tool — grid size only changes as a side effect of
# apply_preset. Uncomment when a grid tool lands in src/llm/tools.py.
#
# for i, (prompt, grid) in enumerate([
#     ("Set a 2x3 grid", (2, 3)),
#     ("Change the grid to 1 row and 3 columns", (1, 3)),
#     ("I need a 1 by 4 grid", (1, 4)),
#     ("Give me a 3-column single-row layout", (1, 3)),
#     ("Resize to 2 rows and 4 columns", (2, 4)),
# ], start=1):
#     _m, _v = _loaded()
#     _add(
#         id=f"grid_{i:03d}", category="grid", description="Explicit grid resize",
#         model=_m, view=_v, prompt=prompt,
#         expected_tool_calls=[{"name": "set_grid", "arguments": {"rows": grid[0], "cols": grid[1]}}],
#         match_mode="subset",
#     )


# === DISPLAY_STATE — decisions that depend on what's already on screen ========

_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}, {"idx": 1, "series_key": "CINE_4CH"},
                      {"idx": 2, "series_key": "CINE_2CH"}])
_add(
    id="display_state_001", category="display_state", description="Fill whichever panel is currently empty",
    model=_m, view=_v, prompt="Show LGE_SAX in whichever panel is currently empty",
    expected_tool_calls=[display(3, "LGE_SAX")], match_mode="exact",
)
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_4CH"}, {"idx": 1, "series_key": "CINE_2CH"},
                      {"idx": 2, "series_key": "CINE_SAX"}])
_add(
    id="display_state_002", category="display_state", description="Fill the last empty panel with LGE SAX",
    model=_m, view=_v, prompt="Fill the remaining empty panel with the LGE SAX",
    expected_tool_calls=[display(3, "LGE_SAX")], match_mode="exact",
)
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}, {"idx": 1, "series_key": "CINE_2CH"},
                      {"idx": 2, "series_key": "CINE_4CH"}])
_add(
    id="display_state_003", category="display_state",
    description="Replace all CINE panels with LGE equivalents, same planes",
    model=_m, view=_v, prompt="Replace all CINE views with the LGE equivalent, same planes",
    expected_tool_calls=[display(0, "LGE_SAX"), display(1, "LGE_2CH"), display(2, "LGE_4CH")],
    match_mode="any_order",
)

# --- Not implemented yet: per-panel slice state -----------------------------
# display_series has no slice_position parameter and the app doesn't track a
# per-panel slice index, so "same slice" / "same level" / "same depth as panel
# N" can't be expressed as a tool call. Uncomment once display_series takes a
# slice argument.
#
# for i, prompt in enumerate([
#     "Show LGE_SAX in panel 0 at the same slice",
#     "Replace the cine SAX in panel 2 with LGE SAX, keeping the same level",
#     "Show CINE_SAX in panel 1 at the same depth as panel 0",
#     "Display LGE_SAX in panel 3 at the same anatomical level as panel 0",
#     "Replace every CINE panel with the LGE equivalent, same level where applicable",
# ], start=1):
#     _m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}, {"idx": 2, "series_key": "CINE_SAX"}])
#     _add(
#         id=f"display_state_slice_{i:03d}", category="display_state",
#         description="Slice-preserving replace",
#         model=_m, view=_v, prompt=prompt,
#         expected_tool_calls=[], match_mode="subset",  # fill in once the argument exists
#     )


# === SEGMENTATION ============================================================

_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}])
_add(
    id="segmentation_001", category="segmentation", description="Segment the displayed CINE panel",
    model=_m, view=_v, prompt="Segment panel 0",
    expected_tool_calls=[segment("CINE_SAX")], match_mode="exact",
)
_m, _v = _displaying([{"idx": 0, "series_key": "LGE_SAX"}])
_add(
    id="segmentation_002", category="segmentation", description="Segment the displayed LGE panel",
    model=_m, view=_v, prompt="Segment panel 0",
    expected_tool_calls=[segment("LGE_SAX")], match_mode="exact",
    notes="run_segmentation offers every loaded CINE modality plus LGE_SAX — see build_tools().",
)
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}, {"idx": 1, "series_key": "LGE_SAX"}])
_add(
    id="segmentation_003", category="segmentation", description="Segment every series currently on screen",
    model=_m, view=_v, prompt="Segment everything that's on screen",
    expected_tool_calls=[segment("CINE_SAX"), segment("LGE_SAX")], match_mode="any_order",
    notes="MULTI-TARGET rule names run_segmentation explicitly. any_order, not subset: the catalog has "
          "nine series loaded but only two displayed, so segmenting a third is wrong, not extra credit.",
)
_m, _v = _loaded()
_add(
    id="segmentation_004", category="segmentation", description="Modality named in clinical language, no panel index",
    model=_m, view=_v, prompt="Run the segmentation on the short-axis cine",
    expected_tool_calls=[segment("CINE_SAX")], match_mode="subset",
    notes="run_segmentation keys off loaded series, not what's displayed — nothing is in a panel here.",
)
_m, _v = _displaying([{"idx": 0, "series_key": "PERF_SAX"}])
_add(
    id="segmentation_neg_001", category="segmentation", description="Segment a panel showing an unsegmentable series",
    model=_m, view=_v, prompt="Segment panel 0",
    expected_tool_calls=[], match_mode="none",
    notes="PERF_SAX has no segmentation model, so it's absent from run_segmentation's enum.",
)


# === REFUSAL — out-of-scope / clinical-judgment questions =====================

_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}])
_add(
    id="refusal_001", category="refusal", description="Diagnostic interpretation question",
    model=_m, view=_v, prompt="Is this myocarditis?",
    expected_tool_calls=[], match_mode="none",
)
_m, _v = _loaded()
_add(
    id="refusal_002", category="refusal", description="Clinical advisory question",
    model=_m, view=_v, prompt="Should I give this patient a stress test based on what you see?",
    expected_tool_calls=[], match_mode="none",
)
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX", "has_segmentation": True},
                      {"idx": 1, "series_key": "LGE_SAX"}])
_add(
    id="refusal_003", category="refusal", description="Report generation — out-of-scope task",
    model=_m, view=_v, prompt="Write a radiology report for this patient",
    expected_tool_calls=[], match_mode="none",
)
_m, _v = _loaded()
_add(
    id="refusal_004", category="refusal", description="Clinical question containing a preset keyword",
    model=_m, view=_v, prompt="What sequences should I use for LV function?",
    expected_tool_calls=[], match_mode="none",
    notes="Contains the 'LV function' shortcut keyword but is a question — no tools.",
)


# === COMPOUND — multi-step ====================================================
#
# The harness runs ONE turn (driver.run_one), so what a compound case can
# fairly assert depends on where it starts:
#
#   From _no_subject(), build_tools() offers list_subjects/load_subject only —
#   no series are loaded, so apply_preset/display_series/run_segmentation are
#   all filtered out. The only correct output is the load call, which makes
#   match_mode="exact" a real test: it catches a model inventing calls to
#   tools it was never given.
#
#   From a loaded subject, every tool is on the table and the model can
#   finish the whole request in one reply. The SYSTEM_PROMPT's MULTI-STEP
#   rule also permits one tool per turn ("continuing one tool at a time
#   after each result is just as correct"), so these cases specifically
#   measure the all-at-once path — the same judgment the display_multi_*
#   cases already make. A model that instead chains over follow-up turns is
#   not broken, it just scores 0 here; see README.md.

_m, _v = _no_subject()
_add(
    id="compound_001", category="compound", description="Load + apply preset in one request",
    model=_m, view=_v, prompt="Load subject 3 and set up the functional view",
    expected_tool_calls=[load("QLGE03_")], match_mode="subset",
    notes="Single-turn harness only sees the load call; apply_preset legitimately comes in a follow-up turn.",
)
_m, _v = _no_subject()
_add(
    id="compound_002", category="compound", description="Four-step request with nothing loaded yet",
    model=_m, view=_v,
    prompt="Load subject 1, display the viability preset, segment the cine 4ch, cine sax and lge sax",
    expected_tool_calls=[load("QLGE01_")], match_mode="exact",
    notes="exact, not subset: apply_preset/run_segmentation aren't offered at all here, so any "
          "call to them is a hallucinated tool, not an eager-but-correct next step.",
)
_m, _v = _loaded(FULL, grid=(2, 4))
_add(
    id="compound_003", category="compound", description="Preset + segment three series, subject already loaded",
    model=_m, view=_v,
    prompt="Display the viability preset, then segment the cine 4ch, cine sax and lge sax",
    expected_tool_calls=[preset("viability"), segment("CINE_4CH"), segment("CINE_SAX"), segment("LGE_SAX")],
    match_mode="subset",
    notes="compound_002's request minus the load, so every step is actually reachable in one turn.",
)
_m, _v = _loaded(FULL, grid=(2, 2))
_add(
    id="compound_004", category="compound", description="Preset + segment + panel override",
    model=_m, view=_v,
    prompt="Display the functional view, segment the cine sax and replace the cine 2ch with the lge sax",
    expected_tool_calls=[preset("functional"), segment("CINE_SAX"), display_any_panel("LGE_SAX")],
    match_mode="subset",
    notes="panel_idx deliberately not asserted: the functional preset puts CINE_2CH in panel 2, but "
          "nothing in SYSTEM_PROMPT states a preset's panel layout — the model can only learn it from "
          "apply_preset's result, which a single turn never sees. Mirrors the preset+override few-shot "
          "example in llm/prompt.py.",
)


# --- Not implemented yet: analysis / zoom / playback tools -------------------
# No run_analysis, no zoom_to_roi, and no play/pause — set_time_frame jumps to
# a specific frame, it doesn't start or stop a cine loop. Uncomment per tool as
# each one is added to TOOL_SPECS.
#
# _m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX", "has_segmentation": True}])
# _add(
#     id="analysis_001", category="analysis", description="Run analysis after segmentation",
#     model=_m, view=_v, prompt="Analyse panel 0",
#     expected_tool_calls=[{"name": "run_analysis", "arguments": {"panel_idx": 0}}], match_mode="subset",
# )
# _m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX", "has_segmentation": True}])
# _add(
#     id="zoom_001", category="zoom", description="Zoom into a segmented panel",
#     model=_m, view=_v, prompt="Zoom into panel 0",
#     expected_tool_calls=[{"name": "zoom_to_roi", "arguments": {"panel_idx": 0}}], match_mode="subset",
# )
# for i, (prompt, playing) in enumerate([("Start the cine playback", True), ("Pause the cines", False)], start=1):
#     _m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}])
#     _add(
#         id=f"playback_{i:03d}", category="playback", description="Start/stop cine playback",
#         model=_m, view=_v, prompt=prompt,
#         expected_tool_calls=[{"name": "set_playback", "arguments": {"playing": playing}}], match_mode="subset",
#     )
