"""Eval test cases for qcardia-app-dev's ChatController tool-calling.

One suite, one vocabulary: every case is written against the app's real
tools and argument names — `search_patients(query=)`, `load_subject(subject_id=)`,
`display_series(panel_idx=, series_key=, axis=)`, `apply_preset(preset=)`,
`run_segmentation(modality=)`. See fakes.py for the fixture classes and
src/llm/ for the real prompt/tool specs this harness exercises unmodified.

Capabilities the app doesn't have yet — flow presets, analysis/playback
tools, and the T1/EGE/anatomy series keys cardisort doesn't produce — have
their cases written out but **commented out**, each under a header saying
what has to land before they're uncommented. Nothing is skipped at runtime:
if a case is in TEST_CASES it gets run and it counts towards the pass rate.

match_mode:
    "exact"      — same calls in same order, no extras
    "any_order"  — same calls, any order, no extras
    "subset"     — every expected call appears somewhere in actual (extras OK)
    "none"       — actual tool_calls must be empty
    "state"      — the calls are run on the fake viewer, and expected_state
                   checks the screen they lead to, whatever route got there
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

from fakes import (
    DEFAULT_CINE_MODALITIES,
    FakeDicomController,
    FakeDisplayAreaWidget,
    FakeModel,
    FakeView,
    phantom_analysis,
    phantom_depths,
)


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
    history: List[dict] = field(default_factory=list)  # earlier turns, before prompt
    # For one expected call (matched by name), assert _execute_tool()'s result
    # text contains this substring — for regression cases where the result
    # text itself is what's being checked (e.g. partial-fill reporting).
    result_contains: dict | None = None  # {"tool_name": str, "contains": str}
    # match_mode "state": takes driver.screen_after()'s dict, returns why it is wrong, or None.
    expected_state: Callable[[dict], str | None] | None = None


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
STRESS_REST = {**CINE_LGE, "PERF_STRESS_SAX": True, "PERF_REST_SAX": True}


# --- Fixture helpers --------------------------------------------------------

def _model(series: dict, subject: str | None = "QLGE01_", analysis: dict | None = None) -> FakeModel:
    return FakeModel(
        series=series,
        folder_path=Path(subject) if subject else None,
        num_time_frames=30 if "CINE_SAX" in series else 0,
        cine_modalities=DEFAULT_CINE_MODALITIES,
        analysis=analysis,
    )


def _view(grid=(2, 2), panels=None, registered_series: set[str] | None = None) -> FakeView:
    return FakeView(FakeDisplayAreaWidget(grid_size=grid, panels=panels or [], registered_series=registered_series))


def _segmented(panels: list[dict] = (), grid=(2, 2), catalog: dict = None, lge: bool = False) -> tuple[FakeModel, FakeView]:
    """As _displaying, with the CINE_SAX and CINE_4CH segmentations measured (see fakes.phantom_analysis),
    and with lge the LGE_SAX one too (fakes.phantom_lge)."""
    catalog = catalog or FULL
    analysis = phantom_analysis()
    if lge:
        analysis["LGE_SAX"] = {"box": (30, 30, 66, 66), "myo_ml": 110.0, "scar_ml": 6.0, "scar_slices": [4, 5, 6]}
    return (_model(catalog, analysis=analysis),
            _view(grid=grid, panels=list(panels), registered_series=set(catalog)))


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

def search_patients() -> dict:
    return {"name": "search_patients", "arguments": {}}


def loaded(folder: str):
    """The state check that exactly this patient's study was loaded, by whatever name, ID or number."""
    return lambda s: None if s["loaded"] == [folder] else f"loaded {s['loaded'] or 'nothing'}, expected {folder}"


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


# --- Screen checks for match_mode "state": each returns why the screen is wrong, or None ---------

def screen(*checks: Callable[[dict], str | None]) -> Callable[[dict], str | None]:
    def check_all(s: dict) -> str | None:
        return next((p for p in (c(s) for c in checks) if p), None)
    return check_all


def grid_is(rows: int, cols: int):
    return lambda s: None if s["grid"] == (rows, cols) else f"grid {s['grid']}, expected {(rows, cols)}"


def cells_are(*cells: tuple[int, int, int, int]):
    return lambda s: None if s["cells"] == sorted(cells) else f"cells {s['cells']}, expected {sorted(cells)}"


def frame_is(frame: int):
    return lambda s: None if s["frame"] == frame else f"frame {s['frame']}, expected {frame}"


def panel_is(idx: int, key: str, slice_no: int | None = None):
    def check(s):
        if idx >= len(s["panels"]):
            return f"no panel {idx}"
        k, sl, _ = s["panels"][idx]
        if k != key or (slice_no is not None and sl != slice_no):
            return f"panel {idx} shows {k} slice {sl}, expected {key}" + (f" slice {slice_no}" if slice_no else "")
    return check


def slices_shown(key: str, slices: list[int]):
    """These slices of key are on screen, one panel each, in any panel order."""
    def check(s):
        shown = sorted(sl for k, sl, _ in s["panels"] if k == key)
        return None if shown == sorted(slices) else f"{key} slices shown {shown}, expected {sorted(slices)}"
    return check


def zoomed(zoom, panels: list[int] | None = None):
    """The given panels (default: every panel with a series) have this zoom."""
    def check(s):
        idxs = panels if panels is not None else [i for i, (k, _, _) in enumerate(s["panels"]) if k]
        wrong = {i: s["panels"][i][2] for i in idxs if s["panels"][i][2] != zoom}
        return None if not wrong else f"zoom {wrong}, expected {zoom!r}"
    return check


def only_zoomed(panels: list[int]):
    def check(s):
        others = [i for i, (_, _, z) in enumerate(s["panels"]) if z is not None and i not in panels]
        return None if not others else f"panels {others} zoomed too"
    return check


def nothing_segmented(s):
    return f"ran run_segmentation({s['segmentations']}) without asking" if s["segmentations"] else None


def nearest(series_key: str, depth: float) -> int:
    """1-based slice of series_key nearest to depth in the phantom."""
    depths = phantom_depths(series_key)
    return min(range(len(depths)), key=lambda i: abs(depths[i] - depth)) + 1


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
    expected_tool_calls=[], match_mode="state", expected_state=loaded("QLGE03_"),
)
_m, _v = _no_subject()
_add(
    id="load_002", category="load", description="Load via 'open patient' synonym",
    model=_m, view=_v, prompt="Open patient 7",
    expected_tool_calls=[], match_mode="state", expected_state=loaded("QLGE07_"),
)
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_2CH"}])
_add(
    id="load_003", category="load", description="Switch subject while one is already loaded",
    model=_m, view=_v, prompt="Switch to subject 5",
    expected_tool_calls=[], match_mode="state", expected_state=loaded("QLGE05_"),
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
    expected_tool_calls=[], match_mode="state", expected_state=loaded("patient_012"),
    available_subjects=[f"patient_{n:03d}" for n in range(1, 41)],
)
_m, _v = _no_subject()
_add(
    id="load_list_001", category="load", description="Which patients exist — a lookup, not a load",
    model=_m, view=_v, prompt="Which patients can I open?",
    expected_tool_calls=[search_patients()], match_mode="exact",
)
_m, _v = _no_subject()
_add(
    id="load_005", category="load", description="Load by name",
    model=_m, view=_v, prompt="Load Adam James",
    expected_tool_calls=[], match_mode="state", expected_state=loaded("QLGE08_"),
    notes="search_patients finds one Adam James (ID 08HD8H7Y9K); load_subject with that ID.",
)
_m, _v = _no_subject()
_add(
    id="load_006", category="load", description="Load by hospital ID",
    model=_m, view=_v, prompt="Open the scans of patient 13HD3H7Y9K",
    expected_tool_calls=[], match_mode="state", expected_state=loaded("QLGE13_"),
)
_m, _v = _no_subject()
_add(
    id="load_007", category="load", description="A misheard name: ask, do not load",
    model=_m, view=_v, prompt="Load Adam Jams",
    expected_tool_calls=[], match_mode="none",
    notes="No exact match; Adam James only sounds alike, so ask whether that is the patient.",
)
_m, _v = _no_subject()
_add(
    id="load_008", category="load", description="Two patients with that name: ask which",
    model=_m, view=_v, prompt="Load Emily",
    expected_tool_calls=[], match_mode="none",
    notes="Emily Clarke and Emily Carter: ask which one.",
)
_m, _v = _no_subject()
_add(
    id="load_neg_003", category="load", description="A name the database does not have",
    model=_m, view=_v, prompt="Load Margaret Thatcher",
    expected_tool_calls=[], match_mode="none",
)
_LOADED_EMILY = [{"role": "user", "content": "load patient 11HD1H7Y9K"},
                 {"role": "assistant", "content": "Loaded the imaging of Emily Clarke, ID 11HD1H7Y9K."}]
_m, _v = _loaded()
_add(
    id="load_009", category="load", description="A question about other patients is a search",
    model=_m, view=_v, prompt="Is there another Emily?", history=list(_LOADED_EMILY),
    expected_tool_calls=[search_patients()], match_mode="exact",
    notes="Emily Carter too: only a search knows. Answer, do not load.",
)
_m, _v = _loaded()
_add(
    id="load_010", category="load", description="Search again after an earlier wrong 'no such patient'",
    model=_m, view=_v, prompt="Load Emely",
    history=[*_LOADED_EMILY, {"role": "user", "content": "how about an emely"},
             {"role": "assistant", "content": "No patient named Emely is in the database."}],
    expected_tool_calls=[search_patients()], match_mode="exact",
    notes="The earlier answer came without a search. Emely sounds like Emily Clarke and Emily Carter: ask which.",
)
_m, _v = _no_subject()
_add(
    id="load_011", category="load", description="One name that sounds alike: ask, do not load it",
    model=_m, view=_v, prompt="Load Grace Walkr",
    expected_tool_calls=[search_patients()], match_mode="exact",
    notes="Ask 'Did you mean Grace Walker?'; searching that name and loading it is the model guessing.",
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
    "Show the cines next to the LGE",
], start=1):
    _m, _v = _loaded(FULL, grid=(2, 4))
    _add(
        id=f"preset_viab_{i:03d}", category="preset", description="Viability trigger phrase",
        model=_m, view=_v, prompt=prompt,
        expected_tool_calls=[preset("viability")], match_mode="subset",
    )

# --- Positive: LGE (needs LGE) ---
for i, prompt in enumerate([
    "I'm assessing for myocardial infarction",
    "Show me the scar imaging",
    "Let me look at the late enhancement",
], start=1):
    _m, _v = _loaded(FULL, grid=(2, 2))
    _add(
        id=f"preset_lge_{i:03d}", category="preset", description="LGE trigger phrase",
        model=_m, view=_v, prompt=prompt,
        expected_tool_calls=[preset("lge")], match_mode="subset",
    )

# --- Positive: perfusion with LGE (needs perfusion and LGE_SAX) ---
for i, prompt in enumerate([
    "Show the perfusion with LGE",
    "I want the perfusion and scar images together",
], start=1):
    _m, _v = _loaded(FULL_STRESS, grid=(2, 3))
    _add(
        id=f"preset_perflge_{i:03d}", category="preset", description="Perfusion with LGE trigger phrase",
        model=_m, view=_v, prompt=prompt,
        expected_tool_calls=[preset("perfusion_lge")], match_mode="subset",
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
    "Show me all the cine SAX slices next to the LAX ones",
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
    id="display_multi_003", category="display_multi", description="All LGE views on screen",
    model=_m, view=_v, prompt="Show all LGE views",
    expected_tool_calls=[], match_mode="state",
    expected_state=lambda s: None if {"LGE_SAX", "LGE_2CH", "LGE_3CH", "LGE_4CH"} <= {k for k, _, _ in s["panels"]}
    else f"panels show {[k for k, _, _ in s['panels']]}",
    notes="The viability preset or four display_series calls both do it.",
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

# === GRID — set_grid: sizes and shifting the panels; set_split_grid: two halves ==

for i, (prompt, grid) in enumerate([
    ("Set a 2x3 grid", (2, 3)),
    ("Change the grid to 1 row and 3 columns", (1, 3)),
    ("I need a 3 by 3 layout", (3, 3)),
    ("Give me a 3-column single-row layout", (1, 3)),
], start=1):
    _m, _v = _loaded()
    _add(
        id=f"grid_{i:03d}", category="grid", description="Explicit grid size",
        model=_m, view=_v, prompt=prompt,
        expected_tool_calls=[], match_mode="state", expected_state=screen(grid_is(*grid)),
    )
_m, _v = _loaded()
_add(
    id="grid_005", category="grid", description="One large panel on the left, a 2x2 on the right",
    model=_m, view=_v, prompt="Give me one large panel on the left and a 2x2 grid on the right",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(grid_is(2, 4), cells_are((0, 0, 2, 2), (0, 2, 1, 1), (0, 3, 1, 1), (1, 2, 1, 1),
                                                     (1, 3, 1, 1))),
    notes="set_split_grid(1, 1, 2, 2): the large panel is as large as the 2x2 beside it.",
)
_m, _v = _loaded()
_add(
    id="grid_007", category="grid", description="One large panel with a 3x3 beside it",
    model=_m, view=_v, prompt="One big panel on the left and a 3 by 3 grid next to it",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(grid_is(3, 6), lambda s: None if s["cells"][0] == (0, 0, 3, 3) and len(s["cells"]) == 10
                          else f"cells {s['cells']}"),
)
_m, _v = _segmented(lge=True)
_add(
    id="grid_008", category="grid", description="LGE large on the left, the four cines in a 2x2 beside it",
    model=_m, view=_v,
    prompt="Show the LGE SAX mid slice large on the left with the four cines in a 2x2 next to it",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(grid_is(2, 4), panel_is(0, "LGE_SAX", 6),
                          lambda s: None if {k for k, _, _ in s["panels"][1:5]} == {"CINE_SAX", "CINE_2CH", "CINE_3CH",
                                                                                   "CINE_4CH"}
                          else f"panels 1-4 show {[k for k, _, _ in s['panels'][1:5]]}"),
)
_m, _v = _displaying([{"idx": i, "series_key": k} for i, k in
                      enumerate(["CINE_SAX", "LGE_SAX", "CINE_3CH", "LGE_4CH", "LGE_2CH", "LGE_3CH"])], grid=(2, 3))
_add(
    id="grid_006", category="grid", description="Make room on the left of the current view for two LAX cines",
    model=_m, view=_v, prompt="Put the 2 chamber and 4 chamber cine on the left of the current view",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(
        grid_is(2, 4), panel_is(1, "CINE_SAX"), panel_is(2, "LGE_SAX"), panel_is(3, "CINE_3CH"),
        panel_is(5, "LGE_4CH"), panel_is(6, "LGE_2CH"), panel_is(7, "LGE_3CH"),
        lambda s: None if {s["panels"][0][0], s["panels"][4][0]} == {"CINE_2CH", "CINE_4CH"}
        else f"left column shows {s['panels'][0][0]}, {s['panels'][4][0]}",
    ),
    notes="The six panels keep their places relative to each other: set_grid(2, 4, shift_cols=1), then "
          "display_series in panels 0 and 4.",
)
def _split_screen(panels: dict[int, str], split=(1, 1, 2, 2)):
    """The phantom with the CINE and LGE segmentations measured, in a split layout showing panels."""
    m, v = _segmented(lge=True)
    w = v.display_area_widget
    w.set_split_layout(split)
    for idx, key in panels.items():
        w.set_panel_series(idx, key)
    return m, v


def split_is(split):
    return lambda s: None if s["split"] == tuple(split) else f"split {s['split']}, expected {tuple(split)}"


_CINES = ["CINE_SAX", "CINE_2CH", "CINE_3CH", "CINE_4CH"]
_m, _v = _split_screen({0: "LGE_SAX", **{i + 1: k for i, k in enumerate(_CINES)}})
_add(
    id="split_001", category="grid", description="Large panel on the right, a 3x3 on the left",
    model=_m, view=_v, prompt="Can you also make a large one on the right with a 3x3 on the left",
    expected_tool_calls=[], match_mode="state", expected_state=split_is((3, 3, 1, 1)),
    notes="set_split_grid(3, 3, 1, 1): panels 0-8 on the left, 9 the large one.",
)
_m, _v = _loaded()
_add(
    id="split_002", category="grid", description="Two 2x2 grids side by side",
    model=_m, view=_v, prompt="Give me two 2x2 grids next to each other",
    expected_tool_calls=[], match_mode="state",
    expected_state=lambda s: None if sorted(s["cells"]) == [(r, c, 1, 1) for r in range(2) for c in range(4)]
    else f"cells {s['cells']}",
    notes="set_split_grid(2, 2, 2, 2) or set_grid(2, 4) look the same.",
)
_m, _v = _split_screen({0: "LGE_SAX", **{i + 1: k for i, k in enumerate(_CINES)}})
_add(
    id="split_003", category="grid", description="Move the large LGE to the right, the cines in a 2x2 left",
    model=_m, view=_v,
    prompt="Put the LGE in a large panel on the right and the four cines in a 2x2 on the left",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(split_is((2, 2, 1, 1)), panel_is(4, "LGE_SAX"), lambda s: None if sorted(k for k, _, _ in s["panels"][:4]) == sorted(_CINES)
                          else f"panels 0-3 show {[k for k, _, _ in s['panels'][:4]]}"),
    notes="set_split_grid(2, 2, 1, 1) keeps panels by number, so the LGE (was 0) ends up small: show it in "
          "panel 4 and the cines in 0-3.",
)
_m, _v = _loaded()
_add(
    id="grid_neg_001", category="grid", description="Grid larger than the viewer supports",
    model=_m, view=_v, prompt="Set a 6x8 grid",
    expected_tool_calls=[], match_mode="none",
    notes="set_grid allows 1-4 rows and 1-8 columns.",
)


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

# === SLICE — slice numbers and positions (the phantom's LGE_SAX has 12 slices to CINE_SAX's 14) ===

_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}])
_add(
    id="slice_001", category="slice", description="A slice number",
    model=_m, view=_v, prompt="Show slice 5 of the cine SAX in panel 1",
    expected_tool_calls=[], match_mode="state", expected_state=screen(panel_is(1, "CINE_SAX", 5)),
)
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX", "slice": 8}])
_add(
    id="slice_002", category="slice", description="Another stack at the depth of a panel",
    model=_m, view=_v, prompt="Show the LGE SAX in panel 1 at the same depth as panel 0",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(panel_is(0, "CINE_SAX", 9), panel_is(1, "LGE_SAX", nearest("LGE_SAX", -73.3))),
    notes="display_series(1, LGE_SAX, match_panel=0).",
)
# Issue #34: the extended functional layout, then all cine swapped for LGE at the same depths.
_lax = {0: "4CH", 6: "3CH", 12: "2CH"}
_sax = [i for i in range(18) if i not in _lax][:14]
_m, _v = _displaying([{"idx": i, "series_key": f"CINE_{p}"} for i, p in _lax.items()]
                     + [{"idx": i, "series_key": "CINE_SAX", "slice": z} for z, i in enumerate(_sax)],
                     grid=(3, 6))
_add(
    id="slice_003", category="slice", description="Swap every cine for LGE, SAX panels keep their depth (#34)",
    model=_m, view=_v, prompt="Swap out all the cine images for LGE",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(*[panel_is(i, f"LGE_{p}") for i, p in _lax.items()],
                          *[panel_is(i, "LGE_SAX", nearest("LGE_SAX", phantom_depths("CINE_SAX")[z]))
                            for z, i in enumerate(_sax)]),
    notes="One display_series per panel with no slice: each SAX panel keeps its position, as LGE slice "
          "numbers differ from cine.",
)


def _spread_cine_sax(count: int):
    def check(s):
        shown = [sl for k, sl, _ in s["panels"] if k == "CINE_SAX"]
        if len(shown) != count or len(set(shown)) != count:
            return f"CINE_SAX slices shown {shown}, expected {count} different ones"
        if shown not in (sorted(shown), sorted(shown, reverse=True)):
            return f"CINE_SAX slices {shown} are not in order across the panels"
    return check


_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}])
_add(
    id="slice_004", category="slice", description="Nine SAX slices spread over a 3x3",
    model=_m, view=_v, prompt="Show 9 of the cine SAX slices in a 3x3",
    expected_tool_calls=[], match_mode="state", expected_state=screen(grid_is(3, 3), _spread_cine_sax(9)),
)
_m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}])
_add(
    id="slice_005", category="slice", description="The apex at the current frame",
    model=_m, view=_v, prompt="Show the most apical cine SAX slice that still has LV, myocardium and RV in panel 0",
    expected_tool_calls=[], match_mode="state", expected_state=screen(panel_is(0, "CINE_SAX", 3)),
)
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}])
_add(
    id="slice_neg_001", category="slice", description="Slice beyond the stack",
    model=_m, view=_v, prompt="Show slice 20 of the cine SAX in panel 0",
    expected_tool_calls=[], match_mode="none", notes="CINE_SAX has 14 slices.",
)


# === FRAME — frame numbers and ED/ES from the segmentation (phantom: ED = 28, ES = 11) ===

_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}])
_add(
    id="frame_001", category="frame", description="A frame number",
    model=_m, view=_v, prompt="Go to frame 10",
    expected_tool_calls=[], match_mode="state", expected_state=screen(frame_is(10)),
)
for i, (prompt, frame) in enumerate([("Show end-systole", 11), ("Go to the ED frame", 28)], start=2):
    _m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}])
    _add(
        id=f"frame_{i:03d}", category="frame", description="ED/ES from the segmentation",
        model=_m, view=_v, prompt=prompt,
        expected_tool_calls=[], match_mode="state", expected_state=screen(frame_is(frame)),
    )
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}])
_add(
    id="frame_neg_001", category="frame", description="Frame beyond the cine",
    model=_m, view=_v, prompt="Go to frame 45",
    expected_tool_calls=[], match_mode="none", notes="The cines have 30 frames.",
)
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}])
_add(
    id="frame_neg_002", category="frame", description="ES asked, nothing segmented",
    model=_m, view=_v, prompt="Go to end systole",
    expected_tool_calls=[], match_mode="none",
    notes="Without a segmentation there is no ES frame: say so and ask whether to segment.",
)


# === NAVIGATION — layout, phase and slice levels in one request =================

_m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}], grid=(1, 1))
_add(
    id="nav_001", category="navigation", description="ES basal slices in a 2x2",
    model=_m, view=_v, prompt="Show the ES basal slices of the cine sax in a 2x2",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(grid_is(2, 2), frame_is(11), slices_shown("CINE_SAX", [11, 10, 9, 8])),
    notes="LV is in slices 3-11 at ES and slice 14 is basal, so the four basal LV slices are 11-8.",
)


def _base_mid_apex(s):
    shown = [sl for k, sl, _ in s["panels"] if k == "CINE_SAX"]
    if sorted(shown) != [3, 7, 11]:
        return f"CINE_SAX slices shown {shown}, expected 3 (apex), 7 (mid) and 11 (base)"


_m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}])
_add(
    id="nav_002", category="navigation", description="Base, mid and apex at ED in a row",
    model=_m, view=_v, prompt="Show a basal, a mid and an apical cine SAX slice at end-diastole in a 1x3 layout",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(grid_is(1, 3), frame_is(28), _base_mid_apex),
    notes="LV is in slices 2-12 at ED, but slice 12 is a crescent (< 50% surrounded) and slice 2 has no RV: 11 to 3.",
)
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}], grid=(1, 1))
_add(
    id="nav_neg_001", category="navigation", description="ES basal slices asked, nothing segmented",
    model=_m, view=_v, prompt="Show the ES basal slices of the cine sax in a 2x2",
    expected_tool_calls=[], match_mode="none",
    notes="ES and the LV range come from the segmentation: ask whether to run it first.",
)

_m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}])
_add(
    id="nav_003", category="navigation", description="SAX at four depths, then ED",
    model=_m, view=_v, prompt="Display the cine SAX at 4 depths and move to the ED frame",
    expected_tool_calls=[], match_mode="state", expected_state=screen(frame_is(28), _spread_cine_sax(4)),
    notes="Four different slices in order over the 2x2; slice numbers, not depths in mm.",
)
for i, (prompt, slice_no) in enumerate([("Display the base slice", 11), ("Now show the apex slice", 3)], start=4):
    _m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}], grid=(1, 1))
    _add(
        id=f"nav_{i:03d}", category="navigation", description="Base or apex by the state block's rule",
        model=_m, view=_v, prompt=prompt,
        expected_tool_calls=[], match_mode="state", expected_state=screen(panel_is(0, "CINE_SAX", slice_no)),
        notes="At frame 1 LV is in slices 2-12 of 14; slice 12 is a crescent and slice 2 has no RV: 11 to 3.",
    )
_m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}, {"idx": 1, "series_key": "LGE_SAX"}])
_add(
    id="nav_006", category="navigation", description="The basal slice of a stack with no LV range of its own",
    model=_m, view=_v, prompt="Show the basal LGE SAX slice in panel 1",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(panel_is(1, "LGE_SAX", nearest("LGE_SAX", phantom_depths("CINE_SAX")[10]))),
    notes="The CINE_SAX LV range, mapped to the nearest LGE_SAX slice.",
)


# === WINDOW — window/level (the phantom's default window is 300/600) ==========

def window_set(key: str, mode: str):
    return {"name": "set_window", "arguments": {"series_key": key, "mode": mode}}


for i, prompt in enumerate(["Make the myocardium dark on the LGE",
                            "Adjust the window/level on the LGE until the myocardium is dark"], start=1):
    _m, _v = _segmented([{"idx": 0, "series_key": "LGE_SAX"}], lge=True)
    _add(
        id=f"window_{i:03d}", category="window", description="LGE window with the myocardium nulled",
        model=_m, view=_v, prompt=prompt,
        expected_tool_calls=[window_set("LGE_SAX", "null_myocardium")], match_mode="exact",
    )
for i, (prompt, key, mode) in enumerate([("Make the LGE a bit darker", "LGE_SAX", "darker"),
                                         ("Much more contrast on the cine", "CINE_SAX", "more_contrast"),
                                         ("Reset the window of the LGE", "LGE_SAX", "reset")], start=3):
    _m, _v = _segmented([{"idx": 0, "series_key": "LGE_SAX"}, {"idx": 1, "series_key": "CINE_SAX"}], lge=True)
    _add(
        id=f"window_{i:03d}", category="window", description=f"Window step: {mode}",
        model=_m, view=_v, prompt=prompt,
        expected_tool_calls=[window_set(key, mode)], match_mode="exact",
    )
_m, _v = _segmented([{"idx": 0, "series_key": "LGE_SAX"}])
_add(
    id="window_neg_001", category="window", description="Nulled myocardium asked, LGE not segmented",
    model=_m, view=_v, prompt="Make the myocardium dark on the LGE",
    expected_tool_calls=[], match_mode="none",
    notes="null_myocardium needs the LGE segmentation: say so and ask whether to run it.",
)


def _two_lge_mid(s):
    """Two LGE_SAX panels at the mid slice (6 in the phantom): the left with contours and the series'
    window, the right without contours and with the myocardium nulled in its own window."""
    shown = [i for i, (k, sl, _) in enumerate(s["panels"]) if k == "LGE_SAX"]
    if len(shown) != 2 or any(s["panels"][i][1] != 6 for i in shown):
        return f"LGE_SAX panels {[s['panels'][i] for i in shown]}, expected two at slice 6"
    left, right = shown
    if not s["contours"][left] or s["contours"][right]:
        return f"contours {s['contours']}: expected on in panel {left} and off in panel {right}"
    if s["panel_windows"][left] is not None or s["panel_windows"][right] is None:
        return f"own windows {s['panel_windows']}: expected only panel {right}"


_m, _v = _segmented([{"idx": 0, "series_key": "LGE_SAX"}], grid=(1, 1), lge=True)
_add(
    id="window_006", category="window", description="One series twice: contours left, nulled window right",
    model=_m, view=_v,
    prompt="Show the LGE SAX mid slice twice side by side, the left one with segmentation and the right one "
           "with the myocardium nulled",
    expected_tool_calls=[], match_mode="state", expected_state=screen(_two_lge_mid),
    notes="run_segmentation shows the cached LGE contours in both; set_contours hides them on the right; "
          "set_window with panels gives only the right panel its own window.",
)


def _own_window_only(panel: int):
    def check(s):
        own = [i for i, w in enumerate(s["panel_windows"]) if w is not None]
        return None if own == [panel] and not s["windows"] else f"own windows in {own}, series windows {s['windows']}"
    return check


_m, _v = _segmented([{"idx": 0, "series_key": "LGE_SAX"}, {"idx": 1, "series_key": "LGE_SAX"}], lge=True)
_add(
    id="window_007", category="window", description="A window step in one panel only",
    model=_m, view=_v, prompt="Make only panel 1 a bit darker",
    expected_tool_calls=[], match_mode="state", expected_state=screen(_own_window_only(1)),
)


# === ADVANCED — layouts grown from a preset, split grids, contours per block ===

def _after_preset(preset: str, catalog: dict = None):
    """The screen right after a preset, on the phantom with every CINE and the LGE_SAX segmentation cached
    (measured, contours drawn in no panel yet)."""
    m, v = _segmented(catalog=catalog or FULL_STRESS, lge=True)
    m.analysis.update({"CINE_2CH": {"box": (60, 50, 200, 190)}, "CINE_3CH": {"box": (60, 50, 200, 190)}})
    FakeDicomController(m, v).apply_preset(preset)
    return m, v


def _keys(view) -> list:
    return [p["series_key"] for p in view.display_area_widget._panels]


def _slices(view) -> list:
    return [p["slice"] + 1 if p["series_key"] else None for p in view.display_area_widget._panels]


def _moved(before_keys: list, before_slices: list, old_cols: int, new_cols: int, shift_cols: int = 0):
    """The panels of an old grid are in the new grid at the same row, shifted right by shift_cols."""
    def check(s):
        for i, (key, sl) in enumerate(zip(before_keys, before_slices)):
            r, c = divmod(i, old_cols)
            j = r * new_cols + c + shift_cols
            if (s["panels"][j][0], s["panels"][j][1]) != (key, sl):
                return f"panel {j} shows {s['panels'][j][:2]}, expected {key} slice {sl} (was panel {i})"
    return check


def _column(panels: list[int], keys: list[str], ordered: bool = True):
    def check(s):
        shown = [s["panels"][i][0] if i < len(s["panels"]) else None for i in panels]
        ok = shown == keys if ordered else sorted(shown, key=str) == sorted(keys)
        return None if ok else f"panels {panels} show {shown}, expected {keys}"
    return check


_LAX_CINES = ["CINE_2CH", "CINE_3CH", "CINE_4CH"]
_m, _v = _after_preset("perfusion_lge")
_add(
    id="adv_001", category="advanced", description="Perfusion 3x3, the three LAX cines in a new right column",
    model=_m, view=_v, prompt="Display the three long-axis cines on the right",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(grid_is(3, 4), _column([3, 7, 11], _LAX_CINES),
                          _moved(_keys(_v), _slices(_v), 3, 4)),
    notes="set_grid(3, 4) keeps the 3x3 on the left; the new column is panels 3, 7 and 11.",
)
_m, _v = _after_preset("perfusion_lge")
_add(
    id="adv_002", category="advanced", description="Perfusion 3x3, the three LGE LAX views in a new left column",
    model=_m, view=_v, prompt="Put the three long-axis LGE views in a column on the left",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(grid_is(3, 4), _column([0, 4, 8], ["LGE_2CH", "LGE_3CH", "LGE_4CH"], ordered=False),
                          _moved(_keys(_v), _slices(_v), 3, 4, shift_cols=1)),
    notes="set_grid(3, 4, shift_cols=1): the 3x3 moves right, the new left column is panels 0, 4 and 8.",
)


def _twice(base_keys: list, rows: int, cols: int):
    """The layout twice side by side in a rows x 2cols grid: the left copy without contours, the right copy
    with contours wherever the series is segmented."""
    segmented = {"CINE_SAX", "CINE_2CH", "CINE_3CH", "CINE_4CH", "LGE_SAX"}

    def check(s):
        at = {cell[:2]: i for i, cell in enumerate(s["cells"])}  # a plain or a split grid look the same
        for i, key in enumerate(base_keys):
            r, c = divmod(i, cols)
            left, right = at.get((r, c)), at.get((r, c + cols))
            if left is None or right is None:
                return f"cells {s['cells']}: no {rows}x{2 * cols} layout"
            if s["panels"][left][0] != key or s["panels"][right][0] != key:
                return f"panels {left} and {right} show {s['panels'][left][0]}, {s['panels'][right][0]}, expected {key}"
            if key in segmented and (s["contours"][left] or not s["contours"][right]):
                return f"{key}: contours {s['contours'][left]} left, {s['contours'][right]} right, expected off, on"
    return check


_m, _v = _after_preset("functional", CINE_ONLY)
_add(
    id="adv_003", category="advanced", description="Functional twice: left without, right with contours",
    model=_m, view=_v,
    prompt="Show this layout twice side by side, the left one without segmentation and the right one with",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(grid_is(2, 4), _twice(_keys(_v), 2, 2)),
)
_m, _v = _after_preset("viability", CINE_LGE)
_add(
    id="adv_004", category="advanced", description="Viability in a 2x8: left 4 columns without, right 4 with",
    model=_m, view=_v,
    prompt="Make it a 2x8 with the same views on the right half, contours only on the right",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(grid_is(2, 8), _twice(_keys(_v), 2, 4)),
)


def _base_to_apex(count: int, first: int, last: int, start: int = 0):
    """count CINE_SAX panels from panel start on, in reading order from first (base) down to last (apex)."""
    def check(s):
        shown = [sl for k, sl, _ in s["panels"][start:start + count] if k == "CINE_SAX"]
        if len(shown) != count or shown != sorted(set(shown), reverse=True) or shown[0] != first or shown[-1] != last:
            return f"CINE_SAX slices {shown}, expected {count} from {first} (base) down to {last} (apex)"
    return check


_m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}], grid=(1, 1))
_add(
    id="adv_005", category="advanced", description="3x3 from base to apex at end-diastole",
    model=_m, view=_v, prompt="Show the cine SAX from base to apex in a 3x3 and go to the ED frame",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(grid_is(3, 3), frame_is(28), _base_to_apex(9, 11, 3)),
    notes="At ED the slices from base to apex are exactly 11 to 3.",
)
_m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}], grid=(1, 1))
_add(
    id="adv_006", category="advanced", description="Large 4CH left, the nine ES SAX slices in a 3x3 beside it",
    model=_m, view=_v,
    prompt="Large cine 4 chamber on the left and next to it the nine SAX slices from base to apex at end-systole",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(grid_is(3, 6), panel_is(0, "CINE_4CH"), frame_is(11), _base_to_apex(9, 11, 3, start=1)),
    notes="set_split_grid(1, 1, 3, 3); at ES the LV slices are exactly 11 to 3.",
)
_m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}], grid=(1, 1), lge=True)
_add(
    id="adv_007", category="advanced", description="Large LGE base left, a 2x2 of cine levels and the 4CH",
    model=_m, view=_v,
    prompt="One big LGE SAX at the base on the left, and next to it a 2x2 with the cine SAX at base, mid and "
           "apex and the cine 4 chamber",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(grid_is(2, 4), panel_is(0, "LGE_SAX", 9),
                          lambda s: None if sorted((k, sl) for k, sl, _ in s["panels"][1:5] if k == "CINE_SAX")
                          == [("CINE_SAX", 3), ("CINE_SAX", 7), ("CINE_SAX", 11)]
                          and "CINE_4CH" in [k for k, _, _ in s["panels"][1:5]]
                          else f"panels 1-4 show {s['panels'][1:5]}"),
)
_m, _v = _after_preset("perfusion_lge")
_lge_row = [(i, sl) for i, (k, sl) in enumerate(zip(_keys(_v), _slices(_v))) if k == "LGE_SAX"]
_add(
    id="adv_008", category="advanced", description="Perfusion 3x3, the LGE row swapped for cine SAX in place",
    model=_m, view=_v, prompt="Replace the LGE row with the cine short axis at the same positions",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(*[panel_is(i, "CINE_SAX", _v.display_area_widget.matching_slice("LGE_SAX", sl - 1,
                                                                                          "CINE_SAX") + 1)
                            for i, sl in _lge_row]),
    notes="display_series per panel without slice: a same-plane swap keeps each panel's position.",
)


def _lge_over_cine(s):
    if s["grid"] != (2, 3):
        return f"grid {s['grid']}, expected (2, 3)"
    top, bottom = s["panels"][:3], s["panels"][3:6]
    if any(k != "LGE_SAX" for k, _, _ in top) or sorted(sl for _, sl, _ in top) != [3, 6, 9]:
        return f"top row {top}, expected LGE_SAX at its base, mid and apex"
    if [(k, sl) for k, sl, _ in bottom] not in ([("CINE_SAX", 11), ("CINE_SAX", 7), ("CINE_SAX", 3)],
                                                [("CINE_SAX", 3), ("CINE_SAX", 7), ("CINE_SAX", 11)]):
        return f"bottom row {bottom}, expected CINE_SAX at the ES base, mid and apex (11, 7, 3)"
    if not all(s["contours"][3:6]):
        return f"contours {s['contours'][3:6]} in the bottom row, expected on"
    if "LGE_SAX" not in s["windows"] and not any(s["panel_windows"][:3]):
        return "no nulled-myocardium window on the LGE"


_m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}], grid=(1, 1), lge=True)
_add(
    id="adv_009", category="advanced", description="LGE levels with nulled myocardium over cine ES levels",
    model=_m, view=_v,
    prompt="Show the LGE SAX at base, mid and apex with the myocardium nulled, and below them the cine SAX at "
           "base, mid and apex at end-systole with segmentation",
    expected_tool_calls=[], match_mode="state", expected_state=_lge_over_cine,
    notes="The LGE levels nearest the CINE_SAX base, mid and apex are 9, 6 and 3 at frame 1 and at ES.",
)


# === PRESET_ADJUST — a preset as the starting point, then changed (issue #42) ===

def _rows_of(s, cols: int) -> list[list[tuple]]:
    return [s["panels"][r * cols:(r + 1) * cols] for r in range(s["grid"][0])]


def _stress_over_lge(s):
    """Stress perfusion on top, below it the LGE SAX slice nearest each stress slice; no rest row."""
    if s["grid"] != (2, 3):
        return f"grid {s['grid']}, expected (2, 3)"
    top, bottom = _rows_of(s, 3)
    if any(k != "PERF_STRESS_SAX" for k, _, _ in top) or any(k != "LGE_SAX" for k, _, _ in bottom):
        return f"rows {top} / {bottom}, expected stress perfusion over LGE_SAX"
    want = [nearest("LGE_SAX", phantom_depths("PERF_STRESS_SAX")[sl - 1]) for _, sl, _ in top]
    if [sl for _, sl, _ in bottom] != want:
        return f"LGE_SAX slices {[sl for _, sl, _ in bottom]}, expected {want} (nearest the stress slices)"


def _perfusion_over_cine(s):
    """The stress and rest rows, and below them the CINE_SAX slice nearest each perfusion column."""
    if s["grid"] != (3, 3):
        return f"grid {s['grid']}, expected (3, 3)"
    stress, rest, bottom = _rows_of(s, 3)
    if any(k != "PERF_STRESS_SAX" for k, _, _ in stress) or any(k != "PERF_REST_SAX" for k, _, _ in rest):
        return f"rows {stress} / {rest}, expected the stress and the rest row"
    want = [nearest("CINE_SAX", phantom_depths("PERF_STRESS_SAX")[sl - 1]) for _, sl, _ in stress]
    got = [(k, sl) for k, sl, _ in bottom]
    if any(k != "CINE_SAX" or abs(sl - w) > 1 for (k, sl), w in zip(got, want)):
        return f"bottom row {got}, expected CINE_SAX near slices {want}"


def _lge_levels(s):
    shown = sorted((k, sl if k == "LGE_SAX" else 0) for k, sl, _ in s["panels"] if k)
    want = sorted([("LGE_2CH", 0), ("LGE_3CH", 0), ("LGE_4CH", 0), ("LGE_SAX", 3), ("LGE_SAX", 6), ("LGE_SAX", 9)])
    return None if shown == want else f"panels {shown}, expected the three LGE LAX views and LGE_SAX 3, 6 and 9"


def _cine_levels(s):
    shown = sorted((k, sl if k == "CINE_SAX" else 0) for k, sl, _ in s["panels"] if k)
    want = sorted([("CINE_2CH", 0), ("CINE_3CH", 0), ("CINE_4CH", 0), ("CINE_SAX", 3), ("CINE_SAX", 7),
                   ("CINE_SAX", 11)])
    return None if shown == want else f"panels {shown}, expected the three LAX cines and CINE_SAX 3, 7 and 11"


def _viability_es_contours(s):
    keys = [k for k, _, _ in s["panels"]]
    if s["grid"] != (2, 4) or set(keys[:4]) != {"CINE_4CH", "CINE_3CH", "CINE_2CH", "CINE_SAX"} \
            or set(keys[4:]) != {"LGE_4CH", "LGE_3CH", "LGE_2CH", "LGE_SAX"}:
        return f"grid {s['grid']}, panels {keys}: expected the cines above their LGE views"
    if not all(s["contours"][:4]):
        return f"contours {s['contours'][:4]} on the cines, expected all on"


def _lge_dark(s):
    keys = {k for k, _, _ in s["panels"]}
    if s["grid"] != (2, 2) or keys != {"LGE_4CH", "LGE_3CH", "LGE_2CH", "LGE_SAX"}:
        return f"grid {s['grid']}, panels {keys}: expected the four LGE views in a 2x2"
    if "LGE_SAX" not in s["windows"] and not any(s["panel_windows"]):
        return "no nulled-myocardium window on the LGE"


_m, _v = _loaded(STRESS_REST, grid=(2, 2))
_add(
    id="preset_adjust_001", category="preset_adjust", description="Perfusion with LGE, only the stress run",
    model=_m, view=_v, prompt="Show the perfusion with LGE, but only the stress run",
    expected_tool_calls=[], match_mode="state", expected_state=_stress_over_lge,
    notes="perfusion_lge (stress, rest, LGE), then without the rest row: the LGE row matches the stress slices.",
)
_m, _v = _loaded(STRESS_REST, grid=(2, 2))
_add(
    id="preset_adjust_002", category="preset_adjust", description="Perfusion with the cine SAX at its depths below",
    model=_m, view=_v, prompt="Show the perfusion, and below it the cine SAX at the same depths",
    expected_tool_calls=[], match_mode="state", expected_state=_perfusion_over_cine,
    notes="perfusion_lge with the LGE row swapped for CINE_SAX in place, or perfusion and a matched cine row.",
)
_m, _v = _segmented(grid=(2, 2), lge=True)
_add(
    id="preset_adjust_003", category="preset_adjust", description="The LGE views, the SAX at base, mid and apex",
    model=_m, view=_v, prompt="Show the LGE views, with the SAX at base, mid and apex",
    expected_tool_calls=[], match_mode="state", expected_state=_lge_levels,
    notes="E.g. a 2x3: the three LGE long axes and LGE_SAX at its base, mid and apex (9, 6, 3).",
)
_m, _v = _segmented(grid=(2, 2), lge=True)
_m.analysis.update({"CINE_2CH": {"box": (60, 50, 200, 190)}, "CINE_3CH": {"box": (60, 50, 200, 190)}})
_add(
    id="preset_adjust_004", category="preset_adjust", description="Viability at end-systole, contours on the cines",
    model=_m, view=_v, prompt="Show viability at end-systole with the contours on the cines",
    expected_tool_calls=[], match_mode="state", expected_state=screen(frame_is(11), _viability_es_contours),
)
_m, _v = _segmented(grid=(2, 2), lge=True)
_add(
    id="preset_adjust_005", category="preset_adjust", description="The LGE views with the myocardium dark",
    model=_m, view=_v, prompt="Show the LGE with the myocardium dark",
    expected_tool_calls=[], match_mode="state", expected_state=_lge_dark,
)
_m, _v = _segmented(grid=(2, 2))
_add(
    id="preset_adjust_006", category="preset_adjust", description="Functional, the SAX at base, mid and apex",
    model=_m, view=_v, prompt="Show the functional views, but the SAX at base, mid and apex",
    expected_tool_calls=[], match_mode="state", expected_state=_cine_levels,
    notes="E.g. a 2x3: the three LAX cines and CINE_SAX at its base, mid and apex (11, 7, 3 at frame 1).",
)


# === CONTOURS — showing and hiding a segmentation per panel ===================

def _base_with_and_without(s):
    shown = [i for i, (k, sl, _) in enumerate(s["panels"]) if k == "CINE_SAX"]
    if len(shown) != 2 or any(s["panels"][i][1] != 11 for i in shown):
        return f"CINE_SAX panels {[s['panels'][i] for i in shown]}, expected two at the base, slice 11"
    if sorted(s["contours"][i] for i in shown) != [False, True]:
        return f"contours {[s['contours'][i] for i in shown]}, expected one with and one without"


_m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}], grid=(1, 1))
_add(
    id="contours_001", category="contours", description="The base twice, with and without segmentation",
    model=_m, view=_v,
    prompt="Show the cine SAX base slice twice, one with segmentation and one without",
    expected_tool_calls=[], match_mode="state", expected_state=screen(_base_with_and_without),
)
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX", "has_segmentation": True},
                      {"idx": 1, "series_key": "CINE_SAX", "has_segmentation": True}])
_add(
    id="contours_002", category="contours", description="Hide the contours in one panel",
    model=_m, view=_v, prompt="Hide the segmentation in panel 1",
    expected_tool_calls=[], match_mode="state",
    expected_state=lambda s: None if s["contours"][:2] == [True, False] else f"contours {s['contours']}",
)

# Asking before segmenting: a repeated request is not a yes.
_ASKED = [{"role": "user", "content": "Go to the ED frame"},
          {"role": "assistant", "content": "ED comes from the CINE_SAX segmentation, which has not been run yet. "
                                           "Shall I run it?"}]
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}])
_add(
    id="consent_001", category="segmentation", description="The request repeated after the question",
    model=_m, view=_v, prompt="go to ed frame", history=list(_ASKED),
    expected_tool_calls=[], match_mode="none",
    notes="Not a yes: ask again whether to segment.",
)
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}])
_add(
    id="consent_002", category="segmentation", description="Yes to the question",
    model=_m, view=_v, prompt="yes", history=list(_ASKED),
    expected_tool_calls=[segment("CINE_SAX")], match_mode="subset",
)
_ASKED_ZOOM = [{"role": "user", "content": "Zoom into the ROI"},
               {"role": "assistant", "content": "Zooming to the ROI needs a CINE_SAX segmentation, which has not "
                                                "been run yet. Shall I run it?"}]
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}])
_add(
    id="consent_003", category="segmentation", description="Yes to segmenting for a zoom",
    model=_m, view=_v, prompt="yes", history=list(_ASKED_ZOOM),
    expected_tool_calls=[segment("CINE_SAX")], match_mode="subset",
    notes="The app waits for the segmentation and asks again with the new state, where the zoom follows.",
)


# === PLAYBACK ==================================================================

def playing_is(playing: bool):
    return lambda s: None if s["playing"] == playing else f"playing {s['playing']}, expected {playing}"


_m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}])
_add(
    id="playback_001", category="playback", description="Play the cines",
    model=_m, view=_v, prompt="Play the cines",
    expected_tool_calls=[], match_mode="state", expected_state=screen(playing_is(True)),
)
for i, (prompt, frame) in enumerate([("Stop the movie", None), ("Stop at end-systole", 11)], start=2):
    _m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}])
    _v.playing = True
    _add(
        id=f"playback_{i:03d}", category="playback", description="Stop the cines" + (", then ES" if frame else ""),
        model=_m, view=_v, prompt=prompt,
        expected_tool_calls=[], match_mode="state",
        expected_state=screen(playing_is(False), *([frame_is(frame)] if frame else [])),
    )

_m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}])
_add(
    id="playback_004", category="playback", description="Slow motion",
    model=_m, view=_v, prompt="Play the cines in slow motion",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(playing_is(True), lambda s: None if s["speed"] == 0.5 else f"speed {s['speed']}"),
)
_m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}])
_v.playing = True
_add(
    id="playback_005", category="playback", description="Faster while playing",
    model=_m, view=_v, prompt="Faster",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(playing_is(True), lambda s: None if s["speed"] == 2 else f"speed {s['speed']}"),
    notes="From 1x the next speed up is 2x.",
)


# === ZOOM ======================================================================

_m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}, {"idx": 1, "series_key": "LGE_SAX"},
                     {"idx": 2, "series_key": "CINE_4CH"}])
_add(
    id="zoom_001", category="zoom", description="Every panel to the ROI",
    model=_m, view=_v, prompt="Zoom all panels to the ROI",
    expected_tool_calls=[], match_mode="state", expected_state=screen(zoomed("roi")),
    notes="LGE_SAX has no segmentation of its own and uses the CINE_SAX one of the same plane.",
)
_m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}, {"idx": 1, "series_key": "CINE_4CH"}])
_add(
    id="zoom_002", category="zoom", description="The heart, in clinical words",
    model=_m, view=_v, prompt="Zoom in on the heart",
    expected_tool_calls=[], match_mode="state", expected_state=screen(zoomed("roi")),
)
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}, {"idx": 1, "series_key": "CINE_4CH"}])
_add(
    id="zoom_003", category="zoom", description="A percentage on every panel",
    model=_m, view=_v, prompt="Zoom to 150%",
    expected_tool_calls=[], match_mode="state", expected_state=screen(zoomed(("percent", 150.0))),
)
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}, {"idx": 1, "series_key": "CINE_4CH"},
                      {"idx": 2, "series_key": "CINE_2CH"}])
_add(
    id="zoom_004", category="zoom", description="A percentage on one panel",
    model=_m, view=_v, prompt="Zoom panel 1 to 50%",
    expected_tool_calls=[], match_mode="state",
    expected_state=screen(zoomed(("percent", 50.0), [1]), only_zoomed([1])),
)
_m, _v = _loaded(CINE_ONLY)
_add(
    id="zoom_005", category="zoom", description="Extended LV function, images filling their panels",
    model=_m, view=_v,
    prompt="Show the extended LV function and make all images fit nicely, without empty bands at the sides",
    expected_tool_calls=[], match_mode="state", expected_state=screen(grid_is(3, 6), zoomed("fill")),
)
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}, {"idx": 1, "series_key": "CINE_4CH"}])
_add(
    id="zoom_006", category="zoom", description="Back to the whole image",
    model=_m, view=_v, prompt="Reset the zoom so I see the whole images again",
    expected_tool_calls=[], match_mode="state", expected_state=screen(zoomed("fit")),
)
_m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX"}, {"idx": 1, "series_key": "LGE_SAX"}])
_add(
    id="zoom_neg_001", category="zoom", description="ROI asked, nothing segmented",
    model=_m, view=_v, prompt="Zoom to the heart in every panel",
    expected_tool_calls=[], match_mode="none",
    notes="No segmentation to frame: ask whether to run it.",
)


# === SEGMENTATION ============================================================

# A loaded subject with a cached segmentation: measured (ED/ES, ROI) but not shown in any panel.
for i, prompt in enumerate(["Segment the cine SAX", "Show the segmentation"], start=1):
    _m, _v = _segmented([{"idx": 0, "series_key": "CINE_SAX"}])
    _add(
        id=f"segmentation_cached_{i:03d}", category="segmentation",
        description="A cached segmentation that no panel shows yet",
        model=_m, view=_v, prompt=prompt,
        expected_tool_calls=[], match_mode="state",
        expected_state=lambda s: None if s["contours"][0] else "panel 0 shows no contours",
        notes="Available is not shown: run_segmentation or set_contours shows the cached result at once.",
    )

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
#   From _no_subject(), build_tools() offers search_patients/load_subject only —
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
    expected_tool_calls=[], match_mode="state", expected_state=loaded("QLGE03_"),
    notes="Single-turn harness only sees the load call; apply_preset legitimately comes in a follow-up turn.",
)
_m, _v = _no_subject()
_add(
    id="compound_002", category="compound", description="Four-step request with nothing loaded yet",
    model=_m, view=_v,
    prompt="Load subject 1, display the viability preset, segment the cine 4ch, cine sax and lge sax",
    expected_tool_calls=[], match_mode="state", expected_state=loaded("QLGE01_"),
    notes="apply_preset/run_segmentation aren't offered here, so the load is the step; the app plans "
          "the rest again once the patient has loaded.",
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


# --- Not implemented yet: run_analysis tool -----------------------------------
# No run_analysis. Uncomment this case when it is added to TOOL_SPECS.
#
# _m, _v = _displaying([{"idx": 0, "series_key": "CINE_SAX", "has_segmentation": True}])
# _add(
#     id="analysis_001", category="analysis", description="Run analysis after segmentation",
#     model=_m, view=_v, prompt="Analyse panel 0",
#     expected_tool_calls=[{"name": "run_analysis", "arguments": {"panel_idx": 0}}], match_mode="subset",
# )
