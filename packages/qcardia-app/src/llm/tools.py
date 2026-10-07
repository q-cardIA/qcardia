"""Tool schemas offered to the chat model, and context-based filtering.

Pure functions, no Qt: `build_tools()` takes a `ToolContext` snapshot of app
state and returns exactly the tools valid for this turn — some are dropped
entirely (their precondition isn't met), others get an enum narrowed to what
actually exists right now (loaded series, available
presets, current grid size), so the model can't request something that
doesn't exist rather than relying on it reading prose correctly.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from utils.hanging_protocols import available_presets

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "list_subjects",
            "description": "List the subject folder names in the data folder.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_subject",
            "description": (
                "Load a patient study by subject folder name. This replaces the "
                "current patient and clears all panels. Use a name from list_subjects."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_id": {
                        "type": "string",
                        "description": "Exact subject folder name from list_subjects.",
                    }
                },
                "required": ["subject_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_segmentation",
            "description": (
                "Run AI segmentation on a loaded stack. modality is the exact "
                "series_key of the stack to segment, e.g. CINE_SAX or LGE_SAX — "
                "same series_key values used by display_series."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "modality": {
                        "type": "string",
                        "enum": ["CINE_SAX", "LGE_SAX"],
                        "description": "Exact series_key of the stack to segment: CINE_SAX or LGE_SAX.",
                    },
                },
                "required": ["modality"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_preset",
            "description": (
                "Apply a clinical hanging protocol: sets the grid AND fills the "
                "panels, replacing the current layout. Prefer this over several "
                "display_series calls. To also show a custom panel, call "
                "apply_preset first, then display_series. Only offered for a "
                "preset when at least one of its required series is loaded; if "
                "a preset you want isn't in the enum below, say the data is "
                "unavailable rather than calling a different tool. Fills every "
                "matching panel it can — its result text reports how many were "
                "actually filled, which may be fewer than the full grid."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "preset": {
                        "type": "string",
                        "enum": ["functional", "functional_complex", "viability", "perfusion"],
                        "description": (
                            "functional: ventricular/LV/RV function, ejection "
                            "fraction, volumes, wall motion (CINE only). "
                            "functional_complex: extended functional, long-axis "
                            "views plus SAX slices from base to apex; for "
                            "per-slice or full SAX views. "
                            "viability: scar, late gadolinium enhancement, "
                            "infarct, fibrosis, transmurality (needs LGE). "
                            "perfusion: perfusion, ischaemia, CAD, stress testing "
                            "(needs PERF_STRESS_SAX, PERF_REST_SAX, or PERF_SAX)."
                        ),
                    }
                },
                "required": ["preset"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "display_series",
            "description": (
                "Show one loaded series in one panel. series_key must appear "
                "exactly in 'Loaded series' or 'Still loading' in the state block."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "panel_idx": {
                        "type": "integer",
                        "description": "0-indexed panel in the current grid: row * cols + col.",
                    },
                    "series_key": {
                        "type": "string",
                        "description": (
                            "Exact key from 'Loaded series' or 'Still loading', e.g. CINE_SAX or "
                            "LGE_4CH. Must match a listed key character for "
                            "character — never invent one."
                        ),
                    },
                    "axis": {
                        "type": "string",
                        "enum": ["transverse", "coronal", "sagittal"],
                        "description": (
                            "Reslicing direction, almost always 'transverse'. This "
                            "is NOT the cardiac plane — the cardiac plane is part "
                            "of series_key. Only use coronal/sagittal if the user "
                            "explicitly asks to reslice."
                        ),
                    },
                },
                "required": ["panel_idx", "series_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_time_frame",
            "description": "Jump all cine panels to a specific time frame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_index": {
                        "type": "integer",
                        "description": (
                            "0-based frame index, range 0 to (total frames - 1). "
                            "The state block shows 'Time frame: current / total' "
                            "using 1-based numbering, so subtract 1."
                        ),
                    }
                },
                "required": ["time_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_segmentation",
            "description": "Save the current segmentation to disk. Only call after a segmentation has been run.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass
class ToolContext:
    series: dict
    cine_modalities: tuple[str, ...]
    n_panels: int
    num_time_frames: int
    has_segmentation: bool
    loading_series: frozenset[str] = frozenset()  # usable already: the tools wait for them

    @classmethod
    def from_app(cls, model, view, loading_series=frozenset()) -> "ToolContext":
        return cls(
            series=dict(model.series),
            cine_modalities=model.cine_modalities,
            n_panels=view.display_area_widget.n_panels,
            num_time_frames=model.num_time_frames,
            has_segmentation=hasattr(model.series.get("CINE_SAX"), "_segmentation_prediction"),
            loading_series=frozenset(loading_series),
        )


def build_tools(ctx: ToolContext) -> list[dict]:
    tools = []
    usable = [*ctx.series, *sorted(ctx.loading_series - set(ctx.series))]
    for spec in TOOL_SPECS:
        name = spec["function"]["name"]

        if name == "run_segmentation":
            modalities = [k for k in (*ctx.cine_modalities, "LGE_SAX") if k in usable]
            if not modalities:
                continue
            tools.append(_narrowed(spec, "modality", modalities))
            continue

        if name == "apply_preset":
            names = available_presets(set(usable))
            if not names:
                continue
            tools.append(_narrowed(spec, "preset", names))
            continue

        if name == "display_series":
            if not usable:
                continue
            spec = _narrowed(spec, "series_key", usable)
            props = spec["function"]["parameters"]["properties"]
            props["panel_idx"]["minimum"] = 0
            props["panel_idx"]["maximum"] = ctx.n_panels - 1
            tools.append(spec)
            continue

        preconditions = {
            "set_time_frame": ctx.num_time_frames > 0,
            "save_segmentation": ctx.has_segmentation,
        }
        if preconditions.get(name, True):
            tools.append(spec)
    return tools


def _narrowed(spec: dict, param: str, values: list) -> dict:
    """Deep-copies just the properties dict (not the whole spec, including
    its static description text) — cheap enough to call on every turn.
    display_series narrows a second property (panel_idx) after this call,
    so every property gets its own fresh copy here, not just `param`'s."""
    properties = copy.deepcopy(spec["function"]["parameters"]["properties"])
    properties[param]["enum"] = values
    return {
        **spec,
        "function": {
            **spec["function"],
            "parameters": {**spec["function"]["parameters"], "properties": properties},
        },
    }
