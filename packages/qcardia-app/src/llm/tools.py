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

from utils.layouts import CINE_SPEEDS, MAX_GRID_COLS, MAX_GRID_ROWS, MAX_HALF
from utils.windowing import WINDOW_STEPS

_ALL_OR_SOME_PANELS = {"type": "array", "items": {"type": "integer"},
                       "description": "Panel numbers; leave out for all panels."}

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
                        "enum": ["functional", "functional_complex", "viability", "perfusion",
                                 "flow", "flow_aortic", "flow_pulmonary"],
                        "description": (
                            "functional: ventricular/LV/RV function, ejection "
                            "fraction, volumes, wall motion (CINE only). "
                            "functional_complex: extended functional, long-axis "
                            "views plus SAX slices from apex to base; for "
                            "per-slice or full SAX views. "
                            "viability: scar, late gadolinium enhancement, "
                            "infarct, fibrosis, transmurality (needs LGE). "
                            "perfusion: perfusion, ischaemia, CAD, stress testing "
                            "(needs PERF_STRESS_SAX, PERF_REST_SAX, or PERF_SAX). "
                            "flow: flow, phase contrast, cardiac output, regurgitation, "
                            "Qp:Qs; aortic flow on top and pulmonary flow below, or "
                            "whichever of the two was acquired (needs PC_AORTA or PC_MPA). "
                            "flow_aortic / flow_pulmonary: only when the user names the "
                            "vessel (aortic flow, aortic stenosis / pulmonary artery flow)."
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
                "exactly in 'Loaded series' or 'Still loading' in the state block. "
                "Without slice or match_panel the panel keeps the position of the stack it replaces "
                "if that stack has the same plane (e.g. CINE_SAX -> LGE_SAX), else the middle slice."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "panel_idx": {
                        "type": "integer",
                        "description": (
                            "Panel number as listed under 'Panels' in the state block. After a "
                            "set_grid call in the same reply, use the numbering of the new grid."
                        ),
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
                    "slice": {
                        "type": "integer",
                        "description": "Slice number to show, 1 to the slice count, as in 'slice 7/14' in the state block.",
                    },
                    "match_panel": {
                        "type": "integer",
                        "description": (
                            "Show the slice nearest to the position of this panel's slice, e.g. LGE_SAX "
                            "at the same depth as the CINE_SAX in panel 0 (slice numbers differ between "
                            "stacks)."
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
                    "frame": {
                        "type": "integer",
                        "description": (
                            "Frame number, 1 to the frame count, as in 'Time frame: 9 / 30'; for ED or ES, "
                            "the frame the state block gives under 'Segmentation'."
                        ),
                    },
                },
                "required": ["frame"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_grid",
            "description": (
                "Change the panel layout to a rows x cols grid; panels are numbered 0, 1, 2 ... in "
                "reading order (left to right, top to bottom). A panel keeps its series when its cell, "
                "moved by shift_rows/shift_cols, is in the new grid; new panels are empty. Example: to "
                "add a column on the left of a 2x3 layout, set_grid(rows=2, cols=4, shift_cols=1); then "
                "panels 0 and 4 are the empty left column. For two halves with their own grids (e.g. one "
                "large panel next to a 2x2), use set_split_grid."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rows": {"type": "integer", "minimum": 1, "maximum": MAX_GRID_ROWS},
                    "cols": {"type": "integer", "minimum": 1, "maximum": MAX_GRID_COLS},
                    "shift_rows": {
                        "type": "integer",
                        "description": "Move the current panels down by this many rows (default 0).",
                    },
                    "shift_cols": {
                        "type": "integer",
                        "description": "Move the current panels right by this many columns (default 0).",
                    },
                },
                "required": ["rows", "cols"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_split_grid",
            "description": (
                "Split the window into two equal halves, each with its own grid; a 1x1 half is one large "
                "panel. E.g. a large panel on the right with a 3x3 on the left = (3, 3, 1, 1); a large "
                "panel on the left with a 2x2 on the right = (1, 1, 2, 2). Panels are numbered through "
                "the left half first, then the right half, each in reading order; they keep their series "
                "by number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "left_rows": {"type": "integer", "minimum": 1, "maximum": MAX_HALF},
                    "left_cols": {"type": "integer", "minimum": 1, "maximum": MAX_HALF},
                    "right_rows": {"type": "integer", "minimum": 1, "maximum": MAX_HALF},
                    "right_cols": {"type": "integer", "minimum": 1, "maximum": MAX_HALF},
                },
                "required": ["left_rows", "left_cols", "right_rows", "right_cols"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zoom",
            "description": (
                "Zoom panels. mode 'roi': the heart, from a segmentation listed as available in the state "
                "block, also when no contours are drawn (a series without its own uses one of the same plane); 'fill': the "
                "image fills the panel, no empty bands at the sides; 'fit': the whole image; "
                "'percent': percent of the fitted size (e.g. 150 to zoom in)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["roi", "fill", "fit", "percent"]},
                    "percent": {"type": "number", "description": "Only for mode 'percent'."},
                    "panels": _ALL_OR_SOME_PANELS,
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_window",
            "description": (
                "Set the window/level (brightness and contrast) of a series, in every panel that shows it. "
                "null_myocardium (LGE_SAX only): healthy myocardium dark with its noise still visible, "
                "scar and blood pool bright but not saturated, from the LGE segmentation. darker, "
                "brighter, more_contrast, less_contrast: a step from the current window. reset: the "
                "series' default window."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "series_key": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["null_myocardium", *WINDOW_STEPS, "reset"],
                    },
                    "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "For the relative modes: 1 = a bit (default), 3 = a lot.",
                    },
                    "panels": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Only these panels (they must show series_key) get this window as their own. "
                            "Leave out for the whole series: then its panels with an own window follow it again."
                        ),
                    },
                },
                "required": ["series_key", "mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_contours",
            "description": (
                "Show or hide the segmentation contours in panels, e.g. one panel with and one without. "
                "Only a segmented series has contours (panel lines say 'seg shown' or 'seg hidden'); this "
                "does not segment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "shown": {"type": "boolean"},
                    "panels": _ALL_OR_SOME_PANELS,
                },
                "required": ["shown"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_playback",
            "description": "Play the cines as a movie or stop them, optionally at another speed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playing": {"type": "boolean", "description": "true = play, false = stop."},
                    "speed": {
                        "type": "number",
                        "enum": list(CINE_SPEEDS),
                        "description": (
                            "Heartbeats per second: 1 = normal, 0.5 = half speed / slow motion, 2 = twice as "
                            "fast. Leave out to keep the current speed (see 'cines playing at' in the state)."
                        ),
                    },
                },
                "required": ["playing"],
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
            spec["function"]["parameters"]["properties"]["panel_idx"]["minimum"] = 0
            tools.append(spec)
            continue

        if name == "set_window":
            loaded = sorted(k for k in ctx.series if k not in ctx.loading_series)
            if loaded:
                tools.append(_narrowed(spec, "series_key", loaded))
            continue

        preconditions = {
            "set_time_frame": ctx.num_time_frames > 0,
            "set_playback": ctx.num_time_frames > 0,
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
