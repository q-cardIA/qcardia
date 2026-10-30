from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

PERF_STRESS_KEY = "PERF_STRESS_SAX"
PERF_REST_KEY = "PERF_REST_SAX"
PERF_KEY = "PERF_SAX"  # a perfusion run that is neither stress nor rest
# A flow acquisition loads as its velocity (the key) and its magnitude (key + suffix).
FLOW_KEYS = {"aorta": "PC_AORTA", "pulmonary": "PC_MPA"}
FLOW_MAGNITUDE_SUFFIX = "_MAG"


@dataclass
class PanelAssignment:
    idx: int
    key: str
    axis: str = "transverse"
    slice_slot: int | None = None
    # Show the slice nearest, in world coordinates, to the same slot's slice of the first of these
    # series that is loaded.
    match_keys: tuple[str, ...] = ()


@dataclass
class Preset:
    label: str
    grid: tuple[int, int]
    panels: list[PanelAssignment] = field(default_factory=list)
    # Other phrases that name this preset, e.g. for search or typed commands.
    triggers: list[str] = field(default_factory=list)
    # Rows with none of their series loaded are left out, e.g. the rest row when only stress exists.
    drop_empty_rows: bool = False

    def for_loaded(self, is_loaded: Callable[[str], bool]) -> Preset:
        if not self.drop_empty_rows:
            return self
        cols = self.grid[1]
        kept = [row for row in range(self.grid[0])
                if any(is_loaded(p.key) for p in self.panels if p.idx // cols == row)]
        new_row = {row: i for i, row in enumerate(kept)}
        panels = [replace(p, idx=new_row[p.idx // cols] * cols + p.idx % cols)
                  for p in self.panels if p.idx // cols in new_row]
        return replace(self, grid=(len(kept), cols), panels=panels)


PRESETS: dict[str, Preset] = {
    "functional": Preset(
        label="the four cines in a 2x2",
        grid=(2, 2),
        panels=[
            PanelAssignment(0, "CINE_4CH"),
            PanelAssignment(1, "CINE_3CH"),
            PanelAssignment(2, "CINE_2CH"),
            PanelAssignment(3, "CINE_SAX"),
        ],
        triggers=[
            "LV function", "RV function", "biventricular function", "cardiac function",
            "ventricular function", "systolic function", "wall motion", "cine assessment",
            "ejection fraction", "EF", "LV volumes", "RV volumes", "left ventricle",
        ],
    ),
    "viability": Preset(
        label="the four cines above their LGE views",
        grid=(2, 4),
        panels=[
            PanelAssignment(0, "CINE_4CH"),
            PanelAssignment(1, "CINE_3CH"),
            PanelAssignment(2, "CINE_2CH"),
            PanelAssignment(3, "CINE_SAX"),
            PanelAssignment(4, "LGE_4CH"),
            PanelAssignment(5, "LGE_3CH"),
            PanelAssignment(6, "LGE_2CH"),
            PanelAssignment(7, "LGE_SAX"),
        ],
        triggers=[
            "viability", "viability assessment", "the cines next to the LGE", "cine and LGE side by side",
            "compare the cines with the LGE", "wall motion and scar together",
            "compare these with the LGE (with cines on screen)", "add the LGE to these cines",
        ],
    ),
    "lge": Preset(
        label="the four LGE views in a 2x2",
        grid=(2, 2),
        panels=[
            PanelAssignment(0, "LGE_4CH"),
            PanelAssignment(1, "LGE_3CH"),
            PanelAssignment(2, "LGE_2CH"),
            PanelAssignment(3, "LGE_SAX"),
        ],
        triggers=[
            "LGE", "late gadolinium enhancement", "late enhancement", "late gad", "scar", "scar assessment",
            "myocardial scar", "infarct", "myocardial infarction", "MI", "fibrosis", "transmurality",
        ],
    ),
    # Long-axis views in column 0, SAX slices basal to apical in the other panels.
    "functional_complex": Preset(
        label="the long-axis cines next to every cine SAX slice",
        grid=(3, 6),
        panels=[
            PanelAssignment(0, "CINE_4CH"),
            PanelAssignment(6, "CINE_3CH"),
            PanelAssignment(12, "CINE_2CH"),
        ] + [
            PanelAssignment(idx, "CINE_SAX", slice_slot=slot)
            for slot, idx in enumerate(i for i in range(18) if i % 6)
        ],
        triggers=[
            "all the cine SAX slices next to the LAX ones", "all SAX slices", "every SAX slice",
            "the whole SAX stack", "LV function per slice", "per-slice function",
        ],
    ),
    # A row per perfusion run (stress above rest), its slices from apex to base.
    "perfusion": Preset(
        label="the perfusion slices, a row per run",
        grid=(3, 3),
        panels=[
            PanelAssignment(row * 3 + slot, key, slice_slot=slot)
            for row, key in enumerate((PERF_STRESS_KEY, PERF_REST_KEY, PERF_KEY))
            for slot in range(3)
        ],
        triggers=[
            "perfusion", "stress perfusion", "rest perfusion", "ischaemia", "inducible ischaemia", "CAD",
            "coronary artery disease", "adenosine", "stress test", "perfusion defect", "myocardial perfusion",
        ],
        drop_empty_rows=True,
    ),
    # As perfusion, and below it the LGE slices nearest to the perfusion slices.
    "perfusion_lge": Preset(
        label="the perfusion slices above the LGE SAX at matching depths",
        grid=(4, 3),
        panels=[
            PanelAssignment(row * 3 + slot, key, slice_slot=slot,
                            match_keys=(PERF_STRESS_KEY, PERF_REST_KEY, PERF_KEY) if key == "LGE_SAX" else ())
            for row, key in enumerate((PERF_STRESS_KEY, PERF_REST_KEY, PERF_KEY, "LGE_SAX"))
            for slot in range(3)
        ],
        triggers=[
            "perfusion with LGE", "perfusion and LGE", "perfusion and scar", "perfusion with the scar images",
            "ischaemia and scar", "stress perfusion with LGE",
        ],
        drop_empty_rows=True,
    ),
    # Velocity on the left, magnitude on the right: aortic flow on top, pulmonary below, or the
    # one of the two that was acquired.
    "flow": Preset(
        label="the flow velocity and magnitude images",
        grid=(2, 2),
        panels=[
            PanelAssignment(row * 2 + col, FLOW_KEYS[vessel] + suffix)
            for row, vessel in enumerate(("aorta", "pulmonary"))
            for col, suffix in enumerate(("", FLOW_MAGNITUDE_SUFFIX))
        ],
        triggers=[
            "flow", "phase contrast", "flow quantification", "flow assessment", "flow measurement",
            "cardiac output", "regurgitant fraction", "regurgitation", "valve flow", "Qp:Qs",
            "phase contrast imaging", "flow CMR",
        ],
        drop_empty_rows=True,
    ),
    "flow_aortic": Preset(
        label="the aortic flow velocity and magnitude",
        grid=(1, 2),
        panels=[
            PanelAssignment(0, FLOW_KEYS["aorta"]),
            PanelAssignment(1, FLOW_KEYS["aorta"] + FLOW_MAGNITUDE_SUFFIX),
        ],
        triggers=["aortic flow", "aortic stenosis", "aortic regurgitation"],
    ),
    "flow_pulmonary": Preset(
        label="the pulmonary flow velocity and magnitude",
        grid=(1, 2),
        panels=[
            PanelAssignment(0, FLOW_KEYS["pulmonary"]),
            PanelAssignment(1, FLOW_KEYS["pulmonary"] + FLOW_MAGNITUDE_SUFFIX),
        ],
        triggers=["pulmonary flow", "pulmonary artery flow", "pulmonary regurgitation", "pulmonary hypertension"],
    ),
}

# The patient sidebar's buttons: the presets a user picks directly. The others are opened in code.
SIDEBAR_PRESETS = {"Function": "functional", "LGE": "lge", "Perfusion": "perfusion", "Flow": "flow"}


def preset_label(name: str) -> str:
    return PRESETS[name].label if name in PRESETS else name.title()


_LGE_KEYS = ("LGE_SAX", "LGE_2CH", "LGE_3CH", "LGE_4CH")
_PERFUSION_KEYS = (PERF_STRESS_KEY, PERF_REST_KEY, PERF_KEY)


def available_presets(loaded_series: set[str]) -> list[str]:
    """A preset needs only one of its required series loaded, not all —
    gated on each preset's own distinguishing series, not shared CINE_* keys."""
    names = []
    if any(pa.key in loaded_series for pa in PRESETS["functional"].panels):
        names += ["functional", "functional_complex"]
    if any(k in loaded_series for k in _LGE_KEYS):
        names += ["viability", "lge"]
    # LGE_SAX alone doesn't count — 'perfusion' needs real perfusion data,
    # matching the tool's own stated precondition.
    if any(k in loaded_series for k in _PERFUSION_KEYS):
        names += ["perfusion", *(["perfusion_lge"] if "LGE_SAX" in loaded_series else [])]
    if any(k in loaded_series for k in FLOW_KEYS.values()):
        names.append("flow")
    if FLOW_KEYS["aorta"] in loaded_series:
        names.append("flow_aortic")
    if FLOW_KEYS["pulmonary"] in loaded_series:
        names.append("flow_pulmonary")
    return names

