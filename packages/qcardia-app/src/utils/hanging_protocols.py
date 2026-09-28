from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PanelAssignment:
    idx: int
    key: str
    axis: str = "transverse"
    slice_slot: int | None = None


@dataclass
class Preset:
    label: str
    grid: tuple[int, int]
    panels: list[PanelAssignment] = field(default_factory=list)
    # Trigger phrases the chat assistant recognizes (see llm.prompt._preset_rules).
    triggers: list[str] = field(default_factory=list)


PRESETS: dict[str, Preset] = {
    "functional": Preset(
        label="Functional",
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
        label="Viability",
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
            "viability", "scar", "LGE", "late gadolinium enhancement", "late enhancement",
            "infarct", "myocardial infarction", "transmurality", "scar assessment",
            "myocardial scar", "fibrosis", "MI",
        ],
    ),
    # Long-axis views in column 0, SAX slices basal to apical in the other panels.
    "functional_complex": Preset(
        label="Functional (Extended)",
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
            "extended functional", "extended function", "full functional", "per-slice function",
            "per slice", "all SAX slices", "every SAX slice", "SAX stack", "basal to apical",
            "detailed function", "complete functional assessment",
        ],
    ),
}

# Not a static PRESETS entry — its grid depends on stress/rest presence, see build_perfusion_preset.
PERFUSION_TRIGGERS = [
    "perfusion", "stress perfusion", "ischaemia", "inducible ischaemia", "CAD",
    "coronary artery disease", "adenosine", "stress test", "perfusion defect",
    "myocardial perfusion", "perfusion and scar",
]

PRESET_BUTTON_ORDER = ["functional", "functional_complex", "viability", "perfusion"]


_LGE_KEYS = ("LGE_SAX", "LGE_2CH", "LGE_3CH", "LGE_4CH")
PERF_STRESS_KEY = "PERF_STRESS_SAX"
PERF_REST_KEY = "PERF_REST_SAX"
_PERFUSION_KEYS = (PERF_STRESS_KEY, PERF_REST_KEY, "PERF_SAX")


def available_presets(loaded_series: set[str]) -> list[str]:
    """A preset needs only one of its required series loaded, not all —
    gated on each preset's own distinguishing series, not shared CINE_* keys."""
    names = []
    if any(pa.key in loaded_series for pa in PRESETS["functional"].panels):
        names += ["functional", "functional_complex"]
    if any(k in loaded_series for k in _LGE_KEYS):
        names.append("viability")
    # LGE_SAX alone doesn't count — 'perfusion' needs real perfusion data,
    # matching the tool's own stated precondition.
    if any(k in loaded_series for k in _PERFUSION_KEYS):
        names.append("perfusion")
    return names


def build_perfusion_preset(
    has_stress: bool, has_rest: bool, fallback_key: str = "PERF_SAX"
) -> Preset:
    """Perfusion grid depends on which acquisitions are present, so it's built
    on demand rather than looked up in PRESETS."""
    if has_stress and has_rest:
        return Preset(
            label="Perfusion",
            grid=(3, 3),
            panels=[
                PanelAssignment(0, PERF_STRESS_KEY),
                PanelAssignment(1, PERF_STRESS_KEY),
                PanelAssignment(2, PERF_STRESS_KEY),
                PanelAssignment(3, PERF_REST_KEY),
                PanelAssignment(4, PERF_REST_KEY),
                PanelAssignment(5, PERF_REST_KEY),
                PanelAssignment(6, "LGE_SAX"),
                PanelAssignment(7, "LGE_SAX"),
                PanelAssignment(8, "LGE_SAX"),
            ],
        )

    if has_rest:
        perf_key = PERF_REST_KEY
    elif has_stress:
        perf_key = PERF_STRESS_KEY
    else:
        perf_key = fallback_key

    return Preset(
        label="Perfusion",
        grid=(2, 3),
        panels=[
            PanelAssignment(0, perf_key),
            PanelAssignment(1, perf_key),
            PanelAssignment(2, perf_key),
            PanelAssignment(3, "LGE_SAX"),
            PanelAssignment(4, "LGE_SAX"),
            PanelAssignment(5, "LGE_SAX"),
        ],
    )
