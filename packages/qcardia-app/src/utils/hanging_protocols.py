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
    ),
}

PRESET_BUTTON_ORDER = ["functional", "viability", "perfusion"]


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
                PanelAssignment(0, "PERF_STRESS_SAX"),
                PanelAssignment(1, "PERF_STRESS_SAX"),
                PanelAssignment(2, "PERF_STRESS_SAX"),
                PanelAssignment(3, "PERF_REST_SAX"),
                PanelAssignment(4, "PERF_REST_SAX"),
                PanelAssignment(5, "PERF_REST_SAX"),
                PanelAssignment(6, "LGE_SAX"),
                PanelAssignment(7, "LGE_SAX"),
                PanelAssignment(8, "LGE_SAX"),
            ],
        )

    if has_rest:
        perf_key = "PERF_REST_SAX"
    elif has_stress:
        perf_key = "PERF_STRESS_SAX"
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
