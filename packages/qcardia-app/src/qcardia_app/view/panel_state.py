from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class AnnotationLayer:
    pass


@dataclass
class SegmentationLayer(AnnotationLayer):
    volume: np.ndarray  # (Z, H, W) uint8, class labels


@dataclass
class PanelState:
    series_key: str
    time_index: int = 0
    crosshair_world: np.ndarray | None = None
    window_level: tuple[float, float] | None = None
    layers: list[AnnotationLayer] = field(default_factory=list)
