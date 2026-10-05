"""Qt-free stand-ins for `.model`/`.view`/`.dicom_controller` — just enough
to drive ChatController's `_context_snapshot`/`_build_tools`/`_execute_tool`
without a real QWidget/QApplication."""
from __future__ import annotations

from pathlib import Path

DEFAULT_CINE_MODALITIES = ("CINE_SAX", "CINE_2CH", "CINE_3CH", "CINE_4CH")


class FakeDisplayAreaWidget:
    def __init__(
        self,
        grid_size: tuple[int, int] = (2, 2),
        panels: list[dict] | None = None,
        registered_series: set[str] | None = None,
    ):
        self._grid_rows, self._grid_cols = grid_size
        # panel_idx -> {"series_key", "axis", "has_segmentation"}
        self._panels: dict[int, dict] = {p["idx"]: p for p in (panels or [])}
        # is_series_registered() is about LOADED series, independent of what's
        # shown in a panel — pass registered_series explicitly for apply_preset tests.
        self._known_series: set[str] = (
            set(registered_series) if registered_series is not None
            else {p["series_key"] for p in (panels or []) if p.get("series_key")}
        )

    @property
    def n_panels(self) -> int:
        return self._grid_rows * self._grid_cols

    @property
    def grid_size(self) -> tuple[int, int]:
        return self._grid_rows, self._grid_cols

    def panel_assignments(self) -> list[dict]:
        out = []
        for i in range(self.n_panels):
            p = self._panels.get(i)
            out.append({
                "idx": i,
                "series_key": p["series_key"] if p else None,
                "axis": p.get("axis", "transverse") if p else "transverse",
                "has_segmentation": p.get("has_segmentation", False) if p else False,
            })
        return out

    def is_series_registered(self, series_key: str) -> bool:
        return series_key in self._known_series

    def set_grid_size(self, rows: int, cols: int):
        self._grid_rows, self._grid_cols = rows, cols

    def n_slices(self, series_key: str, axis: str = "transverse") -> int:
        return 12

    def set_panel_series(self, panel_idx: int, series_key: str, axis: str = "transverse", slice_idx: int | None = None):
        self._panels[panel_idx] = {"series_key": series_key, "axis": axis, "has_segmentation": False}

    def set_panel_empty(self, panel_idx: int):
        self._panels[panel_idx] = {"series_key": None, "axis": "transverse", "has_segmentation": False}


class FakeView:
    def __init__(self, display_area_widget: FakeDisplayAreaWidget):
        self.display_area_widget = display_area_widget


class FakeModel:
    def __init__(
        self,
        series: dict | None = None,
        folder_path: Path | None = None,
        num_time_frames: int = 0,
        cine_modalities: tuple[str, ...] = DEFAULT_CINE_MODALITIES,
    ):
        self.series = series or {}
        self.folder_path = folder_path
        self.num_time_frames = num_time_frames
        self.cine_modalities = cine_modalities


class FakeDicomController:
    """apply_preset delegates to the REAL DICOMController.apply_preset (same
    object.__new__ bypass as ChatController), so its result text can't drift
    from what the app actually does. Other _execute_tool methods aren't
    exercised by test_cases.py and stay unimplemented."""

    def __init__(self, model: FakeModel, view: FakeView, current_time_index: int = 0):
        self.current_time_index = current_time_index
        self._real = _build_real_dicom_controller(model, view)

    def apply_preset(self, name: str):
        return self._real.apply_preset(name)

    @property
    def loading_series(self) -> set[str]:
        return self._real.loading_series

    def panel_assignments(self) -> list[dict]:
        return self._real.panel_assignments()


def _build_real_dicom_controller(model: FakeModel, view: FakeView):
    from controllers.dicom_controller import DICOMController  # lazy: src/ isn't on sys.path yet at import time

    dc = object.__new__(DICOMController)
    dc.model = model
    dc.view = view
    dc._loading_keys = set()  # the patient has loaded
    dc._waiting = {}
    return dc
