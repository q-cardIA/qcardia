"""Qt-free stand-ins for `.model`/`.view`/`.dicom_controller` — just enough
to drive ChatController's `_context_snapshot`/`_build_tools`/`_execute_tool`
without a real QWidget/QApplication.

The fake viewer keeps the state the tools change (grid and merged cells, the
series, slice and zoom of each panel, the cine frame), so a case can check
the screen a reply leads to. Its stacks are a phantom: CINE_SAX and LGE_SAX
cover the same heart with different slice counts, so a slice number does not
carry over from one to the other: the app matches slices by position."""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from utils import app_config
from utils.layouts import layout_cells, split_layout
from view.volume_geometry import VolumeGeometry

DEFAULT_CINE_MODALITIES = ("CINE_SAX", "CINE_2CH", "CINE_3CH", "CINE_4CH")

# Short-axis stacks: (slices, depth of slice 1 in mm, spacing in mm). Slice 1 is apical.
PHANTOM_STACKS = {"CINE_SAX": (14, -137.3, 8.0), "LGE_SAX": (12, -142.0, 10.0),
                  "PERF_SAX": (3, -115.0, 25.0), "PERF_STRESS_SAX": (3, -115.0, 25.0),
                  "PERF_REST_SAX": (3, -115.0, 25.0)}
CINE_FRAMES = 30
PERFUSION_FRAMES = 60


def phantom_depths(series_key: str) -> list[float]:
    n, first, step = PHANTOM_STACKS[series_key]
    return [round(first + i * step, 1) for i in range(n)]


def phantom_analysis() -> dict:
    """CINE_SAX measurements as the app derives them from a segmentation (0-based indices):
    ED = frame 28, ES = frame 11; LV in slices 2-12 at ED and 3-11 at ES, RV too except in slice 2; slice 1 apical."""
    volumes = [round(70 + 50 * abs(t - 10) / 17, 1) if t <= 27 else round(118 - 2 * (t - 27), 1)
               for t in range(CINE_FRAMES)]
    lv = [list(range(1, 12)) if abs(t - 10) > 5 else list(range(2, 11)) for t in range(CINE_FRAMES)]
    # Myocardium surrounds the blood pool everywhere except in slice 12, a crescent where it has LV.
    coverage = [[(0.3 if z == 11 else 1.0) if z in lv[t] else 0.0 for z in range(PHANTOM_STACKS["CINE_SAX"][0])]
                for t in range(CINE_FRAMES)]
    box = (100, 80, 180, 170)
    return {
        "CINE_SAX": {"box": box, "volumes": volumes, "ed": 27, "es": 10,
                     "apex_to_base": list(range(PHANTOM_STACKS["CINE_SAX"][0])), "lv_slices": lv,
                     "lv_myo_rv_slices": [s[1:] if s[0] == 1 else s for s in lv], "lv_coverage": coverage},
        "CINE_4CH": {"box": (60, 50, 200, 190)},
    }


@functools.cache
def phantom_lge() -> tuple[np.ndarray, np.ndarray]:
    """An LGE_SAX volume and its segmentation (1 = healthy myocardium, 2 = scar), 12 slices as the stack."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:96, :96]
    ring = ((yy - 48) ** 2 + (xx - 48) ** 2 <= 20 ** 2) & ((yy - 48) ** 2 + (xx - 48) ** 2 > 12 ** 2)
    seg = np.zeros((PHANTOM_STACKS["LGE_SAX"][0], 96, 96), np.uint8)
    seg[:, ring] = 1
    seg[4:7, ring & (xx > 58)] = 2
    volume = np.full(seg.shape, 60.0)
    volume[seg == 1] = rng.normal(20, 5, (seg == 1).sum())
    volume[seg == 2] = 250
    return volume, seg


DEFAULT_WINDOW = (300.0, 600.0)

# The patient database the cases see (the app reads patients.csv in the data folder): made-up names, two
# Emilys for a request that must ask which one. The folders are the cases' default available_subjects.
_NAMES = ["Lucy Hughes", "Isabel Hall", "Michael King", "Alice White", "William Wilson", "Florence Johnson",
          "Alice Jackson", "Adam James", "George Bell", "Hannah Wright", "Emily Clarke", "Oscar Turner",
          "Grace Walker", "Henry Moore", "Sophie Bennett", "Emily Carter", "Jack Morgan", "Laura Evans",
          "Peter Collins", "Rachel Hughes", "Thomas Fisher"]
TEST_PATIENTS = {f"QLGE{n:02d}_": app_config.Patient(f"QLGE{n:02d}_", name, f"{n:02d}HD{n % 10}H7Y9K")
                 for n, name in enumerate(_NAMES, start=1)}


class FakeDisplayAreaWidget:
    def __init__(
        self,
        grid_size: tuple[int, int] = (2, 2),
        panels: list[dict] | None = None,
        registered_series: set[str] | None = None,
    ):
        self._grid_rows, self._grid_cols = grid_size
        self._cells = layout_cells(*grid_size)
        self.split = None
        self._panels = [self._empty() for _ in self._cells]
        self.segmented: set[str] = set()  # series with a segmentation in the viewer
        for p in panels or []:
            self._panels[p["idx"]] = self._empty()
            self._show(p["idx"], p["series_key"], p.get("axis", "transverse"), p.get("slice"))
            self._panels[p["idx"]]["frame"] = p.get("frame")  # an own frame, 0-based
            if p.get("has_segmentation"):
                self.segmented.add(p["series_key"])
                self._panels[p["idx"]]["contours"] = True
        # is_series_registered() is about LOADED series, independent of what's
        # shown in a panel — pass registered_series explicitly for apply_preset tests.
        self._known_series: set[str] = (
            set(registered_series) if registered_series is not None
            else {p["series_key"] for p in (panels or []) if p.get("series_key")}
        )
        self.current_time_index = 0
        self.perfusion_frames: dict[str, int] = {}  # run -> frame (0-based)
        self.driven_time_axis = "CINE"
        self.windows: dict[str, tuple[float, float]] = {}

    @staticmethod
    def _empty() -> dict:
        return {"series_key": None, "axis": "transverse", "slice": 0, "zoom": None, "contours": False, "window": None,
                "frame": None}

    def _show(self, idx: int, key: str, axis: str, slice_idx: int | None):
        n = self.n_slices(key, axis)
        self._panels[idx].update(series_key=key, axis=axis, zoom=None, frame=None,
                                 slice=n // 2 if slice_idx is None else slice_idx)

    @property
    def n_panels(self) -> int:
        return len(self._cells)

    @property
    def grid_size(self) -> tuple[int, int]:
        return self._grid_rows, self._grid_cols

    def panel_assignments(self) -> list[dict]:
        out = []
        for i, (cell, p) in enumerate(zip(self._cells, self._panels)):
            panel = {"idx": i, "series_key": p["series_key"], "axis": p["axis"],
                     "has_segmentation": self._shows_contours(p), "segmented": p["series_key"] in self.segmented,
                     "own_window": p["window"] is not None}
            n = self.n_slices(p["series_key"], p["axis"]) if p["series_key"] else 0
            if n > 1:
                panel.update(slice=p["slice"], n_slices=n)
            if p["frame"] is not None:
                panel["own_frame"] = (p["frame"], self.panel_frames(i))
            out.append(panel)
        return out

    def state(self) -> dict:
        """The screen, 1-based as the user sees it: grid, merged cells, cine frame, and per panel
        (series_key, slice, zoom)."""
        return {
            "grid": self.grid_size,
            "cells": list(self._cells),
            "split": self.split,
            "frame": self.current_time_index + 1,
            "perfusion_frames": {k: t + 1 for k, t in self.perfusion_frames.items()},
            "driven": self.driven_time_axis,
            "windows": dict(self.windows),
            "panel_windows": [p["window"] for p in self._panels],
            "contours": [self._shows_contours(p) for p in self._panels],
            "panel_frames": [self._frame_of(i) for i in range(self.n_panels)],
            "panels": [(p["series_key"], p["slice"] + 1 if p["series_key"] else None, p["zoom"])
                       for p in self._panels],
        }

    def is_series_registered(self, series_key: str) -> bool:
        return series_key in self._known_series

    def is_panel_empty(self, panel_idx: int) -> bool:
        return self._panels[panel_idx]["series_key"] is None

    def set_grid_size(self, rows: int, cols: int, shift=(0, 0)):
        self._set_layout(rows, cols, layout_cells(rows, cols), shift)

    def set_split_layout(self, split):
        self._set_layout(*split_layout(split), split=split)

    def _set_layout(self, rows, cols, cells, shift=(0, 0), split=None):
        if (split is not None or self.split is not None) and not any(shift):  # by panel number
            moved = {cell[:2]: p for cell, p in zip(cells, self._panels)}
        else:
            moved = {(r + shift[0], c + shift[1]): p for (r, c, _, _), p in zip(self._cells, self._panels)}
        self._grid_rows, self._grid_cols, self._cells, self.split = rows, cols, cells, split
        self._panels = [moved.get(cell[:2], self._empty()) for cell in self._cells]

    def n_slices(self, series_key: str, axis: str = "transverse") -> int:
        return PHANTOM_STACKS.get(series_key, (1,))[0] if axis == "transverse" else 128

    def n_frames(self, time_axis: str) -> int:
        if time_axis.startswith("PERF_"):
            return PERFUSION_FRAMES if time_axis in self._known_series else 0
        return CINE_FRAMES if any(k.startswith("CINE_") for k in self._known_series) else 0

    @staticmethod
    def _axis(key: str) -> str:
        return key if key.startswith("PERF_") else "CINE"

    def panel_frames(self, panel_idx: int) -> int:
        key = self._panels[panel_idx]["series_key"]
        return self.n_frames(self._axis(key)) if key and key.startswith(("CINE_", "PERF_")) else 0

    def _frame_of(self, panel_idx: int) -> int | None:
        """The frame a panel shows, 1-based; None without frames."""
        p = self._panels[panel_idx]
        if not self.panel_frames(panel_idx):
            return None
        return (p["frame"] if p["frame"] is not None else self.time_index(self._axis(p["series_key"]))) + 1

    def hold_frame(self, panels: list[int], t: int):
        for i in panels:
            if self.panel_frames(i):
                self._panels[i]["frame"] = t

    def release_frames(self, axis: str):
        for p in self._panels:
            if p["series_key"] and self._axis(p["series_key"]) == axis:
                p["frame"] = None

    def time_index(self, axis: str) -> int:
        return self.perfusion_frames.get(axis, 0) if axis.startswith("PERF_") else self.current_time_index

    def segmentation(self, series_key: str):
        return None  # no panel shows contours: the controller reads the cache (FakeModel.segment)

    def volume(self, series_key: str):
        return phantom_lge()[0] if series_key == "LGE_SAX" else None

    def window(self, series_key: str, default: bool = False, panel: int | None = None):
        if panel is not None and self._panels[panel]["window"] is not None:
            return self._panels[panel]["window"]
        return DEFAULT_WINDOW if default else self.windows.get(series_key, DEFAULT_WINDOW)

    def set_window(self, series_key: str, window, panels=None):
        window = tuple(round(v, 1) for v in window)
        if panels is not None:
            for i in panels:
                self._panels[i]["window"] = window
            return
        self.windows[series_key] = window
        for p in self._panels:
            if p["series_key"] == series_key:
                p["window"] = None

    def _shows_contours(self, panel: dict) -> bool:
        return panel["contours"] and panel["series_key"] in self.segmented

    def set_panel_contours(self, panel_idx: int, shown: bool):
        self._panels[panel_idx]["contours"] = shown

    def drive_time(self, axis: str = "CINE"):
        self.driven_time_axis = axis

    def has_segmentation(self, series_key: str) -> bool:
        return series_key in self.segmented

    def show_segmentation(self, series_key: str) -> list[int]:
        panels = [i for i, p in enumerate(self._panels) if p["series_key"] == series_key]
        for i in panels:
            self._panels[i]["contours"] = True
        return panels

    def matching_slice(self, source_key: str, z: int, series_key: str) -> int | None:
        if source_key not in PHANTOM_STACKS or series_key not in PHANTOM_STACKS:
            return None
        depth, depths = phantom_depths(source_key)[z], phantom_depths(series_key)
        return min(range(len(depths)), key=lambda i: abs(depths[i] - depth))

    def set_panel_series(self, panel_idx: int, series_key: str, axis: str = "transverse", slice_idx: int | None = None):
        old = self._panels[panel_idx]
        if slice_idx is None and old["series_key"] and old["axis"] == axis == "transverse":
            slice_idx = (old["slice"] if old["series_key"] == series_key
                         else self.matching_slice(old["series_key"], old["slice"], series_key))
        self._show(panel_idx, series_key, axis, slice_idx)
        self._panels[panel_idx].update(window=None, contours=any(
            self._shows_contours(p) for i, p in enumerate(self._panels) if i != panel_idx and p["series_key"] == series_key))

    def set_panel_empty(self, panel_idx: int):
        self._panels[panel_idx] = self._empty()

    def set_panel_waiting(self, panel_idx: int, series_key: str):
        self.set_panel_empty(panel_idx)

    def set_time_index(self, t: int, axis: str | None = None):
        if axis and axis.startswith("PERF_"):
            self.perfusion_frames[axis] = t
        else:
            self.current_time_index = t

    def zoom_panels(self, panels, mode, percent=100.0, analysis=None) -> list[int]:
        boxes = {k: a["box"] for k, a in (analysis or {}).items() if a.get("box")}
        done = []
        for i in panels:
            key = self._panels[i]["series_key"]
            if key is None:
                continue
            if mode == "roi" and not any(k == key or k.rsplit("_", 1)[-1] == key.rsplit("_", 1)[-1]
                                         for k in boxes or {}):
                continue
            self._panels[i]["zoom"] = ("percent", percent) if mode == "percent" else mode
            done.append(i)
        return done


class FakeView:
    def __init__(self, display_area_widget: FakeDisplayAreaWidget, playing: bool = False):
        self.display_area_widget = display_area_widget
        self.playing = playing
        self.cine_speed = self.perfusion_speed = 1.0

    def is_playing(self) -> bool:
        return self.playing

    def log_screen(self):
        pass

    def set_cine_speed(self, speed: float, perfusion: bool = False):
        setattr(self, "perfusion_speed" if perfusion else "cine_speed", speed)

    def toggle_cine(self):
        self.playing = not self.playing


class FakeModel:
    def __init__(
        self,
        series: dict | None = None,
        folder_path: Path | None = None,
        num_time_frames: int = 0,
        cine_modalities: tuple[str, ...] = DEFAULT_CINE_MODALITIES,
        analysis: dict | None = None,
    ):
        self.series = series or {}
        self.folder_path = self.subject_folder = folder_path
        self.num_time_frames = num_time_frames
        self.cine_modalities = cine_modalities
        self.analysis = analysis or {}

    def get_geometry(self, series_key: str):
        if series_key not in PHANTOM_STACKS:
            return None  # a phantom stack: its slices at their depths, 1 mm pixels, slice 1 apical
        return VolumeGeometry(*np.eye(3), [np.array([0.0, 0.0, d]) for d in phantom_depths(series_key)], (1.0, 1.0), 128, 128)

    @property
    def segmentable(self) -> tuple[str, ...]:
        return (*self.cine_modalities, "LGE_SAX")

    def segment(self, series_key: str, cached_only: bool = False):
        return phantom_lge()[1] if series_key == "LGE_SAX" and series_key in self.analysis else None


class FakeDicomController:
    """apply_preset delegates to the REAL DICOMController.apply_preset (same
    object.__new__ bypass as ChatController), so its result text can't drift
    from what the app actually does. run_segmentation only records the request."""

    def __init__(self, model: FakeModel, view: FakeView, current_time_index: int = 0):
        self.current_time_index = current_time_index
        self._real = _build_real_dicom_controller(model, view)
        self._model, self._view = model, view
        self.segmentations: list[str] = []
        view.display_area_widget.segmented |= set(model.analysis)  # a cached segmentation loads with its series

    def apply_preset(self, name: str):
        return self._real.apply_preset(name)

    def show_series_when_loaded(self, panel_idx: int, series_key: str, axis: str = "transverse"):
        self._real.show_series_when_loaded(panel_idx, series_key, axis)

    def segmentation_for(self, series_key: str):
        return self._real.segmentation_for(series_key)

    def is_segmenting(self, series_key: str) -> bool:
        return False

    def run_segmentation(self, series_key: str) -> bool:
        """Records the request; a series measured in the case (a cached segmentation) shows at once."""
        self.segmentations.append(series_key)
        if series_key in self._model.analysis:
            self._view.display_area_widget.segmented.add(series_key)
        return True

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
