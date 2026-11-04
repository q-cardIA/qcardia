from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from qcardia_app.utils import study_log
from qcardia_app.utils.hanging_protocols import FLOW_MAGNITUDE_SUFFIX
from qcardia_app.utils.windowing import window_to_uint8
from qcardia_app.view.custom_graphics_view import CustomQGraphicsView
from qcardia_app.view.loading_overlay import LoadingOverlay
from qcardia_app.view.series_browser_widget import SERIES_KEY_MIME_TYPE
from qcardia_app.utils.layouts import layout_cells, split_layout
from qcardia_app.view.volume_geometry import VolumeGeometry

_EMPTY_TITLE = "Drop a series here"
ROI_PADDING = 0.5  # zoom to ROI: space on each side of the segmentation, as a fraction of its size


@dataclass
class _Panel:
    container: QWidget
    viewer: CustomQGraphicsView
    slider: QSlider
    title_label: QLabel
    slice_label: QLabel
    loading_overlay: LoadingOverlay


CINE_TIME_AXIS = "CINE"


class _PanelDropContainer(QWidget):
    """Grid-cell container that accepts a dragged series card and re-emits its key."""

    series_dropped = Signal(int, str)

    def __init__(self, panel_idx: int, parent=None):
        super().__init__(parent)
        self._panel_idx = panel_idx
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(SERIES_KEY_MIME_TYPE):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(SERIES_KEY_MIME_TYPE):
            event.acceptProposedAction()

    def dropEvent(self, event):
        key = bytes(event.mimeData().data(SERIES_KEY_MIME_TYPE)).decode("utf-8")
        self.series_dropped.emit(self._panel_idx, key)
        event.acceptProposedAction()


class CentralDisplayAreaWidget(QWidget):
    time_step_requested = Signal(int)
    play_toggle_requested = Signal()
    time_axis_changed = Signal(int, int)  # (frames, frame) of the time axis the time slider now drives
    segmentation_requested = Signal(str)  # the contour toggle of a panel whose series has no segmentation yet
    panel_selected = Signal(bool)  # a panel was selected: whether its series has frames
    panels_moved = Signal(object)  # after a layout change: old panel index -> new, for the panels that stay

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        opt = QSize(1280, 720)
        self.setBaseSize(
            QSize(
                int((1325 / opt.width()) * self.parent.baseSize().width()),
                int((950 / opt.height()) * self.parent.baseSize().height()),
            )
        )

        # Series data — keyed by series_key (e.g. "CINE_SAX", "LGE_SAX")
        self._volumes: dict[str, np.ndarray] = {}       # original values: (T,Z,H,W) or (Z,H,W) or (T,H,W)
        self._windows: dict[str, tuple[float, float]] = {}  # (center, width), shared by a series' panels
        self._default_windows: dict[str, tuple[float, float]] = {}
        self._contour_cache: dict[tuple, list[dict] | None] = {}
        self._is_temporal: dict[str, bool] = {}
        self._geometries: dict[str, VolumeGeometry] = {}
        self._segmentations: dict[str, np.ndarray] = {}

        # Per-panel state — grows/shrinks with the grid
        self._panel_series_key: list[str | None] = []
        self._panel_axis: list[str] = []
        self._panel_slice_idx: list[int] = []
        self._panel_depth: list[int] = []
        self._panel_linked: list[bool] = []
        # Per linked plane: the last view as its panel had it, and in patient coordinates (see _physical).
        self._group_views: dict[tuple[str, str], tuple[tuple[float, float, float], tuple | None]] = {}
        self._panel_contours: list[bool] = []  # show the segmentation in this panel
        self._panel_window: list[tuple[float, float] | None] = []  # a panel's own window, else its series'
        self._panel_time: list[str | None] = []  # a panel's own time axis (see hold_frame), else its series'
        self._own_axes = 0  # own time axes made so far, for unique names

        # Shared world-space crosshair; None until the user first clicks
        self._world_crosshair: np.ndarray | None = None
        self._pending_crosshair: tuple[int, int, int] | None = None  # latest drag point, applied once per event loop
        self._pending_window: dict[str | int, tuple[float, float]] = {}  # by series, or by panel if it has its own

        # The cines share one time axis; each perfusion run has its own, as its frame count differs.
        self._time_index: dict[str, int] = {}
        self._time_axis = CINE_TIME_AXIS
        self._selected: int | None = None  # the panel last used, see _show_selected

        # Live panel widgets (rebuilt by set_grid_size)
        self._panels: list[_Panel] = []
        self._dividers: list[QLabel] = []
        self._cells: list[tuple[int, int, int, int]] = []  # per panel: (row, col, rows, cols) it covers
        self._split: tuple[int, int, int, int] | None = None  # (left rows, cols, right rows, cols), see split_layout
        self._grid_rows: int = 0
        self._grid_cols: int = 0

        self.setObjectName("centralDisplayArea")
        self._layout = QGridLayout(self)
        self._layout.setHorizontalSpacing(0)
        self._layout.setVerticalSpacing(0)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self.set_grid_size(2, 2)

    # ── Grid layout ───────────────────────────────────────────────────────

    @property
    def n_panels(self) -> int:
        return len(self._cells)

    @property
    def grid_size(self) -> tuple[int, int]:
        return self._grid_rows, self._grid_cols

    def panel_assignments(self) -> list[dict]:
        """Per-panel state (series key, axis, slice, whether a segmentation exists) —
        used to describe the current layout, e.g. in the session log."""
        panels = []
        for i in range(self.n_panels):
            key, axis = self._panel_series_key[i], self._panel_axis[i]
            panel = {"idx": i, "series_key": key, "axis": axis,
                     "has_segmentation": self._shows_segmentation(i), "segmented": key in self._segmentations,
                     "own_window": self._panel_window[i] is not None}
            n = self._n_slices(key, axis) if key else 0
            if n > 1:
                panel.update(slice=self._panel_slice_idx[i], n_slices=n)
            if self._panel_time[i]:
                panel["own_frame"] = (self.time_index(self._panel_time[i]), self.panel_frames(i))
            panels.append(panel)
        return panels

    def state(self) -> dict:
        """The screen for the session log, 1-based as the user sees it."""
        def panel(i):
            view = self._panels[i].viewer.view()
            return {"series": self._panel_series_key[i], "axis": self._panel_axis[i],
                    "slice": self._panel_slice_idx[i] + 1, "zoom": [round(v, 3) for v in view] if view else None,
                    "window": self._panel_window[i], "contours": self._shows_segmentation(i),
                    "linked": self._panel_linked[i],
                    "own_frame": self.time_index(self._panel_time[i]) + 1 if self._panel_time[i] else None,
                    } if self._panel_series_key[i] else None
        return {"grid": [self._grid_rows, self._grid_cols], "split": self._split,
                "frames": {axis: t + 1 for axis, t in self._time_index.items()},
                "windows": {k: [round(v, 1) for v in w] for k, w in self._windows.items()},
                "panels": [panel(i) for i in range(self.n_panels)]}

    def set_crosshair_shown(self, shown: bool):
        CustomQGraphicsView.crosshair_shown = shown
        for p in self._panels:
            p.viewer.show_crosshair(shown)

    def set_grid_size(self, rows: int, cols: int, shift: tuple[int, int] = (0, 0)):
        """Panels are numbered in reading order. A panel keeps its series if its cell, moved by shift
        (rows, cols), is in the new grid."""
        self._set_layout(rows, cols, layout_cells(rows, cols), shift)

    def set_split_layout(self, split: tuple[int, int, int, int]):
        """The window in two equal halves, each its own grid (see utils.layouts.split_layout)."""
        self._set_layout(*split_layout(split), split=split)

    @property
    def split(self) -> tuple[int, int, int, int] | None:
        return self._split

    def _set_layout(self, rows: int, cols: int, cells, shift: tuple[int, int] = (0, 0), split=None):
        """Panels keep their series by grid cell, or by panel number when a split layout is set or left."""
        by_number = split is not None or self._split is not None
        if cells == self._cells and split == self._split and not any(shift):
            return

        # Preserve series assignments for panels that survive the resize
        old_state = list(zip(self._panel_series_key, self._panel_axis, self._panel_slice_idx, self._panel_depth,
                             self._panel_linked, self._panel_contours, self._panel_window, self._panel_time))
        if by_number and not any(shift):
            moves = {i: i for i in range(min(len(cells), len(old_state)))}  # old panel -> new panel
        else:
            at = {cell[:2]: i for i, cell in enumerate(cells)}
            moves = {i: at[r + shift[0], c + shift[1]] for i, (r, c, _, _) in enumerate(self._cells)
                     if (r + shift[0], c + shift[1]) in at}
        moved = {new: old_state[old] for old, new in moves.items()}
        self._split = split
        old_rows, old_cols = self._grid_rows, self._grid_cols

        # Tear down existing layout and panels
        while self._layout.count():
            self._layout.takeAt(0)
        for w in [p.container for p in self._panels] + self._dividers:
            w.setParent(None)
            w.deleteLater()
        self._panels.clear()
        self._dividers.clear()

        self._grid_rows, self._grid_cols, self._cells = rows, cols, cells
        empty = (None, "transverse", 0, 0, True, True, None, None)
        (self._panel_series_key, self._panel_axis, self._panel_slice_idx, self._panel_depth,
         self._panel_linked, self._panel_contours, self._panel_window, self._panel_time) = map(
            list, zip(*(moved.get(i, empty) for i in range(len(cells)))))

        # Build fresh panels
        for idx in range(len(cells)):
            self._panels.append(self._make_panel(idx))
            self._sync_toggles(idx)

        # Place panels, and dividers wherever two neighbouring cells belong to different panels
        owner = {}
        for idx, (p, (r, c, h, w)) in enumerate(zip(self._panels, cells)):
            self._layout.addWidget(p.container, 2 * r, 2 * c, 2 * h - 1, 2 * w - 1)
            owner.update({(r + i, c + j): idx for i in range(h) for j in range(w)})
        for r in range(rows):
            for c in range(cols):
                around = {owner[r + i, c + j] for i in (0, 1) for j in (0, 1) if r + i < rows and c + j < cols}
                if r + 1 < rows and owner[r, c] != owner[r + 1, c]:
                    self._add_divider(True, 2 * r + 1, 2 * c)
                if c + 1 < cols and owner[r, c] != owner[r, c + 1]:
                    self._add_divider(False, 2 * r, 2 * c + 1)
                if r + 1 < rows and c + 1 < cols and len(around) > 1:
                    self._add_divider(True, 2 * r + 1, 2 * c + 1)

        # Clear stretch factors left over from the previous grid size, otherwise a
        # shrink (e.g. viability 2x4 -> functional 2x2) leaves unused rows/cols
        # from the larger grid still reserving space.
        for r in range(2 * max(rows, old_rows)):
            self._layout.setRowStretch(r, 0)
        for c in range(2 * max(cols, old_cols)):
            self._layout.setColumnStretch(c, 0)

        for r in range(2 * rows):
            self._layout.setRowStretch(r, 1 if r % 2 == 0 else 0)
        for c in range(2 * cols):
            self._layout.setColumnStretch(c, 1 if c % 2 == 0 else 0)

        # Re-render any panels that already have a series assigned
        for i in range(len(cells)):
            if self._panel_series_key[i] is not None:
                # Rebuilt panels start with the empty title.
                self._panels[i].title_label.setText(self._panel_series_key[i])
                self._panels[i].viewer.new_series(self._view_of(i))
                self._update_panel(i)
        self._show_cine_time_if_unseen()
        self._selected = moves.get(self._selected)
        self._show_selected()
        self.panels_moved.emit(moves)
        QTimer.singleShot(0, self, self._relink_views)  # the new panels have their sizes after a layout pass

    def _relink_views(self):
        """The linked view in each linked panel, converted for the size its panel now has."""
        for i in range(self.n_panels):
            if (view := self._view_of(i)) is not None:
                self._panels[i].viewer.set_view(view)

    def _add_divider(self, horizontal: bool, row: int, col: int):
        div = _make_divider(horizontal)
        self._dividers.append(div)
        self._layout.addWidget(div, row, col)

    def _make_panel(self, idx: int) -> _Panel:
        container = _PanelDropContainer(idx)
        container.series_dropped.connect(self._on_panel_series_dropped)
        container.setProperty("role", "viewerCard")
        container.setObjectName(f"panel_{idx}")
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(2)

        hdr = QHBoxLayout()
        hdr.setContentsMargins(6, 4, 6, 4)
        hdr.setSpacing(8)
        title = QLabel(_EMPTY_TITLE)
        title.setProperty("role", "viewerTitle")
        title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)  # clip, never widen the grid
        slice_lbl = QLabel("")
        slice_lbl.setProperty("role", "viewerMeta")
        number = QLabel(str(idx + 1))  # the panel's number for the user, from 1
        number.setProperty("role", "viewerIndex")
        hdr.addWidget(number)
        hdr.addWidget(title, 1)
        hdr.addWidget(slice_lbl)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(0)
        slider.setEnabled(False)
        slider.setVisible(False)
        slider.setMinimumHeight(20)
        slider.setProperty("role", "sliceSlider")
        slider.valueChanged.connect(lambda val, i=idx: self._on_panel_slider_changed(i, val))

        viewer = CustomQGraphicsView(
            view_type=f"panel_{idx}", size=QSize(100, 100), parent=self
        )
        viewer.setProperty("role", "imageViewport")
        viewer.coordinates_changed.connect(
            lambda x, y, i=idx: self._on_panel_coordinates_changed(i, x, y)
        )
        viewer.action_requested.connect(lambda a, n, i=idx: self._on_viewer_action(i, a, n))
        viewer.window_dragged.connect(lambda dx, dy, i=idx: self._on_window_drag(i, dx, dy))
        viewer.view_changed.connect(lambda *view, i=idx: self._on_view_changed(i, view))
        viewer.focused.connect(lambda i=idx: self._select_time_axis(i))

        vbox.addLayout(hdr)
        vbox.addWidget(slider)
        vbox.addWidget(viewer)

        overlay = LoadingOverlay(container)

        return _Panel(
            container=container,
            viewer=viewer,
            slider=slider,
            title_label=title,
            slice_label=slice_lbl,
            loading_overlay=overlay,
        )

    # ── External API ──────────────────────────────────────────────────────

    def register_series(
        self,
        series_key: str,
        volume: np.ndarray,
        is_temporal: bool,
        window: tuple[float, float],
        geometry: VolumeGeometry | None = None,
    ):
        """Register a loaded series volume, its default (center, width) window and optional DICOM geometry."""
        self._volumes[series_key] = volume
        self._windows[series_key] = self._default_windows[series_key] = window
        self._segmentations.pop(series_key, None)  # made from the volume this one replaces
        self._forget_contours(series_key)
        self._is_temporal[series_key] = is_temporal
        if geometry is not None:
            self._geometries[series_key] = geometry

    def clear_series(self):
        """Forget the previous subject's volumes, windows and segmentations."""
        for store in (self._volumes, self._is_temporal, self._geometries, self._segmentations,
                      self._windows, self._default_windows, self._contour_cache):
            store.clear()
        self._world_crosshair = None
        self._group_views.clear()
        self._time_index.clear()
        self._time_axis = CINE_TIME_AXIS

    def refresh_panels(self, series_key: str | None = None):
        """Show each panel's series again (only series_key's panels, if given); empty panels whose series is gone."""
        for i, key in enumerate(self._panel_series_key):
            if key is None or series_key not in (None, key):
                continue
            if key in self._volumes:
                self.set_panel_series(i, key, axis=self._panel_axis[i])
            else:
                self.set_panel_empty(i)

    def set_geometry(self, series_key: str, geom: VolumeGeometry):
        self._geometries[series_key] = geom

    def is_series_registered(self, series_key: str) -> bool:
        return series_key in self._volumes

    def n_slices(self, series_key: str, axis: str = "transverse") -> int:
        return self._n_slices(series_key, axis)

    def set_panel_series(
        self, panel_idx: int, series_key: str, axis: str = "transverse", slice_idx: int | None = None
    ):
        """Assign a registered series to a panel slot, at slice_idx, else at the position of the stack
        it replaces if that is parallel, else at the middle slice."""
        if panel_idx < 0 or panel_idx >= self.n_panels:
            return
        old_key, old_z = self._panel_series_key[panel_idx], self._panel_slice_idx[panel_idx]
        if slice_idx is None and old_key and self._panel_axis[panel_idx] == axis.lower() == "transverse":
            # Same plane, e.g. CINE_SAX -> LGE_SAX: keep the position in the heart (issue #34).
            slice_idx = old_z if old_key == series_key else self.matching_slice(old_key, old_z, series_key)
        self._panel_series_key[panel_idx] = series_key
        self._panel_axis[panel_idx] = axis.lower()
        self._panel_linked[panel_idx] = True
        # Contours as the series' other panels show them: a cached segmentation is in the viewer from the start.
        self._panel_contours[panel_idx] = any(self._shows_segmentation(i) for i in range(self.n_panels)
                                              if i != panel_idx and self._panel_series_key[i] == series_key)
        self._panel_window[panel_idx] = None
        self._time_index.pop(self._panel_time[panel_idx], None)
        self._panel_time[panel_idx] = None  # a new series shows the frame of the others
        self._sync_toggles(panel_idx)

        n = self._n_slices(series_key, axis)
        sl = self._panels[panel_idx].slider
        sl.blockSignals(True)
        sl.setMaximum(max(0, n - 1))
        sl.blockSignals(False)
        sl.setEnabled(n > 1)
        sl.setVisible(n > 1)
        self._set_slice(panel_idx, n // 2 if slice_idx is None else min(slice_idx, max(0, n - 1)))
        self._panel_depth[panel_idx] = self._panel_slice_idx[panel_idx]
        # After the slice: the linked view is found on the panel's slice of the new series.
        self._panels[panel_idx].viewer.new_series(self._view_of(panel_idx))

        self._panels[panel_idx].title_label.setText(series_key)
        self._update_panel(panel_idx)
        self._show_cine_time_if_unseen()
        self._show_selected()

    def matching_slice(self, source_key: str, z: int, series_key: str) -> int | None:
        """The transverse slice of series_key nearest to slice z of source_key, if both stacks have
        geometry and are parallel, e.g. LGE_SAX for CINE_SAX."""
        src, dst = self._geometries.get(source_key), self._geometries.get(series_key)
        if src is None or dst is None or not src.is_parallel_to(dst):
            return None
        if len(dst.ipp_list) != self._n_slices(series_key, "transverse"):
            return None
        return src.nearest_slice(dst, z)

    def zoom_panels(self, panels: list[int], mode: str, percent: float = 100.0, analysis: dict | None = None) -> list[int]:
        """mode 'roi' frames the analyses' boxes (row0, col0, row1, col1) with ROI_PADDING, also in
        panels of a parallel series of the same plane; 'fill' fills the panel with the image, 'fit' shows
        all of it, 'percent' zooms to percent of fit. Returns the panels it zoomed."""
        # list(): the load thread can still add entries
        boxes = {k: a["box"] for k, a in list((analysis or {}).items()) if a.get("box")}
        done = []
        for i in panels:
            key = self._panel_series_key[i]
            if key is None:
                continue
            viewer = self._panels[i].viewer
            if mode == "roi":
                box = self._roi_in_panel(i, boxes or {})
                if box is None:
                    continue
                viewer.frame(box)
            elif mode == "fill":
                viewer.frame(None, cover=True)
            elif mode == "fit":
                viewer.frame(None)
            else:
                viewer.set_zoom(percent / 100)
            self._group_views.pop(self._group(i), None)  # a series dropped in later starts fitted
            done.append(i)
        if mode == "roi":  # one framing per linked plane, the first panel's, so perfusion and LGE match
            for group in {self._group(i) for i in done if self._panel_linked[i]}:
                first, *rest = [i for i in done if self._panel_linked[i] and self._group(i) == group]
                physical = self._physical(first, self._panels[first].viewer.view())
                for i in rest:  # a panel without geometry keeps its own framing
                    if (view := self._relative(i, physical, None)) is not None:
                        self._panels[i].viewer.set_view(view)
        return done

    def _roi_in_panel(self, panel_idx: int, boxes: dict) -> tuple[float, float, float, float] | None:
        key, axis = self._panel_series_key[panel_idx], self._panel_axis[panel_idx]
        if axis != "transverse":  # the boxes are in the acquired (transverse) pixel grid
            return None
        if key in boxes:
            r0, c0, r1, c1 = boxes[key]
        else:
            # Through world coordinates from a segmented series of the same plane, e.g. CINE_SAX for LGE_SAX
            geom = self._geometries.get(key)
            source = next((k for k in boxes if k in self._geometries and geom is not None
                           and k.rsplit("_", 1)[-1] == key.rsplit("_", 1)[-1]
                           and self._geometries[k].is_parallel_to(geom)), None)
            if source is None:
                return None
            sr0, sc0, sr1, sc1 = boxes[source]
            corners = [geom.world_to_pixel(self._geometries[source].pixel_to_world(r, c, 0))[:2]
                       for r in (sr0, sr1) for c in (sc0, sc1)]
            (r0, c0), (r1, c1) = np.min(corners, axis=0), np.max(corners, axis=0)
        pad_r, pad_c = ROI_PADDING * (r1 - r0 + 1), ROI_PADDING * (c1 - c0 + 1)
        return r0 - pad_r, c0 - pad_c, r1 + pad_r, c1 + pad_c

    def set_panel_waiting(self, panel_idx: int, series_key: str):
        """Empty, with the series that it shows once that series has loaded."""
        self.set_panel_empty(panel_idx)
        if 0 <= panel_idx < self.n_panels:
            self._panels[panel_idx].title_label.setText(f"{series_key} · loading…")

    def is_panel_empty(self, panel_idx: int) -> bool:
        return 0 <= panel_idx < self.n_panels and self._panel_series_key[panel_idx] is None

    def set_panel_empty(self, panel_idx: int):
        if panel_idx < 0 or panel_idx >= self.n_panels:
            return
        self._panel_series_key[panel_idx] = None
        self._release_frame(panel_idx)
        p = self._panels[panel_idx]
        p.title_label.setText(_EMPTY_TITLE)
        p.slice_label.setText("")
        p.slider.setEnabled(False)
        p.slider.setVisible(False)
        p.viewer.clear_display()
        self._show_cine_time_if_unseen()
        self._show_selected()

    def get_thumbnail(self, series_key: str) -> np.ndarray | None:
        """Representative 2D uint8 slice (the middle slice at the middle frame) for a browser card."""
        if series_key not in self._volumes:
            return None
        n = self._n_slices(series_key, "transverse")
        mid = n // 2 if n > 0 else 0
        t = self._volumes[series_key].shape[0] // 2 if self._is_temporal.get(series_key) else 0
        return self._display(series_key, self._get_slice_2d(series_key, "transverse", mid, t=t))

    def _on_panel_series_dropped(self, panel_idx: int, series_key: str):
        if self.is_series_registered(series_key):
            self.set_panel_series(panel_idx, series_key)
            study_log.log("series_dropped", panel=panel_idx, series=series_key)

    def set_segmentation(self, series_key: str, seg_volume: np.ndarray):
        self._segmentations[series_key] = seg_volume
        self._forget_contours(series_key)
        for i in range(self.n_panels):
            if self._panel_series_key[i] == series_key:
                self._sync_toggles(i)
                self._update_panel(i)

    def has_segmentation(self, series_key: str) -> bool:
        return series_key in self._segmentations

    def segmentation(self, series_key: str) -> np.ndarray | None:
        return self._segmentations.get(series_key)

    def volume(self, series_key: str) -> np.ndarray | None:
        return self._volumes.get(series_key)

    def window(self, series_key: str, default: bool = False, panel: int | None = None) -> tuple[float, float] | None:
        """(center, width) of series_key, its default window, or the window panel shows it with."""
        if panel is not None and self._panel_window[panel] is not None:
            return self._panel_window[panel]
        return (self._default_windows if default else self._windows).get(series_key)

    def set_window(self, series_key: str, window: tuple[float, float], panels: list[int] | None = None):
        """No narrower than the drag allows. With panels, their own window; without, the series' window,
        which its panels with an own window then follow again."""
        center, width = window
        window = (center, max(width, self._narrowest_width(series_key)))
        if panels is not None:
            for i in panels:
                self._panel_window[i] = window
                self._update_panel(i, contours=False)
            return
        self._windows[series_key] = window
        for i in range(self.n_panels):
            if self._panel_series_key[i] == series_key:
                self._panel_window[i] = None
        self._refresh_images(series_key)

    def _narrowest_width(self, series_key: str) -> float:
        return self._default_windows[series_key][1] / 100

    def set_panel_contours(self, panel_idx: int, shown: bool):
        self._panel_contours[panel_idx] = shown
        self._sync_toggles(panel_idx)
        self._update_panel(panel_idx)

    def drive_time(self, axis: str = CINE_TIME_AXIS):
        """Let the time slider drive the cines, or a perfusion run, e.g. before playing them."""
        self._set_time_axis(axis)

    def time_index(self, axis: str) -> int:
        """The frame (0-based) that the panels of a time axis show."""
        return self._time_index.get(axis, 0)

    def hold_frame(self, panels: list[int], t: int):
        """These panels show frame t on their own (e.g. ED next to ES): the time slider and playback of the
        others leave them. Selected, such a panel has the time slider to itself."""
        for i in panels:
            if self._panel_time_axis(i) is None:
                continue
            if self._panel_time[i] is None:
                self._own_axes += 1
                self._panel_time[i] = f"{self.time_axis(self._panel_series_key[i])}#{self._own_axes}"
                self._sync_toggles(i)
            self.set_time_index(t, self._panel_time[i])
        if self._selected in panels:  # the time slider follows the selected panel
            self._set_time_axis(self._panel_time_axis(self._selected) or CINE_TIME_AXIS)

    def release_frames(self, axis: str):
        """Panels with their own frame of axis (see hold_frame) show axis' frame again."""
        for i in range(self.n_panels):
            if self._panel_time[i] and self._panel_time[i].partition("#")[0] == axis:
                self._release_frame(i)
        self._show_cine_time_if_unseen()

    def _release_frame(self, panel_idx: int):
        own = self._panel_time[panel_idx]
        self._time_index.pop(own, None)
        self._panel_time[panel_idx] = None
        if own and self._time_axis == own:  # the time slider drove this panel alone
            self._set_time_axis(own.partition("#")[0])
        self._sync_toggles(panel_idx)
        self._update_panel(panel_idx)

    def panel_frames(self, panel_idx: int) -> int:
        """The frame count of a panel's series; 0 without frames."""
        if self._panel_time_axis(panel_idx) is None:
            return 0
        return self._volumes[self._panel_series_key[panel_idx]].shape[0]

    def _panel_time_axis(self, panel_idx: int) -> str | None:
        """The time axis a panel's frame follows: its own, else its series'; None without frames."""
        key = self._panel_series_key[panel_idx]
        if not key or not self._is_temporal.get(key):
            return None
        return self._panel_time[panel_idx] or self.time_axis(key)

    def show_segmentation(self, series_key: str) -> list[int]:
        """Turn the contours on in every panel of series_key, also before its segmentation arrives.
        Returns those panels."""
        panels = [i for i in range(self.n_panels) if self._panel_series_key[i] == series_key]
        for i in panels:
            self._panel_contours[i] = True
            self._sync_toggles(i)
            if series_key in self._segmentations:  # else set_segmentation draws them when it arrives
                self._update_panel(i)
        return panels

    def _shows_segmentation(self, panel_idx: int) -> bool:
        return self._panel_contours[panel_idx] and self._panel_series_key[panel_idx] in self._segmentations

    def show_segmentation_loading(self, series_key: str, text: str = "Segmenting…"):
        """Show a small loading overlay only over the panel(s) currently
        displaying series_key, instead of blocking the whole display area —
        so segmenting CINE and LGE at the same time doesn't have one
        overlay's hide_overlay() prematurely dismiss the other's."""
        for i in range(self.n_panels):
            if self._panel_series_key[i] == series_key:
                self._panels[i].loading_overlay.show_message(text)

    def hide_segmentation_loading(self, series_key: str):
        for i in range(self.n_panels):
            if self._panel_series_key[i] == series_key:
                self._panels[i].loading_overlay.hide_overlay()

    @property
    def current_time_index(self) -> int:
        """The cines' frame."""
        return self._time_index.get(CINE_TIME_AXIS, 0)

    @property
    def driven_time_axis(self) -> str:
        """The time axis the time slider drives: the cines', or a perfusion or flow series'."""
        return self._time_axis

    @staticmethod
    def time_axis(series_key: str) -> str:
        if series_key.startswith(("PERF_", "PC_")):
            return series_key.removesuffix(FLOW_MAGNITUDE_SUFFIX)
        return CINE_TIME_AXIS

    def n_frames(self, time_axis: str) -> int:
        keys = {key for key in self._volumes if self._is_temporal.get(key) and self.time_axis(key) == time_axis}
        keys.update(self._panel_series_key[i] for i in range(self.n_panels) if self._panel_time[i] == time_axis)
        return max((self._volumes[key].shape[0] for key in keys), default=0)

    def set_time_index(self, t: int, axis: str | None = None):
        """Show frame t of axis, by default the time axis that the time slider drives. Frame t of another
        axis given here also moves the slider when it drives that axis."""
        moved = axis or self._time_axis
        self._time_index[moved] = t
        for i in range(self.n_panels):
            if self._panel_time_axis(i) == moved:
                self._update_panel(i)
        if axis is not None and axis == self._time_axis:
            self.time_axis_changed.emit(self.n_frames(axis), t)

    def _show_cine_time_if_unseen(self):
        """Once no panel shows the time axis the time slider drives (a perfusion run, a panel's own frame),
        it drives the cines again."""
        if self._time_axis != CINE_TIME_AXIS and not any(
                self._panel_time_axis(i) == self._time_axis for i in range(self.n_panels)):
            self._set_time_axis(CINE_TIME_AXIS)

    def _select_time_axis(self, panel_idx: int):
        """The time slider follows the panel last used: its own frame, a perfusion run's, else the cines'."""
        self._selected = panel_idx
        self._set_time_axis(self._panel_time_axis(panel_idx) or CINE_TIME_AXIS)
        self._show_selected()

    def _show_selected(self):
        """Whether the selected panel has frames, for the time bar; also after its series changes."""
        key = self._panel_series_key[self._selected] if self._selected is not None else None
        self.panel_selected.emit(key is None or bool(self._is_temporal.get(key)))

    def _set_time_axis(self, axis: str):
        """Let the time slider drive axis, at that axis' frame."""
        if axis != self._time_axis:
            self._time_axis = axis
            self.time_axis_changed.emit(self.n_frames(axis), self._time_index.get(axis, 0))

    # ── Slice/axis helpers ────────────────────────────────────────────────

    def _n_slices(self, series_key: str, axis: str) -> int:
        vol = self._volumes.get(series_key)
        if vol is None:
            return 0
        is_temporal = self._is_temporal.get(series_key, False)
        if vol.ndim == 4:                               # (T, Z, H, W)
            return {"transverse": vol.shape[1],
                    "coronal":    vol.shape[2],
                    "sagittal":   vol.shape[3]}.get(axis, vol.shape[1])
        if vol.ndim == 3 and not is_temporal:           # (Z, H, W)
            return {"transverse": vol.shape[0],
                    "coronal":    vol.shape[1],
                    "sagittal":   vol.shape[2]}.get(axis, vol.shape[0])
        return 0                                        # (T, H, W) — no z navigation

    @staticmethod
    def _extract_slice(vol_3d: np.ndarray, axis: str, idx: int) -> np.ndarray:
        if axis == "transverse":
            return vol_3d[min(max(0, idx), vol_3d.shape[0] - 1)]
        if axis == "coronal":
            return vol_3d[:, min(max(0, idx), vol_3d.shape[1] - 1), :]
        # sagittal
        return vol_3d[:, :, min(max(0, idx), vol_3d.shape[2] - 1)]

    def _get_slice_2d(
        self, series_key: str, axis: str, z: int, t: int
    ) -> np.ndarray | None:
        vol = self._volumes.get(series_key)
        if vol is None:
            return None
        is_temporal = self._is_temporal.get(series_key, False)
        if vol.ndim == 4:                               # (T, Z, H, W)
            return self._extract_slice(vol[min(t, vol.shape[0] - 1)], axis, z)
        if vol.ndim == 3 and is_temporal:              # (T, H, W)
            return vol[min(t, vol.shape[0] - 1)]
        if vol.ndim == 3:                              # (Z, H, W)
            return self._extract_slice(vol, axis, z)
        return None

    def _display(self, series_key: str, slice_2d: np.ndarray, panel_idx: int | None = None) -> np.ndarray:
        return window_to_uint8(slice_2d, *self.window(series_key, panel=panel_idx))

    def _forget_contours(self, series_key: str):
        self._contour_cache = {k: v for k, v in self._contour_cache.items() if k[0] != series_key}

    def _get_contours(self, series_key: str, axis: str, z: int, t: int) -> list[dict] | None:
        key = (series_key, axis, z, t)
        if key not in self._contour_cache:
            self._contour_cache[key] = self._compute_contours(series_key, axis, z, t)
        return self._contour_cache[key]

    def _compute_contours(
        self, series_key: str, axis: str, z: int, t: int
    ) -> list[dict] | None:
        seg = self._segmentations.get(series_key)
        if seg is None:
            return None
        try:
            is_temporal = self._is_temporal.get(series_key, False)
            if seg.ndim == 4:                          # (T, Z, H, W)
                seg_2d = self._extract_slice(seg[min(t, seg.shape[0] - 1)], axis, z)
            elif seg.ndim == 3 and is_temporal:        # (T, H, W)
                seg_2d = seg[min(t, seg.shape[0] - 1)]
            elif seg.ndim == 3:                        # (Z, H, W)
                seg_2d = self._extract_slice(seg, axis, z)
            elif seg.ndim == 2:
                seg_2d = seg
            else:
                return None
            # LGE class 1 = myocardium, class 2 = scar; scar tissue is still
            # myocardium anatomically, so its contour should include scar pixels
            # rather than leaving a gap where the wall is scarred.
            mask_unions = {1: {2}} if series_key.startswith("LGE") else None
            return _extract_segmentation_contours(seg_2d, mask_unions)
        except Exception as e:
            logging.error("Contour error for %s: %s", series_key, e)
            return None

    # ── Panel rendering ───────────────────────────────────────────────────

    def _update_panel(self, panel_idx: int, contours: bool = True):
        key = self._panel_series_key[panel_idx]
        if key is None:
            return
        p = self._panels[panel_idx]
        axis = self._panel_axis[panel_idx]
        z = self._panel_slice_idx[panel_idx]
        t = self.time_index(self._panel_time_axis(panel_idx))

        slice_2d = self._get_slice_2d(key, axis, z, t)
        if slice_2d is None:
            return

        geom = self._geometries.get(key)
        p.viewer.set_pixel_spacing(geom.pixel_spacing if geom is not None else (1.0, 1.0))
        p.viewer.set_contour_line_mm(2.5 if key.startswith("LGE") else 2.0)
        if contours:
            shown = self._get_contours(key, axis, z, t) if self._panel_contours[panel_idx] else None
            p.viewer.update_contour_paths(shown)
        p.viewer.update_slice_view(self._display(key, slice_2d, panel_idx))

        n = self._n_slices(key, axis)
        p.slice_label.setText(f"Slice: {z + 1}/{n}" if n > 1 else "")

    def update_all_panels(self):
        for i in range(self.n_panels):
            self._update_panel(i)

    # ── World-coordinate crosshair sync ──────────────────────────────────

    def _on_panel_coordinates_changed(self, panel_idx: int, sig_x: int, sig_y: int):
        # A drag sends many points: keep the latest and apply it once the queued events are handled.
        if self._pending_crosshair is None:
            QTimer.singleShot(0, self._apply_pending_crosshair)
        self._pending_crosshair = (panel_idx, sig_x, sig_y)

    def _apply_pending_crosshair(self):
        if self._pending_crosshair is None:
            return
        panel_idx, col, row = self._pending_crosshair  # signal emits (col, row) as (x, y)
        self._pending_crosshair = None
        if panel_idx >= self.n_panels:
            return
        key = self._panel_series_key[panel_idx]
        if key is None:
            return
        geom = self._geometries.get(key)
        if geom is not None:
            z = self._panel_slice_idx[panel_idx]
            self._world_crosshair = geom.pixel_to_world(row, col, z)
            self._broadcast_world_crosshair(panel_idx)
        else:
            self._panels[panel_idx].viewer.set_crosshair_from_volume_indices(row, col)

    def _on_panel_slider_changed(self, panel_idx: int, value: int):
        if self._panel_series_key[panel_idx] is not None:
            self._scroll(panel_idx, value - self._panel_depth[panel_idx])

    def _scroll(self, source: int, step: int):
        key, axis = self._panel_series_key[source], self._panel_axis[source]
        depth, n = self._panel_depth[source], self._n_slices(key, axis)
        step = min(max(depth + step, min(depth, 0)), max(depth, n - 1)) - depth
        if not step:
            return
        stacks = Counter(zip(self._panel_series_key, self._panel_axis))
        for i in self._linked_with(source) if self._panel_linked[source] else [source]:
            stack = (self._panel_series_key[i], self._panel_axis[i])
            if (i != source and stack != (key, axis) and stacks[stack] == 1
                    and key in self._geometries and stack[0] in self._geometries):
                continue
            new_depth = self._panel_depth[i] + step
            if self._set_slice(i, min(max(0, new_depth), self._n_slices(*stack) - 1)):
                self._update_panel(i)
            self._panel_depth[i] = new_depth

        new_z = self._panel_slice_idx[source]
        geom = self._geometries.get(key)
        if geom is not None:
            # Keep the existing in-plane crosshair position; move to the new slice
            if self._world_crosshair is not None:
                result = geom.world_to_pixel(self._world_crosshair)
                row = result[0] if result is not None else geom.n_rows / 2.0
                col = result[1] if result is not None else geom.n_cols / 2.0
            else:
                row, col = geom.n_rows / 2.0, geom.n_cols / 2.0
            self._world_crosshair = geom.pixel_to_world(row, col, new_z)
            self._broadcast_world_crosshair(source, scrolled=True)

    def _broadcast_world_crosshair(self, source: int, scrolled: bool = False):
        stacks = Counter(zip(self._panel_series_key, self._panel_axis))
        for i in range(self.n_panels):
            key = self._panel_series_key[i]
            if key is None:
                continue
            geom = self._geometries.get(key)
            if geom is None:
                self._update_panel(i)
                continue
            result = geom.world_to_pixel(self._world_crosshair)
            if result is None:
                self._update_panel(i)
                continue
            r, c, new_z = result
            row = max(0, min(int(round(r)), geom.n_rows - 1))
            col = max(0, min(int(round(c)), geom.n_cols - 1))
            n = self._n_slices(key, self._panel_axis[i])
            follows = n > 0 and (i == source or stacks[key, self._panel_axis[i]] == 1 and (
                not scrolled or self._panel_linked[i] and self._panel_linked[source]))
            if follows and self._set_slice(i, max(0, min(int(round(new_z)), n - 1))):
                self._update_panel(i)
            self._panels[i].viewer.set_crosshair_from_volume_indices(row, col)

    def _group(self, panel_idx: int) -> tuple[str, str] | None:
        """Panels of one plane link zoom and pan, e.g. CINE_SAX with LGE_SAX, PC_AORTA with PC_AORTA_MAG."""
        key = self._panel_series_key[panel_idx]
        if key is None:
            return None
        return key.removesuffix(FLOW_MAGNITUDE_SUFFIX).rsplit("_", 1)[-1], self._panel_axis[panel_idx]

    def _linked_with(self, source: int) -> list[int]:
        group = self._group(source)
        return [i for i in range(self.n_panels) if self._panel_linked[i] and self._group(i) == group]

    def _view_of(self, panel_idx: int) -> tuple[float, float, float] | None:
        stored = self._group_views.get(self._group(panel_idx)) if self._panel_linked[panel_idx] else None
        return self._relative(panel_idx, stored[1], stored[0]) if stored else None

    def _physical(self, panel_idx: int, view) -> tuple[float, np.ndarray] | None:
        """A panel's view (zoom relative to fit, centre as offsets in image widths) as mm per screen pixel
        and the world point in the centre; None without a geometry, e.g. in a reformatted plane."""
        geom = self._transverse_geometry(panel_idx)
        if view is None or geom is None:
            return None
        zoom, fx, fy = view
        world = geom.pixel_to_world((0.5 + fy) * geom.n_rows, (0.5 + fx) * geom.n_cols, self._panel_slice_idx[panel_idx])
        return geom.pixel_spacing[1] / (self._fit_scale(panel_idx, geom) * zoom), world

    def _relative(self, panel_idx: int, physical, fallback):
        """A view in patient coordinates as this panel's own view, for its image size and field of view."""
        geom = self._transverse_geometry(panel_idx)
        if physical is None or geom is None:
            return fallback
        mm_per_px, world = physical
        delta = world - geom.ipp_list[self._panel_slice_idx[panel_idx]]  # projected onto the panel's slice
        row, col = delta @ geom.col_dir / geom.pixel_spacing[0], delta @ geom.row_dir / geom.pixel_spacing[1]
        return (geom.pixel_spacing[1] / mm_per_px / self._fit_scale(panel_idx, geom),
                col / geom.n_cols - 0.5, row / geom.n_rows - 0.5)

    def _transverse_geometry(self, panel_idx: int):
        """The geometry of a panel's series in its acquired plane; None without one or in a reformat."""
        key = self._panel_series_key[panel_idx]
        geom = self._geometries.get(key) if self._panel_axis[panel_idx] == "transverse" else None
        return geom if geom is not None and len(geom.ipp_list) == self._n_slices(key, "transverse") else None

    def _fit_scale(self, panel_idx: int, geom) -> float:
        """Screen pixels per image pixel with the whole image in the panel (as CustomQGraphicsView fits it)."""
        port = self._panels[panel_idx].viewer.viewport()
        return max(min(port.width() / geom.n_cols, port.height() / geom.n_rows), 1e-6)

    def _on_view_changed(self, source: int, view: tuple[float, float, float]):
        group = self._group(source)
        if self._panel_linked[source] and group:
            physical = self._physical(source, view)
            self._group_views[group] = view, physical
            for i in self._linked_with(source):
                if i != source:
                    self._panels[i].viewer.set_view(self._relative(i, physical, view))

    def _set_slice(self, panel_idx: int, z: int) -> bool:
        """Set a panel's slice and slider without triggering the slider's own handler; returns whether it changed."""
        changed = z != self._panel_slice_idx[panel_idx]
        if changed:
            self._panel_slice_idx[panel_idx] = self._panel_depth[panel_idx] = z
        slider = self._panels[panel_idx].slider
        slider.blockSignals(True)
        slider.setValue(z)
        slider.blockSignals(False)
        return changed

    def _sync_toggles(self, panel_idx: int):
        # Lit only while contours show: a series without a segmentation has none to show.
        own = bool(self._panel_time[panel_idx]) if self._panel_time_axis(panel_idx) else None
        self._panels[panel_idx].viewer.set_panel_toggles(self._panel_linked[panel_idx],
                                                        self._shows_segmentation(panel_idx), own)

    # ── Viewer input ─────────────────────────────────────────────────────

    def _on_viewer_action(self, panel_idx: int, action: str, amount: int):
        self._select_time_axis(panel_idx)  # a wheel scroll can come without focus
        key = self._panel_series_key[panel_idx]
        if (action == "slice" and key and self._is_temporal.get(key)
                and self._n_slices(key, self._panel_axis[panel_idx]) <= 1):
            action = "frame"  # single-slice cine (LAX): scrolling steps time instead, as in Medis
        slider = self._panels[panel_idx].slider
        if action == "frame":
            self.time_step_requested.emit(amount)
        elif action == "play":
            self.play_toggle_requested.emit()
        elif key is None:
            return
        elif action == "slice":
            self._scroll(panel_idx, amount)
        elif action in ("first", "last"):
            slider.setValue(0 if action == "first" else slider.maximum())
        elif action == "reset":
            if self._panel_window[panel_idx] is not None:  # its own window: back to the series' window
                self._panel_window[panel_idx] = None
                self._update_panel(panel_idx)
            else:
                self._windows[key] = self._default_windows[key]
                self._refresh_images(key)
        elif action == "link":
            linked = self._panel_linked[panel_idx] = not self._panel_linked[panel_idx]
            self._sync_toggles(panel_idx)
            if linked:
                self._panels[panel_idx].viewer.set_view(self._view_of(panel_idx))
        elif action == "own_frame" and self._panel_time_axis(panel_idx):
            if self._panel_time[panel_idx]:
                self._release_frame(panel_idx)  # back on the others' frame
            else:
                self.hold_frame([panel_idx], self.time_index(self._panel_time_axis(panel_idx)))
            self._select_time_axis(panel_idx)  # the time slider: this panel alone, or the others again
        elif action == "contours":
            if key in self._segmentations:
                self._panel_contours[panel_idx] = not self._panel_contours[panel_idx]
            else:  # show it once it is there: from the cache, or segmented now
                self._panel_contours[panel_idx] = True
                self.segmentation_requested.emit(key)
            self._sync_toggles(panel_idx)
            self._update_panel(panel_idx)

    def _on_window_drag(self, panel_idx: int, dx: float, dy: float):
        key = self._panel_series_key[panel_idx]
        if key is None:
            return
        if not self._pending_window:
            QTimer.singleShot(0, self._apply_pending_window)
        target = panel_idx if self._panel_window[panel_idx] is not None else key  # its own window: only this panel
        sum_dx, sum_dy = self._pending_window.get(target, (0.0, 0.0))
        self._pending_window[target] = (sum_dx + dx, sum_dy + dy)

    def _apply_pending_window(self):
        """Right = wider window (less contrast), down = higher level (darker), scaled to the width."""
        pending, self._pending_window = self._pending_window, {}
        for target, (dx, dy) in pending.items():
            panel = target if isinstance(target, int) else None
            key = self._panel_series_key[panel] if panel is not None else target
            if key not in self._windows or (panel is not None and self._panel_window[panel] is None):
                continue
            center, width = self._panel_window[panel] if panel is not None else self._windows[key]
            window = (center + dy * width / 300, max(self._narrowest_width(key), width * 1.005 ** dx))
            if panel is not None:
                self.set_window(key, window, [panel])
            else:
                self._windows[key] = window
                self._refresh_images(key)

    def _refresh_images(self, series_key: str):
        for i in range(self.n_panels):
            if self._panel_series_key[i] == series_key:
                self._update_panel(i, contours=False)

    # ── Contour editing stubs (to be implemented properly later) ─────────

    def set_contour_edit_mode(self, enabled: bool):
        for p in self._panels:
            p.viewer.set_edit_mode(enabled)

    def set_contour_edit_tool(self, tool: str):
        for p in self._panels:
            p.viewer.set_edit_tool(tool)


# ── Module-level helpers ──────────────────────────────────────────────────

def _make_divider(horizontal: bool) -> QLabel:
    div = QLabel()
    if horizontal:
        div.setFixedHeight(3)
    else:
        div.setFixedWidth(3)
    div.setProperty("role", "divider")
    return div


def _extract_segmentation_contours(
    seg_2d: np.ndarray, mask_unions: dict[int, set[int]] | None = None
) -> list[dict]:
    contours = []
    mask_unions = mask_unions or {}
    class_ids = [c for c in np.unique(seg_2d) if c != 0]
    for class_id in class_ids:
        include_ids = {int(class_id)} | mask_unions.get(int(class_id), set())
        mask = np.isin(seg_2d, list(include_ids)).astype(np.uint8)
        if not mask.any():
            continue
        # RETR_CCOMP keeps hole boundaries too, so a ring-shaped mask (e.g. the
        # myocardium wall around the blood pool) renders both its outer
        # (epicardial) and inner (endocardial) contour, not just the outer one.
        found, _ = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in found:
            if len(contour) < 3:
                continue
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = [(float(point[0][1]), float(point[0][0])) for point in approx]
            if len(points) >= 3:
                contours.append({"class_id": int(class_id), "points": points})
    return contours
