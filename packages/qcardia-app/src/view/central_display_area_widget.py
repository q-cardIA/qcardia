from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from view.custom_graphics_view import CustomQGraphicsView
from view.series_browser_widget import SERIES_KEY_MIME_TYPE
from view.volume_geometry import VolumeGeometry


@dataclass
class _Panel:
    container: QWidget
    viewer: CustomQGraphicsView
    slider: QSlider
    title_label: QLabel
    slice_label: QLabel


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
        self._volumes: dict[str, np.ndarray] = {}       # (T,Z,H,W) or (Z,H,W) or (T,H,W)
        self._is_temporal: dict[str, bool] = {}
        self._geometries: dict[str, VolumeGeometry] = {}
        self._segmentations: dict[str, np.ndarray] = {}

        # Per-panel state — grows/shrinks with the grid
        self._panel_series_key: list[str | None] = []
        self._panel_axis: list[str] = []
        self._panel_slice_idx: list[int] = []

        # Shared world-space crosshair; None until the user first clicks
        self._world_crosshair: np.ndarray | None = None

        self.current_time_index: int = 0

        # Live panel widgets (rebuilt by set_grid_size)
        self._panels: list[_Panel] = []
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
        return self._grid_rows * self._grid_cols

    @property
    def grid_size(self) -> tuple[int, int]:
        return self._grid_rows, self._grid_cols

    def panel_assignments(self) -> list[dict]:
        """Per-panel state (series key, axis, whether a segmentation exists) —
        used by the chat assistant to describe the current layout."""
        return [
            {
                "idx": i,
                "series_key": self._panel_series_key[i],
                "axis": self._panel_axis[i],
                "has_segmentation": self._panel_series_key[i] in self._segmentations,
            }
            for i in range(self.n_panels)
        ]

    def set_grid_size(self, rows: int, cols: int):
        if rows == self._grid_rows and cols == self._grid_cols:
            return

        # Preserve series assignments for panels that survive the resize
        old_keys = list(self._panel_series_key)
        old_axes = list(self._panel_axis)
        old_slices = list(self._panel_slice_idx)
        old_rows, old_cols = self._grid_rows, self._grid_cols

        # Tear down existing layout and panels
        while self._layout.count():
            self._layout.takeAt(0)
        for p in self._panels:
            p.container.setParent(None)
            p.container.deleteLater()
        self._panels.clear()

        self._grid_rows = rows
        self._grid_cols = cols
        n = rows * cols

        # Extend or truncate per-panel state lists to match new count
        self._panel_series_key = (old_keys + [None] * n)[:n]
        self._panel_axis       = (old_axes + ["transverse"] * n)[:n]
        self._panel_slice_idx  = (old_slices + [0] * n)[:n]

        # Build fresh panels
        for idx in range(n):
            p = self._make_panel(idx)
            self._panels.append(p)

        # Place panels and dividers in the grid
        for idx, p in enumerate(self._panels):
            r, c = divmod(idx, cols)
            self._layout.addWidget(p.container, 2 * r, 2 * c)

        for r in range(rows - 1):
            div = _make_divider(horizontal=True)
            self._layout.addWidget(div, 2 * r + 1, 0, 1, 2 * cols - 1)

        for c in range(cols - 1):
            div = _make_divider(horizontal=False)
            self._layout.addWidget(div, 0, 2 * c + 1, 2 * rows - 1, 1)

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
        for i in range(n):
            if self._panel_series_key[i] is not None:
                self._update_panel(i)

    def _make_panel(self, idx: int) -> _Panel:
        container = _PanelDropContainer(idx)
        container.series_dropped.connect(self._on_panel_series_dropped)
        container.setProperty("role", "viewerCard")
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(2)

        hdr = QHBoxLayout()
        hdr.setContentsMargins(6, 4, 6, 4)
        hdr.setSpacing(8)
        title = QLabel("Empty")
        title.setProperty("role", "viewerTitle")
        slice_lbl = QLabel("")
        slice_lbl.setProperty("role", "viewerMeta")
        hdr.addWidget(title)
        hdr.addStretch(1)
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

        vbox.addLayout(hdr)
        vbox.addWidget(slider)
        vbox.addWidget(viewer)

        return _Panel(
            container=container,
            viewer=viewer,
            slider=slider,
            title_label=title,
            slice_label=slice_lbl,
        )

    # ── External API ──────────────────────────────────────────────────────

    def register_series(
        self,
        series_key: str,
        volume: np.ndarray,
        is_temporal: bool,
        geometry: VolumeGeometry | None = None,
    ):
        """Register a loaded series volume (and optional DICOM geometry)."""
        self._volumes[series_key] = volume
        self._is_temporal[series_key] = is_temporal
        if geometry is not None:
            self._geometries[series_key] = geometry

    def set_geometry(self, series_key: str, geom: VolumeGeometry):
        self._geometries[series_key] = geom

    def is_series_registered(self, series_key: str) -> bool:
        return series_key in self._volumes

    def set_panel_series(
        self, panel_idx: int, series_key: str, axis: str = "transverse"
    ):
        """Assign a registered series to a panel slot."""
        if panel_idx < 0 or panel_idx >= self.n_panels:
            return
        self._panel_series_key[panel_idx] = series_key
        self._panel_axis[panel_idx] = axis.lower()

        n = self._n_slices(series_key, axis)
        mid = n // 2 if n > 0 else 0
        self._panel_slice_idx[panel_idx] = mid

        sl = self._panels[panel_idx].slider
        sl.blockSignals(True)
        sl.setMaximum(max(0, n - 1))
        sl.setValue(mid)
        sl.setEnabled(n > 1)
        sl.setVisible(n > 1)
        sl.blockSignals(False)

        self._panels[panel_idx].title_label.setText(f"{series_key} / {axis}")
        self._update_panel(panel_idx)

    def set_panel_empty(self, panel_idx: int):
        if panel_idx < 0 or panel_idx >= self.n_panels:
            return
        self._panel_series_key[panel_idx] = None
        p = self._panels[panel_idx]
        p.title_label.setText("Empty")
        p.slice_label.setText("")
        p.slider.setEnabled(False)
        p.slider.setVisible(False)
        p.viewer.clear_display()

    def get_thumbnail(self, series_key: str) -> np.ndarray | None:
        """Representative 2D uint8 slice (center transverse slice, t=0) for a browser card."""
        if series_key not in self._volumes:
            return None
        n = self._n_slices(series_key, "transverse")
        mid = n // 2 if n > 0 else 0
        return self._get_slice_2d(series_key, "transverse", mid, t=0)

    def _on_panel_series_dropped(self, panel_idx: int, series_key: str):
        if self.is_series_registered(series_key):
            self.set_panel_series(panel_idx, series_key)

    def refresh_series(self, series_key: str):
        """Re-render any panel currently showing series_key against its (possibly new) volume."""
        for i in range(self.n_panels):
            if self._panel_series_key[i] == series_key:
                self.set_panel_series(i, series_key, axis=self._panel_axis[i])

    def set_segmentation(self, series_key: str, seg_volume: np.ndarray):
        self._segmentations[series_key] = seg_volume
        for i in range(self.n_panels):
            if self._panel_series_key[i] == series_key:
                self._update_panel(i)

    def set_time_index(self, t: int):
        self.current_time_index = t
        for i in range(self.n_panels):
            key = self._panel_series_key[i]
            if key and self._is_temporal.get(key, False):
                self._update_panel(i)

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

    def _get_contours(
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

    def _update_panel(self, panel_idx: int):
        key = self._panel_series_key[panel_idx]
        if key is None:
            return
        p = self._panels[panel_idx]
        axis = self._panel_axis[panel_idx]
        z = self._panel_slice_idx[panel_idx]
        t = self.current_time_index

        slice_2d = self._get_slice_2d(key, axis, z, t)
        if slice_2d is None:
            return

        geom = self._geometries.get(key)
        p.viewer.set_pixel_spacing(geom.pixel_spacing if geom is not None else (1.0, 1.0))
        p.viewer.set_contour_line_mm(2.5 if key.startswith("LGE") else 2.0)
        p.viewer.update_contour_paths(self._get_contours(key, axis, z, t))
        p.viewer.update_slice_view(slice_2d)

        n = self._n_slices(key, axis)
        p.slice_label.setText(f"Slice: {z + 1}/{n}" if n > 1 else "")

    def update_all_panels(self):
        for i in range(self.n_panels):
            self._update_panel(i)

    # ── World-coordinate crosshair sync ──────────────────────────────────

    def _on_panel_coordinates_changed(self, panel_idx: int, sig_x: int, sig_y: int):
        # Signal emits (col, row) as (x, y)
        col, row = sig_x, sig_y
        key = self._panel_series_key[panel_idx]
        if key is None:
            return
        geom = self._geometries.get(key)
        if geom is not None:
            z = self._panel_slice_idx[panel_idx]
            self._world_crosshair = geom.pixel_to_world(row, col, z)
            self._broadcast_world_crosshair()
        else:
            self._panels[panel_idx].viewer.set_crosshair_from_volume_indices(row, col)

    def _on_panel_slider_changed(self, panel_idx: int, value: int):
        key = self._panel_series_key[panel_idx]
        if key is None:
            return
        n = self._n_slices(key, self._panel_axis[panel_idx])
        new_z = min(max(0, value), max(0, n - 1))
        self._panel_slice_idx[panel_idx] = new_z

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
            self._broadcast_world_crosshair()
        else:
            self._update_panel(panel_idx)

    def _broadcast_world_crosshair(self):
        """Project _world_crosshair into every panel, updating slice index and crosshair."""
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
            if n > 0:
                new_z = max(0, min(int(round(new_z)), n - 1))
                self._panel_slice_idx[i] = new_z
                sl = self._panels[i].slider
                sl.blockSignals(True)
                sl.setValue(new_z)
                sl.blockSignals(False)
            # _update_panel calls update_slice_view first (which preserves crosshair_pos),
            # then we overwrite with the projected position.
            self._update_panel(i)
            self._panels[i].viewer.set_crosshair_from_volume_indices(row, col)

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
