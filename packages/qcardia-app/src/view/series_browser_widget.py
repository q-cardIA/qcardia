from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QMimeData, QPoint, Qt, Signal
from PySide6.QtGui import QDrag, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

SERIES_KEY_MIME_TYPE = "application/x-qcardia-series-key"

_GROUP_ORDER = ["CINE", "LGE", "PERF"]
_N_COLS = 2
_CARD_SIZE = 84

_REASSIGN_KEYS = [
    "CINE_SAX", "CINE_2CH", "CINE_3CH", "CINE_4CH",
    "LGE_SAX", "LGE_2CH", "LGE_3CH", "LGE_4CH",
]


@dataclass
class SeriesCardInfo:
    label: str
    group: str
    thumbnail: np.ndarray | None
    draggable: bool
    path: Path
    series_key: str | None = None


def _to_pixmap(arr: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape
    rgb = np.stack([arr] * 3, axis=-1)
    qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class _ReassignDialog(QDialog):
    def __init__(self, current_key: str | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reassign Classification")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Assign this series to:"))
        self._combo = QComboBox()
        self._combo.addItems(_REASSIGN_KEYS)
        if current_key in _REASSIGN_KEYS:
            self._combo.setCurrentText(current_key)
        layout.addWidget(self._combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_key(self) -> str:
        return self._combo.currentText()


class _ThumbnailCard(QFrame):
    reclassify_requested = Signal(object, str)  # (SeriesCardInfo, new_key)

    def __init__(self, info: SeriesCardInfo, parent=None):
        super().__init__(parent)
        self.info = info
        self._draggable = info.draggable
        self.series_key = info.series_key
        self.setProperty("role", "seriesCard" if info.draggable else "seriesCardMuted")
        self.setFixedSize(_CARD_SIZE, _CARD_SIZE)
        if info.draggable:
            self.setCursor(Qt.OpenHandCursor)
        self._drag_start = QPoint()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        self._thumb_lbl = QLabel()
        self._thumb_lbl.setAlignment(Qt.AlignCenter)
        self._key_lbl = QLabel(info.label)
        self._key_lbl.setProperty(
            "role", "seriesCardLabel" if info.draggable else "seriesCardLabelMuted"
        )
        self._key_lbl.setAlignment(Qt.AlignCenter)
        self._key_lbl.setWordWrap(True)
        layout.addWidget(self._thumb_lbl, 1)
        layout.addWidget(self._key_lbl)

    def set_thumbnail(self, arr: np.ndarray | None):
        if arr is None:
            return
        pix = _to_pixmap(arr)
        side = _CARD_SIZE - 12
        scaled = pix.scaled(side, side, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self._thumb_lbl.setPixmap(scaled)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        act = menu.addAction("Reassign classification…")
        if menu.exec(event.globalPos()) == act:
            dlg = _ReassignDialog(self.info.series_key, self)
            if dlg.exec() == QDialog.Accepted:
                self.reclassify_requested.emit(self.info, dlg.selected_key())

    def mousePressEvent(self, event):
        if not self._draggable:
            return
        if event.button() == Qt.LeftButton:
            self._drag_start = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._draggable:
            return
        if not (event.buttons() & Qt.LeftButton):
            return
        if (event.pos() - self._drag_start).manhattanLength() < QApplication.startDragDistance():
            return
        mime = QMimeData()
        mime.setData(SERIES_KEY_MIME_TYPE, self.series_key.encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        if self._thumb_lbl.pixmap():
            drag.setPixmap(self._thumb_lbl.pixmap())
        drag.exec(Qt.CopyAction)


class SeriesBrowserWidget(QFrame):
    """Right-side panel of series thumbnails, grouped by modality. Loaded series are
    draggable onto a viewer panel; everything else is a view-only card."""

    reclassify_requested = Signal(object, str)  # (SeriesCardInfo, new_key)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("seriesBrowser")
        self.setProperty("role", "seriesBrowserPanel")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setFixedWidth(200)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        title = QLabel("Series")
        title.setProperty("role", "sidebarSectionTitle")
        outer.addWidget(title)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(8)
        self._content_layout.addStretch(1)
        self._scroll.setWidget(self._content)
        outer.addWidget(self._scroll, 1)

    def populate(self, cards: list[SeriesCardInfo]):
        """Rebuild all cards, grouped by modality; Unclassified always sorts last."""
        while self._content_layout.count() > 1:
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        groups: dict[str, list[SeriesCardInfo]] = {}
        for card in cards:
            groups.setdefault(card.group, []).append(card)
        unclassified = groups.pop("Unclassified", None)

        ordered = [g for g in _GROUP_ORDER if g in groups]
        ordered += sorted(g for g in groups if g not in ordered)
        if unclassified is not None:
            groups["Unclassified"] = unclassified
            ordered.append("Unclassified")

        insert_pos = self._content_layout.count() - 1
        for group in ordered:
            hdr = QLabel(group)
            hdr.setProperty("role", "sidebarSectionTitle")
            self._content_layout.insertWidget(insert_pos, hdr)
            insert_pos += 1

            container = QWidget()
            grid = QGridLayout(container)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(6)
            sorted_cards = sorted(groups[group], key=lambda c: (not c.draggable, c.label))
            for j, info in enumerate(sorted_cards):
                card = _ThumbnailCard(info)
                card.set_thumbnail(info.thumbnail)
                card.reclassify_requested.connect(self.reclassify_requested)
                grid.addWidget(card, *divmod(j, _N_COLS))
            self._content_layout.insertWidget(insert_pos, container)
            insert_pos += 1
