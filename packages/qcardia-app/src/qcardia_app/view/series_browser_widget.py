from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QMimeData, QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QDrag, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from qcardia_app.view import theme

SERIES_KEY_MIME_TYPE = "application/x-qcardia-series-key"

_GROUP_ORDER = ["CINE", "LGE", "PERF"]
_N_COLS = 2
_GAP = 2  # px between thumbnails

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
    loading: bool = False  # its series is still being loaded


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


class _ThumbnailCard(QWidget):
    """Painted, so resizing the panel only repaints: the image is cropped to a square once."""

    reclassify_requested = Signal(object, str)  # (SeriesCardInfo, new_key)

    def __init__(self, info: SeriesCardInfo, parent=None):
        super().__init__(parent)
        self.info = info
        self._draggable = info.draggable
        self.series_key = info.series_key
        self.setProperty("role", "seriesCard")
        self.setAttribute(Qt.WA_Hover)
        self.setToolTip(info.label)
        self._pixmap: QPixmap | None = None
        self._scaled: QPixmap | None = None  # _pixmap at the card's size
        if info.draggable:
            self.setCursor(Qt.OpenHandCursor)
        self._drag_start = QPoint()

    def set_thumbnail(self, arr: np.ndarray | None):
        if arr is None:
            return
        pix = _to_pixmap(arr)
        side = min(pix.width(), pix.height())
        self._pixmap = pix.copy((pix.width() - side) // 2, (pix.height() - side) // 2, side, side)
        self._scaled = None

    def set_loaded(self, series_key: str):
        self.info.loading, self.info.draggable = False, True
        self.series_key = self.info.series_key = series_key
        self._draggable = True
        self.setCursor(Qt.OpenHandCursor)

    def set_side(self, side: int):
        if self.width() != side or self.height() != side:
            self.setFixedSize(side, side)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect())
        clip = QPainterPath()
        clip.addRoundedRect(rect, theme.RADIUS, theme.RADIUS)
        painter.setClipPath(clip)
        painter.setOpacity(1.0 if self._draggable else 0.45)
        if self._pixmap is not None:
            painter.drawPixmap(0, 0, self._scaled_pixmap())
        else:
            painter.fillRect(self.rect(), theme.color("bg2"))
        painter.setOpacity(1.0)
        metrics = self.fontMetrics()
        if self.info.loading:
            painter.setPen(theme.color("text_faint"))
            painter.drawText(self.rect().adjusted(0, 0, 0, -metrics.height()), Qt.AlignCenter, "Loading…")
        banner = self.rect().adjusted(0, self.height() - metrics.height() - 4, 0, 0)
        painter.fillRect(banner, theme.color("banner"))
        painter.setPen(theme.color("text" if self._draggable else "text_muted"))
        painter.drawText(banner.adjusted(4, 0, -4, 0), Qt.AlignCenter,
                         metrics.elidedText(self.info.label, Qt.ElideRight, banner.width() - 8))
        if self._draggable and self.underMouse():
            painter.setClipping(False)
            painter.setPen(QPen(theme.color("accent"), 1.5))
            painter.drawRoundedRect(rect.adjusted(0.75, 0.75, -0.75, -0.75), theme.RADIUS, theme.RADIUS)

    def _scaled_pixmap(self) -> QPixmap:
        ratio = self.devicePixelRatioF()
        width, height = round(self.width() * ratio), round(self.height() * ratio)
        if self._scaled is None or (self._scaled.width(), self._scaled.height()) != (width, height):
            self._scaled = self._pixmap.scaled(width, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self._scaled.setDevicePixelRatio(ratio)
        return self._scaled

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
        if self._pixmap is not None:
            drag.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        drag.exec(Qt.CopyAction)


class SeriesBrowserWidget(QFrame):
    """Right-side panel of series thumbnails, grouped by modality. Loaded series are
    draggable onto a viewer panel; everything else is a view-only card."""

    reclassify_requested = Signal(object, str)  # (SeriesCardInfo, new_key)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("seriesBrowser")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setMinimumWidth(theme.scaled(160))
        self._cards: list[_ThumbnailCard] = []
        self._collapsed: set[str] = set()  # group names, kept across a reload

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        self._load_bar = theme.thin_loading_bar()
        outer.addWidget(self._load_bar)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.installEventFilter(self)
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(2)
        self._content_layout.addStretch(1)
        self._scroll.setWidget(self._content)
        outer.addWidget(self._scroll, 1)

    def populate(self, cards: list[SeriesCardInfo]):
        """Rebuild all cards, grouped by modality; Unclassified always sorts last."""
        while self._content_layout.count() > 1:
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards = []

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
            container = QWidget()
            grid = QGridLayout(container)
            grid.setContentsMargins(0, 0, 0, 4)
            grid.setSpacing(_GAP)
            grid.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            sorted_cards = sorted(groups[group], key=lambda c: (not (c.draggable or c.loading), c.label))
            for j, info in enumerate(sorted_cards):
                card = _ThumbnailCard(info)
                card.set_thumbnail(info.thumbnail)
                card.reclassify_requested.connect(self.reclassify_requested)
                grid.addWidget(card, *divmod(j, _N_COLS))
                self._cards.append(card)

            header = QPushButton(f"{group}  {len(sorted_cards)}")
            header.setProperty("role", "seriesGroupHeader")
            header.setCheckable(True)
            header.setChecked(group not in self._collapsed)
            header.setCursor(Qt.PointingHandCursor)
            header.toggled.connect(lambda shown, g=group, c=container, h=header: self._fold(g, c, h, shown))
            self._fold(group, container, header, header.isChecked())
            self._content_layout.insertWidget(insert_pos, header)
            self._content_layout.insertWidget(insert_pos + 1, container)
            insert_pos += 2
        self._fit_cards()

    def update_card(self, path: Path, thumbnail: np.ndarray | None = None, series_key: str | None = None):
        """Fill in a card while the patient loads: its thumbnail, and its series once loaded.
        With series_key, only that series' card: a flow velocity and magnitude share a folder."""
        for card in self._cards:
            if card.info.path == path and series_key in (None, card.info.label):
                if series_key is not None:
                    card.set_loaded(series_key)
                card.set_thumbnail(thumbnail)
                card.update()

    def set_load_progress(self, loaded: int, total: int):
        """The bar above the cards: how many of the patient's series have loaded."""
        self._load_bar.setRange(0, total)
        self._load_bar.setValue(loaded)
        self._load_bar.setToolTip(f"{loaded} of {total} series loaded")
        self._load_bar.setVisible(loaded < total)

    def stop_loading(self):
        """Cards whose series failed to load stop saying they load."""
        self._load_bar.hide()
        for card in self._cards:
            if card.info.loading:
                card.info.loading = False
                card.update()

    def _fold(self, group: str, container: QWidget, header: QPushButton, shown: bool):
        container.setVisible(shown)
        header.setIcon(theme.line_icon("chevron_down" if shown else "chevron_right", "text_faint"))
        (self._collapsed.discard if shown else self._collapsed.add)(group)

    def _fit_cards(self):
        # Room for the scroll bar is always kept: sizing from the viewport would let the bar and the
        # card size keep changing each other at some panel heights.
        width = self._scroll.width() - self._scroll.verticalScrollBar().sizeHint().width()
        side = max(40, (width - _GAP * (_N_COLS - 1)) // _N_COLS)
        for card in self._cards:
            card.set_side(side)

    def eventFilter(self, obj, event):
        if obj is self._scroll and event.type() == QEvent.Resize:
            self._fit_cards()
        return super().eventFilter(obj, event)
