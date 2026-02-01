from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class NavigationPanelWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("navigationPanel")
        self._build()
        self._style()

    def _build(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(10)

        self.title = QLabel("Analysis")
        self.title.setObjectName("analysisTitle")
        self.layout.addWidget(self.title)

        self.baseplan_combo = QComboBox()
        self.baseplan_combo.addItems(
            ["Global function", "Regional wall motion", "LGE"]
        )  # placeholder entries
        self.layout.addWidget(self.baseplan_combo)

        sliders_row = QHBoxLayout()
        self.slider_z = self._vertical_slider()
        self.slider_y = self._vertical_slider()
        self.slider_x = self._vertical_slider()
        sliders_row.addWidget(self.slider_z)
        sliders_row.addWidget(self.slider_y)
        sliders_row.addWidget(self.slider_x)
        self.layout.addLayout(sliders_row)

        self.display_combo = QComboBox()
        self.display_combo.addItems(["Display helplines: On", "Display helplines: Off"])
        self.layout.addWidget(self.display_combo)

        self.labels_combo = QComboBox()
        self.labels_combo.addItems(["Ground Truth Labels", "Predicted Labels"])
        self.layout.addWidget(self.labels_combo)

        self.labels_table = QTableWidget(0, 3)
        self.labels_table.setHorizontalHeaderLabels(["Color", "Structure", "Toggle"])
        self.labels_table.verticalHeader().setVisible(False)
        self.labels_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.labels_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._populate_labels()
        self.layout.addWidget(self.labels_table)

        self.dose_combo = QComboBox()
        self.dose_combo.addItems(["Ground Truth Dose", "Predicted Dose"])
        self.layout.addWidget(self.dose_combo)

        self.dose_table = QTableWidget(0, 3)
        self.dose_table.setHorizontalHeaderLabels(["Color", "Structure", "Toggle"])
        self.dose_table.verticalHeader().setVisible(False)
        self.dose_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.dose_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._populate_dose()
        self.layout.addWidget(self.dose_table)

        self.layout.addStretch(1)

    def _vertical_slider(self):
        s = QSlider(Qt.Vertical)
        s.setMinimum(0)
        s.setMaximum(100)
        s.setValue(50)
        s.setFixedHeight(140)
        return s

    def _populate_labels(self):
        data = [
            ("#D6D1AB", "RING"),
            ("#444444", "Skin"),
            ("#005BFF", "CTV"),
            ("#8B5A2B", "RECTUM"),
            ("#FFFF00", "BLADDER"),
            ("#00FF00", "FEMORALHEADS"),
        ]
        for color, structure in data:
            row = self.labels_table.rowCount()
            self.labels_table.insertRow(row)
            color_item = QTableWidgetItem()
            color_item.setBackground(
                QtGui.QColor(color) if False else Qt.black
            )  # placeholder color logic
            color_item.setText("")
            self.labels_table.setItem(row, 0, color_item)
            self.labels_table.setItem(row, 1, QTableWidgetItem(structure))
            cb = QCheckBox()
            cb.setChecked(True)
            self.labels_table.setCellWidget(row, 2, cb)
        self.labels_table.resizeColumnsToContents()

    def _populate_dose(self):
        data = [
            ("#FF0000", "102"),
            ("#FF4000", "100"),
            ("#FF8000", "98"),
            ("#FFC000", "95"),
            ("#FFFF00", "90"),
            ("#00FF00", "80"),
        ]
        for color, structure in data:
            row = self.dose_table.rowCount()
            self.dose_table.insertRow(row)
            color_item = QTableWidgetItem()
            color_item.setText("")
            self.dose_table.setItem(row, 0, color_item)
            self.dose_table.setItem(row, 1, QTableWidgetItem(structure))
            cb = QCheckBox()
            cb.setChecked(False)
            self.dose_table.setCellWidget(row, 2, cb)
        self.dose_table.resizeColumnsToContents()

    def _style(self):
        self.setFixedWidth(300)
        self.setStyleSheet(
            """
            #navigationPanel {background-color:#22262E; border:1px solid #2F3440;}
            #navTitle {color:#E6E6E6; font-size:16px; font-weight:bold;}
            QComboBox {background-color:#2F3440; color:#E6E6E6; border:1px solid #3A404C; padding:4px;}
            QSlider::groove:vertical {background:#2F3440; width:6px; border-radius:3px;}
            QSlider::handle:vertical {background:#85aaff; height:14px; margin:0 -4px; border-radius:7px;}
            QTableWidget {background-color:#2A2E37; color:#E6E6E6; gridline-color:#3A404C;}
            QHeaderView::section {background-color:#2F3440; color:#E6E6E6; border:0px; padding:4px; font-weight:bold;}
            """
        )
