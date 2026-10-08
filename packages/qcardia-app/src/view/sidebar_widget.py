from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QSizePolicy, QVBoxLayout

from utils.app_config import patient_label
from utils.hanging_protocols import PRESET_BUTTON_ORDER, preset_label
from view import theme


class SidebarWidget(QFrame):
    load_patient_requested = Signal()
    save_requested = Signal()
    preset_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self._build()
        self._style()

    def _build(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 0, 8, 8)
        self.layout.setSpacing(2)

        # Load patient button - cardisort classifies and loads every recognized series
        self.btn_load_patient = QPushButton("Load Patient…")
        self.btn_load_patient.setProperty("role", "sidebarButton")
        self.btn_load_patient.clicked.connect(self.load_patient_requested.emit)
        self.layout.addWidget(self.btn_load_patient)
        self._patient_label = QLabel("No patient loaded")
        self._patient_label.setProperty("role", "sidebarStatus")
        self.layout.addWidget(self._patient_label)

        self._section("Labels")
        # The window gives this button its menu: one entry per view that has a segmentation model.
        self.btn_predict = QPushButton("Predict Labels")
        self.btn_predict.setProperty("role", "sidebarButton")
        self.layout.addWidget(self.btn_predict)
        # Checkable; the window keeps it in step with Tools > Edit Contours.
        self.btn_edit_contours = QPushButton("Edit Contours")
        self.btn_edit_contours.setProperty("role", "sidebarButton")
        self.btn_edit_contours.setCheckable(True)
        self.layout.addWidget(self.btn_edit_contours)
        self.btn_save = QPushButton("Save New Labels")
        self.btn_save.setProperty("role", "sidebarButton")
        self.btn_save.clicked.connect(self.save_requested.emit)
        self.layout.addWidget(self.btn_save)

        self._section("Layouts")
        for preset_name in PRESET_BUTTON_ORDER:
            btn = QPushButton(preset_label(preset_name))
            btn.setProperty("role", "sidebarButton")
            btn.clicked.connect(lambda _, name=preset_name: self.preset_requested.emit(name))
            self.layout.addWidget(btn)

        self.layout.addStretch(1)

    def _section(self, title: str):
        label = QLabel(title)
        label.setProperty("role", "sidebarSectionTitle")
        self.layout.addSpacing(10)
        self.layout.addWidget(label)

    def show_patient(self, folder: Path, n_series: int):
        name = patient_label(folder)
        self._patient_label.setText(f"{name} · {n_series} series")
        self._patient_label.setToolTip(f"{name}\nStudy folder {folder}")

    def _style(self):
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setMinimumWidth(theme.scaled(210))
