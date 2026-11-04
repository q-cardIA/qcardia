from pathlib import Path

from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QSizePolicy, QVBoxLayout

from qcardia_app.utils.app_config import patient_of
from qcardia_app.utils.hanging_protocols import SIDEBAR_PRESETS
from qcardia_app.view import theme


class SidebarWidget(QFrame):
    load_patient_requested = Signal()
    preset_requested = Signal(str)
    _ICONS = {"functional": "heart", "lge": "lge", "perfusion": "uptake", "flow": "flow"}  # by preset

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self._build()
        self._style()

    def _build(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 0, 8, 8)
        self.layout.setSpacing(2)

        # Patient Data opens to the loaded patient and a button to load another (cardisort classifies and
        # loads every recognized series of a patient folder); pressed again, it closes.
        self.btn_patient = QPushButton(" Patient Data")  # the space: a wider gap after the icon
        self.btn_patient.setProperty("role", "sidebarButton")
        self.btn_patient.setCheckable(True)
        self.layout.addWidget(self.btn_patient)
        patient_box = QFrame()
        box_layout = QVBoxLayout(patient_box)
        box_layout.setContentsMargins(26, 0, 0, 4)  # under the button's text
        box_layout.setSpacing(2)
        self._patient_label = QLabel("No patient loaded")
        self._patient_label.setProperty("role", "sidebarStatus")
        box_layout.addWidget(self._patient_label)
        self.btn_load_patient = QPushButton("Load a patient")
        self.btn_load_patient.setProperty("role", "sidebarButton")
        self.btn_load_patient.clicked.connect(self.load_patient_requested.emit)
        box_layout.addWidget(self.btn_load_patient)
        patient_box.hide()
        self.btn_patient.toggled.connect(patient_box.setVisible)
        self.layout.addWidget(patient_box)

        # The presets a user picks directly; greyed out until a patient has their series (set_available).
        self._preset_buttons = {}
        for text, preset_name in SIDEBAR_PRESETS.items():
            btn = self._preset_buttons[preset_name] = QPushButton(f" {text}")
            btn.setProperty("role", "sidebarButton")
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, name=preset_name: self.preset_requested.emit(name))
            self.layout.addWidget(btn)

        self.layout.addStretch(1)
        self._apply_icons()
        theme.notifier.changed.connect(self._apply_icons)

    def _apply_icons(self):
        for btn, kind in ((self.btn_patient, "folder"), *((b, self._ICONS[n]) for n, b in self._preset_buttons.items())):
            btn.setIcon(theme.line_icon(kind, "text_muted"))
            btn.setIconSize(QSize(18, 18))

    def set_available(self, presets: list[str]):
        """Only the presets the loaded patient has the series for can be picked."""
        for name, btn in self._preset_buttons.items():
            btn.setEnabled(name in presets)

    def show_patient(self, folder: Path, n_series: int):
        p = patient_of(folder)
        self._patient_label.setText("\n".join([*((p.name, f"ID {p.hospital_id}") if p else (folder.name,)),
                                               f"{n_series} series"]))
        self._patient_label.setToolTip(f"Study folder {folder}")
        self.btn_load_patient.setText("Load another patient")

    def _style(self):
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
