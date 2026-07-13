from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QSizePolicy, QVBoxLayout

from utils.hanging_protocols import PRESET_BUTTON_ORDER, PRESETS


class SidebarWidget(QFrame):
    load_patient_requested = Signal()
    predict_requested = Signal()
    predict_lge_requested = Signal()
    save_requested = Signal()
    preset_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self._build()
        self._style()

    def _build(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(6)

        self.title_label = QLabel("Home")
        self.title_label.setProperty("role", "sidebarSectionTitle")
        self.layout.addWidget(self.title_label)

        # Load patient button - cardisort classifies and loads every recognized series
        self.btn_load_patient = QPushButton("Load Patient")
        self.btn_load_patient.setProperty("role", "sidebarButton")
        self.btn_load_patient.clicked.connect(self.load_patient_requested.emit)
        self.layout.addWidget(self.btn_load_patient)

        # These buttons will be created and shown after patient loading
        self.btn_predict = None
        self.btn_save = None

        # Predict segmentation for currently loaded LGE stack
        self.btn_predict_lge = QPushButton("  Predict LGE")
        self.btn_predict_lge.setProperty("role", "sidebarSubButton")
        self.btn_predict_lge.clicked.connect(self.predict_lge_requested.emit)
        self.layout.addWidget(self.btn_predict_lge)

        presets_label = QLabel("LAYOUTS")
        presets_label.setProperty("role", "sidebarSectionTitle")
        self.layout.addWidget(presets_label)

        for preset_name in PRESET_BUTTON_ORDER:
            label = PRESETS[preset_name].label if preset_name in PRESETS else preset_name.title()
            btn = QPushButton(label)
            btn.setProperty("role", "sidebarButton")
            btn.clicked.connect(lambda _, name=preset_name: self.preset_requested.emit(name))
            self.layout.addWidget(btn)

        self.layout.addStretch(1)

    def enable_post_load(self):
        """Show sub-buttons after patient data is loaded."""
        if self.btn_predict is not None:
            return  # Already created

        # Find position to insert (before Predict LGE)
        predict_lge_index = self.layout.indexOf(self.btn_predict_lge)

        # Create and insert Predict labels button (sub-button style)
        self.btn_predict = QPushButton("  Predict labels")
        self.btn_predict.setProperty("role", "sidebarSubButton")
        self.btn_predict.clicked.connect(self.predict_requested.emit)
        self.layout.insertWidget(predict_lge_index, self.btn_predict)

        # Create and insert Save button (sub-button style)
        self.btn_save = QPushButton("  Save new labels")
        self.btn_save.setProperty("role", "sidebarSubButton")
        self.btn_save.clicked.connect(self.save_requested.emit)
        self.layout.insertWidget(predict_lge_index + 1, self.btn_save)

    def _style(self):
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setFixedWidth(210)
        self.setProperty("role", "sidebarPanel")
