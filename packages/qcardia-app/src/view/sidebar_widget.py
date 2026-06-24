from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QSizePolicy, QVBoxLayout


class SidebarWidget(QFrame):
    load_requested = Signal()
    load_image_requested = Signal()
    predict_requested = Signal()
    predict_lge_requested = Signal()
    save_requested = Signal()

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

        # Load patient button (Cine SAX)
        self.btn_load = QPushButton("Cine SAX")
        self.btn_load.setProperty("role", "sidebarButton")
        self.btn_load.clicked.connect(self._on_load_patient_clicked)
        self.layout.addWidget(self.btn_load)

        # These buttons will be created and shown after patient loading
        self.btn_predict = None
        self.btn_save = None
        self.submenu_visible = False

        # Load Image button - always visible (LGE)
        self.btn_load_image = QPushButton("LGE")
        self.btn_load_image.setProperty("role", "sidebarButton")
        self.btn_load_image.clicked.connect(self.load_image_requested.emit)
        self.layout.addWidget(self.btn_load_image)

        # Predict segmentation for currently loaded LGE stack
        self.btn_predict_lge = QPushButton("  Predict LGE")
        self.btn_predict_lge.setProperty("role", "sidebarSubButton")
        self.btn_predict_lge.clicked.connect(self.predict_lge_requested.emit)
        self.layout.addWidget(self.btn_predict_lge)

        self.layout.addStretch(1)

    def _on_load_patient_clicked(self):
        """Handle Load patient button click - load or toggle submenu."""
        # If submenu exists, just toggle it without loading
        if self.btn_predict is not None:
            self._toggle_submenu()
        else:
            # First time - emit signal to load patient
            self.load_requested.emit()

    def _toggle_submenu(self):
        """Toggle visibility of predict and save sub-buttons."""
        self.submenu_visible = not self.submenu_visible
        if self.btn_predict is not None:
            self.btn_predict.setVisible(self.submenu_visible)
        if self.btn_save is not None:
            self.btn_save.setVisible(self.submenu_visible)

    def enable_post_load(self):
        """Show sub-buttons after patient data is loaded."""
        if self.btn_predict is not None:
            return  # Already created

        # Find position to insert (after Load patient, before Load Image)
        load_image_index = self.layout.indexOf(self.btn_load_image)

        # Create and insert Predict labels button (sub-button style)
        self.btn_predict = QPushButton("  Predict labels")
        self.btn_predict.setProperty("role", "sidebarSubButton")
        self.btn_predict.clicked.connect(self.predict_requested.emit)
        self.layout.insertWidget(load_image_index, self.btn_predict)

        # Create and insert Save button (sub-button style)
        self.btn_save = QPushButton("  Save new labels")
        self.btn_save.setProperty("role", "sidebarSubButton")
        self.btn_save.clicked.connect(self.save_requested.emit)
        self.layout.insertWidget(load_image_index + 1, self.btn_save)

        # Make them visible and track state
        self.submenu_visible = True

    def _style(self):
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setFixedWidth(210)
        self.setProperty("role", "sidebarPanel")
