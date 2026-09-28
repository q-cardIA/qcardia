import os
import sys

from PySide6.QtCore import QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from utils.app_config import CHAT_AVAILABLE, data_dir, set_data_dir

class WelcomeWidget(QWidget):

    transition_signal = Signal()

    def __init__(self, parent=None):
        super().__init__()
        self.setObjectName("welcomeRoot")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.parent = parent
        self._build()
        self._size()
        self._connect()

    @property
    def assistant_disabled(self) -> bool:
        return not (CHAT_AVAILABLE and self._toggle_llm_assistant.isChecked())

    @property
    def use_cardisort_cache(self) -> bool:
        return self._toggle_preproc_labels.isChecked()

    @property
    def use_segmentation_cache(self) -> bool:
        return self._toggle_preproc_seg.isChecked()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _asset(self, filename):
        return os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../assets/images",
            filename,
        )

    def _icon_btn(self, text, image_file, role):
        btn = QPushButton(f"  {text}")
        btn.setProperty("role", role)
        pix = QPixmap(self._asset(image_file))
        if not pix.isNull():
            btn.setIcon(QIcon(pix))
            btn.setIconSize(QSize(20, 20))
        return btn

    def _toggle_switch(self, default: bool, enabled: bool = True) -> QPushButton:
        """An ON/OFF pill button standing in for a checkbox — used in the
        options dropdown instead of QCheckBox so all three toggles read as
        one consistent switch style."""
        btn = QPushButton("ON" if default else "OFF")
        btn.setCheckable(True)
        btn.setChecked(default)
        btn.setEnabled(enabled)
        btn.setProperty("role", "welcomeToggleSwitch")
        btn.toggled.connect(lambda checked, b=btn: b.setText("ON" if checked else "OFF"))
        return btn

    @staticmethod
    def _toggle_row(toggle_btn: QPushButton, label_text: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        label = QLabel(label_text)
        label.setProperty("role", "welcomeOptionLabel")
        row.addWidget(toggle_btn)
        row.addWidget(label)
        row.addStretch(1)
        return row

    def _show_data_dir(self):
        self._data_dir_label.setText(f"Data folder: {data_dir().name}")
        self._data_dir_label.setToolTip(str(data_dir()))

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(60, 0, 60, 40)
        root.setSpacing(0)

        # ---- Header (logo + title + subtitle) ----
        root.addStretch(2)

        header = QHBoxLayout()
        header.setSpacing(20)
        header.setContentsMargins(0, 0, 0, 0)

        logo_label = QLabel()
        pix = QPixmap(self._asset("q-cardIA.png")).scaled(
            64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        logo_label.setPixmap(pix)
        logo_label.setFixedSize(64, 64)

        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        title_col.setContentsMargins(0, 0, 0, 0)

        self.top_logo_panel_label = QLabel("q-cardIA")
        self.top_logo_panel_label.setProperty("role", "welcomeAppName")

        subtitle = QLabel("Quantitative cardiac image analysis")
        subtitle.setProperty("role", "welcomeHeadline")

        title_col.addWidget(self.top_logo_panel_label)
        title_col.addWidget(subtitle)

        header.addStretch(1)
        header.addWidget(logo_label, 0, Qt.AlignVCenter)
        header.addLayout(title_col)
        header.addStretch(1)

        root.addLayout(header)
        root.addSpacing(36)

        # ---- Card (left + right panels) ----
        self.central_label = QWidget()
        self.central_label.setObjectName("centralContainer")
        self.central_label.setAttribute(Qt.WA_StyledBackground, True)
        self.central_label.setProperty("role", "welcomeCentral")

        card_layout = QHBoxLayout(self.central_label)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        self.left_panel = self._build_left_panel()
        self.right_panel = self._build_right_panel()

        divider = QLabel()
        divider.setFixedWidth(1)
        divider.setProperty("role", "divider")
        divider.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        card_layout.addWidget(self.left_panel)
        card_layout.addWidget(divider)
        card_layout.addWidget(self.right_panel)

        card_row = QHBoxLayout()
        card_row.addStretch(1)
        card_row.addWidget(self.central_label, 4)
        card_row.addStretch(1)
        root.addLayout(card_row)

        root.addStretch(3)

    def _build_left_panel(self):
        panel = QWidget()
        panel.setAttribute(Qt.WA_StyledBackground, True)
        panel.setProperty("role", "welcomeSidePanel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(12)

        title = QLabel("Get started")
        title.setProperty("role", "welcomeSectionTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addSpacing(8)

        button_row = QWidget()
        button_row_layout = QHBoxLayout(button_row)
        button_row_layout.setContentsMargins(0, 0, 0, 0)
        button_row_layout.setSpacing(0)

        self.left_panel_single_patient_pushbutton = self._icon_btn(
            "Single patient mode", "patient-icon.png", "welcomeMainAction"
        )
        self._chevron_btn = QPushButton("···")
        self._chevron_btn.setCheckable(True)
        self._chevron_btn.setChecked(False)
        self._chevron_btn.setToolTip("Assistant and preprocessing options")
        self._chevron_btn.setProperty("role", "welcomeChevron")

        button_row_layout.addWidget(self.left_panel_single_patient_pushbutton, 1)
        button_row_layout.addWidget(self._chevron_btn)

        self._options_panel = QFrame()
        self._options_panel.setProperty("role", "welcomeOptionsPanel")
        self._options_panel.hide()
        options_layout = QVBoxLayout(self._options_panel)
        options_layout.setContentsMargins(16, 12, 16, 12)
        options_layout.setSpacing(10)

        self._toggle_llm_assistant = self._toggle_switch(default=True, enabled=CHAT_AVAILABLE)
        if not CHAT_AVAILABLE:
            self._toggle_llm_assistant.setToolTip("Ollama not detected")
        options_layout.addLayout(self._toggle_row(self._toggle_llm_assistant, "LLM assistant"))

        self._toggle_preproc_labels = self._toggle_switch(default=True)
        options_layout.addLayout(self._toggle_row(self._toggle_preproc_labels, "Precomputed subject labels"))

        self._toggle_preproc_seg = self._toggle_switch(default=True)
        options_layout.addLayout(self._toggle_row(self._toggle_preproc_seg, "Preprocessed segmentation"))

        self._change_data_dir_btn = QPushButton("Edit")
        self._change_data_dir_btn.setProperty("role", "welcomeToggleSwitch")
        data_dir_row = self._toggle_row(self._change_data_dir_btn, "")
        self._data_dir_label = data_dir_row.itemAt(1).widget()  # the row's label
        self._show_data_dir()
        options_layout.addLayout(data_dir_row)

        # Zero-spacing container so button_row and _options_panel touch —
        # the panel's own layout spacing would otherwise gap them apart.
        button_group = QWidget()
        button_group_layout = QVBoxLayout(button_group)
        button_group_layout.setContentsMargins(0, 0, 0, 0)
        button_group_layout.setSpacing(0)
        button_group_layout.addWidget(button_row)
        button_group_layout.addWidget(self._options_panel)
        layout.addWidget(button_group)

        desc = QLabel("Load and analyse a single patient's cardiac MRI study.")
        desc.setProperty("role", "welcomeDescription")
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addStretch(1)
        return panel

    def _build_right_panel(self):
        panel = QWidget()
        panel.setAttribute(Qt.WA_StyledBackground, True)
        panel.setProperty("role", "welcomeSidePanel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(10)

        title = QLabel("Resources")
        title.setProperty("role", "welcomeSectionTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addSpacing(8)

        self.right_panel_community_pushbutton = self._icon_btn(
            "Research community", "globe-icon.png", "welcomeInfoAction"
        )
        self.right_panel_about_software_pushbutton = self._icon_btn(
            "About q-cardIA", "circle_question_icon.png", "welcomeInfoAction"
        )
        self.right_panel_helpwiki_pushbutton = self._icon_btn(
            "Tutorials and wiki",
            "help_wavy_question_icon_blue.png",
            "welcomeInfoAction",
        )
        self.right_panel_issues_pushbutton = self._icon_btn(
            "Issues and suggestions", "github-icon.png", "welcomeInfoAction"
        )

        for btn in (
            self.right_panel_community_pushbutton,
            self.right_panel_about_software_pushbutton,
            self.right_panel_helpwiki_pushbutton,
            self.right_panel_issues_pushbutton,
        ):
            layout.addWidget(btn)

        layout.addStretch(1)
        return panel

    # ------------------------------------------------------------------
    # Size
    # ------------------------------------------------------------------

    def _size(self):
        opt = QSize(1280, 720)
        if self.parent is not None:
            ps = self.parent.size()
            if ps.width() > 0 and ps.height() > 0:
                opt = ps
            else:
                self.parent.resize(opt)
        self.setMinimumSize(opt)
        self.resize(opt)

        btn_h = 52

        self.left_panel_single_patient_pushbutton.setFixedHeight(btn_h)
        self._chevron_btn.setFixedSize(44, btn_h)
        self.left_panel.setMinimumWidth(340)
        self.left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.right_panel.setMinimumWidth(340)
        self.right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        for btn in (
            self.right_panel_community_pushbutton,
            self.right_panel_about_software_pushbutton,
            self.right_panel_helpwiki_pushbutton,
            self.right_panel_issues_pushbutton,
        ):
            btn.setFixedHeight(btn_h)

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def _connect(self):
        self.left_panel_single_patient_pushbutton.clicked.connect(
            self.transition_signal.emit
        )
        self._chevron_btn.toggled.connect(self._on_chevron_toggled)
        self._change_data_dir_btn.clicked.connect(self._on_change_data_dir)
        self.right_panel_community_pushbutton.clicked.connect(self.community_clicked)
        self.right_panel_about_software_pushbutton.clicked.connect(self.about_clicked)
        self.right_panel_helpwiki_pushbutton.clicked.connect(self.help_clicked)
        self.right_panel_issues_pushbutton.clicked.connect(self.issues_clicked)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_chevron_toggled(self, checked: bool):
        self._options_panel.setVisible(checked)
        # unpolish/polish forces Qt to re-match QSS after a property change.
        for widget in (self.left_panel_single_patient_pushbutton, self._chevron_btn):
            widget.setProperty("panelOpen", checked)
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def _on_change_data_dir(self):
        options = QFileDialog.ShowDirsOnly
        if sys.platform == "darwin":  # native dialog crashes there; Qt's own can't open subfolders on Windows
            options |= QFileDialog.DontUseNativeDialog
        folder = QFileDialog.getExistingDirectory(self, "Select data folder", str(data_dir()), options)
        if folder:
            set_data_dir(folder)
            self._show_data_dir()

    def community_clicked(self):
        QDesktopServices.openUrl(QUrl("https://github.com/q-cardIA"))

    def about_clicked(self):
        QDesktopServices.openUrl(QUrl("https://github.com/q-cardIA/qcardia"))

    def help_clicked(self):
        QDesktopServices.openUrl(QUrl("https://github.com/q-cardIA/qcardia/wiki"))

    def issues_clicked(self):
        QDesktopServices.openUrl(QUrl("https://github.com/q-cardIA/qcardia/issues"))
