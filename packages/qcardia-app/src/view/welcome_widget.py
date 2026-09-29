import os

from PySide6.QtCore import QPoint, QSize, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from model.background_worker import CallableWorker
from utils import bindings, llm_devices
from utils.app_config import COMMUNITY_URL, ISSUES_URL, REPO_URL, WIKI_URL, data_dir, set_data_dir
from view.controls_dialog import ControlsDialog
from view.dialogs import pick_directory
from view import theme
from view.theme import AppearanceDialog, PopupComboBox, thin_loading_bar


class WelcomeWidget(QWidget):

    transition_signal = Signal()

    def __init__(self, parent=None):
        super().__init__()
        self.setObjectName("welcomeRoot")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.parent = parent
        self._device_worker = None
        self._build()
        self._size()
        self._connect()
        # In the background: this used to run at import and stalled startup.
        self._start_device_probe(llm_devices.LOCAL_DEVICE, llm_devices.FAST_PROBE_TIMEOUT)

    @property
    def assistant_disabled(self) -> bool:
        return not (llm_devices.chat_available() and self._toggle_llm_assistant.isChecked())

    @property
    def use_cardisort_cache(self) -> bool:
        return self._toggle_preproc_labels.isChecked()

    @property
    def use_segmentation_cache(self) -> bool:
        return self._toggle_preproc_seg.isChecked()

    @property
    def study_logging(self) -> bool:
        return self._toggle_study_log.isChecked()

    @property
    def participant_id(self) -> str:
        return self._participant_edit.text()

    @property
    def condition(self) -> str:
        return self._condition_edit.text()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _asset(self, filename):
        return os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../assets/images",
            filename,
        )

    def _icon_btn(self, text, image_file, role, token="text_muted"):
        btn = QPushButton(f"  {text}")
        btn.setProperty("role", role)
        btn.setIcon(theme.tinted_icon(self._asset(image_file), token))
        btn.setIconSize(QSize(20, 20))
        return btn

    def _retint_main_action(self):
        self.left_panel_single_patient_pushbutton.setIcon(
            theme.tinted_icon(self._asset("patient-icon.png"), "on_accent"))

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

    def _edit_row(self, layout) -> tuple[QPushButton, QLabel]:
        button = QPushButton("Edit")
        button.setProperty("role", "welcomeToggleSwitch")
        row = self._toggle_row(button, "")
        layout.addLayout(row)
        return button, row.itemAt(1).widget()

    def _show_data_dir(self):
        self._data_dir_label.setText(f"Data folder: {data_dir().name}")
        self._data_dir_label.setToolTip(str(data_dir()))

    def _show_controls(self):
        self._controls_label.setText(f"Controls: {bindings.preset_name(bindings.current())}")

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

        card_layout = QHBoxLayout(self.central_label)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(6)  # the gap between the analysis screen's cards

        self.left_panel = self._build_left_panel()
        self.right_panel = self._build_right_panel()

        card_layout.addWidget(self.left_panel, 1)
        card_layout.addWidget(self.right_panel, 1)

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
            "Single patient mode", "patient-icon.png", "welcomeMainAction", "on_accent"
        )
        theme.notifier.changed.connect(self._retint_main_action)
        self._chevron_btn = QPushButton("···")
        self._chevron_btn.setCheckable(True)
        self._chevron_btn.setChecked(False)
        self._chevron_btn.setToolTip("Assistant and preprocessing options")
        self._chevron_btn.setProperty("role", "welcomeChevron")

        button_row_layout.addWidget(self.left_panel_single_patient_pushbutton, 1)
        button_row_layout.addWidget(self._chevron_btn)

        self._options_panel = QFrame(self)
        theme.notifier.changed.connect(self._replace_options)
        self._options_panel.setProperty("role", "welcomeOptionsPanel")
        self._options_panel.hide()
        options_layout = QVBoxLayout(self._options_panel)
        options_layout.setContentsMargins(16, 12, 16, 12)
        options_layout.setSpacing(10)

        # Defaults to this computer, so nobody has to touch this to start.
        device_row = QHBoxLayout()
        device_row.setSpacing(10)
        self._device_combo = PopupComboBox()
        self._device_combo.setProperty("role", "welcomeDeviceCombo")
        for device in llm_devices.load_devices():
            self._device_combo.addItem(device.label, device)
            if device.note:
                self._device_combo.setItemData(
                    self._device_combo.count() - 1, device.note, Qt.ToolTipRole
                )
        device_label = QLabel("LLM device")
        device_label.setProperty("role", "welcomeOptionLabel")
        device_row.addWidget(self._device_combo, 1)
        device_row.addWidget(device_label)
        device_row.addStretch(1)
        options_layout.addLayout(device_row)

        # Directly under the combo, so the activity reads as belonging to that choice.
        self._device_bar = thin_loading_bar()
        options_layout.addWidget(self._device_bar)

        self._device_status = QLabel("Checking…")
        self._device_status.setProperty("role", "welcomeDeviceStatus")
        self._device_status.setWordWrap(True)
        options_layout.addWidget(self._device_status)

        # Disabled until the first probe: an assistant that cannot answer helps nobody.
        self._toggle_llm_assistant = self._toggle_switch(default=True, enabled=False)
        options_layout.addLayout(self._toggle_row(self._toggle_llm_assistant, "LLM assistant"))

        self._toggle_preproc_labels = self._toggle_switch(default=True)
        options_layout.addLayout(self._toggle_row(self._toggle_preproc_labels, "Precomputed subject labels"))

        self._toggle_preproc_seg = self._toggle_switch(default=True)
        options_layout.addLayout(self._toggle_row(self._toggle_preproc_seg, "Preprocessed segmentation"))

        self._toggle_study_log = self._toggle_switch(default=False)
        study_row = self._toggle_row(self._toggle_study_log, "Study logging")
        self._participant_edit, self._condition_edit = QLineEdit(), QLineEdit()
        for edit, placeholder in ((self._participant_edit, "Participant ID"), (self._condition_edit, "Condition")):
            edit.setPlaceholderText(placeholder)
            edit.setEnabled(False)
            self._toggle_study_log.toggled.connect(edit.setEnabled)
            study_row.insertWidget(study_row.count() - 1, edit)
        options_layout.addLayout(study_row)

        self._change_data_dir_btn, self._data_dir_label = self._edit_row(options_layout)
        self._show_data_dir()
        self._edit_controls_btn, self._controls_label = self._edit_row(options_layout)
        self._show_controls()
        self._edit_appearance_btn, self._appearance_label = self._edit_row(options_layout)
        self._appearance_label.setText(theme.appearance_summary())

        layout.addWidget(button_row)
        self._button_row = button_row  # the options panel drops down from here, over the screen

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
        # The parent may be maximized or full screen: never pin the window's minimum to that.
        self.setMinimumSize(opt.boundedTo(QSize(1280, 720)))
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
        self._edit_controls_btn.clicked.connect(self._on_edit_controls)
        self._edit_appearance_btn.clicked.connect(self._on_edit_appearance)
        self._device_combo.currentIndexChanged.connect(self._on_device_selected)
        self.right_panel_community_pushbutton.clicked.connect(self.community_clicked)
        self.right_panel_about_software_pushbutton.clicked.connect(self.about_clicked)
        self.right_panel_helpwiki_pushbutton.clicked.connect(self.help_clicked)
        self.right_panel_issues_pushbutton.clicked.connect(self.issues_clicked)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    # -- LLM device selection -------------------------------------------

    def _on_device_selected(self, index: int):
        device = self._device_combo.itemData(index)
        if device is not None:
            self._start_device_probe(device, llm_devices.DEVICE_PROBE_TIMEOUT)

    def _start_device_probe(self, device, timeout: float):
        """Validate off the GUI thread, holding Start: a dead host takes the full timeout."""
        self._set_device_busy(True)
        self._device_status.setText(f"Checking {device.label}…")
        worker = CallableWorker(llm_devices.activate, device, timeout=timeout)
        worker.finished_ok.connect(lambda result, d=device: self._on_device_probed(d, result))
        worker.error.connect(lambda exc, d=device: self._on_device_probe_failed(d, exc))
        # Held on self so the thread isn't garbage-collected mid-run.
        self._device_worker = worker
        worker.start()

    def _set_device_busy(self, busy: bool):
        self._device_combo.setEnabled(not busy)
        self.left_panel_single_patient_pushbutton.setEnabled(not busy)
        self._device_bar.setVisible(busy)

    def _on_device_probed(self, device, result):
        self._set_device_busy(False)
        if result.ok:
            self._device_status.setText(f"{device.label}: {result.detail}")
        else:
            # Name what is still in use: activate() leaves the previous device active.
            self._device_status.setText(
                f"{device.label} unavailable. {result.detail} "
                f"Still using {llm_devices.current().label}."
            )
        enabled = llm_devices.chat_available()
        self._toggle_llm_assistant.setEnabled(enabled)
        if not enabled:
            self._toggle_llm_assistant.setToolTip("No reachable Ollama server")

    def _on_device_probe_failed(self, device, exc: Exception):
        self._set_device_busy(False)
        self._device_status.setText(f"{device.label} check failed: {exc}")
        self._toggle_llm_assistant.setEnabled(llm_devices.chat_available())

    def _place_options(self):
        panel, row = self._options_panel, self._button_row
        panel.setFixedWidth(row.width())
        panel.adjustSize()
        top = row.mapTo(self, QPoint(0, 0))
        y = top.y() + row.height()
        if y + panel.height() > self.height():  # no room below: open upward, within the window
            y = max(0, top.y() - panel.height())
        panel.move(top.x(), y)
        panel.raise_()

    def _replace_options(self):
        if self._options_panel.isVisible():  # e.g. a text size change moves the button row
            QTimer.singleShot(0, self._place_options)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._options_panel.isVisible():
            self._place_options()

    def _on_chevron_toggled(self, checked: bool):
        if checked:
            self._place_options()
        self._options_panel.setVisible(checked)
        for widget in (self.left_panel_single_patient_pushbutton, self._chevron_btn):
            theme.set_style_property(widget, "panelOpen", checked)

    def _on_edit_controls(self):
        ControlsDialog(self).exec()
        self._show_controls()

    def _on_edit_appearance(self):
        AppearanceDialog(self).exec()
        self._appearance_label.setText(theme.appearance_summary())

    def _on_change_data_dir(self):
        folder = pick_directory(self, "Select data folder")
        if folder:
            set_data_dir(folder)
            self._show_data_dir()

    def community_clicked(self):
        QDesktopServices.openUrl(QUrl(COMMUNITY_URL))

    def about_clicked(self):
        QDesktopServices.openUrl(QUrl(REPO_URL))

    def help_clicked(self):
        QDesktopServices.openUrl(QUrl(WIKI_URL))

    def issues_clicked(self):
        QDesktopServices.openUrl(QUrl(ISSUES_URL))
