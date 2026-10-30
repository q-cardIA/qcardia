import os

from PySide6.QtCore import QPoint, QSize, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from qcardia_app.utils import bindings
from qcardia_app.utils.app_config import COMMUNITY_URL, ISSUES_URL, REPO_URL, WIKI_URL, data_dir, patient_source, set_data_dir
from qcardia_app.view.controls_dialog import ControlsDialog
from qcardia_app.view.dialogs import pick_directory
from qcardia_app.view import theme
from qcardia_app.view.theme import AppearanceDialog


class WelcomeWidget(QWidget):

    transition_signal = Signal()

    def __init__(self, parent=None, plugins=()):
        super().__init__()
        self.setObjectName("welcomeRoot")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.parent = parent
        self._plugins = list(plugins)
        self._start_holds: set[str] = set()  # see hold_start
        self._build()
        self._size()
        self._connect()

    def hold_start(self, key: str, held: bool):
        """Disable the start button while a plugin checks something, e.g. a server. key names the
        holder; the button is enabled again when no holder is left."""
        (self._start_holds.add if held else self._start_holds.discard)(key)
        self.left_panel_single_patient_pushbutton.setEnabled(not self._start_holds)

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
            "../assets/images",
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
        self._show_chevron()

    def _show_chevron(self):
        kind = "chevron_up" if self._chevron_btn.isChecked() else "chevron_down"
        self._chevron_btn.setIcon(theme.line_icon(kind, "on_accent"))

    def toggle_switch(self, default: bool, enabled: bool = True) -> QPushButton:
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
    def toggle_row(toggle_btn: QPushButton, label_text: str) -> QHBoxLayout:
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
        row = self.toggle_row(button, "")
        layout.addLayout(row)
        return button, row.itemAt(1).widget()

    def _show_data_dir(self):
        source = patient_source()
        csv = source.suffix.lower() == ".csv"
        self._data_dir_label.setText(f"{'Patient list' if csv else 'Data folder'}: {source.name}")
        self._data_dir_label.setToolTip(str(source))

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
        logo_size = theme.scaled(88)
        pix = QPixmap(self._asset("q-cardIA.png")).scaled(
            logo_size, logo_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        logo_label.setPixmap(pix)
        logo_label.setFixedSize(logo_size, logo_size)

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
        self._chevron_btn = QPushButton()
        self._chevron_btn.setCheckable(True)
        self._chevron_btn.setChecked(False)
        self._chevron_btn.setToolTip("Session options")
        self._chevron_btn.setProperty("role", "welcomeChevron")
        self._chevron_btn.setIconSize(QSize(18, 18))
        self._show_chevron()

        button_row_layout.addWidget(self.left_panel_single_patient_pushbutton, 1)
        button_row_layout.addWidget(self._chevron_btn)

        self._options_panel = QFrame(self)
        theme.notifier.changed.connect(self._replace_options)
        self._options_panel.setProperty("role", "welcomeOptionsPanel")
        self._options_panel.hide()
        options_layout = QVBoxLayout(self._options_panel)
        options_layout.setContentsMargins(16, 12, 16, 12)
        options_layout.setSpacing(10)

        # Options from plugins first, then the app's own.
        for plugin in self._plugins:
            widget = plugin.welcome_options(self)
            if widget is not None:
                options_layout.addWidget(widget)

        self._toggle_preproc_labels = self.toggle_switch(default=True)
        options_layout.addLayout(self.toggle_row(self._toggle_preproc_labels, "Precomputed subject labels"))

        self._toggle_preproc_seg = self.toggle_switch(default=True)
        options_layout.addLayout(self.toggle_row(self._toggle_preproc_seg, "Preprocessed segmentation"))

        self._toggle_study_log = self.toggle_switch(default=False)
        study_row = self.toggle_row(self._toggle_study_log, "Study logging")
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
        self.right_panel_community_pushbutton.clicked.connect(self.community_clicked)
        self.right_panel_about_software_pushbutton.clicked.connect(self.about_clicked)
        self.right_panel_helpwiki_pushbutton.clicked.connect(self.help_clicked)
        self.right_panel_issues_pushbutton.clicked.connect(self.issues_clicked)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

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
        self._show_chevron()
        for widget in (self.left_panel_single_patient_pushbutton, self._chevron_btn):
            theme.set_style_property(widget, "panelOpen", checked)

    def _on_edit_controls(self):
        ControlsDialog(self).exec()
        self._show_controls()

    def _on_edit_appearance(self):
        AppearanceDialog(self).exec()
        self._appearance_label.setText(theme.appearance_summary())

    def _on_change_data_dir(self):
        """A patient list (.csv, its paths relative to it), or a folder with one subfolder per subject."""
        menu = QMenu(self)
        as_list = menu.addAction("Patient list (.csv)…")
        menu.addAction("Data folder…")
        chosen = menu.exec(self._change_data_dir_btn.mapToGlobal(self._change_data_dir_btn.rect().bottomLeft()))
        if chosen is None:
            return
        if chosen is as_list:
            path = QFileDialog.getOpenFileName(self, "Select patient list", str(data_dir()), "Patient lists (*.csv)")[0]
        else:
            path = pick_directory(self, "Select data folder")
        if path:
            set_data_dir(path)
            self._show_data_dir()

    def community_clicked(self):
        QDesktopServices.openUrl(QUrl(COMMUNITY_URL))

    def about_clicked(self):
        QDesktopServices.openUrl(QUrl(REPO_URL))

    def help_clicked(self):
        QDesktopServices.openUrl(QUrl(WIKI_URL))

    def issues_clicked(self):
        QDesktopServices.openUrl(QUrl(ISSUES_URL))
