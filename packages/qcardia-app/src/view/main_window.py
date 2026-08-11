from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from utils.app_config import CHAT_AVAILABLE, ICON_PATH
from view.central_display_area_widget import CentralDisplayAreaWidget
from view.chat_widget import ChatWidget
from view.loading_overlay import LoadingOverlay
from view.series_browser_widget import SeriesBrowserWidget
from view.sidebar_widget import SidebarWidget
from view.welcome_widget import WelcomeWidget


def load_app_icon() -> QIcon | None:
    return QIcon(str(ICON_PATH)) if ICON_PATH.is_file() else None


class MainWindow(QMainWindow):

    load_patient_button_signal = Signal(str)
    seg_button_signal = Signal()
    lge_seg_button_signal = Signal()
    save_button_signal = Signal()
    time_frame_changed_signal = Signal(int)
    preset_requested_signal = Signal(str)
    reclassify_requested_signal = Signal(object, str)

    def __init__(self):
        super().__init__()
        icon = load_app_icon()
        if icon is not None:
            self.setWindowIcon(icon)
        self.folder_selected = False
        self.welcome_widget = WelcomeWidget(self)
        self.setCentralWidget(self.welcome_widget)
        self.welcome_widget.transition_signal.connect(self.build_analysis_interface)
        self.setWindowState(Qt.WindowMaximized)

    def build_analysis_interface(self):
        self.sidebar = SidebarWidget(self)
        self.sidebar.load_patient_requested.connect(self.on_load_patient_clicked)
        self.sidebar.predict_requested.connect(self.emit_segmentation_signal)
        self.sidebar.predict_lge_requested.connect(self.emit_lge_segmentation_signal)
        self.sidebar.save_requested.connect(self.emit_save_signal)
        self.sidebar.preset_requested.connect(self.preset_requested_signal.emit)

        self.display_area_widget = CentralDisplayAreaWidget(self)
        self.loading_overlay = LoadingOverlay(self.display_area_widget)
        self.series_browser_widget = SeriesBrowserWidget(self)
        self.series_browser_widget.reclassify_requested.connect(
            self.reclassify_requested_signal.emit
        )
        self.chat_widget = ChatWidget(self)
        screen_width = QGuiApplication.primaryScreen().availableGeometry().width()
        self.chat_widget.setMaximumWidth(int(screen_width * 0.75))

        time_controls_container = QWidget()
        time_layout = QHBoxLayout(time_controls_container)
        time_layout.setContentsMargins(6, 4, 6, 4)
        time_layout.setSpacing(8)
        self.time_label = QLabel("Time Frame: 0 / 0")
        self.time_label.setProperty("secondary", "true")
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.setEnabled(False)
        self.time_slider.valueChanged.connect(self.on_time_slider_changed)
        self.time_slider.setProperty("role", "timeSlider")
        time_layout.addWidget(self.time_label)
        time_layout.addWidget(self.time_slider)

        self.top_bar = QWidget()
        self.top_bar.setProperty("role", "topBar")
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(8, 4, 8, 4)
        top_layout.setSpacing(12)
        self.toggle_sidebar_btn = QPushButton("Hide Menu")
        self.toggle_sidebar_btn.setCheckable(True)
        self.toggle_sidebar_btn.setProperty("role", "sidebarToggle")
        self.toggle_sidebar_btn.toggled.connect(self._on_toggle_sidebar)
        self.toggle_series_browser_btn = QPushButton("Hide Series Panel")
        self.toggle_series_browser_btn.setCheckable(True)
        self.toggle_series_browser_btn.setProperty("role", "sidebarToggle")
        self.toggle_series_browser_btn.toggled.connect(self._on_toggle_series_browser)
        self.toggle_chat_btn = QPushButton("Hide Assistant")
        self.toggle_chat_btn.setCheckable(True)
        self.toggle_chat_btn.setProperty("role", "sidebarToggle")
        self.toggle_chat_btn.toggled.connect(self._on_toggle_chat)
        self.contour_edit_btn = QPushButton("Edit Contours")
        self.contour_edit_btn.setCheckable(True)
        self.contour_edit_btn.setProperty("role", "contourToggle")
        self.contour_edit_btn.toggled.connect(self._on_contour_edit_toggled)
        self.add_point_btn = QPushButton("Add Point")
        self.add_point_btn.setCheckable(True)
        self.add_point_btn.setEnabled(False)
        self.add_point_btn.setProperty("role", "contourTool")
        self.add_point_btn.toggled.connect(self._on_add_point_toggled)
        self.remove_point_btn = QPushButton("Remove Point")
        self.remove_point_btn.setCheckable(True)
        self.remove_point_btn.setEnabled(False)
        self.remove_point_btn.setProperty("role", "contourTool")
        self.remove_point_btn.toggled.connect(self._on_remove_point_toggled)
        self.app_title = QLabel("q-cardIA v0.0.1")
        self.app_title.setProperty("role", "appTitle")
        top_layout.addWidget(self.toggle_sidebar_btn)
        top_layout.addWidget(self.toggle_series_browser_btn)
        top_layout.addWidget(self.toggle_chat_btn)
        top_layout.addWidget(self.contour_edit_btn)
        top_layout.addWidget(self.add_point_btn)
        top_layout.addWidget(self.remove_point_btn)
        top_layout.addWidget(self.app_title)
        top_layout.addStretch(1)

        central = QWidget()
        main_v = QVBoxLayout(central)
        main_v.setContentsMargins(0, 0, 0, 0)
        main_v.setSpacing(0)
        main_v.addWidget(self.top_bar)
        content_row = QSplitter(Qt.Horizontal)
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setHandleWidth(4)
        content_row.addWidget(self.sidebar)
        content_row.addWidget(self.display_area_widget)
        content_row.addWidget(self.series_browser_widget)
        content_row.addWidget(self.chat_widget)
        content_row.setStretchFactor(1, 1)  # display area absorbs extra space when resizing
        for i in range(content_row.count()):
            content_row.setCollapsible(i, False)

        # Give the chat panel a real default width (not just its minimum) so
        # user message bubbles — capped at a fraction of the panel's own
        # width — actually have room to grow, rather than defaulting to
        # sizeHint-thin. sidebar/series_browser are setFixedWidth() in their
        # own classes (210/200) — use those exact values here too, otherwise
        # any slack between what we ask for and their real fixed width shows
        # up as dead, unassigned space in the splitter.
        # minimumWidth() (not width()) — setFixedWidth() sets it immediately,
        # while actual widget geometry isn't resolved until the first layout pass.
        sidebar_width = self.sidebar.minimumWidth()
        series_width = self.series_browser_widget.minimumWidth()
        chat_default_width = min(int(screen_width * 0.15), self.chat_widget.maximumWidth())
        display_width = max(400, screen_width - sidebar_width - series_width - chat_default_width)
        content_row.setSizes([sidebar_width, display_width, series_width, chat_default_width])

        main_v.addWidget(content_row, 1)
        main_v.addWidget(time_controls_container)
        self.setCentralWidget(central)

        if not CHAT_AVAILABLE or self.welcome_widget.assistant_disabled:
            self.chat_widget.hide()
            self.toggle_chat_btn.hide()

    def _directory_dialog_options(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        options |= (
            QFileDialog.DontUseNativeDialog
        )  # use Qt dialog to avoid macOS NSOpenPanel crash
        return options

    def on_load_patient_clicked(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Patient Folder", "", self._directory_dialog_options()
        )
        if not folder_path:
            return
        self.folder_selected = True
        self.sidebar.enable_post_load()
        self.load_patient_button_signal.emit(folder_path)

    def show_loading(self, text: str):
        self.sidebar.btn_load_patient.setEnabled(False)
        self.loading_overlay.show_message(text)

    def hide_loading(self):
        self.sidebar.btn_load_patient.setEnabled(True)
        self.loading_overlay.hide_overlay()

    def _on_toggle_sidebar(self, checked):
        if checked:
            self.sidebar.hide()
            self.toggle_sidebar_btn.setText("Show Menu")
        else:
            self.sidebar.show()
            self.toggle_sidebar_btn.setText("Hide Menu")

    def _on_toggle_series_browser(self, checked):
        if checked:
            self.series_browser_widget.hide()
            self.toggle_series_browser_btn.setText("Show Series Panel")
        else:
            self.series_browser_widget.show()
            self.toggle_series_browser_btn.setText("Hide Series Panel")

    def _on_toggle_chat(self, checked):
        if checked:
            self.chat_widget.hide()
            self.toggle_chat_btn.setText("Show Assistant")
        else:
            self.chat_widget.show()
            self.toggle_chat_btn.setText("Hide Assistant")

    def _on_contour_edit_toggled(self, checked):
        self.display_area_widget.set_contour_edit_mode(checked)
        self.contour_edit_btn.setText(
            "Editing Contours" if checked else "Edit Contours"
        )
        self.add_point_btn.setEnabled(checked)
        self.remove_point_btn.setEnabled(checked)
        if not checked:
            self.add_point_btn.setChecked(False)
            self.remove_point_btn.setChecked(False)
            self.display_area_widget.set_contour_edit_tool("move")

    def _on_add_point_toggled(self, checked):
        if checked:
            self.remove_point_btn.setChecked(False)
            self.display_area_widget.set_contour_edit_tool("add")
        elif not self.remove_point_btn.isChecked():
            self.display_area_widget.set_contour_edit_tool("move")

    def _on_remove_point_toggled(self, checked):
        if checked:
            self.add_point_btn.setChecked(False)
            self.display_area_widget.set_contour_edit_tool("remove")
        elif not self.add_point_btn.isChecked():
            self.display_area_widget.set_contour_edit_tool("move")

    def has_folder_been_selected(self):
        return self.folder_selected

    def emit_segmentation_signal(self):
        if self.has_folder_been_selected():
            self.seg_button_signal.emit()

    def emit_lge_segmentation_signal(self):
        self.lge_seg_button_signal.emit()

    def emit_save_signal(self):
        if self.has_folder_been_selected():
            self.save_button_signal.emit()

    def update_time_slider_range(self, num_frames):
        self.time_slider.setMaximum(num_frames - 1)
        self.time_slider.setValue(0)
        self.time_slider.setEnabled(num_frames > 1)
        self.time_label.setText(f"Time Frame: 1 / {num_frames}")

    def on_time_slider_changed(self, value):
        max_val = self.time_slider.maximum()
        self.time_label.setText(f"Time Frame: {value + 1} / {max_val + 1}")
        self.time_frame_changed_signal.emit(value)

    def show_error(self, title, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec()

    def show_warning(self, title, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec()

    def show_info(self, title, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec()
