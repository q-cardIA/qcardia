from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from view.CentralDisplayAreaWidget import CentralDisplayAreaWidget
from view.NavigationPanelWidget import NavigationPanelWidget
from view.SidebarWidget import SidebarWidget
from view.welcome_widget import WelcomeWidget


class MainWindow(QMainWindow):

    load_button_signal = Signal(str)
    load_lge_button_signal = Signal(str)
    seg_button_signal = Signal()
    lge_seg_button_signal = Signal()
    save_button_signal = Signal()
    time_frame_changed_signal = Signal(int)

    def __init__(self):
        super().__init__()
        self.folder_selected = False
        # Create welcome widget first, then set application palette so it applies to all widgets
        self.welcome_widget = WelcomeWidget(self)
        self.set_dark_palette()
        self.setCentralWidget(self.welcome_widget)
        # Welcome transitions to analysis UI (data loaded later from left panel)
        self.welcome_widget.transition_signal.connect(self.build_analysis_interface)
        # Default to full-screen so the welcome layout fills the display
        self.setWindowState(Qt.WindowMaximized)

    def set_dark_palette(self):
        palette = QPalette()
        brush = QBrush(QColor(255, 255, 255, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.WindowText, brush)
        # (Palette setup code continues unchanged below)
        try:
            QApplication.setPalette(palette)
        except Exception:
            self.setPalette(palette)

    def build_analysis_interface(self):
        """Construct analysis UI without loading data."""
        # Sidebar
        self.sidebar = SidebarWidget(self)
        self.sidebar.load_requested.connect(self.on_load_button_clicked)
        self.sidebar.load_image_requested.connect(self.on_load_image_clicked)
        self.sidebar.predict_requested.connect(self.emit_segmentation_signal)
        self.sidebar.predict_lge_requested.connect(self.emit_lge_segmentation_signal)
        self.sidebar.save_requested.connect(self.emit_save_signal)

        # Central display (3 orthogonal viewers + empty cell)
        self.display_area_widget = CentralDisplayAreaWidget(self)

        # Navigation panel (right)
        self.navigation_panel = NavigationPanelWidget(self)

        # Time controls below central area (keep existing semantics)
        time_controls_container = QWidget()
        time_layout = QHBoxLayout(time_controls_container)
        time_layout.setContentsMargins(6, 4, 6, 4)
        time_layout.setSpacing(8)
        self.time_label = QLabel("Time Frame: 0 / 0")
        self.time_label.setStyleSheet("color: #E6E6E6; font-size:12px")
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.setEnabled(False)
        self.time_slider.valueChanged.connect(self.on_time_slider_changed)
        self.time_slider.setStyleSheet(
            """QSlider::groove:horizontal {background:#2F3440; height:6px; border-radius:3px;}\nQSlider::handle:horizontal {background:#85aaff; width:14px; margin:-4px 0; border-radius:7px;}\nQSlider::handle:horizontal:hover {background:#a0c0ff;}"""
        )
        time_layout.addWidget(self.time_label)
        time_layout.addWidget(self.time_slider)

        # Top bar
        self.top_bar = QWidget()
        self.top_bar.setStyleSheet("QWidget{background-color:#1E1E1E;}")
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(8, 4, 8, 4)
        top_layout.setSpacing(12)
        self.toggle_sidebar_btn = QPushButton("Hide Menu")
        self.toggle_sidebar_btn.setCheckable(True)
        self.toggle_sidebar_btn.setStyleSheet(
            "QPushButton{background-color:#2F3440; color:#E6E6E6; border:none; padding:6px 12px; border-radius:4px;}"
            "QPushButton:hover{background-color:#3d4450;}"
            "QPushButton:checked{background-color:#3d4450;}"
        )
        self.toggle_sidebar_btn.toggled.connect(self._on_toggle_sidebar)
        self.app_title = QLabel("q-cardIA v0.0.1")
        self.app_title.setStyleSheet("color:#E6E6E6; font-size:18px; font-weight:bold")
        top_layout.addWidget(self.toggle_sidebar_btn)
        top_layout.addWidget(self.app_title)
        top_layout.addStretch(1)

        # Assemble main layout
        central = QWidget()
        main_v = QVBoxLayout(central)
        main_v.setContentsMargins(0, 0, 0, 0)
        main_v.setSpacing(0)
        main_v.addWidget(self.top_bar)
        content_row = QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(0)
        content_row.addWidget(self.sidebar)
        content_row.addWidget(self.display_area_widget, 1)
        content_row.addWidget(self.navigation_panel)
        main_v.addLayout(content_row)
        main_v.addWidget(time_controls_container)
        self.setCentralWidget(central)

    def _directory_dialog_options(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        options |= (
            QFileDialog.DontUseNativeDialog
        )  # use Qt dialog to avoid macOS NSOpenPanel crash
        return options

    def on_load_button_clicked(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder", "", self._directory_dialog_options()
        )
        if not folder_path:
            return
        self.folder_selected = True
        self.sidebar.enable_post_load()
        self.load_button_signal.emit(folder_path)

    def on_load_image_clicked(self):
        """Prompt for an LGE folder and emit the selection to controllers."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select DICOM Folder", "", self._directory_dialog_options()
        )
        if not folder_path:
            return

        self.load_lge_button_signal.emit(folder_path)

    def _on_toggle_sidebar(self, checked):
        if checked:
            self.sidebar.hide()
            self.toggle_sidebar_btn.setText("Show Menu")
        else:
            self.sidebar.show()
            self.toggle_sidebar_btn.setText("Hide Menu")

    def update_display(self, image_stack, slice_positions=None):
        """Update display with new image stack, centering the view."""
        if hasattr(self, "display_area_widget"):
            self.display_area_widget.displayed_image = image_stack
            self.display_area_widget.displayed_image_uid = None
            center_indices = [
                image_stack.shape[0] // 2,
                image_stack.shape[1] // 2,
                image_stack.shape[2] // 2,
            ]
            self.display_area_widget.point_clicker_position = center_indices
            if (
                slice_positions is not None
                and len(slice_positions) == image_stack.shape[0]
            ):
                self.display_area_widget.primary_slice_positions = [
                    float(val) for val in slice_positions
                ]
            else:
                self.display_area_widget.primary_slice_positions = None
            self.display_area_widget.update_viewers_image()

    def update_display_with_position(
        self, image_stack, preserve_position=False, slice_positions=None
    ):
        """Update display, optionally preserving crosshair position."""
        if hasattr(self, "display_area_widget"):
            da = self.display_area_widget

            if preserve_position:
                # Get current position and clamp to new volume bounds
                pos = getattr(da, "point_clicker_position", [0, 0, 0])
                x = min(max(0, pos[0]), image_stack.shape[0] - 1)
                y = min(max(0, pos[1]), image_stack.shape[1] - 1)
                z = min(max(0, pos[2]), image_stack.shape[2] - 1)
                da.displayed_image = image_stack
                da.point_clicker_position = [x, y, z]
            else:
                # Center the view
                center_indices = [
                    image_stack.shape[0] // 2,
                    image_stack.shape[1] // 2,
                    image_stack.shape[2] // 2,
                ]
                da.displayed_image = image_stack
                da.point_clicker_position = center_indices

            da.displayed_image_uid = None
            if (
                slice_positions is not None
                and len(slice_positions) == image_stack.shape[0]
            ):
                da.primary_slice_positions = [float(val) for val in slice_positions]
            else:
                da.primary_slice_positions = None
            da.update_viewers_image()

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
        """Update the time slider to reflect the number of temporal frames."""
        if hasattr(self, "time_slider"):
            self.time_slider.setMaximum(num_frames - 1)
            self.time_slider.setValue(0)
            self.time_slider.setEnabled(num_frames > 1)
            self.time_label.setText(f"Time Frame: 1 / {num_frames}")

    def on_time_slider_changed(self, value):
        """Handle time slider changes and update the display."""
        if hasattr(self, "time_slider") and hasattr(self, "time_label"):
            max_val = self.time_slider.maximum()
            self.time_label.setText(f"Time Frame: {value + 1} / {max_val + 1}")
            # Emit signal to controller to update the displayed volume
            if hasattr(self, "time_frame_changed_signal"):
                self.time_frame_changed_signal.emit(value)
