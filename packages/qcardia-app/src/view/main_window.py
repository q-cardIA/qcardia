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
from view.utils import SoftwareConfigResources
from view.welcome_widget import WelcomeWidget


class MainWindow(QMainWindow):

    load_button_signal = Signal(str)
    seg_button_signal = Signal()
    save_button_signal = Signal()
    time_frame_changed_signal = Signal(int)

    def __init__(self):
        super(MainWindow, self).__init__()
        self.folder_selected = False

        # Set default window size
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)

        # Set up the stylesheet for the application using shared dark theme colors
        colors = SoftwareConfigResources.getInstance().stylesheet_components
        self.setStyleSheet(
            """
            MainWindow {
                background-color: %s;
            }
            #analysisRightPanel {
                background-color: %s;
                border: none;
            }
            QPushButton {
                background-color: %s;
                color: %s;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: %s;
            }
            QPushButton:pressed {
                background-color: %s;
            }
            #segButton {
                background-color: %s;
            }
            #segButton:hover {
                background-color: %s;
            }
        """
            % (
                colors["Color2"],
                colors["Color2"],
                colors["Color3"],
                colors["Color7"],
                colors["Color4"],
                colors["Color5"],
                colors["Process"],
                colors["Process_pressed"],
            )
        )

        # Create welcome widget first, then set application palette so it applies to all widgets
        self.welcome_widget = WelcomeWidget(self)
        self.set_dark_palette()
        self.setCentralWidget(self.welcome_widget)

        # Welcome transitions to analysis UI (data loaded later from left panel)
        self.welcome_widget.transition_signal.connect(self.build_analysis_interface)

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
        self.right_panel = QWidget()
        self.right_panel.setObjectName("analysisRightPanel")
        self.right_panel.setFixedWidth(200)
        self.right_panel_layout = QVBoxLayout(self.right_panel)
        self.right_panel_layout.setSpacing(10)
        self.right_panel_layout.setContentsMargins(10, 10, 10, 10)

        self.time_label = QLabel("Time Frame: 0 / 0")
        self.time_label.setStyleSheet("color: white; font-size: 12px;")
        self.right_panel_layout.addWidget(self.time_label)

        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.setValue(0)
        self.time_slider.setEnabled(False)
        self.time_slider.setStyleSheet(
            """
            QSlider::groove:horizontal { background: #3a3f4a; height: 8px; border-radius: 4px; }
            QSlider::handle:horizontal { background: #85aaff; width: 16px; margin: -4px 0; border-radius: 8px; }
            QSlider::handle:horizontal:hover { background: #a0c0ff; }
            """
        )
        self.time_slider.valueChanged.connect(self.on_time_slider_changed)
        self.right_panel_layout.addWidget(self.time_slider)
        self.right_panel_layout.addSpacing(20)

        self.seg_button = QPushButton("Run Segmentation")
        self.seg_button.setObjectName("segButton")
        self.seg_button.clicked.connect(self.emit_segmentation_signal)
        self.seg_button.setEnabled(False)
        self.right_panel_layout.addWidget(self.seg_button)

        self.save_button = QPushButton("Save Results")
        self.save_button.setObjectName("saveButton")
        self.save_button.clicked.connect(self.emit_save_signal)
        self.save_button.setEnabled(False)
        self.right_panel_layout.addWidget(self.save_button)
        self.right_panel_layout.addStretch()

        self.left_panel = QWidget()
        self.left_panel.setObjectName("analysisLeftPanel")
        self.left_panel.setFixedWidth(180)
        self.left_panel_layout = QVBoxLayout(self.left_panel)
        self.left_panel_layout.setSpacing(10)
        self.left_panel_layout.setContentsMargins(10, 10, 10, 10)

        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.on_load_button_clicked)
        self.left_panel_layout.addWidget(self.load_button)
        self.left_panel_layout.addStretch()

        self.display_area_widget = CentralDisplayAreaWidget(self)

        self.central_widget = QWidget()
        self.center_panel_layout = QHBoxLayout(self.central_widget)
        self.center_panel_layout.setSpacing(0)
        self.center_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.center_panel_layout.addWidget(self.left_panel)
        self.center_panel_layout.addWidget(self.display_area_widget)
        self.center_panel_layout.addWidget(self.right_panel)
        self.setCentralWidget(self.central_widget)

    def on_load_button_clicked(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder_path:
            return
        self.folder_selected = True
        self.seg_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.load_button_signal.emit(folder_path)

    def update_display(self, image_stack):
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
            self.display_area_widget.update_viewers_image()

    def update_display_with_position(self, image_stack, preserve_position=False):
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
            da.update_viewers_image()

    def has_folder_been_selected(self):
        return self.folder_selected

    def emit_segmentation_signal(self):
        if self.has_folder_been_selected():
            self.seg_button_signal.emit()

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
