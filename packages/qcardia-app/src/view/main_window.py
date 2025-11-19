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

        self.welcome_widget.transition_signal.connect(self.select_folder)

    def set_dark_palette(self):
        palette = QPalette()
        brush = QBrush(QColor(255, 255, 255, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.WindowText, brush)
        brush1 = QBrush(QColor(0, 0, 0, 0))
        brush1.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Button, brush1)
        brush2 = QBrush(QColor(66, 73, 90, 255))
        brush2.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Light, brush2)
        brush3 = QBrush(QColor(55, 61, 75, 255))
        brush3.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Midlight, brush3)
        brush4 = QBrush(QColor(22, 24, 30, 255))
        brush4.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Dark, brush4)
        brush5 = QBrush(QColor(29, 32, 40, 255))
        brush5.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Mid, brush5)
        brush6 = QBrush(QColor(210, 210, 210, 255))
        brush6.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Text, brush6)
        palette.setBrush(QPalette.Active, QPalette.BrightText, brush)
        palette.setBrush(QPalette.Active, QPalette.ButtonText, brush)
        palette.setBrush(QPalette.Active, QPalette.Base, brush1)
        palette.setBrush(QPalette.Active, QPalette.Window, brush1)
        brush7 = QBrush(QColor(0, 0, 0, 255))
        brush7.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Shadow, brush7)
        brush8 = QBrush(QColor(85, 170, 255, 255))
        brush8.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Highlight, brush8)
        palette.setBrush(QPalette.Active, QPalette.Link, brush8)
        brush9 = QBrush(QColor(255, 0, 127, 255))
        brush9.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.LinkVisited, brush9)
        palette.setBrush(QPalette.Active, QPalette.AlternateBase, brush4)
        brush10 = QBrush(QColor(44, 49, 60, 255))
        brush10.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.ToolTipBase, brush10)
        palette.setBrush(QPalette.Active, QPalette.ToolTipText, brush6)
        brush11 = QBrush(QColor(210, 210, 210, 128))
        brush11.setStyle(Qt.NoBrush)
        # if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Active, QPalette.PlaceholderText, brush11)
        # endif
        palette.setBrush(QPalette.Inactive, QPalette.WindowText, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.Light, brush2)
        palette.setBrush(QPalette.Inactive, QPalette.Midlight, brush3)
        palette.setBrush(QPalette.Inactive, QPalette.Dark, brush4)
        palette.setBrush(QPalette.Inactive, QPalette.Mid, brush5)
        palette.setBrush(QPalette.Inactive, QPalette.Text, brush6)
        palette.setBrush(QPalette.Inactive, QPalette.BrightText, brush)
        palette.setBrush(QPalette.Inactive, QPalette.ButtonText, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Base, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.Window, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.Shadow, brush7)
        palette.setBrush(QPalette.Inactive, QPalette.Highlight, brush8)
        palette.setBrush(QPalette.Inactive, QPalette.Link, brush8)
        palette.setBrush(QPalette.Inactive, QPalette.LinkVisited, brush9)
        palette.setBrush(QPalette.Inactive, QPalette.AlternateBase, brush4)
        palette.setBrush(QPalette.Inactive, QPalette.ToolTipBase, brush10)
        palette.setBrush(QPalette.Inactive, QPalette.ToolTipText, brush6)
        brush12 = QBrush(QColor(210, 210, 210, 128))
        brush12.setStyle(Qt.NoBrush)
        # if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush12)
        # endif
        palette.setBrush(QPalette.Disabled, QPalette.WindowText, brush4)
        palette.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Light, brush2)
        palette.setBrush(QPalette.Disabled, QPalette.Midlight, brush3)
        palette.setBrush(QPalette.Disabled, QPalette.Dark, brush4)
        palette.setBrush(QPalette.Disabled, QPalette.Mid, brush5)
        palette.setBrush(QPalette.Disabled, QPalette.Text, brush4)
        palette.setBrush(QPalette.Disabled, QPalette.BrightText, brush)
        palette.setBrush(QPalette.Disabled, QPalette.ButtonText, brush4)
        palette.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Window, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Shadow, brush7)
        brush13 = QBrush(QColor(51, 153, 255, 255))
        brush13.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Disabled, QPalette.Highlight, brush13)
        palette.setBrush(QPalette.Disabled, QPalette.Link, brush8)
        palette.setBrush(QPalette.Disabled, QPalette.LinkVisited, brush9)
        palette.setBrush(QPalette.Disabled, QPalette.AlternateBase, brush10)
        palette.setBrush(QPalette.Disabled, QPalette.ToolTipBase, brush10)
        palette.setBrush(QPalette.Disabled, QPalette.ToolTipText, brush6)
        brush14 = QBrush(QColor(210, 210, 210, 128))
        brush14.setStyle(Qt.NoBrush)
        # if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush14)
        # endif
        # Apply palette at the application level so it affects all widgets
        try:
            QApplication.setPalette(palette)
        except Exception:
            # Fallback to setting on this window if QApplication isn't available
            self.setPalette(palette)

    def select_folder(self):
        # self.ui = Ui_MainWindow()
        # self.ui.setupUi(self)
        # self.setCentralWidget(self.ui)

        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.folder_selected = True
            # self.setCentralWidget(self.ui.centralwidget)

            # self.base_layout = QVBoxLayout(self)
            # self.base_layout.setContentsMargins(0, 0, 0, 0)
            # self.base_layout.setSpacing(0)
            # self.view_stackedwidget = QStackedWidget()

            # self.right_panel_stackedwidget.addWidget(self.layers_panel)
            # Set up the right panel
            self.right_panel = QWidget()
            self.right_panel.setObjectName("analysisRightPanel")
            self.right_panel.setFixedWidth(200)  # Set fixed width for the panel

            # Create layout for right panel
            self.right_panel_layout = QVBoxLayout(self.right_panel)
            self.right_panel_layout.setSpacing(10)
            self.right_panel_layout.setContentsMargins(10, 10, 10, 10)

            # Time frame controls
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
                QSlider::groove:horizontal {
                    background: #3a3f4a;
                    height: 8px;
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    background: #85aaff;
                    width: 16px;
                    margin: -4px 0;
                    border-radius: 8px;
                }
                QSlider::handle:horizontal:hover {
                    background: #a0c0ff;
                }
            """
            )
            self.time_slider.valueChanged.connect(self.on_time_slider_changed)
            self.right_panel_layout.addWidget(self.time_slider)

            # Add some spacing
            self.right_panel_layout.addSpacing(20)

            # Button for segmentation
            self.seg_button = QPushButton("Run Segmentation")
            self.seg_button.setObjectName("segButton")
            self.seg_button.clicked.connect(self.emit_segmentation_signal)
            self.right_panel_layout.addWidget(self.seg_button)

            # Button for saving
            self.save_button = QPushButton("Save Results")
            self.save_button.setObjectName("saveButton")
            self.save_button.clicked.connect(self.emit_save_signal)
            self.right_panel_layout.addWidget(self.save_button)

            # Add stretcher to push buttons to the top
            self.right_panel_layout.addStretch()

            # Set up the main display area
            self.display_area_widget = CentralDisplayAreaWidget(self)

            # Create a central widget to hold the layout
            self.central_widget = QWidget()
            self.center_panel_layout = QHBoxLayout(self.central_widget)
            self.center_panel_layout.setSpacing(0)
            self.center_panel_layout.setContentsMargins(0, 0, 0, 0)
            self.center_panel_layout.addWidget(self.display_area_widget)
            self.center_panel_layout.addWidget(self.right_panel)

            # Set the widget containing the layout as central widget
            self.setCentralWidget(self.central_widget)
            self.load_button_signal.emit(folder_path)

    def update_display(self, image_stack):
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
