from __future__ import annotations

import sys
from importlib import import_module


def main() -> int:
    try:
        qtcore = import_module("PySide6.QtCore")
        qtwidgets = import_module("PySide6.QtWidgets")
    except ModuleNotFoundError:
        print("PySide6 is required. Install it with: pip install PySide6")
        return 1

    Qt = qtcore.Qt
    QApplication = qtwidgets.QApplication
    QCheckBox = qtwidgets.QCheckBox
    QComboBox = qtwidgets.QComboBox
    QFrame = qtwidgets.QFrame
    QGridLayout = qtwidgets.QGridLayout
    QGroupBox = qtwidgets.QGroupBox
    QHBoxLayout = qtwidgets.QHBoxLayout
    QLabel = qtwidgets.QLabel
    QLineEdit = qtwidgets.QLineEdit
    QListWidget = qtwidgets.QListWidget
    QMainWindow = qtwidgets.QMainWindow
    QPushButton = qtwidgets.QPushButton
    QProgressBar = qtwidgets.QProgressBar
    QRadioButton = qtwidgets.QRadioButton
    QSlider = qtwidgets.QSlider
    QSpinBox = qtwidgets.QSpinBox
    QTabWidget = qtwidgets.QTabWidget
    QTableWidget = qtwidgets.QTableWidget
    QTableWidgetItem = qtwidgets.QTableWidgetItem
    QTextEdit = qtwidgets.QTextEdit
    QVBoxLayout = qtwidgets.QVBoxLayout
    QWidget = qtwidgets.QWidget

    from pyside6_theme import apply_miraviewer_theme, set_variant

    class DemoWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("MiraViewer Theme Preview (PySide6)")
            self.resize(1080, 760)

            self._build_menu()

            root = QWidget()
            self.setCentralWidget(root)
            root_layout = QVBoxLayout(root)
            root_layout.setContentsMargins(16, 16, 16, 16)
            root_layout.setSpacing(12)

            header = self._build_header_panel()
            root_layout.addWidget(header)

            controls = self._build_controls_panel()
            root_layout.addWidget(controls)

            tabs = self._build_tabs()
            root_layout.addWidget(tabs, stretch=1)

            self.statusBar().showMessage("Theme preview loaded")

        def _build_menu(self) -> None:
            file_menu = self.menuBar().addMenu("File")
            file_menu.addAction("Import")
            file_menu.addAction("Export")
            file_menu.addSeparator()
            file_menu.addAction("Exit", self.close)

            help_menu = self.menuBar().addMenu("Help")
            help_menu.addAction("Keyboard Shortcuts")
            help_menu.addAction("About")

        def _build_header_panel(self):
            panel = QFrame()
            panel.setProperty("role", "panel")

            layout = QHBoxLayout(panel)
            layout.setContentsMargins(14, 12, 14, 12)
            layout.setSpacing(10)

            title_stack = QVBoxLayout()
            title_stack.setSpacing(2)
            title = QLabel("MiraViewer style preview")
            subtitle = QLabel("Dark cards, muted text, accent CTAs, and border rhythm")
            subtitle.setProperty("textRole", "secondary")
            accent_hint = QLabel("Accent: #3b82f6")
            accent_hint.setProperty("textRole", "accent")
            title_stack.addWidget(title)
            title_stack.addWidget(subtitle)
            title_stack.addWidget(accent_hint)

            actions = QHBoxLayout()
            actions.setSpacing(8)

            import_button = QPushButton("Import")
            set_variant(import_button, "primary")

            help_button = QPushButton("Help")
            set_variant(help_button, "ghost")

            danger_button = QPushButton("Delete all")
            set_variant(danger_button, "danger")

            disabled_button = QPushButton("Disabled")
            disabled_button.setEnabled(False)

            actions.addWidget(import_button)
            actions.addWidget(help_button)
            actions.addWidget(danger_button)
            actions.addWidget(disabled_button)

            layout.addLayout(title_stack, stretch=1)
            layout.addLayout(actions)
            return panel

        def _build_controls_panel(self):
            panel = QFrame()
            panel.setProperty("role", "panel")

            layout = QGridLayout(panel)
            layout.setContentsMargins(14, 12, 14, 12)
            layout.setHorizontalSpacing(10)
            layout.setVerticalSpacing(10)

            name_label = QLabel("Patient name")
            name_label.setProperty("textRole", "secondary")
            name_input = QLineEdit()
            name_input.setPlaceholderText("e.g. Jane Doe")

            sequence_label = QLabel("Sequence")
            sequence_label.setProperty("textRole", "secondary")
            sequence_combo = QComboBox()
            sequence_combo.addItems(["T1", "T2", "FLAIR", "DWI"])

            slice_label = QLabel("Slice")
            slice_label.setProperty("textRole", "secondary")
            slice_spinner = QSpinBox()
            slice_spinner.setRange(1, 240)
            slice_spinner.setValue(93)

            zoom_label = QLabel("Zoom")
            zoom_label.setProperty("textRole", "secondary")
            zoom_slider = QSlider(Qt.Orientation.Horizontal)
            zoom_slider.setRange(25, 400)
            zoom_slider.setValue(140)

            zoom_value = QLabel("140%")
            zoom_value.setProperty("textRole", "tertiary")
            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(35)

            def update_zoom(value: int) -> None:
                zoom_value.setText(f"{value}%")
                progress.setValue(min(100, max(0, (value - 25) // 3)))

            zoom_slider.valueChanged.connect(update_zoom)

            options_box = QGroupBox("Options")
            options_layout = QVBoxLayout(options_box)
            options_layout.setSpacing(6)
            options_layout.addWidget(QCheckBox("Reverse slice order"))
            options_layout.addWidget(QCheckBox("Enable overlay loop"))
            options_layout.addWidget(QRadioButton("Grid view"))
            options_layout.addWidget(QRadioButton("Overlay view"))

            layout.addWidget(name_label, 0, 0)
            layout.addWidget(name_input, 0, 1)
            layout.addWidget(sequence_label, 0, 2)
            layout.addWidget(sequence_combo, 0, 3)

            layout.addWidget(slice_label, 1, 0)
            layout.addWidget(slice_spinner, 1, 1)
            layout.addWidget(zoom_label, 1, 2)
            layout.addWidget(zoom_slider, 1, 3)

            layout.addWidget(QLabel("Import progress"), 2, 0)
            layout.addWidget(progress, 2, 1, 1, 2)
            layout.addWidget(zoom_value, 2, 3)

            layout.addWidget(options_box, 0, 4, 3, 1)
            layout.setColumnStretch(1, 1)
            layout.setColumnStretch(3, 1)

            return panel

        def _build_tabs(self):
            tabs = QTabWidget()

            table = QTableWidget(4, 3)
            table.setHorizontalHeaderLabels(["Date", "Sequence", "Slices"])
            rows = [
                ("2026-02-11", "T1", "188"),
                ("2025-12-02", "FLAIR", "176"),
                ("2025-08-20", "T2", "184"),
                ("2025-03-17", "DWI", "120"),
            ]
            for row_index, row in enumerate(rows):
                for col_index, value in enumerate(row):
                    table.setItem(
                        row_index,
                        col_index,
                        QTableWidgetItem(value),
                    )

            list_widget = QListWidget()
            list_widget.addItems(
                [
                    "Use wheel for slice navigation",
                    "Press Space for quick A/B compare",
                    "Drag rectangle to set exclusion region",
                    "Use primary button for key actions",
                ]
            )

            notes = QTextEdit()
            notes.setPlainText(
                "This tab demonstrates text-edit styling.\n"
                "\n"
                "The goal of this demo is visual parity with\n"
                "the React viewer:\n"
                "dark surfaces, subtle borders, muted typography,\n"
                "and blue accent controls."
            )

            tabs.addTab(table, "Studies")
            tabs.addTab(list_widget, "Tips")
            tabs.addTab(notes, "Notes")
            return tabs

    app = QApplication(sys.argv)
    apply_miraviewer_theme(app)

    window = DemoWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
