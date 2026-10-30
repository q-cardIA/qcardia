import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog, QHeaderView, QLineEdit, QPushButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from qcardia_app.utils.app_config import data_dir, list_subjects, patients


def pick_directory(parent, title: str) -> str:
    """A folder, starting in the data folder; "" when cancelled.

    Qt's own dialog only on macOS, where the native one crashes; on Windows it lists a big folder
    only after many seconds (a shell lookup per entry) and cannot open subfolders.
    """
    options = QFileDialog.ShowDirsOnly
    if sys.platform == "darwin":
        options |= QFileDialog.DontUseNativeDialog
    return QFileDialog.getExistingDirectory(parent, title, str(data_dir()), options)


def pick_patient(parent) -> str:
    """A patient folder, chosen by the name or hospital ID of the patient list; "" when cancelled. Without a
    patient list (or with Browse folder), the folder dialog."""
    listed = [p for p in patients(list_subjects()) if p.name]
    if not listed:
        return pick_directory(parent, "Select Patient Folder")
    dialog = QDialog(parent)
    dialog.setWindowTitle("Load Patient")
    search = QLineEdit(dialog)
    search.setPlaceholderText("Name or hospital ID")
    found = QTreeWidget(dialog)  # two columns: the name at the left border, the ID at the right one
    found.setColumnCount(2)
    found.setHeaderHidden(True)
    found.setRootIsDecorated(False)
    found.header().setStretchLastSection(False)
    found.header().setSectionResizeMode(0, QHeaderView.Stretch)
    found.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
    items = []
    for p in sorted(listed, key=lambda p: p.name):
        items.append(QTreeWidgetItem([p.name, p.hospital_id]))
        items[-1].setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
        items[-1].setData(0, Qt.UserRole, str(data_dir() / p.folder))
    found.addTopLevelItems(items)

    def show_matches(text: str):
        words = text.lower().split()
        for item in items:
            item.setHidden(not all(w in f"{item.text(0)} {item.text(1)}".lower() for w in words))
        found.setCurrentItem(next((item for item in items if not item.isHidden()), None))

    chosen = {}

    def browse():
        path = pick_directory(dialog, "Select Patient Folder")
        if path:
            chosen["path"] = path
            dialog.accept()

    search.textChanged.connect(show_matches)
    found.itemDoubleClicked.connect(lambda *_: dialog.accept())
    buttons = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel, dialog)
    folder_button = QPushButton("Browse folder…", dialog)
    buttons.addButton(folder_button, QDialogButtonBox.ResetRole)
    folder_button.clicked.connect(browse)
    buttons.accepted.connect(lambda: found.currentItem() and dialog.accept())
    search.returnPressed.connect(buttons.accepted)
    buttons.rejected.connect(dialog.reject)
    layout = QVBoxLayout(dialog)
    for widget in (search, found, buttons):
        layout.addWidget(widget)
    dialog.resize(420, 480)
    show_matches("")
    search.setFocus()
    if dialog.exec() != QDialog.Accepted:
        return ""
    if "path" in chosen:
        return chosen["path"]
    return found.currentItem().data(0, Qt.UserRole)
