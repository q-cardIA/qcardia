import sys

from PySide6.QtWidgets import QFileDialog

from utils.app_config import data_dir


def pick_directory(parent, title: str) -> str:
    """A folder, starting in the data folder; "" when cancelled.

    Qt's own dialog only on macOS, where the native one crashes; on Windows it lists a big folder
    only after many seconds (a shell lookup per entry) and cannot open subfolders.
    """
    options = QFileDialog.ShowDirsOnly
    if sys.platform == "darwin":
        options |= QFileDialog.DontUseNativeDialog
    return QFileDialog.getExistingDirectory(parent, title, str(data_dir()), options)
