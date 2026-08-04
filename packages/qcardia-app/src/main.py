import logging
import sys

from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from controllers.chat_controller import ChatController
from controllers.dicom_controller import DICOMController
from model.dicom_model import DICOMModel
from utils.app_config import CHAT_AVAILABLE, ICON_PATH
from view.theme import apply_theme
from view.main_window import MainWindow

logging.basicConfig(level=logging.INFO)


def _set_windows_taskbar_icon():
    """Without an explicit AppUserModelID, Windows groups this process under
    python.exe and shows its icon in the taskbar/alt-tab instead of ours."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("qcardia.app.viewer")
    except Exception:
        pass


def main():
    _set_windows_taskbar_icon()
    app = QApplication()
    if ICON_PATH.is_file():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    apply_theme(app)
    main_window = MainWindow()

    model = DICOMModel()
    controller = DICOMController(main_window, model)

    if CHAT_AVAILABLE:
        # chat_widget only exists after build_analysis_interface() runs, which
        # itself only fires once the welcome screen's transition_signal fires.
        # assistant_disabled reflects the welcome screen's opt-out toggle,
        # read at that same point (its checked state isn't final until then).
        def _wire_chat():
            if main_window.welcome_widget.assistant_disabled:
                return
            main_window._chat_controller = ChatController(main_window, controller, model)

        main_window.welcome_widget.transition_signal.connect(_wire_chat)

    main_window.show()

    if ICON_PATH.is_file():
        # Windows' taskbar button is created from python.exe's own icon the
        # moment the window first appears; the WM_SETICON calls above (made
        # before/at construction, while heavy imports like torch/tf-keras are
        # still loading) can be issued before the native window handle is
        # ready to receive them and get missed. Re-apply once the event loop
        # is actually pumping messages so Explorer picks up the real icon
        # immediately instead of only after some later repaint.
        def _reapply_icon():
            icon = QIcon(str(ICON_PATH))
            app.setWindowIcon(icon)
            main_window.setWindowIcon(icon)

        QTimer.singleShot(0, _reapply_icon)

    app.exec()


if __name__ == "__main__":
    main()
