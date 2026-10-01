import logging
import os
import subprocess
import sys

# torch (via MKL) and faster-whisper's ctranslate2 backend each bundle their
# own Intel OpenMP runtime (libiomp5md.dll); Windows loading both aborts the
# process with "OMP: Error #15". Must be set before either is imported below.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from controllers.chat_controller import ChatController
from controllers.dicom_controller import DICOMController
from model.background_worker import CallableWorker
from model.dicom_model import DICOMModel
from model.voice_input import VOICE_AVAILABLE, TranscriptionWorker
from utils import llm_devices, study_log
from utils.app_config import APP_DIR, data_dir
from view.theme import apply_theme
from view.main_window import MainWindow, load_app_icon
from view.ui_logger import install_ui_logger

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
    icon = load_app_icon()
    if icon is not None:
        app.setWindowIcon(icon)
    apply_theme(app)
    main_window = MainWindow()

    # Belt and braces over llm_devices' atexit hook, which an abrupt Qt exit can skip.
    app.aboutToQuit.connect(llm_devices.shutdown)

    model = DICOMModel()
    controller = DICOMController(main_window, model)

    def _configure_preprocessing():
        model.use_cardisort_cache = main_window.welcome_widget.use_cardisort_cache
        model.use_segmentation_cache = main_window.welcome_widget.use_segmentation_cache

    main_window.welcome_widget.transition_signal.connect(_configure_preprocessing)

    def _start_study_log():
        welcome = main_window.welcome_widget
        if welcome.study_logging:
            study_log.start_session(
                welcome.participant_id, welcome.condition, assistant=not welcome.assistant_disabled,
                cardisort_cache=welcome.use_cardisort_cache,
                segmentation_cache=welcome.use_segmentation_cache, data_folder=str(data_dir()),
            )
            install_ui_logger(main_window)

    def _end_study_log():
        path = study_log.end_session()
        if path is not None:
            # Detached, so the report finishes after the app has closed.
            flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            subprocess.Popen([sys.executable, str(APP_DIR / "tests" / "study" / "study_report.py"), str(path)],
                             creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    main_window.welcome_widget.transition_signal.connect(_start_study_log)
    app.aboutToQuit.connect(_end_study_log)

    # Deferred, not checked at import: the LLM device is picked on the welcome screen.
    def _wire_chat():
        if main_window.welcome_widget.assistant_disabled:
            return
        main_window._chat_controller = ChatController(main_window, controller, model)
        if VOICE_AVAILABLE:
            # Off the GUI thread, so the first push-to-talk isn't a cold model load.
            main_window._voice_preload_worker = CallableWorker(TranscriptionWorker.preload_model)
            main_window._voice_preload_worker.start()

    main_window.welcome_widget.transition_signal.connect(_wire_chat)

    main_window.show()

    if icon is not None:
        # Windows can miss the earlier WM_SETICON calls if the native window
        # handle isn't ready yet — re-apply once the event loop is pumping.
        def _reapply_icon():
            app.setWindowIcon(icon)
            main_window.setWindowIcon(icon)

        QTimer.singleShot(0, _reapply_icon)

    app.exec()


if __name__ == "__main__":
    main()
