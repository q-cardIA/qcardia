import logging
import os
import sys

# torch (via MKL) and other libraries a plugin may load can each bundle their
# own Intel OpenMP runtime (libiomp5md.dll); Windows loading two aborts the
# process with "OMP: Error #15". Must be set before torch is imported below.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from qcardia_app.controllers.dicom_controller import DICOMController
from qcardia_app.model.dicom_model import DICOMModel
from qcardia_app.plugins import AppContext, load_plugins
from qcardia_app.utils import study_log
from qcardia_app.utils.app_config import patient_source
from qcardia_app.view.theme import apply_theme
from qcardia_app.view.main_window import MainWindow, load_app_icon
from qcardia_app.view.ui_logger import install_ui_logger

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
    plugins = load_plugins()
    main_window = MainWindow(plugins)
    for plugin in plugins:
        app.aboutToQuit.connect(plugin.shutdown)

    model = DICOMModel()
    controller = DICOMController(main_window, model)
    context = AppContext(main_window, controller, model)

    def _configure_preprocessing():
        model.use_cardisort_cache = main_window.welcome_widget.use_cardisort_cache
        model.use_segmentation_cache = main_window.welcome_widget.use_segmentation_cache

    main_window.session_started.connect(_configure_preprocessing)
    main_window.session_ended.connect(controller.reset_session)

    def _session_busy() -> str | None:
        if controller.is_busy():
            return "A patient is still loading or being segmented."
        return next((reason for plugin in plugins if (reason := plugin.busy_reason())), None)

    main_window.session_busy = _session_busy

    study = []  # whether the running session is a named study session, which gets a report

    def _start_study_log():
        # Always a session log: named in a study session, else a backup without a name.
        welcome = main_window.welcome_widget
        study[:] = [welcome.study_logging]
        study_log.start_session(
            *((welcome.participant_id, welcome.condition) if welcome.study_logging else ("", "")),
            study=welcome.study_logging,
            cardisort_cache=welcome.use_cardisort_cache,
            segmentation_cache=welcome.use_segmentation_cache, patients=str(patient_source()),
            plugins={plugin.name: plugin.session_info() for plugin in plugins},
        )
        install_ui_logger(main_window)

    def _end_study_log():
        path = study_log.end_session()
        if path is not None:
            for plugin in plugins:
                plugin.session_log_closed(path, study[0])

    main_window.session_started.connect(_start_study_log)
    main_window.session_ended.connect(_end_study_log)
    app.aboutToQuit.connect(_end_study_log)

    # After the analysis screen is built, while the start screen's choices are still readable.
    def _start_plugins():
        for plugin in plugins:
            plugin.start_session(context)

    def _end_plugins():
        for plugin in plugins:
            plugin.end_session()

    main_window.session_started.connect(_start_plugins)
    main_window.session_ended.connect(_end_plugins)

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
