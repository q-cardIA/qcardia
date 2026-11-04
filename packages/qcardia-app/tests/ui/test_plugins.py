"""The plugin hooks: a fake plugin adds a start-screen option and a panel, and is told when the
session starts and ends. Runs without a display (QT_QPA_PLATFORM=offscreen)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDockWidget, QLabel  # noqa: E402

from qcardia_app.plugins import AppContext, Plugin, load_plugins  # noqa: E402
from qcardia_app.view.main_window import MainWindow  # noqa: E402
from qcardia_app.view.theme import apply_theme  # noqa: E402


class FakePlugin(Plugin):
    name = "fake"

    def __init__(self):
        self.events = []
        self.busy = None

    def welcome_options(self, parent):
        self.events.append("welcome_options")
        return QLabel("Fake option", parent)

    def start_session(self, ctx):
        self.events.append("start_session")
        ctx.window.add_panel(QLabel("Fake panel"), "Fake", "fakeDock", "&Fake", width=200)

    def end_session(self):
        self.events.append("end_session")

    def busy_reason(self):
        return self.busy


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    return app


def docks(window):
    return sorted(d.objectName() for d in window.findChildren(QDockWidget))


def test_load_plugins_returns_plugins_only():
    assert all(isinstance(p, Plugin) for p in load_plugins())


def test_window_without_plugins(app):
    window = MainWindow()
    window._start_session()
    app.processEvents()
    assert docks(window) == ["patientDock", "resultsDock", "seriesDock"]


def test_plugin_hooks(app):
    plugin = FakePlugin()
    window = MainWindow([plugin])
    assert plugin.events == ["welcome_options"]
    context = AppContext(window, controller=None, model=None)
    window.session_started.connect(lambda: plugin.start_session(context))
    window.session_ended.connect(plugin.end_session)
    window.session_busy = plugin.busy_reason

    window.welcome_widget.left_panel_single_patient_pushbutton.click()
    app.processEvents()
    assert "fakeDock" in docks(window)
    assert "&Fake" in [a.text() for a in window._view_menu.actions()]

    window.return_to_start()
    app.processEvents()
    assert plugin.events == ["welcome_options", "start_session", "end_session", "welcome_options"]


def test_hold_start(app):
    window = MainWindow()
    start = window.welcome_widget.left_panel_single_patient_pushbutton
    window.welcome_widget.hold_start("a", True)
    window.welcome_widget.hold_start("b", True)
    window.welcome_widget.hold_start("a", False)
    assert not start.isEnabled()
    window.welcome_widget.hold_start("b", False)
    assert start.isEnabled()


_QUIT_SCRIPT = '''
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
import qcardia_app.main as app_main
from qcardia_app.plugins import Plugin

calls = []

class Recorder(Plugin):
    name = "recorder"
    def start_session(self, ctx): calls.append("start_session")
    def end_session(self): calls.append("end_session")
    def session_log_closed(self, path, study): calls.append("session_log_closed")
    def shutdown(self): calls.append("shutdown")

app_main.load_plugins = lambda: [Recorder()]
exec_ = QApplication.exec

def scripted_exec(self):
    window = next(w for w in self.topLevelWidgets() if type(w).__name__ == "MainWindow")
    QTimer.singleShot(0, window.welcome_widget.left_panel_single_patient_pushbutton.click)
    QTimer.singleShot(300, window.close)  # quit during the session
    QTimer.singleShot(600, self.quit)
    return exec_()

QApplication.exec = scripted_exec
app_main.main()
print("CALLS", ",".join(calls))
'''


def test_quit_during_session_order(tmp_path):
    """Closing the window during a session ends it for plugins, and shutdown() comes last."""
    import subprocess
    import sys

    script = tmp_path / "quit.py"
    script.write_text(_QUIT_SCRIPT, encoding="utf-8")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
    line = next((ln for ln in result.stdout.splitlines() if ln.startswith("CALLS ")), "")
    calls = line.removeprefix("CALLS ").split(",")
    assert calls == ["start_session", "session_log_closed", "end_session", "shutdown"], result.stderr[-2000:]
