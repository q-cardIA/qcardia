"""Save screenshots of the app in its main states, for checking UI changes.

Runs the real app without showing a window: the welcome screen and its options, the
appearance and controls dialogs, one subject loaded, each hanging-protocol preset, a
sample chat, and the same screen in other accents and text sizes. Run from the repo
root, so the segmentation weights resolve:

    uv run tests/ui/screenshots.py
    uv run tests/ui/screenshots.py --subject QLGE05_ --size 1280x800 --out screenshots/after
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

import main as app_main  # noqa: E402
from utils import llm_devices  # noqa: E402
from utils.app_config import SETTINGS, data_dir  # noqa: E402
from utils.hanging_protocols import PRESETS  # noqa: E402
from view import theme  # noqa: E402
from view.controls_dialog import ControlsDialog  # noqa: E402

SETTLE_MS = 500  # time for layouts, thumbnails and fades to finish before a grab
PROBE_TIMEOUT_S = 30


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", default="QLGE01_", help="subject folder inside the data folder")
    parser.add_argument("--size", default="1600x1000", help="window size as WIDTHxHEIGHT")
    parser.add_argument("--out", type=Path, default=REPO / "screenshots", help="output folder")
    parser.add_argument("--accent", choices=list(theme.ACCENTS), help="start with this accent")
    parser.add_argument("--text-size", choices=list(theme.FONT_SIZES), help="start with this text size")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    width, height = (int(v) for v in args.size.lower().split("x"))
    original_appearance = (theme.accent_name(), theme.font_size_name())
    original_docks = SETTINGS.value("window/dock_state")  # Return to Start saves the layout
    SETTINGS.setValue("appearance/accent", args.accent or original_appearance[0])
    SETTINGS.setValue("appearance/font_size", args.text_size or original_appearance[1])
    state = {}

    def window():
        return next(w for w in QApplication.instance().topLevelWidgets() if type(w).__name__ == "MainWindow")

    def save(widget: QWidget, name: str):
        path = args.out / f"{name}.png"
        widget.grab().save(str(path))
        print(f"saved {path.name}")

    def offscreen(widget: QWidget, w: int | None = None, h: int | None = None) -> QWidget:
        widget.setAttribute(Qt.WA_DontShowOnScreen)
        if w:
            widget.resize(w, h)
        QWidget.show(widget)
        return widget

    # Each step does one thing, then the next runs after SETTLE_MS; a step returning
    # False waits for something else (a load) to call next_step itself.
    def shot(name: str):
        return lambda: save(window(), name)

    def welcome():
        offscreen(window(), width, height)
        save(window(), "01_welcome")

    def welcome_options():
        row = window().welcome_widget._button_row
        state["row_y"] = row.mapTo(window(), row.rect().topLeft()).y()
        window().welcome_widget._chevron_btn.click()

    def welcome_options_shot():
        save(window(), "02_welcome_options")
        print("welcome button row moved on open:", state.get("row_y"), "->",
              window().welcome_widget._button_row.mapTo(window(), window().welcome_widget._button_row.rect().topLeft()).y())

    def dialogs():
        save(offscreen(theme.AppearanceDialog(window())), "03_appearance_dialog")
        save(offscreen(ControlsDialog(window())), "04_controls_dialog")

    def enter_analysis():
        window().welcome_widget.left_panel_single_patient_pushbutton.click()

    def load():
        state["controller"].load_patient(data_dir() / args.subject, on_ready=on_loaded)
        return False

    def on_loaded(success: bool):
        if not success:
            print(f"could not load {args.subject}")
            finish()
            return
        if state["controller"].loading_series:  # the shots need all series
            QTimer.singleShot(100, lambda: on_loaded(True))
            return
        next_step()

    def loaded_shot():
        save(window(), "06_loaded")
        w = window()
        print("shortcuts:", {a.text(): a.shortcut().toString() for a in (w.toggle_sidebar_action, w.toggle_series_browser_action,
              w.toggle_chat_action, w.full_screen_action, w.contour_edit_action)})
        print("patient line:", w.sidebar._patient_label.text())

    def preset_step(name: str):
        def run():
            result = state["controller"].apply_preset(name)
            print(f"preset {name}: {result[0]}/{result[1]} panels filled" if result else f"preset {name}: unknown")
        return run

    def preset_shot(name: str):
        return shot(f"07_preset_{name}")

    def focus_and_chat():
        state["controller"].apply_preset("functional")
        window().activateWindow()
        window().display_area_widget._panels[0].viewer.setFocus(Qt.MouseFocusReason)
        chat = getattr(window(), "chat_widget", None)
        if chat is None or not chat.isVisible():
            return
        chat._add_bubble("user", "Load QLGE01_ and show the functional layout")
        chat.append_thinking_chunk("The user wants a subject loaded and a layout applied, so two tool calls.")
        chat.begin_assistant_message()
        chat.append_tool_call("load_subject", {"subject_id": args.subject}).set_result("Loaded 8 series")
        chat.append_tool_call("apply_preset", {"preset": "functional"}).set_result("Filled 4 of 4 panels")
        chat.append_tool_call("run_segmentation", {"modality": "LGE"}).set_result("Segmentation failed: model not found")
        chat.append_assistant_chunk(f"I loaded **{args.subject}** and applied the *functional* layout: "
                                    "the long-axis CINE views and the short-axis stack.")
        chat.finalize_assistant_message()
        chat.append_system_message("Model switched to qwen3.5")
        chat._input.setPlainText("Then segment the short-axis stack and tell me the ejection fraction "
                                 "and the end-diastolic volume of the left ventricle, please.")
        chat._rebuild_model_menu()
        print("model menu:", [a.text() or "---" for a in chat._model_menu.actions()])

    def grid_without_crosshair():
        window()._rebuild_predict_menu()
        print("predict menu:", [(a.text(), a.isEnabled()) for a in window().predict_menu.actions() if a.text()])
        window().display_area_widget.set_grid_size(1, 3)
        window().display_area_widget.set_crosshair_shown(False)

    def grid_shot():
        save(window(), "13_grid_1x3_no_crosshair")
        window().display_area_widget.set_crosshair_shown(True)
        state["controller"].apply_preset("functional")

    def segment():
        state["controller"].run_segmentation("CINE_SAX")

    def wait_for_segmentation():
        waited = [0]

        def poll():
            controller = state["controller"]
            if controller._seg_worker is None and not controller._seg_queue or waited[0] > 600:
                next_step()
                return
            waited[0] += 1
            QTimer.singleShot(500, poll)
        poll()
        return False

    def edit_contours():
        window().contour_edit_action.toggle()

    def edit_contours_shot():
        save(window(), "12_edit_contours")
        window().contour_edit_action.toggle()

    def measure_resize_and_drag():
        import time
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QMouseEvent
        w = window()
        w.reset_panel_layout()
        QApplication.processEvents()
        browser = w.series_browser_widget
        t0 = time.perf_counter()
        for width in range(220, 420, 10):
            w.resizeDocks([w.series_dock], [width], Qt.Horizontal)
            QApplication.processEvents()
            browser.repaint()
        print(f"series resize: {(time.perf_counter() - t0) / 20 * 1000:.1f} ms per step")
        t0 = time.perf_counter()
        for _ in range(20):
            browser._fit_cards()
            browser.repaint()
        print(f"  of which the Series panel itself: {(time.perf_counter() - t0) / 20 * 1000:.1f} ms")
        t0 = time.perf_counter()
        for _ in range(20):
            w.display_area_widget.repaint()
        print(f"  and a repaint of the images: {(time.perf_counter() - t0) / 20 * 1000:.1f} ms")
        def drag(dock, dx):
            w.resizeDocks([w.series_dock, w.chat_dock], [220, 280], Qt.Horizontal)
            QApplication.processEvents()
            before = (w.series_dock.width(), w.chat_dock.width())
            x, y = dock.geometry().left() - 3, dock.geometry().center().y()  # the gap left of that panel
            for kind, px, buttons in ((QMouseEvent.MouseButtonPress, x, Qt.LeftButton),
                                      (QMouseEvent.MouseMove, x + dx // 2, Qt.LeftButton),
                                      (QMouseEvent.MouseMove, x + dx, Qt.LeftButton),
                                      (QMouseEvent.MouseButtonRelease, x + dx, Qt.NoButton)):
                pos = QPointF(px, y)
                QApplication.sendEvent(w, QMouseEvent(kind, pos, w.mapToGlobal(pos), Qt.LeftButton, buttons, Qt.NoModifier))
                QApplication.processEvents()
            after = (w.series_dock.width(), w.chat_dock.width())
            print(f"drag left edge of {dock.windowTitle()} by {dx}: series {before[0]} -> {after[0]}, "
                  f"assistant {before[1]} -> {after[1]}")
        drag(w.series_dock, -120)
        drag(w.chat_dock, -120)
        drag(w.chat_dock, 60)

    def assistant_under_patient():
        w = window()
        w.splitDockWidget(w.sidebar_dock, w.chat_dock, Qt.Vertical)

    def merged_tabs():
        w = window()
        w.reset_panel_layout()
        w.tabifyDockWidget(w.series_dock, w.chat_dock)
        w.chat_dock.raise_()

    def tabs_shot():
        save(window(), "15_series_and_assistant_as_tabs")
        w = window()
        w.series_dock.raise_()  # the first tab
        QApplication.processEvents()
        w._update_dock_titles()
        QApplication.processEvents()
        print("first tab selected, series top:", w.series_browser_widget.property("top"),
              "| assistant top:", w.chat_widget.property("top"))
        save(w, "15b_first_tab_selected")
        w.chat_dock.toggleViewAction().trigger()  # View > Assistant off, while tabbed with Series
        QApplication.processEvents()
        w._update_dock_titles()
        print("tab closed: series has its title bar again:",
              w.series_dock.titleBarWidget() is w._dock_title_bars[w.series_dock][0])
        w.chat_dock.toggleViewAction().trigger()
        w.reset_panel_layout()

    def back_to_start():
        w, controller = window(), state["controller"]
        shown = []
        w.show_info = lambda title, text: shown.append(text)  # no modal dialog in a script
        controller._load_worker = object()  # as if a load were still running
        w.return_to_start()
        print("return refused while busy:", bool(shown) and hasattr(w, "sidebar_dock") and w.sidebar_dock is not None)
        controller._load_worker = None
        w.toggle_cine()
        w.return_to_start()
        del w.show_info
        esc = [a for a in w.actions() if a.shortcut().toString() == "Esc"]
        print("after return: series left", len(controller.model.series), "| cine timer running", w._play_timer.isActive(),
              "| Esc actions", len(esc))

    def back_shot():
        save(window(), "16_back_at_start")
        start_button = window().welcome_widget.left_panel_single_patient_pushbutton
        waited = [0]

        def click_when_ready():  # the new welcome screen probes the device again first
            if not start_button.isEnabled() and waited[0] < 60:
                waited[0] += 1
                QTimer.singleShot(500, click_when_ready)
                return
            start_button.click()
            QTimer.singleShot(SETTLE_MS, next_step)
        click_when_ready()
        return False

    def second_session_shot():
        print("second session Esc actions:", len([a for a in window().actions() if a.shortcut().toString() == "Esc"]))
        chat = getattr(window(), "_chat_controller", None)
        print("second session: chat controller", type(chat).__name__ if chat else None,
              "| docks", [d.objectName() for d in (window().sidebar_dock, window().series_dock, window().chat_dock)])
        save(window(), "17_second_session")

    def appearance(accent: str, size: str):
        return lambda: theme.set_appearance(accent, size)

    def appearance_shot(accent: str, size: str):
        return shot(f"09_appearance_{accent.lower()}_{size.lower().replace(' ', '_')}")

    steps = [welcome, welcome_options, welcome_options_shot, dialogs, enter_analysis, shot("05_analysis_empty"),
             load, loaded_shot]
    for name in PRESETS:
        steps += [preset_step(name), preset_shot(name)]
    steps += [focus_and_chat, shot("08_focus_and_chat"), grid_without_crosshair, grid_shot,
              measure_resize_and_drag, assistant_under_patient, shot("14_assistant_under_patient"), merged_tabs, tabs_shot, segment, shot("10_segmenting"), wait_for_segmentation, shot("11_contours"),
              edit_contours, edit_contours_shot]
    steps += [back_to_start, back_shot, second_session_shot]
    size = theme.font_size_name()
    for accent in ("Purple", "Amber"):
        steps += [appearance(accent, size), appearance_shot(accent, size)]
    queue = list(steps)

    def next_step():
        if not queue:
            finish()
            return
        step = queue.pop(0)
        if step() is not False:
            QTimer.singleShot(SETTLE_MS, next_step)

    def finish():
        QApplication.instance().quit()  # the finally block below restores the appearance

    waited = [0.0]

    def start():
        # The chat panel only appears once the device probe has answered.
        if not llm_devices.chat_available() and waited[0] < PROBE_TIMEOUT_S:
            waited[0] += 0.5
            QTimer.singleShot(500, start)
            return
        next_step()

    original_init = app_main.DICOMController.__init__

    def keep_controller(self, *a, **kw):
        original_init(self, *a, **kw)
        state["controller"] = self

    app_main.DICOMController.__init__ = keep_controller

    # main() shows the window itself; mark it first so it never appears on screen.
    app_main.MainWindow.show = lambda self: offscreen(self, width, height)

    original_exec = QApplication.exec

    def exec_with_script(*_):
        QTimer.singleShot(0, start)
        return original_exec()

    QApplication.exec = exec_with_script
    try:
        app_main.main()
    finally:
        SETTINGS.setValue("appearance/accent", original_appearance[0])
        SETTINGS.setValue("appearance/font_size", original_appearance[1])
        if original_docks is None:
            SETTINGS.remove("window/dock_state")
        else:
            SETTINGS.setValue("window/dock_state", original_docks)
        SETTINGS.sync()  # the event loop has ended, so nothing else would write this to disk


if __name__ == "__main__":
    main()
