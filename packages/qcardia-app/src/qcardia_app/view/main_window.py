from __future__ import annotations

import math


from PySide6.QtCore import QEvent, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QGuiApplication, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QDockWidget,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSlider,
    QStyle,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from qcardia_app.utils import study_log
from qcardia_app.utils.app_config import ICON_PATH, ISSUES_URL, REPO_URL, SETTINGS, WIKI_URL
from qcardia_app.utils.layouts import CINE_SPEEDS
from qcardia_app.view.central_display_area_widget import CentralDisplayAreaWidget
from qcardia_app.view.dialogs import pick_patient
from qcardia_app.view.loading_overlay import LoadingOverlay
from qcardia_app.view.results_widget import ResultsWidget
from qcardia_app.view.series_browser_widget import SeriesBrowserWidget
from qcardia_app.view.sidebar_widget import SidebarWidget
from qcardia_app.view import theme
from qcardia_app.view.theme import StickyMenu, segment_group
from qcardia_app.view.welcome_widget import WelcomeWidget


def load_app_icon() -> QIcon | None:
    return QIcon(str(ICON_PATH)) if ICON_PATH.is_file() else None


_TAB_DRAG_OUT_PX = 24  # how far past its tab bar a tab must be dragged to unstack its panel
GRID_SIZES = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5))  # rows x columns


class _ContourIsland(QFrame):

    tool_changed = Signal(str)
    save_requested = Signal()
    done_requested = Signal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("contourIsland")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)
        self._tool_buttons = []
        for tool, label in (("move", "Move"), ("add", "Add Point"), ("remove", "Remove Point")):
            button = QPushButton(label)
            button.toggled.connect(lambda on, t=tool: on and self.tool_changed.emit(t))
            self._tool_buttons.append(button)
        layout.addWidget(segment_group(self._tool_buttons))
        self._tool_buttons[0].setChecked(True)
        layout.addSpacing(8)
        save = QPushButton("Save Labels")
        save.clicked.connect(self.save_requested.emit)
        done = QPushButton("Done")
        done.setProperty("role", "primary")
        done.clicked.connect(self.done_requested.emit)
        layout.addWidget(save)
        layout.addWidget(done)
        parent.installEventFilter(self)
        self.hide()

    def reset(self):
        self._tool_buttons[0].setChecked(True)

    def showEvent(self, event):
        super().showEvent(event)
        self._place()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Resize:
            self._place()
        return False

    def _place(self):
        self.adjustSize()
        self.move((self.parentWidget().width() - self.width()) // 2, 10)
        self.raise_()


class _DockTitle(QWidget):
    """Dock title bar: the panel's name and a close button. Mouse events it ignores reach the
    dock, so dragging the bar still moves, stacks or tabs the panel."""

    def __init__(self, dock: QDockWidget):
        super().__init__(dock)
        self.setProperty("role", "dockTitle")
        self.setAttribute(Qt.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 8, 4)
        name = QLabel(dock.windowTitle())
        name.setProperty("role", "dockTitleLabel")
        close = QPushButton()
        close.setProperty("role", "dockClose")
        close.setIcon(theme.line_icon("close", "text_faint"))
        close.setIconSize(QSize(12, 12))
        name.setFixedHeight(22)  # the close button's height, so both share a centre line
        name.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        close.setToolTip(f"Hide {dock.windowTitle()} (View menu shows it again)")
        close.clicked.connect(dock.close)
        layout.addWidget(name)
        layout.addStretch(1)
        layout.addWidget(close)
        self.setCursor(Qt.OpenHandCursor)


class MainWindow(QMainWindow):

    load_patient_button_signal = Signal(str)
    seg_button_signal = Signal(str)
    save_button_signal = Signal()
    time_frame_changed_signal = Signal(int)
    preset_requested_signal = Signal(str)
    session_started = Signal()  # the analysis screen is built; the welcome screen's choices are still readable
    session_ended = Signal()
    reclassify_requested_signal = Signal(object, str)
    zoom_requested = Signal(str)  # "roi", "fill" or "fit", in every panel
    contours_requested = Signal(bool)  # contours on or off in every panel
    window_requested = Signal(str)  # "null_myocardium" (LGE_SAX) or "reset" (every series)
    phase_requested = Signal(str)  # "ED" or "ES"
    panels_moved = Signal(object)  # see CentralDisplayAreaWidget.panels_moved

    def __init__(self, plugins: list | tuple = ()):
        super().__init__()
        self.setObjectName("mainWindow")
        self.plugins = list(plugins)  # qcardia_app.plugins.Plugin instances
        icon = load_app_icon()
        if icon is not None:
            self.setWindowIcon(icon)
        self.segmentable_series: tuple[str, ...] = ()  # set by DICOMController
        self._default_dock_state = None  # set while a session's docks exist
        self._dock_title_bars: dict[QDockWidget, tuple[QWidget, QWidget]] = {}
        self._edge_drag = None
        self._tab_drag: int | None = None
        # One deferred update per burst of dock moves, visibility changes and tab switches.
        self._dock_titles_timer = QTimer(self)
        self._dock_titles_timer.setSingleShot(True)
        self._dock_titles_timer.timeout.connect(self._update_dock_titles)
        self.session_busy = lambda: None  # set by main: why the session cannot end yet, or None
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)
        self._cine_speed, self._perfusion_speed, self._play_step = 1.0, 1.0, 1
        self._play_range: tuple[int, int] | None = None  # (first, last) frame of a loop, see play_frames
        # One state record per burst of user input.
        self._screen_log_timer = QTimer(self, singleShot=True, interval=300, timeout=self.log_screen)
        self.full_screen_action = QAction("&Full Screen", self)
        self.full_screen_action.setCheckable(True)
        self.full_screen_action.setShortcut("F11")
        self.full_screen_action.toggled.connect(lambda on: self.showFullScreen() if on else self.showMaximized())
        # Esc leaves full screen; only enabled there, so everywhere else Esc still reaches the widgets.
        exit_full_screen = QAction(self)
        exit_full_screen.setShortcut(Qt.Key_Escape)
        exit_full_screen.setEnabled(False)
        exit_full_screen.triggered.connect(lambda: self.full_screen_action.setChecked(False))
        self.full_screen_action.toggled.connect(exit_full_screen.setEnabled)
        self.addActions([self.full_screen_action, exit_full_screen])  # also on the start screen
        self.welcome_widget = WelcomeWidget(self, self.plugins)
        self.setCentralWidget(self.welcome_widget)
        self.welcome_widget.transition_signal.connect(self._start_session)
        self.setWindowState(Qt.WindowMaximized)

    def build_analysis_interface(self):
        self.sidebar = SidebarWidget(self)
        self.sidebar.load_patient_requested.connect(self.on_load_patient_clicked)
        self.sidebar.preset_requested.connect(self.preset_requested_signal.emit)

        self.display_area_widget = CentralDisplayAreaWidget(self)
        self.loading_overlay = LoadingOverlay(self.display_area_widget)
        self.series_browser_widget = SeriesBrowserWidget(self)
        self.series_browser_widget.reclassify_requested.connect(
            self.reclassify_requested_signal.emit
        )
        screen_width = QGuiApplication.primaryScreen().availableGeometry().width()
        self._plugin_docks: list[tuple[QDockWidget, int | None]] = []  # see add_panel
        self.results_widget = ResultsWidget(self)

        time_controls_container = QWidget()
        time_layout = QHBoxLayout(time_controls_container)
        time_layout.setContentsMargins(8, 4, 8, 4)
        time_layout.setSpacing(8)
        self.time_label = QLabel("Time Frame: 0 / 0")
        self.time_label.setProperty("secondary", "true")
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.setEnabled(False)
        self.time_slider.valueChanged.connect(self.on_time_slider_changed)
        self.time_slider.setProperty("role", "timeSlider")
        time_layout.addWidget(self.time_label)
        time_layout.addWidget(self.time_slider)
        self.display_area_widget.time_step_requested.connect(self.step_time_frame)
        self.display_area_widget.time_axis_changed.connect(self.show_time_axis)
        self.display_area_widget.play_toggle_requested.connect(self.toggle_cine)
        self.display_area_widget.segmentation_requested.connect(
            lambda key: key in self.segmentable_series and self.seg_button_signal.emit(key))
        self.display_area_widget.panels_moved.connect(self.panels_moved.emit)

        self.top_bar = QWidget()
        self.top_bar.setProperty("role", "topBar")
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(4, 2, 8, 2)
        top_layout.setSpacing(8)
        self.app_title = QLabel("q-cardIA v0.0.1")
        self.app_title.setProperty("role", "appTitle")
        self._build_docks()
        self.menu_bar = self._build_menu_bar()
        top_layout.addWidget(self.menu_bar, 0, Qt.AlignVCenter)
        top_layout.addStretch(1)
        top_layout.addWidget(self.app_title)

        viewer_column = QFrame()
        viewer_column.setObjectName("viewerColumn")
        viewer_layout = QVBoxLayout(viewer_column)
        viewer_layout.setContentsMargins(4, 4, 4, 0)
        viewer_layout.setSpacing(0)
        viewer_layout.addWidget(self.display_area_widget, 1)
        viewer_layout.addWidget(time_controls_container)
        # Hidden while a panel without frames (LGE) is selected; its space stays, so the panels do not move.
        keep = time_controls_container.sizePolicy()
        keep.setRetainSizeWhenHidden(True)
        time_controls_container.setSizePolicy(keep)
        self.display_area_widget.panel_selected.connect(time_controls_container.setVisible)
        self.contour_island = _ContourIsland(viewer_column)
        self.contour_island.tool_changed.connect(self.display_area_widget.set_contour_edit_tool)
        self.contour_island.save_requested.connect(self.emit_save_signal)
        self.contour_island.done_requested.connect(lambda: self.contour_edit_action.setChecked(False))

        self.setMenuWidget(self.top_bar)
        self.setCentralWidget(viewer_column)
        self.setContentsMargins(6, 0, 6, 6)
        # Sizes and the default layout only take once the docks are laid out, so after this event.
        # Plugins add their panels on session_started, which is before this runs.
        QTimer.singleShot(0, lambda: self._settle_docks(screen_width))

    def add_panel(self, widget: QWidget, title: str, name: str, menu_text: str,
                  shortcut: str | None = None, width: int | None = None) -> QDockWidget:
        """A side panel from a plugin: right of the series browser, tabbed with Results, and with
        its own entry in the View menu. Call it from Plugin.start_session, so the saved panel
        layout also restores this panel. name must be unique: the saved layout finds the panel by it.
        """
        dock = self._dock(widget, title, name, Qt.RightDockWidgetArea)
        self.splitDockWidget(self.series_dock, dock, Qt.Horizontal)
        if not self._plugin_docks:
            self.tabifyDockWidget(dock, self.results_dock)
            dock.raise_()
        action = dock.toggleViewAction()
        action.setText(menu_text)
        if shortcut:
            action.setShortcut(shortcut)
        self._view_menu.insertAction(self._reset_layout_action, action)
        self._plugin_docks.append((dock, width))
        return dock

    def _start_session(self):
        self.build_analysis_interface()
        self.session_started.emit()

    def return_to_start(self):
        reason = self.session_busy()
        if reason:
            self.show_info("Please wait", f"{reason} Return to the start screen once it has finished.")
            return
        self._play_timer.stop()
        self.session_ended.emit()
        SETTINGS.setValue("window/dock_state", self.saveState())
        self._default_dock_state = None
        self._dock_titles_timer.stop()
        self._dock_title_bars = {}
        plugin_docks = [dock for dock, _ in self._plugin_docks]
        self._plugin_docks = []
        for dock in (self.sidebar_dock, self.series_dock, self.results_dock, *plugin_docks):
            self.removeDockWidget(dock)
            dock.deleteLater()
        self.top_bar.deleteLater()
        self.setMenuWidget(None)
        self.setContentsMargins(0, 0, 0, 0)
        self.welcome_widget = WelcomeWidget(self, self.plugins)
        self.welcome_widget.transition_signal.connect(self._start_session)
        self.setCentralWidget(self.welcome_widget)

    def _build_docks(self):
        self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks)
        self.setTabPosition(Qt.AllDockWidgetAreas, QTabWidget.North)
        self.sidebar_dock = self._dock(self.sidebar, "Protocols", "patientDock", Qt.LeftDockWidgetArea)
        self.series_dock = self._dock(self.series_browser_widget, "Series", "seriesDock", Qt.RightDockWidgetArea)
        self.results_dock = self._dock(self.results_widget, "Results", "resultsDock", Qt.RightDockWidgetArea)
        self.splitDockWidget(self.series_dock, self.results_dock, Qt.Horizontal)

    def _settle_docks(self, screen_width: int):
        docks = [self.sidebar_dock, self.series_dock]
        widths = [self.sidebar.minimumSizeHint().width(), theme.scaled(200)]
        for dock, width in self._plugin_docks:
            if width is not None:
                docks.append(dock)
                widths.append(min(width, int(screen_width * 0.75)))
        self.resizeDocks(docks, widths, Qt.Horizontal)
        self.results_dock.hide()  # also in the default layout
        self._default_dock_state = self.saveState()
        saved = SETTINGS.value("window/dock_state")
        if saved is not None:
            self.restoreState(saved)
            self.results_dock.hide()  # hidden at every start; View > Results opens it

    def _update_dock_titles(self):
        """A dock merged into tabs is named by its tab, so its own title bar goes."""
        first_tabs = self._first_tab_titles()
        for dock, (title, blank) in self._dock_title_bars.items():
            # A closed panel stays in its tab group, but then no tab row shows, so it must not count.
            tabbed = any(other.toggleViewAction().isChecked() for other in self.tabifiedDockWidgets(dock))
            if dock.titleBarWidget() is not (blank if tabbed else title):
                dock.setTitleBarWidget(blank if tabbed else title)
            # Square top corners under a title bar, square top-left under the first tab.
            top = "title" if not tabbed else "firstTab" if dock.windowTitle() in first_tabs else "laterTab"
            theme.set_style_property(dock.widget(), "top", top)

    def _first_tab_titles(self) -> set[str]:
        """Titles of the docks shown as the first, selected tab; Qt builds these tab bars itself."""
        titles = set()
        for bar in self.findChildren(QTabBar, options=Qt.FindDirectChildrenOnly):
            if not bar.property("watched"):
                bar.setProperty("watched", True)
                bar.setElideMode(Qt.ElideNone)  # a tab as wide as its title; too many tabs scroll instead
                bar.currentChanged.connect(lambda _: self._dock_titles_timer.start())
                bar.installEventFilter(self)
            if bar.isVisible() and bar.currentIndex() == 0:
                titles.add(bar.tabText(0))
        return titles

    def _dock(self, panel: QWidget, title: str, name: str, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(name)  # restoreState finds each dock by this name
        dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._dock_title_bars[dock] = (_DockTitle(dock), QWidget(dock))
        dock.setTitleBarWidget(self._dock_title_bars[dock][0])
        # Painted too, so no seam of the lighter base shows between title and body.
        dock.setAttribute(Qt.WA_StyledBackground, True)
        panel.setProperty("role", "dockPanel")
        panel.setProperty("top", "title")
        dock.setWidget(panel)
        self.addDockWidget(area, dock)
        # After the move settles, as tabifiedDockWidgets only knows the new grouping then.
        dock.dockLocationChanged.connect(self._dock_titles_timer.start)
        dock.visibilityChanged.connect(self._dock_titles_timer.start)
        return dock

    def reset_panel_layout(self):
        self.restoreState(self._default_dock_state)
        self._update_dock_titles()

    def event(self, event):
        """The edge of a side panel that faces the images resizes that panel alone: the side grows or
        shrinks and the panels beside it keep their width. Qt's own separators would trade width
        between neighbours, or share it among the whole side."""
        kind = event.type()
        if kind == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            grip = self._panel_edge_at(event.position().toPoint())
            if grip is not None:
                dock, sign = grip
                self._edge_drag = (dock, sign, event.position().x(), dock.width())
                return True
        elif kind == QEvent.MouseMove and self._edge_drag:
            if not event.buttons() & Qt.LeftButton:
                self._edge_drag = None
                return super().event(event)
            dock, sign, x0, w0 = self._edge_drag
            width = max(dock.minimumSizeHint().width(), round(w0 + sign * (event.position().x() - x0)))
            self.resizeDocks([dock], [width], Qt.Horizontal)
            return True
        elif kind == QEvent.MouseButtonRelease and self._edge_drag:
            self._edge_drag = None
            return True
        return super().event(event)

    def eventFilter(self, obj, event):
        """A tab dragged out of its tab bar unstacks that panel: dragged down, it goes under the rest of
        its group; dragged sideways, beside it. Qt's own drag-out needs floating panels, which are off."""
        if isinstance(obj, QTabBar):
            kind = event.type()
            if kind == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._tab_drag = obj.tabAt(event.position().toPoint())
            elif kind == QEvent.MouseMove and self._tab_drag is not None and event.buttons() & Qt.LeftButton:
                pos = event.position()
                below, beside = pos.y() - obj.height(), max(-pos.x(), pos.x() - obj.width())
                if self._tab_drag >= 0 and max(below, beside) > _TAB_DRAG_OUT_PX:
                    title, self._tab_drag = obj.tabText(self._tab_drag), None
                    self._unstack(title, Qt.Vertical if below >= beside else Qt.Horizontal)
                    return True
            elif kind == QEvent.MouseButtonRelease:
                self._tab_drag = None
        return super().eventFilter(obj, event)

    def _unstack(self, title: str, orientation):
        dock = next((d for d in self._dock_title_bars if d.windowTitle() == title), None)
        rest = [d for d in self.tabifiedDockWidgets(dock) if d.toggleViewAction().isChecked()] if dock else []
        if rest:
            self.splitDockWidget(rest[0], dock, orientation)
            self._dock_titles_timer.start()

    def _panel_edge_at(self, pos) -> tuple[QDockWidget, int] | None:
        gap = self.style().pixelMetric(QStyle.PM_DockWidgetSeparatorExtent) + 6
        for dock in self._dock_title_bars:
            if not dock.isVisible() or dock.isFloating():
                continue
            rect = dock.geometry()
            if not rect.top() <= pos.y() <= rect.bottom():
                continue
            if self.dockWidgetArea(dock) == Qt.RightDockWidgetArea and rect.left() - gap <= pos.x() < rect.left():
                return dock, -1
            if self.dockWidgetArea(dock) == Qt.LeftDockWidgetArea and rect.right() < pos.x() <= rect.right() + gap:
                return dock, 1
        return None

    def closeEvent(self, event):
        if self._default_dock_state is not None:
            SETTINGS.setValue("window/dock_state", self.saveState())
            SETTINGS.sync()
        super().closeEvent(event)

    def _build_menu_bar(self) -> QMenuBar:
        bar = QMenuBar()
        bar.setObjectName("appMenu")

        file_menu = bar.addMenu("&File")
        file_menu.addAction("&Load Files...", QKeySequence.Open, self.on_load_patient_clicked)
        file_menu.addAction("&Save New Labels", QKeySequence.Save, self.emit_save_signal)
        file_menu.addSeparator()
        file_menu.addAction("&Return to Start Screen", self.return_to_start)
        file_menu.addSeparator()
        file_menu.addAction("E&xit", QKeySequence.Quit, self.close)

        view_menu = StickyMenu("&View", bar)
        bar.addMenu(view_menu)
        view_menu.addAction(self.full_screen_action)
        view_menu.addSeparator()
        self.toggle_sidebar_action = self._panel_action(view_menu, self.sidebar_dock, "&Protocols", "Ctrl+B")
        self.toggle_series_browser_action = self._panel_action(view_menu, self.series_dock, "&Series", "Ctrl+Shift+Q")
        self._panel_action(view_menu, self.results_dock, "&Results")
        # Panels from plugins go between Results and this entry, see add_panel.
        self._view_menu = view_menu
        self._reset_layout_action = view_menu.addAction("&Reset Panel Layout", self.reset_panel_layout)
        view_menu.addSeparator()
        crosshair = view_menu.addAction("&Crosshair")
        crosshair.setCheckable(True)
        crosshair.toggled.connect(self.display_area_widget.set_crosshair_shown)
        crosshair.toggled.connect(lambda on: SETTINGS.setValue("view/crosshair", on))
        crosshair.setChecked(SETTINGS.value("view/crosshair", True, type=bool))
        self.display_area_widget.set_crosshair_shown(crosshair.isChecked())
        view_menu.addSeparator()
        self.zoom_menu = view_menu.addMenu("&Zoom")
        for text, mode in (("Zoom to &Heart", "roi"), ("&Fill Panels", "fill"), ("Fit &Whole Image", "fit")):
            self.zoom_menu.addAction(text, lambda m=mode: self.zoom_requested.emit(m))
        view_menu.addSeparator()
        self.contours_menu = view_menu.addMenu("&Contours")
        self.contours_menu.addAction("&Show in All Panels", lambda: self.contours_requested.emit(True))
        self.contours_menu.addAction("&Hide in All Panels", lambda: self.contours_requested.emit(False))
        self.window_menu = view_menu.addMenu("&Window")
        self.window_menu.addAction("&Null Myocardium (LGE)", lambda: self.window_requested.emit("null_myocardium"))
        self.window_menu.addAction("&Reset All Windows", lambda: self.window_requested.emit("reset"))
        self.playback_menu = StickyMenu("&Playback", view_menu)
        self.playback_menu.aboutToShow.connect(self._rebuild_playback_menu)
        view_menu.addMenu(self.playback_menu)

        # The simple grids as diagrams, beside View as in CVI42; the split layouts are not in this menu.
        self.grid_menu = StickyMenu("&Grid", bar)
        self.grid_menu.aboutToShow.connect(self._rebuild_grid_menu)
        # The native macOS menu bar hides an empty menu, so aboutToShow would never fire.
        self._rebuild_grid_menu()
        bar.addMenu(self.grid_menu)

        tools_menu = StickyMenu("&Tools", bar)
        bar.addMenu(tools_menu)
        self.predict_menu = QMenu("Predict &Labels", self)
        self.predict_menu.aboutToShow.connect(self._rebuild_predict_menu)
        tools_menu.addMenu(self.predict_menu)
        tools_menu.addSeparator()
        self.contour_edit_action = tools_menu.addAction("&Edit Contours")
        self.contour_edit_action.setCheckable(True)
        self.contour_edit_action.setShortcut("Ctrl+E")
        self.contour_edit_action.toggled.connect(self._on_contour_edit_toggled)

        help_menu = bar.addMenu("&Help")
        for label, url in (("&Tutorials and Wiki", WIKI_URL), ("&Report an Issue", ISSUES_URL),
                           ("&About q-cardIA", REPO_URL)):
            help_menu.addAction(label, lambda u=url: QDesktopServices.openUrl(QUrl(u)))
        return bar

    def _rebuild_grid_menu(self):
        """Rebuilt on opening: a layout preset may have changed the grid since."""
        self.grid_menu.clear()
        group = QActionGroup(self.grid_menu)
        current = self.display_area_widget.grid_size
        split = self.display_area_widget.split
        for rows, cols in GRID_SIZES:
            action = self.grid_menu.addAction(theme.line_icon(f"grid:{rows}x{cols}", "text_muted"), f"{rows} \u00d7 {cols}")
            action.setCheckable(True)
            action.setChecked((rows, cols) == current and split is None)
            group.addAction(action)
            action.triggered.connect(lambda _=False, r=rows, c=cols: self.display_area_widget.set_grid_size(r, c))

    def _rebuild_playback_menu(self):
        """Rebuilt on opening, so the play state and the speed are current."""
        menu = self.playback_menu
        menu.clear()
        menu.addAction("&Pause" if self.is_playing() else "&Play", self.toggle_cine)
        self.speed_menus = []  # kept: a submenu built here must outlive this call
        for title, perfusion, one in (("&Speed", False, "one heartbeat per second"),
                                      ("P&erfusion Speed", True, "a run in four seconds")):
            speeds = menu.addMenu(title)
            self.speed_menus.append(speeds)
            group = QActionGroup(speeds)
            for speed in CINE_SPEEDS:
                action = speeds.addAction(f"{speed:g}\u00d7" + (f" ({one})" if speed == 1 else ""))
                action.setCheckable(True)
                action.setChecked(speed == (self._perfusion_speed if perfusion else self._cine_speed))
                group.addAction(action)
                action.triggered.connect(lambda _=False, s=speed, p=perfusion: self.set_cine_speed(s, p))
        loops = menu.addMenu("&Loop from This Frame")
        self.speed_menus.append(loops)
        looping = self._play_range[1] - self._play_range[0] + 1 if self._play_range and self.is_playing() else None
        for count in (3, 5, 7, 10):
            action = loops.addAction(f"{count} frames")
            action.setCheckable(True)
            action.setChecked(count == looping)
            action.setEnabled(self.time_slider.maximum() > 0)
            action.triggered.connect(lambda _=False, c=count: self._loop_from_here(c))
        menu.addSeparator()
        menu.addAction("Go to &End-Diastole", lambda: self.phase_requested.emit("ED"))
        menu.addAction("Go to End-&Systole", lambda: self.phase_requested.emit("ES"))

    def _predict(self, key: str):
        """Segments key (a cached segmentation is there already) and draws its contours in its panels."""
        self.display_area_widget.show_segmentation(key)
        self.seg_button_signal.emit(key)

    def _rebuild_predict_menu(self):
        self.predict_menu.clear()
        loaded = [k for k in self.segmentable_series if self.display_area_widget.is_series_registered(k)]
        for key in self.segmentable_series:
            action = self.predict_menu.addAction(key, lambda k=key: self._predict(k))
            action.setEnabled(key in loaded)
        self.predict_menu.addSeparator()
        every = self.predict_menu.addAction("All Loaded Views", lambda: [self._predict(k) for k in loaded])
        every.setEnabled(bool(loaded))

    def _panel_action(self, menu, dock: QDockWidget, text: str, shortcut: str | None = None) -> QAction:
        action = dock.toggleViewAction()
        action.setText(text)
        if shortcut:
            action.setShortcut(shortcut)
        menu.addAction(action)
        return action

    def on_load_patient_clicked(self):
        folder_path = pick_patient(self)
        if not folder_path:
            return
        self.load_patient_button_signal.emit(folder_path)

    def show_loading(self, text: str):
        self.sidebar.btn_load_patient.setEnabled(False)
        self.loading_overlay.show_message(text)

    def hide_loading(self):
        self.sidebar.btn_load_patient.setEnabled(True)
        self.loading_overlay.hide_overlay()

    def _on_contour_edit_toggled(self, checked):
        self.display_area_widget.set_contour_edit_mode(checked)
        self.display_area_widget.set_contour_edit_tool("move")
        self.contour_island.reset()
        self.contour_island.setVisible(checked)

    def emit_save_signal(self):
        self.save_button_signal.emit()

    def update_time_slider_range(self, num_frames):
        self.show_time_axis(num_frames, 0)

    def show_time_axis(self, num_frames: int, frame: int):
        """Let the time slider drive another time axis (the cines, or a perfusion run) at its frame."""
        self._play_timer.stop()
        self._play_range = None
        self.time_slider.blockSignals(True)  # the display already shows this frame
        self.time_slider.setMaximum(max(num_frames - 1, 0))
        self.time_slider.setValue(frame)
        self.time_slider.blockSignals(False)
        self.time_slider.setEnabled(num_frames > 1)
        self.time_label.setText(f"Time Frame: {frame + 1} / {num_frames}")

    def step_time_frame(self, step: int):
        n = self.time_slider.maximum() + 1
        if n > 1:
            self.time_slider.setValue((self.time_slider.value() + step) % n)

    def _play_tick(self):
        if self._play_range is None:
            self.step_time_frame(self._play_step)
            return
        first, last = self._play_range
        frame = self.time_slider.value() + self._play_step
        self.time_slider.setValue(frame if first <= frame <= last else first)

    def _loop_from_here(self, count: int):
        """Loop count frames from the frame shown, of the series the time slider drives."""
        first = self.time_slider.value()
        last = min(first + count - 1, self.time_slider.maximum())
        if last > first:
            self.play_frames(first, last)

    def play_frames(self, first: int, last: int):
        """Loop frames first to last (0-based) of the time axis that the time slider drives; one loop takes as
        long as a whole run at the current speed."""
        self._play_range = (first, last)
        self.time_slider.setValue(first)
        self.time_label.setText(f"Time Frame: {first + 1} / {self.time_slider.maximum() + 1}  (loop {first + 1}-{last + 1})")
        self._play_timer.start(self._play_interval())

    def log_screen(self):
        """The screen to the session log, as far as it changed since the last time."""
        if hasattr(self, "display_area_widget"):
            study_log.log_state({**self.display_area_widget.state(), "playing": self.is_playing(),
                                 "speed": self._cine_speed, "perfusion_speed": self._perfusion_speed})

    def schedule_screen_log(self):
        self._screen_log_timer.start()

    def is_playing(self) -> bool:
        return self._play_timer.isActive()

    @property
    def cine_speed(self) -> float:
        return self._cine_speed

    @property
    def perfusion_speed(self) -> float:
        return self._perfusion_speed

    def set_cine_speed(self, speed: float, perfusion: bool = False):
        """The cines' speed (heartbeats per second), or the perfusion's; also while playing."""
        if perfusion:
            self._perfusion_speed = speed
        else:
            self._cine_speed = speed
        if self._play_timer.isActive():
            self._play_timer.start(self._play_interval())

    def _play_interval(self) -> int:
        """ms per step: one heartbeat per second at speed 1, and a perfusion run (one contrast passage) in
        four seconds. At most 50 steps per second, so faster playback skips frames."""
        perfusion = self.display_area_widget.driven_time_axis.startswith("PERF_")
        frames = self._play_range[1] - self._play_range[0] + 1 if self._play_range else self.time_slider.maximum() + 1
        fps = frames * (self._perfusion_speed / 4 if perfusion else self._cine_speed)
        self._play_step = max(1, math.ceil(fps / 50))
        return max(20, round(1000 * self._play_step / fps))

    def toggle_cine(self):
        """Play or pause the cine at the chosen speed, over all frames."""
        self._play_range = None
        self.time_label.setText(f"Time Frame: {self.time_slider.value() + 1} / {self.time_slider.maximum() + 1}")
        if self._play_timer.isActive():
            self._play_timer.stop()
        elif self.time_slider.maximum() > 0:
            self._play_timer.start(self._play_interval())

    def on_time_slider_changed(self, value):
        max_val = self.time_slider.maximum()
        loop = f"  (loop {self._play_range[0] + 1}-{self._play_range[1] + 1})" if self._play_range else ""
        self.time_label.setText(f"Time Frame: {value + 1} / {max_val + 1}{loop}")
        self.time_frame_changed_signal.emit(value)

    def show_error(self, title, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec()

    def show_warning(self, title, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec()

    def show_info(self, title, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec()
