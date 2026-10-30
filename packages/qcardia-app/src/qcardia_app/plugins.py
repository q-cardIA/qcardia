"""Optional extensions: installed packages that add panels and options to the app.

A plugin is a class that derives from Plugin. Its package registers it under the entry point
group "qcardia_app.plugins" in its pyproject.toml:

    [project.entry-points."qcardia_app.plugins"]
    my_plugin = "my_package.plugin:MyPlugin"

At startup, the app makes one instance of each registered plugin. A plugin that fails to load
is skipped, so it cannot stop the app.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

    from qcardia_app.controllers.dicom_controller import DICOMController
    from qcardia_app.model.dicom_model import DICOMModel
    from qcardia_app.view.main_window import MainWindow

ENTRY_POINT_GROUP = "qcardia_app.plugins"

log = logging.getLogger(__name__)


@dataclass
class AppContext:
    """What a plugin gets at the start of a session."""

    window: MainWindow
    controller: DICOMController
    model: DICOMModel


class Plugin:
    """Base class for plugins. A plugin overrides only the hooks it needs."""

    name = "plugin"

    def welcome_options(self, parent: QWidget) -> QWidget | None:
        """A widget for the options area of the start screen, or None.

        Called each time the start screen is built. The widget is deleted with the start screen,
        so keep the chosen values in the plugin, not only in the widget.
        """
        return None

    def start_session(self, ctx: AppContext) -> None:
        """The analysis screen is built. Add panels here with ctx.window.add_panel."""

    def end_session(self) -> None:
        """The user returns to the start screen. The panels are removed after this."""

    def busy_reason(self) -> str | None:
        """Why the session cannot end yet, for example a running task, or None."""
        return None

    def session_info(self) -> dict:
        """Values for the session log, for example the plugin's settings."""
        return {}

    def session_log_closed(self, path: Path, study: bool) -> None:
        """The session log is complete at path. study is True for a named study session."""

    def shutdown(self) -> None:
        """The app quits. Stop threads and release external resources."""


def load_plugins() -> list[Plugin]:
    """One instance of each installed plugin, in a fixed order (by entry point name)."""
    plugins = []
    for entry_point in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda ep: ep.name):
        try:
            plugin = entry_point.load()()
        except Exception:
            log.exception("Plugin %r could not load; the app continues without it", entry_point.name)
            continue
        if not isinstance(plugin, Plugin):
            log.error("Plugin %r is not a qcardia_app.plugins.Plugin; skipped", entry_point.name)
            continue
        plugins.append(plugin)
        log.info("Plugin loaded: %s", plugin.name)
    return plugins
