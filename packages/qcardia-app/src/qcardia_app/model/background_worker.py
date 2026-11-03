"""QThread base for any blocking call, so it runs off the GUI thread."""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class CallableWorker(QThread):
    finished_ok = Signal(object)  # whatever `func` returns
    error = Signal(object)  # the raised exception itself, so callers can inspect its type
    progress = Signal(object)  # whatever `func` passes to its `report` callback, with report=True

    def __init__(self, func, *args, report: bool = False, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = {**kwargs, "report": self.progress.emit} if report else kwargs

    def run(self):
        try:
            result = self._func(*self._args, **self._kwargs)
        except Exception as e:
            self.error.emit(e)
            return
        self.finished_ok.emit(result)
