"""Generic QThread wrapper for running one blocking call off the GUI thread.

Used for anything slow enough to freeze the UI (segmentation inference, ...)
without writing a bespoke QThread subclass per operation.
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class CallableWorker(QThread):
    finished_ok = Signal(object)  # whatever `func` returns
    error = Signal(object)  # the raised exception itself, so callers can inspect its type

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._func(*self._args, **self._kwargs)
        except Exception as e:
            self.error.emit(e)
            return
        self.finished_ok.emit(result)
